/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * <tsimmerman.ss@phystech.edu>, <alex.rom23@mail.ru> wrote this file.  As long as you
 * retain this notice you can do whatever you want with this stuff. If we meet
 * some day, and you think this stuff is worth it, you can buy me a beer in
 * return.
 * ----------------------------------------------------------------------------
 */

#include "opencl_include.hpp"
#include "selector.hpp"
#include "utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>

#include <boost/algorithm/string/replace.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/option.hpp>
#include <type_traits>

#include "linmath/contiguous_matrix.hpp"

#define STRINGIFY0(v) #v
#define STRINGIFY(v) STRINGIFY0(v)

#ifndef TYPE__
#define TYPE__ float
#endif

namespace po = boost::program_options;
namespace linmath = throttle::linmath;

using matrix_type = linmath::contiguous_matrix<TYPE__>;

namespace app {

struct matrix_sizes {
  matrix_type::size_type ax, ay, by;
};

struct ndrange_query {
  cl::NDRange global, local;
};

struct i_matmult_kernel_source {
  virtual std::string get_kernel_source() const = 0;
  virtual std::string get_kernel_name() const = 0;

  virtual ndrange_query get_ndranges(matrix_sizes) const = 0;

  ~i_matmult_kernel_source() {}
};

class naive_matmult_kernel : public i_matmult_kernel_source {
  std::string m_source;

public:
  naive_matmult_kernel(std::string type) {
    static const std::string matmult_naive =
        R"(
    __kernel void matmult_naive(__global TYPE *A, __global TYPE *B, __global TYPE *C, int AX, int AY, int BY) {
      int i = get_global_id(0);
      int j = get_global_id(1);

      TYPE sum = 0;
      for (int k = 0; k < AY; ++k) {
        sum += A[i * AY + k] * B[k * BY + j];
      }

      C[i * BY + j] = sum;
    })";
    auto type_macro_def = clutils::kernel_define("TYPE", type);
    m_source = type_macro_def + matmult_naive;
  }

  std::string get_kernel_source() const override { return m_source; }
  std::string get_kernel_name() const override { return "matmult_naive"; }

  ndrange_query get_ndranges(matrix_sizes sizes) const override {
    return {cl::NDRange{sizes.ax, sizes.by}, cl::NDRange{}};
  }
};

class matmult_kernel_localmem : public i_matmult_kernel_source {
  std::string m_source;
  unsigned    m_local_size;

public:
  matmult_kernel_localmem(std::string type, unsigned local_size) : m_local_size{local_size} {
    static const std::string matmult_localmem =
        R"(
    __kernel void matmult_localmem(__global TYPE *A, __global TYPE *B, __global TYPE *C, int AX, int AY, int BY) {
      int tile_row = get_group_id(0);
      int tile_col = get_group_id(1);

      int local_row = get_local_id(0);
      int local_col = get_local_id(1);

      __local tile_A[TILE_SIZE * TILE_SIZE];
      __local tile_B[TILE_SIZE * TILE_SIZE];

      int global_row = TILE_SIZE * tile_row + local_row;
      int global_col = TILE_SIZE * tile_col + local_col;

      int tile_count = AY / TILE_SIZE;
      TYPE sum = 0;

      for (int t = 0; t < tile_count; ++t) {
        // Step 1. Here each work group thread is responsible for copying data into the corresponding slot in the tile_A, tile_B
        tile_A[local_row * TILE_SIZE + local_col] = A[global_row * AY + t * TILE_SIZE + local_col];
        tile_B[local_row * TILE_SIZE + local_col] = B[BY * (t * TILE_SIZE + local_row) + global_col];

        // Barrier here to finish loading all the data before proceeding.
        barrier(CLK_LOCAL_MEM_FENCE);

        // Step 2. Calculate part of the resulting tile corresponding to this thread and accumulate it in sum.
        for (int k = 0; k < TILE_SIZE; ++k) {
          sum += tile_A[TILE_SIZE * local_row + k] * tile_B[k * TILE_SIZE + local_col];
        }

        // Wait for all threads to finish before reloading new tiles.
        barrier(CLK_LOCAL_MEM_FENCE);
      }

      C[global_row * BY + global_col] = sum;
    })";

    auto type_macro_def = clutils::kernel_define("TYPE", type);
    auto local_size_macro_def = clutils::kernel_define("TILE_SIZE", local_size);
    m_source = type_macro_def + local_size_macro_def + matmult_localmem;
  }

  std::string get_kernel_source() const override { return m_source; }
  std::string get_kernel_name() const override { return "matmult_localmem"; }

  ndrange_query get_ndranges(matrix_sizes sizes) const override {
    if ((sizes.ax % m_local_size) != 0 || (sizes.ay % m_local_size) != 0 || (sizes.by % m_local_size) != 0)
      throw std::invalid_argument{"Local size should evenly divide the matrix dimension"};
    return {cl::NDRange{sizes.ax, sizes.by}, cl::NDRange{m_local_size, m_local_size}};
  }
};

struct profiling_info {
  std::chrono::milliseconds gpu_pure;
  std::chrono::milliseconds gpu_wall;
};

class matmult : private clutils::platform_selector {
  std::unique_ptr<i_matmult_kernel_source> m_source;

  cl::Context      m_ctx;
  cl::CommandQueue m_queue;

  cl::Program m_program;

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int, cl_int, cl_int> m_functor;

public:
  static constexpr clutils::platform_version c_api_version = {2, 2};

  matmult(std::unique_ptr<i_matmult_kernel_source> source)
      : clutils::platform_selector{c_api_version}, m_source{std::move(source)}, m_ctx{m_device},
        m_queue{m_ctx, cl::QueueProperties::Profiling}, m_program{m_ctx, m_source->get_kernel_source(), true},
        m_functor{m_program, m_source->get_kernel_name()} {}

  matrix_type multiply(const matrix_type &mata, const matrix_type &matb, profiling_info *time = nullptr) {
    if (mata.cols() != matb.rows()) throw std::invalid_argument{"Mismatched matrix sizes"};

    const auto mat_size = [](const auto &m) { return std::distance(m.begin(), m.end()); };
    const auto mat_bin_size = [&mat_size](const auto &m) { return mat_size(m) * sizeof(TYPE__); };

    auto wall_start = std::chrono::high_resolution_clock::now();

    matrix_type matc{mata.rows(), matb.cols()};

    cl::Buffer abuf = {m_ctx, CL_MEM_READ_ONLY, mat_bin_size(mata)};
    cl::Buffer bbuf = {m_ctx, CL_MEM_READ_ONLY, mat_bin_size(matb)};
    cl::Buffer cbuf = {m_ctx, CL_MEM_WRITE_ONLY, mat_bin_size(matc)};

    cl::copy(m_queue, mata.begin(), mata.end(), abuf);
    cl::copy(m_queue, matb.begin(), matb.end(), bbuf);

    auto            ranges = m_source->get_ndranges({mata.rows(), mata.cols(), matb.cols()});
    cl::EnqueueArgs args = {m_queue, ranges.global, ranges.local};

    auto evnt = m_functor(args, abuf, bbuf, cbuf, mata.rows(), mata.cols(), matb.cols());
    evnt.wait();
    cl::copy(m_queue, cbuf, matc.begin(), matc.end());
    auto wall_end = std::chrono::high_resolution_clock::now();

    std::chrono::nanoseconds pure_start{evnt.getProfilingInfo<CL_PROFILING_COMMAND_START>()},
        pure_end{evnt.getProfilingInfo<CL_PROFILING_COMMAND_END>()};

    auto pure = std::chrono::duration_cast<std::chrono::milliseconds>(pure_end - pure_start);
    auto wall = std::chrono::duration_cast<std::chrono::milliseconds>(wall_end - wall_start);

    if (time) *time = {pure, wall};
    return matc;
  }
};

} // namespace app

template <typename T> auto create_random_number_generator(T lower, T upper) {
  std::random_device rnd_device;
  std::mt19937       mersenne_engine{rnd_device()};

  if constexpr (std::is_floating_point_v<T>) {
    std::uniform_real_distribution<T> dist{lower, upper};

    return [dist, mersenne_engine](auto &vec) mutable {
      std::generate(vec.begin(), vec.end(), [&]() { return dist(mersenne_engine); });
    };
  }

  else {
    std::uniform_int_distribution<T> dist{lower, upper};

    return [dist, mersenne_engine](auto &vec) mutable {
      std::generate(vec.begin(), vec.end(), [&]() { return dist(mersenne_engine); });
    };
  }
}

int main(int argc, char *argv[]) try {
  po::options_description desc("Available options");

  TYPE__   lower, upper;
  unsigned ax, ay, by, lsz;

  std::string kernel_name;
  desc.add_options()("help,h", "Print this help message")("print,p", "Print on failure")(
      "skip,s", "Skip cpu calculation")("lower,l", po::value<TYPE__>(&lower)->default_value(0),
                                        "Low bound for random integer")(
      "upper,u", po::value<TYPE__>(&upper)->default_value(32), "Upper bound for random integer")(
      "ax", po::value<unsigned>(&ax)->default_value(512),
      "Number of rows in matrix A")("ay", po::value<unsigned>(&ay)->default_value(512), "Number of cols in matrix A")(
      "by", po::value<unsigned>(&by)->default_value(512), "Number of cols in matrix B")(
      "kernel,k", po::value<std::string>(&kernel_name)->default_value("localmem"),
      "Which kernel to use: naive, localmem")("lsz", po::value<unsigned>(&lsz)->default_value(1),
                                              "Local iteration size");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  bool skip_cpu = vm.count("skip");
  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  std::unique_ptr<app::i_matmult_kernel_source> source;
  if (kernel_name == "naive") {
    source = std::make_unique<app::naive_matmult_kernel>(STRINGIFY(TYPE__));
  } else if (kernel_name == "localmem") {
    source = std::make_unique<app::matmult_kernel_localmem>(STRINGIFY(TYPE__), lsz);
  } else {
    std::cout << "Unknown type of kernel: " << kernel_name << "\n";
    return EXIT_FAILURE;
  }

  std::cout << "Multiplying A [" << ax << " x " << ay << "] by B [" << ay << " x " << by << "]\n";
  app::matmult mult = {std::move(source)};

  matrix_type a{ax, ay}, b{ay, by};

  auto random_filler = create_random_number_generator<TYPE__>(lower, upper);
  random_filler(a);
  random_filler(b);

  std::chrono::milliseconds wall;

  matrix_type c;
  if (!skip_cpu) {
    auto wall_start = std::chrono::high_resolution_clock::now();
    c = a * b;
    auto wall_end = std::chrono::high_resolution_clock::now();
    wall = std::chrono::duration_cast<std::chrono::milliseconds>(wall_end - wall_start);
  }

  static const auto matrix_print = [](auto name, auto &mat) {
    std::cout << name << " : \n";
    for (unsigned i = 0; i < mat.rows(); ++i) {
      for (auto v : mat[i]) {
        std::cout << v << "\t";
      }
      std::cout << "\n";
    }
  };

  app::profiling_info prof_info;

  auto res = mult.multiply(a, b, &prof_info);
  if (!skip_cpu) {
    std::cout << "CPU wall time: " << wall.count() << " ms\n";
  }

  std::cout << "GPU wall time: " << prof_info.gpu_wall.count() << " ms\n";
  std::cout << "GPU pure time: " << prof_info.gpu_pure.count() << " ms\n";

  const auto validate_results = [&c, &res, &a, &b]() {
    if (c == res) {
      std::cout << "GPU matrix multiplication works fine\n";
      return EXIT_SUCCESS;
    } else {
      std::cout << "GPU matrix multiplication is borked\n";

      matrix_print("Matrix A", a);
      matrix_print("Matrix B", b);

      matrix_print("Matrix from GPU", res);
      matrix_print("Correct", c);

      return EXIT_FAILURE;
    }
  };

  if (skip_cpu) return EXIT_SUCCESS;
  return validate_results();
} catch (cl::BuildError &e) {
  std::cerr << "Compilation failed:\n";
  for (const auto &v : e.getBuildLog()) {
    std::cerr << v.second << "\n";
  }
} catch (cl::Error &e) {
  std::cerr << "OpenCL error: " << e.what() << "(" << e.err() << ")\n";
} catch (std::exception &e) {
  std::cerr << "Encountered error: " << e.what() << "\n";
} catch (...) {
  std::cerr << "Unknown error\n";
}