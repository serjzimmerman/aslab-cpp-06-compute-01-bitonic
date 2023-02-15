/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * <tsimmerman.ss@phystech.edu>, <alex.rom23@mail.ru> wrote this file.  As long as you
 * retain this notice you can do whatever you want with this stuff. If we meet
 * some day, and you think this stuff is worth it, you can buy us a beer in
 * return.
 * ----------------------------------------------------------------------------
 */

#include "opencl_include.hpp"
#include "selector.hpp"
#include "utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>

#include <boost/program_options.hpp>
#include <boost/program_options/option.hpp>
#include <type_traits>

#include "linmath/contiguous_matrix.hpp"

#define STRINGIFY0(v) #v
#define STRINGIFY(v) STRINGIFY0(v)

#ifndef TYPE__
#define TYPE__ int
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

using clutils::profiling_info;
class i_matmult {
public:
  virtual matrix_type operator()(const matrix_type &, const matrix_type &, profiling_info *) = 0;

  matrix_type multiply(const matrix_type &mata, const matrix_type &matb, profiling_info *time = nullptr) {
    return operator()(mata, matb, time);
  }

  virtual ~i_matmult() {}
};

class gpu_matmult : public i_matmult, protected clutils::platform_selector {
protected:
  cl::Context m_ctx;
  cl::CommandQueue m_queue;

protected:
  static constexpr clutils::platform_version c_api_version = {2, 2};

  gpu_matmult()
      : clutils::platform_selector{c_api_version}, m_ctx{m_device}, m_queue{m_ctx, cl::QueueProperties::Profiling} {}

  using func_signature = cl::Event(cl::Buffer, cl::Buffer, cl::Buffer);
  matrix_type run_boilerplate(const matrix_type &mata, const matrix_type &matb, std::function<func_signature> func,
                              profiling_info *time) {
    if (mata.cols() != matb.rows()) throw std::invalid_argument{"Mismatched matrix sizes"};

    const auto mat_size = [](const auto &m) { return std::distance(m.begin(), m.end()); };
    const auto mat_bin_size = [&mat_size](const auto &m) { return mat_size(m) * sizeof(matrix_type::value_type); };

    auto wall_start = std::chrono::high_resolution_clock::now();

    matrix_type matc = {mata.rows(), matb.cols()};

    cl::Buffer bufa = {m_ctx, CL_MEM_READ_ONLY, mat_bin_size(mata)};
    cl::Buffer bufb = {m_ctx, CL_MEM_READ_ONLY, mat_bin_size(matb)};
    cl::Buffer bufc = {m_ctx, CL_MEM_WRITE_ONLY, mat_bin_size(matc)};

    cl::copy(m_queue, mata.begin(), mata.end(), bufa);
    cl::copy(m_queue, matb.begin(), matb.end(), bufb);

    auto event = func(bufa, bufb, bufc);
    event.wait();
    cl::copy(m_queue, bufc, matc.begin(), matc.end());
    auto wall_end = std::chrono::high_resolution_clock::now();

    std::chrono::nanoseconds pure_start{event.getProfilingInfo<CL_PROFILING_COMMAND_START>()},
        pure_end{event.getProfilingInfo<CL_PROFILING_COMMAND_END>()};

    auto pure = std::chrono::duration_cast<std::chrono::milliseconds>(pure_end - pure_start);
    auto wall = std::chrono::duration_cast<std::chrono::milliseconds>(wall_end - wall_start);

    if (time) *time = {pure, wall};
    return matc;
  }
};

class naive_matmult : public gpu_matmult {
  struct kernel {
    using functor_type = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int, cl_int, cl_int>;

    static std::string source(std::string type) {
      static const std::string naive_source =
          R"(
      __kernel void naive(__global TYPE *A, __global TYPE *B, __global TYPE *C, int AX, int AY, int BY) {
        int i = get_global_id(0);
        int j = get_global_id(1);

        TYPE sum = 0;
        for (int k = 0; k < AY; ++k) {
          sum += A[i * AY + k] * B[k * BY + j];
        }

        C[i * BY + j] = sum;
      })";

      auto type_macro_def = clutils::kernel_define("TYPE", type);
      return type_macro_def + naive_source;
    }

    static std::string entry() { return "naive"; }
  };

private:
  cl::Program m_program;
  kernel::functor_type m_functor;

public:
  naive_matmult()
      : gpu_matmult{}, m_program{m_ctx, kernel::source(STRINGIFY(TYPE__)), true}, m_functor{m_program,
                                                                                            kernel::entry()} {}

  matrix_type operator()(const matrix_type &mata, const matrix_type &matb, profiling_info *time = nullptr) override {
    const auto func = [&](auto bufa, auto bufb, auto bufc) {
      cl::EnqueueArgs args = {m_queue, {mata.rows(), matb.cols()}};
      return m_functor(args, bufa, bufb, bufc, mata.rows(), mata.cols(), matb.cols());
    };
    return gpu_matmult::run_boilerplate(mata, matb, func, time);
  }
};

class tiled_matmult : public gpu_matmult {
  struct kernel {
    using functor_type = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int, cl_int, cl_int>;

    static std::string source(std::string type, unsigned local_size) {
      static const std::string matmult_tiled =
          R"(
      __kernel void tiled(__global TYPE *A, __global TYPE *B, __global TYPE *C, int AX, int AY, int BY) {
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

      return type_macro_def + local_size_macro_def + matmult_tiled;
    }

    static std::string entry() { return "tiled"; }
  };

private:
  cl::Program m_program;
  kernel::functor_type m_functor;

  unsigned m_tile_size;

public:
  tiled_matmult(unsigned tile_size)
      : gpu_matmult{}, m_program{m_ctx, kernel::source(STRINGIFY(TYPE__), tile_size), true},
        m_functor{m_program, kernel::entry()}, m_tile_size{tile_size} {}

  matrix_type operator()(const matrix_type &mata, const matrix_type &matb, profiling_info *time = nullptr) {
    if (mata.rows() % m_tile_size != 0 || mata.cols() % m_tile_size != 0 || matb.cols() % m_tile_size != 0 ||
        matb.rows() % m_tile_size != 0)
      throw std::invalid_argument{"Matrix sizes should be divisible by the tile size"};

    const auto func = [&](auto bufa, auto bufb, auto bufc) {
      cl::EnqueueArgs args = {m_queue, {mata.rows(), matb.cols()}, {m_tile_size, m_tile_size}};
      return m_functor(args, bufa, bufb, bufc, mata.rows(), mata.cols(), matb.cols());
    };

    return gpu_matmult::run_boilerplate(mata, matb, func, time);
  }
};

class tiled_arbitrary_matmult : public gpu_matmult {
  struct kernel {
    using functor_type = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int, cl_int, cl_int, cl_int>;

    static std::string source(std::string type, unsigned local_size) {
      static const std::string tiled_arbitrary_size =
          R"(
      __kernel void tiled_arbitrary(__global TYPE *A, __global TYPE *B, __global TYPE *C, int AX, int AY, int BY, int tile_count) {
        int tile_row = get_group_id(0);
        int tile_col = get_group_id(1);

        int local_row = get_local_id(0);
        int local_col = get_local_id(1);

        __local tile_A[TILE_SIZE * TILE_SIZE];
        __local tile_B[TILE_SIZE * TILE_SIZE];

        int global_row = TILE_SIZE * tile_row + local_row;
        int global_col = TILE_SIZE * tile_col + local_col;

        int row_out_of_bounds = (global_row >= AX);
        int col_out_of_bounds = (global_col >= BY);

        TYPE sum = 0;

        for (int t = 0; t < tile_count; ++t) {
          // Step 1. Here each work group thread is responsible for copying data into the corresponding slot in the tile_A, tile_B
          int curr_tiled_col = t * TILE_SIZE + local_col;
          int curr_tiled_row = t * TILE_SIZE + local_row;

          tile_A[local_row * TILE_SIZE + local_col] = ((curr_tiled_col >= AY || row_out_of_bounds) ? 0 : A[global_row * AY + curr_tiled_col]);
          tile_B[local_row * TILE_SIZE + local_col] = ((curr_tiled_row >= AY || col_out_of_bounds) ? 0 : B[BY * curr_tiled_row + global_col]);

          // Barrier here to finish loading all the data before proceeding.
          barrier(CLK_LOCAL_MEM_FENCE);

          // Step 2. Calculate part of the resulting tile corresponding to this thread and accumulate it in sum.
          for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[TILE_SIZE * local_row + k] * tile_B[k * TILE_SIZE + local_col];
          }

          // Wait for all threads to finish before reloading new tiles.
          barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (row_out_of_bounds || col_out_of_bounds) return;
        C[global_row * BY + global_col] = sum;
      })";

      auto type_macro_def = clutils::kernel_define("TYPE", type);
      auto local_size_macro_def = clutils::kernel_define("TILE_SIZE", local_size);

      return type_macro_def + local_size_macro_def + tiled_arbitrary_size;
    }

    static std::string entry() { return "tiled_arbitrary"; }
  };

private:
  cl::Program m_program;
  kernel::functor_type m_functor;

  unsigned m_tile_size;

public:
  tiled_arbitrary_matmult(unsigned tile_size)
      : gpu_matmult{}, m_program{m_ctx, kernel::source(STRINGIFY(TYPE__), tile_size), true},
        m_functor{m_program, kernel::entry()}, m_tile_size{tile_size} {}

  matrix_type operator()(const matrix_type &mata, const matrix_type &matb, profiling_info *time = nullptr) override {
    const auto func = [&](auto bufa, auto bufb, auto bufc) {
      const auto tile_sz = m_tile_size;
      const auto recalc_size = [tile_sz](auto sz) {
        if (sz % tile_sz == 0) return sz / tile_sz;
        return (sz / tile_sz + 1);
      };

      auto recalc_rows = recalc_size(mata.rows()) * tile_sz;
      auto recalc_cols = recalc_size(matb.cols()) * tile_sz;

      cl::EnqueueArgs args = {m_queue, {recalc_rows, recalc_cols}, {tile_sz, tile_sz}};

      int tile_count = recalc_size(mata.cols());
      return m_functor(args, bufa, bufb, bufc, mata.rows(), mata.cols(), matb.cols(), tile_count);
    };
    return gpu_matmult::run_boilerplate(mata, matb, func, time);
  }
};

} // namespace app

int main(int argc, char *argv[]) try {
  po::options_description desc("Available options");

  TYPE__ lower, upper;
  unsigned ax, ay, by, lsz;

  std::string kernel_name;
  desc.add_options()("help,h", "Print this help message")("print,p", "Print on failure")(
      "skip,s", "Skip cpu calculation")("lower,l", po::value<TYPE__>(&lower)->default_value(0),
                                        "Low bound for random integer")(
      "upper,u", po::value<TYPE__>(&upper)->default_value(32), "Upper bound for random integer")(
      "ax", po::value<unsigned>(&ax)->default_value(512),
      "Number of rows in matrix A")("ay", po::value<unsigned>(&ay)->default_value(512), "Number of cols in matrix A")(
      "by", po::value<unsigned>(&by)->default_value(512), "Number of cols in matrix B")(
      "kernel,k", po::value<std::string>(&kernel_name)->default_value("naive"),
      "Which kernel to use: naive, tiled, tiledarb")("lsz", po::value<unsigned>(&lsz)->default_value(8),
                                                     "Local iteration size");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  bool skip_cpu = vm.count("skip");
  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  std::unique_ptr<app::i_matmult> mult;
  if (kernel_name == "naive") {
    mult = std::make_unique<app::naive_matmult>();
  } else if (kernel_name == "tiled") {
    mult = std::make_unique<app::tiled_matmult>(lsz);
  } else if (kernel_name == "tiledarb") {
    mult = std::make_unique<app::tiled_arbitrary_matmult>(lsz);
  } else {
    std::cout << "Unknown type of kernel: " << kernel_name << "\n";
    return EXIT_FAILURE;
  }

  const auto print_sep = []() { std::cout << " -------- \n"; };
  std::cout << "Multiplying A [" << ax << " x " << ay << "] by B [" << ay << " x " << by << "]\n";
  print_sep();

  matrix_type a{ax, ay}, b{ay, by};

  auto random_filler = clutils::create_random_number_generator<matrix_type::value_type>(lower, upper);
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

  auto res = mult->multiply(a, b, &prof_info);
  if (!skip_cpu) {
    std::cout << "CPU wall time: " << wall.count() << " ms\n";
  }

  std::cout << "GPU wall time: " << prof_info.wall.count() << " ms\n";
  std::cout << "GPU pure time: " << prof_info.pure.count() << " ms\n";

  print_sep();

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