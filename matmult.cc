/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * <tsimmerman.ss@phystech.edu>, <alex.rom23@mail.ru> wrote this file.  As long as you
 * retain this notice you can do whatever you want with this stuff. If we meet
 * some day, and you think this stuff is worth it, you can buy me a beer in
 * return.
 * ----------------------------------------------------------------------------
 */

#include "CL/cl_ext.h"
#include "opencl_include.hpp"
#include "selector.hpp"
#include "utils.hpp"

#include <algorithm>
#include <bits/chrono.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>

#include <boost/algorithm/string/replace.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/option.hpp>

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

struct i_matmult_kernel_source {
  virtual std::string get_kernel_source() const = 0;
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
    m_source = boost::algorithm::replace_all_copy(matmult_naive, "TYPE", type);
  }

  std::string get_kernel_source() const override { return m_source; }
};

struct profiling_info {
  std::chrono::microseconds gpu_pure;
  std::chrono::microseconds gpu_wall;
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
        m_functor{m_program, "matmult_naive"} {}

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

    cl::NDRange     global = {matc.rows(), matc.cols()};
    cl::EnqueueArgs args = {m_queue, global};

    auto evnt = m_functor(args, abuf, bbuf, cbuf, mata.rows(), mata.cols(), matb.cols());
    evnt.wait();
    cl::copy(m_queue, cbuf, matc.begin(), matc.end());
    auto wall_end = std::chrono::high_resolution_clock::now();

    std::chrono::nanoseconds pure_start{evnt.getProfilingInfo<CL_PROFILING_COMMAND_START>()},
        pure_end{evnt.getProfilingInfo<CL_PROFILING_COMMAND_END>()};

    auto pure = std::chrono::duration_cast<std::chrono::microseconds>(pure_end - pure_start);
    auto wall = std::chrono::duration_cast<std::chrono::microseconds>(wall_end - wall_start);

    if (time) *time = {pure, wall};
    return matc;
  }
};

} // namespace app

int main(int argc, char *argv[]) try {
  po::options_description desc("Available options");

  int      lower, upper;
  unsigned ax, ay, by;
  desc.add_options()("help,h", "Print this help message")("lower,l", po::value<int>(&lower)->default_value(0),
                                                          "Low bound for random integer")(
      "upper,u", po::value<int>(&upper)->default_value(32), "Upper bound for random integer")(
      "ax", po::value<unsigned>(&ax)->default_value(512),
      "Number of rows in matrix A")("ay", po::value<unsigned>(&ay)->default_value(512), "Number of cols in matrix A")(
      "by", po::value<unsigned>(&by)->default_value(512), "Number of cols in matrix B");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  std::cout << "Multiplying A [" << ax << " x " << ay << "] by B [" << ay << " x " << by << "]\n";
  app::matmult mult = {std::make_unique<app::naive_matmult_kernel>(STRINGIFY(TYPE__))};

  std::random_device rnd_device;
  std::mt19937       mersenne_engine{rnd_device()};

  std::uniform_int_distribution<TYPE__> dist{lower, upper};

  const auto random = [&dist, &mersenne_engine] { return dist(mersenne_engine); };
  const auto fill_random_matrix = [random](auto &vec) { std::generate(vec.begin(), vec.end(), random); };

  matrix_type a{ax, ay}, b{ay, by};
  fill_random_matrix(a);
  fill_random_matrix(b);

  auto wall_start = std::chrono::high_resolution_clock::now();
  auto c = a * b;
  auto wall_end = std::chrono::high_resolution_clock::now();
  auto wall = std::chrono::duration_cast<std::chrono::microseconds>(wall_end - wall_start);

  app::profiling_info prof_info;

  auto res = mult.multiply(a, b, &prof_info);
  bool correct = (c == res);

  std::cout << "CPU wall time: " << wall.count() / 1000 << " ms\n";
  std::cout << "GPU wall time: " << prof_info.gpu_wall.count() / 1000 << " ms\n";
  std::cout << "GPU pure time: " << prof_info.gpu_pure.count() / 1000 << " ms\n";

  if (correct) {
    std::cout << "GPU matrix multiplication works fine\n";
    return EXIT_SUCCESS;
  } else {
    std::cout << "GPU matrix multiplication is borked\n";
    return EXIT_FAILURE;
  }

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