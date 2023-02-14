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
#include <random>
#include <span>
#include <stdexcept>
#include <string>

#include <boost/algorithm/string/replace.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/option.hpp>

#define STRINGIFY0(v) #v
#define STRINGIFY(v) STRINGIFY0(v)

#ifndef TYPE__
#define TYPE__ int
#endif

namespace po = boost::program_options;

namespace app {

static const std::string adder_kernel =
    R"(__kernel void vec_add(__global TYPE *A, __global TYPE *B, __global TYPE *C) {
  int i = get_global_id(0);
  C[i] = A[i] + B[i];
})";

std::string substitute_type(std::string input) {
  return boost::algorithm::replace_all_copy(input, "TYPE", STRINGIFY(TYPE__));
}

class vecadd : private clutils::platform_selector {
  cl::Context m_ctx;
  cl::CommandQueue m_queue;

  cl::Program m_program;
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> m_functor;

public:
  static constexpr clutils::platform_version c_api_version = {2, 2};

  vecadd()
      : clutils::platform_selector{c_api_version}, m_ctx{m_device}, m_queue{m_ctx, cl::QueueProperties::Profiling},
        m_program{m_ctx, substitute_type(adder_kernel), true}, m_functor{m_program, "vec_add"} {}

  std::vector<TYPE__> add(std::span<const TYPE__> spa, std::span<const TYPE__> spb,
                          std::chrono::microseconds *time = nullptr) {
    if (spa.size() != spb.size()) throw std::invalid_argument{"Mismatched vector sizes"};

    const auto size = spa.size();
    const auto bin_size = clutils::sizeof_container(spa);

    std::vector<TYPE__> cvec;
    cvec.resize(size);

    cl::Buffer abuf = {m_ctx, CL_MEM_READ_ONLY, bin_size};
    cl::Buffer bbuf = {m_ctx, CL_MEM_READ_ONLY, bin_size};
    cl::Buffer cbuf = {m_ctx, CL_MEM_WRITE_ONLY, bin_size};

    cl::copy(m_queue, spa.begin(), spa.end(), abuf);
    cl::copy(m_queue, spb.begin(), spb.end(), bbuf);

    cl::NDRange global = {size};
    cl::EnqueueArgs args = {m_queue, global};

    auto evnt = m_functor(args, abuf, bbuf, cbuf);

    evnt.wait();
    std::chrono::nanoseconds time_start{evnt.getProfilingInfo<CL_PROFILING_COMMAND_START>()},
        time_end{evnt.getProfilingInfo<CL_PROFILING_COMMAND_END>()};

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    if (time) *time = duration;

    cl::copy(m_queue, cbuf, cvec.begin(), cvec.end());
    return cvec;
  }
};

} // namespace app

int main(int argc, char *argv[]) try {
  po::options_description desc("Available options");

  int lower, upper;
  unsigned count;
  desc.add_options()("help,h", "Print this help message")("lower,l", po::value<int>(&lower)->default_value(0),
                                                          "Low bound for random integer")(
      "upper,u", po::value<int>(&upper)->default_value(32),
      "Upper bound for random integer")("count,c", po::value<unsigned>(&count)->default_value(1048576),
                                        "Length of arrays to sum")("print,p", "Verbose print");

  bool print = false;

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  print = vm.count("print");

  app::vecadd adder;

  const auto print_array = [print](auto name, auto vec) {
    if (!print) return;
    std::cout << name << " := { ";
    for (const auto &v : vec) {
      std::cout << v << " ";
    }
    std::cout << "}\n";
  };

  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};

  std::uniform_int_distribution<TYPE__> dist{lower, upper};

  const auto random = [&dist, &mersenne_engine] { return dist(mersenne_engine); };
  const auto fill_random_vector = [random]<typename T>(std::vector<T> &vec, auto count) {
    std::generate_n(std::back_inserter(vec), count, random);
  };

  std::vector<TYPE__> a, b;
  fill_random_vector(a, count);
  fill_random_vector(b, count);

  print_array("A", a);
  print_array("B", b);

  std::chrono::microseconds pure_time;

  auto res = adder.add(a, b, &pure_time);
  print_array("C", res);
  bool correct = (a.size() == res.size());

  if (correct) {
    for (size_t i = 0; i < a.size(); ++i) {
      if (a[i] + b[i] == res[i]) continue;
      std::cout << "Mismatch at position i = " << i << "\n";
      correct = false;
      break;
    }
  }

  std::cout << "GPU pure time: " << pure_time.count() << " us\n";

  if (correct) {
    std::cout << "GPU vector add works fine\n";
    return EXIT_SUCCESS;
  } else {
    std::cout << "GPU vector add is borked\n";
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