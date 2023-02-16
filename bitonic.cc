/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * <tsimmerman.ss@phystech.edu>, <alex.rom23@mail.ru> wrote this file.  As long as you
 * retain this notice you can do whatever you want with this stuff. If we meet
 * some day, and you think this stuff is worth it, you can buy us a beer in
 * return.
 * ----------------------------------------------------------------------------
 */

#define STRINGIFY0(v) #v
#define STRINGIFY(v) STRINGIFY0(v)

#ifndef TYPE__
#define TYPE__ int
#endif

#include "bitonic.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <boost/program_options/option.hpp>

using vector_type = std::vector<TYPE__>;

namespace po = boost::program_options;

void vprint(const std::string title, const auto &vec) {
  std::cout << title << ": { ";
  for (auto &elem : vec) {
    std::cout << elem << " ";
  }
  std::cout << "}\n";
}

int validate_results(const auto &origin, const auto &res, const auto &check) {
  if (std::equal(res.begin(), res.end(), check.begin())) {
    std::cout << "Bitonic sort works fine\n";
    return EXIT_SUCCESS;
  } else {
    std::cout << "Bitonic sort is broken\n";
    vprint("Original", origin);
    vprint("Result", res);
    vprint("Correct", check);
    return EXIT_FAILURE;
  }
}

template <typename T> struct type_name {};
template <> struct type_name<TYPE__> {
  static constexpr const char *name_str = STRINGIFY(TYPE__);
};

int main(int argc, char **argv) try {
  po::options_description desc("Avaliable options");

  TYPE__ lower, upper;
  unsigned num, lsz;

  std::string kernel_name;
  const auto maximum = std::numeric_limits<TYPE__>::max(), minimum = std::numeric_limits<TYPE__>::min();
  desc.add_options()("help,h", "Print this help message")("print,p", "Print on failure")("skip,s", "Skip std::sort")(
      "lower,l", po::value<TYPE__>(&lower)->default_value(minimum), "Lower bound for random integers")(
      "upper,u", po::value<TYPE__>(&upper)->default_value(maximum), "Upper bound for random integers")(
      "num,n", po::value<unsigned>(&num)->default_value(2), "n dor 2^n length vector of integers")(
      "kernel,k", po::value<std::string>(&kernel_name)->default_value("naive"),
      "Which kernel to use: naive, cpu, local")("lsz", po::value<unsigned>(&lsz)->default_value(256),
                                                "Local iteration size");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  bool skip_std_sort = vm.count("skip");
  if (vm.count("help")) {
    std::cout << desc << "\n ";
    return EXIT_SUCCESS;
  }

  const unsigned size = (1 << num);
  std::unique_ptr<bitonic::i_bitonic_sort<TYPE__>> sorter;

  if (kernel_name == "naive") {
    sorter = std::make_unique<bitonic::naive_bitonic<TYPE__, type_name<TYPE__>>>();
  } else if (kernel_name == "cpu") {
    sorter = std::make_unique<bitonic::cpu_bitonic_sort<TYPE__>>();
  } else if (kernel_name == "local") {
    sorter = std::make_unique<bitonic::local_bitonic<TYPE__, type_name<TYPE__>>>(lsz);
  } else {
    std::cout << "Unknown type of kernel: " << kernel_name << "\n ";
    return EXIT_FAILURE;
  }

  const auto print_sep = []() { std::cout << " -------- \n"; };

  std::cout << "Sorting vector of size = " << size << "\n";
  print_sep();

  vector_type origin;
  origin.resize(size);

  auto rand_gen = clutils::create_random_number_generator<TYPE__>(lower, upper);
  rand_gen(origin);

  std::chrono::milliseconds wall;
  auto check = origin;

  if (!skip_std_sort) {
    auto wall_start = std::chrono::high_resolution_clock::now();
    std::sort(check.begin(), check.end());
    auto wall_end = std::chrono::high_resolution_clock::now();
    wall = std::chrono::duration_cast<std::chrono::milliseconds>(wall_end - wall_start);
  }

  clutils::profiling_info prof_info;
  auto vec = origin;

  sorter->sort(vec, &prof_info);

  if (!skip_std_sort) std::cout << "std::sort wall time: " << wall.count() << " ms\n";

  std::cout << "bitonic wall time: " << prof_info.wall.count() << " ms\n";
  std::cout << "bitonic pure time: " << prof_info.pure.count() << " ms\n";

  print_sep();

  if (skip_std_sort) return EXIT_SUCCESS;
  return validate_results(origin, vec, check);

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
