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

#include "popl.hpp"

#ifdef PAR_CPU_SORT

#include <parallel/algorithm>
#define CPU_SORT ::__gnu_parallel::sort
#define CPU_SORT_NAME "__gnu_parallel::sort"

#else

#define CPU_SORT std::sort
#define CPU_SORT_NAME "std::sort"

#endif

using vector_type = std::vector<TYPE__>;

void vprint(const std::string title, const auto &vec) {
  std::cout << title << ": { ";
  for (auto &elem : vec) {
    std::cout << elem << " ";
  }
  std::cout << "}\n";
}

int validate_results(const auto &origin, const auto &res, const auto &check, bool print_on_failure) {
  if (std::equal(res.begin(), res.end(), check.begin())) {
    std::cout << "Bitonic sort works fine\n";
    return EXIT_SUCCESS;
  }

  std::cout << "Bitonic sort is broken\n";

  if (print_on_failure) {
    vprint("Original", origin);
    vprint("Result", res);
    vprint("Correct", check);
  }

  return EXIT_FAILURE;
}

template <typename T> struct type_name {};
template <> struct type_name<TYPE__> {
  static constexpr const char *name_str = STRINGIFY(TYPE__);
};

int main(int argc, char **argv) try {
  const auto maximum = std::numeric_limits<TYPE__>::max(), minimum = std::numeric_limits<TYPE__>::min();

  popl::OptionParser op("Avaliable options");
  auto help_option = op.add<popl::Switch>("h", "help", "Print this help message");
  auto print_option = op.add<popl::Switch>("p", "print", "Print on failure");
  auto skip_option = op.add<popl::Switch>("s", "skip", "Skip comparing with std::sort");

  auto lower_option = op.add<popl::Implicit<TYPE__>>("", "lower", "Lower bound", minimum);
  auto upper_option = op.add<popl::Implicit<TYPE__>>("", "upper", "Upper bound", maximum);

  auto num_option = op.add<popl::Implicit<unsigned>>("", "num", "Length of the array to sort = 2^n", 24);
  auto kernel_option =
      op.add<popl::Implicit<std::string>>("", "kernel", "Which kernel to use: naive, cpu, local", "naive");
  auto lsz_option = op.add<popl::Implicit<unsigned>>("", "lsz", "Local memory size", 256);

  op.parse(argc, argv);

  if (help_option->is_set()) {
    std::cout << op << "\n ";
    return EXIT_SUCCESS;
  }

  const bool skip_std_sort = skip_option->is_set(), print_on_failure = print_option->is_set();
  const auto lower = lower_option->value(), upper = upper_option->value();
  const auto num = num_option->value();
  const auto kernel_name = kernel_option->value();
  const auto lsz = lsz_option->value();

  if (lower >= upper) {
    std::cout << "Error: lower bound can't be greater than the upper bound\n";
    return EXIT_FAILURE;
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

  if (kernel_name != "local" && lsz_option->is_set()) {
    std::cout << "Warning: local size provided but kernel used is not \"local\", ignoring --lsz option\n";
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
    CPU_SORT(check.begin(), check.end());
    auto wall_end = std::chrono::high_resolution_clock::now();
    wall = std::chrono::duration_cast<std::chrono::milliseconds>(wall_end - wall_start);
  }

  clutils::profiling_info prof_info;
  auto vec = origin;

  sorter->sort(vec, &prof_info);

  if (!skip_std_sort) std::cout << CPU_SORT_NAME << " wall time: " << wall.count() << " ms\n";

  std::cout << "bitonic wall time: " << prof_info.wall.count() << " ms\n";
  std::cout << "bitonic pure time: " << prof_info.pure.count() << " ms\n";

  print_sep();

  if (skip_std_sort) return EXIT_SUCCESS;
  return validate_results(origin, vec, check, print_on_failure);

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
