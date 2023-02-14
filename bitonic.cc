/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * <tsimmerman.ss@phystech.edu>, <alex.rom23@mail.ru> wrote this file.  As long as you
 * retain this notice you can do whatever you want with this stuff. If we meet
 * some day, and you think this stuff is worth it, you can buy me a beer in
 * return.
 * ----------------------------------------------------------------------------
 */

#define TYPE__ int

#include "bitonic.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <boost/program_options/option.hpp>

namespace po = boost::program_options;

template <typename T> void vprint(const std::string title, const std::vector<T> &vec) {
  std::cout << title << ": { ";
  for (auto &elem : vec)
    std::cout << elem << " ";
  std::cout << "}\n";
}

template <typename T>
int validate_results(const std::vector<T> &origin, const std::vector<T> &res, const std::vector<T> &check) {
  if (std::equal(res.begin(), res.end(), check.begin())) {
    std::cout << "Bitonic sort works fine\n";
    return EXIT_SUCCESS;
  } else {
    std::cout << "Bitonic sort is broken\n";
    vprint("Origin", origin);
    vprint("Result", res);
    vprint("Correct", check);
    return EXIT_FAILURE;
  }
}

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


int main(int argc, char **argv) try {
  po::options_description desc("Avaliable options");

  TYPE__   lower{}, upper{};
  unsigned num{};

  std::string kernel_name;
  desc.add_options()("help,h", "Print this help message")("print,p", "Print on failure")("skip,s", "Skip std sort")(
      "lower,l", po::value<TYPE__>(&lower)->default_value(0), "Lower bound for random number")(
      "upper,u", po::value<TYPE__>(&upper)->default_value(100),
      "Upper bound for random number")("num,n", po::value<unsigned>(&num)->default_value(16777216), "Random numbers' count. Should be a power of 2")(
      "kernel,k", po::value<std::string>(&kernel_name)->default_value("naive"), "Which kernel to use: naive");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  bool skip_std_sort = vm.count("skip");
  if (vm.count("help")) {
    std::cout << desc << "\n ";
    return EXIT_SUCCESS;
  }

  std::unique_ptr<bitonic::i_bitonic_sort<TYPE__>> sorter;

  if (kernel_name == "naive") {
    sorter = std::make_unique<bitonic::naive_bitonic<TYPE__>>();
  } else {
    std::cout << "Unknown type of kernel: " << kernel_name << "\n ";
    return EXIT_FAILURE;
  }

  std::cout << "Sorting vector of size " << num << "...\n";
  std::vector<TYPE__> origin(num);

  auto rand_gen = create_random_number_generator<TYPE__>(lower, upper);
  rand_gen(origin);

  std::chrono::milliseconds wall{};

  std::vector<TYPE__> check = origin;
  if (!skip_std_sort) {
    auto wall_start = std::chrono::high_resolution_clock::now();
    std::sort(check.begin(), check.end());
    auto wall_end = std::chrono::high_resolution_clock::now();
    wall = std::chrono::duration_cast<std::chrono::milliseconds>(wall_end - wall_start);
  }

  clutils::profiling_info prof_info;
  std::vector<TYPE__>     vec = origin;

  sorter->sort(std::span{vec}, &prof_info);

  if (!skip_std_sort) std::cout << "std::sort wall time: " << wall.count() << " ms\n";
  std::cout << "GPU wall time: " << prof_info.gpu_wall.count() << " ms\n";
  std::cout << "GPU pure time: " << prof_info.gpu_pure.count() << " ms\n";

  if (skip_std_sort) return EXIT_SUCCESS;
  return validate_results(origin, vec, check);

} catch (cl::Error &e) {
  std::cerr << "OpenCL error: " << e.what() << "(" << e.err() << ")\n";
} catch (std::exception &e) {
  std::cerr << "Encountered error: " << e.what() << "\n";
} catch (...) {
  std::cerr << "Unknown error\n";
}
