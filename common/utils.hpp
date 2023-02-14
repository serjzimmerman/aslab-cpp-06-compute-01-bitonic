/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * <tsimmerman.ss@phystech.edu>, <alex.rom23@mail.ru> wrote this file.  As long as you
 * retain this notice you can do whatever you want with this stuff. If we meet
 * some day, and you think this stuff is worth it, you can buy me a beer in
 * return.
 * ----------------------------------------------------------------------------
 */

#pragma once

#include "opencl_include.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace clutils {

template <typename... Ts> std::string kernel_define(std::string symbol, Ts... values) {
  std::stringstream ss;
  ss << "#define " << symbol << " ";
  ((ss << " " << values), ...);
  ss << "\n";
  return ss.str();
}

template <class T> inline std::size_t sizeof_container(const T &container) {
  return sizeof(typename T::value_type) * container.size();
}

struct profiling_info {
  std::chrono::milliseconds gpu_pure{};
  std::chrono::milliseconds gpu_wall{};
};

} // namespace clutils