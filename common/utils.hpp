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

#include <string>
#include <vector>

#include <iostream>

namespace clutils {

inline std::vector<std::string> get_required_device_extensions() {
  return {"cl_khr_byte_addressable_store", "cl_khr_global_int32_base_atomics", "cl_khr_global_int32_extended_atomics"};
}

using support_result = typename std::pair<bool, std::vector<std::string>>;

inline support_result device_supports_extensions(const cl::Device &device, const std::vector<std::string> &extensions) {
  const auto               supported_extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
  std::vector<std::string> missing_extensions;
  for (const auto &ext : extensions) {
    if (supported_extensions.find(ext) == std::string::npos) missing_extensions.push_back(ext);
  }
  return std::make_pair(missing_extensions.empty(), missing_extensions);
}

template <class T> inline std::size_t sizeof_container(const T &container) {
  return sizeof(typename T::value_type) * container.size();
}

} // namespace clutils