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

#include "utils.hpp"

#include <algorithm>
#include <cassert>
#include <charconv>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>

namespace clutils {

struct platform_version {
  int major, minor;
};

inline auto operator<=>(const platform_version &lhs, const platform_version &rhs) {
  if (auto cmp = lhs.major <=> rhs.major; cmp != 0) return cmp;
  return lhs.minor <=> rhs.minor;
}

struct platform_version_ext {
  platform_version ver;
  std::string      platform_specific;
};

// Pass only valid opencl version string to this function
inline platform_version_ext decode_platform_version(std::string version_string) {
  // This should always work https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetPlatformInfo.html
  auto version_start = version_string.find_first_of("0123456789");

  if (version_start == std::string::npos) throw std::invalid_argument{"OpenCL platform version string is invalid"};

  version_string = version_string.substr(version_start);
  auto        version_finish = version_string.find_first_of(" ");
  std::string major_minor_string = version_string.substr(0, version_finish);

  std::string platform_specific;
  if (version_finish != std::string::npos) {
    platform_specific = version_string.substr(version_finish + 1);
  }

  auto to_int = [](std::string_view s) -> std::optional<int> {
    if (int value; std::from_chars(s.begin(), s.end(), value).ec == std::errc{}) {
      return value;
    } else {
      return std::nullopt;
    }
  };

  // Split into minor/major version numbers
  auto sep = major_minor_string.find(".");
  if (sep == std::string::npos) throw std::invalid_argument{"OpenCL platform version string is invalid"};

  auto major_string = major_minor_string.substr(0, sep);
  auto minor_string = major_minor_string.substr(sep + 1);

  auto major = to_int(major_string), minor = to_int(minor_string);
  if (!major || !minor) throw std::invalid_argument{"OpenCL platform version string is invalid"};

  return platform_version_ext{platform_version{major.value(), minor.value()}, platform_specific};
}

class platform_selector {
protected:
  cl::Platform            m_platform;
  std::vector<cl::Device> m_devices;

public:
  platform_selector(platform_version min_ver) {
    std::vector<cl::Platform> platforms;

    cl::Platform::get(&platforms);
    auto chosen_platform = std::find_if(platforms.begin(), platforms.end(), [min_ver](auto p) {
      auto version = decode_platform_version(p.template getInfo<CL_PLATFORM_VERSION>());
      return (version.ver >= min_ver);
    });

    if (chosen_platform == platforms.end()) throw std::runtime_error{"No fitting OpenCL platforms found"};
    m_platform = *chosen_platform;
    m_platform.getDevices(CL_DEVICE_TYPE_GPU, &m_devices);
  }

  const std::array<cl_context_properties, 3> get_context_properties() const {
    return std::array<cl_context_properties, 3>{
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(m_platform()),
        0 // signals end of property list
    };
  }

  const std::vector<cl::Device> &get_available_devices() const & { return m_devices; }
};

inline std::vector<cl::Device> enumerate_suitable_devices(const std::vector<cl::Device>  &devices,
                                                          const std::vector<std::string> &extensions) {
  std::vector<cl::Device> suitable_devices;
  for (auto &d : devices) {
    auto [supports, missing] = device_supports_extensions(d, extensions);
    if (supports) suitable_devices.push_back(d);
  }
  return suitable_devices;
}

}; // namespace clutils