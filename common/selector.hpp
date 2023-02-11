/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * <tsimmerman.ss@phystech.edu>, <alex.rom23@mail.ru> wrote this file.  As long as you
 * retain this notice you can do whatever you want with this stuff. If we meet
 * some day, and you think this stuff is worth it, you can buy me a beer in
 * return.
 * ----------------------------------------------------------------------------
 */

#include <cassert>
#include <charconv>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>

namespace clutils {

struct platform_version {
  int         major, minor;
  std::string platform_specific;
};

// Pass only valid opencl version string to this function
inline platform_version decode_platform_version(std::string version_string) {
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

  return platform_version{major.value(), minor.value(), platform_specific};
}

class cl_platform_selector {};

}; // namespace clutils