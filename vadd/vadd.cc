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

#include <cassert>

int main() {
  clutils::platform_selector pl_selector{{2, 0}};
  auto                       required_extensions = clutils::get_required_device_extensions();
  auto suitable_devices = clutils::enumerate_suitable_devices(pl_selector.get_available_devices(), required_extensions);
  assert(!suitable_devices.empty());
  auto        device = std::move(suitable_devices.front());
  const auto  context_properties = pl_selector.get_context_properties();
  cl::Context context{CL_DEVICE_TYPE_GPU, context_properties.data()};
}