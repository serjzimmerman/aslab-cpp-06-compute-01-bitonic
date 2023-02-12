#include "opencl_include.hpp"
#include "selector.hpp"

int main() {
  clutils::platform_selector pl_selector{{2, 0}};
  auto                       required_extensions = clutils::get_required_device_extensions();
  auto suitable_devices = clutils::enumerate_suitable_devices(pl_selector.get_available_devices(), required_extensions);
  auto device = std::move(suitable_devices.front());
  const auto  context_properties = pl_selector.get_context_properties();
  cl::Context context{CL_DEVICE_TYPE_GPU, context_properties.data()};
}