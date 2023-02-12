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
#include <exception>
#include <fstream>
#include <sstream>

const cl::Program create_program_compile(const std::string &filename) {
  std::fstream fs{filename, std::ifstream::in};
  if (!fs.is_open()) throw std::runtime_error("Failed to open kernel file");
  std::stringstream ss;
  ss << fs.rdbuf();
  fs.close();
  return cl::Program{ss.str(), true};
}

int main() try {
  std::array<float, 5> A{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::array<float, 5> B{-1.0f, -2.0f, -3.0f, -4.0f, -5.0f};
  std::array<float, 5> C{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  std::cout << "A:\n{";
  for (auto elem : A)
    std::cout << " " << elem;
  std::cout << " }\n";

  std::cout << "B:\n{";
  for (auto elem : B)
    std::cout << " " << elem;
  std::cout << " }\n";

  clutils::platform_selector pl_selector{{2, 0}};
  auto                       required_extensions = clutils::get_required_device_extensions();
  auto suitable_devices = clutils::enumerate_suitable_devices(pl_selector.get_available_devices(), required_extensions);
  assert(!suitable_devices.empty());
  auto        device = std::move(suitable_devices.front());
  const auto  context_properties = pl_selector.get_context_properties();
  cl::Context context{CL_DEVICE_TYPE_GPU, context_properties.data()};

  const auto       program = create_program_compile("vadd/vadd.cl");
  cl::Kernel       kernel{program, "vadd"};
  cl::CommandQueue queue{context, device};

  cl::Buffer A_buf{context, CL_MEM_READ_ONLY, clutils::sizeof_container(A)};
  cl::Buffer B_buf{context, CL_MEM_READ_ONLY, clutils::sizeof_container(B)};
  cl::Buffer C_buf{context, CL_MEM_WRITE_ONLY, clutils::sizeof_container(C)};
  cl::copy(A.begin(), A.end(), A_buf);
  cl::copy(B.begin(), B.end(), B_buf);

  kernel.setArg(0, sizeof(cl::Buffer), &A_buf);
  kernel.setArg(1, sizeof(cl::Buffer), &B_buf);
  kernel.setArg(2, sizeof(cl::Buffer), &C_buf);

  cl::NDRange                                           glob_range{C.size()};
  cl::NDRange                                           loc_range{1};
  cl::EnqueueArgs                                       args{queue, glob_range, loc_range};
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> vadd_func{program, "vadd"};

  cl::Event evnt = vadd_func(args, A_buf, B_buf, C_buf);
  evnt.wait();
  cl::copy(queue, C_buf, C.begin(), C.end());

  std::cout << "A + B:\n{";
  for (auto elem : C)
    std::cout << " " << elem;
  std::cout << " }\n";

} catch (cl::Error &e) {
  std::cerr << "OpenCL error: " << e.what() << "\n";
} catch (std::exception &e) {
  std::cerr << "Encountered error: " << e.what() << "\n";
} catch (...) {
  std::cerr << "Unknown error\n";
}