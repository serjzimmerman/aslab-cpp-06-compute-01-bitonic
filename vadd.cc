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
#include "utils.hpp"

#include <CL/opencl.hpp>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>

namespace app {

static const std::string adder_kernel =
    R"(__kernel void vec_add(__global int *A, __global int *B, __global int *C) {
  int i = get_global_id(0);
  C[i] = A[i] + B[i];
})";

class vecadd : private clutils::platform_selector {
  cl::Context      m_ctx;
  cl::CommandQueue m_queue;

  cl::Program                                           m_program;
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> m_functor;

public:
  static constexpr clutils::platform_version c_api_version = {2, 2};

  vecadd()
      : clutils::platform_selector{c_api_version}, m_ctx{m_device}, m_queue{m_ctx},
        m_program{m_ctx, adder_kernel, true}, m_functor{m_program, "vec_add"} {}

  std::vector<cl_int> add(std::span<const cl_int> spa, std::span<const cl_int> spb) {
    if (spa.size() != spb.size()) throw std::invalid_argument{"Mismatched vector sizes"};

    const auto max_local_size = m_device.template getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    const auto size = ((spa.size() / max_local_size) + 1) * max_local_size;
    const auto bin_size = sizeof(cl_int) * size;

    std::vector<cl_int> cvec;
    cvec.resize(size);

    cl::Buffer abuf = {m_ctx, CL_MEM_READ_ONLY, bin_size};
    cl::Buffer bbuf = {m_ctx, CL_MEM_READ_ONLY, bin_size};
    cl::Buffer cbuf = {m_ctx, CL_MEM_WRITE_ONLY, bin_size};

    cl::copy(m_queue, spa.begin(), spa.end(), abuf);
    cl::copy(m_queue, spb.begin(), spb.end(), bbuf);

    cl::NDRange     global = {size}, local = {max_local_size};
    cl::EnqueueArgs args = {m_queue, global, local};

    m_functor(args, abuf, bbuf, cbuf);

    cl::copy(m_queue, cbuf, cvec.begin(), cvec.end());
    cvec.resize(spa.size());

    return cvec;
  }
};

} // namespace app

int main(int, char *[]) try {
  app::vecadd adder;

  const auto print_array = [](auto name, auto vec) {
    std::cout << name << " := { ";
    for (const auto &v : vec) {
      std::cout << v << " ";
    }
    std::cout << "}\n";
  };

  std::vector<cl_int> a = {1, 2, 3, 4}, b = {8, 1, 2, 3};

  print_array("A", a);
  print_array("B", b);

  auto res = adder.add(a, b);

  print_array("C", res);
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