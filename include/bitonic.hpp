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
#include "selector.hpp"
#include "utils.hpp"

#include <bit>
#include <chrono>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

#define STRINGIFY0(v) #v
#define STRINGIFY(v) STRINGIFY0(v)

#ifndef TYPE__
#define TYPE__ int
#endif

namespace bitonic {

template <typename T> struct i_bitonic_sort {
  virtual ~i_bitonic_sort() {}

  void sort(std::span<T> container, clutils::profiling_info *time = nullptr) { return operator()(container, time); }
  virtual void operator()(std::span<T>, clutils::profiling_info *) = 0;
};

template <typename T> class gpu_bitonic : public i_bitonic_sort<T>, protected clutils::platform_selector {
protected:
  cl::Context      m_ctx;
  cl::CommandQueue m_queue;

  static constexpr clutils::platform_version cl_api_version = {2, 2};

  gpu_bitonic()
      : clutils::platform_selector{cl_api_version}, m_ctx{m_device}, m_queue{m_ctx, cl::QueueProperties::Profiling} {}

  using func_signature = cl::Event(cl::Buffer, int, int);

  void run_boilerplate(std::span<T> container, std::function<func_signature> func, clutils::profiling_info *time) {
    unsigned size = container.size();
    if (std::popcount(size) != 1 || size < 2) throw std::runtime_error("Only power-of-two sequences are supported");

    auto wall_start = std::chrono::high_resolution_clock::now();

    cl::Event event;
    int       steps_n = std::countr_zero(size);
    for (int step = 0; step < steps_n; ++step) {
      int stage_n = step;
      for (int stage = stage_n; stage >= 0; --stage) {
        cl::Buffer buff = {m_ctx, CL_MEM_READ_WRITE, clutils::sizeof_container(container)};
        cl::copy(m_queue, container.begin(), container.end(), buff);
        event = func(buff, step, stage);
        event.wait();
        cl::copy(m_queue, buff, container.begin(), container.end());
      }
    }
    auto                     wall_end = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds pure_start{event.getProfilingInfo<CL_PROFILING_COMMAND_START>()},
        pure_end{event.getProfilingInfo<CL_PROFILING_COMMAND_END>()};
    auto pure = std::chrono::duration_cast<std::chrono::milliseconds>(pure_end - pure_start);
    auto wall = std::chrono::duration_cast<std::chrono::milliseconds>(wall_end - wall_start);

    if (time) *time = {pure, wall};
  }
};

template <typename T> class naive_bitonic : public gpu_bitonic<T> {
  struct kernel {
    using functor_type = cl::KernelFunctor<cl::Buffer, int, int>;
    static std::string source(std::string type) {
      static const std::string naive_source = R"(
      __kernel void naive_bitonic (__global TYPE *buff, int step, int stage) {
        int i = get_global_id(0);
        int seq_len = 1 << (stage + 1);
        int power_of_two = 1 << (step - stage);
        int seq_n = i / seq_len;
        int odd = seq_n / power_of_two;
        bool increasing = ((odd % 2) == 0);
        int halflen = seq_len / 2;
        if (i < (seq_len * seq_n) + halflen) {
          int   j = i + halflen;
          if (((buff[i] > buff[j]) && increasing) ||
              ((buff[i] < buff[j]) && !increasing)) {
            TYPE tmp = buff[j];
            buff[i] = buff[j];
            buff[j] = tmp;
          }
        }
      })";

      auto type_macro_def = clutils::kernel_define("TYPE", type);
      return type_macro_def + naive_source;
    }

    static std::string entry() { return "naive_bitonic"; }
  };

private:
  cl::Program          m_program;
  kernel::functor_type m_functor;

public:
  naive_bitonic()
      : gpu_bitonic<T>{}, m_program{gpu_bitonic<T>::m_ctx, kernel::source(STRINGIFY(TYPE__)), true},
        m_functor{m_program, kernel::entry()} {}

  void operator()(std::span<T> container, clutils::profiling_info *time = nullptr) override {
    const auto func = [&](auto buf, auto step, auto stage) {
      cl::EnqueueArgs args = {gpu_bitonic<T>::m_queue, {container.size()}};
      return m_functor(args, buf, step, stage);
    };
    return gpu_bitonic<T>::run_boilerplate(container, func, time);
  }
};
} // namespace bitonic