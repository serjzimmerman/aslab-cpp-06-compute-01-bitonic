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

template <typename T> struct simple_bitonic_sort : public i_bitonic_sort<T> {

  // start, finish - 2^n range of elements to sort
  void operator()(std::span<T> container, clutils::profiling_info *) override {
    unsigned size = container.size();
    if (std::popcount(size) != 1 || size < 2) throw std::runtime_error("Only power-of-two sequences are supported");

    // for 2^n sequence of elements there are n steps
    int steps_n = std::countr_zero(size);
    for (int step = 0; step < steps_n; ++step) {
      int stage_n = step; // i'th step consists of i stages
      for (int stage = stage_n; stage >= 0; --stage) {
        int seq_len = 1 << (stage + 1);
        int pow_of_two = 1 << (step - stage);
        for (int i = 0; i < size; ++i) {
          int  seq_n = i / seq_len;
          int  odd = seq_n / pow_of_two;
          bool increasing = ((odd % 2) == 0);
          int  halflen = seq_len / 2;
          if (i < (seq_len * seq_n) + halflen) {
            int   j = i + halflen;
            auto &x = container[i];
            auto &y = container[j];
            if (((x > y) && increasing) || ((x < y) && !increasing)) std::swap(x, y);
          }
        }
      }
    }
  }
};

template <typename T> class gpu_bitonic : public i_bitonic_sort<T>, protected clutils::platform_selector {
protected:
  cl::Context      m_ctx;
  cl::CommandQueue m_queue;

  static constexpr clutils::platform_version cl_api_version = {2, 2};

  gpu_bitonic()
      : clutils::platform_selector{cl_api_version}, m_ctx{m_device}, m_queue{m_ctx, cl::QueueProperties::Profiling} {}

  using func_signature = cl::Event(cl::Buffer);

  void run_boilerplate(std::span<T> container, std::function<func_signature> func, clutils::profiling_info *time) {

    cl::Buffer buff = {m_ctx, CL_MEM_READ_WRITE, clutils::sizeof_container(container)};
    cl::copy(m_queue, container.begin(), container.end(), buff);
    auto event = func(buff);
    event.wait();
    cl::copy(m_queue, buff, container.begin(), container.end());

    std::chrono::nanoseconds pure_start{event.getProfilingInfo<CL_PROFILING_COMMAND_START>()},
        pure_end{event.getProfilingInfo<CL_PROFILING_COMMAND_END>()};
    auto pure = std::chrono::duration_cast<std::chrono::milliseconds>(pure_end - pure_start);
    if (time) time->gpu_pure += pure;
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
            TYPE tmp = buff[i];
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
    unsigned size = container.size();
    if (std::popcount(size) != 1 || size < 2) throw std::runtime_error("Only power-of-two sequences are supported");
    int steps_n = std::countr_zero(size);

    auto wall_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < steps_n; ++step) {
      int stage_n = step;
      for (int stage = stage_n; stage >= 0; --stage) {
        const auto func = [&](auto buf) {
          cl::EnqueueArgs args = {gpu_bitonic<T>::m_queue, {container.size()}};
          return m_functor(args, buf, step, stage);
        };
        gpu_bitonic<T>::run_boilerplate(container, func, time);
      }
    }

    auto wall_end = std::chrono::high_resolution_clock::now();

    if (time) time->gpu_wall = std::chrono::duration_cast<std::chrono::milliseconds>(wall_end - wall_start);
  }
};
} // namespace bitonic