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
#include <memory>
#include <span>
#include <stdexcept>

namespace bitonic {

struct i_bitonic_kernel_source {
  virtual std::string get_kernel_source() const = 0;
  virtual std::string get_kernel_name() const = 0;

  virtual ~i_bitonic_kernel_source() {}
};

template <typename T> struct i_bitonic_sort {
public:
  virtual ~i_bitonic_sort() {}

  // start, finish - 2^n range of elements to sort
  virtual void operator()(std::span<T>) const = 0;
};

template <typename T> struct simple_bitonic_sort : public i_bitonic_sort<T> {

  // start, finish - 2^n range of elements to sort
  void operator()(std::span<T> container) const override {
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

template <typename T> class gpu_bitonic_sort : public i_bitonic_sort<T>, protected clutils::platform_selector {
private:
  std::unique_ptr<i_bitonic_kernel_source> m_source;

  cl::Context      m_ctx;
  cl::CommandQueue m_queue;
  cl::Program      m_program;

public:
  gpu_bitonic_sort(std::unique_ptr<i_bitonic_kernel_source> &&source)
      : m_source{std::move(source)}, m_ctx{m_device}, m_queue{m_ctx, cl::QueueProperties::Profiling},
        m_program{m_ctx, m_source->get_kernel_source, clutils::opencl_build_option::strict} {}

  void operator()(std::span<T> container) const override {
    unsigned size = container.size();
    if (std::popcount(size) != 1 || size < 2) throw std::runtime_error("Only power-of-two sequences are supported");
  }
};
} // namespace bitonic