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

#include "i_bitonic_sort.hpp"

#include <bit>
#include <iterator>

namespace bitonic {

template <typename T> class simple_bitonic_sort : public i_bitonic_sort<T> {

public:
  // start, finish - 2^n range of elements to sort
  void operator()(std::span<T> container) override {
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
} // namespace bitonic