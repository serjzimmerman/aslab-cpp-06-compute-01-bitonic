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

#include <span>

namespace bitonic {

template <typename T> struct i_bitonic_sort {
public:
  virtual ~i_bitonic_sort() {}

  // start, finish - 2^n range of elements to sort
  virtual void operator()(std::span<T>) = 0;
};

} // namespace bitonic