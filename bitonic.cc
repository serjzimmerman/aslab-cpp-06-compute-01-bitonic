/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * <tsimmerman.ss@phystech.edu>, <alex.rom23@mail.ru> wrote this file.  As long as you
 * retain this notice you can do whatever you want with this stuff. If we meet
 * some day, and you think this stuff is worth it, you can buy me a beer in
 * return.
 * ----------------------------------------------------------------------------
 */

#include "bitonic.hpp"

#include <iostream>
#include <memory>
#include <vector>

int main() {
  std::unique_ptr<bitonic::i_bitonic_sort<int>> int_sorter;
  int_sorter = std::make_unique<bitonic::naive_bitonic<int>>();

  std::vector<int> A = {20, 22, 2,  19, 1,  16, 9,  0, 12, 24, 18, 8,  16, 4,  24, 29,
                        4,  5,  24, 0,  15, 20, 16, 9, 15, 2,  17, 32, 8,  11, 28, 19};
  int_sorter->sort(std::span{A});

  std::cout << "sorted:\n{";
  for (int elem : A)
    std::cout << elem << " ";
  std::cout << "}\n";
}