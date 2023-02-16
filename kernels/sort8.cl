/* Hardcoded sorting network of 8 elements
 *
 *  @kernel    ( {"name" : "sort8_kernel", "entry" : "sort8"} )
 *  @signature ( ["cl::Buffer"] )
 *  @macros    ( [{"type" : "std::string", "name": "TYPE"}] )
 *
 */

#define SWAP_IF(a, b)                                                                                                  \
  if (a > b) {                                                                                                         \
    TYPE temp = a;                                                                                                     \
    a = b;                                                                                                             \
    b = temp;                                                                                                          \
  }

__kernel void sort8(__global TYPE *buf) {
  int i = 8 * get_global_id(0);

  TYPE array[8];
  array[0] = buf[i + 0];
  array[1] = buf[i + 1];
  array[2] = buf[i + 2];
  array[3] = buf[i + 3];
  array[4] = buf[i + 4];
  array[5] = buf[i + 5];
  array[6] = buf[i + 6];
  array[7] = buf[i + 7];

  // Sorting network for 4 elements:
  SWAP_IF(array[0], array[2]);
  SWAP_IF(array[1], array[3]);
  SWAP_IF(array[4], array[6]);
  SWAP_IF(array[5], array[7]);
  SWAP_IF(array[0], array[4]);
  SWAP_IF(array[1], array[5]);
  SWAP_IF(array[2], array[6]);
  SWAP_IF(array[3], array[7]);
  SWAP_IF(array[0], array[1]);
  SWAP_IF(array[2], array[3]);
  SWAP_IF(array[4], array[5]);
  SWAP_IF(array[6], array[7]);
  SWAP_IF(array[2], array[4]);
  SWAP_IF(array[3], array[5]);
  SWAP_IF(array[1], array[4]);
  SWAP_IF(array[3], array[6]);
  SWAP_IF(array[1], array[2]);
  SWAP_IF(array[3], array[4]);
  SWAP_IF(array[5], array[6]);

  buf[i + 0] = array[0];
  buf[i + 1] = array[1];
  buf[i + 2] = array[2];
  buf[i + 3] = array[3];
  buf[i + 4] = array[4];
  buf[i + 5] = array[5];
  buf[i + 6] = array[6];
  buf[i + 7] = array[7];
}