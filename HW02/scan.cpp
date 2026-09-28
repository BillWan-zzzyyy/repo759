#include "scan.h"

void scan(const float *arr, float *output, std::size_t n) {
  // nothing to do for an empty array (also avoids reading arr[0])
  if (n == 0) {
    return;
  }

  // inclusive scan: output[i] = arr[0] + arr[1] + ... + arr[i]
  // each step reuses the previous partial sum, so this is O(n)
  output[0] = arr[0];
  for (std::size_t i = 1; i < n; ++i) {
    output[i] = output[i - 1] + arr[i];
  }
}
