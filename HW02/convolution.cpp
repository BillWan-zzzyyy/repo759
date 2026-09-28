#include "convolution.h"

// Value of f[i, j] with the padding rule from HW02:
// inside the image -> image value, edges -> 1, corners -> 0.
static float get_pixel(const float *image, std::size_t n, long i, long j) {
  // use signed indices here since i or j can be negative near the border
  const long size = static_cast<long>(n);
  const bool i_in = (i >= 0 && i < size);
  const bool j_in = (j >= 0 && j < size);

  if (i_in && j_in) {
    return image[i * size + j];
  }
  // exactly one index is out of range:edge
  if (i_in || j_in) {
    return 1.0f;
  }
  // both indices are out of range:corner
  return 0.0f;
}

void convolve(const float *image, float *output, std::size_t n,
              const float *mask, std::size_t m) {
  // (m - 1) / 2 shifts the mask so that its center sits on (x, y)
  const long half = static_cast<long>((m - 1) / 2);

  // loop over every pixel of the output image
  for (std::size_t x = 0; x < n; ++x) {
    for (std::size_t y = 0; y < n; ++y) {
      float sum = 0.0f;
      for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
          const long row = static_cast<long>(x + i) - half;
          const long col = static_cast<long>(y + j) - half;
          sum += mask[i * m + j] * get_pixel(image, n, row, col);
        }
      }
      output[x * n + y] = sum;
    }
  }
}
