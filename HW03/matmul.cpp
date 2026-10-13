#include "matmul.h"

#include <algorithm>

// Parallel version of mmul2 from HW02 (loop order i, k, j).
void mmul(const float* A, const float* B, float* C, const std::size_t n) {
  // C has to start from zero since the loops only do C += A * B
  std::fill(C, C + n * n, 0.0f);

  // split the rows of C between the threads, each thread only writes its own
  // rows so there is no race on C
#pragma omp parallel for
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < n; ++k) {
      for (std::size_t j = 0; j < n; ++j) {
        C[i * n + j] += A[i * n + k] * B[k * n + j];
      }
    }
  }
}
