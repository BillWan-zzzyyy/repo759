#include "matmul.h"

#include <algorithm>

// C has to start from zero since every version only does C += A * B.

// mmul1: loop order (i, j, k). The inner loop is the dot product of row i
// of A and column j of B, so B is read with a stride of n.
void mmul1(const double* A, const double* B, double* C, const unsigned int n) {
  std::fill(C, C + n * n, 0.0);
  for (unsigned int i = 0; i < n; ++i) {
    for (unsigned int j = 0; j < n; ++j) {
      for (unsigned int k = 0; k < n; ++k) {
        C[i * n + j] += A[i * n + k] * B[k * n + j];
      }
    }
  }
}

// mmul2: loop order (i, k, j). A[i][k] is fixed in the inner loop and
// row k of B and row i of C are walked through contiguously.
void mmul2(const double* A, const double* B, double* C, const unsigned int n) {
  std::fill(C, C + n * n, 0.0);
  for (unsigned int i = 0; i < n; ++i) {
    for (unsigned int k = 0; k < n; ++k) {
      for (unsigned int j = 0; j < n; ++j) {
        C[i * n + j] += A[i * n + k] * B[k * n + j];
      }
    }
  }
}

// mmul3: loop order (j, k, i). B[k][j] is fixed in the inner loop, but
// A and C are both walked down a column (stride n).
void mmul3(const double* A, const double* B, double* C, const unsigned int n) {
  std::fill(C, C + n * n, 0.0);
  for (unsigned int j = 0; j < n; ++j) {
    for (unsigned int k = 0; k < n; ++k) {
      for (unsigned int i = 0; i < n; ++i) {
        C[i * n + j] += A[i * n + k] * B[k * n + j];
      }
    }
  }
}

// mmul4: same loops as mmul1, only A and B are std::vector<double>.
void mmul4(const std::vector<double>& A, const std::vector<double>& B,
           double* C, const unsigned int n) {
  std::fill(C, C + n * n, 0.0);
  for (unsigned int i = 0; i < n; ++i) {
    for (unsigned int j = 0; j < n; ++j) {
      for (unsigned int k = 0; k < n; ++k) {
        C[i * n + j] += A[i * n + k] * B[k * n + j];
      }
    }
  }
}
