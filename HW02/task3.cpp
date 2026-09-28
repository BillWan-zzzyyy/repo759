#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "matmul.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main() {
  const unsigned int n = 1024;

  // mmul1-3 take raw arrays, mmul4 takes vectors, so keep both copies.
  double* A = new double[n * n];
  double* B = new double[n * n];
  double* C = new double[n * n];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (unsigned int i = 0; i < n * n; ++i) {
    A[i] = dist(gen);
    B[i] = dist(gen);
  }
  std::vector<double> A_vec(A, A + n * n);
  std::vector<double> B_vec(B, B + n * n);

  high_resolution_clock::time_point start;
  high_resolution_clock::time_point stop;
  duration<double, std::milli> elapsed;

  std::cout << n << "\n";

  start = high_resolution_clock::now();
  mmul1(A, B, C, n);
  stop = high_resolution_clock::now();
  elapsed = stop - start;
  std::cout << elapsed.count() << "\n";
  std::cout << C[n * n - 1] << "\n";

  start = high_resolution_clock::now();
  mmul2(A, B, C, n);
  stop = high_resolution_clock::now();
  elapsed = stop - start;
  std::cout << elapsed.count() << "\n";
  std::cout << C[n * n - 1] << "\n";

  start = high_resolution_clock::now();
  mmul3(A, B, C, n);
  stop = high_resolution_clock::now();
  elapsed = stop - start;
  std::cout << elapsed.count() << "\n";
  std::cout << C[n * n - 1] << "\n";

  start = high_resolution_clock::now();
  mmul4(A_vec, B_vec, C, n);
  stop = high_resolution_clock::now();
  elapsed = stop - start;
  std::cout << elapsed.count() << "\n";
  std::cout << C[n * n - 1] << "\n";

  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}
