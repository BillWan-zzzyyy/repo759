#include <omp.h>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

#include "matmul.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;

// Returns the parsed value, or 0 if str is not a positive integer.
long parse_positive(const char* str) {
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(str, &end, 10);
  if (errno != 0 || end == str || *end != '\0' || parsed <= 0) {
    return 0;
  }
  return parsed;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " n t\n";
    return 1;
  }

  // n is the matrix size, t is the number of threads
  const long n_arg = parse_positive(argv[1]);
  const long t = parse_positive(argv[2]);
  if (n_arg == 0 || t == 0 || t > 20) {
    std::cerr << "n must be a positive integer and t must be in [1, 20].\n";
    return 1;
  }
  const std::size_t n = static_cast<std::size_t>(n_arg);

  float* A = new float[n * n];
  float* B = new float[n * n];
  float* C = new float[n * n];

  // same range as HW02 task3, stored in row-major order
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t i = 0; i < n * n; ++i) {
    A[i] = dist(gen);
    B[i] = dist(gen);
  }

  // mmul has no thread argument, so set the number of threads here
  omp_set_num_threads(static_cast<int>(t));

  high_resolution_clock::time_point start = high_resolution_clock::now();
  mmul(A, B, C, n);
  high_resolution_clock::time_point stop = high_resolution_clock::now();
  duration<double, std::milli> elapsed = stop - start;

  // first and last element of C, then the time in ms
  std::cout << C[0] << "\n";
  std::cout << C[n * n - 1] << "\n";
  std::cout << elapsed.count() << "\n";

  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}
