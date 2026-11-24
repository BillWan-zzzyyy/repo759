#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <random>

#include "matmul.cuh"

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
    std::cerr << "Usage: " << argv[0] << " n threads_per_block\n";
    return 1;
  }

  const long n_arg = parse_positive(argv[1]);
  const long t_arg = parse_positive(argv[2]);
  if (n_arg == 0 || t_arg == 0 || t_arg > 1024) {
    std::cerr << "n must be a positive integer and threads_per_block must be "
                 "in [1, 1024].\n";
    return 1;
  }
  const size_t n = static_cast<size_t>(n_arg);
  const unsigned int threads_per_block = static_cast<unsigned int>(t_arg);

  // A and B are n x n, row major, with values in [-1, 1]
  float* A = new float[n * n];
  float* B = new float[n * n];
  float* C = new float[n * n];
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n * n; ++i) {
    A[i] = dist(gen);
    B[i] = dist(gen);
  }

  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_C = nullptr;
  cudaMalloc(&d_A, n * n * sizeof(float));
  cudaMalloc(&d_B, n * n * sizeof(float));
  cudaMalloc(&d_C, n * n * sizeof(float));
  cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, n * n * sizeof(float), cudaMemcpyHostToDevice);

  // time the matmul call with CUDA events
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  matmul(d_A, d_B, d_C, n, threads_per_block);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);

  cudaMemcpy(C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

  // last element of C, then the time in ms
  std::cout << C[n * n - 1] << "\n";
  std::cout << ms << "\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}
