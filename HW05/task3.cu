#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <random>

#include "vscale.cuh"

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
  if (argc != 2 && argc != 3) {
    std::cerr << "Usage: " << argv[0] << " n [threads_per_block]\n";
    return 1;
  }

  const long n_arg = parse_positive(argv[1]);
  if (n_arg == 0) {
    std::cerr << "n must be a positive integer.\n";
    return 1;
  }
  const unsigned int n = static_cast<unsigned int>(n_arg);

  // 512 threads per block by default; the optional second argument is
  // only used to get the 16 threads per block results in 3(c)
  unsigned int threads_per_block = 512;
  if (argc == 3) {
    const long t_arg = parse_positive(argv[2]);
    if (t_arg == 0 || t_arg > 1024) {
      std::cerr << "threads_per_block must be in [1, 1024].\n";
      return 1;
    }
    threads_per_block = static_cast<unsigned int>(t_arg);
  }

  // a in [-10, 10], b in [0, 1]
  float* a = new float[n];
  float* b = new float[n];
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> a_dist(-10.0f, 10.0f);
  std::uniform_real_distribution<float> b_dist(0.0f, 1.0f);
  for (unsigned int i = 0; i < n; ++i) {
    a[i] = a_dist(gen);
    b[i] = b_dist(gen);
  }

  float* d_a = nullptr;
  float* d_b = nullptr;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

  // enough blocks so that every element gets one thread
  const unsigned int num_blocks = (n + threads_per_block - 1) / threads_per_block;

  // time only the kernel with CUDA events
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  vscale<<<num_blocks, threads_per_block>>>(d_a, d_b, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);

  cudaMemcpy(b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost);

  // time in ms, then the first and last element of the result
  std::cout << ms << "\n";
  std::cout << b[0] << "\n";
  std::cout << b[n - 1] << "\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_a);
  cudaFree(d_b);
  delete[] a;
  delete[] b;

  return 0;
}
