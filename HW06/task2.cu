#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <random>

#include "stencil.cuh"

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
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " n R threads_per_block\n";
    return 1;
  }

  const long n_arg = parse_positive(argv[1]);
  const long R_arg = parse_positive(argv[2]);
  const long t_arg = parse_positive(argv[3]);
  if (n_arg == 0 || R_arg == 0 || t_arg == 0 || t_arg > 1024 ||
      t_arg < 2 * R_arg + 1) {
    std::cerr << "n, R and threads_per_block must be positive integers, "
                 "threads_per_block <= 1024 and threads_per_block >= 2R + 1.\n";
    return 1;
  }
  const unsigned int n = static_cast<unsigned int>(n_arg);
  const unsigned int R = static_cast<unsigned int>(R_arg);
  const unsigned int threads_per_block = static_cast<unsigned int>(t_arg);
  const unsigned int m = 2 * R + 1;

  // image and mask values in [-1, 1]
  float* image = new float[n];
  float* mask = new float[m];
  float* output = new float[n];
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (unsigned int i = 0; i < n; ++i) {
    image[i] = dist(gen);
  }
  for (unsigned int i = 0; i < m; ++i) {
    mask[i] = dist(gen);
  }

  float* d_image = nullptr;
  float* d_mask = nullptr;
  float* d_output = nullptr;
  cudaMalloc(&d_image, n * sizeof(float));
  cudaMalloc(&d_mask, m * sizeof(float));
  cudaMalloc(&d_output, n * sizeof(float));
  cudaMemcpy(d_image, image, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask, mask, m * sizeof(float), cudaMemcpyHostToDevice);

  // time the stencil call with CUDA events
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  stencil(d_image, d_mask, d_output, n, R, threads_per_block);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);

  cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

  // last element of output, then the time in ms
  std::cout << output[n - 1] << "\n";
  std::cout << ms << "\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_image);
  cudaFree(d_mask);
  cudaFree(d_output);
  delete[] image;
  delete[] mask;
  delete[] output;

  return 0;
}
