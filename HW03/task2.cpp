#include <omp.h>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

#include "convolution.h"

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

  // n is the image size, t is the number of threads
  const long n_arg = parse_positive(argv[1]);
  const long t = parse_positive(argv[2]);
  if (n_arg == 0 || t == 0 || t > 20) {
    std::cerr << "n must be a positive integer and t must be in [1, 20].\n";
    return 1;
  }
  const std::size_t n = static_cast<std::size_t>(n_arg);
  // the mask is always 3 x 3 in this task
  const std::size_t m = 3;

  // both matrices are stored in 1D in row-major order
  float* image = new float[n * n];
  float* mask = new float[m * m];
  float* output = new float[n * n];

  // image values in [-10, 10], mask values in [-1, 1]
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> image_dist(-10.0f, 10.0f);
  std::uniform_real_distribution<float> mask_dist(-1.0f, 1.0f);

  for (std::size_t i = 0; i < n * n; ++i) {
    image[i] = image_dist(gen);
  }
  for (std::size_t i = 0; i < m * m; ++i) {
    mask[i] = mask_dist(gen);
  }

  // convolve has no thread argument, so set the number of threads here
  omp_set_num_threads(static_cast<int>(t));

  high_resolution_clock::time_point start = high_resolution_clock::now();
  convolve(image, output, n, mask, m);
  high_resolution_clock::time_point stop = high_resolution_clock::now();
  duration<double, std::milli> elapsed = stop - start;

  // first and last element of the result, then the time in ms
  std::cout << output[0] << "\n";
  std::cout << output[n * n - 1] << "\n";
  std::cout << elapsed.count() << "\n";

  delete[] image;
  delete[] mask;
  delete[] output;

  return 0;
}
