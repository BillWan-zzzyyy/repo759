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
    std::cerr << "Usage: " << argv[0] << " n m\n";
    return 1;
  }

  const long n_arg = parse_positive(argv[1]);
  const long m_arg = parse_positive(argv[2]);
  if (n_arg == 0 || m_arg == 0 || m_arg % 2 == 0) {
    std::cerr << "n must be a positive integer and m a positive odd integer.\n";
    return 1;
  }
  const std::size_t n = static_cast<std::size_t>(n_arg);
  const std::size_t m = static_cast<std::size_t>(m_arg);

  float* image = new float[n * n];
  float* mask = new float[m * m];
  float* output = new float[n * n];

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

  high_resolution_clock::time_point start = high_resolution_clock::now();
  convolve(image, output, n, mask, m);
  high_resolution_clock::time_point stop = high_resolution_clock::now();
  duration<double, std::milli> elapsed = stop - start;

  std::cout << elapsed.count() << "\n";
  std::cout << output[0] << "\n";
  std::cout << output[n * n - 1] << "\n";

  delete[] image;
  delete[] mask;
  delete[] output;

  return 0;
}
