#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

#include "scan.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " n\n";
    return 1;
  }

  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(argv[1], &end, 10);
  if (errno != 0 || end == argv[1] || *end != '\0' || parsed <= 0) {
    std::cerr << "n must be a positive integer.\n";
    return 1;
  }
  const std::size_t n = static_cast<std::size_t>(parsed);

  float* arr = new float[n];
  float* output = new float[n];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t i = 0; i < n; ++i) {
    arr[i] = dist(gen);
  }

  high_resolution_clock::time_point start = high_resolution_clock::now();
  scan(arr, output, n);
  high_resolution_clock::time_point stop = high_resolution_clock::now();
  duration<double, std::milli> elapsed = stop - start;

  std::cout << elapsed.count() << "\n";
  std::cout << output[0] << "\n";
  std::cout << output[n - 1] << "\n";

  delete[] arr;
  delete[] output;

  return 0;
}
