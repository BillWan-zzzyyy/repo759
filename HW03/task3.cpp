#include <omp.h>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

#include "msort.h"

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
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " n t ts\n";
    return 1;
  }

  // n is the array length, t the number of threads, ts the threshold
  const long n_arg = parse_positive(argv[1]);
  const long t = parse_positive(argv[2]);
  const long ts_arg = parse_positive(argv[3]);
  if (n_arg == 0 || t == 0 || t > 20 || ts_arg == 0) {
    std::cerr << "n and ts must be positive integers and t must be in "
                 "[1, 20].\n";
    return 1;
  }
  const std::size_t n = static_cast<std::size_t>(n_arg);
  const std::size_t ts = static_cast<std::size_t>(ts_arg);

  // random integers in [-1000, 1000]
  int* arr = new int[n];
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-1000, 1000);
  for (std::size_t i = 0; i < n; ++i) {
    arr[i] = dist(gen);
  }

  // msort has no thread argument, so set the number of threads here
  omp_set_num_threads(static_cast<int>(t));

  high_resolution_clock::time_point start = high_resolution_clock::now();
  msort(arr, n, ts);
  high_resolution_clock::time_point stop = high_resolution_clock::now();
  duration<double, std::milli> elapsed = stop - start;

  // first and last element of the sorted array, then the time in ms
  std::cout << arr[0] << "\n";
  std::cout << arr[n - 1] << "\n";
  std::cout << elapsed.count() << "\n";

  delete[] arr;

  return 0;
}
