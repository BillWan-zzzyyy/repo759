#include "msort.h"

#include <algorithm>

// Serial sort used when the piece is smaller than the threshold.
// Insertion sort has no recursion, which is what the task asks for.
static void insertion_sort(int* arr, std::size_t n) {
  for (std::size_t i = 1; i < n; ++i) {
    const int key = arr[i];
    std::size_t j = i;
    // shift the bigger elements one step to the right
    while (j > 0 && arr[j - 1] > key) {
      arr[j] = arr[j - 1];
      --j;
    }
    arr[j] = key;
  }
}

// Merges the sorted halves arr[0, mid) and arr[mid, n) into tmp and copies
// the result back into arr.
static void merge(int* arr, int* tmp, std::size_t mid, std::size_t n) {
  std::size_t i = 0;
  std::size_t j = mid;
  std::size_t k = 0;
  while (i < mid && j < n) {
    // <= keeps equal elements in their original order
    if (arr[i] <= arr[j]) {
      tmp[k++] = arr[i++];
    } else {
      tmp[k++] = arr[j++];
    }
  }
  while (i < mid) {
    tmp[k++] = arr[i++];
  }
  while (j < n) {
    tmp[k++] = arr[j++];
  }
  std::copy(tmp, tmp + n, arr);
}

// Recursive part of the merge sort. tmp is the matching part of the
// temporary buffer, so the two halves never touch the same memory.
static void msort_rec(int* arr, int* tmp, std::size_t n,
                      std::size_t threshold) {
  if (n < 2) {
    return;
  }
  // small piece: not worth creating more tasks
  if (n < threshold) {
    insertion_sort(arr, n);
    return;
  }

  const std::size_t mid = n / 2;
  // sort the two halves as two OpenMP tasks
#pragma omp task
  msort_rec(arr, tmp, mid, threshold);
#pragma omp task
  msort_rec(arr + mid, tmp + mid, n - mid, threshold);
  // both halves have to be sorted before merging
#pragma omp taskwait
  merge(arr, tmp, mid, n);
}

void msort(int* arr, const std::size_t n, const std::size_t threshold) {
  // one buffer for all merges, allocated once
  int* tmp = new int[n];

  // create the thread team, then let one thread start the recursion; the
  // tasks it creates are picked up by the other threads
#pragma omp parallel
  {
#pragma omp single
    msort_rec(arr, tmp, n, threshold);
  }

  delete[] tmp;
}
