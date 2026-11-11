#include <cstdio>

// Each thread computes the factorial of (its index + 1) and prints it.
__global__ void factorial_kernel() {
  const int a = threadIdx.x + 1;
  int b = 1;
  for (int i = 2; i <= a; ++i) {
    b *= i;
  }
  std::printf("%d!=%d\n", a, b);
}

int main() {
  // 1 block with 8 threads, one thread for each integer 1..8
  factorial_kernel<<<1, 8>>>();
  // wait until the kernel has finished printing
  cudaDeviceSynchronize();

  return 0;
}
