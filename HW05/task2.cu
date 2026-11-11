#include <cstdio>
#include <random>

// Each thread writes a * x + y into its own entry of dA, where x is
// threadIdx.x and y is blockIdx.x.
__global__ void axpy_kernel(int* dA, int a) {
  const int x = threadIdx.x;
  const int y = blockIdx.x;
  dA[blockIdx.x * blockDim.x + threadIdx.x] = a * x + y;
}

int main() {
  const int num_blocks = 2;
  const int num_threads = 8;
  const int n = num_blocks * num_threads;

  // random integer a in [1, 10]
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(1, 10);
  const int a = dist(gen);

  int* dA = nullptr;
  cudaMalloc(&dA, n * sizeof(int));

  axpy_kernel<<<num_blocks, num_threads>>>(dA, a);

  // cudaMemcpy waits for the kernel to finish before copying
  int hA[n];
  cudaMemcpy(hA, dA, n * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; ++i) {
    if (i > 0) {
      std::printf(" ");
    }
    std::printf("%d", hA[i]);
  }
  std::printf("\n");

  cudaFree(dA);

  return 0;
}
