#include "matmul.cuh"

__global__ void matmul_kernel(const float* A, const float* B, float* C,
                              size_t n) {
  // 1D index over all n * n elements of C, one element per thread
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n * n) {
    return;
  }
  const size_t row = idx / n;
  const size_t col = idx % n;

  float sum = 0.0f;
  for (size_t k = 0; k < n; ++k) {
    sum += A[row * n + k] * B[k * n + col];
  }
  C[idx] = sum;
}

void matmul(const float* A, const float* B, float* C, size_t n,
            unsigned int threads_per_block) {
  // enough blocks so that every element of C gets one thread
  const unsigned int num_blocks =
      static_cast<unsigned int>((n * n + threads_per_block - 1) / threads_per_block);
  matmul_kernel<<<num_blocks, threads_per_block>>>(A, B, C, n);
  cudaDeviceSynchronize();
}
