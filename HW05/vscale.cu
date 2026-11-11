#include "vscale.cuh"

__global__ void vscale(const float* a, float* b, unsigned int n) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  // the last block may have more threads than elements left
  if (i < n) {
    b[i] = a[i] * b[i];
  }
}
