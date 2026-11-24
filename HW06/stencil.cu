#include "stencil.cuh"

__global__ void stencil_kernel(const float* image, const float* mask,
                               float* output, unsigned int n, unsigned int R) {
  // dynamic shared memory holds, in this order:
  // the mask (2R + 1), the image piece of this block (blockDim.x + 2R),
  // and the output of this block (blockDim.x)
  extern __shared__ float shared[];
  float* s_mask = shared;
  float* s_image = s_mask + 2 * R + 1;
  float* s_output = s_image + blockDim.x + 2 * R;

  const unsigned int tid = threadIdx.x;
  const unsigned int i = blockIdx.x * blockDim.x + tid;

  // blockDim.x >= 2R + 1, so one pass is enough to load the mask
  if (tid < 2 * R + 1) {
    s_mask[tid] = mask[tid];
  }

  // s_image[k] holds image[block_start - R + k]; image is 1 outside [0, n)
  const long long block_start = static_cast<long long>(blockIdx.x) * blockDim.x;
  for (unsigned int k = tid; k < blockDim.x + 2 * R; k += blockDim.x) {
    const long long j = block_start - R + k;
    s_image[k] = (j >= 0 && j < n) ? image[j] : 1.0f;
  }
  // everything has to be loaded before any thread uses it
  __syncthreads();

  if (i < n) {
    float sum = 0.0f;
    for (unsigned int j = 0; j <= 2 * R; ++j) {
      sum += s_image[tid + j] * s_mask[j];
    }
    s_output[tid] = sum;
    output[i] = s_output[tid];
  }
}

__host__ void stencil(const float* image, const float* mask, float* output,
                      unsigned int n, unsigned int R,
                      unsigned int threads_per_block) {
  const unsigned int num_blocks = (n + threads_per_block - 1) / threads_per_block;
  // mask + image piece + output, see stencil_kernel
  const size_t shared_bytes =
      (2 * R + 1 + threads_per_block + 2 * R + threads_per_block) * sizeof(float);
  stencil_kernel<<<num_blocks, threads_per_block, shared_bytes>>>(image, mask,
                                                                  output, n, R);
  cudaDeviceSynchronize();
}
