// Independent 96-thread CTA register SiLU service probe.  No routed data,
// model output, weights, or production kernel is used.  Each lane evaluates
// eight independent BF16 inputs and writes one FP32 checksum.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  std::exit(1); } } while (0)

template <bool SourceMath>
__global__ __launch_bounds__(96) void probe(float* output) {
  const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float seed = float((index + j * 73) & 1023) * (1.0f / 512.0f) - 1.0f;
    const float x = __bfloat162float(__float2bfloat16(seed));
    if constexpr (SourceMath) {
      sum += x / (1.0f + expf(-x));
    } else {
      sum = fmaf(x, 0.125f, sum);
    }
  }
  output[index] = sum;
}

int main() {
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  if (std::strstr(prop.name, "H200") == nullptr || prop.multiProcessorCount != 132) {
    std::fprintf(stderr, "requires one 132-SM H200\n");
    return 1;
  }
  constexpr int grids[] = {1320, 13200, 132000, 524288};
  constexpr int trials = 7;
  float* output;
  CHECK(cudaMalloc(&output, size_t(grids[3]) * 96 * sizeof(float)));
  cudaEvent_t begin, end;
  CHECK(cudaEventCreate(&begin));
  CHECK(cudaEventCreate(&end));
  std::puts("variant,grid,block,trial,chains_per_lane,event_ms");
  for (int grid : grids) {
    for (int variant = 0; variant < 2; ++variant) {
      if (variant == 0) probe<false><<<grid, 96>>>(output);
      else probe<true><<<grid, 96>>>(output);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      for (int trial = 0; trial < trials; ++trial) {
        CHECK(cudaEventRecord(begin));
        if (variant == 0) probe<false><<<grid, 96>>>(output);
        else probe<true><<<grid, 96>>>(output);
        CHECK(cudaEventRecord(end));
        CHECK(cudaEventSynchronize(end));
        float elapsed;
        CHECK(cudaEventElapsedTime(&elapsed, begin, end));
        std::printf("%s,%d,96,%d,8,%.9f\n",
                    variant == 0 ? "arithmetic_control" : "source_silu",
                    grid, trial, double(elapsed));
      }
      float sample;
      CHECK(cudaMemcpy(&sample, output + size_t(grid - 1) * 96 + 95,
                       sizeof(sample), cudaMemcpyDeviceToHost));
      if (!__builtin_isfinite(sample)) {
        std::fprintf(stderr, "non-finite checksum\n");
        return 2;
      }
    }
  }
  CHECK(cudaEventDestroy(begin));
  CHECK(cudaEventDestroy(end));
  CHECK(cudaFree(output));
  return 0;
}
