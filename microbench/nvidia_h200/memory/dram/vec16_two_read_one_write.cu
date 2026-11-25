// Independent H200 mixed-DRAM mechanism: two 16-byte reads and one
// 16-byte write per lane. No activation, routing, weights, or model output.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  std::exit(1); } } while (0)

__global__ void two_read_one_write(const int4* a, const int4* b, int4* out) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int4 x = __ldg(a + i);
  int4 y = __ldg(b + i);
  out[i] = make_int4(x.x ^ y.x, x.y ^ y.y, x.z ^ y.z, x.w ^ y.w);
}

int main() {
  cudaDeviceProp props;
  CHECK(cudaGetDeviceProperties(&props, 0));
  if (std::strstr(props.name, "H200") == nullptr || props.multiProcessorCount != 132) {
    std::fprintf(stderr, "requires one 132-SM H200\n");
    return 1;
  }
  constexpr int grids[] = {131072, 262144, 524288};
  constexpr int blocks[] = {96, 128};
  constexpr int trials = 7;
  const size_t max_bytes = size_t(grids[2]) * blocks[1] * sizeof(int4);
  int4 *a, *b, *out;
  CHECK(cudaMalloc(&a, max_bytes));
  CHECK(cudaMalloc(&b, max_bytes));
  CHECK(cudaMalloc(&out, max_bytes));
  CHECK(cudaMemset(a, 0x5a, max_bytes));
  CHECK(cudaMemset(b, 0xa5, max_bytes));
  CHECK(cudaMemset(out, 0, max_bytes));
  cudaEvent_t start, end;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&end));
  std::puts("grid,block,trial,read_bytes,write_bytes,event_ms");
  for (int grid : grids) {
    for (int block : blocks) {
      const size_t elements = size_t(grid) * block;
      const size_t role_bytes = elements * sizeof(int4);
      if (role_bytes <= 62914560) return 2;
      two_read_one_write<<<grid, block>>>(a, b, out);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      for (int trial = 0; trial < trials; ++trial) {
        CHECK(cudaEventRecord(start));
        two_read_one_write<<<grid, block>>>(a, b, out);
        CHECK(cudaEventRecord(end));
        CHECK(cudaEventSynchronize(end));
        CHECK(cudaGetLastError());
        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, end));
        std::printf("%d,%d,%d,%zu,%zu,%.9f\n", grid, block, trial,
                    2 * role_bytes, role_bytes, double(ms));
      }
      int4 sample;
      CHECK(cudaMemcpy(&sample, out + elements - 1, sizeof(sample), cudaMemcpyDeviceToHost));
      if (sample.x != -1 || sample.y != -1 || sample.z != -1 || sample.w != -1) {
        std::fprintf(stderr, "mixed-stream output validation failed\n");
        return 3;
      }
    }
  }
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(end));
  CHECK(cudaFree(a));
  CHECK(cudaFree(b));
  CHECK(cudaFree(out));
  return 0;
}
