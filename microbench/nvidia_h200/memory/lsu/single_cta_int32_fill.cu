/* One-CTA contiguous INT32 fill service, independent of MoE alignment. */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); \
  } \
} while (0)

constexpr int THREADS = 1024;
constexpr int ITEMS[] = {262144, 524288, 1048576};
constexpr int BACKING_ITEMS = 32 * 1024 * 1024;

__global__ void fill(int* out, int items, int value, unsigned long long* cycles) {
  const unsigned long long start = clock64();
  for (int i = threadIdx.x; i < items; i += THREADS) out[i] = value;
  __syncthreads();
  if (threadIdx.x == 0) cycles[0] = clock64() - start;
}

int main() {
  int* backing = nullptr;
  unsigned long long* cycles = nullptr;
  CHECK(cudaMalloc(&backing, BACKING_ITEMS * sizeof(int)));
  CHECK(cudaMalloc(&cycles, sizeof(unsigned long long)));
  cudaEvent_t start, end;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&end));
  std::printf("repeat,items,address_mode,sample,event_ms,cta_cycles,checksum\n");
  for (int repeat = 0; repeat < 2; ++repeat) {
    for (int items : ITEMS) {
      for (int order = 0; order < 2; ++order) {
        const int mode = (order + repeat) % 2;
        for (int warm = 0; warm < 8; ++warm)
          fill<<<1, THREADS>>>(backing, items, items, cycles);
        CHECK(cudaDeviceSynchronize());
        for (int sample = 0; sample < 7; ++sample) {
          const int offset = mode ? ((repeat * 7 + sample) * 8 * 1024 * 1024)
                                    % (BACKING_ITEMS - items) : 0;
          int* out = backing + offset;
          CHECK(cudaEventRecord(start));
          fill<<<1, THREADS>>>(out, items, items, cycles);
          CHECK(cudaGetLastError());
          CHECK(cudaEventRecord(end));
          CHECK(cudaEventSynchronize(end));
          float event_ms = 0;
          CHECK(cudaEventElapsedTime(&event_ms, start, end));
          unsigned long long cta_cycles = 0;
          int first = 0, last = 0;
          CHECK(cudaMemcpy(&cta_cycles, cycles, sizeof(cta_cycles), cudaMemcpyDeviceToHost));
          CHECK(cudaMemcpy(&first, out, sizeof(int), cudaMemcpyDeviceToHost));
          CHECK(cudaMemcpy(&last, out + items - 1, sizeof(int), cudaMemcpyDeviceToHost));
          if (!(event_ms > 0) || cta_cycles == 0 || first != items || last != items)
            return 2;
          std::printf("%d,%d,%d,%d,%.9g,%llu,%d\n", repeat, items,
                      mode, sample, event_ms, cta_cycles, first + last);
        }
      }
    }
  }
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(end));
  CHECK(cudaFree(backing));
  CHECK(cudaFree(cycles));
  return 0;
}
