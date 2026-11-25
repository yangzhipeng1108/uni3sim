/* Single-CTA branch/reconvergence cost for synthetic load->shared atomics.
 * This independent mechanism probe does not scan/pad/sort production routes,
 * construct a prefix, write expert IDs, or execute a complete MoE operator.
 */
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
constexpr int ITEMS[] = {131072, 262144, 524288};
constexpr int EXPERTS[] = {128, 512};

__global__ void init_ids(int* ids, int items, int experts) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < items;
       i += gridDim.x * blockDim.x) ids[i] = (i * 17) & (experts - 1);
}

template <bool GUARDED>
__global__ void service(const int* __restrict__ ids, int items, int experts,
                        int* out, unsigned long long* cycles) {
  extern __shared__ int counts[];
  const unsigned long long start = clock64();
  for (int e = threadIdx.x; e < experts; e += THREADS) counts[e] = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < items; i += THREADS) {
    const int expert = ids[i];
    if constexpr (GUARDED) {
      if (expert >= 0 && expert < experts) atomicAdd(&counts[expert], 1);
    } else {
      atomicAdd(&counts[expert], 1);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int total = 0;
    for (int e = 0; e < experts; ++e) total += counts[e];
    out[0] = total;
    cycles[0] = clock64() - start;
  }
}

static void launch(bool guarded, int* ids, int items, int experts,
                   int* out, unsigned long long* cycles) {
  if (guarded) service<true><<<1, THREADS, experts * sizeof(int)>>>(
      ids, items, experts, out, cycles);
  else service<false><<<1, THREADS, experts * sizeof(int)>>>(
      ids, items, experts, out, cycles);
  CHECK(cudaGetLastError());
}

int main() {
  int *ids = nullptr, *out = nullptr;
  unsigned long long* cycles = nullptr;
  CHECK(cudaMalloc(&ids, ITEMS[2] * sizeof(int)));
  CHECK(cudaMalloc(&out, sizeof(int)));
  CHECK(cudaMalloc(&cycles, sizeof(unsigned long long)));
  cudaEvent_t start, end;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&end));
  std::printf("repeat,items,experts,guarded,sample,event_ms,cta_cycles,checksum\n");
  for (int repeat = 0; repeat < 2; ++repeat) {
    for (int items : ITEMS) {
      for (int experts : EXPERTS) {
        init_ids<<<132, 256>>>(ids, items, experts);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        for (int order = 0; order < 2; ++order) {
          const bool guarded = bool((order + repeat) % 2);
          for (int warmup = 0; warmup < 4; ++warmup) {
            launch(guarded, ids, items, experts, out, cycles);
          }
          CHECK(cudaDeviceSynchronize());
          for (int sample = 0; sample < 7; ++sample) {
            CHECK(cudaEventRecord(start));
            launch(guarded, ids, items, experts, out, cycles);
            CHECK(cudaEventRecord(end));
            CHECK(cudaEventSynchronize(end));
            float event_ms = 0;
            CHECK(cudaEventElapsedTime(&event_ms, start, end));
            int checksum = 0;
            unsigned long long cta_cycles = 0;
            CHECK(cudaMemcpy(&checksum, out, sizeof(int), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(&cta_cycles, cycles, sizeof(cta_cycles), cudaMemcpyDeviceToHost));
            if (checksum != items || !(event_ms > 0) || cta_cycles == 0) return 2;
            std::printf("%d,%d,%d,%d,%d,%.9g,%llu,%d\n", repeat, items,
                        experts, int(guarded), sample, event_ms, cta_cycles, checksum);
          }
        }
      }
    }
  }
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(end));
  CHECK(cudaFree(ids));
  CHECK(cudaFree(out));
  CHECK(cudaFree(cycles));
  return 0;
}
