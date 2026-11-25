/* Independent single-CTA global-read/shared-atomic dependency probe.
 *
 * No expert weights, routing output, prefix scan, sorting or complete MoE
 * operator. The synthetic expert IDs are a uniform deterministic permutation.
 * Three kernels hold the grid and loop geometry fixed and isolate global
 * input reads, shared atomics, and their load-to-atomic dependency.
 */
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); \
  } \
} while (0)

constexpr int THREADS = 1024;
constexpr int SAMPLES = 7;
constexpr int REPEAT = 2;
constexpr int ITEMS[] = {131072, 262144, 524288};
constexpr int EXPERTS[] = {128, 512};

__global__ void init_ids(int* ids, int items, int experts) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < items;
       i += blockDim.x * gridDim.x) {
    ids[i] = (i * 17) & (experts - 1);
  }
}

__global__ void read_only(const int* __restrict__ ids, int items, int* out) {
  unsigned sum = 0;
  for (int i = threadIdx.x; i < items; i += THREADS) {
    sum += static_cast<unsigned>(ids[i]);
  }
  out[threadIdx.x] = static_cast<int>(sum);
}

__global__ void atomic_only(int items, int experts, int* out) {
  extern __shared__ int count[];
  for (int e = threadIdx.x; e < experts; e += THREADS) count[e] = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < items; i += THREADS) {
    atomicAdd(&count[(i * 17) & (experts - 1)], 1);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int total = 0;
    for (int e = 0; e < experts; ++e) total += count[e];
    out[0] = total;
  }
}

__global__ void load_atomic(const int* __restrict__ ids, int items,
                            int experts, int* out) {
  extern __shared__ int count[];
  for (int e = threadIdx.x; e < experts; e += THREADS) count[e] = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < items; i += THREADS) {
    const int expert = ids[i];
    atomicAdd(&count[expert], 1);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int total = 0;
    for (int e = 0; e < experts; ++e) total += count[e];
    out[0] = total;
  }
}

static void launch(int mode, const int* ids, int items, int experts, int* out) {
  if (mode == 0) read_only<<<1, THREADS>>>(ids, items, out);
  else if (mode == 1) atomic_only<<<1, THREADS, experts * sizeof(int)>>>(items, experts, out);
  else load_atomic<<<1, THREADS, experts * sizeof(int)>>>(ids, items, experts, out);
  CHECK(cudaGetLastError());
}

int main() {
  int* ids = nullptr;
  int* out = nullptr;
  CHECK(cudaMalloc(&ids, ITEMS[2] * sizeof(int)));
  CHECK(cudaMalloc(&out, THREADS * sizeof(int)));
  cudaEvent_t start, end;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&end));
  std::printf("repeat,items,experts,mode,sample,event_ms,checksum\n");
  for (int repeat = 0; repeat < REPEAT; ++repeat) {
    for (int items : ITEMS) {
      for (int experts : EXPERTS) {
        init_ids<<<132, 256>>>(ids, items, experts);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        for (int order = 0; order < 3; ++order) {
          int mode = (order + repeat) % 3;
          launch(mode, ids, items, experts, out);
          CHECK(cudaDeviceSynchronize());
          for (int sample = 0; sample < SAMPLES; ++sample) {
            CHECK(cudaEventRecord(start));
            launch(mode, ids, items, experts, out);
            CHECK(cudaEventRecord(end));
            CHECK(cudaEventSynchronize(end));
            float ms = 0;
            CHECK(cudaEventElapsedTime(&ms, start, end));
            if (!(ms > 0)) return 2;
            int checksum = 0;
            if (mode == 0) {
              std::vector<int> host(THREADS);
              CHECK(cudaMemcpy(host.data(), out, THREADS * sizeof(int), cudaMemcpyDeviceToHost));
              for (int value : host) checksum += value;
              const int expected = items / experts * (experts * (experts - 1) / 2);
              if (checksum != expected) return 3;
            } else {
              CHECK(cudaMemcpy(&checksum, out, sizeof(int), cudaMemcpyDeviceToHost));
              if (checksum != items) return 4;
            }
            std::printf("%d,%d,%d,%d,%d,%.9g,%d\n", repeat, items,
                        experts, mode, sample, ms, checksum);
          }
        }
      }
    }
  }
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(end));
  CHECK(cudaFree(ids));
  CHECK(cudaFree(out));
  return 0;
}
