// Independent Hopper persistent-task queue service probe.
// 132 CTAs in 66 two-CTA clusters; only one thread per cluster competes for
// one global counter. No tensor map, matrix multiply, or production payload.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK(call) do { cudaError_t error = (call); if (error != cudaSuccess) { \
  fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
  std::exit(1); } } while (0)

constexpr int kGrid = 132;
constexpr int kCluster = 2;
constexpr int kBlock = 384;
constexpr int kSharedBytes = 213204;
constexpr int kLeaders = kGrid / kCluster;

__global__ void __launch_bounds__(kBlock, 1)
atomic_queue_probe(unsigned int* counter, unsigned int* sink,
                   int claims_per_leader) {
  if (threadIdx.x != 0 || blockIdx.x % kCluster != 0) return;
  unsigned int checksum = static_cast<unsigned int>(blockIdx.x);
  const unsigned int limit =
      kLeaders * static_cast<unsigned int>(claims_per_leader);
#pragma unroll 1
  for (int claim = 0; claim < claims_per_leader; ++claim) {
    const unsigned int old = atomicAdd(counter, 1u);
    if (old >= limit) break;
    checksum ^= old + static_cast<unsigned int>(claim);
  }
  sink[blockIdx.x / kCluster] = checksum;
}

static void launch(unsigned int* counter, unsigned int* sink, int claims) {
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(kGrid);
  config.blockDim = dim3(kBlock);
  config.dynamicSmemBytes = kSharedBytes;
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = kCluster;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;
  config.attrs = attribute;
  config.numAttrs = 1;
  CHECK(cudaLaunchKernelEx(&config, atomic_queue_probe, counter, sink, claims));
}

int main() {
  int device = -1;
  CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties = {};
  CHECK(cudaGetDeviceProperties(&properties, device));
  if (properties.major != 9 || properties.minor != 0 ||
      properties.multiProcessorCount != 132) {
    fprintf(stderr, "expected an H200 with 132 SMs\n");
    return 2;
  }
  CHECK(cudaFuncSetAttribute(
      atomic_queue_probe, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSharedBytes));
  unsigned int* counter = nullptr;
  unsigned int* sink = nullptr;
  CHECK(cudaMalloc(&counter, sizeof(unsigned int)));
  CHECK(cudaMalloc(&sink, kLeaders * sizeof(unsigned int)));
  cudaEvent_t start, end;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&end));

  const int claims[] = {0, 1, 8, 64, 515};
  for (int count : claims) {
    for (int repetition = -2; repetition < 7; ++repetition) {
      CHECK(cudaMemset(counter, 0, sizeof(unsigned int)));
      CHECK(cudaEventRecord(start));
      launch(counter, sink, count);
      CHECK(cudaEventRecord(end));
      CHECK(cudaEventSynchronize(end));
      unsigned int actual = 0;
      CHECK(cudaMemcpy(&actual, counter, sizeof(actual),
                       cudaMemcpyDeviceToHost));
      const unsigned int expected = kLeaders * static_cast<unsigned int>(count);
      if (actual != expected) {
        fprintf(stderr, "counter mismatch %u != %u\n", actual, expected);
        return 3;
      }
      if (repetition >= 0) {
        float milliseconds = 0.0f;
        CHECK(cudaEventElapsedTime(&milliseconds, start, end));
        printf("claims_per_leader=%d,leaders=%d,atomic_ops=%u,repeat=%d,"
               "milliseconds=%.9f\n", count, kLeaders, actual,
               repetition, milliseconds);
      }
    }
  }
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(end));
  CHECK(cudaFree(counter));
  CHECK(cudaFree(sink));
  return 0;
}
