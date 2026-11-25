// Independent Hopper mechanism probe: one 192 x 64 BF16-sized operand box
// per cluster and K step. There is no matrix instruction, output tile, or
// production scheduler. The two modes use the same barriers and cluster sync:
// multicast issues one 3D TMA for two CTAs; unicast issues one per CTA.
// Build: nvcc -O3 -std=c++17 -arch=sm_90a tma3d_cluster_operand.cu -o probe
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;

#define CHECK(call) do { cudaError_t error = (call); if (error != cudaSuccess) { \
  fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
  exit(1); } } while (0)

constexpr int kBoxBytes = 192 * 64 * 2;
constexpr int kClusters = 66;
constexpr int kSteps = 80;

__device__ __forceinline__ uint32_t smem_address(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void wait_barrier(uint64_t* barrier, int phase) {
  asm volatile("{ .reg .pred done; WAIT_%=: "
               "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1; "
               "@!done bra WAIT_%=; }" ::
               "r"(smem_address(barrier)), "r"(phase) : "memory");
}

__global__ __cluster_dims__(2, 1, 1) void operand_stream(
    const __grid_constant__ CUtensorMap tensor_map, uint32_t* checksum,
    int multicast) {
  __shared__ __align__(128) uint8_t box[kBoxBytes];
  __shared__ __align__(8) uint64_t barrier;
  auto cluster = cg::this_cluster();
  const int cluster_id = static_cast<int>(blockIdx.x) / 2;
  const int rank = cluster.block_rank();
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::
                 "r"(smem_address(&barrier)) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  cluster.sync();

  uint32_t sum = 0;
  for (int step = 0; step < kSteps; ++step) {
    if (threadIdx.x == 0) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::
                   "r"(smem_address(&barrier)), "r"(kBoxBytes) : "memory");
    }
    // Ensure both CTAs have armed their local completion barrier.
    cluster.sync();
    if (threadIdx.x == 0) {
      const int coordinate = cluster_id * kSteps + step;
      if (multicast) {
        if (rank == 0) {
          const uint16_t mask = 0x3;
          asm volatile(
              "cp.async.bulk.tensor.3d.shared::cluster.global."
              "mbarrier::complete_tx::bytes.multicast::cluster "
              "[%0], [%1, {%4, %5, %6}], [%2], %3;" ::
              "r"(smem_address(box)), "l"(&tensor_map),
              "r"(smem_address(&barrier)), "h"(mask),
              "r"(0), "r"(0), "r"(coordinate) : "memory");
        }
      } else {
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cluster.global."
            "mbarrier::complete_tx::bytes "
            "[%0], [%1, {%3, %4, %5}], [%2];" ::
            "r"(smem_address(box)), "l"(&tensor_map),
            "r"(smem_address(&barrier)),
            "r"(0), "r"(0), "r"(coordinate) : "memory");
      }
      wait_barrier(&barrier, step & 1);
      sum += box[(step * 128) % kBoxBytes];
    }
    cluster.sync();
  }
  if (threadIdx.x == 0) checksum[blockIdx.x] = sum;
}

static CUtensorMap make_map(void* source) {
  void* address = nullptr;
  cudaDriverEntryPointQueryResult status;
  CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &address,
                                cudaEnableDefault, &status));
  if (address == nullptr || status != cudaDriverEntryPointSuccess) {
    fprintf(stderr, "cuTensorMapEncodeTiled unavailable\n");
    exit(1);
  }
  auto encode = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(address);
  CUtensorMap map;
  cuuint64_t dimensions[3] = {128, 192, kClusters * kSteps};
  cuuint64_t strides[2] = {128, kBoxBytes};
  cuuint32_t box[3] = {128, 192, 1};
  cuuint32_t element_strides[3] = {1, 1, 1};
  CUresult result = encode(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, source,
                           dimensions, strides, box, element_strides,
                           CU_TENSOR_MAP_INTERLEAVE_NONE,
                           CU_TENSOR_MAP_SWIZZLE_128B,
                           CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (result != CUDA_SUCCESS) {
    fprintf(stderr, "cuTensorMapEncodeTiled failed: %d\n", int(result));
    exit(1);
  }
  return map;
}

int main() {
  uint8_t* source = nullptr;
  uint32_t* checksum = nullptr;
  CHECK(cudaMalloc(&source, size_t(kBoxBytes) * kClusters * kSteps));
  CHECK(cudaMalloc(&checksum, 2 * kClusters * sizeof(uint32_t)));
  CHECK(cudaMemset(source, 1, size_t(kBoxBytes) * kClusters * kSteps));
  CUtensorMap map = make_map(source);
  cudaEvent_t start, finish;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&finish));
  printf("{\"box_bytes\":%d,\"clusters\":%d,\"steps\":%d,\"runs\":[",
         kBoxBytes, kClusters, kSteps);
  for (int mode = 0; mode != 2; ++mode) {
    operand_stream<<<2 * kClusters, 128>>>(map, checksum, mode == 0);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    printf("%s{\"mode\":\"%s\",\"milliseconds\":[",
           mode ? "," : "", mode ? "unicast" : "multicast");
    for (int sample = 0; sample < 7; ++sample) {
      CHECK(cudaEventRecord(start));
      operand_stream<<<2 * kClusters, 128>>>(map, checksum, mode == 0);
      CHECK(cudaEventRecord(finish));
      CHECK(cudaEventSynchronize(finish));
      CHECK(cudaGetLastError());
      float milliseconds = 0;
      CHECK(cudaEventElapsedTime(&milliseconds, start, finish));
      printf("%s%.6f", sample ? "," : "", milliseconds);
    }
    std::vector<uint32_t> values(2 * kClusters);
    CHECK(cudaMemcpy(values.data(), checksum, values.size() * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost));
    for (uint32_t value : values) {
      if (value != kSteps) {
        fprintf(stderr, "unexpected checksum: %u instead of %d\n", value,
                kSteps);
        return 2;
      }
    }
    printf("]}");
  }
  printf("]}\n");
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(finish));
  CHECK(cudaFree(checksum));
  CHECK(cudaFree(source));
  return 0;
}
