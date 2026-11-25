// Independent Hopper 3D TMA probe. Each cluster issues eight transfers into
// eight distinct shared boxes per completion wait, reducing the cluster sync
// frequency versus tma3d_cluster_operand.cu. No MMA or GEMM output.
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
constexpr int kBatch = 8;

__device__ __forceinline__ uint32_t smem_address(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void wait_barrier(uint64_t* barrier, int phase) {
  asm volatile("{ .reg .pred done; WAIT_%=: "
               "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1; "
               "@!done bra WAIT_%=; }" ::
               "r"(smem_address(barrier)), "r"(phase) : "memory");
}

__global__ __cluster_dims__(2, 1, 1) void batched_operand_stream(
    const __grid_constant__ CUtensorMap tensor_map, uint32_t* checksum,
    int multicast) {
  extern __shared__ __align__(128) uint8_t box[];
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
  for (int batch = 0; batch < kSteps / kBatch; ++batch) {
    if (threadIdx.x == 0) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::
                   "r"(smem_address(&barrier)), "r"(kBoxBytes * kBatch) : "memory");
    }
    cluster.sync();
    if (threadIdx.x == 0) {
      for (int slot = 0; slot < kBatch; ++slot) {
        const int coordinate = cluster_id * kSteps + batch * kBatch + slot;
        uint8_t* destination = box + slot * kBoxBytes;
        if (multicast) {
          if (rank == 0) {
            const uint16_t mask = 0x3;
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cluster.global."
                "mbarrier::complete_tx::bytes.multicast::cluster "
                "[%0], [%1, {%4, %5, %6}], [%2], %3;" ::
                "r"(smem_address(destination)), "l"(&tensor_map),
                "r"(smem_address(&barrier)), "h"(mask),
                "r"(0), "r"(0), "r"(coordinate) : "memory");
          }
        } else {
          asm volatile(
              "cp.async.bulk.tensor.3d.shared::cluster.global."
              "mbarrier::complete_tx::bytes "
              "[%0], [%1, {%3, %4, %5}], [%2];" ::
              "r"(smem_address(destination)), "l"(&tensor_map),
              "r"(smem_address(&barrier)),
              "r"(0), "r"(0), "r"(coordinate) : "memory");
        }
      }
      wait_barrier(&barrier, batch & 1);
      for (int slot = 0; slot < kBatch; ++slot) {
        sum += box[slot * kBoxBytes];
      }
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
  CHECK(cudaFuncSetAttribute(batched_operand_stream,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kBoxBytes * kBatch));
  cudaEvent_t start, finish;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&finish));
  printf("{\"box_bytes\":%d,\"clusters\":%d,\"steps\":%d,\"runs\":[",
         kBoxBytes, kClusters, kSteps);
  for (int mode = 0; mode != 2; ++mode) {
    batched_operand_stream<<<2 * kClusters, 128, kBoxBytes * kBatch>>>(
        map, checksum, mode == 0);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    printf("%s{\"mode\":\"%s\",\"milliseconds\":[",
           mode ? "," : "", mode ? "unicast" : "multicast");
    for (int sample = 0; sample < 7; ++sample) {
      CHECK(cudaEventRecord(start));
      batched_operand_stream<<<2 * kClusters, 128, kBoxBytes * kBatch>>>(
          map, checksum, mode == 0);
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
