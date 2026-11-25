// Independent H200 48-KiB cp.async panel ring. The 16-KiB A and 32-KiB B
// panel sizes and four 48-KiB shared slots follow the observed BM128/BN256/
// BK64 launch. This probe has no MMA, expert routing, or matrix output.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call) do { const cudaError_t error = (call); if (error != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(error)); std::exit(1); } } while (0)

constexpr int kThreads = 256;
constexpr int kSmCount = 132;
constexpr int kWaves = 16;
constexpr int kGrid = kSmCount * kWaves;
constexpr int kPanelBytes = (128 + 256) * 64 * 2;
constexpr int kStages = 4;
constexpr int kSharedBytes = kStages * kPanelBytes;
constexpr int kRounds = 32 * 8;
constexpr int kBackingBytes = 48 * 1024 * 1024;
constexpr int kBackingPanels = kBackingBytes / kPanelBytes;
static_assert(kPanelBytes == 49152 && kSharedBytes == 196608);
static_assert(kBackingBytes % kPanelBytes == 0);

__device__ __forceinline__ void issue_panel(
    std::uint8_t* destination, const std::uint8_t* source) {
  for (int vector = threadIdx.x; vector < kPanelBytes / 16;
       vector += blockDim.x) {
    const std::uint32_t shared_address = static_cast<std::uint32_t>(
        __cvta_generic_to_shared(destination + vector * 16));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                 "r"(shared_address), "l"(source + vector * 16) : "memory");
  }
  asm volatile("cp.async.commit_group;\n" :: : "memory");
}

__global__ __launch_bounds__(kThreads, 1) void cp_async_ring(
    const std::uint8_t* __restrict__ input,
    std::uint32_t* __restrict__ checksums) {
  extern __shared__ __align__(16) std::uint8_t shared[];
  for (int round = 0; round < kStages; ++round) {
    const int panel = (blockIdx.x * 17 + round) % kBackingPanels;
    issue_panel(shared + round * kPanelBytes, input + panel * kPanelBytes);
  }
  std::uint32_t checksum = 0;
  for (int round = 0; round < kRounds; ++round) {
    if (round < kRounds - kStages) {
      asm volatile("cp.async.wait_group 3;\n" :: : "memory");
    } else {
      asm volatile("cp.async.wait_group 0;\n" :: : "memory");
    }
    __syncthreads();
    const int slot = round % kStages;
    checksum += reinterpret_cast<volatile std::uint32_t*>(
        shared + slot * kPanelBytes)[threadIdx.x];
    __syncthreads();
    const int next = round + kStages;
    if (next < kRounds) {
      const int panel = (blockIdx.x * 17 + next) % kBackingPanels;
      issue_panel(shared + slot * kPanelBytes, input + panel * kPanelBytes);
    }
  }
  if (threadIdx.x == 0) checksums[blockIdx.x] = checksum;
}

int main() {
  cudaDeviceProp properties{};
  CHECK(cudaGetDeviceProperties(&properties, 0));
  int devices = 0;
  CHECK(cudaGetDeviceCount(&devices));
  if (devices != 1 || std::strcmp(properties.name, "NVIDIA H200") != 0 ||
      properties.major != 9 || properties.multiProcessorCount != kSmCount) {
    std::fprintf(stderr, "requires one 132-SM NVIDIA H200\n");
    return 2;
  }
  CHECK(cudaFuncSetAttribute(cp_async_ring,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes));
  int occupancy = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, cp_async_ring, kThreads, kSharedBytes));
  if (occupancy != 1) {
    std::fprintf(stderr, "expected one resident CTA per SM, got %d\n", occupancy);
    return 3;
  }
  std::uint8_t* input = nullptr;
  std::uint32_t* checksums = nullptr;
  CHECK(cudaMalloc(&input, kBackingBytes));
  CHECK(cudaMalloc(&checksums, kGrid * sizeof(std::uint32_t)));
  CHECK(cudaMemset(input, 0x5a, kBackingBytes));
  CHECK(cudaMemset(checksums, 0, kGrid * sizeof(std::uint32_t)));
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  std::vector<float> samples;
  for (int sample = -2; sample < 7; ++sample) {
    CHECK(cudaEventRecord(start));
    cp_async_ring<<<kGrid, kThreads, kSharedBytes>>>(input, checksums);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaGetLastError());
    if (sample >= 0) {
      float milliseconds = 0;
      CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
      samples.push_back(milliseconds);
    }
  }
  std::uint32_t first = 0;
  CHECK(cudaMemcpy(&first, checksums, sizeof(first), cudaMemcpyDeviceToHost));
  if (first != static_cast<std::uint32_t>(
          static_cast<std::uint64_t>(0x5a5a5a5a) * kRounds)) {
    std::fprintf(stderr, "checksum disagrees with completed panel copies\n");
    return 4;
  }
  std::sort(samples.begin(), samples.end());
  const std::uint64_t logical_bytes =
      static_cast<std::uint64_t>(kGrid) * kRounds * kPanelBytes;
  std::printf("{\"device\":\"%s\",\"sm_count\":%d,\"threads\":%d,"
              "\"shared_bytes\":%d,\"resident_ctas_per_sm\":%d,"
              "\"grid\":%d,\"rounds\":%d,\"backing_bytes\":%d,"
              "\"logical_bytes\":%llu,\"samples_ms\":[",
              properties.name, kSmCount, kThreads, kSharedBytes, occupancy,
              kGrid, kRounds, kBackingBytes,
              static_cast<unsigned long long>(logical_bytes));
  for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
    std::printf("%s%.9f", i ? "," : "", samples[i]);
  }
  std::printf("],\"checksum\":%u}\n", first);
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  CHECK(cudaFree(checksums));
  CHECK(cudaFree(input));
  return 0;
}
