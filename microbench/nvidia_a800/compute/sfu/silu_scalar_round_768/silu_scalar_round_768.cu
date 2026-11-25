/* Independent SM80 scalar-SiLU recurrence and CTA-overlap mechanism probe. */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK_CUDA(call) do { \
  const cudaError_t error = (call); \
  if (error != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

namespace {

constexpr int kThreads = 768;
constexpr int kSmCount = 108;
constexpr int kResidentCtas = 2;
constexpr int kRounds = 64;
constexpr int kTrials = 3;
constexpr int kRotations = 3;
constexpr int kStaticSharedBytes = 1024;
constexpr int kSharedWords = kStaticSharedBytes / sizeof(unsigned int);
constexpr const char* kExpectedDevice = "NVIDIA A800-SXM4-80GB";
constexpr const char* kHeader =
    "run_rotation,trial,order_position,grid_blocks,block_threads,rounds,"
    "resident_cta_limit,active_sms,minimum_ctas_per_sm,maximum_ctas_per_sm,"
    "maximum_sm_span_cycles,median_cta_cycles,checksum";

__device__ __forceinline__ unsigned int read_smid() {
  unsigned int value;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
  return value;
}

__device__ __forceinline__ std::uint64_t read_clock64() {
  std::uint64_t value;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value));
  return value;
}

__device__ __forceinline__ __nv_bfloat16 independent_bf16(int thread) {
  const std::uint16_t bits = static_cast<std::uint16_t>(
      0x3e00U + (static_cast<unsigned int>(thread) & 0x001fU));
  return __ushort_as_bfloat16(bits);
}

__device__ __forceinline__ float scalar_silu(float value) {
  return value / (1.0F + expf(-value));
}

__global__ __launch_bounds__(kThreads, kResidentCtas)
void silu_scalar_round_768(
    float* __restrict__ final_sink, std::uint64_t* __restrict__ starts,
    std::uint64_t* __restrict__ ends, unsigned int* __restrict__ smids) {
  __shared__ unsigned int residency_class_smem[kSharedWords];
  if (threadIdx.x < kSharedWords) {
    residency_class_smem[threadIdx.x] = static_cast<unsigned int>(threadIdx.x);
  }
  __syncthreads();

  const float shared_lifetime = static_cast<float>(
      residency_class_smem[threadIdx.x & (kSharedWords - 1)]) * 1e-12F;
  float lifetime0 = static_cast<float>((threadIdx.x + 1) & 31) * 1e-12F;
  float lifetime1 = static_cast<float>((threadIdx.x + 3) & 31) * 1e-12F;
  float lifetime2 = static_cast<float>((threadIdx.x + 5) & 31) * 1e-12F;
  float lifetime3 = static_cast<float>((threadIdx.x + 7) & 31) * 1e-12F;
  float lifetime4 = static_cast<float>((threadIdx.x + 11) & 31) * 1e-12F;
  float lifetime5 = static_cast<float>((threadIdx.x + 13) & 31) * 1e-12F;
  float lifetime6 = static_cast<float>((threadIdx.x + 17) & 31) * 1e-12F;
  asm volatile("" : "+f"(lifetime0), "+f"(lifetime1), "+f"(lifetime2),
               "+f"(lifetime3), "+f"(lifetime4), "+f"(lifetime5),
               "+f"(lifetime6));
  float state = __bfloat162float(independent_bf16(threadIdx.x));

  __syncthreads();
  if (threadIdx.x == 0) {
    smids[blockIdx.x] = read_smid();
    starts[blockIdx.x] = read_clock64();
  }
  __syncthreads();

#pragma unroll 1
  for (int round = 0; round < kRounds; ++round) {
    state = scalar_silu(state);
  }

  __syncthreads();
  if (threadIdx.x == 0) ends[blockIdx.x] = read_clock64();
  final_sink[blockIdx.x * blockDim.x + threadIdx.x] = state + shared_lifetime
      + lifetime0 + lifetime1 + lifetime2 + lifetime3 + lifetime4 + lifetime5
      + lifetime6;
}

void run_cell(int grid, int rotation, int trial, int position, float* final_sink,
              std::uint64_t* starts, std::uint64_t* ends, unsigned int* smids) {
  int resident_ctas = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &resident_ctas, silu_scalar_round_768, kThreads, kStaticSharedBytes));
  if (resident_ctas != kResidentCtas) {
    std::fprintf(stderr, "unexpected occupancy %d\n", resident_ctas);
    std::exit(EXIT_FAILURE);
  }
  silu_scalar_round_768<<<grid, kThreads>>>(final_sink, starts, ends, smids);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<std::uint64_t> host_starts(grid);
  std::vector<std::uint64_t> host_ends(grid);
  std::vector<unsigned int> host_smids(grid);
  CHECK_CUDA(cudaMemcpy(host_starts.data(), starts, grid * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(host_ends.data(), ends, grid * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(host_smids.data(), smids, grid * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  std::vector<int> ctas_per_sm(kSmCount, 0);
  std::vector<std::uint64_t> earliest(kSmCount, UINT64_MAX);
  std::vector<std::uint64_t> latest(kSmCount, 0);
  std::vector<std::uint64_t> cta_cycles;
  for (int block = 0; block < grid; ++block) {
    const unsigned int sm = host_smids[block];
    if (sm >= kSmCount || host_ends[block] <= host_starts[block]) {
      std::fprintf(stderr, "invalid timing record\n");
      std::exit(EXIT_FAILURE);
    }
    ++ctas_per_sm[sm];
    earliest[sm] = std::min(earliest[sm], host_starts[block]);
    latest[sm] = std::max(latest[sm], host_ends[block]);
    cta_cycles.push_back(host_ends[block] - host_starts[block]);
  }
  const int active_sms = static_cast<int>(std::count_if(
      ctas_per_sm.begin(), ctas_per_sm.end(), [](int value) { return value > 0; }));
  const int min_ctas = *std::min_element(ctas_per_sm.begin(), ctas_per_sm.end());
  const int max_ctas = *std::max_element(ctas_per_sm.begin(), ctas_per_sm.end());
  std::uint64_t maximum_sm_span = 0;
  for (int sm = 0; sm < kSmCount; ++sm) {
    if (ctas_per_sm[sm] > 0) maximum_sm_span = std::max(maximum_sm_span, latest[sm] - earliest[sm]);
  }
  std::sort(cta_cycles.begin(), cta_cycles.end());
  float checksum = 0.0F;
  CHECK_CUDA(cudaMemcpy(&checksum, final_sink, sizeof(checksum), cudaMemcpyDeviceToHost));
  std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%.9g\n",
      rotation, trial, position, grid, kThreads, kRounds, resident_ctas, active_sms,
      min_ctas, max_ctas, static_cast<unsigned long long>(maximum_sm_span),
      static_cast<unsigned long long>(cta_cycles[cta_cycles.size() / 2]), checksum);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s DEVICE TRIALS ROTATION\n", argv[0]);
    return EXIT_FAILURE;
  }
  const int device = std::stoi(argv[1]);
  const int trials = std::stoi(argv[2]);
  const int rotation = std::stoi(argv[3]);
  if (device < 0 || trials != kTrials || rotation < 0 || rotation >= kRotations) {
    std::fprintf(stderr, "device>=0, trials=3, rotation in [0,2] required\n");
    return EXIT_FAILURE;
  }
  CHECK_CUDA(cudaSetDevice(device));
  cudaDeviceProp properties{};
  CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
  if (std::string(properties.name) != kExpectedDevice || properties.major != 8
      || properties.minor != 0 || properties.multiProcessorCount != kSmCount) {
    std::fprintf(stderr, "unexpected device geometry\n");
    return EXIT_FAILURE;
  }
  constexpr int kMaximumGrid = 216;
  float* final_sink = nullptr;
  std::uint64_t* starts = nullptr;
  std::uint64_t* ends = nullptr;
  unsigned int* smids = nullptr;
  CHECK_CUDA(cudaMalloc(&final_sink, kMaximumGrid * kThreads * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&starts, kMaximumGrid * sizeof(std::uint64_t)));
  CHECK_CUDA(cudaMalloc(&ends, kMaximumGrid * sizeof(std::uint64_t)));
  CHECK_CUDA(cudaMalloc(&smids, kMaximumGrid * sizeof(unsigned int)));
  std::printf("%s\n", kHeader);
  const int grids[] = {108, 216};
  for (int trial = 0; trial < kTrials; ++trial) {
    for (int position = 0; position < 2; ++position) {
      const int grid = grids[(position + trial + rotation) % 2];
      run_cell(grid, rotation, trial, position, final_sink, starts, ends, smids);
    }
  }
  CHECK_CUDA(cudaFree(smids));
  CHECK_CUDA(cudaFree(ends));
  CHECK_CUDA(cudaFree(starts));
  CHECK_CUDA(cudaFree(final_sink));
  return EXIT_SUCCESS;
}
