// Independent BF16 grouped address-order probe. It stages routed A and expert
// B panels with cp.async, but has no MMA, activation, or matrix output.
#include <cuda_runtime.h>
#if defined(WITH_OUTPUT_TAIL)
#include <cuda_bf16.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call) do { const cudaError_t error = (call); if (error != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(error)); std::exit(1); } } while (0)

constexpr int kExperts = 128;
constexpr int kTokens = 32768;
constexpr int kTopk = 16;
constexpr int kRoutes = kTokens * kTopk;
constexpr int kBM = 128;
constexpr int kBN = 256;
constexpr int kBK = 64;
constexpr int kGroupM = 16;
constexpr int kThreads = 256;
constexpr int kStages = 4;
constexpr int kAVectors = kBM * kBK * 2 / 16;
constexpr int kBVectors = kBN * kBK * 2 / 16;
constexpr int kPanelBytes = (kAVectors + kBVectors) * 16;
constexpr int kSharedBytes = kStages * kPanelBytes;
static_assert(kPanelBytes == 49152 && kSharedBytes == 196608);

__device__ __forceinline__ void cp_vector(
    unsigned char* destination, const unsigned char* source, int source_bytes) {
  const unsigned int address = static_cast<unsigned int>(
      __cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::
               "r"(address), "l"(source), "r"(source_bytes) : "memory");
}

__device__ __forceinline__ void issue_panel(
    unsigned char* panel, const unsigned char* activation,
    const unsigned char* weights, const int* row_ids, int expert,
    int pid_n, int k_round, int k, int n, int source_topk, int mode) {
  if (mode != 2) {
    for (int vector = threadIdx.x; vector < kAVectors; vector += kThreads) {
      const int row = vector / (kBK * 2 / 16);
      const int k_vector = vector % (kBK * 2 / 16);
      const int source_row = row_ids[row];
      const bool valid = source_row < kRoutes;
      const std::size_t offset = std::size_t(valid ? source_row / source_topk : 0)
          * k * 2 + (k_round * kBK * 2 + k_vector * 16);
      cp_vector(panel + vector * 16, activation + offset, valid ? 16 : 0);
    }
  }
  if (mode != 1) {
    for (int vector = threadIdx.x; vector < kBVectors; vector += kThreads) {
      const int n_row = pid_n * kBN + vector / (kBK * 2 / 16);
      const int k_vector = vector % (kBK * 2 / 16);
      const std::size_t offset = (std::size_t(expert) * n + n_row) * k * 2
          + (k_round * kBK * 2 + k_vector * 16);
      cp_vector(panel + kAVectors * 16 + vector * 16,
                weights + offset, 16);
    }
  }
  asm volatile("cp.async.commit_group;\n" :: : "memory");
}

#if defined(WITH_OUTPUT_TAIL)
// The independent output tail follows the installed vLLM 0.30 PTX's
// warp/column mapping and register->shared->ldmatrix->global layout exchange.
// The preceding operand loop still has no MMA and produces no matrix result.
__device__ __forceinline__ void issue_output_tail(
    const int* sorted_ids, unsigned char* output, unsigned char* shared,
    int pid_m, int pid_n, int tail_enabled) {
  const unsigned tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid & 31;
  int ids[16];
  #pragma unroll
  for (int vector = 0; vector < 16; ++vector)
    ids[vector] = sorted_ids[pid_m * kBM + warp * 16 + vector];
  if (!tail_enabled) return;
  float seed = __uint2float_rn(
      static_cast<unsigned>(blockIdx.x * kThreads + threadIdx.x));
  uint4 transformed[16];
  const unsigned shared_base = static_cast<unsigned>(
      __cvta_generic_to_shared(shared));
  const unsigned write_base = shared_base + (tid & 3) * 8224 +
      ((tid & 4) ? 4112 : 0) + ((tid & 248) << 4);
  const unsigned read_base = shared_base +
      ((((tid & 24) << 10) | ((tid & 7) << 4)) ^
       (((tid & 24) << 2) | ((tid << 1) & 384)) ^
       ((tid & 32) ? 4112 : 0));
  #pragma unroll
  for (int vector = 0; vector < 16; ++vector) {
    uint4 packed;
    auto* word = reinterpret_cast<unsigned*>(&packed);
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const int element = vector * 8 + pair * 2;
      float a = seed + float(element) * 0.0078125f;
      float b = seed + float(element + 1) * 0.0078125f;
      __nv_bfloat162 bf = __floats2bfloat162_rn(a, b);
      word[pair] = *reinterpret_cast<unsigned*>(&bf);
    }
    const unsigned address = write_base ^ ((vector & 7) * 16);
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                 "r"(address), "r"(packed.x), "r"(packed.y),
                 "r"(packed.z), "r"(packed.w) : "memory");
    if (vector == 7 || vector == 15) {
      __syncthreads();
      #pragma unroll
      for (int item = 0; item < 8; ++item) {
        uint4& value = transformed[(vector & 8) + item];
        const unsigned address = read_base + item * 512;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                     "{%0,%1,%2,%3}, [%4];" :
                     "=r"(value.x), "=r"(value.y), "=r"(value.z),
                     "=r"(value.w) : "r"(address) : "memory");
      }
      if (vector == 7) __syncthreads();
    }
  }
  #pragma unroll
  for (int vector = 0; vector < 16; ++vector) {
    const int id = ids[vector];
    if (id >= kRoutes) continue;
    const std::size_t offset = std::size_t(id) * 1536 * 2 +
        pid_n * kBN * 2 + lane * 16;
    *reinterpret_cast<uint4*>(output + offset) = transformed[vector];
  }
}
#endif

__global__ __launch_bounds__(kThreads, 1) void grouped_address_stream(
    const unsigned char* __restrict__ activation,
    const unsigned char* __restrict__ weights,
    const int* __restrict__ sorted_ids,
    const int* __restrict__ tile_experts,
    std::uint32_t* __restrict__ checksums,
    int m_tiles, int k, int n, int source_topk, int mode
#if defined(WITH_OUTPUT_TAIL)
    , unsigned char* __restrict__ output, int tail_enabled
#endif
    ) {
  extern __shared__ __align__(16) unsigned char shared[];
  __shared__ int row_ids[kBM];
  __shared__ std::uint32_t warp_checksums[kThreads / 32];
  const int n_tiles = n / kBN;
  const int group_width = kGroupM * n_tiles;
  const int group = blockIdx.x / group_width;
  const int first_m = group * kGroupM;
  const int group_m = min(m_tiles - first_m, kGroupM);
  const int group_pid = blockIdx.x % group_width;
  const int pid_m = first_m + group_pid % group_m;
  const int pid_n = group_pid / group_m;
  if (threadIdx.x < kBM)
    row_ids[threadIdx.x] = sorted_ids[pid_m * kBM + threadIdx.x];
  __syncthreads();
  const int expert = tile_experts[pid_m];
  const int rounds = k / kBK;
  const int initial = min(rounds, kStages);
  for (int round = 0; round < initial; ++round)
    issue_panel(shared + round * kPanelBytes, activation, weights,
                row_ids, expert, pid_n, round, k, n, source_topk, mode);
  std::uint32_t checksum = 0;
  for (int round = 0; round < rounds; ++round) {
    if (round + kStages < rounds)
      asm volatile("cp.async.wait_group 3;\n" :: : "memory");
    else
      asm volatile("cp.async.wait_group 0;\n" :: : "memory");
    __syncthreads();
    const int slot = round % kStages;
    const int offset = mode == 2 ? kAVectors * 16 : 0;
    checksum += reinterpret_cast<volatile std::uint32_t*>(
        shared + slot * kPanelBytes + offset)[threadIdx.x];
    __syncthreads();
    if (round + kStages < rounds)
      issue_panel(shared + slot * kPanelBytes, activation, weights,
                  row_ids, expert, pid_n, round + kStages, k, n,
                  source_topk, mode);
  }
  for (int offset = 16; offset > 0; offset /= 2)
    checksum += __shfl_down_sync(0xffffffffu, checksum, offset);
  if ((threadIdx.x & 31) == 0)
    warp_checksums[threadIdx.x / 32] = checksum;
  __syncthreads();
  if (threadIdx.x == 0) {
    std::uint32_t total = 0;
    for (int warp = 0; warp < kThreads / 32; ++warp)
      total += warp_checksums[warp];
    checksums[blockIdx.x] = total;
  }
#if defined(WITH_OUTPUT_TAIL)
  __syncthreads();
  issue_output_tail(sorted_ids, output, shared, pid_m, pid_n, tail_enabled);
#endif
}

static std::uint32_t mix(std::uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  return value ^ (value >> 16);
}

int main(int argc, char** argv) {
  if (argc !=
#if defined(WITH_OUTPUT_TAIL)
      4
#else
      3
#endif
      || std::strcmp(argv[1], "w1") != 0 ||
      (std::strcmp(argv[2], "a") != 0 &&
       std::strcmp(argv[2], "b") != 0 &&
       std::strcmp(argv[2], "both") != 0)) {
#if defined(WITH_OUTPUT_TAIL)
    std::fprintf(stderr, "usage: probe w1 both control|tail\n");
#else
    std::fprintf(stderr, "usage: probe w1 a|b|both\n");
#endif
    return 2;
  }
#if defined(WITH_OUTPUT_TAIL)
  if (std::strcmp(argv[2], "both") != 0 ||
      (std::strcmp(argv[3], "control") != 0 &&
       std::strcmp(argv[3], "tail") != 0)) return 2;
  const int tail_enabled = std::strcmp(argv[3], "tail") == 0;
#endif
  const int mode = std::strcmp(argv[2], "a") == 0 ? 1 :
                   std::strcmp(argv[2], "b") == 0 ? 2 : 3;
  cudaDeviceProp properties{};
  int device_count = 0;
  CHECK(cudaGetDeviceCount(&device_count));
  CHECK(cudaGetDeviceProperties(&properties, 0));
  if (device_count != 1 || std::strstr(properties.name, "H200") == nullptr ||
      properties.major != 9 || properties.multiProcessorCount != 132) {
    std::fprintf(stderr, "requires one 132-SM H200\n");
    return 2;
  }
  std::vector<std::vector<int>> buckets(kExperts);
  for (int token = 0; token < kTokens; ++token) {
    const int start = int(mix(std::uint32_t(token) + 42u) % kExperts);
    for (int slot = 0; slot < kTopk; ++slot) {
      const int expert = (start + slot * 13) % kExperts;
      buckets[expert].push_back(token * kTopk + slot);
    }
  }
  std::vector<int> sorted_ids;
  std::vector<int> tile_experts;
  int min_rows = kRoutes, max_rows = 0;
  for (int expert = 0; expert < kExperts; ++expert) {
    min_rows = std::min(min_rows, int(buckets[expert].size()));
    max_rows = std::max(max_rows, int(buckets[expert].size()));
    const int padded = (int(buckets[expert].size()) + kBM - 1) / kBM * kBM;
    sorted_ids.insert(sorted_ids.end(), buckets[expert].begin(),
                      buckets[expert].end());
    sorted_ids.insert(sorted_ids.end(), padded - buckets[expert].size(),
                      kRoutes);
    tile_experts.insert(tile_experts.end(), padded / kBM, expert);
  }
  const int k = 2048;
  const int n = 1536;
  const int m_tiles = int(tile_experts.size());
  const int grid = m_tiles * (n / kBN);
  const std::size_t a_bytes = std::size_t(kTokens) * k * 2;
  const std::size_t b_bytes = std::size_t(kExperts) * n * k * 2;
  unsigned char *activation = nullptr, *weights = nullptr;
  int *gpu_sorted = nullptr, *gpu_experts = nullptr;
  std::uint32_t* gpu_checksums = nullptr;
#if defined(WITH_OUTPUT_TAIL)
  unsigned char* output = nullptr;
#endif
  CHECK(cudaMalloc(&activation, a_bytes));
  CHECK(cudaMalloc(&weights, b_bytes));
  CHECK(cudaMalloc(&gpu_sorted, sorted_ids.size() * sizeof(int)));
  CHECK(cudaMalloc(&gpu_experts, tile_experts.size() * sizeof(int)));
  CHECK(cudaMalloc(&gpu_checksums, grid * sizeof(std::uint32_t)));
#if defined(WITH_OUTPUT_TAIL)
  CHECK(cudaMalloc(&output, std::size_t(kRoutes) * n * 2));
#endif
  CHECK(cudaMemset(activation, 0x5a, a_bytes));
  CHECK(cudaMemset(weights, 0x6b, b_bytes));
  CHECK(cudaMemcpy(gpu_sorted, sorted_ids.data(), sorted_ids.size() * sizeof(int),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(gpu_experts, tile_experts.data(),
                   tile_experts.size() * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaFuncSetAttribute(grouped_address_stream,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes));
  int occupancy = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, grouped_address_stream, kThreads, kSharedBytes));
  if (occupancy != 1) {
    std::fprintf(stderr, "expected one resident CTA/SM, got %d\n", occupancy);
    return 3;
  }
  std::fprintf(stderr,
      "route rows=%d padded=%zu active_m_tiles=%d grid=%d min=%d max=%d "
      "A_bytes=%zu B_bytes=%zu occupancy=%d\n",
      kRoutes, sorted_ids.size(), m_tiles, grid, min_rows, max_rows,
      a_bytes, b_bytes, occupancy);
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  for (int trial = -2; trial < 7; ++trial) {
    CHECK(cudaEventRecord(start));
    grouped_address_stream<<<grid, kThreads, kSharedBytes>>>(
        activation, weights, gpu_sorted, gpu_experts, gpu_checksums,
        m_tiles, k, n, kTopk, mode
#if defined(WITH_OUTPUT_TAIL)
        , output, tail_enabled
#endif
        );
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    if (trial >= 0) {
      float milliseconds = 0;
      CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
      std::printf("sample=%d,mode=%s,milliseconds=%.9g\n",
                  trial, argv[2], milliseconds);
    }
  }
  CHECK(cudaGetLastError());
  std::uint32_t checksum = 0;
  CHECK(cudaMemcpy(&checksum, gpu_checksums, sizeof(checksum),
                   cudaMemcpyDeviceToHost));
  std::fprintf(stderr, "validated checksum=%u\n", checksum);
#if defined(WITH_OUTPUT_TAIL)
  if (tail_enabled) {
    unsigned char one = 0;
    CHECK(cudaMemcpy(&one, output, 1, cudaMemcpyDeviceToHost));
    std::fprintf(stderr, "output_byte=%u\n", unsigned(one));
  }
#endif
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  CHECK(cudaFree(activation));
  CHECK(cudaFree(weights));
  CHECK(cudaFree(gpu_sorted));
  CHECK(cudaFree(gpu_experts));
  CHECK(cudaFree(gpu_checksums));
#if defined(WITH_OUTPUT_TAIL)
  CHECK(cudaFree(output));
#endif
  return 0;
}
