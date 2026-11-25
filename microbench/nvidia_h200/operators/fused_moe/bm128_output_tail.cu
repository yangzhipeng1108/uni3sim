// Independent BF16 grouped-MoE output tail probe. Each CTA converts a
// synthetic FP32 tile and scatters BF16 values through route IDs. No operand
// loads, MMA, activation, expert weights, or complete MoE execution.
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  std::exit(1); } } while (0)

constexpr int kTokens = 32768, kTopk = 16, kExperts = 128;
constexpr int kRoutes = kTokens * kTopk;
constexpr int kBM = 128, kBN = 256, kGroupM = 16, kThreads = 256;
constexpr int kSharedBytes = 196608;

__global__ __launch_bounds__(kThreads, 1) void output_tail(
    const int* __restrict__ sorted_ids, unsigned char* __restrict__ output,
    int m_tiles, int n, int enabled) {
  extern __shared__ unsigned char occupancy_only[];
  const int n_tiles = n / kBN;
  const int group_width = kGroupM * n_tiles;
  const int group = blockIdx.x / group_width;
  const int first_m = group * kGroupM;
  const int group_m = min(m_tiles - first_m, kGroupM);
  const int group_pid = blockIdx.x % group_width;
  const int pid_m = first_m + group_pid % group_m;
  const int pid_n = group_pid / group_m;
#if defined(SOURCE_COLUMN_LAYOUT) || defined(SOURCE_SHARED_LAYOUT)
  // Source PTX: each warp owns 16 M rows and lane owns one 8-BF16 vector
  // within each row. Adjacent lanes therefore store adjacent 16-B vectors.
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  int row_ids[16];
  #pragma unroll
  for (int vector = 0; vector < 16; ++vector)
    row_ids[vector] = sorted_ids[pid_m * kBM + warp * 16 + vector];
  if (!enabled) return;
#else
  const int row = threadIdx.x / 2;
  const int row_id = sorted_ids[pid_m * kBM + row];
  if (!enabled || row_id >= kRoutes) return;
  const int half = threadIdx.x & 1;
  const std::size_t offset = std::size_t(row_id) * n * 2 +
      pid_n * kBN * 2 + half * 128 * 2;
  auto* dst = reinterpret_cast<uint4*>(output + offset);
#endif
  float seed = __uint2float_rn(
      static_cast<unsigned>(blockIdx.x * kThreads + threadIdx.x));
#if defined(SOURCE_SHARED_LAYOUT)
  uint4 transformed[16];
  const unsigned tid = threadIdx.x;
  const unsigned shared_base = static_cast<unsigned>(
      __cvta_generic_to_shared(occupancy_only));
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
        const unsigned source_address = read_base + item * 512;
        uint4& value = transformed[(vector & 8) + item];
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{%0,%1,%2,%3}, [%4];" :
            "=r"(value.x), "=r"(value.y), "=r"(value.z),
            "=r"(value.w) : "r"(source_address) : "memory");
      }
      if (vector == 7) __syncthreads();
    }
  }
  #pragma unroll
  for (int vector = 0; vector < 16; ++vector) {
    const int row_id = row_ids[vector];
    if (row_id >= kRoutes) continue;
    const std::size_t offset = std::size_t(row_id) * n * 2 +
        pid_n * kBN * 2 + lane * 16;
    *reinterpret_cast<uint4*>(output + offset) = transformed[vector];
  }
#else
  #pragma unroll
  for (int vector = 0; vector < 16; ++vector) {
#if defined(SOURCE_COLUMN_LAYOUT)
    const int row_id = row_ids[vector];
    if (row_id >= kRoutes) continue;
    const std::size_t offset = std::size_t(row_id) * n * 2 +
        pid_n * kBN * 2 + lane * 16;
    auto* dst = reinterpret_cast<uint4*>(output + offset);
#endif
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
    dst[
#if defined(SOURCE_COLUMN_LAYOUT)
        0
#else
        vector
#endif
    ] = packed;
  }
  // Keep the dynamic allocation as an occupancy constraint without reading
  // from it; the output instruction path is the only measured device work.
#if !defined(SOURCE_COLUMN_LAYOUT)
  if (occupancy_only == nullptr) dst[0] = make_uint4(0, 0, 0, 0);
#endif
#endif
}

static std::uint32_t mix(std::uint32_t value) {
  value ^= value >> 16; value *= 0x7feb352du;
  value ^= value >> 15; value *= 0x846ca68bu;
  return value ^ (value >> 16);
}

int main(int argc, char** argv) {
  if (argc != 3 || (std::strcmp(argv[1], "w1") != 0 &&
      std::strcmp(argv[1], "w2") != 0) ||
      (std::strcmp(argv[2], "control") != 0 &&
       std::strcmp(argv[2], "store") != 0)) {
    std::fprintf(stderr, "usage: output_tail w1|w2 control|store\n");
    return 2;
  }
  cudaDeviceProp gpu{}; int count = 0;
  CHECK(cudaGetDeviceCount(&count)); CHECK(cudaGetDeviceProperties(&gpu, 0));
  if (count != 1 || std::strstr(gpu.name, "H200") == nullptr ||
      gpu.multiProcessorCount != 132) return 2;
  std::vector<std::vector<int>> buckets(kExperts);
  for (int token = 0; token < kTokens; ++token) {
    const int start = int(mix(std::uint32_t(token) + 42u) % kExperts);
    for (int slot = 0; slot < kTopk; ++slot)
      buckets[(start + slot * 13) % kExperts].push_back(token * kTopk + slot);
  }
  std::vector<int> ids;
  for (int expert = 0; expert < kExperts; ++expert) {
    ids.insert(ids.end(), buckets[expert].begin(), buckets[expert].end());
    ids.insert(ids.end(), (kBM - buckets[expert].size() % kBM) % kBM,
               kRoutes);
  }
  const int n = std::strcmp(argv[1], "w1") == 0 ? 1536 : 2048;
  const int m_tiles = int(ids.size()) / kBM;
  const int grid = m_tiles * (n / kBN);
  int* gpu_ids = nullptr; unsigned char* output = nullptr;
  CHECK(cudaMalloc(&gpu_ids, ids.size() * sizeof(int)));
  CHECK(cudaMalloc(&output, std::size_t(kRoutes) * n * 2));
  CHECK(cudaMemcpy(gpu_ids, ids.data(), ids.size() * sizeof(int),
                   cudaMemcpyHostToDevice));
  CHECK(cudaFuncSetAttribute(output_tail,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes));
  int occupancy = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, output_tail, kThreads, kSharedBytes));
  if (occupancy != 1) return 3;
  const int enabled = std::strcmp(argv[2], "store") == 0;
  std::fprintf(stderr,
      "n=%d routes=%d padded=%zu grid=%d threads=%d shared=%d occupancy=%d\n",
      n, kRoutes, ids.size(), grid, kThreads, kSharedBytes, occupancy);
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
  for (int trial = -2; trial < 7; ++trial) {
    CHECK(cudaEventRecord(start));
    output_tail<<<grid, kThreads, kSharedBytes>>>(
        gpu_ids, output, m_tiles, n, enabled);
    CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
    if (trial >= 0) {
      float milliseconds = 0;
      CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
      std::printf("variant=%s,mode=%s,sample=%d,milliseconds=%.9g\n",
                  argv[1], argv[2], trial, milliseconds);
    }
  }
  CHECK(cudaGetLastError());
  if (enabled) {
    unsigned char one = 0;
    CHECK(cudaMemcpy(&one, output, 1, cudaMemcpyDeviceToHost));
    std::fprintf(stderr, "output_byte=%u\n", unsigned(one));
  }
  CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
  CHECK(cudaFree(gpu_ids)); CHECK(cudaFree(output));
  return 0;
}
