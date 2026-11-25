// Independent BM128/BN256/BK64 grouped-GEMM operand read probe for H200.
// Reads BF16 activation and weight tiles into shared memory in 32 K rounds.
// The checksum keeps each load live. There is no MMA, routing, or MoE output.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); \
  } \
} while (0)

constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 64;
constexpr int K = 2048;
constexpr int TOKENS = 4096;
constexpr int GRID = 4096;
constexpr int THREADS = 256;
constexpr int A_WORDS = BM * BK * 2 / sizeof(uint4);
constexpr int B_WORDS = BN * BK * 2 / sizeof(uint4);

__global__ void initialize(uint32_t* words, size_t count) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) words[i] = uint32_t(i * 747796405u + 2891336453u);
}

template<int WEIGHT_TILES>
__global__ void operand_read(const uint4* __restrict__ activation,
                             const uint4* __restrict__ weights,
                             const int* __restrict__ row_indices,
                             uint32_t* __restrict__ output) {
  extern __shared__ uint4 tile[];
  __shared__ uint32_t warp_checksums[THREADS / 32];
  const int tid = threadIdx.x;
  const int weight_tile = blockIdx.x % WEIGHT_TILES;
  uint32_t checksum = 0;
  for (int group = 0; group < K / BK; ++group) {
    for (int word = tid; word < A_WORDS; word += THREADS) {
      int row = word / (BK * 2 / sizeof(uint4));
      int column = word % (BK * 2 / sizeof(uint4));
      int source_row = row_indices[blockIdx.x * BM + row];
      tile[word] = activation[size_t(source_row) * (K * 2 / sizeof(uint4))
                              + group * (BK * 2 / sizeof(uint4)) + column];
    }
    for (int word = tid; word < B_WORDS; word += THREADS) {
      tile[A_WORDS + word] = weights[
          size_t(weight_tile) * (K * BN * 2 / sizeof(uint4))
          + group * B_WORDS + word];
    }
    __syncthreads();
    for (int word = tid; word < A_WORDS + B_WORDS; word += THREADS) {
      uint4 v = tile[word];
      checksum ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    __syncthreads();
  }
  for (int offset = 16; offset > 0; offset /= 2)
    checksum ^= __shfl_xor_sync(0xffffffffu, checksum, offset);
  if ((tid & 31) == 0) warp_checksums[tid / 32] = checksum;
  __syncthreads();
  if (tid == 0) {
    uint32_t total = 0;
    for (int i = 0; i < THREADS / 32; ++i) total ^= warp_checksums[i];
    output[blockIdx.x] = total;
  }
}

template<int WEIGHT_TILES>
void run_case(const char* name) {
  constexpr size_t activation_bytes = size_t(TOKENS) * K * 2;
  constexpr size_t weight_bytes = size_t(WEIGHT_TILES) * K * BN * 2;
  constexpr size_t index_count = size_t(GRID) * BM;
  uint32_t *activation, *weights, *output;
  int* indices;
  CHECK(cudaMalloc(&activation, activation_bytes));
  CHECK(cudaMalloc(&weights, weight_bytes));
  CHECK(cudaMalloc(&indices, index_count * sizeof(int)));
  CHECK(cudaMalloc(&output, GRID * sizeof(uint32_t)));
  initialize<<<(activation_bytes / 4 + 255) / 256, 256>>>(activation, activation_bytes / 4);
  initialize<<<(weight_bytes / 4 + 255) / 256, 256>>>(weights, weight_bytes / 4);
  CHECK(cudaGetLastError());
  int* host_indices = static_cast<int*>(malloc(index_count * sizeof(int)));
  if (host_indices == nullptr) exit(2);
  for (size_t i = 0; i < index_count; ++i)
    host_indices[i] = int((i * 4051u + (i / BM) * 17u) % TOKENS);
  CHECK(cudaMemcpy(indices, host_indices, index_count * sizeof(int), cudaMemcpyHostToDevice));
  free(host_indices);
  constexpr int shared_bytes = (A_WORDS + B_WORDS) * sizeof(uint4);
  CHECK(cudaFuncSetAttribute(operand_read<WEIGHT_TILES>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  for (int i = 0; i < 30; ++i)
    operand_read<WEIGHT_TILES><<<GRID, THREADS, shared_bytes>>>(
        reinterpret_cast<uint4*>(activation), reinterpret_cast<uint4*>(weights),
        indices, output);
  CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, end;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&end));
  constexpr size_t logical_bytes = size_t(GRID) * K * (BM + BN) * 2;
  printf("case,sample,weight_tiles,activation_bytes,weight_bytes,logical_bytes,milliseconds,logical_gb_s\n");
  for (int i = 0; i < 7; ++i) {
    CHECK(cudaEventRecord(start));
    operand_read<WEIGHT_TILES><<<GRID, THREADS, shared_bytes>>>(
        reinterpret_cast<uint4*>(activation), reinterpret_cast<uint4*>(weights),
        indices, output);
    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));
    float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, end));
    printf("%s,%d,%d,%zu,%zu,%zu,%.9g,%.9g\n", name, i, WEIGHT_TILES,
           activation_bytes, weight_bytes, logical_bytes, milliseconds,
           double(logical_bytes) / (milliseconds * 1e6));
  }
  uint32_t checksum = 0;
  CHECK(cudaMemcpy(&checksum, output, sizeof(checksum), cudaMemcpyDeviceToHost));
  fprintf(stderr, "validated %s checksum=%u\n", name, checksum);
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(end));
  CHECK(cudaFree(activation));
  CHECK(cudaFree(weights));
  CHECK(cudaFree(indices));
  CHECK(cudaFree(output));
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s l2|stream\n", argv[0]);
    return 2;
  }
  cudaDeviceProp prop;
  int device_count = 0;
  CHECK(cudaGetDeviceCount(&device_count));
  CHECK(cudaGetDeviceProperties(&prop, 0));
  if (device_count != 1 || strstr(prop.name, "H200") == nullptr
      || prop.major != 9 || prop.multiProcessorCount != 132) {
    fprintf(stderr, "requires one 132-SM H200, got %s with %d SMs\n",
            prop.name, prop.multiProcessorCount);
    return 2;
  }
  if (strcmp(argv[1], "l2") == 0)
    run_case<32>("l2_48mib_unique");
  else if (strcmp(argv[1], "stream") == 0)
    run_case<128>("stream_144mib_unique");
  else return 2;
  return 0;
}
