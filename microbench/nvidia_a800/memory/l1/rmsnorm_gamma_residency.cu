#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kGrid = 1024;
constexpr int kBlock = 1024;
constexpr int kStaticSharedBytes = 168;

struct Case {
  const char* name;
  int hidden;
  bool private_gamma;
};

constexpr Case kCases[] = {
    {"shared_gamma_h2048", 2048, false},
    {"shared_gamma_h8192", 8192, false},
    {"shared_gamma_h16384", 16384, false},
    {"private_gamma_h16384_control", 16384, true},
    {"private_gamma_h2048_control", 2048, true},
    {"private_gamma_h8192_control", 8192, true},
};

__device__ __forceinline__ uint16_t load_cached_bf16(const uint16_t* address) {
  uint16_t value;
  asm volatile("ld.global.ca.u16 %0, [%1];" : "=h"(value) : "l"(address));
  return value;
}

__device__ __forceinline__ float bf16_to_float(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

extern "C" __global__ __launch_bounds__(kBlock, 2)
void rmsnorm_gamma_residency_probe(const uint16_t* input_a,
                                   const uint16_t* input_b,
                                   const uint16_t* gamma,
                                   uint16_t* output,
                                   int hidden,
                                   int private_gamma) {
  __shared__ volatile unsigned char occupancy_anchor[kStaticSharedBytes];
  const size_t row = static_cast<size_t>(blockIdx.x) * hidden;
  float acc_a = 0.0f;
  float acc_b = 0.0f;

  for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
    acc_a += bf16_to_float(load_cached_bf16(input_a + row + column));
  }
  for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
    acc_b += bf16_to_float(load_cached_bf16(input_b + row + column));
  }
  if (threadIdx.x == 0) {
    occupancy_anchor[kStaticSharedBytes - 1] =
        static_cast<unsigned char>(__float_as_uint(acc_a + acc_b));
  }
  __syncthreads();

  uint32_t register_anchor_0;
  uint32_t register_anchor_1;
  uint32_t register_anchor_2;
  uint32_t register_anchor_3;
  uint32_t register_anchor_4;
  uint32_t register_anchor_5;
  uint32_t register_anchor_6;
  uint32_t register_anchor_7;
  uint32_t register_anchor_8;
  uint32_t register_anchor_9;
  uint32_t register_anchor_10;
  uint32_t register_anchor_11;
  uint32_t register_anchor_12;
  uint32_t register_anchor_13;
  uint32_t register_anchor_14;
  uint32_t register_anchor_15;
  uint32_t register_anchor_16;
  uint32_t register_anchor_17;
  uint32_t register_anchor_18;
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_0) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_1) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_2) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_3) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_4) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_5) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_6) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_7) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_8) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_9) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_10) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_11) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_12) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_13) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_14) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_15) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_16) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_17) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_18) : "r"(threadIdx.x));
  asm volatile(
      "add.u32 %0, %0, %1;\n\t"
      "add.u32 %0, %0, %2;\n\t"
      "add.u32 %0, %0, %3;\n\t"
      "add.u32 %0, %0, %4;\n\t"
      "add.u32 %0, %0, %5;\n\t"
      "add.u32 %0, %0, %6;\n\t"
      "add.u32 %0, %0, %7;"
      "add.u32 %0, %0, %8;\n\t"
      "add.u32 %0, %0, %9;\n\t"
      "add.u32 %0, %0, %10;\n\t"
      "add.u32 %0, %0, %11;\n\t"
      "add.u32 %0, %0, %12;\n\t"
      "add.u32 %0, %0, %13;\n\t"
      "add.u32 %0, %0, %14;\n\t"
      "add.u32 %0, %0, %15;\n\t"
      "add.u32 %0, %0, %16;\n\t"
      "add.u32 %0, %0, %17;\n\t"
      "add.u32 %0, %0, %18;"
      : "+r"(register_anchor_0)
      : "r"(register_anchor_1), "r"(register_anchor_2),
        "r"(register_anchor_3), "r"(register_anchor_4),
        "r"(register_anchor_5), "r"(register_anchor_6),
        "r"(register_anchor_7), "r"(register_anchor_8),
        "r"(register_anchor_9), "r"(register_anchor_10),
        "r"(register_anchor_11), "r"(register_anchor_12),
        "r"(register_anchor_13), "r"(register_anchor_14),
        "r"(register_anchor_15), "r"(register_anchor_16),
        "r"(register_anchor_17), "r"(register_anchor_18));

  const size_t gamma_row = private_gamma ? row : 0;
  for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
    const float input =
        bf16_to_float(load_cached_bf16(input_a + row + column));
    const float weight =
        bf16_to_float(load_cached_bf16(gamma + gamma_row + column));
    const float value = input * weight + acc_a * 0.0f + acc_b * 0.0f +
                        static_cast<float>(register_anchor_0) * 0.0f +
                        static_cast<float>(occupancy_anchor[kStaticSharedBytes - 1]) * 0.0f;
    output[row + column] = static_cast<uint16_t>(__float_as_uint(value) >> 16);
  }
}

void CHECK_CUDA(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    std::exit(1);
  }
}

size_t gamma_elements(const Case& probe_case) {
  return static_cast<size_t>(probe_case.hidden) *
         (probe_case.private_gamma ? kGrid : 1);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s CASE_ID\n", argv[0]);
    return 2;
  }
  char* end = nullptr;
  const long case_id = std::strtol(argv[1], &end, 10);
  if (*argv[1] == '\0' || *end != '\0' || case_id < 0 ||
      case_id >= static_cast<long>(sizeof(kCases) / sizeof(kCases[0]))) {
    std::fprintf(stderr, "CASE_ID must be in [0, 5]\n");
    return 2;
  }
  const Case& probe_case = kCases[case_id];
  cudaDeviceProp properties{};
  CHECK_CUDA(cudaGetDeviceProperties(&properties, 0),
             "cudaGetDeviceProperties");
  int resident_ctas_per_sm = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                 &resident_ctas_per_sm, rmsnorm_gamma_residency_probe,
                 kBlock, 0),
             "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
  const size_t input_elements = static_cast<size_t>(kGrid) * probe_case.hidden;
  const size_t input_bytes = input_elements * sizeof(uint16_t);
  const size_t gamma_bytes = gamma_elements(probe_case) * sizeof(uint16_t);
  uint16_t* input_a = nullptr;
  uint16_t* input_b = nullptr;
  uint16_t* gamma = nullptr;
  uint16_t* output = nullptr;
  CHECK_CUDA(cudaMalloc(&input_a, input_bytes), "cudaMalloc input_a");
  CHECK_CUDA(cudaMalloc(&input_b, input_bytes), "cudaMalloc input_b");
  CHECK_CUDA(cudaMalloc(&gamma, gamma_bytes), "cudaMalloc gamma");
  CHECK_CUDA(cudaMalloc(&output, input_bytes), "cudaMalloc output");
  CHECK_CUDA(cudaMemset(input_a, 0x3f, input_bytes), "cudaMemset input_a");
  CHECK_CUDA(cudaMemset(input_b, 0x40, input_bytes), "cudaMemset input_b");
  CHECK_CUDA(cudaMemset(gamma, 0x3f, gamma_bytes), "cudaMemset gamma");
  CHECK_CUDA(cudaMemset(output, 0, input_bytes), "cudaMemset output");
  rmsnorm_gamma_residency_probe<<<kGrid, kBlock>>>(
      input_a, input_b, gamma, output, probe_case.hidden,
      probe_case.private_gamma ? 1 : 0);
  CHECK_CUDA(cudaGetLastError(), "rmsnorm_gamma_residency_probe launch");
  CHECK_CUDA(cudaDeviceSynchronize(), "rmsnorm_gamma_residency_probe sync");
  uint16_t checksum = 0;
  CHECK_CUDA(cudaMemcpy(&checksum, output, sizeof(checksum),
                        cudaMemcpyDeviceToHost), "cudaMemcpy checksum");
  const size_t private_input_read_bytes = 3 * input_bytes;
  const size_t logical_gamma_read_bytes = input_bytes;
  const size_t logical_read_bytes =
      private_input_read_bytes + logical_gamma_read_bytes;
  std::printf("%s,%s,%d,%d,%d,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%d,%u\n",
              properties.name, probe_case.name,
              probe_case.hidden, kGrid, kBlock, input_bytes, input_bytes,
              gamma_bytes, input_bytes, private_input_read_bytes,
              logical_gamma_read_bytes, logical_read_bytes,
              resident_ctas_per_sm, static_cast<unsigned>(checksum));
  CHECK_CUDA(cudaFree(output), "cudaFree output");
  CHECK_CUDA(cudaFree(gamma), "cudaFree gamma");
  CHECK_CUDA(cudaFree(input_b), "cudaFree input_b");
  CHECK_CUDA(cudaFree(input_a), "cudaFree input_a");
  return 0;
}
