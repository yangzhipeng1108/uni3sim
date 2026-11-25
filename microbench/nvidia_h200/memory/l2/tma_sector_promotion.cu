// Independent TMA L2-promotion sector accounting; no matrix arithmetic.
#define main original_operand_stream_main
#include "fp8_tma_operand_stream.cu"
#undef main

static CUtensorMap promoted_map(void* base, int promotion) {
  CUtensorMap map;
  cuuint64_t dims[2] = {16384, 132 * 64};
  cuuint64_t strides[1] = {16384};
  cuuint32_t box[2] = {128, 64}, element_strides[2] = {1, 1};
  CUtensorMapL2promotion selected;
  switch (promotion) {
    case 0: selected = CU_TENSOR_MAP_L2_PROMOTION_NONE; break;
    case 128: selected = CU_TENSOR_MAP_L2_PROMOTION_L2_128B; break;
    case 256: selected = CU_TENSOR_MAP_L2_PROMOTION_L2_256B; break;
    default: std::exit(2);
  }
  CUresult status = encode_tiled()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
      base, dims, strides, box, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      selected, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (status != CUDA_SUCCESS) std::exit(3);
  return map;
}

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  const int promotion = std::atoi(argv[1]), rounds = std::atoi(argv[2]);
  if (rounds != 8 && rounds != 128) return 2;
  uint8_t *input, *output, *evict;
  CHECK(cudaMalloc(&input, 16384ull * 132 * 64));
  CHECK(cudaMalloc(&output, 132));
  CHECK(cudaMalloc(&evict, 256ull * 1024 * 1024));
  CHECK(cudaMemset(input, 1, 16384ull * 132 * 64));
  auto map = promoted_map(input, promotion);
  constexpr int shared_bytes = 8 * 64 * 128 + 256;
  CHECK(cudaFuncSetAttribute(tma_stream_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  for (int repeat = 0; repeat < 3; ++repeat) {
    if (rounds == 128) CHECK(cudaMemset(evict, repeat, 256ull * 1024 * 1024));
    tma_stream_kernel<<<132, 384, shared_bytes>>>(map, map, output,
        8, 64, 0, rounds, 0, 0, 256, 132);
    CHECK(cudaDeviceSynchronize());
  }
  std::printf("{\"promotion_bytes\":%d,\"rounds\":%d,\"payload_bytes\":%llu}\n",
      promotion, rounds, 132ull * 64 * 128 * rounds);
  CHECK(cudaFree(evict)); CHECK(cudaFree(output)); CHECK(cudaFree(input));
}
