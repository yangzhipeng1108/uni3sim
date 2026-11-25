#pragma once

#include <stdint.h>

#if defined(GEMM_V3_STORE_AICORE)
#define GEMM_V3_STORE_INLINE __aicore__ inline
#else
#define GEMM_V3_STORE_INLINE constexpr
#endif

namespace gemm_v3_store_contract {

constexpr int64_t kBfloat16 = 0;
constexpr int64_t kFloat32 = 1;
constexpr int64_t kK = 16;
constexpr int64_t kMaximumM = 128;
constexpr int64_t kMaximumN = 256;
constexpr int64_t kShortInvocations = 32;
constexpr int64_t kLongInvocations = 96;
constexpr int64_t kFiniteShortInvocations = 32;
constexpr int64_t kFiniteLongInvocations = 96;
constexpr int64_t kLongStreamShortInvocations = 96;
constexpr int64_t kLongStreamLongInvocations = 1195;
constexpr int64_t kMaximumBlocks = 24;
constexpr int64_t kCycleCacheLineBytes = 64;
constexpr int64_t kCycleSlotElements =
    kCycleCacheLineBytes / static_cast<int64_t>(sizeof(int64_t));

GEMM_V3_STORE_INLINE bool is_frozen_cell(
        int64_t dtype, int64_t m, int64_t n, int64_t block_dim) {
    const bool frozen_geometry =
        (dtype == kBfloat16 && m == 128 && n == 256)
        || (dtype == kBfloat16 && m == 128 && n == 128)
        || (dtype == kBfloat16 && m == 64 && n == 256)
        || (dtype == kFloat32 && m == 128 && n == 128);
    return frozen_geometry && (block_dim == 1 || block_dim == 24);
}

GEMM_V3_STORE_INLINE bool is_frozen_finite_cell(
        int64_t dtype, int64_t m, int64_t n, int64_t block_dim) {
    if (dtype != kBfloat16) {
        return false;
    }
    return (m == 16 && n == 176 && block_dim == 22)
        || (m == 32 && n == 256
            && (block_dim == 16 || block_dim == 20 || block_dim == 22))
        || (m == 48 && n == 256 && block_dim == 22)
        || (m == 64 && n == 256
            && (block_dim == 20 || block_dim == 22))
        || (m == 128 && n == 256 && block_dim == 24);
}

GEMM_V3_STORE_INLINE bool is_frozen_long_stream_cell(
        int64_t dtype, int64_t m, int64_t n, int64_t block_dim) {
    return dtype == kBfloat16 && m == 128 && n == 256
        && block_dim == 24;
}

GEMM_V3_STORE_INLINE int64_t output_element_bytes(int64_t dtype) {
    return dtype == kBfloat16 ? 2 : dtype == kFloat32 ? 4 : 0;
}

GEMM_V3_STORE_INLINE int64_t tile_elements(int64_t m, int64_t n) {
    return m > 0 && n > 0 ? m * n : 0;
}

GEMM_V3_STORE_INLINE int64_t block_elements(
        int64_t m, int64_t n, int64_t invocations) {
    const int64_t elements = tile_elements(m, n);
    return elements > 0 && invocations > 0
        ? elements * (invocations + 1) : 0;
}

}  // namespace gemm_v3_store_contract

#undef GEMM_V3_STORE_INLINE
