#ifndef MIDLAYER_ASCEND_910B_BLOCK_REDUCE_MAX_CONTRACT_H_
#define MIDLAYER_ASCEND_910B_BLOCK_REDUCE_MAX_CONTRACT_H_

#include <cstdint>

namespace block_reduce_max_contract {

constexpr uint32_t kFloatBlockSize = 8;
constexpr uint32_t kFloatVectorSize = 64;
constexpr uint32_t kMaximumColumnCount = 1024;
constexpr uint32_t kMaximumRowChunk = 64;
constexpr uint32_t kMaximumSourceElements = 8192;
constexpr uint32_t kReduceUbSize = 1024;
constexpr uint32_t kTemporaryElements = 2048;
constexpr uint32_t kRowOutputElements = 64;
constexpr uint32_t kColumnAlignmentUnmasked = 16;
constexpr uint32_t kColumnAlignmentMasked = 32;
constexpr double kSystemCounterTickNs = 20.0;

#ifdef BLOCK_REDUCE_MAX_CONTRACT_AICORE
#define BLOCK_REDUCE_MAX_INLINE __aicore__ inline
#else
#define BLOCK_REDUCE_MAX_INLINE inline
#endif

BLOCK_REDUCE_MAX_INLINE uint32_t RoundUp(
        uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

BLOCK_REDUCE_MAX_INLINE uint32_t CeilDiv(
        uint32_t value, uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

BLOCK_REDUCE_MAX_INLINE bool IsSupportedCell(
        uint32_t row_chunk_round, uint32_t column_count,
        uint32_t column_alignment) {
    return row_chunk_round >= kFloatBlockSize
        && row_chunk_round <= kMaximumRowChunk
        && row_chunk_round % kFloatBlockSize == 0
        && column_count >= 1
        && column_count <= kMaximumColumnCount
        && (column_alignment == kColumnAlignmentUnmasked
            || column_alignment == kColumnAlignmentMasked)
        && row_chunk_round * RoundUp(column_count, column_alignment)
            <= kMaximumSourceElements;
}

#undef BLOCK_REDUCE_MAX_INLINE

}  // namespace block_reduce_max_contract

#endif  // MIDLAYER_ASCEND_910B_BLOCK_REDUCE_MAX_CONTRACT_H_
