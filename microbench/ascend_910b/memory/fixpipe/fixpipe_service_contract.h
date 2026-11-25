#pragma once

#include <stdint.h>

#if defined(FIXPIPE_SERVICE_AICORE)
#define FIXPIPE_SERVICE_INLINE __aicore__ inline
#else
#define FIXPIPE_SERVICE_INLINE constexpr
#endif

namespace fixpipe_service_contract {

constexpr int64_t kTargetBytes = 512 * 1024;
constexpr int64_t kLargeRowThreshold = 16;
constexpr int64_t kMinimumLargeRowInvocations = 128;
constexpr int64_t kMiddleIntervalMultiplier = 2;
constexpr int64_t kLongIntervalMultiplier = 4;
constexpr int64_t kMaximumInt64 = 9223372036854775807LL;
constexpr int64_t kMaximumBlocks = 24;
constexpr int64_t kDestinationGuardElements = 16;
constexpr int64_t kDestinationAlignmentElements = 128;
constexpr int64_t kDestinationBaseOffsetElements =
    kDestinationAlignmentElements;
constexpr int64_t kBlockAlignmentElements = 1024;
constexpr int64_t kCycleCacheLineBytes = 64;
constexpr int64_t kCycleSlotElements =
    kCycleCacheLineBytes / static_cast<int64_t>(sizeof(int64_t));

struct InvocationCounts {
    int64_t short_count;
    int64_t long_count;
};

struct DestinationExtent {
    int64_t tile_span_elements;
    int64_t slot_pitch_elements;
    int64_t slot_count;
    int64_t block_elements;
    int64_t block_offset;
    int64_t maximum_invocation_offset;
    int64_t maximum_destination_offset;
    int64_t warmup_destination_offset;
    int64_t destination_end_offset;
};

FIXPIPE_SERVICE_INLINE bool checked_positive_multiply(
        int64_t left, int64_t right, int64_t* product) {
    if (product == nullptr || left <= 0 || right <= 0
            || left > kMaximumInt64 / right) {
        return false;
    }
    *product = left * right;
    return true;
}

FIXPIPE_SERVICE_INLINE bool checked_nonnegative_multiply(
        int64_t left, int64_t right, int64_t* product) {
    if (product == nullptr || left < 0 || right < 0
            || (left != 0
                && right > kMaximumInt64 / left)) {
        return false;
    }
    *product = left * right;
    return true;
}

FIXPIPE_SERVICE_INLINE bool checked_nonnegative_add(
        int64_t left, int64_t right, int64_t* sum) {
    if (sum == nullptr || left < 0 || right < 0
            || left > kMaximumInt64 - right) {
        return false;
    }
    *sum = left + right;
    return true;
}

FIXPIPE_SERVICE_INLINE bool checked_positive_round_up(
        int64_t value, int64_t alignment, int64_t* rounded) {
    if (rounded == nullptr || value <= 0 || alignment <= 0) {
        return false;
    }
    const int64_t remainder = value % alignment;
    if (remainder == 0) {
        *rounded = value;
        return true;
    }
    return checked_nonnegative_add(value, alignment - remainder, rounded);
}

FIXPIPE_SERVICE_INLINE bool checked_row_major_span(
        int64_t m, int64_t n, int64_t dst_stride, int64_t* span_elements) {
    if (span_elements == nullptr || m <= 0 || n <= 0
            || dst_stride < n) {
        return false;
    }
    int64_t row_offset = 0;
    return checked_nonnegative_multiply(
            m - 1, dst_stride, &row_offset)
        && checked_nonnegative_add(row_offset, n, span_elements);
}

FIXPIPE_SERVICE_INLINE bool checked_destination_extent(
        int64_t block_index, int64_t invocations, int64_t destination_slots,
        int64_t tile_span_elements, DestinationExtent* extent) {
    if (extent == nullptr || block_index < 0 || invocations <= 0
            || destination_slots <= 0 || destination_slots > invocations
            || tile_span_elements <= 0) {
        return false;
    }
    int64_t guarded_span_elements = 0;
    int64_t slot_pitch_elements = 0;
    int64_t arena_slot_count = 0;
    int64_t unaligned_block_elements = 0;
    int64_t block_elements = 0;
    int64_t block_offset = 0;
    int64_t maximum_invocation_offset = 0;
    int64_t maximum_destination_offset = 0;
    int64_t warmup_destination_offset = 0;
    int64_t destination_end_offset = 0;
    if (!checked_nonnegative_add(
                kDestinationBaseOffsetElements, tile_span_elements,
                &guarded_span_elements)
            || !checked_nonnegative_add(
                guarded_span_elements, kDestinationGuardElements,
                &guarded_span_elements)
            || !checked_positive_round_up(
                guarded_span_elements, kDestinationAlignmentElements,
                &slot_pitch_elements)
            || !checked_nonnegative_add(
                destination_slots, 1, &arena_slot_count)
            || !checked_positive_multiply(
                arena_slot_count, slot_pitch_elements,
                &unaligned_block_elements)
            || !checked_positive_round_up(
                unaligned_block_elements, kBlockAlignmentElements,
                &block_elements)
            || !checked_nonnegative_multiply(
                block_index, block_elements, &block_offset)
            || !checked_nonnegative_multiply(
                destination_slots - 1, slot_pitch_elements,
                &maximum_invocation_offset)
            || !checked_nonnegative_add(
                maximum_invocation_offset, kDestinationBaseOffsetElements,
                &maximum_invocation_offset)
            || !checked_nonnegative_add(
                block_offset, maximum_invocation_offset,
                &maximum_destination_offset)
            || !checked_nonnegative_multiply(
                destination_slots, slot_pitch_elements,
                &warmup_destination_offset)
            || !checked_nonnegative_add(
                block_offset, warmup_destination_offset,
                &warmup_destination_offset)
            || !checked_nonnegative_add(
                warmup_destination_offset, kDestinationBaseOffsetElements,
                &warmup_destination_offset)
            || !checked_nonnegative_add(
                block_offset, block_elements, &destination_end_offset)) {
        return false;
    }
    extent->tile_span_elements = tile_span_elements;
    extent->slot_pitch_elements = slot_pitch_elements;
    extent->slot_count = destination_slots;
    extent->block_elements = block_elements;
    extent->block_offset = block_offset;
    extent->maximum_invocation_offset = maximum_invocation_offset;
    extent->maximum_destination_offset = maximum_destination_offset;
    extent->warmup_destination_offset = warmup_destination_offset;
    extent->destination_end_offset = destination_end_offset;
    return true;
}

FIXPIPE_SERVICE_INLINE bool frozen_invocation_counts(
        int64_t m, int64_t n, int64_t stream_depth,
        InvocationCounts* counts) {
    if (counts == nullptr || stream_depth <= 0) {
        return false;
    }
    int64_t tile_elements = 0;
    int64_t tile_bytes = 0;
    if (!checked_positive_multiply(m, n, &tile_elements)
            || !checked_positive_multiply(
                tile_elements, static_cast<int64_t>(sizeof(float)),
                &tile_bytes)) {
        return false;
    }
    int64_t target_tiles = kTargetBytes / tile_bytes
        + (kTargetBytes % tile_bytes != 0 ? 1 : 0);
    if (m >= kLargeRowThreshold
            && target_tiles < kMinimumLargeRowInvocations) {
        target_tiles = kMinimumLargeRowInvocations;
    }
    const int64_t depth_groups = target_tiles / stream_depth
        + (target_tiles % stream_depth != 0 ? 1 : 0);
    int64_t short_count = 0;
    int64_t long_count = 0;
    if (!checked_positive_multiply(
            depth_groups, stream_depth, &short_count)
            || !checked_positive_multiply(
                short_count, kLongIntervalMultiplier, &long_count)) {
        return false;
    }
    counts->short_count = short_count;
    counts->long_count = long_count;
    return true;
}

FIXPIPE_SERVICE_INLINE bool is_frozen_invocation_count(
        int64_t m, int64_t n, int64_t stream_depth,
        int64_t invocations) {
    InvocationCounts counts{0, 0};
    return frozen_invocation_counts(m, n, stream_depth, &counts)
        && (invocations == counts.short_count
            || invocations
                == counts.short_count * kMiddleIntervalMultiplier
            || invocations == counts.long_count);
}

}  // namespace fixpipe_service_contract

#undef FIXPIPE_SERVICE_INLINE
