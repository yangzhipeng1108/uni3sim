// Isolate CANN 8.5.1 GroupedMatmul's nonempty integer prelude.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr uint32_t kAicCount = 24;
constexpr uint32_t kResultStrideElements = 8;
constexpr uint32_t kThresholdBlockNum = 8;
constexpr uint32_t kThresholdDimM = 5;

__aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1U) / divisor;
}

__aicore__ inline uint32_t AlignDown(uint32_t value, uint32_t divisor) {
    return value / divisor * divisor;
}

__aicore__ inline uint32_t GreatestCommonDivisor(
        uint32_t left, uint32_t right) {
    while (right != 0U) {
        const uint32_t remainder = left % right;
        left = right;
        right = remainder;
    }
    return left;
}

__aicore__ inline uint32_t LeastCommonMultiple(
        uint32_t left, uint32_t right) {
    return left / GreatestCommonDivisor(left, right) * right;
}

__aicore__ inline void MNBlockIdxCompute(
        uint32_t block_dim_m, uint32_t block_dim_n, uint32_t relative_block,
        uint32_t* m_index, uint32_t* n_index) {
    if (block_dim_m <= kThresholdDimM) {
        *m_index = relative_block / block_dim_n;
        *n_index = relative_block % block_dim_n;
        return;
    }
    const uint32_t threshold_m_dim_n = kThresholdBlockNum * block_dim_n;
    const uint32_t full_extent = AlignDown(
        block_dim_m * block_dim_n, threshold_m_dim_n);
    const uint32_t current_threshold_m = relative_block >= full_extent
        ? block_dim_m % kThresholdBlockNum : kThresholdBlockNum;
    const uint32_t current_threshold_m_threshold_n =
        current_threshold_m * kThresholdBlockNum;
    const uint32_t local_n_extent = AlignDown(
        current_threshold_m * block_dim_n,
        current_threshold_m_threshold_n);
    const uint32_t current_threshold_n =
        relative_block % threshold_m_dim_n >= local_n_extent
        ? block_dim_n % kThresholdBlockNum : kThresholdBlockNum;
    const uint32_t local_relative =
        relative_block % threshold_m_dim_n % current_threshold_m_threshold_n;
    *m_index = local_relative % current_threshold_m
        + relative_block / threshold_m_dim_n * kThresholdBlockNum;
    *n_index = (local_relative + local_relative / LeastCommonMultiple(
        current_threshold_m, current_threshold_n)) % current_threshold_n
        + relative_block % threshold_m_dim_n
        / current_threshold_m_threshold_n * kThresholdBlockNum;
}

}  // namespace

extern "C" __global__ __aicore__ void grouped_matmul_prelude_control(
        __gm__ int64_t* output_cycles, __gm__ uint64_t* output_checksums,
        int64_t tile_count, int64_t k_rounds, int64_t repeats) {
    const uint32_t block_index = AscendC::GetBlockIdx();
    if (block_index >= kAicCount) {
        return;
    }

    // The frozen short-source CANN program has base-M16/N256/K64,
    // eight A1/B1 K groups and step-K four for both operands.
    constexpr uint32_t kLogicalM = 16;
    constexpr uint32_t kLogicalN = 256;
    constexpr uint32_t kBaseM = 16;
    constexpr uint32_t kBaseN = 256;
    constexpr uint32_t kBaseK = 64;
    constexpr uint32_t kStepKA = 4;
    constexpr uint32_t kStepKB = 4;

    volatile uint64_t checksum = static_cast<uint64_t>(block_index + 1U);
    const int64_t begin = AscendC::GetSystemCycle();
    for (int64_t repeat = 0; repeat < repeats; ++repeat) {
        for (int64_t tile = 0; tile < tile_count; ++tile) {
            const uint32_t block_dim_m = CeilDiv(kLogicalM, kBaseM);
            const uint32_t block_dim_n = CeilDiv(kLogicalN, kBaseN);
            const uint32_t relative_block = static_cast<uint32_t>(tile)
                % (block_dim_m * block_dim_n);
            uint32_t m_index = 0;
            uint32_t n_index = 0;
            MNBlockIdxCompute(
                block_dim_m, block_dim_n, relative_block, &m_index, &n_index);

            const uint32_t tail_n = n_index * kBaseN;
            const uint32_t current_n = n_index < block_dim_n - 1U
                ? kBaseN : kLogicalN - tail_n;
            const uint32_t current_m = m_index < block_dim_m - 1U
                ? kBaseM : kLogicalM - m_index * kBaseM;
            const uint64_t x_offset = static_cast<uint64_t>(m_index)
                * kBaseM * static_cast<uint64_t>(k_rounds) * kBaseK;
            const uint64_t w_offset = static_cast<uint64_t>(tail_n) * kBaseK;
            const uint64_t y_offset = static_cast<uint64_t>(m_index)
                * kBaseM * kLogicalN + tail_n;

            uint32_t a_group_index = 0;
            uint32_t b_group_index = 0;
            uint32_t a_l1_offset = 0;
            uint32_t b_l1_offset = 0;
            bool initialize_l0c = true;
            for (int64_t k_index = 0; k_index < k_rounds; ++k_index) {
                if (k_index % kStepKA == 0) {
                    ++a_group_index;
                    a_l1_offset = 0;
                }
                if (k_index % kStepKB == 0) {
                    ++b_group_index;
                    b_l1_offset = 0;
                }
                const uint32_t l0a_slot = static_cast<uint32_t>(k_index) & 1U;
                const uint32_t l0b_slot = static_cast<uint32_t>(k_index) & 1U;
                const bool final_round = k_index + 1 == k_rounds;
                checksum += static_cast<uint64_t>(current_m + current_n + kBaseK
                    + a_group_index + b_group_index + a_l1_offset
                    + b_l1_offset + l0a_slot + l0b_slot
                    + (initialize_l0c ? 1U : 0U) + (final_round ? 1U : 0U));
                a_l1_offset += kBaseK;
                b_l1_offset += kBaseK;
                initialize_l0c = false;
            }
            checksum += x_offset + w_offset + y_offset;
        }
    }
    const int64_t end = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<int64_t> cycle_slot;
    cycle_slot.SetGlobalBuffer(
        output_cycles + block_index * kResultStrideElements);
    cycle_slot.SetValue(0, end - begin);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(cycle_slot);
    AscendC::GlobalTensor<uint64_t> checksum_slot;
    checksum_slot.SetGlobalBuffer(
        output_checksums + block_index * kResultStrideElements);
    checksum_slot.SetValue(0, checksum);
    AscendC::DataCacheCleanAndInvalid<
        uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(checksum_slot);
}
