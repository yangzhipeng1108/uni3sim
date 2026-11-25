// Isolate the controller-to-first-source-tile dependency in CANN 8.5.1
// GroupedMatmul::GMMProcess::Process.  This AIC-only probe retains the
// source-defined cumulative group-list loop, MN configuration, AIC ring, and
// diagonal block index calculation for the uniform BF16 CUMSUM program.  It
// records the point immediately before the first MMCompute call that each AIC
// would make, but never enters MMCompute or touches a matrix operand.
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

__aicore__ inline uint32_t GreatestCommonDivisor(uint32_t left, uint32_t right) {
    while (right != 0U) {
        const uint32_t remainder = left % right;
        left = right;
        right = remainder;
    }
    return left;
}

__aicore__ inline uint32_t LeastCommonMultiple(uint32_t left, uint32_t right) {
    return left / GreatestCommonDivisor(left, right) * right;
}

// This is the source's MNBlockIdxCompute control path, retained because its
// result determines the first source tile's M/N offset before MMCompute.
__aicore__ inline void MNBlockIdxCompute(
        uint32_t block_dim_m, uint32_t block_dim_n, uint32_t cur_block,
        uint32_t count, uint32_t threshold_m_dim_n, uint32_t* m_index,
        uint32_t* n_index) {
    if (block_dim_m <= kThresholdDimM || kThresholdDimM == 1U) {
        *m_index = (cur_block - count) / block_dim_n;
        *n_index = (cur_block - count) % block_dim_n;
        return;
    }
    const uint32_t relative_block = cur_block - count;
    const uint32_t full_extent = (
        block_dim_m * block_dim_n / threshold_m_dim_n * threshold_m_dim_n);
    const uint32_t current_threshold_m = relative_block >= full_extent
        ? block_dim_m % kThresholdBlockNum : kThresholdBlockNum;
    const uint32_t current_threshold_mn = current_threshold_m * kThresholdBlockNum;
    const uint32_t local_n_extent = (
        current_threshold_m * block_dim_n / current_threshold_mn
        * current_threshold_mn);
    const uint32_t current_threshold_n = (
        relative_block % threshold_m_dim_n >= local_n_extent
        ? block_dim_n % kThresholdBlockNum : kThresholdBlockNum);
    const uint32_t local_relative = (
        relative_block % threshold_m_dim_n % current_threshold_mn);
    *m_index = (
        local_relative % current_threshold_m
        + relative_block / threshold_m_dim_n * kThresholdBlockNum);
    *n_index = (
        (local_relative + local_relative / LeastCommonMultiple(
            current_threshold_m, current_threshold_n)) % current_threshold_n
        + relative_block % threshold_m_dim_n / current_threshold_mn
        * kThresholdBlockNum);
}

}  // namespace

extern "C" __global__ __aicore__ void grouped_matmul_first_issue_control(
        __gm__ int64_t* group_offsets, __gm__ int64_t* first_issue_cycles,
        __gm__ int64_t* completion_cycles, __gm__ uint64_t* output_checksums,
        int64_t group_count, int64_t repeats, int64_t projection_k,
        int64_t projection_n, int64_t base_m, int64_t base_n) {
    const uint32_t block_index = AscendC::GetBlockIdx();
    if (block_index >= kAicCount) {
        return;
    }
    AscendC::GlobalTensor<int64_t> group_list;
    group_list.SetGlobalBuffer(group_offsets);
    uint64_t checksum = static_cast<uint64_t>(block_index + 1U);
    int64_t total_first_issue_cycles = 0;
    int64_t total_completion_cycles = 0;
    bool issued_any_tile = true;

    for (int64_t repeat = 0; repeat < repeats; ++repeat) {
        const int64_t begin = AscendC::GetSystemCycle();
        // GMMProcess::Process fences the prior source task before scanning
        // the group list. The probe has no prior matrix task, but retaining
        // this source operation prevents silently measuring a different
        // controller sequence.
        AscendC::WaitPreTaskEnd();
        int64_t first_issue = -1;
        uint32_t previous_m = 0;
        uint32_t previous_k = 0;
        uint32_t previous_n = 0;
        uint64_t weight_offset = 0;
        uint64_t n_offset = 0;
        uint64_t m_offset = 0;
        uint64_t x_offset = 0;
        uint64_t y_offset = 0;
        int64_t scale_index = -1;
        int32_t previous_offset = 0;
        uint32_t count = 0;

        for (int64_t group_index = 0; group_index < group_count;
             ++group_index) {
            // GMMProcess::UpdateMnConfig for the BF16-NZ static program.
            weight_offset += static_cast<uint64_t>(
                (previous_k + 15U) & ~15U) * ((previous_n + 15U) & ~15U);
            n_offset += previous_n;
            ++scale_index;
            m_offset += previous_m;
            x_offset += static_cast<uint64_t>(previous_m) * previous_k;
            y_offset += static_cast<uint64_t>(previous_m) * previous_n;

            const int32_t offset = static_cast<int32_t>(
                group_list.GetValue(group_index));
            const int32_t split_value = offset - previous_offset;
            previous_offset = offset;
            previous_m = static_cast<uint32_t>(split_value);
            previous_k = static_cast<uint32_t>(projection_k);
            previous_n = static_cast<uint32_t>(projection_n);
            if (split_value <= 0) {
                continue;
            }

            const uint32_t block_dim_m = CeilDiv(previous_m, base_m);
            const uint32_t block_dim_n = CeilDiv(previous_n, base_n);
            const uint32_t current_count = count + block_dim_m * block_dim_n;
            uint32_t current_block = block_index >= count
                ? block_index : block_index + kAicCount;
            const uint32_t threshold_m_dim_n = kThresholdBlockNum * block_dim_n;
            while (current_block < current_count) {
                uint32_t m_index = 0;
                uint32_t n_index = 0;
                MNBlockIdxCompute(
                    block_dim_m, block_dim_n, current_block, count,
                    threshold_m_dim_n, &m_index, &n_index);
                // This is immediately before MMCompute's per-tile source
                // setup.  No tensor address, transfer, Cube, or output work
                // is executed by this probe.
                if (first_issue < 0) {
                    first_issue = AscendC::GetSystemCycle() - begin;
                }
                checksum += static_cast<uint64_t>(m_index + 1U)
                    * static_cast<uint64_t>(n_index + 1U)
                    * static_cast<uint64_t>(group_index + 1U);
                current_block += kAicCount;
            }
            count = current_count % kAicCount;
        }
        // The source closes the task after PostCompute. This probe has no
        // compute task to drain, so preserve the task boundary without adding
        // matrix, MTE, Cube, or FixPipe work.
        AscendC::SetNextTaskStart();
        const int64_t completion = AscendC::GetSystemCycle() - begin;
        if (first_issue < 0) {
            issued_any_tile = false;
            first_issue = 0;
        }
        total_first_issue_cycles += first_issue;
        total_completion_cycles += completion;
        checksum += weight_offset + n_offset + m_offset + x_offset + y_offset
            + static_cast<uint64_t>(scale_index + 1) + count
            + static_cast<uint64_t>(previous_offset);
    }

    const uint32_t slot = block_index * kResultStrideElements;
    AscendC::GlobalTensor<int64_t> first_slot;
    first_slot.SetGlobalBuffer(first_issue_cycles + slot);
    first_slot.SetValue(0, total_first_issue_cycles / repeats);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(first_slot);
    AscendC::GlobalTensor<int64_t> completion_slot;
    completion_slot.SetGlobalBuffer(completion_cycles + slot);
    completion_slot.SetValue(0, total_completion_cycles / repeats);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(completion_slot);
    AscendC::GlobalTensor<uint64_t> checksum_slot;
    checksum_slot.SetGlobalBuffer(output_checksums + slot);
    checksum_slot.SetValue(0, issued_any_tile ? checksum : 0);
    AscendC::DataCacheCleanAndInvalid<
        uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(checksum_slot);
}
