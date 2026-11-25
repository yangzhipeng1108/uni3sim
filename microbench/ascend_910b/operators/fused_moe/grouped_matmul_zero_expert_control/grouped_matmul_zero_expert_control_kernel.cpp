// Isolate the zero-row branch of CANN 8.5.1 GroupedMatmul::GMMProcess.
// Every AIC scans the cumulative group list, updates MNConfig, reads one
// int64 group offset, materializes the static M/K/N fields, and continues
// before MMCompute. No matrix, vector, or production operator work executes.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr uint32_t kAicCount = 24;
constexpr uint32_t kResultStrideElements = 8;

__aicore__ inline uint32_t Align16(uint32_t value) {
    return (value + 15U) & ~15U;
}

}  // namespace

extern "C" __global__ __aicore__ void grouped_matmul_zero_expert_control(
        __gm__ int64_t* group_offsets, __gm__ int64_t* output_cycles,
        __gm__ uint64_t* output_checksums, int64_t group_count,
        int64_t repeats, int64_t projection_k, int64_t projection_n,
        int64_t base_m, int64_t base_n) {
    const uint32_t block_index = AscendC::GetBlockIdx();
    if (block_index >= kAicCount) {
        return;
    }
    AscendC::GlobalTensor<int64_t> group_list;
    group_list.SetGlobalBuffer(group_offsets);

    uint64_t checksum = static_cast<uint64_t>(block_index + 1U);
    const int64_t begin = AscendC::GetSystemCycle();
    for (int64_t repeat = 0; repeat < repeats; ++repeat) {
        uint32_t m = 0;
        uint32_t k = 0;
        uint32_t n = 0;
        uint32_t single_m = 0;
        uint32_t single_n = 0;
        uint64_t weight_offset = 0;
        uint64_t n_offset = 0;
        uint64_t m_offset = 0;
        uint64_t x_offset = 0;
        uint64_t y_offset = 0;
        int64_t scale_index = -1;
        int32_t previous_offset = 0;
        for (int64_t group_index = 0; group_index < group_count;
             ++group_index) {
            weight_offset += static_cast<uint64_t>(Align16(k)) * Align16(n);
            n_offset += n;
            ++scale_index;
            m_offset += m;
            x_offset += static_cast<uint64_t>(m) * k;
            y_offset += static_cast<uint64_t>(m) * n;

            const int32_t offset = static_cast<int32_t>(
                group_list.GetValue(group_index));
            const int32_t split_value = offset - previous_offset;
            previous_offset = offset;
            m = static_cast<uint32_t>(split_value);
            k = static_cast<uint32_t>(projection_k);
            n = static_cast<uint32_t>(projection_n);
            single_m = static_cast<uint32_t>(base_m);
            single_n = static_cast<uint32_t>(base_n);
            if (split_value <= 0) {
                continue;
            }
            checksum += static_cast<uint64_t>(
                (m + single_m - 1U) / single_m
                + (n + single_n - 1U) / single_n);
        }
        checksum += weight_offset + n_offset + m_offset + x_offset
            + y_offset + static_cast<uint64_t>(scale_index + 1)
            + static_cast<uint64_t>(previous_offset);
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
