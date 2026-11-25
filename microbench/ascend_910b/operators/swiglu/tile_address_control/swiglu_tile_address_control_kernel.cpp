// Isolate CANN 8.5.1 SwiGLU's per-tile address and branch selection control.
//
// The timed mode 0 executes only the source's uint64 quotient/remainder
// pair. Mode 1 executes that same pair followed by CaclOneTileOffsetParam's
// five multiply/three add DAG and the exact four-way curCalLen/copy-param
// selection from glu_tiling_kernel.hpp. No TPipe, queue, CopyIn, CopyOut,
// UB allocation, vector operation, or production SwiGLU call is present.
// Subtracting mode 0 from mode 1 therefore cannot re-price Forward's existing
// source-counted CANN uint64 div/rem mechanism.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr uint64_t kOutputValuesPerWorker = 8;
__aicore__ inline void BranchDomain(
        uint64_t branch, uint64_t column_tile_count,
        uint64_t base_column_tile_count, uint64_t& start_tile) {
    // branch 0/1 use base rows; branch 2/3 begin at the first tail row.
    // Both dimensions remain runtime kernel parameters in the source call.
    if (branch > 3 || column_tile_count == 0
            || base_column_tile_count == 0) {
        start_tile = 0;
        return;
    }
    const uint64_t tail_column_offset = (branch & 1U) == 0
        ? 0 : base_column_tile_count;
    if (branch < 2) {
        // ``tile_idx`` advances one row per repetition. Keep all repeats in
        // the base-row source path while preserving dynamic row arithmetic.
        start_tile = tail_column_offset;
    } else {
        start_tile = column_tile_count + tail_column_offset;
    }
}

__aicore__ inline uint64_t DivRemOnly(
        uint64_t tile_index, uint64_t column_tile_count) {
    const uint64_t row_tile_index = tile_index / column_tile_count;
    const uint64_t column_tile_index = tile_index % column_tile_count;
    return row_tile_index + column_tile_index;
}

__aicore__ inline uint64_t AddressAndSelection(
        uint64_t tile_index, uint64_t column_tile_count,
        uint64_t base_row_length, uint64_t combined_column_length,
        uint64_t base_column_length, uint64_t column_length,
        uint64_t base_row_tile_count, uint64_t base_column_tile_count) {
    // This is the arithmetic and control body of CaclOneTileParam after its
    // existing rowTileIdx/colTileIdx div/rem. Keep the source's operand and
    // store ordering intact: CaclOneTileOffsetParam first, selection second.
    const uint64_t row_tile_index = tile_index / column_tile_count;
    const uint64_t column_tile_index = tile_index % column_tile_count;
    const uint64_t gm_row_offset = row_tile_index * base_row_length;
    const uint64_t split_offset_1 = gm_row_offset * combined_column_length
        + column_tile_index * base_column_length;
    const uint64_t split_offset_2 = split_offset_1 + column_length;
    const uint64_t independent_offset = gm_row_offset * column_length
        + column_tile_index * base_column_length;

    uint64_t current_calculation_length = 0;
    uint64_t current_copy_parameter_tag = 0;
    if (row_tile_index < base_row_tile_count) {
        if (column_tile_index < base_column_tile_count) {
            current_calculation_length = 0x101;
            current_copy_parameter_tag = 0x11;
        } else {
            current_calculation_length = 0x102;
            current_copy_parameter_tag = 0x12;
        }
    } else {
        if (column_tile_index < base_column_tile_count) {
            current_calculation_length = 0x201;
            current_copy_parameter_tag = 0x21;
        } else {
            current_calculation_length = 0x202;
            current_copy_parameter_tag = 0x22;
        }
    }
    return gm_row_offset + split_offset_1 + split_offset_2
        + independent_offset + current_calculation_length
        + current_copy_parameter_tag;
}

}  // namespace

extern "C" __global__ __aicore__ void swiglu_tile_address_control(
        __gm__ uint64_t* output, int64_t branch, int64_t mode,
        int64_t repetitions, int64_t column_tile_count,
        int64_t base_row_length, int64_t combined_column_length,
        int64_t base_column_length, int64_t column_length,
        int64_t base_row_tile_count, int64_t base_column_tile_count) {
    const uint64_t worker = AscendC::GetBlockIdx();
    const uint64_t worker_count = AscendC::GetBlockNum();
    if (worker_count == 0 || worker_count > 48 || branch < 0 || branch > 3
            || mode < 0 || mode > 1 || repetitions <= 0
            || column_tile_count <= 0 || base_row_length <= 0
            || combined_column_length <= 0 || base_column_length <= 0
            || column_length <= 0 || base_row_tile_count <= 0
            || base_column_tile_count <= 0) {
        AscendC::GlobalTensor<uint64_t> invalid;
        invalid.SetGlobalBuffer(output + worker * kOutputValuesPerWorker);
        invalid.SetValue(0, 0);
        invalid.SetValue(1, 0);
        return;
    }

    const uint64_t repeat_count = static_cast<uint64_t>(repetitions);
    // CANN's fields are populated from a runtime tiling payload.  Keep these
    // values as launch-time ABI inputs instead of compile-time constants so
    // the probe measures the same dynamic scalar program rather than a
    // strength-reduced special case.
    const uint64_t runtime_column_tile_count =
        static_cast<uint64_t>(column_tile_count);
    const uint64_t runtime_base_row_length =
        static_cast<uint64_t>(base_row_length);
    const uint64_t runtime_combined_column_length =
        static_cast<uint64_t>(combined_column_length);
    const uint64_t runtime_base_column_length =
        static_cast<uint64_t>(base_column_length);
    const uint64_t runtime_column_length =
        static_cast<uint64_t>(column_length);
    const uint64_t runtime_base_row_tile_count =
        static_cast<uint64_t>(base_row_tile_count);
    const uint64_t runtime_base_column_tile_count =
        static_cast<uint64_t>(base_column_tile_count);
    uint64_t start_tile = 0;
    BranchDomain(static_cast<uint64_t>(branch), runtime_column_tile_count,
                 runtime_base_column_tile_count, start_tile);
    const uint64_t tile_index_base = start_tile + worker;

    uint64_t checksum = 0;
    const uint64_t start_cycle = AscendC::GetSystemCycle();
#pragma unroll 1
    for (uint64_t repeat = 0; repeat < repeat_count; ++repeat) {
        // Advancing by colTileNum changes the source row index while retaining
        // the selected base/tail column class for every worker in this cell.
        const uint64_t tile_index = tile_index_base
            + repeat * runtime_column_tile_count;
        if (mode == 0) {
            checksum += DivRemOnly(tile_index, runtime_column_tile_count);
        } else {
            checksum += AddressAndSelection(
                tile_index, runtime_column_tile_count,
                runtime_base_row_length, runtime_combined_column_length,
                runtime_base_column_length, runtime_column_length,
                runtime_base_row_tile_count,
                runtime_base_column_tile_count);
        }
    }
    const uint64_t end_cycle = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<uint64_t> result;
    result.SetGlobalBuffer(output + worker * kOutputValuesPerWorker);
    result.SetValue(0, end_cycle - start_cycle);
    result.SetValue(1, checksum);
    result.SetValue(2, static_cast<uint64_t>(branch));
    result.SetValue(3, static_cast<uint64_t>(mode));
    result.SetValue(4, repeat_count);
    result.SetValue(5, worker_count);
    AscendC::DataCacheCleanAndInvalid<
        uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}
