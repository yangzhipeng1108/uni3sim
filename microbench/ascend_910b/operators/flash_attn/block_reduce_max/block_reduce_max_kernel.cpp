// Minimal CANN 8.5.1 FIA online-softmax local-row-max instruction mirror.
// The timed region contains only CalcLocalRowMax's BlockReduceMax/Max/mask
// sequence and the PIPE_V barriers required by that production source path.
#include "kernel_operator.h"

#define BLOCK_REDUCE_MAX_CONTRACT_AICORE
#include "block_reduce_max_contract.h"
#undef BLOCK_REDUCE_MAX_CONTRACT_AICORE

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

using namespace block_reduce_max_contract;

__aicore__ inline void SetVecMask(int32_t len) {
    uint64_t mask = 0;
    uint64_t one = 1;
    uint64_t temp = len % kFloatVectorSize;
    for (int64_t index = 0; index < temp; ++index) {
        mask |= one << index;
    }
    if (len == 128 || len == 0) {
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    } else if (len >= kFloatVectorSize) {
        AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
    } else {
        AscendC::SetVectorMask<int8_t>(0x0, mask);
    }
}

__aicore__ inline void SetBlockReduceMask(int32_t len) {
    if (len > 8 || len < 1) {
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        return;
    }
    uint64_t sub_mask = ((uint64_t)1 << len) - 1;
    uint64_t mask_value = (sub_mask << 48) + (sub_mask << 32)
        + (sub_mask << 16) + sub_mask + (sub_mask << 56)
        + (sub_mask << 40) + (sub_mask << 24) + (sub_mask << 8);
    AscendC::SetVectorMask<int8_t>(mask_value, mask_value);
}

__aicore__ inline void RowmaxSPECTILE512(
        const AscendC::LocalTensor<float>& source,
        const AscendC::LocalTensor<float>& row_max,
        const AscendC::LocalTensor<float>& temporary,
        uint32_t row_chunk_round, uint32_t column_count,
        uint32_t column_count_round) {
    AscendC::BlockReduceMax<float, false>(
        temporary, source,
        row_chunk_round * column_count_round / kFloatVectorSize,
        0, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::BlockReduceMax<float, false>(
        temporary[kReduceUbSize], temporary,
        row_chunk_round * column_count_round
            / kFloatBlockSize / kFloatVectorSize,
        0, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::BlockReduceMax<float, false>(
        row_max, temporary[kReduceUbSize],
        row_chunk_round * column_count_round
            / kFloatVectorSize / kFloatVectorSize,
        0, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void RowmaxSPECTILE256(
        const AscendC::LocalTensor<float>& source,
        const AscendC::LocalTensor<float>& row_max,
        const AscendC::LocalTensor<float>& temporary,
        uint32_t row_chunk_round, uint32_t column_count,
        uint32_t column_count_round) {
    AscendC::BlockReduceMax<float, false>(
        temporary, source,
        row_chunk_round * column_count_round / kFloatVectorSize,
        0, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    SetVecMask(32);
    AscendC::BlockReduceMax<float, false>(
        temporary[kReduceUbSize], temporary, row_chunk_round,
        0, 1, 1, 4);
    AscendC::PipeBarrier<PIPE_V>();
    SetBlockReduceMask(4);
    AscendC::BlockReduceMax<float, false>(
        row_max, temporary[kReduceUbSize],
        CeilDiv(row_chunk_round * kFloatBlockSize, kFloatVectorSize),
        0, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
}

__aicore__ inline void RowmaxTAILTILE(
        const AscendC::LocalTensor<float>& source,
        const AscendC::LocalTensor<float>& row_max,
        const AscendC::LocalTensor<float>& temporary,
        uint32_t row_chunk_round, uint32_t column_count,
        uint32_t column_count_round) {
    if (column_count >= kFloatVectorSize) {
        AscendC::BlockReduceMax<float, false>(
            temporary, source, row_chunk_round,
            0, 1, 1, column_count_round / kFloatBlockSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceMax<float, false>(
            row_max, temporary,
            CeilDiv(row_chunk_round * kFloatBlockSize, kFloatVectorSize),
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint64_t segment = 1;
                segment < static_cast<uint64_t>(column_count)
                    / kFloatVectorSize;
                ++segment) {
            AscendC::BlockReduceMax<float, false>(
                temporary, source[segment * kFloatVectorSize],
                row_chunk_round,
                0, 1, 1, column_count_round / kFloatBlockSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceMax<float, false>(
                temporary[kReduceUbSize], temporary,
                CeilDiv(row_chunk_round * kFloatBlockSize,
                        kFloatVectorSize),
                0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            SetVecMask(row_chunk_round);
            AscendC::Max<float, false>(
                row_max, row_max, temporary[kReduceUbSize],
                (uint64_t)0, 1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }
    if (column_count % kFloatVectorSize > 0) {
        SetVecMask(column_count % kFloatVectorSize);
        AscendC::BlockReduceMax<float, false>(
            temporary,
            source[column_count / kFloatVectorSize * kFloatVectorSize],
            row_chunk_round,
            0, 1, 1, column_count_round / kFloatBlockSize);
        AscendC::PipeBarrier<PIPE_V>();
        SetBlockReduceMask(CeilDiv(
            column_count % kFloatVectorSize, kFloatBlockSize));
        if (column_count < kFloatVectorSize) {
            AscendC::BlockReduceMax<float, false>(
                row_max, temporary,
                CeilDiv(row_chunk_round * kFloatBlockSize,
                        kFloatVectorSize),
                0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::BlockReduceMax<float, false>(
                temporary[kReduceUbSize], temporary,
                CeilDiv(row_chunk_round * kFloatBlockSize,
                        kFloatVectorSize),
                0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            SetVecMask(row_chunk_round);
            AscendC::Max<float, false>(
                row_max, row_max, temporary[kReduceUbSize],
                (uint64_t)0, 1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
}

__aicore__ inline void CalcLocalRowMax(
        const AscendC::LocalTensor<float>& source,
        const AscendC::LocalTensor<float>& row_max,
        const AscendC::LocalTensor<float>& temporary,
        uint32_t row_chunk_round, uint32_t column_count,
        uint32_t column_count_round) {
    if (column_count == 512) {
        RowmaxSPECTILE512(
            source, row_max, temporary, row_chunk_round,
            column_count, column_count_round);
    } else if (column_count == 256) {
        RowmaxSPECTILE256(
            source, row_max, temporary, row_chunk_round,
            column_count, column_count_round);
    } else {
        RowmaxTAILTILE(
            source, row_max, temporary, row_chunk_round,
            column_count, column_count_round);
    }
}

}  // namespace

extern "C" __global__ __aicore__ void block_reduce_max_service(
        __gm__ int64_t* gm_cycles, __gm__ float* gm_validation,
        uint32_t row_chunk_round, uint32_t column_count,
        uint32_t column_alignment, uint32_t iterations) {
    using namespace block_reduce_max_contract;
    AscendC::GlobalTensor<int64_t> cycles;
    cycles.SetGlobalBuffer(gm_cycles);
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    if (!IsSupportedCell(
            row_chunk_round, column_count, column_alignment)
            || iterations == 0) {
        cycles.SetValue(0, -1);
        return;
    }
    const uint32_t column_count_round = RoundUp(
        column_count, column_alignment);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> source_buffer;
    pipe.InitBuffer(
        source_buffer, kMaximumSourceElements * sizeof(float));
    auto source = source_buffer.Get<float>();
    AscendC::TBuf<AscendC::TPosition::VECCALC> temporary_buffer;
    pipe.InitBuffer(
        temporary_buffer, kTemporaryElements * sizeof(float));
    auto temporary = temporary_buffer.Get<float>();
    AscendC::TBuf<AscendC::TPosition::VECOUT> row_max_buffer;
    pipe.InitBuffer(
        row_max_buffer, kRowOutputElements * sizeof(float));
    auto row_max = row_max_buffer.Get<float>();

    AscendC::Duplicate<float>(
        source, 1.0f, row_chunk_round * column_count_round);
    for (uint32_t row = 0; row < row_chunk_round; ++row) {
        for (uint32_t column = column_count;
                column < column_count_round; ++column) {
            source.SetValue(row * column_count_round + column, 2.0f);
        }
    }
    AscendC::PipeBarrier<PIPE_V>();
    CalcLocalRowMax(
        source, row_max, temporary, row_chunk_round,
        column_count, column_count_round);

    const int64_t begin = AscendC::GetSystemCycle();
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
        CalcLocalRowMax(
            source, row_max, temporary, row_chunk_round,
            column_count, column_count_round);
    }
    const int64_t end = AscendC::GetSystemCycle();
    cycles.SetValue(0, end - begin);

    AscendC::GlobalTensor<float> validation;
    validation.SetGlobalBuffer(gm_validation);
    for (uint32_t row = 0; row < row_chunk_round; ++row) {
        validation.SetValue(row, row_max.GetValue(row));
    }
}
