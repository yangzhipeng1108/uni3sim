// Independent Ascend 910B AIV Vector-pipe opcode throughput benchmark.
//
// The timed region is UB-resident and contains only a runtime-selected stream
// of FP32 Exp, Div, or Sqrt intrinsics followed by one PIPE_V completion
// barrier.  It
// has no production operator, reduction, MTE2/MTE3 transfer, or GM traffic.
// Two destinations alternate to avoid an artificial immediate WAW dependency.
// Runtime scalar inputs prevent compile-time constant folding.  One result is
// copied to GM only after timing so the host can fail loudly on bad execution.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kExp = 0;
constexpr int64_t kDiv = 1;
constexpr int64_t kExpDiv = 2;
constexpr int64_t kSqrt = 3;
constexpr int64_t kSqrtDiv = 4;
constexpr int64_t kExpSerialized = 5;
constexpr int64_t kDivSerialized = 6;
constexpr int64_t kCastBf16ToFp32Serialized = 7;
constexpr int64_t kCastFp32ToBf16Serialized = 8;
constexpr int64_t kMulsSerialized = 9;
constexpr int64_t kAddsSerialized = 10;
constexpr int64_t kMulSerialized = 11;
constexpr int64_t kNormSqrtSerialized = 12;
constexpr int64_t kNormDivSerialized = 13;
constexpr int64_t kNormSqrtDuplicateDivSerialized = 14;
constexpr int64_t kTwoTbufMaskPrologue = 15;
constexpr int64_t kRmsNormLocalScalarDependency = 16;
constexpr int64_t kLocalScalarFillSv = 17;
constexpr int64_t kLocalUbCopyContiguous = 18;
constexpr int64_t kLocalUbCopyStride64 = 19;
constexpr int64_t kLocalUbCopyStride128 = 20;
constexpr int64_t kSingleTbufInit = 21;
constexpr int64_t kScalarU32DivRemDependency = 22;
constexpr int64_t kScalarU32MulAddDependency = 23;
constexpr int64_t kScalarU64MulDependency = 24;
constexpr int64_t kSwigluDoubleBufferInit = 25;
constexpr int64_t kSwigluSingleBufferInit = 26;
constexpr int64_t kRmsNormHalfIntervalReduce = 27;
constexpr int64_t kSwigluDoubleBufferLifecycle = 28;
constexpr int64_t kSwigluSingleBufferLifecycle = 29;
constexpr int64_t kSwigluDoubleBufferQueueControl = 30;
constexpr int64_t kSwigluSingleBufferQueueControl = 31;
constexpr int64_t kScalarU64DivRemDependency = 32;
constexpr int64_t kRmsNormFourQueueThreeTbufLifecycle = 33;
constexpr int64_t kRmsNormFourQueueControlChain = 34;
constexpr int64_t kRmsNormFourQueueThreeTbufReady = 35;
constexpr int64_t kAddRmsNormLocalScalarHandoff = 36;
constexpr int64_t kAddRmsNormReduceSumFp32 = 37;
constexpr int64_t kVectorToMte2Fence = 38;
constexpr int64_t kAddRmsNormSplitDLocalScalarHandoff = 39;
constexpr int64_t kAddRmsNormSplitDFourQueueFourTbufLifecycle = 40;
constexpr int64_t kAddRmsNormSplitDBlockReduceSumFp32 = 41;
constexpr int64_t kMergeNReduceSumMultiN = 42;
constexpr int64_t kMergeNBroadcastGamma = 43;
constexpr int64_t kMergeNBroadcastRstd = 44;
constexpr int64_t kMergeNPerformanceRstdBrcb = 45;
constexpr int64_t kMergeNPerformanceMulRowRstd = 46;
constexpr int64_t kMergeNPerformanceMulRowGamma = 47;
constexpr int64_t kCompareFp32Gt = 48;
constexpr int64_t kCompareFp32Eq = 49;
constexpr int64_t kCompareFp32Ne = 50;
constexpr int64_t kSelectFp32 = 51;
constexpr int64_t kCastInt32ToInt64 = 52;
// The CANN 8.5.1 RmsNorm MERGE_N (tiling key 2) row-group primitives are
// parameterized by the row count of one group; their column extent is the
// host-tiler ``numColAlign``.  Every production shape that reaches key 2 on
// this SoC has numColAlign == 128, and rms_norm_tiling.cpp:543 then fixes
// rowFactor = ubSize / (128 * 16 + 260) = 84 rows per group.  ``element_count``
// carries the row count for these opcodes.
constexpr uint32_t kMergeNColumns = 128;
constexpr uint32_t kMergeNMaximumRows = 84;
constexpr uint32_t kMergeNElementsPerRepeat = 64;
constexpr uint32_t kMergeNElementsPerBlock = 8;
// The version-matched key-27 SwiGlu payload uses one 6,144-element FP32
// Exp/Div tile.  Four FP32 UB buffers occupy 98,304 bytes at this bound,
// within the 192 KiB physical UB capacity.
constexpr int64_t kMaximumElements = 13312;
// GlobalTensor::SetValue uses the scalar GM data-cache path.  Give every AIV
// its own 64-byte cache line so concurrent per-block cycle writes cannot
// overwrite neighbouring blocks during cache writeback.
constexpr int64_t kCycleValuesPerBlock = 8;
constexpr int64_t kValidationValuesPerBlock = 16;
constexpr uint32_t kLocalCopySourceBytes = 32768;
constexpr uint32_t kLocalCopyDestinationBytes = 131072;
constexpr uint32_t kLocalCopyDestinationStrideBytes = 8192;
constexpr int64_t kLocalCopyUnroll = 16;

// No claim about 910B lowering is made by this API-level preparation. The
// target compiler must accept this exact typed Cast before the experiment
// can run. Its interval measures vector invocations, never a whole GM Cast.
// The Level-2 API uses mask_count(elements) and vconv_s322s64 count mode;
// N/64 must not be interpreted as its ISA repeat count. The N%64 bound below
// only defines this experiment's finite tile domain.
__aicore__ inline void MeasureInt32Int64Cast(
        AscendC::GlobalTensor<int64_t>& cycles, int64_t cycle_index,
        __gm__ float* gm_validation, int64_t block_index,
        uint32_t elements, int64_t iterations) {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> source_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> output0_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> output1_buffer;
    pipe.InitBuffer(source_buffer, elements * sizeof(int32_t));
    pipe.InitBuffer(output0_buffer, elements * sizeof(int64_t));
    pipe.InitBuffer(output1_buffer, elements * sizeof(int64_t));
    auto source = source_buffer.Get<int32_t>();
    auto output0 = output0_buffer.Get<int64_t>();
    auto output1 = output1_buffer.Get<int64_t>();
    const int32_t pattern[16] = {
        (-2147483647 - 1), -2147483647, -65537, -65536, -1, 0, 1,
        65535, 65536, 65537, 2147483646, 2147483647, -2, 2,
        -1073741824, 1073741824};
    for (uint32_t element = 0; element < elements; ++element) {
        source.SetValue(element, pattern[element % 16]);
        output0.SetValue(element, int64_t{0x123456789abcdefLL});
        output1.SetValue(element, int64_t{0x123456789abcdefLL});
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t begin = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; iteration += 2) {
        AscendC::Cast(output0, source, AscendC::RoundMode::CAST_NONE, elements);
        AscendC::Cast(output1, source, AscendC::RoundMode::CAST_NONE, elements);
    }
    AscendC::PipeBarrier<PIPE_V>();
    const int64_t elapsed = AscendC::GetSystemCycle() - begin;
    AscendC::PipeBarrier<PIPE_ALL>();
    int64_t mismatches = 0;
    for (uint32_t element = 0; element < elements; ++element) {
        const int64_t expected = static_cast<int64_t>(pattern[element % 16]);
        if (output0.GetValue(element) != expected) { ++mismatches; }
        if (output1.GetValue(element) != expected) { ++mismatches; }
    }
    cycles.SetValue(cycle_index, elapsed);
    cycles.SetValue(cycle_index + 1, mismatches);
    // Keep exact INT64 samples in the integer result channel. Do not
    // truncate them to FP32 or mistake a low-word check for signed widening.
    const uint32_t sample_lanes[6] = {0, 4, 5, 6, 10, 11};
    for (uint32_t sample = 0; sample < 6; ++sample) {
        cycles.SetValue(cycle_index + 2 + sample,
                         output1.GetValue(sample_lanes[sample]));
    }
    AscendC::GlobalTensor<float> validation;
    validation.SetGlobalBuffer(gm_validation);
    validation.SetValue(block_index * kValidationValuesPerBlock, 1.0f);
}

template <AscendC::CMPMODE mode>
__aicore__ inline void RunFp32Compare(
        const AscendC::LocalTensor<uint8_t>& mask0,
        const AscendC::LocalTensor<uint8_t>& mask1,
        const AscendC::LocalTensor<float>& input_a,
        const AscendC::LocalTensor<float>& input_b,
        uint8_t repeats, int64_t iterations) {
    const AscendC::BinaryRepeatParams params{1, 1, 1, 8, 8, 8};
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; iteration += 2) {
        AscendC::Compare(mask0, input_a, input_b, mode,
                         uint64_t{64}, repeats, params);
        AscendC::Compare(mask1, input_a, input_b, mode,
                         uint64_t{64}, repeats, params);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

// An independent UB-only predicate/select stream.  This is deliberately not
// ArgMax: no reduction, index comparison, shuffle, GM payload or model shape.
// The version-matched ApplyTopKTopPCustom uses this exact uint8-mask,
// Compare/Select repeat API.  A source call is a vector invocation with
// elements/64 repeats, not one scalar comparison or one unspecified ISA op.
__aicore__ inline void MeasureFp32PredicateSelect(
        AscendC::GlobalTensor<int64_t>& cycles, int64_t cycle_index,
        __gm__ float* gm_validation, int64_t block_index,
        int64_t opcode, uint32_t elements, int64_t iterations) {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_a_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_b_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> true_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> false_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> output0_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> output1_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mask0_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mask1_buffer;
    const uint32_t payload_bytes = elements * sizeof(float);
    const uint32_t mask_bytes = (elements / 8 + 31) / 32 * 32;
    pipe.InitBuffer(input_a_buffer, payload_bytes);
    pipe.InitBuffer(input_b_buffer, payload_bytes);
    pipe.InitBuffer(true_buffer, payload_bytes);
    pipe.InitBuffer(false_buffer, payload_bytes);
    pipe.InitBuffer(output0_buffer, payload_bytes);
    pipe.InitBuffer(output1_buffer, payload_bytes);
    pipe.InitBuffer(mask0_buffer, mask_bytes);
    pipe.InitBuffer(mask1_buffer, mask_bytes);
    auto input_a = input_a_buffer.Get<float>();
    auto input_b = input_b_buffer.Get<float>();
    auto input_a_bits = input_a_buffer.Get<uint32_t>();
    auto input_b_bits = input_b_buffer.Get<uint32_t>();
    auto true_values = true_buffer.Get<float>();
    auto false_values = false_buffer.Get<float>();
    auto output0 = output0_buffer.Get<float>();
    auto output1 = output1_buffer.Get<float>();
    auto mask0 = mask0_buffer.Get<uint8_t>();
    auto mask1 = mask1_buffer.Get<uint8_t>();
    // Numeric greater/less/tie, signed zero, infinities and quiet NaNs in
    // either/both operands.  Write exact IEEE bits; never numerically cast
    // INT64 indices to float and pretend to measure index ordering.
    const uint32_t lhs_bits[16] = {
        0x3f800000U, 0xbf800000U, 0x00000000U, 0x80000000U,
        0x7f800000U, 0xff800000U, 0x7fc00001U, 0x3f800000U,
        0x7fc00001U, 0x40000000U, 0xc0000000U, 0x7f800000U,
        0xff800000U, 0x3f800000U, 0xbf800000U, 0x00000000U};
    const uint32_t rhs_bits[16] = {
        0xbf800000U, 0x3f800000U, 0x80000000U, 0x00000000U,
        0x7f800000U, 0xff800000U, 0x3f800000U, 0x7fc00001U,
        0x7fc00001U, 0x3f800000U, 0xbf800000U, 0x3f800000U,
        0x3f800000U, 0x3f800000U, 0xbf800000U, 0x3f800000U};
    for (uint32_t element = 0; element < elements; ++element) {
        const uint32_t lane = element % 16;
        input_a_bits.SetValue(element, lhs_bits[lane]);
        input_b_bits.SetValue(element, rhs_bits[lane]);
        const int32_t magnitude = 10 + static_cast<int32_t>(lane);
        true_values.SetValue(element, static_cast<float>(magnitude));
        false_values.SetValue(element, -static_cast<float>(magnitude));
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const uint8_t repeats = static_cast<uint8_t>(elements / 64);
    const AscendC::BinaryRepeatParams params{1, 1, 1, 8, 8, 8};
    if (opcode == kSelectFp32) {
        // Mask setup is outside the Select-only timing interval.
        AscendC::Compare(mask1, input_a, input_b, AscendC::CMPMODE::GT,
                         uint64_t{64}, repeats, params);
        AscendC::PipeBarrier<PIPE_V>();
    }
    const int64_t begin = AscendC::GetSystemCycle();
    if (opcode == kSelectFp32) {
        #pragma unroll 1
        for (int64_t iteration = 0; iteration < iterations; iteration += 2) {
            AscendC::Select(output0, mask1, true_values, false_values,
                AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                uint64_t{64}, repeats, params);
            AscendC::Select(output1, mask1, true_values, false_values,
                AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                uint64_t{64}, repeats, params);
        }
    } else if (opcode == kCompareFp32Gt) {
        RunFp32Compare<AscendC::CMPMODE::GT>(mask0, mask1,
            input_a, input_b, repeats, iterations);
    } else if (opcode == kCompareFp32Eq) {
        RunFp32Compare<AscendC::CMPMODE::EQ>(mask0, mask1,
            input_a, input_b, repeats, iterations);
    } else {
        RunFp32Compare<AscendC::CMPMODE::NE>(mask0, mask1,
            input_a, input_b, repeats, iterations);
    }
    AscendC::PipeBarrier<PIPE_V>();
    const int64_t elapsed = AscendC::GetSystemCycle() - begin;
    if (opcode != kSelectFp32) {
        // Decode both mask destinations after timing, for actual result
        // validation. This Select must never be charged as Compare service.
        AscendC::Select(output0, mask0, true_values, false_values,
            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
            uint64_t{64}, repeats, params);
        AscendC::Select(output1, mask1, true_values, false_values,
            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
            uint64_t{64}, repeats, params);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const bool gt[16] = {true, false, false, false, false, false, false,
        false, false, true, false, true, false, false, false, false};
    const bool eq[16] = {false, false, true, true, true, true, false,
        false, false, false, false, false, false, true, true, false};
    int64_t mismatches = 0;
    for (uint32_t element = 0; element < elements; ++element) {
        const uint32_t lane = element % 16;
        // Independent per-lane device diagnostic observes false for this
        // exact positive-qNaN payload on either/both operands (lanes 6–8).
        // Validate ordered inequality, not C++ unordered != or isnan.
        // Signaling NaNs and other NaN payloads are outside this experiment.
        const bool ordered_ne = lane != 6 && lane != 7 && lane != 8
            && !eq[lane];
        const bool predicate = opcode == kCompareFp32Eq ? eq[lane]
            : (opcode == kCompareFp32Ne ? ordered_ne : gt[lane]);
        const float magnitude = static_cast<float>(
            10 + static_cast<int32_t>(lane));
        const float expected = predicate ? magnitude : -magnitude;
        if (output0.GetValue(element) != expected) { ++mismatches; }
        if (output1.GetValue(element) != expected) { ++mismatches; }
    }
    cycles.SetValue(cycle_index, elapsed);
    cycles.SetValue(cycle_index + 1, mismatches);
    AscendC::GlobalTensor<float> validation;
    validation.SetGlobalBuffer(gm_validation);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        validation.SetValue(block_index * kValidationValuesPerBlock + lane,
                            output1.GetValue(lane));
    }
}

template <uint16_t buffer_num>
__aicore__ inline int64_t MeasureSwigluBufferLifecycle(
        uint32_t tile_elements) {
    const uint32_t bf16_bytes = tile_elements * sizeof(bfloat16_t);
    const uint32_t fp32_bytes = tile_elements * sizeof(float);
    const int64_t cycle_start = AscendC::GetSystemCycle();
    {
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::QuePosition::VECIN, buffer_num> input_a;
        AscendC::TQue<AscendC::QuePosition::VECIN, buffer_num> input_b;
        AscendC::TQue<AscendC::QuePosition::VECOUT, buffer_num> output_c;
        AscendC::TBuf<AscendC::TPosition::VECCALC> input_temporary;
        AscendC::TBuf<AscendC::TPosition::VECCALC> output_temporary;
        pipe.InitBuffer(input_a, buffer_num, bf16_bytes);
        pipe.InitBuffer(input_b, buffer_num, bf16_bytes);
        pipe.InitBuffer(output_c, buffer_num, bf16_bytes);
        pipe.InitBuffer(input_temporary, fp32_bytes);
        pipe.InitBuffer(output_temporary, fp32_bytes);
    }
    return AscendC::GetSystemCycle() - cycle_start;
}

template <uint16_t buffer_num>
__aicore__ inline int64_t MeasureSwigluQueueControl(
        uint32_t tile_elements) {
    const uint32_t tensor_bytes = tile_elements * sizeof(bfloat16_t);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, buffer_num> input_a;
    AscendC::TQue<AscendC::QuePosition::VECIN, buffer_num> input_b;
    AscendC::TQue<AscendC::QuePosition::VECOUT, buffer_num> output_c;
    pipe.InitBuffer(input_a, buffer_num, tensor_bytes);
    pipe.InitBuffer(input_b, buffer_num, tensor_bytes);
    pipe.InitBuffer(output_c, buffer_num, tensor_bytes);
    const int64_t cycle_start = AscendC::GetSystemCycle();
    auto a = input_a.template AllocTensor<bfloat16_t>();
    input_a.EnQue(a);
    auto b = input_b.template AllocTensor<bfloat16_t>();
    input_b.EnQue(b);
    a = input_a.template DeQue<bfloat16_t>();
    input_a.FreeTensor(a);
    b = input_b.template DeQue<bfloat16_t>();
    input_b.FreeTensor(b);
    auto c = output_c.template AllocTensor<bfloat16_t>();
    output_c.EnQue(c);
    c = output_c.template DeQue<bfloat16_t>();
    output_c.FreeTensor(c);
    return AscendC::GetSystemCycle() - cycle_start;
}

__aicore__ inline int64_t MeasureRmsNormBufferLifecycle(
        uint32_t hidden_elements) {
    const uint32_t bf16_bytes = hidden_elements * sizeof(bfloat16_t);
    const uint32_t fp32_bytes = hidden_elements * sizeof(float);
    constexpr uint32_t kRowFactor = 64;
    constexpr uint32_t kReduceElements = 64;
    const int64_t cycle_start = AscendC::GetSystemCycle();
    {
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_x;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_gamma;
        AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_y;
        AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_rstd;
        AscendC::TBuf<AscendC::TPosition::VECCALC> x_fp32;
        AscendC::TBuf<AscendC::TPosition::VECCALC> sqx;
        AscendC::TBuf<AscendC::TPosition::VECCALC> reduce_fp32;
        pipe.InitBuffer(input_x, 1, bf16_bytes);
        pipe.InitBuffer(input_gamma, 1, bf16_bytes);
        pipe.InitBuffer(output_y, 1, bf16_bytes);
        pipe.InitBuffer(output_rstd, 1, kRowFactor * sizeof(float));
        pipe.InitBuffer(x_fp32, fp32_bytes);
        pipe.InitBuffer(sqx, fp32_bytes);
        pipe.InitBuffer(reduce_fp32, kReduceElements * sizeof(float));
    }
    return AscendC::GetSystemCycle() - cycle_start;
}

__aicore__ inline int64_t MeasureRmsNormBufferReady(
        uint32_t hidden_elements) {
    const uint32_t bf16_bytes = hidden_elements * sizeof(bfloat16_t);
    const uint32_t fp32_bytes = hidden_elements * sizeof(float);
    constexpr uint32_t kRowFactor = 64;
    constexpr uint32_t kReduceElements = 64;
    const int64_t cycle_start = AscendC::GetSystemCycle();
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_x;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_gamma;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_y;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_rstd;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x_fp32;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sqx;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduce_fp32;
    pipe.InitBuffer(input_x, 1, bf16_bytes);
    pipe.InitBuffer(input_gamma, 1, bf16_bytes);
    pipe.InitBuffer(output_y, 1, bf16_bytes);
    pipe.InitBuffer(output_rstd, 1, kRowFactor * sizeof(float));
    pipe.InitBuffer(x_fp32, fp32_bytes);
    pipe.InitBuffer(sqx, fp32_bytes);
    pipe.InitBuffer(reduce_fp32, kReduceElements * sizeof(float));
    AscendC::PipeBarrier<PIPE_ALL>();
    return AscendC::GetSystemCycle() - cycle_start;
}

__aicore__ inline int64_t MeasureRmsNormQueueControl(
        uint32_t hidden_elements) {
    const uint32_t tensor_bytes = hidden_elements * sizeof(bfloat16_t);
    constexpr uint32_t kRowFactor = 64;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_x;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_gamma;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_y;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_rstd;
    pipe.InitBuffer(input_x, 1, tensor_bytes);
    pipe.InitBuffer(input_gamma, 1, tensor_bytes);
    pipe.InitBuffer(output_y, 1, tensor_bytes);
    pipe.InitBuffer(output_rstd, 1, kRowFactor * sizeof(float));
    const int64_t cycle_start = AscendC::GetSystemCycle();
    auto x = input_x.AllocTensor<bfloat16_t>();
    input_x.EnQue(x);
    x = input_x.DeQue<bfloat16_t>();
    input_x.FreeTensor(x);
    auto gamma = input_gamma.AllocTensor<bfloat16_t>();
    input_gamma.EnQue(gamma);
    gamma = input_gamma.DeQue<bfloat16_t>();
    input_gamma.FreeTensor(gamma);
    auto y = output_y.AllocTensor<bfloat16_t>();
    output_y.EnQue(y);
    y = output_y.DeQue<bfloat16_t>();
    output_y.FreeTensor(y);
    auto rstd = output_rstd.AllocTensor<float>();
    output_rstd.EnQue(rstd);
    rstd = output_rstd.DeQue<float>();
    output_rstd.FreeTensor(rstd);
    return AscendC::GetSystemCycle() - cycle_start;
}

__aicore__ inline void RunExp(
        const AscendC::LocalTensor<float>& destination0,
        const AscendC::LocalTensor<float>& destination1,
        const AscendC::LocalTensor<float>& source0,
        uint32_t elements, int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; iteration += 2) {
        AscendC::Exp(destination0, source0, elements);
        AscendC::Exp(destination1, source0, elements);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void RunDiv(
        const AscendC::LocalTensor<float>& destination0,
        const AscendC::LocalTensor<float>& destination1,
        const AscendC::LocalTensor<float>& source0,
        const AscendC::LocalTensor<float>& source1,
        uint32_t elements, int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; iteration += 2) {
        AscendC::Div(destination0, source0, source1, elements);
        AscendC::Div(destination1, source0, source1, elements);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void RunExpDiv(
        const AscendC::LocalTensor<float>& destination0,
        const AscendC::LocalTensor<float>& destination1,
        const AscendC::LocalTensor<float>& source0,
        const AscendC::LocalTensor<float>& source1,
        uint32_t elements, int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        AscendC::Exp(destination0, source0, elements);
        AscendC::Div(destination1, source0, source1, elements);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void RunSqrt(
        const AscendC::LocalTensor<float>& destination0,
        const AscendC::LocalTensor<float>& destination1,
        const AscendC::LocalTensor<float>& source0,
        uint32_t elements, int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; iteration += 2) {
        AscendC::Sqrt(destination0, source0, elements);
        AscendC::Sqrt(destination1, source0, elements);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void RunSqrtDiv(
        const AscendC::LocalTensor<float>& destination0,
        const AscendC::LocalTensor<float>& destination1,
        const AscendC::LocalTensor<float>& source0,
        const AscendC::LocalTensor<float>& source1,
        uint32_t elements, int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        AscendC::Sqrt(destination0, source0, elements);
        AscendC::Div(destination1, source0, source1, elements);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void RunExpSerialized(
        const AscendC::LocalTensor<float>& destination0,
        const AscendC::LocalTensor<float>& destination1,
        const AscendC::LocalTensor<float>& source0,
        uint32_t elements, int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; iteration += 2) {
        AscendC::Exp(destination0, source0, elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(destination1, source0, elements);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void RunDivSerialized(
        const AscendC::LocalTensor<float>& destination0,
        const AscendC::LocalTensor<float>& destination1,
        const AscendC::LocalTensor<float>& source0,
        const AscendC::LocalTensor<float>& source1,
        uint32_t elements, int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; iteration += 2) {
        AscendC::Div(destination0, source0, source1, elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(destination1, source0, source1, elements);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void RunCastBf16ToFp32Serialized(
        const AscendC::LocalTensor<float>& destination0,
        const AscendC::LocalTensor<float>& destination1,
        const AscendC::LocalTensor<bfloat16_t>& source,
        uint32_t elements, int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; iteration += 2) {
        AscendC::Cast(
            destination0, source, AscendC::RoundMode::CAST_NONE, elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(
            destination1, source, AscendC::RoundMode::CAST_NONE, elements);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void RunCastFp32ToBf16Serialized(
        const AscendC::LocalTensor<bfloat16_t>& destination0,
        const AscendC::LocalTensor<bfloat16_t>& destination1,
        const AscendC::LocalTensor<float>& source,
        uint32_t elements, int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; iteration += 2) {
        AscendC::Cast(
            destination0, source, AscendC::RoundMode::CAST_RINT, elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(
            destination1, source, AscendC::RoundMode::CAST_RINT, elements);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void RunMulsSerialized(
        const AscendC::LocalTensor<float>& destination0,
        const AscendC::LocalTensor<float>& destination1,
        const AscendC::LocalTensor<float>& source,
        float scalar, uint32_t elements, int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; iteration += 2) {
        AscendC::Muls(destination0, source, scalar, elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(destination1, source, scalar, elements);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void RunAddsSerialized(
        const AscendC::LocalTensor<float>& destination0,
        const AscendC::LocalTensor<float>& destination1,
        const AscendC::LocalTensor<float>& source,
        float scalar, uint32_t elements, int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; iteration += 2) {
        AscendC::Adds(destination0, source, scalar, elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(destination1, source, scalar, elements);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void RunMulSerialized(
        const AscendC::LocalTensor<float>& destination0,
        const AscendC::LocalTensor<float>& destination1,
        const AscendC::LocalTensor<float>& source0,
        const AscendC::LocalTensor<float>& source1,
        uint32_t elements, int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; iteration += 2) {
        AscendC::Mul(destination0, source0, source1, elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(destination1, source0, source1, elements);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void RunNormSqrtSerialized(
        const AscendC::LocalTensor<float>& rstd,
        uint32_t elements, int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        AscendC::Sqrt(rstd, rstd, elements);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void RunNormDivSerialized(
        const AscendC::LocalTensor<float>& rstd,
        const AscendC::LocalTensor<float>& one,
        uint32_t elements, int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        AscendC::Div(rstd, one, rstd, elements);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void RunNormSqrtDuplicateDivSerialized(
        const AscendC::LocalTensor<float>& rstd,
        const AscendC::LocalTensor<float>& one,
        uint32_t elements, int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        AscendC::Sqrt(rstd, rstd, elements);
        AscendC::Duplicate<float>(one, 1.0f, elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(rstd, one, rstd, elements);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void RunRmsNormLocalScalarDependency(
        const AscendC::LocalTensor<float>& reduction,
        const AscendC::LocalTensor<float>& rstd,
        int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        // Mirror the two V->S handoffs and one S->V handoff in the fixed
        // CANN 8.5.1 key-0 source.  The reduction value is initialized in UB
        // before timing, so this isolates only the local handoff and scalar
        // normalization chain rather than replaying RMSNorm.
        const event_t reduction_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(reduction_event);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(reduction_event);
        const float reduce_out = reduction.GetValue(0);
        const event_t scalar_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(scalar_event);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(scalar_event);
        const float rstd_value = 1 / sqrt(reduce_out * 0.25f + 1.0e-5f);
        rstd.SetValue(0, rstd_value);
        const event_t vector_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(vector_event);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(vector_event);
    }
}

__aicore__ inline void RunAddRmsNormLocalScalarHandoff(
        const AscendC::LocalTensor<float>& reduction,
        const AscendC::LocalTensor<float>& rstd,
        int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        const event_t reduction_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(reduction_event);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(reduction_event);
        const float rstd_value = reduction.GetValue(0);
        const event_t vector_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(vector_event);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(vector_event);
        rstd.SetValue(0, rstd_value);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline float RunAddRmsNormSplitDLocalScalarHandoff(
        const AscendC::LocalTensor<float>& reduction,
        int64_t iterations) {
    float rstd_value = 0.0f;
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        // Exact CANN 8.5.1 key-31 ComputeY control sequence.  Unlike key 30,
        // Split-D reads the already materialized rstd scalar for every
        // output column tile and does not write it back with SetValue.
        const event_t reduction_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(reduction_event);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(reduction_event);
        rstd_value = reduction.GetValue(0);
        const event_t vector_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(vector_event);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(vector_event);
        AscendC::PipeBarrier<PIPE_V>();
    }
    return rstd_value;
}

__aicore__ inline void RunVectorToMte2Fence(int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        const event_t event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
    }
}

__aicore__ inline void RunLocalScalarFillSv(
        const AscendC::LocalTensor<float>& destination,
        uint32_t writes, float negative_value, float positive_value,
        int64_t iterations) {
    const uint32_t half_writes = writes / 2;
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        #pragma unroll 1
        for (uint32_t index = 0; index < half_writes; ++index) {
            destination.SetValue(index, negative_value);
            destination.SetValue(index + half_writes, positive_value);
        }
        const event_t vector_event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(vector_event);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(vector_event);
    }
}

__aicore__ inline void RunLocalUbCopyChain(
        const AscendC::LocalTensor<float>& destination,
        const AscendC::LocalTensor<float>& source,
        uint32_t payload_bytes, uint32_t block_bytes,
        int64_t iterations) {
    const uint16_t block_count = static_cast<uint16_t>(
        payload_bytes / block_bytes);
    const uint16_t block_length = static_cast<uint16_t>(block_bytes / 32);
    const uint16_t stride = block_count == 1 ? 0 : block_length;
    const AscendC::DataCopyParams copy_params = {
        block_count, block_length, stride, stride};
    constexpr uint32_t kDestinationStrideElements =
        kLocalCopyDestinationStrideBytes / sizeof(float);
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations;
         iteration += kLocalCopyUnroll) {
        #pragma unroll 16
        for (int64_t slot = 0; slot < kLocalCopyUnroll; ++slot) {
            DataCopy(
                destination[slot * kDestinationStrideElements],
                source,
                copy_params);
        }
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline uint32_t RunScalarU32DivRemDependency(
        uint32_t value, uint32_t divisor, int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        const uint32_t quotient = value / divisor;
        const uint32_t remainder = value % divisor;
        value = quotient + remainder + 0x9e3779b9U;
    }
    return value;
}

__aicore__ inline uint32_t RunScalarU32MulAddDependency(
        uint32_t value, uint32_t multiplier, int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        value = value * multiplier + 1013904223U;
    }
    return value;
}

__aicore__ inline uint64_t RunScalarU64MulDependency(
        uint64_t value, uint64_t multiplier, int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        value *= multiplier;
    }
    return value;
}

__aicore__ inline uint64_t RunScalarU64DivRemDependency(
        uint64_t value, uint64_t divisor, int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        const uint64_t quotient = value / divisor;
        const uint64_t remainder = value % divisor;
        value = quotient + remainder + 0x9e3779b97f4a7c15ULL;
    }
    return value;
}

__aicore__ inline uint32_t GreatestPowerOfTwo(uint32_t value) {
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return (value + 1) >> 1;
}

__aicore__ inline void RunRmsNormHalfIntervalReduce(
        const AscendC::LocalTensor<float>& source,
        uint32_t elements, int64_t iterations) {
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        uint32_t body_count = GreatestPowerOfTwo(elements);
        const uint32_t tail_count = elements - body_count;
        if (tail_count > 0) {
            AscendC::Add(
                source, source, source[body_count], tail_count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        while (body_count > 64) {
            body_count /= 2;
            AscendC::Add(
                source, source, source[body_count], body_count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::AscendCUtils::SetMask<float>(64);
        AscendC::WholeReduceSum<float, false>(
            source, source, 64, 1, 0, 1, 0);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void RunAddRmsNormReduceSumFp32(
        const AscendC::LocalTensor<float>& destination,
        const AscendC::LocalTensor<float>& source,
        const AscendC::LocalTensor<float>& work,
        uint32_t elements, int64_t iterations) {
    constexpr uint32_t kElementsPerRepeat = 64;
    const uint32_t repeat_times = elements / kElementsPerRepeat;
    const uint32_t tail_count = elements % kElementsPerRepeat;
    const uint32_t body_count = repeat_times * kElementsPerRepeat;
    AscendC::BinaryRepeatParams repeat_parameters;
    repeat_parameters.src0RepStride = 8;
    repeat_parameters.src0BlkStride = 1;
    repeat_parameters.src1RepStride = 0;
    repeat_parameters.src1BlkStride = 1;
    repeat_parameters.dstRepStride = 0;
    repeat_parameters.dstBlkStride = 1;
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        AscendC::Duplicate(work, 0.0f, kElementsPerRepeat);
        AscendC::PipeBarrier<PIPE_V>();
        if (repeat_times > 0) {
            AscendC::Add(
                work, source, work, kElementsPerRepeat, repeat_times,
                repeat_parameters);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (tail_count != 0) {
            AscendC::Add(
                work, source[body_count], work, tail_count, 1,
                repeat_parameters);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::AscendCUtils::SetMask<float>(kElementsPerRepeat);
        AscendC::WholeReduceSum<float, false>(
            destination, work, kElementsPerRepeat, 1, 0, 1, 0);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void RunAddRmsNormSplitDBlockReduceSumFp32(
        const AscendC::LocalTensor<float>& data,
        uint32_t elements, int64_t iterations) {
    // Exact CANN 8.5.1 AddRmsNormSplitD::BlockReduceSumFP32 body. The
    // production helper receives one eight-FP32 partial-sum block per row,
    // after its preceding ReduceSumFP32ToBlock operation. This probe keeps
    // only that UB-resident Vector primitive and its PIPE_V boundaries.
    constexpr uint32_t kElementsPerRepeat = 64;
    constexpr uint32_t kPartialSumsPerRow = 8;
    const uint32_t repeat_times = elements / kElementsPerRepeat;
    const uint32_t tail_count = elements % kElementsPerRepeat;
    const uint32_t destination_offset = repeat_times * kPartialSumsPerRow;
    const uint32_t source_offset = repeat_times * kElementsPerRepeat;
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        if (repeat_times > 0) {
            AscendC::BlockReduceSum(
                data, data, repeat_times, kElementsPerRepeat,
                1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (tail_count > 0) {
            AscendC::BlockReduceSum(
                data[destination_offset], data[source_offset],
                1, tail_count, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

__aicore__ inline void RunMergeNReduceSumMultiN(
        const AscendC::LocalTensor<float>& destination,
        const AscendC::LocalTensor<float>& source,
        const AscendC::LocalTensor<float>& work,
        uint32_t rows, int64_t iterations) {
    // reduce_common.h:91-101 ReduceSumMultiN -> :60 ReduceSumForSmallReduceDim
    // -> :28 ReduceSumForSmallReduceDimPreRepeat.  numCol == numColAlign at
    // kMergeNColumns, so tailCount is zero, and rows <= rowFactor = 84 is
    // below MAX_REP_NUM = 255, so the single-pre-repeat branch is the only
    // one a production row group takes.
    const uint8_t repeat_stride = kMergeNColumns / kMergeNElementsPerBlock;
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        AscendC::Duplicate(work, 0.0f, rows * kMergeNElementsPerRepeat);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t element = 0;
                element + kMergeNElementsPerRepeat <= kMergeNColumns;
                element += kMergeNElementsPerRepeat) {
            AscendC::Add(
                work, source[element], work, kMergeNElementsPerRepeat, rows,
                {1, 1, 1, static_cast<uint8_t>(kMergeNElementsPerBlock),
                 repeat_stride,
                 static_cast<uint8_t>(kMergeNElementsPerBlock)});
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::AscendCUtils::SetMask<float>(kMergeNElementsPerRepeat);
        AscendC::WholeReduceSum<float, false>(
            destination, work, kMergeNElementsPerRepeat, rows, 1, 1,
            kMergeNElementsPerBlock);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void RunMergeNBroadcastGamma(
        const AscendC::LocalTensor<float>& gamma,
        AscendC::LocalTensor<uint8_t> shared,
        uint32_t rows, int64_t iterations) {
    // rms_norm_merge_n.h:151-157 BroadCastGamma.  It replicates one
    // numColAlign gamma vector over the whole rowFactor tile once per launch,
    // in place, before the row-group loop, and is skipped when the host tiler
    // sets is_performance.
    const uint32_t source_shape[2] = {1, kMergeNColumns};
    const uint32_t destination_shape[2] = {rows, kMergeNColumns};
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        AscendC::BroadCast<float, 2, 0>(
            gamma, gamma, destination_shape, source_shape, shared);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void RunMergeNBroadcastRstd(
        const AscendC::LocalTensor<float>& destination,
        const AscendC::LocalTensor<float>& rstd,
        AscendC::LocalTensor<uint8_t> shared,
        uint32_t rows, int64_t iterations) {
    // rms_norm_merge_n.h:271-274.  One row group expands its per-row rstd
    // column into the full {calcRowNum, numColAlign} tile.
    const uint32_t source_shape[2] = {rows, 1};
    const uint32_t destination_shape[2] = {rows, kMergeNColumns};
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        AscendC::BroadCast<float, 2, 1>(
            destination, rstd, destination_shape, source_shape, shared);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void RunMergeNPerformanceRstdBrcb(
        const AscendC::LocalTensor<float>& destination,
        const AscendC::LocalTensor<float>& rstd,
        uint32_t rows, int64_t iterations) {
    // rms_norm_merge_n.h:316-330 doPerformanceRstd.  splidRow is 240 and a
    // production row group never exceeds rowFactor = 84, so only the tail
    // Brcb call executes.
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        AscendC::Brcb(destination, rstd, rows, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void RunMergeNPerformanceMulByRow(
        const AscendC::LocalTensor<float>& destination,
        const AscendC::LocalTensor<float>& source0,
        const AscendC::LocalTensor<float>& source1,
        uint32_t rows, bool is_rstd, int64_t iterations) {
    // rms_norm_merge_n.h:332-376 repeatByRow -> mulRepeat.  singlT is 255 and
    // a production row group never exceeds rowFactor = 84, so only the tail
    // call executes.  mulLoop = numColAlign / 64 and mulTail = 0 at
    // kMergeNColumns; dstRepStride = numColAlign / 8.
    const uint32_t mul_loop = kMergeNColumns / kMergeNElementsPerRepeat;
    const uint8_t destination_repeat_stride =
        kMergeNColumns / kMergeNElementsPerBlock;
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        for (uint32_t index = 0; index < mul_loop; ++index) {
            const uint32_t offset = index * kMergeNElementsPerRepeat;
            if (is_rstd) {
                AscendC::Mul(
                    destination[offset], source0[offset], source1,
                    kMergeNElementsPerRepeat, rows,
                    {1, 1, 0, destination_repeat_stride,
                     destination_repeat_stride, 1});
            } else {
                AscendC::Mul(
                    destination[offset], source0[offset], source1[offset],
                    kMergeNElementsPerRepeat, rows,
                    {1, 1, 1, destination_repeat_stride,
                     destination_repeat_stride, 0});
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

}  // namespace

extern "C" __global__ __aicore__ void vector_opcode(
        __gm__ int64_t* gm_cycles, __gm__ float* gm_validation,
        int64_t opcode, int64_t element_count, int64_t iterations,
        float source_value, float divisor_value) {
    const int64_t block_index = AscendC::GetBlockIdx();
    const int64_t cycle_index = block_index * kCycleValuesPerBlock;
    AscendC::GlobalTensor<int64_t> cycles;
    cycles.SetGlobalBuffer(gm_cycles);
    if ((opcode != kExp && opcode != kDiv && opcode != kExpDiv
                && opcode != kSqrt && opcode != kSqrtDiv
                && opcode != kExpSerialized && opcode != kDivSerialized
                && opcode != kCastBf16ToFp32Serialized
                && opcode != kCastFp32ToBf16Serialized
                && opcode != kMulsSerialized && opcode != kAddsSerialized
                && opcode != kMulSerialized
                && opcode != kNormSqrtSerialized
                && opcode != kNormDivSerialized
                && opcode != kNormSqrtDuplicateDivSerialized
                && opcode != kTwoTbufMaskPrologue
                && opcode != kRmsNormLocalScalarDependency
                && opcode != kLocalScalarFillSv
                && opcode != kLocalUbCopyContiguous
                && opcode != kLocalUbCopyStride64
                && opcode != kLocalUbCopyStride128
                && opcode != kSingleTbufInit
                && opcode != kScalarU32DivRemDependency
                && opcode != kScalarU32MulAddDependency
                && opcode != kScalarU64MulDependency
                && opcode != kSwigluDoubleBufferInit
                && opcode != kSwigluSingleBufferInit
                && opcode != kRmsNormHalfIntervalReduce
                && opcode != kSwigluDoubleBufferLifecycle
                && opcode != kSwigluSingleBufferLifecycle
                && opcode != kSwigluDoubleBufferQueueControl
                && opcode != kSwigluSingleBufferQueueControl
                && opcode != kScalarU64DivRemDependency
                && opcode != kRmsNormFourQueueThreeTbufLifecycle
                && opcode != kRmsNormFourQueueControlChain
                && opcode != kRmsNormFourQueueThreeTbufReady
                && opcode != kAddRmsNormLocalScalarHandoff
                && opcode != kAddRmsNormReduceSumFp32
                && opcode != kVectorToMte2Fence
                && opcode != kAddRmsNormSplitDLocalScalarHandoff
                && opcode != kAddRmsNormSplitDFourQueueFourTbufLifecycle
                && opcode != kAddRmsNormSplitDBlockReduceSumFp32
                && opcode != kMergeNReduceSumMultiN
                && opcode != kMergeNBroadcastGamma
                && opcode != kMergeNBroadcastRstd
                && opcode != kMergeNPerformanceRstdBrcb
                && opcode != kMergeNPerformanceMulRowRstd
                && opcode != kMergeNPerformanceMulRowGamma
                && opcode != kCompareFp32Gt
                && opcode != kCompareFp32Eq
                && opcode != kCompareFp32Ne
                && opcode != kSelectFp32
                && opcode != kCastInt32ToInt64)
            || element_count < 1 || element_count > kMaximumElements
            || iterations <= 0
            || ((opcode != kTwoTbufMaskPrologue
                    && opcode != kSingleTbufInit
                    && opcode != kSwigluDoubleBufferInit
                    && opcode != kSwigluSingleBufferInit
                    && opcode != kSwigluDoubleBufferLifecycle
                    && opcode != kSwigluSingleBufferLifecycle
                    && opcode != kSwigluDoubleBufferQueueControl
                    && opcode != kSwigluSingleBufferQueueControl
                    && opcode != kRmsNormFourQueueThreeTbufLifecycle
                    && opcode != kRmsNormFourQueueControlChain
                    && opcode != kRmsNormFourQueueThreeTbufReady)
                && iterations % 2 != 0)
            || (opcode == kLocalScalarFillSv
                && element_count != 64 && element_count != 128)
            || ((opcode == kMergeNReduceSumMultiN
                    || opcode == kMergeNBroadcastGamma
                    || opcode == kMergeNBroadcastRstd
                    || opcode == kMergeNPerformanceRstdBrcb
                    || opcode == kMergeNPerformanceMulRowRstd
                    || opcode == kMergeNPerformanceMulRowGamma)
                && element_count > kMergeNMaximumRows)
            || ((opcode == kLocalUbCopyContiguous
                    || opcode == kLocalUbCopyStride64
                    || opcode == kLocalUbCopyStride128)
                && (element_count % 32 != 0
                    || element_count > 4096
                    || iterations % kLocalCopyUnroll != 0))
            || divisor_value == 0.0f) {
        cycles.SetValue(cycle_index, -1);
        return;
    }

    if (opcode == kCastInt32ToInt64) {
        if (element_count % 64 != 0 || element_count > 4096) {
            cycles.SetValue(cycle_index, -4);
            return;
        }
        MeasureInt32Int64Cast(cycles, cycle_index, gm_validation,
            block_index, static_cast<uint32_t>(element_count), iterations);
        return;
    }

    if (opcode >= kCompareFp32Gt && opcode <= kSelectFp32) {
        if (element_count % 64 != 0 || element_count > 4096) {
            cycles.SetValue(cycle_index, -3);
            return;
        }
        MeasureFp32PredicateSelect(cycles, cycle_index, gm_validation,
            block_index, opcode, static_cast<uint32_t>(element_count),
            iterations);
        return;
    }

    if (opcode == kSwigluDoubleBufferInit
            || opcode == kSwigluSingleBufferInit) {
        const uint32_t tile_elements = static_cast<uint32_t>(element_count);
        const uint32_t bf16_bytes = tile_elements * sizeof(bfloat16_t);
        const uint32_t fp32_bytes = tile_elements * sizeof(float);
        const int64_t cycle_start = AscendC::GetSystemCycle();
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> input_temporary;
        AscendC::TBuf<AscendC::TPosition::VECCALC> output_temporary;
        if (opcode == kSwigluDoubleBufferInit) {
            AscendC::TQue<AscendC::QuePosition::VECIN, 2> input_a;
            AscendC::TQue<AscendC::QuePosition::VECIN, 2> input_b;
            AscendC::TQue<AscendC::QuePosition::VECOUT, 2> output_c;
            pipe.InitBuffer(input_a, 2, bf16_bytes);
            pipe.InitBuffer(input_b, 2, bf16_bytes);
            pipe.InitBuffer(output_c, 2, bf16_bytes);
            pipe.InitBuffer(input_temporary, fp32_bytes);
            pipe.InitBuffer(output_temporary, fp32_bytes);
        } else {
            AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_a;
            AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_b;
            AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_c;
            pipe.InitBuffer(input_a, 1, bf16_bytes);
            pipe.InitBuffer(input_b, 1, bf16_bytes);
            pipe.InitBuffer(output_c, 1, bf16_bytes);
            pipe.InitBuffer(input_temporary, fp32_bytes);
            pipe.InitBuffer(output_temporary, fp32_bytes);
        }
        const int64_t cycle_end = AscendC::GetSystemCycle();
        cycles.SetValue(cycle_index, cycle_end - cycle_start);
        AscendC::GlobalTensor<float> validation;
        validation.SetGlobalBuffer(gm_validation);
        validation.SetValue(
            block_index * kValidationValuesPerBlock, source_value);
        return;
    }

    if (opcode == kSwigluDoubleBufferLifecycle
            || opcode == kSwigluSingleBufferLifecycle) {
        const uint32_t tile_elements = static_cast<uint32_t>(element_count);
        const int64_t lifecycle_cycles = (
            opcode == kSwigluDoubleBufferLifecycle
            ? MeasureSwigluBufferLifecycle<2>(tile_elements)
            : MeasureSwigluBufferLifecycle<1>(tile_elements));
        cycles.SetValue(cycle_index, lifecycle_cycles);
        AscendC::GlobalTensor<float> validation;
        validation.SetGlobalBuffer(gm_validation);
        validation.SetValue(
            block_index * kValidationValuesPerBlock, source_value);
        return;
    }

    if (opcode == kSwigluDoubleBufferQueueControl
            || opcode == kSwigluSingleBufferQueueControl) {
        const uint32_t tile_elements = static_cast<uint32_t>(element_count);
        const int64_t control_cycles = (
            opcode == kSwigluDoubleBufferQueueControl
            ? MeasureSwigluQueueControl<2>(tile_elements)
            : MeasureSwigluQueueControl<1>(tile_elements));
        cycles.SetValue(cycle_index, control_cycles);
        AscendC::GlobalTensor<float> validation;
        validation.SetGlobalBuffer(gm_validation);
        validation.SetValue(
            block_index * kValidationValuesPerBlock, source_value);
        return;
    }

    if (opcode == kRmsNormFourQueueThreeTbufLifecycle
            || opcode == kRmsNormFourQueueControlChain
            || opcode == kRmsNormFourQueueThreeTbufReady) {
        const uint32_t hidden_elements = static_cast<uint32_t>(element_count);
        int64_t control_cycles;
        if (opcode == kRmsNormFourQueueThreeTbufLifecycle) {
            control_cycles = MeasureRmsNormBufferLifecycle(hidden_elements);
        } else if (opcode == kRmsNormFourQueueControlChain) {
            control_cycles = MeasureRmsNormQueueControl(hidden_elements);
        } else {
            control_cycles = MeasureRmsNormBufferReady(hidden_elements);
        }
        cycles.SetValue(cycle_index, control_cycles);
        AscendC::GlobalTensor<float> validation;
        validation.SetGlobalBuffer(gm_validation);
        validation.SetValue(
            block_index * kValidationValuesPerBlock, source_value);
        return;
    }

    if (opcode == kMergeNReduceSumMultiN
            || opcode == kMergeNBroadcastGamma
            || opcode == kMergeNBroadcastRstd
            || opcode == kMergeNPerformanceRstdBrcb
            || opcode == kMergeNPerformanceMulRowRstd
            || opcode == kMergeNPerformanceMulRowGamma) {
        const uint32_t rows = static_cast<uint32_t>(element_count);
        // Allocate the full rowFactor tile regardless of the measured row
        // count so buffer geometry cannot vary across the sweep.  The tile
        // and shared-temporary extents are the ones rms_norm_merge_n.h:42-51
        // requests at numColAlign == kMergeNColumns.
        constexpr uint32_t kTileElements =
            kMergeNMaximumRows * kMergeNColumns;
        constexpr uint32_t kTileBytes = kTileElements * sizeof(float);
        constexpr uint32_t kSharedBytes =
            kMergeNMaximumRows * kMergeNElementsPerRepeat * sizeof(float);
        constexpr uint32_t kRowBytes =
            kMergeNMaximumRows * kMergeNElementsPerBlock * sizeof(float);
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> tile_a_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> tile_b_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> shared_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> row_buffer;
        pipe.InitBuffer(tile_a_buffer, kTileBytes);
        pipe.InitBuffer(tile_b_buffer, kTileBytes);
        pipe.InitBuffer(shared_buffer, kSharedBytes);
        pipe.InitBuffer(row_buffer, kRowBytes);
        auto tile_a = tile_a_buffer.Get<float>();
        auto tile_b = tile_b_buffer.Get<float>();
        auto shared_float = shared_buffer.Get<float>();
        auto shared = shared_buffer.Get<uint8_t>();
        auto rows_tensor = row_buffer.Get<float>();
        AscendC::Duplicate<float>(tile_a, source_value, kTileElements);
        AscendC::Duplicate<float>(tile_b, divisor_value, kTileElements);
        AscendC::Duplicate<float>(
            rows_tensor, source_value,
            kMergeNMaximumRows * kMergeNElementsPerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        const int64_t cycle_start = AscendC::GetSystemCycle();
        if (opcode == kMergeNReduceSumMultiN) {
            RunMergeNReduceSumMultiN(
                rows_tensor, tile_a, shared_float, rows, iterations);
        } else if (opcode == kMergeNBroadcastGamma) {
            RunMergeNBroadcastGamma(tile_a, shared, rows, iterations);
        } else if (opcode == kMergeNBroadcastRstd) {
            RunMergeNBroadcastRstd(
                tile_a, rows_tensor, shared, rows, iterations);
        } else if (opcode == kMergeNPerformanceRstdBrcb) {
            RunMergeNPerformanceRstdBrcb(
                tile_a, rows_tensor, rows, iterations);
        } else {
            RunMergeNPerformanceMulByRow(
                tile_a, tile_b,
                opcode == kMergeNPerformanceMulRowRstd
                ? rows_tensor : tile_b,
                rows,
                opcode == kMergeNPerformanceMulRowRstd,
                iterations);
        }
        const int64_t cycle_end = AscendC::GetSystemCycle();
        cycles.SetValue(cycle_index, cycle_end - cycle_start);
        AscendC::GlobalTensor<float> validation;
        validation.SetGlobalBuffer(gm_validation);
        validation.SetValue(
            block_index * kValidationValuesPerBlock, tile_a.GetValue(0));
        return;
    }

    if (opcode == kRmsNormHalfIntervalReduce) {
        const uint32_t elements = static_cast<uint32_t>(element_count);
        const size_t buffer_bytes = static_cast<size_t>(elements)
            * sizeof(float);
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> source_buffer;
        pipe.InitBuffer(source_buffer, buffer_bytes);
        auto source = source_buffer.Get<float>();
        AscendC::Duplicate<float>(source, 0.0f, elements);
        AscendC::PipeBarrier<PIPE_V>();
        const int64_t cycle_start = AscendC::GetSystemCycle();
        RunRmsNormHalfIntervalReduce(source, elements, iterations);
        const int64_t cycle_end = AscendC::GetSystemCycle();
        cycles.SetValue(cycle_index, cycle_end - cycle_start);
        AscendC::GlobalTensor<float> validation;
        validation.SetGlobalBuffer(gm_validation);
        validation.SetValue(
            block_index * kValidationValuesPerBlock, source.GetValue(0));
        return;
    }

    if (opcode == kAddRmsNormReduceSumFp32) {
        const uint32_t elements = static_cast<uint32_t>(element_count);
        const size_t source_bytes = static_cast<size_t>(elements)
            * sizeof(float);
        constexpr size_t kReductionBytes = 64 * sizeof(float);
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> source_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> work_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> destination_buffer;
        pipe.InitBuffer(source_buffer, source_bytes);
        pipe.InitBuffer(work_buffer, kReductionBytes);
        pipe.InitBuffer(destination_buffer, kReductionBytes);
        auto source = source_buffer.Get<float>();
        auto work = work_buffer.Get<float>();
        auto destination = destination_buffer.Get<float>();
        AscendC::Duplicate<float>(source, source_value, elements);
        AscendC::PipeBarrier<PIPE_V>();
        const int64_t cycle_start = AscendC::GetSystemCycle();
        RunAddRmsNormReduceSumFp32(
            destination, source, work, elements, iterations);
        const int64_t cycle_end = AscendC::GetSystemCycle();
        cycles.SetValue(cycle_index, cycle_end - cycle_start);
        AscendC::GlobalTensor<float> validation;
        validation.SetGlobalBuffer(gm_validation);
        validation.SetValue(
            block_index * kValidationValuesPerBlock,
            destination.GetValue(0));
        return;
    }

    if (opcode == kAddRmsNormSplitDBlockReduceSumFp32) {
        const uint32_t elements = static_cast<uint32_t>(element_count);
        if (elements % 8 != 0) {
            cycles.SetValue(cycle_index, -4);
            return;
        }
        const size_t source_bytes = static_cast<size_t>(elements)
            * sizeof(float);
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> data_buffer;
        pipe.InitBuffer(data_buffer, source_bytes);
        auto data = data_buffer.Get<float>();
        AscendC::Duplicate<float>(data, source_value, elements);
        AscendC::PipeBarrier<PIPE_V>();
        const int64_t cycle_start = AscendC::GetSystemCycle();
        RunAddRmsNormSplitDBlockReduceSumFp32(
            data, elements, iterations);
        const int64_t cycle_end = AscendC::GetSystemCycle();
        cycles.SetValue(cycle_index, cycle_end - cycle_start);
        AscendC::GlobalTensor<float> validation;
        validation.SetGlobalBuffer(gm_validation);
        validation.SetValue(
            block_index * kValidationValuesPerBlock,
            data.GetValue(0));
        return;
    }

    if (opcode == kLocalUbCopyContiguous
            || opcode == kLocalUbCopyStride64
            || opcode == kLocalUbCopyStride128) {
        const uint32_t payload_bytes = static_cast<uint32_t>(element_count);
        const uint32_t block_bytes = (
            opcode == kLocalUbCopyContiguous
            ? payload_bytes
            : (opcode == kLocalUbCopyStride64 ? 64 : 128));
        if (payload_bytes % block_bytes != 0) {
            cycles.SetValue(cycle_index, -3);
            return;
        }
        AscendC::TPipe copy_pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> source_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> destination_buffer;
        copy_pipe.InitBuffer(source_buffer, kLocalCopySourceBytes);
        copy_pipe.InitBuffer(destination_buffer, kLocalCopyDestinationBytes);
        auto source = source_buffer.Get<float>();
        auto destination = destination_buffer.Get<float>();
        AscendC::Duplicate<float>(
            source, source_value, kLocalCopySourceBytes / sizeof(float));
        AscendC::PipeBarrier<PIPE_V>();
        const int64_t cycle_start = AscendC::GetSystemCycle();
        RunLocalUbCopyChain(
            destination, source, payload_bytes, block_bytes, iterations);
        const int64_t cycle_end = AscendC::GetSystemCycle();
        cycles.SetValue(cycle_index, cycle_end - cycle_start);
        AscendC::GlobalTensor<float> validation;
        validation.SetGlobalBuffer(gm_validation);
        validation.SetValue(
            block_index * kValidationValuesPerBlock,
            destination.GetValue(0));
        return;
    }

    if (opcode == kSingleTbufInit) {
        if (element_count != 1024 && element_count != 2048) {
            cycles.SetValue(cycle_index, -3);
            return;
        }
        const int64_t cycle_start = AscendC::GetSystemCycle();
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
        pipe.InitBuffer(buffer, static_cast<uint32_t>(element_count));
        const int64_t cycle_end = AscendC::GetSystemCycle();
        cycles.SetValue(cycle_index, cycle_end - cycle_start);
        AscendC::GlobalTensor<float> validation;
        validation.SetGlobalBuffer(gm_validation);
        validation.SetValue(
            block_index * kValidationValuesPerBlock, source_value);
        return;
    }

    if (opcode == kScalarU32DivRemDependency
            || opcode == kScalarU32MulAddDependency
            || opcode == kScalarU64MulDependency
            || opcode == kScalarU64DivRemDependency) {
        const uint32_t operand = static_cast<uint32_t>(element_count);
        uint64_t result;
        const int64_t cycle_start = AscendC::GetSystemCycle();
        if (opcode == kScalarU32DivRemDependency) {
            result = RunScalarU32DivRemDependency(
                0xf1234567U, operand, iterations);
        } else if (opcode == kScalarU32MulAddDependency) {
            result = RunScalarU32MulAddDependency(
                0xf1234567U, operand, iterations);
        } else if (opcode == kScalarU64MulDependency) {
            result = RunScalarU64MulDependency(
                0xfedcba9876543211ULL,
                2ULL * operand + 1ULL, iterations);
        } else {
            result = RunScalarU64DivRemDependency(
                0xfedcba9876543211ULL,
                2ULL * operand + 1ULL, iterations);
        }
        const int64_t cycle_end = AscendC::GetSystemCycle();
        cycles.SetValue(cycle_index, cycle_end - cycle_start);
        cycles.SetValue(
            cycle_index + 1,
            static_cast<int64_t>(result & 0x00ffffffULL));
        AscendC::GlobalTensor<float> validation;
        validation.SetGlobalBuffer(gm_validation);
        validation.SetValue(
            block_index * kValidationValuesPerBlock, source_value);
        return;
    }

    if (opcode == kTwoTbufMaskPrologue) {
        if (element_count != 2048 && element_count != 4096) {
            cycles.SetValue(cycle_index, -3);
            return;
        }
        const int64_t cycle_start = AscendC::GetSystemCycle();
        AscendC::TPipe prologue_pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> fp32_buffer;
        AscendC::TBuf<AscendC::TPosition::VECCALC> bf16_buffer;
        prologue_pipe.InitBuffer(fp32_buffer, 147456);
        prologue_pipe.InitBuffer(bf16_buffer, 49152);
        auto temporary = fp32_buffer.Get<float>();
        const uint32_t mask_elements = static_cast<uint32_t>(element_count);
        AscendC::Duplicate<float>(temporary, -1.0f, mask_elements);
        uint64_t mask[2] = {
            static_cast<uint64_t>(0xaaaaaaaaaaaaaaaa),
            static_cast<uint64_t>(0xaaaaaaaaaaaaaaaa),
        };
        AscendC::Duplicate<float>(
            temporary, 0.0f, mask,
            static_cast<uint8_t>(mask_elements / 64), 1, 8);
        AscendC::ResetMask();
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds<float>(
            temporary[mask_elements], temporary, 1.0f, mask_elements);
        AscendC::PipeBarrier<PIPE_V>();
        const int64_t cycle_end = AscendC::GetSystemCycle();
        cycles.SetValue(cycle_index, cycle_end - cycle_start);
        AscendC::GlobalTensor<float> validation;
        validation.SetGlobalBuffer(gm_validation);
        validation.SetValue(
            block_index * kValidationValuesPerBlock,
            temporary[mask_elements].GetValue(0));
        return;
    }

    const uint32_t elements = static_cast<uint32_t>(element_count);
    const size_t buffer_bytes = static_cast<size_t>(
        (element_count * sizeof(float) + 31) / 32 * 32);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> source_buffer0;
    AscendC::TBuf<AscendC::TPosition::VECIN> source_buffer1;
    AscendC::TBuf<AscendC::TPosition::VECOUT> destination_buffer0;
    AscendC::TBuf<AscendC::TPosition::VECOUT> destination_buffer1;
    AscendC::TBuf<AscendC::TPosition::VECIN> bf16_source_buffer;
    AscendC::TBuf<AscendC::TPosition::VECOUT> bf16_destination_buffer0;
    AscendC::TBuf<AscendC::TPosition::VECOUT> bf16_destination_buffer1;
    pipe.InitBuffer(source_buffer0, buffer_bytes);
    pipe.InitBuffer(source_buffer1, buffer_bytes);
    pipe.InitBuffer(destination_buffer0, buffer_bytes);
    pipe.InitBuffer(destination_buffer1, buffer_bytes);
    const size_t bf16_buffer_bytes = static_cast<size_t>(
        (element_count * sizeof(bfloat16_t) + 31) / 32 * 32);
    pipe.InitBuffer(bf16_source_buffer, bf16_buffer_bytes);
    pipe.InitBuffer(bf16_destination_buffer0, bf16_buffer_bytes);
    pipe.InitBuffer(bf16_destination_buffer1, bf16_buffer_bytes);
    auto source0 = source_buffer0.Get<float>();
    auto source1 = source_buffer1.Get<float>();
    auto destination0 = destination_buffer0.Get<float>();
    auto destination1 = destination_buffer1.Get<float>();
    auto bf16_source = bf16_source_buffer.Get<bfloat16_t>();
    auto bf16_destination0 = bf16_destination_buffer0.Get<bfloat16_t>();
    auto bf16_destination1 = bf16_destination_buffer1.Get<bfloat16_t>();

    AscendC::Duplicate<float>(source0, source_value, elements);
    AscendC::Duplicate<float>(source1, divisor_value, elements);
    AscendC::Duplicate<float>(destination0, 0.0f, elements);
    const float destination1_initial = (
        opcode == kNormSqrtSerialized
        || opcode == kNormSqrtDuplicateDivSerialized)
        ? source_value
        : (opcode == kNormDivSerialized ? divisor_value : 0.0f);
    AscendC::Duplicate<float>(destination1, destination1_initial, elements);
    // The AIV backend does not support C++ scalar casts to/from BF16.  Build
    // the BF16 source through the same LocalTensor Cast intrinsic used by the
    // production kernels; initialization stays outside the timed region.
    AscendC::Cast(
        bf16_source, source0, AscendC::RoundMode::CAST_RINT, elements);
    AscendC::PipeBarrier<PIPE_V>();

    float split_d_handoff_result = 0.0f;
    const int64_t cycle_start = AscendC::GetSystemCycle();
    if (opcode == kExp) {
        RunExp(destination0, destination1, source0, elements, iterations);
    } else if (opcode == kDiv) {
        RunDiv(
            destination0, destination1, source0, source1,
            elements, iterations);
    } else if (opcode == kExpDiv) {
        RunExpDiv(
            destination0, destination1, source0, source1,
            elements, iterations);
    } else if (opcode == kSqrt) {
        RunSqrt(destination0, destination1, source0, elements, iterations);
    } else if (opcode == kSqrtDiv) {
        RunSqrtDiv(
            destination0, destination1, source0, source1,
            elements, iterations);
    } else if (opcode == kExpSerialized) {
        RunExpSerialized(
            destination0, destination1, source0, elements, iterations);
    } else if (opcode == kDivSerialized) {
        RunDivSerialized(
            destination0, destination1, source0, source1,
            elements, iterations);
    } else if (opcode == kCastBf16ToFp32Serialized) {
        RunCastBf16ToFp32Serialized(
            destination0, destination1, bf16_source, elements, iterations);
    } else if (opcode == kCastFp32ToBf16Serialized) {
        RunCastFp32ToBf16Serialized(
            bf16_destination0, bf16_destination1, source0,
            elements, iterations);
    } else if (opcode == kMulsSerialized) {
        RunMulsSerialized(
            destination0, destination1, source0, divisor_value,
            elements, iterations);
    } else if (opcode == kAddsSerialized) {
        RunAddsSerialized(
            destination0, destination1, source0, divisor_value,
            elements, iterations);
    } else if (opcode == kMulSerialized) {
        RunMulSerialized(
            destination0, destination1, source0, source1,
            elements, iterations);
    } else if (opcode == kNormSqrtSerialized) {
        RunNormSqrtSerialized(destination1, elements, iterations);
    } else if (opcode == kNormDivSerialized) {
        RunNormDivSerialized(destination1, source0, elements, iterations);
    } else if (opcode == kNormSqrtDuplicateDivSerialized) {
        RunNormSqrtDuplicateDivSerialized(
            destination1, source1, elements, iterations);
    } else if (opcode == kRmsNormLocalScalarDependency) {
        RunRmsNormLocalScalarDependency(
            source0, destination1, iterations);
    } else if (opcode == kAddRmsNormLocalScalarHandoff) {
        RunAddRmsNormLocalScalarHandoff(
            source0, destination1, iterations);
    } else if (opcode == kAddRmsNormSplitDLocalScalarHandoff) {
        split_d_handoff_result = RunAddRmsNormSplitDLocalScalarHandoff(
            source0, iterations);
    } else if (opcode == kVectorToMte2Fence) {
        RunVectorToMte2Fence(iterations);
    } else if (opcode == kLocalScalarFillSv) {
        RunLocalScalarFillSv(
            destination1, elements, source_value, divisor_value, iterations);
    } else {
        cycles.SetValue(cycle_index, -2);
        return;
    }
    const int64_t cycle_end = AscendC::GetSystemCycle();
    cycles.SetValue(cycle_index, cycle_end - cycle_start);
    if (opcode == kAddRmsNormSplitDLocalScalarHandoff) {
        destination1.SetValue(0, split_d_handoff_result);
    }

    AscendC::GlobalTensor<float> validation;
    validation.SetGlobalBuffer(gm_validation);
    if (opcode == kCastFp32ToBf16Serialized) {
        AscendC::Cast(
            destination0, bf16_destination1,
            AscendC::RoundMode::CAST_NONE, elements);
        AscendC::PipeBarrier<PIPE_V>();
    }
    const float validation_value = (
        opcode == kCastFp32ToBf16Serialized
        ? destination0.GetValue(0) : destination1.GetValue(0));
    validation.SetValue(
        block_index * kValidationValuesPerBlock, validation_value);
}
