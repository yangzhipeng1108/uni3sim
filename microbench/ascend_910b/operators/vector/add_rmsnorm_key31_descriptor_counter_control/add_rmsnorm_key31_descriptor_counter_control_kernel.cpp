// Non-production CANN 8.5.1 control for the key-31 DataCopyCustom paths.
//
// It intentionally contains no AddRMSNorm arithmetic, reduction, output
// scale, full source loop, or timing instrumentation.  It compares only the
// fixed source's BF16 GM->UB queue geometry with its FP32 UB->GM rstd copy;
// msprof counters, never an elapsed duration, are the intended observation.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr uint32_t kBf16Elements = 8192;
constexpr uint32_t kRstdElements = 64;
constexpr uint32_t kBf16Bytes = kBf16Elements * sizeof(bfloat16_t);
constexpr uint32_t kRstdBytes = kRstdElements * sizeof(float);
constexpr uint32_t kStatusStride = 8;

enum class Mode : uint32_t {
    kGammaCopy = 0,
    kInputXCopy = 1,
    kGammaQueueControl = 2,
    kRstdCopy = 3,
    kRstdQueueControl = 4,
};

__aicore__ inline void CopyIn(
        AscendC::LocalTensor<bfloat16_t> dst,
        AscendC::GlobalTensor<bfloat16_t> src) {
    AscendC::DataCopyParams params;
    params.blockLen = kBf16Bytes;
    params.blockCount = 1;
    AscendC::DataCopyPadParams pad;
    // This is the AICORE-220 DataCopyCustom<T>(LocalTensor, GlobalTensor)
    // body in the fixed CANN RMSNorm base: one MTE2 DataCopyPad call.
    AscendC::DataCopyPad(dst, src, params, pad);
}

__aicore__ inline void CopyOut(
        AscendC::GlobalTensor<float> dst,
        AscendC::LocalTensor<float> src) {
    AscendC::DataCopyParams params;
    params.blockLen = kRstdBytes;
    params.blockCount = 1;
    // This is the AICORE-220 DataCopyCustom<float>(GlobalTensor, LocalTensor)
    // body in the fixed CANN RMSNorm base: one MTE3 DataCopyPad call.
    AscendC::DataCopyPad(dst, src, params);
}

__aicore__ inline uint64_t GammaLike(
        AscendC::TPipe& pipe, AscendC::GlobalTensor<bfloat16_t> source,
        uint32_t repeats, bool issue_copy) {
    // CANN 8.5.1 AddRmsNormSplitD fixes BUFFER_NUM to one.
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> gamma_queue;
    pipe.InitBuffer(gamma_queue, 1, kBf16Bytes);
    uint64_t checksum = 0;
    for (uint32_t index = 0; index < repeats; ++index) {
        auto produced = gamma_queue.AllocTensor<bfloat16_t>();
        if (issue_copy) {
            CopyIn(produced, source);
        } else {
            AscendC::Duplicate(
                produced, static_cast<bfloat16_t>(0.5f), kBf16Elements);
            AscendC::PipeBarrier<PIPE_V>();
        }
        gamma_queue.EnQue(produced);
        auto consumed = gamma_queue.DeQue<bfloat16_t>();
        // The queue lifecycle is the observable source effect.  The copy
        // intrinsic is side-effecting; a scalar BF16 read would add a V->S
        // mechanism outside the DMA/queue control being isolated here.
        checksum += 1;
        gamma_queue.FreeTensor(consumed);
    }
    return checksum;
}

__aicore__ inline uint64_t XLike(
        AscendC::TPipe& pipe, AscendC::GlobalTensor<bfloat16_t> source,
        uint32_t repeats) {
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_queue;
    // Exact key-31 x1/x2 allocation: one queue tensor holds adjacent tiles.
    pipe.InitBuffer(input_queue, 1, 2 * kBf16Bytes);
    uint64_t checksum = 0;
    for (uint32_t index = 0; index < repeats; ++index) {
        auto produced = input_queue.AllocTensor<bfloat16_t>();
        CopyIn(produced, source);
        input_queue.EnQue(produced);
        auto consumed = input_queue.DeQue<bfloat16_t>();
        checksum += 1;
        input_queue.FreeTensor(consumed);
    }
    return checksum;
}

__aicore__ inline uint64_t RstdLike(
        AscendC::TPipe& pipe, AscendC::GlobalTensor<float> sink,
        uint32_t repeats, bool issue_copy) {
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> rstd_queue;
    pipe.InitBuffer(rstd_queue, 1, kRstdBytes);
    uint64_t checksum = 0;
    for (uint32_t index = 0; index < repeats; ++index) {
        auto produced = rstd_queue.AllocTensor<float>();
        AscendC::Duplicate(produced, 1.0f, kRstdElements);
        AscendC::PipeBarrier<PIPE_V>();
        rstd_queue.EnQue(produced);
        auto consumed = rstd_queue.DeQue<float>();
        if (issue_copy) {
            CopyOut(sink, consumed);
        }
        checksum += 1;
        rstd_queue.FreeTensor(consumed);
    }
    return checksum;
}

}  // namespace

extern "C" __global__ __aicore__ void add_rmsnorm_key31_descriptor_counter_control(
        __gm__ bfloat16_t* source, __gm__ float* sink,
        __gm__ uint64_t* status, uint32_t mode, uint32_t repeats) {
    if (repeats == 0 || mode > static_cast<uint32_t>(Mode::kRstdQueueControl)) {
        return;
    }
    const uint32_t worker = AscendC::GetBlockIdx();
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<bfloat16_t> input;
    input.SetGlobalBuffer(source + worker * kBf16Elements);
    AscendC::GlobalTensor<float> output;
    output.SetGlobalBuffer(sink + worker * kRstdElements);
    uint64_t checksum = 0;
    const Mode selected = static_cast<Mode>(mode);
    if (selected == Mode::kGammaCopy) {
        checksum = GammaLike(pipe, input, repeats, true);
    } else if (selected == Mode::kInputXCopy) {
        checksum = XLike(pipe, input, repeats);
    } else if (selected == Mode::kGammaQueueControl) {
        checksum = GammaLike(pipe, input, repeats, false);
    } else if (selected == Mode::kRstdCopy) {
        checksum = RstdLike(pipe, output, repeats, true);
    } else if (selected == Mode::kRstdQueueControl) {
        checksum = RstdLike(pipe, output, repeats, false);
    } else {
        return;
    }
    AscendC::GlobalTensor<uint64_t> result;
    result.SetGlobalBuffer(status + worker * kStatusStride);
    result.SetValue(0, checksum);
    result.SetValue(1, static_cast<uint64_t>(mode));
    result.SetValue(2, static_cast<uint64_t>(repeats));
    result.SetValue(3, static_cast<uint64_t>(worker));
    AscendC::DataCacheCleanAndInvalid<
        uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}
