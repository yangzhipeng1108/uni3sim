// Isolate the CANN 8.5.1 elementwise scheduler's source-control lifecycle.
//
// This probe deliberately omits Add/Mul tensor payloads.  It preserves the
// source-shaped seven-node traversal, twelve logical buffer lookups and, in
// the full variant, the exact per-tile MTE2/V/MTE3 mutex lifecycle.  All
// variants share the same outer loop, timer and terminal PIPE_ALL fence.
#include "kernel_operator.h"

#define ELEMENTWISE_CONTROL_CONTRACT_AICORE
#include "elementwise_control_contract.h"
#undef ELEMENTWISE_CONTROL_CONTRACT_AICORE

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

using elementwise_control_contract::Variant;

__aicore__ inline uint64_t Mix(
        uint64_t checksum, uint64_t node, uint64_t value) {
    checksum ^= value + node * 0x9e3779b97f4a7c15ULL;
    return checksum * 0xbf58476d1ce4e5b9ULL + 0x94d049bb133111ebULL;
}

__aicore__ inline uint64_t LookupBuffer(
        AscendC::LocalTensor<uint8_t>& pool, uint8_t buffer_id) {
    AscendC::LocalTensor<uint8_t> view = pool[
        static_cast<int64_t>(buffer_id)
        * elementwise_control_contract::kBufferBytes];
    view.SetBufferLen(elementwise_control_contract::kBufferBytes);
    return reinterpret_cast<uint64_t>(view.GetPhyAddr());
}

template <bool WithMutex, pipe_t Pipe>
__aicore__ inline uint64_t Visit(
        AscendC::LocalTensor<uint8_t>& pool, uint8_t buffer_id,
        uint64_t checksum, uint64_t node) {
    if constexpr (WithMutex) {
        AscendC::GetBuffImpl<Pipe, false>(buffer_id);
    }
    checksum = Mix(checksum, node, LookupBuffer(pool, buffer_id));
    if constexpr (WithMutex) {
        AscendC::ReleaseBuffImpl<Pipe, false>(buffer_id);
    }
    return checksum;
}

template <bool WithMutex>
__aicore__ inline uint64_t TraverseDag(
        AscendC::LocalTensor<uint8_t>& pool, int64_t tile,
        uint64_t checksum) {
    const uint8_t pong = static_cast<uint8_t>(
        (tile & 1) * elementwise_control_contract::kPingPongOffset);
    const uint8_t x1 = static_cast<uint8_t>(pong + 0);
    const uint8_t x2 = static_cast<uint8_t>(pong + 1);
    const uint8_t x1_fp32 = static_cast<uint8_t>(pong + 2);
    const uint8_t x2_fp32 = static_cast<uint8_t>(pong + 3);
    const uint8_t result_fp32 = static_cast<uint8_t>(pong + 4);
    const uint8_t result_bf16 = static_cast<uint8_t>(pong + 5);

    // copy_in_x1, copy_in_x2: two MTE2 output buffer lifecycles.
    checksum = Visit<WithMutex, PIPE_MTE2>(pool, x1, checksum, 1);
    checksum = Visit<WithMutex, PIPE_MTE2>(pool, x2, checksum, 2);
    // Two casts, binary op, and result cast: nine Vector lifecycles.
    checksum = Visit<WithMutex, PIPE_V>(pool, x1, checksum, 3);
    checksum = Visit<WithMutex, PIPE_V>(pool, x1_fp32, checksum, 3);
    checksum = Visit<WithMutex, PIPE_V>(pool, x2, checksum, 4);
    checksum = Visit<WithMutex, PIPE_V>(pool, x2_fp32, checksum, 4);
    checksum = Visit<WithMutex, PIPE_V>(pool, x1_fp32, checksum, 5);
    checksum = Visit<WithMutex, PIPE_V>(pool, x2_fp32, checksum, 5);
    checksum = Visit<WithMutex, PIPE_V>(pool, result_fp32, checksum, 5);
    checksum = Visit<WithMutex, PIPE_V>(pool, result_fp32, checksum, 6);
    checksum = Visit<WithMutex, PIPE_V>(pool, result_bf16, checksum, 6);
    // copy_out_y: one MTE3 input buffer lifecycle.
    return Visit<WithMutex, PIPE_MTE3>(
        pool, result_bf16, checksum, 7);
}

__aicore__ inline void StoreResult(
        AscendC::GlobalTensor<int64_t>& result,
        int64_t cycles, uint64_t checksum) {
    result.SetValue(0, cycles);
    result.SetValue(1, static_cast<int64_t>(checksum));
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}

}  // namespace

extern "C" __global__ __aicore__ void elementwise_control(
        __gm__ int64_t* gm_results, int64_t tile_count,
        int64_t variant_value) {
    const int64_t worker = AscendC::GetBlockIdx();
    AscendC::GlobalTensor<int64_t> result;
    result.SetGlobalBuffer(
        gm_results
        + worker * elementwise_control_contract::kResultSlotElements);
    if (worker < 0 || worker >= elementwise_control_contract::kAivWorkers
            || !elementwise_control_contract::IsFrozenTileCount(tile_count)
            || !elementwise_control_contract::IsFrozenVariant(variant_value)) {
        StoreResult(result, -1, 0);
        return;
    }

    const Variant variant = static_cast<Variant>(variant_value);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    uint64_t checksum = static_cast<uint64_t>(worker + 1);
    const int64_t start = AscendC::GetSystemCycle();
    if (variant == Variant::kControlBaseline) {
        for (int64_t tile = 0; tile < tile_count; ++tile) {
            checksum = Mix(checksum, 0, static_cast<uint64_t>(tile & 1));
        }
    } else {
        pipe.InitBuffer(buffer, elementwise_control_contract::kPoolBytes);
        AscendC::LocalTensor<uint8_t> pool = buffer.Get<uint8_t>();
        if (variant == Variant::kDagTbuf) {
            for (int64_t tile = 0; tile < tile_count; ++tile) {
                checksum = TraverseDag<false>(pool, tile, checksum);
            }
        } else {
            for (int64_t tile = 0; tile < tile_count; ++tile) {
                checksum = TraverseDag<true>(pool, tile, checksum);
            }
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t end = AscendC::GetSystemCycle();
    StoreResult(result, end - start, checksum);
}
