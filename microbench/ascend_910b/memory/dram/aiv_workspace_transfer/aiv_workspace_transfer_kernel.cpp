// Independent short AIV workspace-transfer chains from CANN 8.5.1 FIA.
//
// Vec1 mirrors the D64, M=4 DealBmm1ResBaseBlock byte geometry at aligned
// S2=512 and 1024. Vec2 mirrors the corresponding final-head state read and
// BF16 output write in DealBmm2ResBaseBlock, including explicit
// MTE3-to-MTE2 completion.
// No attention arithmetic, MatMul, cross-core synchronization, or production
// input is present in the timed region.
#include "kernel_operator.h"

#define AIV_WORKSPACE_TRANSFER_CONTRACT_AICORE
#include "aiv_workspace_transfer_contract.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

namespace {

using namespace aiv_workspace_transfer_contract;

__aicore__ inline void CompleteMte2BeforeConsumer() {
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
}

__aicore__ inline void CompleteMte3BeforeNextMte2() {
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

__aicore__ inline void CompleteVectorBeforeMte3() {
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
}

__aicore__ inline void Vec1Step(
        Mode mode, AscendC::LocalTensor<half>& local,
        AscendC::GlobalTensor<half>& source,
        AscendC::GlobalTensor<half>& destination) {
    int64_t load_elements = kVec1LoadBytes / sizeof(half);
    int64_t store_elements = kVec1StoreBytes / sizeof(half);
    if (mode == Mode::kVec1M4S512OrderedChain) {
        load_elements = kVec1M4S512LoadBytes / sizeof(half);
        store_elements = kVec1M4S512StoreBytes / sizeof(half);
    } else if (mode == Mode::kVec1M4S1024OrderedChain) {
        load_elements = kVec1M4S1024LoadBytes / sizeof(half);
        store_elements = kVec1M4S1024StoreBytes / sizeof(half);
    }
    if (mode == Mode::kVec1Load4096Isolated) {
        AscendC::DataCopy(local, source, load_elements);
        CompleteMte2BeforeConsumer();
    } else if (mode == Mode::kVec1Store2048Isolated) {
        AscendC::DataCopy(destination, local, store_elements);
        CompleteMte3BeforeNextMte2();
    } else if (mode == Mode::kVec1OrderedChain
               || mode == Mode::kVec1M4S512OrderedChain
               || mode == Mode::kVec1M4S1024OrderedChain) {
        AscendC::DataCopy(local, source, load_elements);
        CompleteMte2BeforeConsumer();
        AscendC::DataCopy(destination, local, store_elements);
        CompleteMte3BeforeNextMte2();
    }
}

__aicore__ inline void Vec2Step(
        Mode mode, AscendC::LocalTensor<float>& local,
        AscendC::GlobalTensor<float>& state) {
    int64_t read_elements = kVec2StateBytes / sizeof(float);
    int64_t store_elements = kVec2StateBytes / sizeof(float);
    if (mode == Mode::kVec2M4FinalOrderedChain) {
        read_elements = kVec2M4FinalReadBytes / sizeof(float);
        store_elements = kVec2M4FinalStoreBytes / sizeof(float);
    }
    if (mode == Mode::kVec2Load256Isolated) {
        AscendC::DataCopy(local, state, read_elements);
        CompleteMte2BeforeConsumer();
    } else if (mode == Mode::kVec2Store256Isolated) {
        AscendC::DataCopy(state, local, store_elements);
        CompleteMte3BeforeNextMte2();
    } else if (IsVec2Ordered(mode)) {
        // The production source places this dependency before its previous
        // workspace-state read.  Completing every preceding iteration's
        // store here preserves that exact MTE3 -> MTE2 ordering.
        CompleteMte3BeforeNextMte2();
        AscendC::DataCopy(local, state, read_elements);
        CompleteMte2BeforeConsumer();
        AscendC::DataCopy(state, local, store_elements);
    }
}

}  // namespace

extern "C" __global__ __aicore__ void aiv_workspace_transfer(
        __gm__ half* gm_vec1_source, __gm__ half* gm_vec1_destination,
        __gm__ float* gm_vec2_state, __gm__ int64_t* gm_cycles,
        __gm__ float* gm_validation, int64_t mode_value,
        int64_t iterations, int64_t validation_only) {
    using namespace aiv_workspace_transfer_contract;
    if ASCEND_IS_AIC {
        return;
    } else {
        const int64_t block_index = AscendC::GetBlockIdx();
        const int64_t lane = block_index % kAivPerAic;
        AscendC::GlobalTensor<int64_t> cycle_output;
        cycle_output.SetGlobalBuffer(
            gm_cycles + block_index * kCycleSlotElements);
        AscendC::GlobalTensor<float> validation_output;
        validation_output.SetGlobalBuffer(
            gm_validation + block_index * kValidationSlotElements);
        if (!IsSupportedMode(mode_value) || iterations <= 0) {
            cycle_output.SetValue(0, -1);
            cycle_output.SetValue(1, -1);
            validation_output.SetValue(0, -1.0f);
            return;
        }
        const Mode mode = static_cast<Mode>(mode_value);
        // For the target D64 task, mDealSize=4.  SetMSplitInfo assigns that
        // row to VEC0 and leaves VEC1 with vecDealM=0, so all measured Vec1
        // and Vec2 mechanisms execute on the primary (even) AIV only.
        const bool active = lane == 0;
        if (!active) {
            cycle_output.SetValue(0, 0);
            cycle_output.SetValue(1, 0);
            validation_output.SetValue(0, 0.0f);
            return;
        }

        const int64_t half_stride = kActorStrideBytes / sizeof(half);
        const int64_t float_stride = kActorStrideBytes / sizeof(float);
        AscendC::GlobalTensor<half> vec1_source;
        AscendC::GlobalTensor<half> vec1_destination;
        AscendC::GlobalTensor<float> vec2_state;
        vec1_source.SetGlobalBuffer(
            gm_vec1_source + block_index * half_stride);
        vec1_destination.SetGlobalBuffer(
            gm_vec1_destination + block_index * half_stride);
        vec2_state.SetGlobalBuffer(
            gm_vec2_state + block_index * float_stride);

        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECIN> vec1_buffer;
        AscendC::TBuf<AscendC::TPosition::VECIN> vec2_buffer;
        pipe.InitBuffer(vec1_buffer, kVec1M4S1024LoadBytes);
        pipe.InitBuffer(vec2_buffer, kVec2M4FinalReadBytes);
        auto vec1_local = vec1_buffer.Get<half>();
        auto vec2_local = vec2_buffer.Get<float>();
        AscendC::Duplicate<half>(
            vec1_local, static_cast<half>(kLocalSentinel),
            kVec1M4S1024LoadBytes / sizeof(half));
        AscendC::Duplicate<float>(
            vec2_local, kLocalSentinel,
            kVec2M4FinalReadBytes / sizeof(float));
        AscendC::PipeBarrier<PIPE_V>();

        if (validation_only != 0) {
            float loaded_value = 0.0f;
            if (IsVec2Ordered(mode)) {
                const int64_t read_elements = (
                    mode == Mode::kVec2M4FinalOrderedChain
                    ? kVec2M4FinalReadBytes : kVec2StateBytes)
                    / sizeof(float);
                const int64_t store_elements = (
                    mode == Mode::kVec2M4FinalOrderedChain
                    ? kVec2M4FinalStoreBytes : kVec2StateBytes)
                    / sizeof(float);
                AscendC::DataCopy(
                    vec2_local, vec2_state, read_elements);
                CompleteMte2BeforeConsumer();
                loaded_value = vec2_local.GetValue(0);
                AscendC::Duplicate<float>(
                    vec2_local, kVec2ChainStoreSentinel,
                    store_elements);
                CompleteVectorBeforeMte3();
                AscendC::DataCopy(
                    vec2_state, vec2_local, store_elements);
                CompleteMte3BeforeNextMte2();
            } else if (UsesVec1(mode)) {
                Vec1Step(mode, vec1_local, vec1_source, vec1_destination);
                loaded_value = static_cast<float>(vec1_local.GetValue(0));
            } else {
                Vec2Step(mode, vec2_local, vec2_state);
                loaded_value = vec2_local.GetValue(0);
            }
            cycle_output.SetValue(0, 1);
            validation_output.SetValue(0, loaded_value);
            validation_output.SetValue(
                1, IsVec2Ordered(mode)
                    ? kVec2ChainStoreSentinel : loaded_value);
            AscendC::DataCacheCleanAndInvalid<
                int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(cycle_output);
            AscendC::DataCacheCleanAndInvalid<
                float, AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(validation_output);
            return;
        }

        #pragma unroll 1
        for (int64_t warmup = 0; warmup < kWarmupIterations; ++warmup) {
            if (UsesVec1(mode)) {
                Vec1Step(mode, vec1_local, vec1_source, vec1_destination);
            } else {
                Vec2Step(mode, vec2_local, vec2_state);
            }
        }
        // Measure N and 4N back-to-back inside this persistent invocation.
        // Both samples therefore belong to this exact physical AIV actor;
        // the host must difference each actor pair before aggregating actors.
        const int64_t short_start = AscendC::GetSystemCycle();
        #pragma unroll 1
        for (int64_t iteration = 0; iteration < iterations; ++iteration) {
            if (UsesVec1(mode)) {
                Vec1Step(mode, vec1_local, vec1_source, vec1_destination);
            } else {
                Vec2Step(mode, vec2_local, vec2_state);
            }
        }
        if (IsVec2Ordered(mode)) {
            CompleteMte3BeforeNextMte2();
        }
        const int64_t short_end = AscendC::GetSystemCycle();

        const int64_t long_start = AscendC::GetSystemCycle();
        #pragma unroll 1
        for (int64_t iteration = 0; iteration < 4 * iterations; ++iteration) {
            if (UsesVec1(mode)) {
                Vec1Step(mode, vec1_local, vec1_source, vec1_destination);
            } else {
                Vec2Step(mode, vec2_local, vec2_state);
            }
        }
        if (IsVec2Ordered(mode)) {
            CompleteMte3BeforeNextMte2();
        }
        const int64_t long_end = AscendC::GetSystemCycle();
        cycle_output.SetValue(0, short_end - short_start);
        cycle_output.SetValue(1, long_end - long_start);
        AscendC::DataCacheCleanAndInvalid<
            int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(cycle_output);

        const float value = UsesVec1(mode)
            ? static_cast<float>(vec1_local.GetValue(0))
            : vec2_local.GetValue(0);
        validation_output.SetValue(0, value);
        AscendC::DataCacheCleanAndInvalid<
            float, AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(validation_output);
    }
}
