// Independent resident MatMulV3 key-65632 base-tile service probe.
//
// The timed loop keeps the source-visible local Matmul API and L0C handoff
// lifecycle for one BF16 128x128x384 base tile.  GM-to-L1 population is
// completed before timing; each iteration writes one FP32 workspace tile,
// but the probe omits the production outer loops and AIV output conversion.
#include "kernel_operator.h"
#include "adv_api/matmul_intf.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

using namespace AscendC;
using namespace matmul;

constexpr int32_t kTileM = 128;
constexpr int32_t kTileN = 128;
constexpr int32_t kMaximumTileK = 384;
constexpr int32_t kBaseK = 128;
constexpr int32_t kWorkerCount = 24;
constexpr int32_t kL1Bytes = 512 * 1024;
constexpr int32_t kL1SlotElements = 48 * 1024;
constexpr int32_t kL0CBytes = 64 * 1024;
constexpr int32_t kCycleSlotElements = 8;

using AL1 = MatmulType<TPosition::A1, CubeFormat::NZ, bfloat16_t, false>;
using BL1 = MatmulType<TPosition::B1, CubeFormat::NZ, bfloat16_t, false>;
using CL0 = MatmulType<TPosition::CO1, CubeFormat::NZ, float>;
using Bias = MatmulType<TPosition::GM, CubeFormat::ND, float>;

constexpr static MatmulConfig kMdlConfig = GetMDLConfig();

constexpr tiling::TCubeTiling kTiling = []() {
    tiling::TCubeTiling result{};
    result.usedCoreNum = kWorkerCount;
    result.M = kTileM;
    result.N = kTileN;
    result.Ka = kMaximumTileK;
    result.Kb = kMaximumTileK;
    result.singleCoreM = kTileM;
    result.singleCoreN = kTileN;
    result.singleCoreK = kMaximumTileK;
    result.baseM = kTileM;
    result.baseN = kTileN;
    result.baseK = kBaseK;
    result.depthA1 = 9;
    result.depthB1 = 6;
    result.stepM = 3;
    result.stepN = 1;
    result.iterateOrder = 1;
    result.batchM = 1;
    result.batchN = 1;
    result.singleBatchM = 1;
    result.singleBatchN = 1;
    result.stepKa = 3;
    result.stepKb = 3;
    result.dbL0A = 2;
    result.dbL0B = 2;
    result.dbL0C = 2;
    return result;
}();

__aicore__ inline void CopyNd2Nz(
        LocalTensor<bfloat16_t>& destination,
        GlobalTensor<bfloat16_t>& source,
        int32_t rows, int32_t columns) {
    Nd2NzParams parameters;
    parameters.ndNum = 1;
    parameters.nValue = static_cast<uint16_t>(rows);
    parameters.dValue = static_cast<uint16_t>(columns);
    parameters.srcNdMatrixStride = 0;
    parameters.srcDValue = static_cast<uint16_t>(columns);
    parameters.dstNzC0Stride = static_cast<uint16_t>(rows);
    parameters.dstNzNStride = 1;
    parameters.dstNzMatrixStride = 0;
    DataCopy(destination, source, parameters);
}

}  // namespace

extern "C" __global__ __aicore__ void single_core_split_k_tile(
        __gm__ bfloat16_t* gm_a, __gm__ bfloat16_t* gm_b,
        __gm__ float* gm_workspace, __gm__ int64_t* gm_cycles,
        int64_t tile_k, int64_t iterations) {
    using namespace AscendC;
    using namespace matmul;

    if ((tile_k != 256 && tile_k != kMaximumTileK)
            || iterations <= 0 || GetBlockNum() != kWorkerCount) {
        return;
    }

    TPipe pipe;
    TBuf<TPosition::A1> l1_buffer;
    TQue<TPosition::CO1, 1> l0c_ping_buffer;
    TQue<TPosition::CO1, 1> l0c_pong_buffer;
    pipe.InitBuffer(l1_buffer, kL1Bytes);
    pipe.InitBuffer(l0c_ping_buffer, 1, kL0CBytes);
    pipe.InitBuffer(l0c_pong_buffer, 1, kL0CBytes);

    auto b_l1 = l1_buffer.Get<bfloat16_t>();
    auto a_l1 = l1_buffer.Get<bfloat16_t>()[2 * kL1SlotElements];
    auto l0c_ping = l0c_ping_buffer.AllocTensor<float>();
    auto l0c_pong = l0c_pong_buffer.AllocTensor<float>();

    GlobalTensor<bfloat16_t> a_global;
    GlobalTensor<bfloat16_t> b_global;
    a_global.SetGlobalBuffer(gm_a);
    b_global.SetGlobalBuffer(gm_b);
    CopyNd2Nz(a_l1, a_global, kTileM, tile_k);
    CopyNd2Nz(b_l1, b_global, tile_k, kTileN);
    PipeBarrier<PIPE_MTE2>();
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

    MatmulImpl<AL1, BL1, CL0, Bias, kMdlConfig> mm;
    mm.SetSubBlockIdx(0);
    PRELOAD(4);
    mm.Init(&kTiling, &pipe);
    mm.SetOrgShape(
        kTileM, kTileN, kMaximumTileK, kMaximumTileK, kTileN);
    const event_t ping_event = static_cast<event_t>(
        GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
    const event_t pong_event = static_cast<event_t>(
        GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
    SetFlag<HardEvent::FIX_M>(ping_event);
    SetFlag<HardEvent::FIX_M>(pong_event);

    const int64_t begin = GetSystemCycle();
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        const event_t event = (iteration & 1) == 0
            ? ping_event : pong_event;
        auto destination = (iteration & 1) == 0 ? l0c_ping : l0c_pong;
        WaitFlag<HardEvent::FIX_M>(event);
        mm.SetSingleShape(kTileM, kTileN, tile_k);
        mm.SetTensorA(a_l1, false);
        mm.SetTensorB(b_l1, false);
        mm.Iterate(false, destination);
        SetFlag<HardEvent::M_FIX>(event);
        WaitFlag<HardEvent::M_FIX>(event);
        GlobalTensor<float> workspace;
        workspace.SetGlobalBuffer(
            gm_workspace + GetBlockIdx() * kTileM * kTileN);
        FixpipeParamsV220 parameters;
        parameters.mSize = kTileM;
        parameters.nSize = kTileN;
        parameters.srcStride = kTileM;
        parameters.dstStride = kTileN;
        parameters.ndNum = 1;
        Fixpipe<float, float, CFG_ROW_MAJOR>(
            workspace, destination, parameters);
        SetFlag<HardEvent::FIX_M>(event);
        mm.ClearBias();
        mm.End();
    }
    const int64_t end = GetSystemCycle();
    WaitFlag<HardEvent::FIX_M>(ping_event);
    WaitFlag<HardEvent::FIX_M>(pong_event);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(ping_event);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(pong_event);

    GlobalTensor<int64_t> cycles;
    cycles.SetGlobalBuffer(
        gm_cycles + GetBlockIdx() * kCycleSlotElements);
    cycles.SetValue(0, end - begin);
    cycles.SetValue(1, iterations);
}
