// MTE2 producer queue probe for CANN deterministic Split-K's L2-split body.
//
// This is not a GEMM. It preserves the source's GM-to-A1/B1 descriptor
// topology and consumes each queued operand through MTE1 into L0 before
// releasing its A1/B1 slot. There is deliberately no Cube, FixPipe,
// workspace, AIV reduction, output tile, or production-workload duration.
// The MTE1 consumer is needed because the previous MTE2-only version reused
// producer storage without the source's consumer lifetime and deadlocked.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kElementBytes = sizeof(bfloat16_t);
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kAicWorkers = 24;
constexpr int64_t kFullChunkWorkers = 3;
constexpr int64_t kNSlabs = 2;
constexpr int64_t kNTilesPerSlab = 32;
constexpr int64_t kKChunks = 75;
constexpr int64_t kKChunk = 384;
constexpr int64_t kKTail = 256;
constexpr int64_t kPhysicalK = 28672;
constexpr int64_t kPhysicalN = 8192;
constexpr int64_t kMaximumATailRows = 128;
constexpr int64_t kBRows = 128;
constexpr int64_t kBaseK = 128;
constexpr int64_t kBaseN = 128;
constexpr uint64_t kMStepBit = 16;

static_assert(kFullChunkWorkers * 4 + (kAicWorkers - kFullChunkWorkers) * 3
              == kKChunks);
static_assert(kNSlabs * kNTilesPerSlab * kBRows == kPhysicalN);
static_assert(kKChunk % kBaseK == 0);

__aicore__ inline int64_t Align16(int64_t value) {
    return (value + 15) / 16 * 16;
}

__aicore__ inline void CopyNd2Nz(
        AscendC::LocalTensor<bfloat16_t>& destination,
        AscendC::GlobalTensor<bfloat16_t>& source, int64_t rows,
        int64_t columns, int64_t source_row_stride) {
    AscendC::Nd2NzParams parameters;
    parameters.ndNum = 1;
    parameters.nValue = static_cast<uint16_t>(rows);
    parameters.dValue = static_cast<uint16_t>(columns);
    parameters.srcNdMatrixStride = 0;
    parameters.srcDValue = static_cast<uint16_t>(source_row_stride);
    parameters.dstNzC0Stride = static_cast<uint16_t>(Align16(rows));
    parameters.dstNzNStride = 1;
    parameters.dstNzMatrixStride = 0;
    AscendC::DataCopy(destination, source, parameters);
}

// These are the source's non-transposed A1->A2 and B1->B2 base-tile loads.
// The timed consumer intentionally stops at the MTE1 completion boundary.
__aicore__ inline void LoadA(
        AscendC::LocalTensor<bfloat16_t>& destination,
        const AscendC::LocalTensor<bfloat16_t>& source, int64_t a_rows) {
    AscendC::LoadData3DParamsV2Pro parameters;
    parameters.channelSize = kBaseK;
    parameters.extConfig = (static_cast<uint64_t>(a_rows) << kMStepBit)
        | static_cast<uint64_t>(kBaseK);
    AscendC::LoadData(destination, source, parameters);
}

__aicore__ inline void LoadB(
        AscendC::LocalTensor<bfloat16_t>& destination,
        const AscendC::LocalTensor<bfloat16_t>& source) {
    AscendC::LoadData3DParamsV2Pro parameters;
    parameters.channelSize = kBaseN;
    parameters.extConfig = (static_cast<uint64_t>(kBaseK) << kMStepBit)
        | static_cast<uint64_t>(kBaseN);
    parameters.fMatrixCtrl = true;
    AscendC::LoadData(destination, source, parameters);
}

__aicore__ inline void ConsumeBaseKPair(
        AscendC::TQue<AscendC::TPosition::A2, 2>& l0a_queue,
        AscendC::TQue<AscendC::TPosition::B2, 2>& l0b_queue,
        const AscendC::LocalTensor<bfloat16_t>& a_local,
        const AscendC::LocalTensor<bfloat16_t>& b_local, int64_t a_rows) {
    auto l0a = l0a_queue.AllocTensor<bfloat16_t>();
    auto l0b = l0b_queue.AllocTensor<bfloat16_t>();
    LoadA(l0a, a_local, a_rows);
    LoadB(l0b, b_local);
    l0a_queue.EnQue(l0a);
    l0b_queue.EnQue(l0b);
    auto completed_l0a = l0a_queue.DeQue<bfloat16_t>();
    auto completed_l0b = l0b_queue.DeQue<bfloat16_t>();
    l0a_queue.FreeTensor(completed_l0a);
    l0b_queue.FreeTensor(completed_l0b);
}

__aicore__ inline int64_t ChunkStart(int64_t block) {
    if (block < kFullChunkWorkers) {
        return block * 4;
    }
    return block * 3 + kFullChunkWorkers;
}

__aicore__ inline int64_t ChunkCount(int64_t block) {
    return block < kFullChunkWorkers ? 4 : 3;
}

}  // namespace

extern "C" __global__ __aicore__ void aic_mte2_deterministic_split_k(
        __gm__ bfloat16_t* gm_a_source, __gm__ bfloat16_t* gm_b_source,
        __gm__ int64_t* gm_output, int64_t a_rows) {
    const int64_t block = AscendC::GetBlockIdx();
    const int64_t chunk_start = ChunkStart(block);
    const int64_t chunk_count = ChunkCount(block);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> a_queue;
    AscendC::TQue<AscendC::TPosition::B1, 2> b_queue;
    AscendC::TQue<AscendC::TPosition::A2, 2> l0a_queue;
    AscendC::TQue<AscendC::TPosition::B2, 2> l0b_queue;
    pipe.InitBuffer(
        a_queue, 1, kMaximumATailRows * kKChunk * kElementBytes);
    pipe.InitBuffer(b_queue, 2, kBRows * kKChunk * kElementBytes);
    pipe.InitBuffer(
        l0a_queue, 2, kMaximumATailRows * kBaseK * kElementBytes);
    pipe.InitBuffer(l0b_queue, 2, kBaseK * kBaseN * kElementBytes);
    AscendC::GlobalTensor<bfloat16_t> a_source;
    AscendC::GlobalTensor<bfloat16_t> b_source;

    // Mc2MatMulMultiCoreSplitKDivideL2cache initializes four commands before
    // issuing the two N slabs. Do not add a per-descriptor MTE2 barrier.
    PRELOAD(4);
    const int64_t start = AscendC::GetSystemCycle();
    int64_t descriptor_count = 0;
    int64_t base_pair_count = 0;
    #pragma unroll 1
    for (int64_t n_slab = 0; n_slab < kNSlabs; ++n_slab) {
        #pragma unroll 1
        for (int64_t local_chunk = 0;
                local_chunk < chunk_count; ++local_chunk) {
            const int64_t k_chunk_index = chunk_start + local_chunk;
            const int64_t k_offset = k_chunk_index * kKChunk;
            const int64_t k_extent = (
                k_chunk_index == kKChunks - 1 ? kKTail : kKChunk);
            a_source.SetGlobalBuffer(gm_a_source + k_offset);
            auto a_producer = a_queue.AllocTensor<bfloat16_t>();
            CopyNd2Nz(
                a_producer, a_source, a_rows, k_extent, kPhysicalK);
            a_queue.EnQue(a_producer);
            auto a_consumer = a_queue.DeQue<bfloat16_t>();
            ++descriptor_count;
            #pragma unroll 1
            for (int64_t n_tile = 0; n_tile < kNTilesPerSlab; ++n_tile) {
                const int64_t physical_n_tile = (
                    n_slab * kNTilesPerSlab + n_tile);
                b_source.SetGlobalBuffer(
                    gm_b_source + physical_n_tile * kBRows * kPhysicalK
                    + k_offset);
                auto b_producer = b_queue.AllocTensor<bfloat16_t>();
                CopyNd2Nz(
                    b_producer, b_source, kBRows, k_extent, kPhysicalK);
                b_queue.EnQue(b_producer);
                auto b_consumer = b_queue.DeQue<bfloat16_t>();
                ++descriptor_count;
                #pragma unroll 1
                for (int64_t base_pair = 0;
                        base_pair < k_extent / kBaseK; ++base_pair) {
                    ConsumeBaseKPair(
                        l0a_queue, l0b_queue, a_consumer, b_consumer, a_rows);
                    ++base_pair_count;
                }
                b_queue.FreeTensor(b_consumer);
            }
            a_queue.FreeTensor(a_consumer);
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t end = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(
        gm_output + block * kCycleSlotBytes / sizeof(int64_t));
    output.SetValue(0, end - start);
    output.SetValue(1, descriptor_count);
    output.SetValue(2, chunk_start);
    output.SetValue(3, chunk_count);
    output.SetValue(4, base_pair_count);
    output.SetValue(5, a_rows);
}
