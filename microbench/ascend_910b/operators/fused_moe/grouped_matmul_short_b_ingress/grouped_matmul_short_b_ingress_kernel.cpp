// Independent CANN 8.5.1 short-GroupedMatmul B-ingress source probe.
//
// The kernel deliberately retains only the BF16 B-side source path: group
// traversal, AIC-ring tile assignment, the two K=256 B1 fills per M64/N256
// output tile, B1 ping-pong and MTE2 completion.  It does not invoke Matmul,
// routing, vector work, output stores or a production FusedMoE operator.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

using namespace AscendC;

constexpr int64_t kAicCount = 24;
constexpr int64_t kMaximumGroups = 512;
constexpr int64_t kBaseM = 64;
constexpr int64_t kBaseN = 256;
constexpr int64_t kProjectionK = 512;
constexpr int64_t kProjectionN = 2048;
constexpr int64_t kNTiles = kProjectionN / kBaseN;
constexpr int64_t kBGroupK = 256;
constexpr int64_t kBGroupsPerTile = kProjectionK / kBGroupK;
constexpr int64_t kB1Elements = kBGroupK * kBaseN;
constexpr int64_t kResultStrideElements = 8;

static_assert(kProjectionN % kBaseN == 0);
static_assert(kProjectionK % kBGroupK == 0);

__aicore__ inline int64_t CeilDiv(int64_t value, int64_t divisor) {
    return (value + divisor - 1) / divisor;
}

__aicore__ inline void CopyB1OnTheFly(
        LocalTensor<bfloat16_t>& destination,
        GlobalTensor<bfloat16_t>& source) {
    // CANN's BF16-NZ B1 descriptor consists of sixteen C0-wide copies.  The
    // source matrix is N-major with the short program's K=512 row stride.
    constexpr int64_t kC0 = 16;
    constexpr int64_t kCopyWidths = kBaseN / kC0;
    constexpr int64_t kDestinationWidthStride =
        (kBGroupK / kC0) * kC0 * kC0;
    constexpr int64_t kSourceGap =
        kProjectionK * sizeof(bfloat16_t) / 32 - 1;
    DataCopyEnhancedParams enhanced {};
    #pragma unroll 1
    for (int64_t width = 0; width < kCopyWidths; ++width) {
        DataCopy(destination[width * kDestinationWidthStride],
                 source[width * kC0],
                 {static_cast<uint16_t>(kBGroupK), 1,
                  static_cast<uint16_t>(kSourceGap), 0}, enhanced);
    }
}

}  // namespace

extern "C" __global__ __aicore__ void grouped_matmul_short_b_ingress(
        __gm__ int64_t* group_rows, __gm__ bfloat16_t* packed_b,
        __gm__ int64_t* result, int64_t group_count) {
    const int64_t worker = GetBlockIdx();
    if (worker >= kAicCount || group_count <= 0 || group_count > kMaximumGroups) {
        return;
    }

    TPipe pipe;
    TBuf<TPosition::B1> b1_ping_buffer;
    TBuf<TPosition::B1> b1_pong_buffer;
    pipe.InitBuffer(b1_ping_buffer, kB1Elements * sizeof(bfloat16_t));
    pipe.InitBuffer(b1_pong_buffer, kB1Elements * sizeof(bfloat16_t));
    auto b1_ping = b1_ping_buffer.Get<bfloat16_t>();
    auto b1_pong = b1_pong_buffer.Get<bfloat16_t>();

    GlobalTensor<int64_t> rows;
    rows.SetGlobalBuffer(group_rows);
    int64_t source_count = 0;
    int64_t populated_group_index = 0;
    int64_t b1_sequence = 0;
    int64_t issued_b_groups = 0;

    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);

    // The source task begins after the preceding task.  This probe has no
    // preceding Matmul task, but preserving the boundary avoids silently
    // measuring a different MTE2 source lifecycle.
    WaitPreTaskEnd();
    const int64_t begin = GetSystemCycle();
    #pragma unroll 1
    for (int64_t group = 0; group < group_count; ++group) {
        const int64_t row_count = rows.GetValue(group);
        if (row_count < 0) {
            return;
        }
        if (row_count == 0) {
            continue;
        }
        const int64_t m_tiles = CeilDiv(row_count, kBaseM);
        const int64_t group_tile_count = m_tiles * kNTiles;
        const int64_t group_end = source_count + group_tile_count;
        int64_t source_block = worker >= source_count
            ? worker : worker + kAicCount;
        while (source_block < group_end) {
            const int64_t relative_block = source_block - source_count;
            const int64_t n_index = relative_block % kNTiles;
            GlobalTensor<bfloat16_t> b_source;
            const int64_t packed_group_offset = populated_group_index
                * kProjectionN * kProjectionK;
            const int64_t n_offset = n_index * kBaseN * kProjectionK;
            #pragma unroll 1
            for (int64_t b_group = 0; b_group < kBGroupsPerTile; ++b_group) {
                const int64_t slot = b1_sequence & 1;
                auto destination = slot == 0 ? b1_ping : b1_pong;
                b_source.SetGlobalBuffer(
                    packed_b + packed_group_offset + n_offset
                    + b_group * kBGroupK);
                b_source.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
                WaitFlag<HardEvent::MTE1_MTE2>(slot);
                CopyB1OnTheFly(destination, b_source);
                SetFlag<HardEvent::MTE2_MTE1>(slot);
                // The test stops at B ingress.  Return B1 only after its
                // MTE2 producer has completed, so it cannot fabricate more
                // than the source program's two B1 credits.
                WaitFlag<HardEvent::MTE2_MTE1>(slot);
                SetFlag<HardEvent::MTE1_MTE2>(slot);
                ++b1_sequence;
                ++issued_b_groups;
            }
            source_block += kAicCount;
        }
        source_count = group_end % kAicCount;
        ++populated_group_index;
    }
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    PipeBarrier<PIPE_MTE2>();
    const int64_t completion = GetSystemCycle() - begin;
    SetNextTaskStart();

    GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(result + worker * kResultStrideElements);
    output.SetValue(0, completion);
    output.SetValue(1, issued_b_groups);
    output.SetValue(2, source_count);
    output.SetValue(3, group_count);
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE,
        DcciDst::CACHELINE_OUT>(output);
}
