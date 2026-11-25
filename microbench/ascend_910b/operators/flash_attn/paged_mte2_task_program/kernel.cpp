// Independent CANN 8.5.1 regular-FIA MM1 paged-helper task-program probe.
//
// This deliberately covers only the source-valid MM1 K ingress boundary:
// block-table lookup -> two page ND2NZ copies -> MTE2/MTE1 completion -> two
// L0/MMAD/Fixpipe panels -> K-L1 slot release.  It excludes P/V, Softmax,
// AIV work and all Attention output composition, so it cannot reproduce a
// production Attention duration.  The task count and K-slot lifetime match
// the 24-AIC S2=4096 regular-GQA program selected by CANN 8.5.1.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kWorkers = 24;
constexpr int64_t kAlignedM = 16;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kPageRows = 128;
constexpr int64_t kPagesPerHelper = 2;
constexpr int64_t kHelperRows = kPageRows * kPagesPerHelper;
constexpr int64_t kHelpersPerTask = 16;
constexpr int64_t kMaximumTasks = 43;
constexpr int64_t kPagesPerWorker = (
    kMaximumTasks * kHelpersPerTask * kPagesPerHelper);
constexpr int64_t kElementsPerBlock = 32 / sizeof(bfloat16_t);
constexpr int64_t kPageElements = kPageRows * kHeadDim;
constexpr int64_t kCycleSlotElements = 64 / sizeof(int64_t);
constexpr int64_t kOutputElements = 2 * kAlignedM * kHeadDim;

__aicore__ inline int64_t TaskCountForProfile(
        int64_t worker, int64_t profile) {
    // Exact CANN source task distribution for B128/S4096/HQ64/HKV8/D128:
    // groups 0..8 and even groups 10..22 own 43 tasks; odd groups 9..23 own
    // 42.  Uniform 42/43 profiles are pre-registered holdouts only.
    if (profile == 0) {
        return (worker < 9 || (worker >= 10 && worker % 2 == 0)) ? 43 : 42;
    }
    if (profile == 1) {
        return 42;
    }
    if (profile == 2) {
        return 43;
    }
    if (profile == 3) {
        constexpr int64_t target_task_counts[kWorkers] = {
            36, 36, 36, 36, 36, 36, 37, 36,
            26, 18, 18, 18, 19, 18, 19, 13,
            9, 9, 9, 9, 10, 9, 10, 9,
        };
        return target_task_counts[worker];
    }
    return -1;
}

__aicore__ inline int64_t HelpersForTask(
        int64_t worker, int64_t task, int64_t profile) {
    if (profile != 3) {
        return kHelpersPerTask;
    }
    constexpr int64_t target_s2_2048_tasks[kWorkers] = {
        36, 36, 36, 36, 36, 36, 37, 36,
        15, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    };
    constexpr int64_t target_s2_4096_tasks[kWorkers] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        11, 18, 18, 18, 19, 18, 19, 7,
        0, 0, 0, 0, 0, 0, 0, 0,
    };
    const int64_t s2_2048_count = target_s2_2048_tasks[worker];
    const int64_t s2_4096_count = target_s2_4096_tasks[worker];
    if (task < s2_2048_count) {
        return 8;
    }
    if (task < s2_2048_count + s2_4096_count) {
        return 16;
    }
    return 32;
}

__aicore__ inline int64_t PageOffsetForTask(
        int64_t worker, int64_t task, int64_t profile) {
    // The source program's three S2 domains are contiguous within each AIC.
    // Keep this address calculation closed-form: replaying all earlier tasks
    // would add a scalar loop that the selected CANN task program does not
    // execute, and would therefore invalidate this mechanism-only probe.
    if (profile != 3) {
        return task * kHelpersPerTask * kPagesPerHelper;
    }
    constexpr int64_t target_s2_2048_tasks[kWorkers] = {
        36, 36, 36, 36, 36, 36, 37, 36,
        15, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    };
    constexpr int64_t target_s2_4096_tasks[kWorkers] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        11, 18, 18, 18, 19, 18, 19, 7,
        0, 0, 0, 0, 0, 0, 0, 0,
    };
    const int64_t s2_2048_count = target_s2_2048_tasks[worker];
    const int64_t s2_4096_count = target_s2_4096_tasks[worker];
    if (task < s2_2048_count) {
        return task * 8 * kPagesPerHelper;
    }
    if (task < s2_2048_count + s2_4096_count) {
        return s2_2048_count * 8 * kPagesPerHelper
            + (task - s2_2048_count) * 16 * kPagesPerHelper;
    }
    return s2_2048_count * 8 * kPagesPerHelper
        + s2_4096_count * 16 * kPagesPerHelper
        + (task - s2_2048_count - s2_4096_count)
            * 32 * kPagesPerHelper;
}

__aicore__ inline void CopyPage(
        AscendC::LocalTensor<bfloat16_t>& destination,
        AscendC::GlobalTensor<bfloat16_t>& source,
        int64_t destination_page, int64_t source_page) {
    AscendC::Nd2NzParams parameters;
    parameters.ndNum = 1;
    parameters.nValue = static_cast<uint16_t>(kPageRows);
    parameters.dValue = static_cast<uint16_t>(kHeadDim);
    parameters.srcNdMatrixStride = 0;
    parameters.srcDValue = static_cast<uint16_t>(kHeadDim);
    parameters.dstNzC0Stride = static_cast<uint16_t>(kHelperRows);
    parameters.dstNzNStride = 1;
    parameters.dstNzMatrixStride = 0;
    AscendC::DataCopy(
        destination[destination_page * kPageRows * kElementsPerBlock],
        source[source_page * kPageElements], parameters);
}

__aicore__ inline void ResetLoad3dConfig() {
    constexpr uint8_t padding[] = {0, 0, 0, 255};
    AscendC::SetFmatrix(
        kHeadDim / 16, 16, padding, AscendC::FmatrixMode::FMATRIX_LEFT);
    AscendC::SetLoadDataPaddingValue<bfloat16_t>(0);
}

__aicore__ inline void LoadA(
        AscendC::LocalTensor<bfloat16_t>& destination,
        const AscendC::LocalTensor<bfloat16_t>& source) {
    AscendC::LoadData3DParamsV2<bfloat16_t> parameters;
    parameters.l1H = 1;
    parameters.l1W = 16;
    parameters.padList[0] = 0;
    parameters.padList[1] = 0;
    parameters.padList[2] = 0;
    parameters.padList[3] = 255;
    parameters.mStartPt = 0;
    parameters.kStartPt = 0;
    parameters.mExtension = kAlignedM;
    parameters.kExtension = kHeadDim;
    parameters.channelSize = kHeadDim;
    parameters.strideW = 1;
    parameters.strideH = 1;
    parameters.filterW = 1;
    parameters.filterSizeW = 0;
    parameters.filterH = 1;
    parameters.filterSizeH = 0;
    parameters.dilationFilterW = 1;
    parameters.dilationFilterH = 1;
    parameters.enTranspose = 0;
    parameters.fMatrixCtrl = 0;
    AscendC::LoadData(destination, source, parameters);
    ResetLoad3dConfig();
}

__aicore__ inline void LoadB(
        AscendC::LocalTensor<bfloat16_t>& destination,
        const AscendC::LocalTensor<bfloat16_t>& source,
        int64_t source_row_offset) {
    AscendC::LoadData2DParams parameters;
    parameters.startIndex = 0;
    parameters.repeatTimes = 8;
    parameters.srcStride = 1;
    parameters.dstGap = 0;
    parameters.ifTranspose = false;
    for (int64_t index = 0; index < kHeadDim / 16; ++index) {
        AscendC::LoadData(
            destination[128 * index * 16],
            source[source_row_offset * 16 + kHelperRows * index * 16],
            parameters);
    }
}

__aicore__ inline void IssueFixpipe(
        AscendC::GlobalTensor<float>& destination,
        const AscendC::LocalTensor<float>& source, int64_t actual_m) {
    AscendC::FixpipeParamsV220 parameters;
    parameters.mSize = actual_m;
    parameters.nSize = 128;
    parameters.srcStride = kAlignedM;
    parameters.dstStride = 128;
    parameters.ndNum = 1;
    AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
        destination, source, parameters);
}

__aicore__ inline void StoreResult(
        AscendC::GlobalTensor<int64_t>& destination,
        int64_t cycles, int64_t valid) {
    destination.SetValue(0, cycles);
    destination.SetValue(1, valid);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(destination);
}

}  // namespace

extern "C" __global__ __aicore__ void fia_paged_mte2_task_program(
        __gm__ bfloat16_t* gm_q, __gm__ bfloat16_t* gm_source,
        __gm__ int32_t* gm_block_table, __gm__ float* gm_output,
        __gm__ int64_t* gm_cycles, int64_t profile, int64_t actual_m) {
    const int64_t worker = AscendC::GetBlockIdx();
    AscendC::GlobalTensor<int64_t> cycle_slot;
    cycle_slot.SetGlobalBuffer(gm_cycles + worker * kCycleSlotElements);
    const int64_t task_count = TaskCountForProfile(worker, profile);
    if (AscendC::GetBlockNum() != kWorkers || task_count <= 0
            || task_count > kMaximumTasks || (actual_m != 4 && actual_m != 8)
            || (profile == 3 && actual_m != 4)) {
        StoreResult(cycle_slot, -1, 0);
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> q_l1_buffer;
    AscendC::TBuf<AscendC::TPosition::B1> k_l1_buffer;
    AscendC::TBuf<AscendC::TPosition::A2> a_l0_buffer;
    AscendC::TBuf<AscendC::TPosition::B2> b_l0_buffer;
    AscendC::TBuf<AscendC::TPosition::CO1> c_l0_buffer;
    pipe.InitBuffer(q_l1_buffer, kAlignedM * kHeadDim * sizeof(bfloat16_t));
    pipe.InitBuffer(
        k_l1_buffer, 3 * kHelperRows * kHeadDim * sizeof(bfloat16_t));
    pipe.InitBuffer(
        a_l0_buffer, 2 * kAlignedM * kHeadDim * sizeof(bfloat16_t));
    pipe.InitBuffer(b_l0_buffer, 2 * 128 * kHeadDim * sizeof(bfloat16_t));
    pipe.InitBuffer(c_l0_buffer, 2 * kAlignedM * 128 * sizeof(float));
    auto q_l1 = q_l1_buffer.Get<bfloat16_t>();
    auto k_l1 = k_l1_buffer.Get<bfloat16_t>();
    auto a_l0 = a_l0_buffer.Get<bfloat16_t>();
    auto b_l0 = b_l0_buffer.Get<bfloat16_t>();
    auto c_l0 = c_l0_buffer.Get<float>();

    AscendC::GlobalTensor<bfloat16_t> q_source;
    q_source.SetGlobalBuffer(gm_q + worker * kAlignedM * kHeadDim);
    AscendC::Nd2NzParams q_parameters;
    q_parameters.ndNum = 1;
    q_parameters.nValue = static_cast<uint16_t>(actual_m);
    q_parameters.dValue = static_cast<uint16_t>(kHeadDim);
    q_parameters.srcNdMatrixStride = 0;
    q_parameters.srcDValue = static_cast<uint16_t>(kHeadDim);
    q_parameters.dstNzC0Stride = static_cast<uint16_t>(kAlignedM);
    q_parameters.dstNzNStride = 1;
    q_parameters.dstNzMatrixStride = 0;
    AscendC::DataCopy(q_l1, q_source, q_parameters);
    AscendC::PipeBarrier<PIPE_MTE2>();
    ResetLoad3dConfig();

    for (int64_t index = 0; index < 3; ++index) {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(index + 4);
    }
    for (int64_t index = 0; index < 2; ++index) {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(index + 3);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(index + 3);
    }

    AscendC::GlobalTensor<bfloat16_t> source;
    source.SetGlobalBuffer(
        gm_source + worker * kPagesPerWorker * kPageElements);
    AscendC::GlobalTensor<int32_t> block_table;
    block_table.SetGlobalBuffer(gm_block_table + worker * kPagesPerWorker);
    AscendC::GlobalTensor<float> output;
    output.SetGlobalBuffer(gm_output + worker * kOutputElements);

    int64_t k_slot = 0;
    int64_t ab_slot = 0;
    int64_t c_slot = 0;
    int64_t checksum = 0;
    const int64_t begin = AscendC::GetSystemCycle();
    for (int64_t task = 0; task < task_count; ++task) {
        const int64_t helper_count = HelpersForTask(worker, task, profile);
        const int64_t task_page_offset = PageOffsetForTask(
            worker, task, profile);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
            static_cast<uint32_t>(4 + k_slot));
        auto k_panel = k_l1[k_slot * kHelperRows * kHeadDim];
        for (int64_t helper = 0; helper < helper_count; ++helper) {
            for (int64_t page = 0; page < kPagesPerHelper; ++page) {
                const int64_t table_offset = (
                    task_page_offset + (helper * kPagesPerHelper) + page);
                const int64_t source_page = block_table.GetValue(table_offset);
                checksum += source_page - table_offset;
                CopyPage(k_panel, source, page, source_page);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(
                static_cast<uint32_t>(4 + k_slot));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(
                static_cast<uint32_t>(4 + k_slot));
            for (int64_t tile = 0; tile < 2; ++tile) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(
                    static_cast<uint32_t>(3 + c_slot));
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(
                    static_cast<uint32_t>(3 + ab_slot));
                auto local_a = a_l0[ab_slot * kAlignedM * kHeadDim];
                auto local_b = b_l0[ab_slot * 128 * kHeadDim];
                auto local_c = c_l0[c_slot * kAlignedM * 128];
                LoadA(local_a, q_l1);
                LoadB(local_b, k_panel, tile * 128);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID7);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID7);
                AscendC::MmadParams mm_parameters;
                mm_parameters.m = kAlignedM;
                mm_parameters.n = 128;
                mm_parameters.k = kHeadDim;
                mm_parameters.cmatrixInitVal = true;
                mm_parameters.cmatrixSource = false;
                AscendC::Mmad(local_c, local_a, local_b, mm_parameters);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(
                    static_cast<uint32_t>(3 + ab_slot));
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(
                    static_cast<uint32_t>(3 + c_slot));
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(
                    static_cast<uint32_t>(3 + c_slot));
                auto output_slot = output[c_slot * kAlignedM * 128];
                IssueFixpipe(output_slot, local_c, actual_m);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(
                    static_cast<uint32_t>(3 + c_slot));
                ab_slot = (ab_slot + 1) % 2;
                c_slot = (c_slot + 1) % 2;
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
            static_cast<uint32_t>(4 + k_slot));
        k_slot = (k_slot + 1) % 3;
    }
    for (int64_t index = 0; index < 3; ++index) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(index + 4);
    }
    for (int64_t index = 0; index < 2; ++index) {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(index + 3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(index + 3);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t end = AscendC::GetSystemCycle();
    StoreResult(cycle_slot, end - begin, checksum == 0 ? 1 : 0);
}
