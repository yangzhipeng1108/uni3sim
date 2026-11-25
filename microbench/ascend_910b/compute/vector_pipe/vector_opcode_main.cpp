// Host driver for the independent AIV Vector-pipe opcode benchmark.
#include <acl/acl.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <string>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",   \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

#define CLOCK_CHECK(call) do {                                            \
    if ((call) != 0) {                                                    \
        std::perror(#call);                                               \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_vector_opcode(
    uint32_t blockDim, void* stream, void* gm_cycles, void* gm_validation,
    int64_t opcode, int64_t element_count, int64_t iterations,
    float source_value, float divisor_value);

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
constexpr int64_t kMaximumBlockDim = 48;
constexpr int64_t kCycleValuesPerBlock = 8;
constexpr int64_t kValidationValuesPerBlock = 16;
constexpr double kSystemCounterTickNs = 20.0;

int64_t ParseInteger(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        std::fprintf(stderr, "%s must be an integer, got %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

float SourceExactBlockReduceSumFirstValue(
        int64_t elements, int64_t iterations, float source_value) {
    // CANN 8.5.1 add_rms_norm_base.h::BlockReduceSumFP32 splits count into
    // 64-FP32 vector repeats followed by a multiple-of-eight tail. Each
    // repeat emits one sum for each eight-FP32 row partial at dst + 8 * rep.
    // The production helper deliberately aliases dst and src, so model that
    // source-visible recurrence exactly for the benchmark validation.
    constexpr int64_t kElementsPerRepeat = 64;
    constexpr int64_t kElementsPerPartial = 8;
    std::vector<float> data(static_cast<size_t>(elements), source_value);
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        const std::vector<float> source = data;
        const int64_t repeat_times = elements / kElementsPerRepeat;
        const int64_t tail_count = elements % kElementsPerRepeat;
        for (int64_t repeat = 0; repeat < repeat_times; ++repeat) {
            for (int64_t partial = 0;
                    partial < kElementsPerRepeat / kElementsPerPartial;
                    ++partial) {
                float sum = 0.0f;
                const int64_t source_offset = repeat * kElementsPerRepeat
                    + partial * kElementsPerPartial;
                for (int64_t element = 0;
                        element < kElementsPerPartial; ++element) {
                    sum += source[static_cast<size_t>(source_offset + element)];
                }
                data[static_cast<size_t>(repeat * kElementsPerPartial
                    + partial)] = sum;
            }
        }
        const int64_t tail_partials = tail_count / kElementsPerPartial;
        const int64_t tail_source_offset = repeat_times * kElementsPerRepeat;
        const int64_t tail_destination_offset = repeat_times
            * kElementsPerPartial;
        for (int64_t partial = 0; partial < tail_partials; ++partial) {
            float sum = 0.0f;
            const int64_t source_offset = tail_source_offset
                + partial * kElementsPerPartial;
            for (int64_t element = 0; element < kElementsPerPartial;
                    ++element) {
                sum += source[static_cast<size_t>(source_offset + element)];
            }
            data[static_cast<size_t>(tail_destination_offset + partial)] = sum;
        }
    }
    return data.front();
}

struct Cell {
    int64_t opcode;
    int64_t elements;
    int64_t block_dim;
    int64_t iterations;
};

const char* OpcodeName(int64_t opcode) {
    if (opcode == kCastInt32ToInt64) {
        return "cast_int32_to_int64";
    }
    if (opcode == kCompareFp32Gt) {
        return "compare_fp32_gt";
    }
    if (opcode == kCompareFp32Eq) {
        return "compare_fp32_eq";
    }
    if (opcode == kCompareFp32Ne) {
        return "compare_fp32_ordered_ne";
    }
    if (opcode == kSelectFp32) {
        return "select_fp32";
    }
    if (opcode == kExp) {
        return "exp";
    }
    if (opcode == kDiv) {
        return "div";
    }
    if (opcode == kExpDiv) {
        return "exp_div";
    }
    if (opcode == kSqrt) {
        return "sqrt";
    }
    if (opcode == kSqrtDiv) {
        return "sqrt_div";
    }
    if (opcode == kExpSerialized) {
        return "exp_serialized";
    }
    if (opcode == kDivSerialized) {
        return "div_serialized";
    }
    if (opcode == kCastBf16ToFp32Serialized) {
        return "cast_bf16_to_fp32_serialized";
    }
    if (opcode == kCastFp32ToBf16Serialized) {
        return "cast_fp32_to_bf16_serialized";
    }
    if (opcode == kMulsSerialized) {
        return "muls_serialized";
    }
    if (opcode == kAddsSerialized) {
        return "adds_serialized";
    }
    if (opcode == kMulSerialized) {
        return "mul_serialized";
    }
    if (opcode == kNormSqrtSerialized) {
        return "norm_sqrt_serialized";
    }
    if (opcode == kNormDivSerialized) {
        return "norm_div_serialized";
    }
    if (opcode == kNormSqrtDuplicateDivSerialized) {
        return "norm_sqrt_duplicate_div_serialized";
    }
    if (opcode == kTwoTbufMaskPrologue) {
        return "two_tbuf_mask_prologue";
    }
    if (opcode == kRmsNormLocalScalarDependency) {
        return "rmsnorm_local_scalar_dependency";
    }
    if (opcode == kLocalScalarFillSv) {
        return "local_scalar_fill_s_v";
    }
    if (opcode == kLocalUbCopyContiguous) {
        return "local_ub_copy_contiguous";
    }
    if (opcode == kLocalUbCopyStride64) {
        return "local_ub_copy_stride_64b";
    }
    if (opcode == kLocalUbCopyStride128) {
        return "local_ub_copy_stride_128b";
    }
    if (opcode == kSingleTbufInit) {
        return "single_tbuf_init";
    }
    if (opcode == kScalarU32DivRemDependency) {
        return "scalar_u32_divrem_dependency";
    }
    if (opcode == kScalarU32MulAddDependency) {
        return "scalar_u32_mul_add_dependency";
    }
    if (opcode == kScalarU64MulDependency) {
        return "scalar_u64_mul_dependency";
    }
    if (opcode == kScalarU64DivRemDependency) {
        return "scalar_u64_divrem_dependency";
    }
    if (opcode == kRmsNormFourQueueThreeTbufLifecycle) {
        return "rmsnorm_four_queue_three_tbuf_lifecycle";
    }
    if (opcode == kRmsNormFourQueueControlChain) {
        return "rmsnorm_four_queue_control_chain";
    }
    if (opcode == kRmsNormFourQueueThreeTbufReady) {
        return "rmsnorm_four_queue_three_tbuf_ready";
    }
    if (opcode == kAddRmsNormLocalScalarHandoff) {
        return "add_rmsnorm_local_scalar_handoff";
    }
    if (opcode == kAddRmsNormReduceSumFp32) {
        return "add_rmsnorm_reduce_sum_fp32";
    }
    if (opcode == kAddRmsNormSplitDBlockReduceSumFp32) {
        return "add_rmsnorm_split_d_block_reduce_sum_fp32";
    }
    if (opcode == kVectorToMte2Fence) {
        return "vector_to_mte2_fence";
    }
    if (opcode == kAddRmsNormSplitDLocalScalarHandoff) {
        return "add_rmsnorm_split_d_local_scalar_handoff";
    }
    if (opcode == kMergeNReduceSumMultiN) {
        return "merge_n_reduce_sum_multi_n";
    }
    if (opcode == kMergeNBroadcastGamma) {
        return "merge_n_broadcast_gamma";
    }
    if (opcode == kMergeNBroadcastRstd) {
        return "merge_n_broadcast_rstd";
    }
    if (opcode == kMergeNPerformanceRstdBrcb) {
        return "merge_n_performance_rstd_brcb";
    }
    if (opcode == kMergeNPerformanceMulRowRstd) {
        return "merge_n_performance_mul_row_rstd";
    }
    if (opcode == kMergeNPerformanceMulRowGamma) {
        return "merge_n_performance_mul_row_gamma";
    }
    if (opcode == kSwigluDoubleBufferInit) {
        return "swiglu_double_buffer_init";
    }
    if (opcode == kSwigluSingleBufferInit) {
        return "swiglu_single_buffer_init";
    }
    if (opcode == kRmsNormHalfIntervalReduce) {
        return "rmsnorm_half_interval_reduce";
    }
    if (opcode == kSwigluDoubleBufferLifecycle) {
        return "swiglu_double_buffer_lifecycle";
    }
    if (opcode == kSwigluSingleBufferLifecycle) {
        return "swiglu_single_buffer_lifecycle";
    }
    if (opcode == kSwigluDoubleBufferQueueControl) {
        return "swiglu_double_buffer_queue_control";
    }
    if (opcode == kSwigluSingleBufferQueueControl) {
        return "swiglu_single_buffer_queue_control";
    }
    std::fprintf(stderr, "unknown opcode %ld\n", opcode);
    std::exit(EXIT_FAILURE);
}

std::string JoinCycles(const std::vector<int64_t>& cycles) {
    std::string result;
    for (size_t index = 0; index < cycles.size(); ++index) {
        if (index != 0) {
            result.push_back('|');
        }
        result += std::to_string(cycles[index]);
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 2) % 4 != 0) {
        std::fprintf(
            stderr,
            "usage: %s --batch opcode elements block_dim iterations "
            "[opcode elements block_dim iterations ...]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int cell_count = (argc - 2) / 4;
    if (cell_count > 64) {
        std::fprintf(stderr, "a batch cannot contain more than 64 cells\n");
        return EXIT_FAILURE;
    }
    std::vector<Cell> cells;
    cells.reserve(static_cast<size_t>(cell_count));
    for (int argument = 2; argument < argc; argument += 4) {
        const Cell cell{
            ParseInteger(argv[argument], "opcode"),
            ParseInteger(argv[argument + 1], "elements"),
            ParseInteger(argv[argument + 2], "block_dim"),
            ParseInteger(argv[argument + 3], "iterations"),
        };
        if ((cell.opcode != kExp && cell.opcode != kDiv
                && cell.opcode != kExpDiv && cell.opcode != kSqrt
                && cell.opcode != kSqrtDiv
                && cell.opcode != kExpSerialized
                && cell.opcode != kDivSerialized
                && cell.opcode != kCastBf16ToFp32Serialized
                && cell.opcode != kCastFp32ToBf16Serialized
                && cell.opcode != kMulsSerialized
                && cell.opcode != kAddsSerialized
                && cell.opcode != kMulSerialized
                && cell.opcode != kNormSqrtSerialized
                && cell.opcode != kNormDivSerialized
                && cell.opcode != kNormSqrtDuplicateDivSerialized
                && cell.opcode != kTwoTbufMaskPrologue
                && cell.opcode != kRmsNormLocalScalarDependency
                && cell.opcode != kLocalScalarFillSv
                && cell.opcode != kLocalUbCopyContiguous
                && cell.opcode != kLocalUbCopyStride64
                && cell.opcode != kLocalUbCopyStride128
                && cell.opcode != kSingleTbufInit
                && cell.opcode != kScalarU32DivRemDependency
                && cell.opcode != kScalarU32MulAddDependency
                && cell.opcode != kScalarU64MulDependency
                && cell.opcode != kSwigluDoubleBufferInit
                && cell.opcode != kSwigluSingleBufferInit
                && cell.opcode != kRmsNormHalfIntervalReduce
                && cell.opcode != kSwigluDoubleBufferLifecycle
                && cell.opcode != kSwigluSingleBufferLifecycle
                && cell.opcode != kSwigluDoubleBufferQueueControl
                && cell.opcode != kSwigluSingleBufferQueueControl
                && cell.opcode != kScalarU64DivRemDependency
                && cell.opcode != kRmsNormFourQueueThreeTbufLifecycle
                && cell.opcode != kRmsNormFourQueueControlChain
                && cell.opcode != kRmsNormFourQueueThreeTbufReady
                && cell.opcode != kAddRmsNormLocalScalarHandoff
                && cell.opcode != kAddRmsNormSplitDLocalScalarHandoff
                && cell.opcode != kAddRmsNormReduceSumFp32
                && cell.opcode != kVectorToMte2Fence
                && cell.opcode != kAddRmsNormSplitDFourQueueFourTbufLifecycle
                && cell.opcode != kAddRmsNormSplitDBlockReduceSumFp32
                && cell.opcode != kMergeNReduceSumMultiN
                && cell.opcode != kMergeNBroadcastGamma
                && cell.opcode != kMergeNBroadcastRstd
                && cell.opcode != kMergeNPerformanceRstdBrcb
                && cell.opcode != kMergeNPerformanceMulRowRstd
                && cell.opcode != kMergeNPerformanceMulRowGamma
                && cell.opcode != kCompareFp32Gt
                && cell.opcode != kCompareFp32Eq
                && cell.opcode != kCompareFp32Ne
                && cell.opcode != kSelectFp32
                && cell.opcode != kCastInt32ToInt64)
                || cell.elements < 1 || cell.elements > 13312
                || ((cell.opcode >= kCompareFp32Gt
                        && cell.opcode <= kCastInt32ToInt64)
                    && (cell.elements % 64 != 0 || cell.elements > 4096))
                || ((cell.opcode == kMergeNReduceSumMultiN
                        || cell.opcode == kMergeNBroadcastGamma
                        || cell.opcode == kMergeNBroadcastRstd
                        || cell.opcode == kMergeNPerformanceRstdBrcb
                        || cell.opcode == kMergeNPerformanceMulRowRstd
                        || cell.opcode == kMergeNPerformanceMulRowGamma)
                    && cell.elements > 84)
                || cell.block_dim <= 0 || cell.block_dim > kMaximumBlockDim
                || cell.iterations <= 0
                || ((cell.opcode != kTwoTbufMaskPrologue
                        && cell.opcode != kSingleTbufInit
                        && cell.opcode != kSwigluDoubleBufferInit
                        && cell.opcode != kSwigluSingleBufferInit
                        && cell.opcode != kSwigluDoubleBufferLifecycle
                        && cell.opcode != kSwigluSingleBufferLifecycle
                        && cell.opcode != kSwigluDoubleBufferQueueControl
                        && cell.opcode != kSwigluSingleBufferQueueControl
                        && cell.opcode
                        != kRmsNormFourQueueThreeTbufLifecycle
                        && cell.opcode != kRmsNormFourQueueControlChain
                        && cell.opcode != kRmsNormFourQueueThreeTbufReady)
                    && cell.iterations % 2 != 0)
                || cell.iterations > std::numeric_limits<int64_t>::max() / 4) {
            std::fprintf(stderr, "invalid vector opcode benchmark cell\n");
            return EXIT_FAILURE;
        }
        cells.push_back(cell);
    }

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    void* device_cycles = nullptr;
    void* device_validation = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_cycles,
        kMaximumBlockDim * kCycleValuesPerBlock * sizeof(int64_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_validation,
        kMaximumBlockDim * kValidationValuesPerBlock * sizeof(float),
        ACL_MEM_MALLOC_HUGE_FIRST));

    auto run = [&](const Cell& cell, int64_t iterations,
                   bool require_positive_cycles,
                   std::vector<int64_t>& block_cycles, int64_t& wall_ns) {
        const bool norm_sqrt_source = (
            cell.opcode == kSqrt || cell.opcode == kSqrtDiv
            || cell.opcode == kNormSqrtSerialized
            || cell.opcode == kNormSqrtDuplicateDivSerialized
            || cell.opcode == kRmsNormLocalScalarDependency);
        const float source_value = norm_sqrt_source
            ? 4.0f : (cell.opcode == kNormDivSerialized ? 1.0f : -1.0f);
        constexpr float kDivisorValue = 2.0f;
        const bool scalar_integer_dependency = (
            cell.opcode == kScalarU32DivRemDependency
            || cell.opcode == kScalarU32MulAddDependency
            || cell.opcode == kScalarU64MulDependency
            || cell.opcode == kScalarU64DivRemDependency);
        uint64_t expected_integer = 0;
        float expected;
        if (cell.opcode == kExp || cell.opcode == kExpSerialized) {
            expected = std::exp(source_value);
        } else if (cell.opcode == kSqrt
                   || cell.opcode == kNormSqrtSerialized) {
            if (cell.opcode == kNormSqrtSerialized) {
                expected = std::pow(
                    source_value, std::pow(0.5, iterations));
            } else {
                expected = std::sqrt(source_value);
            }
        } else if (cell.opcode == kNormDivSerialized) {
            expected = kDivisorValue;
        } else if (cell.opcode == kNormSqrtDuplicateDivSerialized) {
            expected = std::pow(
                source_value, std::pow(-0.5, iterations));
        } else if (cell.opcode == kMulsSerialized
                   || cell.opcode == kMulSerialized) {
            expected = source_value * kDivisorValue;
        } else if (cell.opcode == kAddsSerialized) {
            expected = source_value + kDivisorValue;
        } else if (cell.opcode == kCastBf16ToFp32Serialized
                   || cell.opcode == kCastFp32ToBf16Serialized) {
            expected = source_value;
        } else if (cell.opcode == kTwoTbufMaskPrologue) {
            // The alternating mask clears lane zero.  The subsequent Adds
            // writes the disjoint tensor view starting at mask_elements, so
            // the validated first lane remains zero.
            expected = 0.0f;
        } else if (cell.opcode == kRmsNormLocalScalarDependency) {
            expected = 1.0f / std::sqrt(source_value * 0.25f + 1.0e-5f);
        } else if (cell.opcode == kAddRmsNormLocalScalarHandoff) {
            expected = source_value;
        } else if (cell.opcode == kAddRmsNormSplitDLocalScalarHandoff) {
            expected = source_value;
        } else if (cell.opcode == kAddRmsNormReduceSumFp32) {
            expected = source_value * static_cast<float>(cell.elements);
        } else if (cell.opcode == kAddRmsNormSplitDBlockReduceSumFp32) {
            expected = SourceExactBlockReduceSumFirstValue(
                cell.elements, iterations, source_value);
        } else if (cell.opcode == kVectorToMte2Fence) {
            expected = 0.0f;
        } else if (cell.opcode == kLocalScalarFillSv) {
            expected = source_value;
        } else if (cell.opcode == kLocalUbCopyContiguous
                   || cell.opcode == kLocalUbCopyStride64
                   || cell.opcode == kLocalUbCopyStride128) {
            expected = source_value;
        } else if (cell.opcode == kSingleTbufInit) {
            expected = source_value;
        } else if (cell.opcode == kSwigluDoubleBufferInit
                   || cell.opcode == kSwigluSingleBufferInit
                   || cell.opcode == kSwigluDoubleBufferLifecycle
                   || cell.opcode == kSwigluSingleBufferLifecycle
                   || cell.opcode == kSwigluDoubleBufferQueueControl
                   || cell.opcode == kSwigluSingleBufferQueueControl
                   || cell.opcode == kRmsNormFourQueueThreeTbufLifecycle
                   || cell.opcode == kRmsNormFourQueueControlChain
                   || cell.opcode == kRmsNormFourQueueThreeTbufReady) {
            expected = source_value;
        } else if (cell.opcode == kRmsNormHalfIntervalReduce) {
            expected = 0.0f;
        } else if (cell.opcode == kMergeNReduceSumMultiN
                   || cell.opcode == kMergeNBroadcastGamma
                   || cell.opcode == kMergeNBroadcastRstd
                   || cell.opcode == kMergeNPerformanceRstdBrcb) {
            // ReduceSumMultiN writes only the row tensor; the two BroadCast
            // calls and Brcb replicate the uniform source value, so the
            // validated first tile lane keeps it.
            expected = source_value;
        } else if (cell.opcode == kMergeNPerformanceMulRowRstd) {
            expected = source_value * kDivisorValue;
        } else if (cell.opcode == kMergeNPerformanceMulRowGamma) {
            expected = kDivisorValue * kDivisorValue;
        } else if (scalar_integer_dependency) {
            const uint32_t operand = static_cast<uint32_t>(cell.elements);
            uint64_t value;
            if (cell.opcode == kScalarU32DivRemDependency) {
                uint32_t current = 0xf1234567U;
                for (int64_t iteration = 0; iteration < iterations;
                     ++iteration) {
                    const uint32_t quotient = current / operand;
                    const uint32_t remainder = current % operand;
                    current = quotient + remainder + 0x9e3779b9U;
                }
                value = current;
            } else if (cell.opcode == kScalarU32MulAddDependency) {
                uint32_t current = 0xf1234567U;
                for (int64_t iteration = 0; iteration < iterations;
                     ++iteration) {
                    current = current * operand + 1013904223U;
                }
                value = current;
            } else if (cell.opcode == kScalarU64MulDependency) {
                uint64_t current = 0xfedcba9876543211ULL;
                const uint64_t multiplier = 2ULL * operand + 1ULL;
                for (int64_t iteration = 0; iteration < iterations;
                     ++iteration) {
                    current *= multiplier;
                }
                value = current;
            } else {
                uint64_t current = 0xfedcba9876543211ULL;
                const uint64_t divisor = 2ULL * operand + 1ULL;
                for (int64_t iteration = 0; iteration < iterations;
                     ++iteration) {
                    const uint64_t quotient = current / divisor;
                    const uint64_t remainder = current % divisor;
                    current = quotient + remainder
                        + 0x9e3779b97f4a7c15ULL;
                }
                value = current;
            }
            expected_integer = value & 0x00ffffffULL;
            expected = source_value;
        } else {
            expected = source_value / kDivisorValue;
        }
        ACL_CHECK(aclrtMemset(
            device_cycles,
            kMaximumBlockDim * kCycleValuesPerBlock * sizeof(int64_t),
            0xff,
            kMaximumBlockDim * kCycleValuesPerBlock * sizeof(int64_t)));
        ACL_CHECK(aclrtMemset(
            device_validation,
            kMaximumBlockDim * kValidationValuesPerBlock * sizeof(float),
            0, kMaximumBlockDim * kValidationValuesPerBlock * sizeof(float)));
        struct timespec start;
        struct timespec end;
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &start));
        ACL_CHECK(aclrtlaunch_vector_opcode(
            static_cast<uint32_t>(cell.block_dim), stream,
            device_cycles, device_validation, cell.opcode, cell.elements,
            iterations, source_value, kDivisorValue));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &end));
        wall_ns = static_cast<int64_t>(end.tv_sec - start.tv_sec)
            * 1000000000LL + static_cast<int64_t>(end.tv_nsec - start.tv_nsec);

        std::vector<int64_t> padded_cycles(
            static_cast<size_t>(cell.block_dim) * kCycleValuesPerBlock);
        std::vector<float> validation(
            static_cast<size_t>(cell.block_dim)
            * kValidationValuesPerBlock);
        ACL_CHECK(aclrtMemcpy(
            padded_cycles.data(), padded_cycles.size() * sizeof(int64_t),
            device_cycles, padded_cycles.size() * sizeof(int64_t),
            ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(
            validation.data(), validation.size() * sizeof(float),
            device_validation, validation.size() * sizeof(float),
            ACL_MEMCPY_DEVICE_TO_HOST));
        block_cycles.resize(static_cast<size_t>(cell.block_dim));
        for (int64_t block = 0; block < cell.block_dim; ++block) {
            const int64_t cycles = padded_cycles[
                static_cast<size_t>(block) * kCycleValuesPerBlock];
            block_cycles[static_cast<size_t>(block)] = cycles;
            if (cycles < 0 || (require_positive_cycles && cycles == 0)) {
                std::fprintf(
                    stderr,
                    "block %ld produced invalid %s cycles: %ld\n",
                    block,
                    require_positive_cycles ? "measured" : "warmup",
                    cycles);
                std::exit(EXIT_FAILURE);
            }
            if (scalar_integer_dependency) {
                const int64_t result = padded_cycles[
                    static_cast<size_t>(block) * kCycleValuesPerBlock + 1];
                if (result != static_cast<int64_t>(expected_integer)) {
                    std::fprintf(
                        stderr,
                        "block %ld integer validation failed: got %ld "
                        "expected %ld\n",
                        block, result,
                        static_cast<int64_t>(expected_integer));
                    std::exit(EXIT_FAILURE);
                }
            }
            if (cell.opcode >= kCompareFp32Gt && cell.opcode <= kSelectFp32) {
                const int64_t mismatches = padded_cycles[
                    static_cast<size_t>(block) * kCycleValuesPerBlock + 1];
                if (mismatches != 0) {
                    std::fprintf(stderr,
                        "block %ld full typed output validation failed: "
                        "%ld mismatched elements\n", block, mismatches);
                    // Preserve the observed lane truth values before failing.
                    // A mismatch count alone cannot establish NaN semantics.
                    std::fprintf(stderr,
                        "typed_validation_values opcode=%ld block=%ld values=",
                        cell.opcode, block);
                    for (int64_t lane = 0; lane < 16; ++lane) {
                        std::fprintf(stderr, "%s%.9g", lane == 0 ? "" : ",",
                            validation[static_cast<size_t>(block)
                                * kValidationValuesPerBlock + lane]);
                    }
                    std::fprintf(stderr, "\n");
                    std::exit(EXIT_FAILURE);
                }
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
                for (int64_t lane = 0; lane < 16; ++lane) {
                    float lhs, rhs;
                    std::memcpy(&lhs, &lhs_bits[lane], sizeof(float));
                    std::memcpy(&rhs, &rhs_bits[lane], sizeof(float));
                    const bool predicate = cell.opcode == kCompareFp32Eq
                        ? lhs == rhs : (cell.opcode == kCompareFp32Ne
                        ? (!std::isnan(lhs) && !std::isnan(rhs) && lhs != rhs)
                        : lhs > rhs);
                    const float magnitude = static_cast<float>(10 + lane);
                    const float lane_expected = predicate
                        ? magnitude : -magnitude;
                    const float actual = validation[
                        static_cast<size_t>(block)
                        * kValidationValuesPerBlock + lane];
                    if (actual != lane_expected) {
                        std::fprintf(stderr,
                            "block %ld lane %ld typed validation failed: "
                            "got %.9f expected %.9f\n", block, lane,
                            actual, lane_expected);
                        std::exit(EXIT_FAILURE);
                    }
                }
                continue;
            }
            if (cell.opcode == kCastInt32ToInt64) {
                const size_t offset = static_cast<size_t>(block)
                    * kCycleValuesPerBlock;
                if (padded_cycles[offset + 1] != 0) {
                    std::fprintf(stderr,
                        "block %ld INT32->INT64 output mismatch count: %ld\n",
                        block, padded_cycles[offset + 1]);
                    std::exit(EXIT_FAILURE);
                }
                const int64_t expected_samples[6] = {
                    (-2147483647LL - 1), -1, 0, 1, 2147483646, 2147483647};
                for (size_t sample = 0; sample < 6; ++sample) {
                    if (padded_cycles[offset + 2 + sample]
                            != expected_samples[sample]) {
                        std::fprintf(stderr,
                            "block %ld signed INT64 sample %zu mismatch\n",
                            block, sample);
                        std::exit(EXIT_FAILURE);
                    }
                }
                continue;
            }
            const float value = validation[
                static_cast<size_t>(block) * kValidationValuesPerBlock];
            if (!std::isfinite(value) || std::fabs(value - expected) > 1e-5f) {
                std::fprintf(
                    stderr,
                    "block %ld validation failed: got %.9f expected %.9f\n",
                    block, value, expected);
                std::exit(EXIT_FAILURE);
            }
        }
    };

    for (const Cell& cell : cells) {
        std::vector<int64_t> warmup_cycles;
        int64_t warmup_wall_ns = 0;
        // SYS_CNT has a 20 ns tick.  A short warmup can legitimately
        // quantize to zero, but the negative sentinel and kernel error codes
        // must still fail loudly.  Published short/long intervals below keep
        // the stricter positive-cycle requirement.
        run(cell, 16, false, warmup_cycles, warmup_wall_ns);

        std::vector<int64_t> short_cycles;
        int64_t short_wall_ns = 0;
        std::vector<int64_t> long_cycles;
        int64_t long_wall_ns = 0;
        run(cell, cell.iterations, true, short_cycles, short_wall_ns);
        run(cell, cell.iterations * 4, true, long_cycles, long_wall_ns);
        // Host wall intervals are retained only as raw provenance.  Host
        // scheduling can make the separately synchronized long interval
        // shorter than the short interval, even when the device intervals
        // are valid.  Service time is therefore derived from the fixed
        // 50-MHz SYS_CNT contract, never from a host-wall difference.
        if (short_wall_ns <= 0 || long_wall_ns <= 0) {
            std::fprintf(stderr, "host wall intervals must be positive\n");
            return EXIT_FAILURE;
        }
        for (size_t block = 0; block < short_cycles.size(); ++block) {
            if (cell.opcode != kTwoTbufMaskPrologue
                    && cell.opcode != kSingleTbufInit
                    && cell.opcode != kSwigluDoubleBufferInit
                    && cell.opcode != kSwigluSingleBufferInit
                    && cell.opcode != kSwigluDoubleBufferLifecycle
                    && cell.opcode != kSwigluSingleBufferLifecycle
                    && cell.opcode != kSwigluDoubleBufferQueueControl
                    && cell.opcode != kSwigluSingleBufferQueueControl
                    && cell.opcode
                    != kRmsNormFourQueueThreeTbufLifecycle
                    && cell.opcode != kRmsNormFourQueueControlChain
                    && cell.opcode != kRmsNormFourQueueThreeTbufReady
                    && long_cycles[block] <= short_cycles[block]) {
                std::fprintf(
                    stderr, "block %zu long cycles must exceed short cycles\n",
                    block);
                return EXIT_FAILURE;
            }
        }
        const int64_t short_critical = *std::max_element(
            short_cycles.begin(), short_cycles.end());
        const int64_t long_critical = *std::max_element(
            long_cycles.begin(), long_cycles.end());
        const double tick_ns = kSystemCounterTickNs;
        const double per_iteration_ns = (
            (cell.opcode == kTwoTbufMaskPrologue
                    || cell.opcode == kSingleTbufInit
                    || cell.opcode == kSwigluDoubleBufferInit
                    || cell.opcode == kSwigluSingleBufferInit
                    || cell.opcode == kSwigluDoubleBufferLifecycle
                    || cell.opcode == kSwigluSingleBufferLifecycle
                    || cell.opcode == kSwigluDoubleBufferQueueControl
                    || cell.opcode == kSwigluSingleBufferQueueControl
                    || cell.opcode == kRmsNormFourQueueThreeTbufLifecycle
                    || cell.opcode == kRmsNormFourQueueControlChain
                    || cell.opcode == kRmsNormFourQueueThreeTbufReady)
            ? 0.5 * static_cast<double>(short_critical + long_critical)
                * tick_ns
            : static_cast<double>(long_critical - short_critical) * tick_ns
                / static_cast<double>(cell.iterations * 3));
        const double elements_per_ns = static_cast<double>(cell.elements)
            / per_iteration_ns;
        if (!std::isfinite(tick_ns) || tick_ns <= 0.0
                || !std::isfinite(per_iteration_ns)
                || per_iteration_ns <= 0.0
                || !std::isfinite(elements_per_ns)
                || elements_per_ns <= 0.0) {
            std::fprintf(stderr, "derived opcode measurement is invalid\n");
            return EXIT_FAILURE;
        }
        std::printf(
            "%s,%ld,%ld,%ld,%s,%ld,%s,%ld,%.9f,%.9f,%.9f,pass\n",
            OpcodeName(cell.opcode), cell.elements, cell.block_dim,
            cell.iterations, JoinCycles(short_cycles).c_str(), short_wall_ns,
            JoinCycles(long_cycles).c_str(), long_wall_ns, tick_ns,
            per_iteration_ns, elements_per_ns);
        std::fflush(stdout);
    }

    ACL_CHECK(aclrtFree(device_cycles));
    ACL_CHECK(aclrtFree(device_validation));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
