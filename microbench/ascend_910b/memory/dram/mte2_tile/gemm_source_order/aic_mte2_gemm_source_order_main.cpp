// ACL host for the independent MatMulV2 ingress-program probes.
#include <acl/acl.h>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",   \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_aic_mte2_gemm_source_order(
    uint32_t blockDim, void* stream, void* gm_a_source, void* gm_b_source,
    void* gm_output, int64_t m_groups, int64_t worker_m_tiles,
    int64_t worker_n_tiles, int64_t logical_n_tiles,
    int64_t output_tiles_per_worker, int64_t k_slabs, int64_t b_rows,
    int64_t source_row_stride);
extern "C" uint32_t aclrtlaunch_aic_mte2_gemm_ingress_completion(
    uint32_t blockDim, void* stream, void* gm_a_source, void* gm_b_source,
    void* gm_output, int64_t m_groups, int64_t worker_m_tiles,
    int64_t worker_n_tiles, int64_t logical_n_tiles,
    int64_t output_tiles_per_worker, int64_t k_slabs, int64_t b_rows,
    int64_t source_row_stride);
#ifdef GEMM_ENABLE_CUBE_COMPLETION
extern "C" uint32_t aclrtlaunch_aic_mte2_gemm_pipeline_completion(
    uint32_t blockDim, void* stream, void* gm_a_source, void* gm_b_source,
    void* gm_output, int64_t m_groups, int64_t worker_m_tiles,
    int64_t worker_n_tiles, int64_t logical_n_tiles,
    int64_t output_tiles_per_worker, int64_t k_slabs, int64_t b_rows,
    int64_t source_row_stride);
#endif
#ifdef GEMM_ENABLE_DIRTY_PREAMBLE
extern "C" uint32_t aclrtlaunch_aiv_l2_dirty_preamble(
    uint32_t blockDim, void* stream, void* gm_destination,
    int64_t total_bytes);
#endif

namespace {

constexpr int64_t kElementBytes = 2;
constexpr int64_t kCycleSlotBytes = 64;
#ifndef GEMM_SOURCE_A_ROWS
#define GEMM_SOURCE_A_ROWS 128
#endif
#ifndef GEMM_SOURCE_A_COLUMNS
#define GEMM_SOURCE_A_COLUMNS 512
#endif
constexpr int64_t kARows = GEMM_SOURCE_A_ROWS;
constexpr int64_t kAColumns = GEMM_SOURCE_A_COLUMNS;
constexpr int64_t kDefaultBRows = 256;
constexpr double kSystemCounterTickNs = 20.0;

struct CycleSpan {
    int64_t maximum_worker_cycles;
    int64_t launch_envelope_cycles;
};

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

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    const bool has_dirty_preamble = argc == 23;
    if ((argc != 21 && !has_dirty_preamble)
            || std::strcmp(argv[1], "--trials") != 0
            || std::strcmp(argv[3], "--active-aics") != 0
            || std::strcmp(argv[5], "--m-groups") != 0
            || std::strcmp(argv[7], "--worker-m-tiles") != 0
            || std::strcmp(argv[9], "--worker-n-tiles") != 0
            || std::strcmp(argv[11], "--logical-n-tiles") != 0
            || std::strcmp(argv[13], "--output-tiles-per-worker") != 0
            || std::strcmp(argv[15], "--k-slabs") != 0
            || std::strcmp(argv[17], "--b-rows") != 0
            || std::strcmp(argv[19], "--source-row-stride") != 0
            || (has_dirty_preamble
                && std::strcmp(argv[21], "--dirty-preamble-mib") != 0)) {
        std::fprintf(
            stderr,
            "usage: %s --trials COUNT --active-aics COUNT --m-groups COUNT "
            "--worker-m-tiles COUNT --worker-n-tiles COUNT "
            "--logical-n-tiles COUNT "
            "--output-tiles-per-worker COUNT --k-slabs COUNT "
            "--b-rows COUNT --source-row-stride COUNT "
            "[--dirty-preamble-mib COUNT]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t trials = ParseInteger(argv[2], "trials");
    const int64_t active_aics = ParseInteger(argv[4], "active_aics");
    const int64_t m_groups = ParseInteger(argv[6], "m_groups");
    const int64_t worker_m_tiles = ParseInteger(argv[8], "worker_m_tiles");
    const int64_t worker_n_tiles = ParseInteger(argv[10], "worker_n_tiles");
    const int64_t logical_n_tiles = ParseInteger(argv[12], "logical_n_tiles");
    const int64_t output_tiles_per_worker = ParseInteger(
        argv[14], "output_tiles_per_worker");
    const int64_t k_slabs = ParseInteger(argv[16], "k_slabs");
    const int64_t b_rows = ParseInteger(argv[18], "b_rows");
    const int64_t source_row_stride = ParseInteger(
        argv[20], "source_row_stride");
    const int64_t dirty_preamble_mib = (
        has_dirty_preamble
        ? ParseInteger(argv[22], "dirty_preamble_mib") : 0);
#ifndef GEMM_ENABLE_DIRTY_PREAMBLE
    if (dirty_preamble_mib != 0) {
        std::fprintf(stderr, "this binary has no dirty-preamble kernel\n");
        return EXIT_FAILURE;
    }
#endif
    if (trials < 3 || trials > 24) {
        std::fprintf(stderr, "trials must be in [3,24]\n");
        return EXIT_FAILURE;
    }
    if (active_aics <= 0 || active_aics > 24) {
        std::fprintf(stderr, "active_aics must be in [1,24]\n");
        return EXIT_FAILURE;
    }
    if (m_groups <= 0 || active_aics % m_groups != 0) {
        std::fprintf(
            stderr, "m_groups must be a positive divisor of active_aics\n");
        return EXIT_FAILURE;
    }
    if (output_tiles_per_worker <= 0 || output_tiles_per_worker > 32) {
        std::fprintf(
            stderr, "output_tiles_per_worker must be in [1,32]\n");
        return EXIT_FAILURE;
    }
    if (worker_m_tiles <= 0 || worker_n_tiles <= 0
            || worker_m_tiles * worker_n_tiles
                != output_tiles_per_worker) {
        std::fprintf(stderr, "worker tile rectangle disagrees with output count\n");
        return EXIT_FAILURE;
    }
    const int64_t n_groups = active_aics / m_groups;
    if (logical_n_tiles < worker_n_tiles
            || logical_n_tiles > n_groups * worker_n_tiles) {
        std::fprintf(stderr, "logical N tiles do not fit the worker grid\n");
        return EXIT_FAILURE;
    }
    if (k_slabs <= 0 || k_slabs > 64) {
        std::fprintf(stderr, "k_slabs must be in [1,64]\n");
        return EXIT_FAILURE;
    }
    if (b_rows <= 0 || b_rows > kDefaultBRows || b_rows % 16 != 0) {
        std::fprintf(stderr, "b_rows must be a multiple of 16 in [16,256]\n");
        return EXIT_FAILURE;
    }
    if (source_row_stride <= (k_slabs - 1) * kAColumns
            || source_row_stride > k_slabs * kAColumns
            || source_row_stride % 64 != 0) {
        std::fprintf(
            stderr, "source_row_stride must define exact 64-aligned K\n");
        return EXIT_FAILURE;
    }
    if (dirty_preamble_mib < 0 || dirty_preamble_mib > 192) {
        std::fprintf(stderr, "dirty_preamble_mib must be in [0,192]\n");
        return EXIT_FAILURE;
    }

#ifdef GEMM_ENABLE_CUBE_COMPLETION
    constexpr int64_t kModeCount = 3;
#else
    constexpr int64_t kModeCount = 2;
#endif
    const int64_t source_programs = kModeCount * (trials + 1);
    const int64_t rows = m_groups * worker_m_tiles * kARows;
    const int64_t columns = logical_n_tiles * b_rows;
    const int64_t a_source_elements = rows * source_row_stride;
    const int64_t b_source_elements = columns * source_row_stride;
    const size_t a_program_bytes = static_cast<size_t>(
        a_source_elements * kElementBytes);
    const size_t b_program_bytes = static_cast<size_t>(
        b_source_elements * kElementBytes);
    const size_t a_source_bytes = static_cast<size_t>(
        source_programs) * a_program_bytes;
    const size_t b_source_bytes = static_cast<size_t>(
        source_programs) * b_program_bytes;
    const size_t output_bytes = static_cast<size_t>(
        active_aics * kCycleSlotBytes);
    void* device_a_source = nullptr;
    void* device_b_source = nullptr;
    void* device_output = nullptr;
    void* device_dirty_preamble = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_a_source, a_source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_b_source, b_source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    const size_t dirty_preamble_bytes = static_cast<size_t>(
        dirty_preamble_mib) * 1024 * 1024;
    if (dirty_preamble_bytes > 0) {
        ACL_CHECK(aclrtMalloc(
            &device_dirty_preamble, dirty_preamble_bytes,
            ACL_MEM_MALLOC_HUGE_FIRST));
    }
    ACL_CHECK(aclrtMemset(device_a_source, a_source_bytes, 0, a_source_bytes));
    ACL_CHECK(aclrtMemset(device_b_source, b_source_bytes, 0, b_source_bytes));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    auto launch = [&](int64_t program, int64_t mode) -> CycleSpan {
        ACL_CHECK(aclrtMemset(device_output, output_bytes, 0, output_bytes));
#ifdef GEMM_ENABLE_DIRTY_PREAMBLE
        if (dirty_preamble_bytes > 0) {
            ACL_CHECK(aclrtlaunch_aiv_l2_dirty_preamble(
                48, stream, device_dirty_preamble,
                static_cast<int64_t>(dirty_preamble_bytes)));
        }
#endif
        auto* a_source = static_cast<char*>(device_a_source)
            + program * a_program_bytes;
        auto* b_source = static_cast<char*>(device_b_source)
            + program * b_program_bytes;
        if (mode == 0) {
            ACL_CHECK(aclrtlaunch_aic_mte2_gemm_source_order(
                static_cast<uint32_t>(active_aics), stream,
                a_source, b_source, device_output, m_groups,
                worker_m_tiles, worker_n_tiles, logical_n_tiles,
                output_tiles_per_worker, k_slabs, b_rows, source_row_stride));
        } else if (mode == 1) {
            ACL_CHECK(aclrtlaunch_aic_mte2_gemm_ingress_completion(
                static_cast<uint32_t>(active_aics), stream,
                a_source, b_source, device_output, m_groups,
                worker_m_tiles, worker_n_tiles, logical_n_tiles,
                output_tiles_per_worker, k_slabs, b_rows, source_row_stride));
#ifdef GEMM_ENABLE_CUBE_COMPLETION
        } else if (mode == 2) {
            ACL_CHECK(aclrtlaunch_aic_mte2_gemm_pipeline_completion(
                static_cast<uint32_t>(active_aics), stream,
                a_source, b_source, device_output, m_groups,
                worker_m_tiles, worker_n_tiles, logical_n_tiles,
                output_tiles_per_worker, k_slabs, b_rows, source_row_stride));
#endif
        } else {
            std::fprintf(stderr, "unsupported ingress mode %ld\n", mode);
            std::exit(EXIT_FAILURE);
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<int64_t> output(output_bytes / sizeof(int64_t), 0);
        ACL_CHECK(aclrtMemcpy(
            output.data(), output_bytes, device_output, output_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        int64_t maximum_cycles = 0;
        int64_t minimum_entry = std::numeric_limits<int64_t>::max();
        int64_t maximum_end = 0;
        for (int64_t block = 0; block < active_aics; ++block) {
            const size_t offset = static_cast<size_t>(
                block * kCycleSlotBytes / sizeof(int64_t));
            const int64_t expected_count = mode == 0
                ? output_tiles_per_worker * k_slabs
                : output_tiles_per_worker * source_row_stride / 64;
            if (output[offset] <= 0 || output[offset + 1] != expected_count
                    || output[offset + 2] <= 0
                    || output[offset + 3] < output[offset + 2]) {
                std::fprintf(
                    stderr,
                    "ingress block validation failed: block=%ld mode=%ld "
                    "cycles=%ld count=%ld expected_count=%ld\n",
                    block, mode, output[offset], output[offset + 1],
                    expected_count);
                std::exit(EXIT_FAILURE);
            }
            maximum_cycles = std::max(maximum_cycles, output[offset]);
            minimum_entry = std::min(minimum_entry, output[offset + 2]);
            maximum_end = std::max(maximum_end, output[offset + 3]);
        }
        return {maximum_cycles, maximum_end - minimum_entry};
    };

    for (int64_t mode = 0; mode < kModeCount; ++mode) {
        launch(kModeCount * trials + mode, mode);
    }
    std::printf(
        "trial,mode,max_cycles,duration_ns,launch_envelope_cycles,"
        "launch_envelope_ns,validation\n");
    for (int64_t trial = 0; trial < trials; ++trial) {
        for (int64_t position = 0; position < kModeCount; ++position) {
            const int64_t mode = (trial + position) % kModeCount;
            const int64_t program = kModeCount * trial + mode;
            const CycleSpan span = launch(program, mode);
            std::printf(
                "%ld,%s,%ld,%.9f,%ld,%.9f,pass\n", trial,
                mode == 0 ? "mte2_only" : (
                    mode == 1 ? "mte2_mte1_completion"
                              : "mte2_mte1_cube_completion"),
                span.maximum_worker_cycles,
                span.maximum_worker_cycles * kSystemCounterTickNs,
                span.launch_envelope_cycles,
                span.launch_envelope_cycles * kSystemCounterTickNs);
            std::fflush(stdout);
        }
    }

    ACL_CHECK(aclrtFree(device_a_source));
    ACL_CHECK(aclrtFree(device_b_source));
    ACL_CHECK(aclrtFree(device_output));
    if (device_dirty_preamble != nullptr) {
        ACL_CHECK(aclrtFree(device_dirty_preamble));
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
