// ACL host for the exact 4096x8192x8192 MatMulV2 ingress-region probe.
#include <acl/acl.h>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",   \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_aic_mte2_gemm_region_source_order(
    uint32_t blockDim, void* stream, void* gm_a_source, void* gm_b_source,
    void* gm_output, int64_t k_slabs, int64_t m_regions,
    int64_t n_regions, int64_t m_block, int64_t n_block,
    int64_t m_tiles, int64_t n_tiles);
extern "C" uint32_t aclrtlaunch_aic_mte2_gemm_region_ingress_completion(
    uint32_t blockDim, void* stream, void* gm_a_source, void* gm_b_source,
    void* gm_output, int64_t k_slabs, int64_t m_regions,
    int64_t n_regions, int64_t m_block, int64_t n_block,
    int64_t m_tiles, int64_t n_tiles);
extern "C" uint32_t aclrtlaunch_aic_mte2_gemm_region_pipeline_completion(
    uint32_t blockDim, void* stream, void* gm_a_source, void* gm_b_source,
    void* gm_output, int64_t k_slabs, int64_t m_regions,
    int64_t n_regions, int64_t m_block, int64_t n_block,
    int64_t m_tiles, int64_t n_tiles);

namespace {

constexpr int64_t kActiveAics = 24;
constexpr int64_t kElementBytes = 2;
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kKColumnsPerSlab = 512;
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

int64_t GreatestCommonDivisor(int64_t left, int64_t right) {
    while (right != 0) {
        const int64_t remainder = left % right;
        left = right;
        right = remainder;
    }
    return left;
}

int64_t WorkerRounds(
        int64_t block, int64_t region,
        int64_t m_block, int64_t n_block) {
    const int64_t tiles_per_region = m_block * n_block;
    const int64_t tail_aics_per_region =
        tiles_per_region % kActiveAics;
    const int64_t rounds =
        (tiles_per_region + kActiveAics - 1) / kActiveAics;
    const int64_t block_start =
        region * tail_aics_per_region % kActiveAics;
    const int64_t relative_block =
        (block + kActiveAics - block_start) % kActiveAics;
    return tail_aics_per_region == 0
        || relative_block < tail_aics_per_region ? rounds : rounds - 1;
}

int64_t WorkerTileChecksum(
        int64_t block, int64_t m_tiles, int64_t n_tiles,
        int64_t m_regions, int64_t n_regions,
        int64_t m_block, int64_t n_block,
        int64_t* valid_tile_count) {
    const int64_t tail_aics_per_region =
        m_block * n_block % kActiveAics;
    const int64_t local_index_skew_period = (
        m_block / GreatestCommonDivisor(m_block, n_block) * n_block);
    int64_t checksum = 0;
    for (int64_t region = 0; region < m_regions * n_regions; ++region) {
        const int64_t block_start =
            region * tail_aics_per_region % kActiveAics;
        const int64_t rounds = WorkerRounds(
            block, region, m_block, n_block);
        for (int64_t round = 0; round < rounds; ++round) {
            const int64_t local_block = (
                (block + kActiveAics - block_start) % kActiveAics
                + round * kActiveAics);
            const int64_t local_m = local_block % m_block;
            const int64_t local_n =
                (local_block + local_block / local_index_skew_period)
                % n_block;
            const int64_t m_region = region / n_regions;
            const int64_t n_position = region % n_regions;
            const int64_t n_region = (m_region & 1) != 0
                ? n_regions - n_position - 1 : n_position;
            const int64_t m_tile = m_region * m_block + local_m;
            const int64_t n_tile = n_region * n_block + local_n;
            if (m_tile >= m_tiles || n_tile >= n_tiles) {
                continue;
            }
            checksum += m_tile * n_tiles + n_tile + 1;
            ++*valid_tile_count;
        }
    }
    return checksum;
}

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc != 17 || std::strcmp(argv[1], "--trials") != 0
            || std::strcmp(argv[3], "--k-slabs") != 0
            || std::strcmp(argv[5], "--m-regions") != 0
            || std::strcmp(argv[7], "--n-regions") != 0
            || std::strcmp(argv[9], "--m-block") != 0
            || std::strcmp(argv[11], "--n-block") != 0
            || std::strcmp(argv[13], "--m-tiles") != 0
            || std::strcmp(argv[15], "--n-tiles") != 0) {
        std::fprintf(
            stderr, "usage: %s --trials COUNT --k-slabs COUNT "
            "--m-regions COUNT --n-regions COUNT "
            "--m-block COUNT --n-block COUNT "
            "--m-tiles COUNT --n-tiles COUNT\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t trials = ParseInteger(argv[2], "trials");
    const int64_t k_slabs = ParseInteger(argv[4], "k_slabs");
    const int64_t m_regions = ParseInteger(argv[6], "m_regions");
    const int64_t n_regions = ParseInteger(argv[8], "n_regions");
    const int64_t m_block = ParseInteger(argv[10], "m_block");
    const int64_t n_block = ParseInteger(argv[12], "n_block");
    const int64_t m_tiles = ParseInteger(argv[14], "m_tiles");
    const int64_t n_tiles = ParseInteger(argv[16], "n_tiles");
    if (trials < 3 || trials > 24) {
        std::fprintf(stderr, "trials must be in [3,24]\n");
        return EXIT_FAILURE;
    }
    if (k_slabs <= 0 || k_slabs > 32) {
        std::fprintf(stderr, "k_slabs must be in [1,32]\n");
        return EXIT_FAILURE;
    }
    for (const auto& item : std::vector<std::pair<const char*, int64_t>>{
            {"m_regions", m_regions}, {"n_regions", n_regions},
            {"m_block", m_block}, {"n_block", n_block}}) {
        if (item.second <= 0 || item.second > 64) {
            std::fprintf(stderr, "%s must be in [1,64]\n", item.first);
            return EXIT_FAILURE;
        }
    }
    for (const auto& item : std::vector<std::pair<const char*, int64_t>>{
            {"m_tiles", m_tiles}, {"n_tiles", n_tiles}}) {
        if (item.second <= 0 || item.second > 512) {
            std::fprintf(stderr, "%s must be in [1,512]\n", item.first);
            return EXIT_FAILURE;
        }
    }
    if (m_tiles > m_regions * m_block || n_tiles > n_regions * n_block) {
        std::fprintf(stderr, "logical tiles exceed region schedule\n");
        return EXIT_FAILURE;
    }
    const int64_t m = m_tiles * 128;
    const int64_t n = n_tiles * 256;
    constexpr int64_t kModeCount = 3;
    const int64_t source_programs = kModeCount * (trials + 1);
    const int64_t k = k_slabs * kKColumnsPerSlab;
    const size_t a_program_bytes = static_cast<size_t>(
        m * k * kElementBytes);
    const size_t b_program_bytes = static_cast<size_t>(
        n * k * kElementBytes);
    const size_t a_source_bytes = static_cast<size_t>(
        source_programs) * a_program_bytes;
    const size_t b_source_bytes = static_cast<size_t>(
        source_programs) * b_program_bytes;
    const size_t output_bytes = static_cast<size_t>(
        kActiveAics * kCycleSlotBytes);
    void* device_a_source = nullptr;
    void* device_b_source = nullptr;
    void* device_output = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_a_source, a_source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_b_source, b_source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(device_a_source, a_source_bytes, 0, a_source_bytes));
    ACL_CHECK(aclrtMemset(device_b_source, b_source_bytes, 0, b_source_bytes));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    auto launch = [&](int64_t program, int64_t mode) -> int64_t {
        ACL_CHECK(aclrtMemset(device_output, output_bytes, 0, output_bytes));
        auto* a_source = static_cast<char*>(device_a_source)
            + program * a_program_bytes;
        auto* b_source = static_cast<char*>(device_b_source)
            + program * b_program_bytes;
        if (mode == 0) {
            ACL_CHECK(aclrtlaunch_aic_mte2_gemm_region_source_order(
                static_cast<uint32_t>(kActiveAics), stream,
                a_source, b_source, device_output, k_slabs,
                m_regions, n_regions, m_block, n_block, m_tiles, n_tiles));
        } else if (mode == 1) {
            ACL_CHECK(aclrtlaunch_aic_mte2_gemm_region_ingress_completion(
                static_cast<uint32_t>(kActiveAics), stream,
                a_source, b_source, device_output, k_slabs,
                m_regions, n_regions, m_block, n_block, m_tiles, n_tiles));
        } else if (mode == 2) {
            ACL_CHECK(aclrtlaunch_aic_mte2_gemm_region_pipeline_completion(
                static_cast<uint32_t>(kActiveAics), stream,
                a_source, b_source, device_output, k_slabs,
                m_regions, n_regions, m_block, n_block, m_tiles, n_tiles));
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
        int64_t aggregate_tiles = 0;
        int64_t aggregate_checksum = 0;
        for (int64_t block = 0; block < kActiveAics; ++block) {
            const size_t offset = static_cast<size_t>(
                block * kCycleSlotBytes / sizeof(int64_t));
            int64_t expected_tiles = 0;
            const int64_t expected_checksum = WorkerTileChecksum(
                block, m_tiles, n_tiles, m_regions, n_regions,
                m_block, n_block, &expected_tiles);
            const int64_t expected_count = (
                expected_tiles * k_slabs * (mode == 0 ? 1 : 8));
            if (output[offset] <= 0 || output[offset + 1] != expected_count
                    || output[offset + 2] != expected_tiles
                    || output[offset + 3] != expected_checksum) {
                std::fprintf(
                    stderr,
                    "region validation failed: block=%ld mode=%ld "
                    "cycles=%ld count=%ld/%ld tiles=%ld/%ld "
                    "checksum=%ld/%ld\n",
                    block, mode, output[offset], output[offset + 1],
                    expected_count, output[offset + 2], expected_tiles,
                    output[offset + 3], expected_checksum);
                std::exit(EXIT_FAILURE);
            }
            maximum_cycles = std::max(maximum_cycles, output[offset]);
            aggregate_tiles += output[offset + 2];
            aggregate_checksum += output[offset + 3];
        }
        const int64_t expected_total_tiles = m_tiles * n_tiles;
        const int64_t expected_total_checksum =
            expected_total_tiles * (expected_total_tiles + 1) / 2;
        if (aggregate_tiles != expected_total_tiles
                || aggregate_checksum != expected_total_checksum) {
            std::fprintf(
                stderr,
                "aggregate region validation failed: tiles=%ld/%ld "
                "checksum=%ld/%ld\n",
                aggregate_tiles, expected_total_tiles,
                aggregate_checksum, expected_total_checksum);
            std::exit(EXIT_FAILURE);
        }
        return maximum_cycles;
    };

    for (int64_t mode = 0; mode < kModeCount; ++mode) {
        launch(kModeCount * trials + mode, mode);
    }
    std::printf(
        "trial,mode,max_cycles,duration_ns,m_tiles,n_tiles,k_slabs,"
        "aggregate_tiles,aggregate_checksum,validation\n");
    for (int64_t trial = 0; trial < trials; ++trial) {
        for (int64_t position = 0; position < kModeCount; ++position) {
            const int64_t mode = (trial + position) % kModeCount;
            const int64_t program = kModeCount * trial + mode;
            const int64_t maximum_cycles = launch(program, mode);
            std::printf(
                "%ld,%s,%ld,%.9f,%ld,%ld,%ld,%ld,%ld,pass\n",
                trial,
                mode == 0 ? "mte2_only" : (
                    mode == 1 ? "mte2_mte1_completion"
                              : "mte2_mte1_cube_completion"),
                maximum_cycles, maximum_cycles * kSystemCounterTickNs,
                m_tiles, n_tiles, k_slabs, m_tiles * n_tiles,
                m_tiles * n_tiles * (m_tiles * n_tiles + 1) / 2);
            std::fflush(stdout);
        }
    }

    ACL_CHECK(aclrtFree(device_a_source));
    ACL_CHECK(aclrtFree(device_b_source));
    ACL_CHECK(aclrtFree(device_output));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
