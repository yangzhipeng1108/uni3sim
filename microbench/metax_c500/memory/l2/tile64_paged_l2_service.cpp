/*
 * MetaX C500 tile64 paged K/V residency microbenchmark.
 *
 * This is not FlashAttention. A CTA only reads one BF16-width K tile and
 * one BF16-width V tile for each causal KV step. It contains no MMA,
 * softmax, reduction, page-table load, or attention output. The four frozen
 * cells vary only query-head reuse (one or four consumers per page owner) and
 * the combined K/V pool size immediately below or above the documented 8-MiB
 * L2 capacity. The page number is computed in registers so downstream read
 * traffic is attributable to K/V data rather than a page-table stream.
 *
 * Device-event time is diagnostic. The profile marker contains exactly one
 * target launch, and production/evaluation duration is never consumed.
 *
 * Build in the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -std=c++17 \
 *       -o tile64_paged_l2_service tile64_paged_l2_service.cpp
 */

#include <mc_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <thread>
#include <unistd.h>
#include <vector>

#define CHECK(call) do {                                                   \
    mcError_t error = (call);                                              \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",              \
                     __FILE__, __LINE__, mcGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

namespace {

constexpr int kExpectedAps = 104;
constexpr int kExpectedWaveSize = 64;
constexpr int kExpectedL2Bytes = 8 * 1024 * 1024;
constexpr int kBlockThreads = 256;
constexpr int kDynamicSharedBytes = 32 * 1024;
constexpr int kQueryHeads = 32;
constexpr int kQueryTiles = 64;
constexpr int kBatch = 128;
constexpr int kQueryTileTokens = 64;
constexpr int kKvTileTokens = 64;
constexpr int kHeadDim = 128;
constexpr int kElementBytes = 2;
constexpr int kTileBytes = kKvTileTokens * kHeadDim * kElementBytes;
constexpr int kVectorsPerTile = kTileBytes / 16;
constexpr int kWarmupLaunches = 3;
constexpr int kEventTrials = 3;
constexpr int kAttachWaitSeconds = 15;
constexpr std::uint64_t kPageMultiplier = 8191ULL;
constexpr std::uint64_t kPageAddend = 17ULL;
constexpr const char* kCausalKvStepsRule = "q_tile_index_plus_one";

struct alignas(16) Vector128 {
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t z;
    std::uint32_t w;
};

static_assert(sizeof(Vector128) == 16, "request vector must be 16 bytes");
static_assert(kTileBytes == 16 * 1024,
              "tile64xD128 BF16 must be 16 KiB");
static_assert(kVectorsPerTile % kBlockThreads == 0,
              "each thread must issue the same number of tile loads");

struct CaseSpec {
    const char* name;
    const char* kernel_identity_token;
    int consumer_reuse_group_size;
    int pool_pages_per_operand;
};

constexpr CaseSpec kCases[] = {
    {"reuse1_pool224", "c500fa_gqa_r1_p224", 1, 224},
    {"reuse4_pool224", "c500fa_gqa_r4_p224", 4, 224},
    {"reuse1_pool288", "c500fa_gqa_r1_p288", 1, 288},
    {"reuse4_pool288", "c500fa_gqa_r4_p288", 4, 288},
};

template <int ConsumerReuseGroupSize, int PoolPagesPerOperand>
__device__ void paged_tile64_load_body(
        const Vector128* key_pages, const Vector128* value_pages,
        std::uint32_t* sink) {
    extern __shared__ std::uint32_t occupancy_control[];
    const int q_head = static_cast<int>(blockIdx.x);
    const int q_tile = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    occupancy_control[threadIdx.x] = static_cast<std::uint32_t>(
        q_head + q_tile + batch + threadIdx.x + 1);
    __syncthreads();

    constexpr int owner_count = kQueryHeads / ConsumerReuseGroupSize;
    const int owner = q_head / ConsumerReuseGroupSize;
    std::uint32_t checksum = occupancy_control[
        (threadIdx.x + 1) % blockDim.x];
    #pragma unroll 1
    for (int step = 0; step <= q_tile; ++step) {
        const std::uint64_t linear = (
            (static_cast<std::uint64_t>(batch) * owner_count + owner)
            * kQueryTiles + q_tile) * kQueryTiles + step;
        const std::uint32_t page = static_cast<std::uint32_t>(
            (linear * kPageMultiplier + kPageAddend)
            % PoolPagesPerOperand);
        const Vector128* key_tile = key_pages
            + static_cast<std::size_t>(page) * kVectorsPerTile;
        const Vector128* value_tile = value_pages
            + static_cast<std::size_t>(page) * kVectorsPerTile;
        #pragma unroll
        for (int vector = threadIdx.x; vector < kVectorsPerTile;
                vector += kBlockThreads) {
            const Vector128 key = key_tile[vector];
            const Vector128 value = value_tile[vector];
            checksum += key.x + key.y + key.z + key.w;
            checksum += value.x + value.y + value.z + value.w;
        }
    }
    const std::size_t cta = (
        static_cast<std::size_t>(batch) * kQueryTiles + q_tile)
        * kQueryHeads + q_head;
    sink[cta * blockDim.x + threadIdx.x] = checksum;
}

#define DEFINE_TILE64_KERNEL(NAME, REUSE, PAGES)                           \
extern "C" __global__ void NAME(                                         \
        const Vector128* key_pages, const Vector128* value_pages,          \
        std::uint32_t* sink) {                                             \
    paged_tile64_load_body<REUSE, PAGES>(key_pages, value_pages, sink);    \
}

DEFINE_TILE64_KERNEL(c500fa_gqa_r1_p224, 1, 224)
DEFINE_TILE64_KERNEL(c500fa_gqa_r4_p224, 4, 224)
DEFINE_TILE64_KERNEL(c500fa_gqa_r1_p288, 1, 288)
DEFINE_TILE64_KERNEL(c500fa_gqa_r4_p288, 4, 288)

const CaseSpec& parse_case(const char* text) {
    for (const CaseSpec& spec : kCases) {
        if (std::strcmp(spec.name, text) == 0) return spec;
    }
    std::fprintf(stderr, "unknown tile64 residency case: %s\n", text);
    std::exit(EXIT_FAILURE);
}

int parse_positive_integer(const char* text, const char* label) {
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0 || value > 2147483647L) {
        std::fprintf(stderr, "%s must be a positive integer\n", label);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

void require_device_contract() {
    mcDeviceProp_t properties {};
    CHECK(mcGetDeviceProperties(&properties, 0));
    if (std::strstr(properties.name, "C500") == nullptr
            || properties.multiProcessorCount != kExpectedAps
            || properties.warpSize != kExpectedWaveSize
            || properties.l2CacheSize != kExpectedL2Bytes) {
        std::fprintf(
            stderr,
            "expected MetaX C500 AP=104 wave=64 L2=8MiB; got name=%s AP=%d wave=%d L2=%d\n",
            properties.name, properties.multiProcessorCount,
            properties.warpSize, properties.l2CacheSize);
        std::exit(EXIT_FAILURE);
    }
}

void launch_case(
        const CaseSpec& spec, const Vector128* key_pages,
        const Vector128* value_pages, std::uint32_t* sink) {
    const dim3 grid(kQueryHeads, kQueryTiles, kBatch);
#define LAUNCH_TILE64_CASE(NAME)                                           \
    if (std::strcmp(spec.kernel_identity_token, #NAME) == 0) {             \
        NAME<<<grid, kBlockThreads, kDynamicSharedBytes>>>(                \
            key_pages, value_pages, sink);                                 \
    }
    LAUNCH_TILE64_CASE(c500fa_gqa_r1_p224)
    else LAUNCH_TILE64_CASE(c500fa_gqa_r4_p224)
    else LAUNCH_TILE64_CASE(c500fa_gqa_r1_p288)
    else LAUNCH_TILE64_CASE(c500fa_gqa_r4_p288)
    else {
        std::fprintf(stderr, "unbound tile64 symbol: %s\n",
                     spec.kernel_identity_token);
        std::exit(EXIT_FAILURE);
    }
#undef LAUNCH_TILE64_CASE
    CHECK(mcGetLastError());
}

void require_nonzero_sink(const std::vector<std::uint32_t>& sink) {
    if (std::none_of(sink.begin(), sink.end(),
                     [](std::uint32_t value) { return value != 0; })) {
        std::fprintf(stderr, "tile64 residency sink validation failed\n");
        std::exit(EXIT_FAILURE);
    }
}

unsigned long long operand_read_bytes() {
    const unsigned long long causal_steps =
        static_cast<unsigned long long>(kQueryTiles)
        * (kQueryTiles + 1) / 2;
    return static_cast<unsigned long long>(kBatch) * kQueryHeads
        * causal_steps * kTileBytes;
}

void print_sample(
        std::FILE* output, const CaseSpec& spec, const char* mode,
        int event_trials, float event_ms_min, float event_ms_median,
        float event_ms_max, const std::vector<std::uint32_t>& sink) {
    const unsigned long long workgroups =
        static_cast<unsigned long long>(kQueryHeads) * kQueryTiles * kBatch;
    const unsigned long long operand_bytes = operand_read_bytes();
    const unsigned long long pool_bytes =
        static_cast<unsigned long long>(spec.pool_pages_per_operand)
        * 2 * kTileBytes;
    const unsigned long long sink_write_bytes = workgroups * kBlockThreads
        * sizeof(std::uint32_t);
    std::fprintf(
        output,
        "metadata,2,metax_c500,paged_tile64_gqa_l2_residency,production_or_evaluation_duration_consumed=0,mcprofiler_instrumented_duration_consumed=0\n");
    std::fprintf(
        output,
        "sample_header,mode,case_id,kernel_identity,grid_x,grid_y,grid_z,workgroups,block,dynamic_shared_bytes,query_heads,q_tiles,batch,q_tile_tokens,kv_tile_tokens,head_dim,consumer_reuse_group_size,page_owner_count,pool_pages_per_operand,combined_kv_working_set_bytes,causal_kv_steps_rule,logical_k_read_bytes,logical_v_read_bytes,sink_write_bytes,event_trials,event_ms_min,event_ms_median,event_ms_max,first_sink,last_sink\n");
    std::fprintf(
        output,
        "sample,%s,%s,%s,%d,%d,%d,%llu,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%s,%llu,%llu,%llu,%d,%.9f,%.9f,%.9f,%08x,%08x\n",
        mode, spec.name, spec.kernel_identity_token, kQueryHeads, kQueryTiles,
        kBatch, workgroups, kBlockThreads, kDynamicSharedBytes, kQueryHeads,
        kQueryTiles, kBatch, kQueryTileTokens, kKvTileTokens, kHeadDim,
        spec.consumer_reuse_group_size,
        kQueryHeads / spec.consumer_reuse_group_size,
        spec.pool_pages_per_operand, pool_bytes, kCausalKvStepsRule,
        operand_bytes, operand_bytes, sink_write_bytes, event_trials,
        event_ms_min, event_ms_median, event_ms_max,
        sink.front(), sink.back());
    if (std::fflush(output) != 0) {
        std::perror("failed to flush tile64 sample output");
        std::exit(EXIT_FAILURE);
    }
}

std::FILE* open_profile_output(const char* output_path) {
    if (output_path == nullptr) return stdout;
    const int descriptor = open(output_path, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (descriptor < 0) {
        std::perror("cannot create exclusive profile output");
        std::exit(EXIT_FAILURE);
    }
    std::FILE* output = fdopen(descriptor, "w");
    if (output == nullptr) {
        std::perror("cannot open profile output stream");
        close(descriptor);
        std::exit(EXIT_FAILURE);
    }
    return output;
}

void run_case(const CaseSpec& spec, bool profile_once, int attach_wait_seconds,
              const char* profile_output_path) {
    require_device_contract();
    const std::size_t page_bytes =
        static_cast<std::size_t>(spec.pool_pages_per_operand) * kTileBytes;
    const std::size_t workgroups =
        static_cast<std::size_t>(kQueryHeads) * kQueryTiles * kBatch;
    std::vector<std::uint32_t> host_sink(workgroups * kBlockThreads);

    Vector128* key_pages = nullptr;
    Vector128* value_pages = nullptr;
    std::uint32_t* sink = nullptr;
    CHECK(mcMalloc(&key_pages, page_bytes));
    CHECK(mcMalloc(&value_pages, page_bytes));
    CHECK(mcMalloc(&sink, host_sink.size() * sizeof(std::uint32_t)));
    CHECK(mcMemset(key_pages, 0x3f, page_bytes));
    CHECK(mcMemset(value_pages, 0x5a, page_bytes));
    CHECK(mcMemset(sink, 0, host_sink.size() * sizeof(std::uint32_t)));

    for (int warmup = 0; warmup < kWarmupLaunches; ++warmup) {
        launch_case(spec, key_pages, value_pages, sink);
    }
    CHECK(mcDeviceSynchronize());

    float event_ms_min = 0.0f;
    float event_ms_median = 0.0f;
    float event_ms_max = 0.0f;
    if (profile_once) {
        if (attach_wait_seconds != kAttachWaitSeconds) {
            std::fprintf(stderr,
                         "profile attach wait must be exactly %d seconds\n",
                         kAttachWaitSeconds);
            std::exit(EXIT_FAILURE);
        }
        std::this_thread::sleep_for(
            std::chrono::seconds(attach_wait_seconds));
        CHECK(mcProfilerStart());
        launch_case(spec, key_pages, value_pages, sink);
        CHECK(mcDeviceSynchronize());
        CHECK(mcProfilerStop());
    } else {
        std::vector<float> trials;
        trials.reserve(kEventTrials);
        for (int trial = 0; trial < kEventTrials; ++trial) {
            mcEvent_t start;
            mcEvent_t stop;
            CHECK(mcEventCreate(&start));
            CHECK(mcEventCreate(&stop));
            CHECK(mcEventRecord(start));
            launch_case(spec, key_pages, value_pages, sink);
            CHECK(mcEventRecord(stop));
            CHECK(mcEventSynchronize(stop));
            float elapsed = 0.0f;
            CHECK(mcEventElapsedTime(&elapsed, start, stop));
            CHECK(mcEventDestroy(stop));
            CHECK(mcEventDestroy(start));
            if (!(elapsed > 0.0f)) {
                std::fprintf(stderr,
                             "device-event duration must be positive\n");
                std::exit(EXIT_FAILURE);
            }
            trials.push_back(elapsed);
        }
        std::sort(trials.begin(), trials.end());
        event_ms_min = trials.front();
        event_ms_median = trials[trials.size() / 2];
        event_ms_max = trials.back();
    }

    CHECK(mcMemcpy(host_sink.data(), sink,
                   host_sink.size() * sizeof(std::uint32_t),
                   mcMemcpyDeviceToHost));
    require_nonzero_sink(host_sink);
    std::FILE* output = open_profile_output(profile_output_path);
    print_sample(output, spec, profile_once ? "profile" : "event",
                 profile_once ? 0 : kEventTrials, event_ms_min,
                 event_ms_median, event_ms_max, host_sink);
    if (output != stdout && std::fclose(output) != 0) {
        std::perror("failed to close profile output");
        std::exit(EXIT_FAILURE);
    }

    CHECK(mcFree(sink));
    CHECK(mcFree(value_pages));
    CHECK(mcFree(key_pages));
}

void print_cases() {
    for (const CaseSpec& spec : kCases) {
        std::printf(
            "%s,symbol=%s,grid=%dx%dx%d,block=%d,dynamic_shared_bytes=%d,reuse_group=%d,pool_pages_per_operand=%d\n",
            spec.name, spec.kernel_identity_token, kQueryHeads, kQueryTiles,
            kBatch, kBlockThreads, kDynamicSharedBytes,
            spec.consumer_reuse_group_size,
            spec.pool_pages_per_operand);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--list-cases") == 0) {
        print_cases();
        return EXIT_SUCCESS;
    }
    if (argc == 3 && std::strcmp(argv[1], "--event") == 0) {
        run_case(parse_case(argv[2]), false, 0, nullptr);
        return EXIT_SUCCESS;
    }
    if (argc == 4 && std::strcmp(argv[1], "--profile-once") == 0) {
        run_case(parse_case(argv[2]), true,
                 parse_positive_integer(argv[3], "attach wait seconds"),
                 nullptr);
        return EXIT_SUCCESS;
    }
    if (argc == 5 && std::strcmp(argv[1], "--profile-once") == 0) {
        run_case(parse_case(argv[2]), true,
                 parse_positive_integer(argv[3], "attach wait seconds"),
                 argv[4]);
        return EXIT_SUCCESS;
    }
    std::fprintf(
        stderr,
        "usage: %s --list-cases | --event CASE | --profile-once CASE 15 [PROFILE_OUTPUT]\n",
        argv[0]);
    return EXIT_FAILURE;
}
