/*
 * MetaX C500 operator-independent cross-slot smoke benchmark.
 *
 * Each cell pairs one slot-2 stream (MMA or Vector-ALU/MTE) with one stream
 * from slot 3 (global-memory store) or slot 4 (shared-memory store). Four
 * separate compile-time kernels retain identical stage work: slot2 isolated,
 * cross-slot isolated, independent work on two waves, and one-wave work whose
 * cross-slot stores consume the slot-2 result. There is no loop-internal
 * barrier and no device-side dynamic phase dispatch. The fixed prime loop
 * count and four static operations per iteration do not reproduce a
 * production operator's instruction proportions.
 */

#include <mc_runtime.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <thread>
#include <vector>

#define CHECK(call) do {                                                   \
    mcError_t error = (call);                                              \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",               \
                     __FILE__, __LINE__, mcGetErrorString(error));          \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

namespace {

constexpr int kExpectedApCount = 104;
constexpr int kExpectedWaveSize = 64;
constexpr int kExpectedL2Bytes = 8 * 1024 * 1024;
constexpr int kGridBlocks = kExpectedApCount;
constexpr int kBlockThreads = 2 * kExpectedWaveSize;
constexpr int kWavesPerBlock = 2;
constexpr int kIterations = 4093;
constexpr int kStaticOpsPerIteration = 4;
constexpr int kAddressRounds = 257;
constexpr int kSharedWords = kStaticOpsPerIteration * kExpectedWaveSize;
constexpr int kDynamicSharedBytes = 64 * 1024;
constexpr int kWarmupLaunches = 3;
constexpr int kEventTrials = 7;
constexpr int kAttachWaitSeconds = 15;

static_assert(kBlockThreads == 128, "cross-slot cell requires two waves");
static_assert(kIterations != 64 && kIterations != 256,
              "mechanism count must not mirror a production tile");
static_assert(kSharedWords == 256, "shared store ring must remain fixed");

typedef __NATIVE_VECTOR__(2, int32_t) v2i;
typedef __NATIVE_VECTOR__(4, float) v4f;

struct alignas(16) Vector128 {
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t z;
    std::uint32_t w;
};

static_assert(sizeof(Vector128) == 16, "global store must be 128 bits");

enum class Slot2 { kMma, kMte };
enum class CrossSlot { kGvm, kBsm };
enum class Mode { kSlot2Isolated, kCrossIsolated, kIndependent, kDependent };

struct ModeSpec {
    Slot2 slot2;
    CrossSlot cross;
    Mode mode;
    const char* cell;
    const char* name;
    const char* kernel_identity;
};

#define CELL_MODES(SLOT2_ENUM, CROSS_ENUM, CELL_NAME)                     \
    {SLOT2_ENUM, CROSS_ENUM, Mode::kSlot2Isolated, #CELL_NAME,             \
     "slot2_isolated", "c500_cross_" #CELL_NAME "_slot2"},              \
    {SLOT2_ENUM, CROSS_ENUM, Mode::kCrossIsolated, #CELL_NAME,             \
     "cross_isolated", "c500_cross_" #CELL_NAME "_cross"},              \
    {SLOT2_ENUM, CROSS_ENUM, Mode::kIndependent, #CELL_NAME,               \
     "independent", "c500_cross_" #CELL_NAME "_independent"},           \
    {SLOT2_ENUM, CROSS_ENUM, Mode::kDependent, #CELL_NAME,                 \
     "dependent", "c500_cross_" #CELL_NAME "_dependent"}

constexpr ModeSpec kModes[] = {
    CELL_MODES(Slot2::kMma, CrossSlot::kGvm, mma_gvm),
    CELL_MODES(Slot2::kMte, CrossSlot::kGvm, mte_gvm),
    CELL_MODES(Slot2::kMma, CrossSlot::kBsm, mma_bsm),
    CELL_MODES(Slot2::kMte, CrossSlot::kBsm, mte_bsm),
};

#undef CELL_MODES

__device__ __forceinline__ v4f mma_bf16(v2i lhs, v2i rhs, v4f accumulator) {
    return __builtin_mxc_mma_16x16x16bf16(lhs, rhs, accumulator);
}

__device__ __forceinline__ float add_f32(float lhs, float rhs) {
    float output;
    asm volatile("add_f32 %0, %1, %2"
                 : "=r"(output) : "r"(lhs), "r"(rhs));
    return output;
}

template <Slot2 slot2>
__device__ __forceinline__ float slot2_stream(float seed) {
    if constexpr (slot2 == Slot2::kMma) {
        const v2i lhs = {0x3b803c00, 0x3a803b00};
        const v2i rhs = {0x3b003a80, 0x3c003b80};
        v4f c0 = {seed, seed + 0.125f, seed + 0.25f, seed + 0.5f};
        v4f c1 = {seed + 0.25f, seed + 0.5f, seed + 1.0f, seed + 0.125f};
        v4f c2 = {seed + 0.5f, seed + 1.0f, seed + 0.125f, seed + 0.25f};
        v4f c3 = {seed + 1.0f, seed + 0.125f, seed + 0.25f, seed + 0.5f};
        #pragma unroll 1
        for (int iteration = 0; iteration < kIterations; ++iteration) {
            c0 = mma_bf16(lhs, rhs, c0);
            c1 = mma_bf16(lhs, rhs, c1);
            c2 = mma_bf16(lhs, rhs, c2);
            c3 = mma_bf16(lhs, rhs, c3);
        }
        return c0.x + c1.y + c2.z + c3.w;
    } else if constexpr (slot2 == Slot2::kMte) {
        float v0 = seed;
        float v1 = seed + 0.125f;
        float v2 = seed + 0.25f;
        float v3 = seed + 0.5f;
        const float increment = 0.00000011920928955078125f;
        #pragma unroll 1
        for (int iteration = 0; iteration < kIterations; ++iteration) {
            v0 = add_f32(v0, increment);
            v1 = add_f32(v1, increment);
            v2 = add_f32(v2, increment);
            v3 = add_f32(v3, increment);
        }
        return v0 + v1 + v2 + v3;
    } else {
        __builtin_trap();
        return seed;
    }
}

__device__ __forceinline__ void gvm_store_stream(
        Vector128* output, std::size_t logical_thread,
        std::size_t logical_threads, float token) {
    #pragma unroll 1
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        const std::uint32_t bits = __float_as_uint(
            token + static_cast<float>(iteration) * 0.0000009536743164f);
        const std::size_t round = static_cast<std::size_t>(
            iteration % kAddressRounds);
        #pragma unroll
        for (int slot = 0; slot < kStaticOpsPerIteration; ++slot) {
            const std::size_t index = (
                (static_cast<std::size_t>(slot) * kAddressRounds + round)
                * logical_threads + logical_thread);
            output[index] = {
                bits + static_cast<std::uint32_t>(slot), bits ^ 0x13579bdfU,
                bits ^ 0x2468ace0U, bits + 17U};
        }
        asm volatile("" ::: "memory");
    }
}

__device__ __forceinline__ void bsm_store_stream(
        volatile std::uint32_t* shared_words, int lane, float token) {
    #pragma unroll 1
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        const std::uint32_t bits = __float_as_uint(
            token + static_cast<float>(iteration) * 0.0000009536743164f);
        #pragma unroll
        for (int slot = 0; slot < kStaticOpsPerIteration; ++slot) {
            shared_words[slot * kExpectedWaveSize + lane] = (
                bits ^ static_cast<std::uint32_t>(slot * 0x13579bdfU));
        }
    }
}

template <CrossSlot cross>
__device__ __forceinline__ void cross_stream(
        Vector128* output, volatile std::uint32_t* shared_words,
        std::size_t logical_thread, std::size_t logical_threads,
        int lane, float token) {
    if constexpr (cross == CrossSlot::kGvm) {
        gvm_store_stream(output, logical_thread, logical_threads, token);
    } else if constexpr (cross == CrossSlot::kBsm) {
        bsm_store_stream(shared_words, lane, token);
    } else {
        __builtin_trap();
    }
}

template <Slot2 slot2, CrossSlot cross, Mode mode>
__device__ __forceinline__ void cross_slot_body(
        Vector128* output, float* sink,
        volatile std::uint32_t* shared_words) {
    const int wave = static_cast<int>(threadIdx.x) / kExpectedWaveSize;
    const int lane = static_cast<int>(threadIdx.x) & (kExpectedWaveSize - 1);
    const std::size_t logical_thread = (
        static_cast<std::size_t>(blockIdx.x) * kExpectedWaveSize + lane);
    const std::size_t logical_threads = (
        static_cast<std::size_t>(gridDim.x) * kExpectedWaveSize);
    const float seed = 0.125f + static_cast<float>(lane) * 0.0009765625f;

    if constexpr (mode == Mode::kSlot2Isolated) {
        if (wave == 0) sink[logical_thread] = slot2_stream<slot2>(seed);
    } else if constexpr (mode == Mode::kCrossIsolated) {
        if (wave == 1) {
            cross_stream<cross>(
                output, shared_words, logical_thread, logical_threads,
                lane, seed);
        }
    } else if constexpr (mode == Mode::kIndependent) {
        if (wave == 0) {
            sink[logical_thread] = slot2_stream<slot2>(seed);
        } else {
            cross_stream<cross>(
                output, shared_words, logical_thread, logical_threads,
                lane, seed);
        }
    } else if constexpr (mode == Mode::kDependent) {
        if (wave == 0) {
            const float token = slot2_stream<slot2>(seed);
            sink[logical_thread] = token;
            cross_stream<cross>(
                output, shared_words, logical_thread, logical_threads,
                lane, token);
        }
    } else {
        __builtin_trap();
    }
}

#define DEFINE_GVM_KERNEL(NAME, SLOT2_ENUM, MODE_ENUM)                    \
extern "C" __global__ void NAME(Vector128* output, float* sink) {         \
    extern __shared__ volatile std::uint32_t shared_words[];               \
    cross_slot_body<SLOT2_ENUM, CrossSlot::kGvm, MODE_ENUM>(               \
        output, sink, shared_words);                                       \
}

#define DEFINE_BSM_KERNEL(NAME, SLOT2_ENUM, MODE_ENUM)                    \
extern "C" __global__ void NAME(Vector128* output, float* sink) {         \
    extern __shared__ volatile std::uint32_t shared_words[];               \
    cross_slot_body<SLOT2_ENUM, CrossSlot::kBsm, MODE_ENUM>(               \
        output, sink, shared_words);                                       \
}

DEFINE_GVM_KERNEL(c500_cross_mma_gvm_slot2, Slot2::kMma, Mode::kSlot2Isolated)
DEFINE_GVM_KERNEL(c500_cross_mma_gvm_cross, Slot2::kMma, Mode::kCrossIsolated)
DEFINE_GVM_KERNEL(c500_cross_mma_gvm_independent, Slot2::kMma, Mode::kIndependent)
DEFINE_GVM_KERNEL(c500_cross_mma_gvm_dependent, Slot2::kMma, Mode::kDependent)
DEFINE_GVM_KERNEL(c500_cross_mte_gvm_slot2, Slot2::kMte, Mode::kSlot2Isolated)
DEFINE_GVM_KERNEL(c500_cross_mte_gvm_cross, Slot2::kMte, Mode::kCrossIsolated)
DEFINE_GVM_KERNEL(c500_cross_mte_gvm_independent, Slot2::kMte, Mode::kIndependent)
DEFINE_GVM_KERNEL(c500_cross_mte_gvm_dependent, Slot2::kMte, Mode::kDependent)
DEFINE_BSM_KERNEL(c500_cross_mma_bsm_slot2, Slot2::kMma, Mode::kSlot2Isolated)
DEFINE_BSM_KERNEL(c500_cross_mma_bsm_cross, Slot2::kMma, Mode::kCrossIsolated)
DEFINE_BSM_KERNEL(c500_cross_mma_bsm_independent, Slot2::kMma, Mode::kIndependent)
DEFINE_BSM_KERNEL(c500_cross_mma_bsm_dependent, Slot2::kMma, Mode::kDependent)
DEFINE_BSM_KERNEL(c500_cross_mte_bsm_slot2, Slot2::kMte, Mode::kSlot2Isolated)
DEFINE_BSM_KERNEL(c500_cross_mte_bsm_cross, Slot2::kMte, Mode::kCrossIsolated)
DEFINE_BSM_KERNEL(c500_cross_mte_bsm_independent, Slot2::kMte, Mode::kIndependent)
DEFINE_BSM_KERNEL(c500_cross_mte_bsm_dependent, Slot2::kMte, Mode::kDependent)

#undef DEFINE_GVM_KERNEL
#undef DEFINE_BSM_KERNEL

const ModeSpec& parse_mode(const char* cell, const char* mode) {
    for (const ModeSpec& spec : kModes) {
        if (std::strcmp(spec.cell, cell) == 0
                && std::strcmp(spec.name, mode) == 0) return spec;
    }
    std::fprintf(stderr, "unsupported cross-slot cell/mode: %s/%s\n",
                 cell, mode);
    std::exit(EXIT_FAILURE);
}

int parse_nonnegative_int(const char* text, const char* label) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > 2147483647L) {
        std::fprintf(stderr, "%s must be a non-negative integer\n", label);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

void require_device_contract(int device_id, mcDeviceProp_t* properties) {
    CHECK(mcSetDevice(device_id));
    CHECK(mcGetDeviceProperties(properties, device_id));
    if (std::strstr(properties->name, "C500") == nullptr
            || properties->multiProcessorCount != kExpectedApCount
            || properties->warpSize != kExpectedWaveSize
            || properties->l2CacheSize != kExpectedL2Bytes) {
        std::fprintf(stderr, "expected MetaX C500 AP=104 wave=64 L2=8MiB\n");
        std::exit(EXIT_FAILURE);
    }
}

#define DISPATCH_BY_ID(ACTION)                                            \
    ACTION(c500_cross_mma_gvm_slot2)                                      \
    ACTION(c500_cross_mma_gvm_cross)                                      \
    ACTION(c500_cross_mma_gvm_independent)                                \
    ACTION(c500_cross_mma_gvm_dependent)                                  \
    ACTION(c500_cross_mte_gvm_slot2)                                      \
    ACTION(c500_cross_mte_gvm_cross)                                      \
    ACTION(c500_cross_mte_gvm_independent)                                \
    ACTION(c500_cross_mte_gvm_dependent)                                  \
    ACTION(c500_cross_mma_bsm_slot2)                                      \
    ACTION(c500_cross_mma_bsm_cross)                                      \
    ACTION(c500_cross_mma_bsm_independent)                                \
    ACTION(c500_cross_mma_bsm_dependent)                                  \
    ACTION(c500_cross_mte_bsm_slot2)                                      \
    ACTION(c500_cross_mte_bsm_cross)                                      \
    ACTION(c500_cross_mte_bsm_independent)                                \
    ACTION(c500_cross_mte_bsm_dependent)

void launch(const ModeSpec& spec, Vector128* output, float* sink) {
#define LAUNCH(NAME)                                                       \
    if (std::strcmp(spec.kernel_identity, #NAME) == 0) {                   \
        NAME<<<kGridBlocks, kBlockThreads, kDynamicSharedBytes>>>(         \
            output, sink);                                                 \
    } else
    DISPATCH_BY_ID(LAUNCH) {
        std::fprintf(stderr, "unbound cross-slot kernel: %s\n",
                     spec.kernel_identity);
        std::exit(EXIT_FAILURE);
    }
#undef LAUNCH
    CHECK(mcGetLastError());
}

void read_resources(
        const ModeSpec& spec, mcFuncAttributes* attributes,
        int* resident_blocks_per_ap) {
#define READ(NAME)                                                         \
    if (std::strcmp(spec.kernel_identity, #NAME) == 0) {                   \
        CHECK(mcFuncGetAttributes(                                        \
            attributes, reinterpret_cast<const void*>(NAME)));            \
        CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(                \
            resident_blocks_per_ap, NAME, kBlockThreads,                   \
            kDynamicSharedBytes));                                         \
        return;                                                            \
    }
    DISPATCH_BY_ID(READ)
#undef READ
    std::fprintf(stderr, "unbound cross-slot kernel: %s\n",
                 spec.kernel_identity);
    std::exit(EXIT_FAILURE);
}

#undef DISPATCH_BY_ID

std::uint64_t monotonic_raw_ns() {
    timespec value {};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) {
        std::perror("clock_gettime");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL
        + static_cast<std::uint64_t>(value.tv_nsec);
}

void run_mode(
        const ModeSpec& spec, int device_id, bool profile_once,
        bool controlled_event, int attach_wait_seconds) {
    mcDeviceProp_t properties {};
    require_device_contract(device_id, &properties);
    mcFuncAttributes attributes {};
    int resident_blocks_per_ap = 0;
    read_resources(spec, &attributes, &resident_blocks_per_ap);
    constexpr std::size_t expected_static_shared_bytes = 0;
    if (attributes.numRegs <= 0
            || attributes.sharedSizeBytes != expected_static_shared_bytes
            || resident_blocks_per_ap != 1) {
        std::fprintf(
            stderr,
            "cross-slot resources violate registers/shared/one-CTA contract: "
            "kernel=%s numRegs=%d sharedSizeBytes=%zu "
            "expectedStaticSharedSizeBytes=%zu dynamicSharedSizeBytes=%d "
            "residentBlocksPerAp=%d expectedResidentBlocksPerAp=1\n",
            spec.kernel_identity, attributes.numRegs,
            static_cast<std::size_t>(attributes.sharedSizeBytes),
            expected_static_shared_bytes, kDynamicSharedBytes,
            resident_blocks_per_ap);
        std::exit(EXIT_FAILURE);
    }
    const std::size_t logical_threads = (
        static_cast<std::size_t>(kGridBlocks) * kExpectedWaveSize);
    const std::size_t output_vectors = (
        logical_threads * kAddressRounds * kStaticOpsPerIteration);
    Vector128* output = nullptr;
    float* sink = nullptr;
    CHECK(mcMalloc(&output, output_vectors * sizeof(Vector128)));
    CHECK(mcMalloc(&sink, logical_threads * sizeof(float)));
    CHECK(mcMemset(output, 0, output_vectors * sizeof(Vector128)));
    CHECK(mcMemset(sink, 0, logical_threads * sizeof(float)));
    for (int warmup = 0; warmup < kWarmupLaunches; ++warmup) {
        launch(spec, output, sink);
    }
    CHECK(mcDeviceSynchronize());

    std::vector<float> event_ms;
    std::vector<std::uint64_t> starts;
    std::vector<std::uint64_t> ends;
    if (profile_once) {
        if (attach_wait_seconds != kAttachWaitSeconds) {
            std::fprintf(stderr, "attach wait must be exactly 15 seconds\n");
            std::exit(EXIT_FAILURE);
        }
        std::this_thread::sleep_for(std::chrono::seconds(attach_wait_seconds));
        CHECK(mcProfilerStart());
        launch(spec, output, sink);
        CHECK(mcDeviceSynchronize());
        CHECK(mcProfilerStop());
    } else {
        event_ms.reserve(kEventTrials);
        starts.reserve(kEventTrials);
        ends.reserve(kEventTrials);
        for (int trial = 0; trial < kEventTrials; ++trial) {
            if (controlled_event && raise(SIGSTOP) != 0) {
                std::perror("raise(SIGSTOP)");
                std::exit(EXIT_FAILURE);
            }
            mcEvent_t start;
            mcEvent_t end;
            CHECK(mcEventCreate(&start));
            CHECK(mcEventCreate(&end));
            starts.push_back(monotonic_raw_ns());
            CHECK(mcEventRecord(start));
            launch(spec, output, sink);
            CHECK(mcEventRecord(end));
            CHECK(mcEventSynchronize(end));
            ends.push_back(monotonic_raw_ns());
            float elapsed_ms = 0.0f;
            CHECK(mcEventElapsedTime(&elapsed_ms, start, end));
            CHECK(mcEventDestroy(end));
            CHECK(mcEventDestroy(start));
            if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f) {
                std::fprintf(stderr, "device-event duration is invalid\n");
                std::exit(EXIT_FAILURE);
            }
            event_ms.push_back(elapsed_ms);
        }
    }

    std::vector<float> host_sink(logical_threads);
    CHECK(mcMemcpy(host_sink.data(), sink, logical_threads * sizeof(float),
                   mcMemcpyDeviceToHost));
    double checksum = 0.0;
    for (float value : host_sink) {
        if (!std::isfinite(value)) {
            std::fprintf(stderr, "cross-slot sink is non-finite\n");
            std::exit(EXIT_FAILURE);
        }
        checksum += value;
    }
    if (!std::isfinite(checksum)) {
        std::fprintf(stderr, "cross-slot checksum is invalid\n");
        std::exit(EXIT_FAILURE);
    }

    std::vector<float> sorted = event_ms;
    std::sort(sorted.begin(), sorted.end());
    const float minimum = profile_once ? 0.0f : sorted.front();
    const float median = profile_once ? 0.0f : sorted[sorted.size() / 2];
    const float maximum = profile_once ? 0.0f : sorted.back();
    std::printf(
        "metadata,3,%s,metax_c500,cross_slot_smoke,0,0,%d,%d,%d,%d\n",
        profile_once ? "profile" : "event", device_id,
        properties.multiProcessorCount, properties.warpSize,
        properties.l2CacheSize);
    std::printf(
        "sample,%s,%s,%s,%d,%d,%d,%d,%d,%d,%.9f,%.9f,%.9f,%.17g,%d,%zu,%d,%d\n",
        spec.cell, spec.name, spec.kernel_identity, kGridBlocks,
        kBlockThreads, kWavesPerBlock, kIterations,
        kStaticOpsPerIteration, profile_once ? 0 : kEventTrials,
        minimum, median, maximum, checksum, attributes.numRegs,
        attributes.sharedSizeBytes, kDynamicSharedBytes,
        resident_blocks_per_ap);
    if (!profile_once) {
        for (int trial = 0; trial < kEventTrials; ++trial) {
            std::printf("trial,%d,%llu,%llu,%.9f\n", trial + 1,
                static_cast<unsigned long long>(starts[trial]),
                static_cast<unsigned long long>(ends[trial]), event_ms[trial]);
        }
    }
    std::fflush(stdout);
    CHECK(mcFree(sink));
    CHECK(mcFree(output));
}

void print_inspection(int device_id) {
    mcDeviceProp_t properties {};
    require_device_contract(device_id, &properties);
    std::printf("inspect,3,metax_c500,%d,%s,%d,%d,%d,%d,%d,%d\n",
                device_id, properties.name, properties.multiProcessorCount,
                properties.warpSize, properties.l2CacheSize, kGridBlocks,
                kBlockThreads, kIterations);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 4 && std::strcmp(argv[1], "--inspect") == 0
            && std::strcmp(argv[2], "--device-id") == 0) {
        print_inspection(parse_nonnegative_int(argv[3], "device-id"));
        return EXIT_SUCCESS;
    }
    const bool event_mode = (
        argc == 8 && (std::strcmp(argv[1], "--event") == 0
        || std::strcmp(argv[1], "--event-controlled") == 0)
        && std::strcmp(argv[2], "--device-id") == 0
        && std::strcmp(argv[4], "--cell") == 0
        && std::strcmp(argv[6], "--mode") == 0);
    const bool profile_mode = (
        argc == 10 && std::strcmp(argv[1], "--profile-once") == 0
        && std::strcmp(argv[2], "--device-id") == 0
        && std::strcmp(argv[4], "--cell") == 0
        && std::strcmp(argv[6], "--mode") == 0
        && std::strcmp(argv[8], "--attach-wait-seconds") == 0);
    if (!event_mode && !profile_mode) {
        std::fprintf(stderr, "invalid cross-slot command\n");
        return EXIT_FAILURE;
    }
    const int device_id = parse_nonnegative_int(argv[3], "device-id");
    const ModeSpec& spec = parse_mode(argv[5], argv[7]);
    const int attach_wait_seconds = profile_mode
        ? parse_nonnegative_int(argv[9], "attach-wait-seconds") : 0;
    run_mode(
        spec, device_id, profile_mode,
        event_mode && std::strcmp(argv[1], "--event-controlled") == 0,
        attach_wait_seconds);
    return EXIT_SUCCESS;
}
