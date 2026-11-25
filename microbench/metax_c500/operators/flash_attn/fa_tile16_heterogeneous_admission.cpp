/*
 * MetaX C500 heterogeneous direct-CTA admission mechanism probe.
 *
 * This is not FlashAttention.  It launches the resource-matched 64-thread,
 * 8-KiB-dynamic-shared-memory CTA used by the C500 tile16 trait, but each
 * CTA only holds its resident slot for a fixed synthetic source-work unit
 * count.  There is no attention tensor, MMA, softmax, production data, or
 * duration input.  The exact mode contains 608/256/160 source-ordinal CTA
 * runs with 8/16/32 work units; the holdout contains the same multiset in a
 * four-period interleaving.  Comparing their recorded admission windows
 * tests scheduler semantics, not an operator latency.
 */

#include <mc_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do {                                                   \
    const mcError_t error = (call);                                        \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "%s failed: %s\n", #call,                   \
                     mcGetErrorString(error));                             \
        std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
} while (0)

namespace {

constexpr int kExpectedApCount = 104;
constexpr int kExpectedWaveSize = 64;
constexpr int kThreads = kExpectedWaveSize;
constexpr int kDynamicSharedBytes = 8 * 1024;
constexpr int kGrid = 1024;
constexpr int kShortCtas = 608;
constexpr int kMediumCtas = 256;
constexpr int kLongCtas = 160;
constexpr int kShortWorkUnits = 8;
constexpr int kMediumWorkUnits = 16;
constexpr int kLongWorkUnits = 32;
constexpr unsigned long long kCyclesPerWorkUnit = 562500ULL;
constexpr int kRegisterPressureScalars = 128;
constexpr int kWarmups = 1;
constexpr int kTrials = 5;

static_assert(kShortCtas + kMediumCtas + kLongCtas == kGrid);
static_assert(kShortWorkUnits * 2 == kMediumWorkUnits);
static_assert(kMediumWorkUnits * 2 == kLongWorkUnits);

struct Counters {
    unsigned int active;
    unsigned int peak;
    unsigned int entered;
    unsigned int finished;
};

__device__ __forceinline__ float register_pressure(float seed) {
    float values[kRegisterPressureScalars];
#pragma unroll
    for (int index = 0; index < kRegisterPressureScalars; ++index) {
        const float addend = static_cast<float>(index + 1) * 0.000001f;
        asm volatile("add_f32 %0, %1, %2"
                     : "=r"(values[index]) : "r"(seed), "r"(addend));
    }
    float checksum = 0.0f;
#pragma unroll
    for (int index = 0; index < kRegisterPressureScalars; ++index) {
        asm volatile("add_f32 %0, %1, %2"
                     : "=r"(checksum) : "r"(checksum), "r"(values[index]));
    }
    return checksum;
}

__host__ __device__ __forceinline__ int source_work_units(
        unsigned int ordinal, int mode) {
    if (mode == 0) {
        if (ordinal < static_cast<unsigned int>(kShortCtas)) {
            return kShortWorkUnits;
        }
        if (ordinal < static_cast<unsigned int>(kShortCtas + kMediumCtas)) {
            return kMediumWorkUnits;
        }
        return kLongWorkUnits;
    }
    if (mode == 1) {
        // Four exact periods: 4 * (19 short + 8 medium + 5 long).
        const unsigned int within_period = ordinal & 31U;
        if (within_period < 19U) {
            return kShortWorkUnits;
        }
        if (within_period < 27U) {
            return kMediumWorkUnits;
        }
        return kLongWorkUnits;
    }
    __builtin_trap();
    return 0;
}

__global__ __launch_bounds__(kThreads) void heterogeneous_admission(
        Counters* counters, unsigned int* entry_tickets,
        unsigned int* finish_tickets, unsigned long long* start_ticks,
        unsigned long long* end_ticks, float* sinks, int mode) {
    extern __shared__ unsigned char dynamic_shared[];
    const float checksum = register_pressure(
        1.0f + static_cast<float>(threadIdx.x) * 0.000003814697265625f);
    dynamic_shared[threadIdx.x] = static_cast<unsigned char>(threadIdx.x);
    __syncthreads();
    if (threadIdx.x != 0) {
        return;
    }

    const unsigned int ordinal = static_cast<unsigned int>(blockIdx.x);
    const unsigned int ticket = atomicAdd(&counters->entered, 1U);
    entry_tickets[ordinal] = ticket;
    const unsigned int active = atomicAdd(&counters->active, 1U) + 1U;
    atomicMax(&counters->peak, active);
    const unsigned long long start = clock64();
    const unsigned long long hold_cycles = kCyclesPerWorkUnit
        * static_cast<unsigned long long>(source_work_units(ordinal, mode));
    while (clock64() - start < hold_cycles) {
        asm volatile("" ::: "memory");
    }
    const unsigned long long end = clock64();
    atomicSub(&counters->active, 1U);
    start_ticks[ordinal] = start;
    end_ticks[ordinal] = end;
    finish_tickets[ordinal] = atomicAdd(&counters->finished, 1U);
    sinks[ordinal] = checksum + static_cast<float>(ticket & 1U);
}

int parse_device(const char* text) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > 3) {
        std::fprintf(stderr, "DEVICE must be in [0, 3]\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

void require_device(int device) {
    CHECK(mcSetDevice(device));
    mcDeviceProp_t properties {};
    CHECK(mcGetDeviceProperties(&properties, device));
    if (properties.multiProcessorCount != kExpectedApCount
            || properties.warpSize != kExpectedWaveSize) {
        std::fprintf(stderr, "expected C500 AP=104 and wave=64\n");
        std::exit(EXIT_FAILURE);
    }
}

void require_permutation(
        const std::vector<unsigned int>& tickets, const char* label) {
    std::vector<unsigned int> sorted = tickets;
    std::sort(sorted.begin(), sorted.end());
    for (unsigned int index = 0; index < static_cast<unsigned int>(kGrid);
         ++index) {
        if (sorted[index] != index) {
            std::fprintf(stderr, "%s ticket coverage is invalid\n", label);
            std::exit(EXIT_FAILURE);
        }
    }
}

void run_mode(
        int mode, Counters* counters, unsigned int* entry_tickets,
        unsigned int* finish_tickets, unsigned long long* start_ticks,
        unsigned long long* end_ticks, float* sinks) {
    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        CHECK(mcMemset(counters, 0, sizeof(Counters)));
        heterogeneous_admission<<<kGrid, kThreads, kDynamicSharedBytes>>>(
            counters, entry_tickets, finish_tickets, start_ticks, end_ticks,
            sinks, mode);
        CHECK(mcGetLastError());
        CHECK(mcDeviceSynchronize());
    }

    std::vector<unsigned int> host_entry(kGrid);
    std::vector<unsigned int> host_finish(kGrid);
    std::vector<unsigned long long> host_start(kGrid);
    std::vector<unsigned long long> host_end(kGrid);
    std::vector<float> host_sinks(kGrid);
    for (int trial = 0; trial < kTrials; ++trial) {
        CHECK(mcMemset(counters, 0, sizeof(Counters)));
        heterogeneous_admission<<<kGrid, kThreads, kDynamicSharedBytes>>>(
            counters, entry_tickets, finish_tickets, start_ticks, end_ticks,
            sinks, mode);
        CHECK(mcGetLastError());
        CHECK(mcDeviceSynchronize());

        Counters observed {};
        CHECK(mcMemcpy(&observed, counters, sizeof(Counters),
                       mcMemcpyDeviceToHost));
        CHECK(mcMemcpy(host_entry.data(), entry_tickets,
                       kGrid * sizeof(unsigned int), mcMemcpyDeviceToHost));
        CHECK(mcMemcpy(host_finish.data(), finish_tickets,
                       kGrid * sizeof(unsigned int), mcMemcpyDeviceToHost));
        CHECK(mcMemcpy(host_start.data(), start_ticks,
                       kGrid * sizeof(unsigned long long),
                       mcMemcpyDeviceToHost));
        CHECK(mcMemcpy(host_end.data(), end_ticks,
                       kGrid * sizeof(unsigned long long),
                       mcMemcpyDeviceToHost));
        CHECK(mcMemcpy(host_sinks.data(), sinks, kGrid * sizeof(float),
                       mcMemcpyDeviceToHost));
        if (observed.active != 0U || observed.entered != kGrid
                || observed.finished != kGrid || observed.peak < 832U) {
            std::fprintf(stderr, "admission completion invariant failed\n");
            std::exit(EXIT_FAILURE);
        }
        require_permutation(host_entry, "entry");
        require_permutation(host_finish, "finish");
        for (int ordinal = 0; ordinal < kGrid; ++ordinal) {
            const unsigned long long elapsed = (
                host_end[ordinal] - host_start[ordinal]);
            const unsigned long long required = kCyclesPerWorkUnit
                * static_cast<unsigned long long>(source_work_units(
                    static_cast<unsigned int>(ordinal), mode));
            if (host_end[ordinal] < host_start[ordinal]
                    || elapsed < required || host_sinks[ordinal] <= 0.0f) {
                std::fprintf(stderr, "CTA trace coverage is invalid\n");
                std::exit(EXIT_FAILURE);
            }
            std::printf(
                "record,mode=%s,trial=%d,ordinal=%d,work_units=%d,"
                "entry=%u,finish=%u,start=%llu,end=%llu\n",
                mode == 0 ? "source_runs" : "interleaved_holdout", trial,
                ordinal, source_work_units(static_cast<unsigned int>(ordinal),
                                            mode), host_entry[ordinal],
                host_finish[ordinal], host_start[ordinal], host_end[ordinal]);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s DEVICE\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device = parse_device(argv[1]);
    require_device(device);
    mcFuncAttributes attributes {};
    CHECK(mcFuncGetAttributes(
        &attributes, reinterpret_cast<const void*>(heterogeneous_admission)));
    int resident_blocks = 0;
    CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_blocks, heterogeneous_admission, kThreads,
        kDynamicSharedBytes));
    if (attributes.numRegs <= 0 || attributes.sharedSizeBytes != 0
            || resident_blocks != 8) {
        std::fprintf(stderr, "resource-matched binary contract failed\n");
        return EXIT_FAILURE;
    }
    std::printf(
        "metadata,ap=%d,wave=%d,threads=%d,dynamic_smem=%d,registers=%d,"
        "resident_blocks=%d,grid=%d,source_counts=%d:%d:%d,"
        "work_units=%d:%d:%d,cycles_per_unit=%llu\n",
        kExpectedApCount, kExpectedWaveSize, kThreads, kDynamicSharedBytes,
        attributes.numRegs, resident_blocks, kGrid, kShortCtas, kMediumCtas,
        kLongCtas, kShortWorkUnits, kMediumWorkUnits, kLongWorkUnits,
        static_cast<unsigned long long>(kCyclesPerWorkUnit));

    Counters* counters = nullptr;
    unsigned int* entry_tickets = nullptr;
    unsigned int* finish_tickets = nullptr;
    unsigned long long* start_ticks = nullptr;
    unsigned long long* end_ticks = nullptr;
    float* sinks = nullptr;
    CHECK(mcMalloc(&counters, sizeof(Counters)));
    CHECK(mcMalloc(&entry_tickets, kGrid * sizeof(unsigned int)));
    CHECK(mcMalloc(&finish_tickets, kGrid * sizeof(unsigned int)));
    CHECK(mcMalloc(&start_ticks, kGrid * sizeof(unsigned long long)));
    CHECK(mcMalloc(&end_ticks, kGrid * sizeof(unsigned long long)));
    CHECK(mcMalloc(&sinks, kGrid * sizeof(float)));
    run_mode(0, counters, entry_tickets, finish_tickets, start_ticks,
             end_ticks, sinks);
    run_mode(1, counters, entry_tickets, finish_tickets, start_ticks,
             end_ticks, sinks);
    CHECK(mcFree(sinks));
    CHECK(mcFree(end_ticks));
    CHECK(mcFree(start_ticks));
    CHECK(mcFree(finish_tickets));
    CHECK(mcFree(entry_tickets));
    CHECK(mcFree(counters));
    return EXIT_SUCCESS;
}
