/*
 * MetaX C500 direct-to-shared gvmcnt threshold microbenchmark.
 *
 * This is an isolated memory-completion experiment, not a GEMM: every
 * 256-thread CTA issues an eleven-request (prime and deliberately unrelated
 * to production tiles) global-to-shared window. Each round executes a
 * bsmcnt(0)-only completion and a gvmcnt(N)+bsmcnt(0) completion with their
 * order alternated by CTA and round. Per-CTA clock64 spans therefore isolate
 * source-visible gvm thresholds from the common shared-memory completion
 * with identical traffic, allocation, CTA count, and code shape. No
 * production duration is read.
 *
 * Build inside the MACA 3.7.2 C500 profiling container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -std=c++17 \
 *       gvmcnt_threshold_service.cpp -o gvmcnt_threshold_service
 */

#include <mc_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(call) do {                                                   \
    mcError_t error = (call);                                              \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",                \
                     __FILE__, __LINE__, mcGetErrorString(error));        \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

namespace {

constexpr std::size_t kSourceBytes = 128ULL * 1024 * 1024;
constexpr int kBlockThreads = 256;
constexpr int kExpectedAps = 104;
constexpr int kExpectedWaveSize = 64;
constexpr int kRequestWindow = 11;
constexpr int kRounds = 13;
constexpr int kSamplesPerThreshold = 17;
constexpr std::size_t kDynamicSharedBytes = 64 * 1024;

using v4u = __NATIVE_VECTOR__(4, std::uint32_t);

static_assert(sizeof(v4u) == 16, "direct load must be 128 bits");
static_assert(kBlockThreads % kExpectedWaveSize == 0,
              "CTA must contain complete wave64 groups");
static_assert(kRequestWindow == 11,
              "window must remain non-production and prime");
static_assert(2ULL * kRounds * kRequestWindow * kExpectedAps
                  * kBlockThreads * sizeof(v4u) <= kSourceBytes,
              "paired windows must use distinct source traffic");
static_assert(kRequestWindow * kBlockThreads * sizeof(v4u)
                  <= kDynamicSharedBytes,
              "all direct-load destinations must fit in shared memory");

int parse_nonnegative_int(const char* text, const char* label) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
            || value < 0 || value > INT_MAX) {
        std::fprintf(stderr, "%s must be a non-negative integer\n", label);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

__host__ __device__ __forceinline__ std::uint32_t mix(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

__global__ void initialize_source(v4u* source, long long elements) {
    const long long index = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long current = index; current < elements; current += stride) {
        const std::uint32_t value = mix(static_cast<std::uint32_t>(current));
        source[current] = {value, mix(value), mix(value ^ 0x6a09e667U),
                           mix(value ^ 0xbb67ae85U)};
    }
}

__device__ __forceinline__ void direct_load(void* destination,
                                             const v4u* source) {
    __builtin_mxc_ldg_b128_bsm(
        destination, const_cast<v4u*>(source), 0, -1, true, true, false,
        true);
}

template <int RemainingRequests>
__global__ void gvmcnt_threshold_cycles(
        const v4u* source, unsigned long long* paired_cycles, v4u* sink) {
    extern __shared__ std::uint8_t shared_bytes[];
    const int local_thread = static_cast<int>(threadIdx.x);
    const long long global_thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + local_thread);
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    unsigned long long baseline_cycles = 0;
    unsigned long long threshold_cycles = 0;
    for (int round = 0; round < kRounds; ++round) {
        const bool baseline_first = ((static_cast<int>(blockIdx.x) + round) & 1)
                == 0;
        #pragma unroll
        for (int phase = 0; phase < 2; ++phase) {
            const bool is_baseline = phase == (baseline_first ? 0 : 1);
            const int source_window = 2 * round + phase;
            const unsigned long long start = clock64();
            #pragma unroll
            for (int request = 0; request < kRequestWindow; ++request) {
                const long long source_index = (
                    (static_cast<long long>(source_window) * kRequestWindow
                        + request) * stride + global_thread);
                const std::size_t destination = static_cast<std::size_t>(
                    (request * kBlockThreads + local_thread) * sizeof(v4u));
                direct_load(shared_bytes + destination, source + source_index);
            }
            if (!is_baseline) {
                __builtin_mxc_arrive(64 + RemainingRequests);
            }
            __builtin_mxc_arrive_bsmcnt(0);
            __builtin_mxc_barrier_inst();
            const unsigned long long elapsed = clock64() - start;
            if (is_baseline) {
                baseline_cycles += elapsed;
            } else {
                threshold_cycles += elapsed;
            }
        }
    }
    const std::size_t final_offset = static_cast<std::size_t>(
        ((kRequestWindow - 1) * kBlockThreads + local_thread) * sizeof(v4u));
    sink[global_thread] = *reinterpret_cast<v4u*>(shared_bytes + final_offset);
    if (local_thread == 0) {
        paired_cycles[2 * blockIdx.x] = baseline_cycles;
        paired_cycles[2 * blockIdx.x + 1] = threshold_cycles;
    }
}

void launch_threshold(int remaining, const v4u* source,
                      unsigned long long* cycles, v4u* sink) {
    if (remaining == 2) {
        gvmcnt_threshold_cycles<2><<<
            kExpectedAps, kBlockThreads, kDynamicSharedBytes>>>(
                source, cycles, sink);
    } else if (remaining == 4) {
        gvmcnt_threshold_cycles<4><<<
            kExpectedAps, kBlockThreads, kDynamicSharedBytes>>>(
                source, cycles, sink);
    } else if (remaining == 5) {
        gvmcnt_threshold_cycles<5><<<
            kExpectedAps, kBlockThreads, kDynamicSharedBytes>>>(
                source, cycles, sink);
    } else if (remaining == 6) {
        gvmcnt_threshold_cycles<6><<<
            kExpectedAps, kBlockThreads, kDynamicSharedBytes>>>(
                source, cycles, sink);
    } else {
        std::fprintf(stderr, "unsupported gvmcnt remaining count %d\n",
                     remaining);
        std::exit(EXIT_FAILURE);
    }
}

unsigned long long median(std::vector<unsigned long long> values) {
    if (values.empty()) {
        std::fprintf(stderr, "cannot take an empty median\n");
        std::exit(EXIT_FAILURE);
    }
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5 || std::strcmp(argv[1], "--device-id") != 0
            || std::strcmp(argv[3], "--trial") != 0) {
        std::fprintf(stderr, "usage: %s --device-id DEVICE --trial TRIAL\n",
                     argv[0]);
        return EXIT_FAILURE;
    }
    const int device_id = parse_nonnegative_int(argv[2], "device-id");
    const int trial = parse_nonnegative_int(argv[4], "trial");
    if (trial < 1 || trial > 3) {
        std::fprintf(stderr, "trial must be in [1, 3]\n");
        return EXIT_FAILURE;
    }
    CHECK(mcSetDevice(device_id));
    mcDeviceProp_t properties {};
    CHECK(mcGetDeviceProperties(&properties, device_id));
    if (std::strstr(properties.name, "C500") == nullptr
            || properties.multiProcessorCount != kExpectedAps
            || properties.warpSize != kExpectedWaveSize) {
        std::fprintf(stderr, "unexpected device %s aps=%d wave=%d\n",
                     properties.name, properties.multiProcessorCount,
                     properties.warpSize);
        return EXIT_FAILURE;
    }
    v4u* source = nullptr;
    v4u* sink = nullptr;
    unsigned long long* cycles = nullptr;
    CHECK(mcMalloc(&source, kSourceBytes));
    CHECK(mcMalloc(&sink, static_cast<std::size_t>(kExpectedAps)
                         * kBlockThreads * sizeof(v4u)));
    CHECK(mcMalloc(&cycles, 2ULL * kExpectedAps
                         * sizeof(unsigned long long)));
    initialize_source<<<kExpectedAps, kBlockThreads>>>(
        source, static_cast<long long>(kSourceBytes / sizeof(v4u)));
    CHECK(mcGetLastError());
    CHECK(mcDeviceSynchronize());
    for (int remaining : std::array<int, 4>{2, 4, 5, 6}) {
        launch_threshold(remaining, source, cycles, sink);
    }
    CHECK(mcGetLastError());
    CHECK(mcDeviceSynchronize());
    std::vector<std::pair<int, int>> plan;
    for (int remaining : std::array<int, 4>{2, 4, 5, 6}) {
        for (int sample = 0; sample < kSamplesPerThreshold; ++sample) {
            plan.emplace_back(remaining, sample);
        }
    }
    std::mt19937 generator(961748927U + static_cast<unsigned>(trial));
    std::shuffle(plan.begin(), plan.end(), generator);
    std::vector<unsigned long long> host_cycles(2 * kExpectedAps);
    std::vector<unsigned long long> baseline_cycles(kExpectedAps);
    std::vector<unsigned long long> threshold_cycles(kExpectedAps);
    std::vector<long long> paired_deltas(kExpectedAps);
    std::printf(
        "metadata,3,MetaX C500,%d,%d,%d,%d,%d,%d,%zu\n",
        device_id, properties.multiProcessorCount, properties.warpSize,
        kBlockThreads, kRequestWindow, kRounds, kDynamicSharedBytes);
    for (std::size_t order = 0; order < plan.size(); ++order) {
        const int remaining = plan[order].first;
        launch_threshold(remaining, source, cycles, sink);
        CHECK(mcGetLastError());
        CHECK(mcMemcpy(host_cycles.data(), cycles,
                       host_cycles.size() * sizeof(unsigned long long),
                       mcMemcpyDeviceToHost));
        if (std::any_of(host_cycles.begin(), host_cycles.end(),
                        [](unsigned long long value) { return value == 0; })) {
            std::fprintf(stderr, "zero clock64 span from threshold kernel\n");
            return EXIT_FAILURE;
        }
        for (int cta = 0; cta < kExpectedAps; ++cta) {
            baseline_cycles[cta] = host_cycles[2 * cta];
            threshold_cycles[cta] = host_cycles[2 * cta + 1];
            paired_deltas[cta] = static_cast<long long>(threshold_cycles[cta])
                    - static_cast<long long>(baseline_cycles[cta]);
        }
        const auto baseline_bounds = std::minmax_element(
            baseline_cycles.begin(), baseline_cycles.end());
        const auto threshold_bounds = std::minmax_element(
            threshold_cycles.begin(), threshold_cycles.end());
        const auto delta_bounds = std::minmax_element(
            paired_deltas.begin(), paired_deltas.end());
        const long long delta_min = *delta_bounds.first;
        const long long delta_max = *delta_bounds.second;
        std::sort(paired_deltas.begin(), paired_deltas.end());
        std::printf(
            "sample,%zu,%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%llu,%lld,%lld,%lld\n",
            order, remaining, plan[order].second, kExpectedAps,
            *baseline_bounds.first, median(baseline_cycles),
            *baseline_bounds.second, *threshold_bounds.first,
            median(threshold_cycles), *threshold_bounds.second,
            delta_min, paired_deltas[paired_deltas.size() / 2], delta_max);
    }
    CHECK(mcFree(cycles));
    CHECK(mcFree(sink));
    CHECK(mcFree(source));
    return EXIT_SUCCESS;
}
