/*
 * MetaX C500 dependent L2/DRAM endpoint-latency microbenchmark.
 *
 * One 16-lane group follows a randomized linked list.  Every lane loads one
 * 64-bit word from the same aligned 128-byte node, so each loop iteration
 * retains one coalesced 128-byte access and a loop-carried next-node
 * dependency.  The lower 32 bits hold the identical next-node index in every
 * lane; randomized upper bits are accumulated and sunk so all lane payloads
 * remain live.  A register recurrence reports a diagnostic control baseline;
 * it is deliberately not subtracted from the dependent-chain endpoint.
 *
 * Below-L2 footprints are fully warmed and repeat their ring for at least
 * 65,536 timed hops.  Larger footprints warm only a short prefix and measure
 * the unrepeated remainder as a DRAM-miss control.  clock64()
 * timestamp ticks are the primary mechanism output; ordinary MACA events
 * provide an independent elapsed-time cross-check.  The tick frequency must
 * be taken from macainfo and verified against event time.  This benchmark does
 * not measure queue depth, request credits, bandwidth, or production-operator
 * latency.
 *
 * Build inside the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -o l2_latency l2_latency.cpp
 */

#include <mc_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#define CHECK(call) do {                                                   \
    mcError_t error = (call);                                              \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",              \
                     __FILE__, __LINE__, mcGetErrorString(error));          \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

constexpr int kLanes = 16;
constexpr int kTransactionBytes = 128;
constexpr int kTrials = 7;
constexpr int kLargeFootprintWarmSteps = 256;
constexpr int kMinimumHotTimedSteps = 65536;

__global__ void warm_chain(
        const std::uint64_t* buffer, int steps, std::uint32_t* sink,
        std::uint32_t* start_node) {
    int lane = static_cast<int>(threadIdx.x);
    if (lane >= kLanes) {
        return;
    }
    volatile const std::uint64_t* live_buffer = buffer;
    std::uint32_t node = 0;
    std::uint32_t checksum = static_cast<std::uint32_t>(lane + 1);
    #pragma unroll 1
    for (int step = 0; step < steps; ++step) {
        std::uint64_t value = live_buffer[
            static_cast<long long>(node) * kLanes + lane];
        node = static_cast<std::uint32_t>(value);
        checksum ^= static_cast<std::uint32_t>(value >> 32);
    }
    sink[lane] = node ^ checksum;
    if (lane == 0) {
        start_node[0] = node;
    }
}

__global__ void timed_chain(
        const std::uint64_t* buffer, const std::uint32_t* start_node,
        int steps, std::uint32_t* sink,
        unsigned long long* elapsed_ticks) {
    int lane = static_cast<int>(threadIdx.x);
    if (lane >= kLanes) {
        return;
    }
    volatile const std::uint64_t* live_buffer = buffer;
    std::uint32_t node = start_node[0];
    asm volatile("" : "+r"(node));
    std::uint32_t checksum = static_cast<std::uint32_t>(lane + 1);
    unsigned long long start = clock64();
    #pragma unroll 1
    for (int step = 0; step < steps; ++step) {
        std::uint64_t value = live_buffer[
            static_cast<long long>(node) * kLanes + lane];
        node = static_cast<std::uint32_t>(value);
        checksum ^= static_cast<std::uint32_t>(value >> 32);
    }
    unsigned long long end = clock64();
    sink[lane] = node ^ checksum;
    if (lane == 0) {
        elapsed_ticks[0] = end - start;
    }
}

__global__ void timed_control(
        int steps, int nodes, std::uint32_t* sink,
        unsigned long long* elapsed_ticks) {
    int lane = static_cast<int>(threadIdx.x);
    if (lane >= kLanes) {
        return;
    }
    std::uint32_t node = 0;
    std::uint32_t checksum = static_cast<std::uint32_t>(lane + 1);
    unsigned long long start = clock64();
    #pragma unroll 1
    for (int step = 0; step < steps; ++step) {
        ++node;
        if (node == static_cast<std::uint32_t>(nodes)) {
            node = 0;
        }
        checksum ^= node;
        asm volatile("" : "+r"(node), "+r"(checksum));
    }
    unsigned long long end = clock64();
    sink[lane] = node ^ checksum;
    if (lane == 0) {
        elapsed_ticks[0] = end - start;
    }
}

struct Timing {
    unsigned long long ticks;
    double event_ms;
};

static Timing launch_timed_chain(
        const std::uint64_t* buffer, const std::uint32_t* start_node,
        int steps, std::uint32_t* sink,
        unsigned long long* elapsed_ticks) {
    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    CHECK(mcEventRecord(start));
    timed_chain<<<1, kLanes>>>(
        buffer, start_node, steps, sink, elapsed_ticks);
    CHECK(mcEventRecord(end));
    CHECK(mcDeviceSynchronize());
    float elapsed_ms = 0.0f;
    CHECK(mcEventElapsedTime(&elapsed_ms, start, end));
    CHECK(mcEventDestroy(start));
    CHECK(mcEventDestroy(end));
    unsigned long long ticks = 0;
    CHECK(mcMemcpy(
        &ticks, elapsed_ticks, sizeof(ticks), mcMemcpyDeviceToHost));
    if (ticks == 0 || !std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f) {
        std::fprintf(stderr,
                     "invalid memory timing ticks=%llu event_ms=%.9g\n",
                     ticks, elapsed_ms);
        std::exit(EXIT_FAILURE);
    }
    return Timing{ticks, static_cast<double>(elapsed_ms)};
}

static Timing launch_timed_control(
        int steps, int nodes, std::uint32_t* sink,
        unsigned long long* elapsed_ticks) {
    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    CHECK(mcEventRecord(start));
    timed_control<<<1, kLanes>>>(steps, nodes, sink, elapsed_ticks);
    CHECK(mcEventRecord(end));
    CHECK(mcDeviceSynchronize());
    float elapsed_ms = 0.0f;
    CHECK(mcEventElapsedTime(&elapsed_ms, start, end));
    CHECK(mcEventDestroy(start));
    CHECK(mcEventDestroy(end));
    unsigned long long ticks = 0;
    CHECK(mcMemcpy(
        &ticks, elapsed_ticks, sizeof(ticks), mcMemcpyDeviceToHost));
    if (ticks == 0 || !std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f) {
        std::fprintf(stderr,
                     "invalid control timing ticks=%llu event_ms=%.9g\n",
                     ticks, elapsed_ms);
        std::exit(EXIT_FAILURE);
    }
    return Timing{ticks, static_cast<double>(elapsed_ms)};
}

static std::vector<std::uint64_t> make_chain(int nodes, int seed) {
    std::vector<int> permutation(nodes);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::mt19937 order_generator(seed);
    std::shuffle(
        permutation.begin(), permutation.end(), order_generator);
    std::mt19937 payload_generator(seed ^ 0x5bd1e995U);
    std::vector<std::uint64_t> buffer(
        static_cast<std::size_t>(nodes) * kLanes);
    for (int position = 0; position < nodes; ++position) {
        int current = permutation[position];
        int next = permutation[(position + 1) % nodes];
        for (int lane = 0; lane < kLanes; ++lane) {
            std::uint64_t payload = (
                static_cast<std::uint64_t>(payload_generator()) << 32);
            buffer[
                static_cast<std::size_t>(current) * kLanes + lane] = (
                    payload | static_cast<std::uint32_t>(next));
        }
    }
    return buffer;
}

struct ChainState {
    std::uint32_t node;
    std::vector<std::uint32_t> sink;
};

static ChainState host_chain_state(
        const std::vector<std::uint64_t>& buffer, int steps,
        std::uint32_t start_node) {
    std::vector<std::uint32_t> checksum(kLanes);
    for (int lane = 0; lane < kLanes; ++lane) {
        checksum[lane] = static_cast<std::uint32_t>(lane + 1);
    }
    std::uint32_t node = start_node;
    for (int step = 0; step < steps; ++step) {
        std::size_t base = static_cast<std::size_t>(node) * kLanes;
        std::uint32_t next = static_cast<std::uint32_t>(buffer[base]);
        for (int lane = 0; lane < kLanes; ++lane) {
            std::uint64_t value = buffer[base + lane];
            if (static_cast<std::uint32_t>(value) != next) {
                std::fprintf(
                    stderr,
                    "host chain lane mismatch step=%d node=%u lane=%d\n",
                    step, node, lane);
                std::exit(EXIT_FAILURE);
            }
            checksum[lane] ^= static_cast<std::uint32_t>(value >> 32);
        }
        node = next;
    }
    std::vector<std::uint32_t> sink(kLanes);
    for (int lane = 0; lane < kLanes; ++lane) {
        sink[lane] = node ^ checksum[lane];
    }
    return ChainState{node, std::move(sink)};
}

static std::vector<std::uint32_t> control_checksum(int steps, int nodes) {
    std::vector<std::uint32_t> expected(kLanes);
    for (int lane = 0; lane < kLanes; ++lane) {
        std::uint32_t node = 0;
        std::uint32_t checksum = static_cast<std::uint32_t>(lane + 1);
        for (int step = 0; step < steps; ++step) {
            ++node;
            if (node == static_cast<std::uint32_t>(nodes)) {
                node = 0;
            }
            checksum ^= node;
        }
        expected[lane] = node ^ checksum;
    }
    return expected;
}

static void validate_sink(
        const std::uint32_t* device_sink,
        const std::vector<std::uint32_t>& expected,
        const char* phase, int footprint_kib, int trial) {
    std::vector<std::uint32_t> observed(kLanes);
    CHECK(mcMemcpy(
        observed.data(), device_sink,
        kLanes * sizeof(std::uint32_t), mcMemcpyDeviceToHost));
    for (int lane = 0; lane < kLanes; ++lane) {
        if (observed[lane] != expected[lane]) {
            std::fprintf(
                stderr,
                "sink mismatch phase=%s footprint_kib=%d trial=%d lane=%d "
                "observed=%u expected=%u\n",
                phase, footprint_kib, trial, lane,
                observed[lane], expected[lane]);
            std::exit(EXIT_FAILURE);
        }
    }
}

int main() {
    mcDeviceProp_t properties;
    CHECK(mcGetDeviceProperties(&properties, 0));
    constexpr int expected_aps = 104;
    constexpr int expected_warp_size = 64;
    constexpr int expected_l2_bytes = 8 * 1024 * 1024;
    if (std::strstr(properties.name, "C500") == nullptr
            || properties.multiProcessorCount != expected_aps
            || properties.warpSize != expected_warp_size
            || properties.l2CacheSize != expected_l2_bytes
            || properties.clockRate <= 0) {
        std::fprintf(stderr,
                     "unsupported device name=%s aps=%d warp=%d l2=%d "
                     "clock_khz=%d; expected C500/%d/%d/%d/positive\n",
                     properties.name, properties.multiProcessorCount,
                     properties.warpSize, properties.l2CacheSize,
                     properties.clockRate, expected_aps,
                     expected_warp_size, expected_l2_bytes);
        return EXIT_FAILURE;
    }

    std::uint32_t* device_sink = nullptr;
    std::uint32_t* device_start_node = nullptr;
    unsigned long long* device_ticks = nullptr;
    CHECK(mcMalloc(&device_sink, kLanes * sizeof(std::uint32_t)));
    CHECK(mcMalloc(&device_start_node, sizeof(std::uint32_t)));
    CHECK(mcMalloc(&device_ticks, sizeof(unsigned long long)));
    std::printf(
        "device=%s aps=%d warp_size=%d l2_bytes=%d clock_rate_khz=%d "
        "transaction_bytes=%d lanes=%d trials=%d\n",
        properties.name, properties.multiProcessorCount, properties.warpSize,
        properties.l2CacheSize, properties.clockRate,
        kTransactionBytes, kLanes, kTrials);
    std::printf(
        "footprint_kib,tier_hint,trial,nodes,warm_steps,timed_steps,"
        "memory_ticks,"
        "control_ticks,raw_ticks_per_step,control_ticks_per_step,"
        "memory_event_ms,control_event_ms,memory_event_ns_per_step,"
        "control_event_ns_per_step,memory_tick_event_ghz,"
        "control_tick_event_ghz\n");

    const int footprints_kib[] = {64, 256, 1024, 2048, 65536};
    for (int footprint_kib : footprints_kib) {
        std::size_t footprint_bytes = (
            static_cast<std::size_t>(footprint_kib) * 1024);
        if (footprint_bytes % kTransactionBytes != 0) {
            std::fprintf(stderr,
                         "footprint %zu is not transaction aligned\n",
                         footprint_bytes);
            return EXIT_FAILURE;
        }
        int nodes = static_cast<int>(
            footprint_bytes / kTransactionBytes);
        std::vector<std::uint64_t> host_buffer = make_chain(
            nodes, 42 + footprint_kib);
        std::uint64_t* device_buffer = nullptr;
        CHECK(mcMalloc(&device_buffer, footprint_bytes));
        CHECK(mcMemcpy(
            device_buffer, host_buffer.data(), footprint_bytes,
            mcMemcpyHostToDevice));
        bool below_l2 = (
            footprint_bytes < static_cast<std::size_t>(properties.l2CacheSize));
        int warm_steps = below_l2 ? nodes : kLargeFootprintWarmSteps;
        int hot_repeats = (
            (kMinimumHotTimedSteps + nodes - 1) / nodes);
        int timed_steps = (
            below_l2 ? nodes * hot_repeats : nodes - warm_steps);
        if (timed_steps <= 0) {
            std::fprintf(stderr,
                         "non-positive timed steps footprint_kib=%d nodes=%d "
                         "warm_steps=%d\n",
                         footprint_kib, nodes, warm_steps);
            return EXIT_FAILURE;
        }
        const char* tier_hint = (
            below_l2 ? "l2_candidate" : "dram_candidate");
        ChainState expected_warm = host_chain_state(
            host_buffer, warm_steps, 0);
        ChainState expected_memory = host_chain_state(
            host_buffer, timed_steps, expected_warm.node);
        std::vector<std::uint32_t> expected_control = control_checksum(
            timed_steps, nodes);

        warm_chain<<<1, kLanes>>>(
            device_buffer, warm_steps,
            device_sink, device_start_node);
        CHECK(mcDeviceSynchronize());
        std::uint32_t observed_start_node = 0;
        CHECK(mcMemcpy(
            &observed_start_node, device_start_node,
            sizeof(observed_start_node), mcMemcpyDeviceToHost));
        if (observed_start_node != expected_warm.node) {
            std::fprintf(stderr,
                         "warm start mismatch footprint_kib=%d observed=%u "
                         "expected=%u\n",
                         footprint_kib, observed_start_node,
                         expected_warm.node);
            return EXIT_FAILURE;
        }
        validate_sink(
            device_sink, expected_warm.sink,
            "warm", footprint_kib, -1);

        for (int trial = 0; trial < kTrials; ++trial) {
            warm_chain<<<1, kLanes>>>(
                device_buffer, warm_steps,
                device_sink, device_start_node);
            Timing memory = launch_timed_chain(
                device_buffer, device_start_node,
                timed_steps, device_sink, device_ticks);
            validate_sink(
                device_sink, expected_memory.sink,
                "memory", footprint_kib, trial);
            Timing control = launch_timed_control(
                timed_steps, nodes, device_sink, device_ticks);
            validate_sink(
                device_sink, expected_control,
                "control", footprint_kib, trial);
            double raw_ticks = (
                static_cast<double>(memory.ticks) / timed_steps);
            double control_ticks = (
                static_cast<double>(control.ticks) / timed_steps);
            double memory_event_ns = (
                memory.event_ms * 1.0e6 / timed_steps);
            double control_event_ns = (
                control.event_ms * 1.0e6 / timed_steps);
            double memory_tick_event_ghz = (
                static_cast<double>(memory.ticks)
                / (memory.event_ms * 1.0e6));
            double control_tick_event_ghz = (
                static_cast<double>(control.ticks)
                / (control.event_ms * 1.0e6));
            if (!std::isfinite(raw_ticks) || raw_ticks <= 0.0
                    || !std::isfinite(control_ticks)
                    || control_ticks <= 0.0
                    || !std::isfinite(memory_event_ns)
                    || memory_event_ns <= 0.0
                    || !std::isfinite(control_event_ns)
                    || control_event_ns <= 0.0
                    || !std::isfinite(memory_tick_event_ghz)
                    || memory_tick_event_ghz <= 0.0
                    || !std::isfinite(control_tick_event_ghz)
                    || control_tick_event_ghz <= 0.0) {
                std::fprintf(stderr,
                             "invalid normalized timing footprint_kib=%d "
                             "trial=%d\n", footprint_kib, trial);
                return EXIT_FAILURE;
            }
            std::printf(
                "%d,%s,%d,%d,%d,%d,%llu,%llu,%.9f,%.9f,%.9f,%.9f,"
                "%.9f,%.9f,%.9f,%.9f\n",
                footprint_kib, tier_hint, trial, nodes, warm_steps,
                timed_steps,
                memory.ticks, control.ticks,
                raw_ticks, control_ticks,
                memory.event_ms, control.event_ms,
                memory_event_ns, control_event_ns,
                memory_tick_event_ghz, control_tick_event_ghz);
        }
        CHECK(mcFree(device_buffer));
    }

    CHECK(mcFree(device_sink));
    CHECK(mcFree(device_start_node));
    CHECK(mcFree(device_ticks));
    return EXIT_SUCCESS;
}
