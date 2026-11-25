/*
 * MetaX C500 BM32 round-cadence and per-CTA-span mechanism probe.
 *
 * This is an independent mechanism microbenchmark, not a FusedMoE
 * implementation.  It keeps the deployed BM32/BN64/BK128 compiler resource
 * envelope (two 128-bit A loads and four 128-bit B loads per 256-thread CTA
 * per K round, two shared-memory stages, one CTA barrier per round, shared
 * reads, register-only BF16 MMA) and the compiler-observed prefetch phase
 * ordering, then varies only launch-independent mechanism axes:
 *
 *   - rounds: the K-loop trip count, sweeping fill/drain amortization;
 *   - grid: the number of serial CTAs per AP at one resident CTA/AP;
 *   - footprint mode: unique DRAM-streaming operands, small L2-resident
 *     operands, or a resident-A/streaming-B mixture.
 *
 * An additional early-exit kernel launches the same geometry, performs one
 * scalar load, and returns, isolating the serial cost of a no-work CTA at
 * this resource point.  No routing, expert selection, output tile,
 * production tensor shape, or production/Serving duration is present.
 *
 * Build in the prescribed C500 profiler container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch=native \
 *       --maca-path=/opt/maca -O3 -std=c++17 \
 *       bm32_round_cadence.cpp -o bm32_round_cadence
 *
 * Run (idle device, one GPU):
 *   MACA_VISIBLE_DEVICES=1 ./bm32_round_cadence
 */

#include <mc_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(call) do {                                                   \
    const mcError_t error = (call);                                        \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",              \
                     __FILE__, __LINE__, mcGetErrorString(error));        \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

namespace {

constexpr int kAps = 104;
constexpr int kWaveSize = 64;
constexpr int kThreads = 256;
constexpr int kAStreams = 2;
constexpr int kBStreams = 4;
constexpr int kStageCount = 2;
constexpr int kVectorBytes = 16;
constexpr int kDynamicSharedBytes = 49'152;
constexpr int kTrials = 7;
constexpr int kWarmups = 3;

// DRAM-streaming footprints: large enough that no launch revisits a line.
constexpr std::size_t kStreamABytes = 512ULL * 1024ULL * 1024ULL;
constexpr std::size_t kStreamBBytes = 1ULL * 1024ULL * 1024ULL * 1024ULL;
// L2-resident footprints: 2.5 MiB total, far below the reported 8-MiB L2.
constexpr long long kResidentAVectors = 32 * 1024;   // 512 KiB
constexpr long long kResidentBVectors = 128 * 1024;  // 2 MiB

typedef __NATIVE_VECTOR__(4, std::uint32_t) v4u;
typedef __NATIVE_VECTOR__(2, std::int32_t) v2i;
typedef __NATIVE_VECTOR__(4, float) v4f;

static_assert(sizeof(v4u) == kVectorBytes,
              "each operand request must be 128 bits");
static_assert(kThreads % kWaveSize == 0,
              "the probe requires complete wave64 groups");
static_assert(kStageCount * (kAStreams + kBStreams) * kThreads
                  * kVectorBytes == kDynamicSharedBytes,
              "dynamic shared allocation must be exactly two BM32 stages");

constexpr int kModeStream = 0;    // A and B stream unique DRAM lines
constexpr int kModeResident = 1;  // A and B stay in small L2-resident windows
constexpr int kModeMixed = 2;     // A resident window, B streams unique lines

struct OperandRegisters {
    v4u a[kAStreams];
    v4u b[kBStreams];
};

__device__ __forceinline__ v4u load_register(const v4u* source) {
    return __builtin_mxc_ldg_b128(
        const_cast<v4u*>(source), 0, -1, true, true, false, false);
}

__device__ __forceinline__ v4f mma_bf16(v2i a, v2i b, v4f c) {
    return __builtin_mxc_mma_16x16x16bf16(a, b, c);
}

__device__ __forceinline__ long long stream_index(
        long long round, int slot, int streams, long long global_thread,
        long long stride, long long resident_vectors, int mode) {
    if (mode == kModeResident
            || (mode == kModeMixed && streams == kAStreams)) {
        // One shared L2-resident window: every CTA reads the same panel.
        return ((round * streams + slot) * kThreads
                + (global_thread % kThreads)) % resident_vectors;
    }
    return (round * streams + slot) * stride + global_thread;
}

__device__ __forceinline__ OperandRegisters load_operands(
        const v4u* activation, const v4u* weight, long long round,
        long long global_thread, long long stride, int mode) {
    OperandRegisters result {};
    #pragma unroll
    for (int slot = 0; slot < kAStreams; ++slot) {
        result.a[slot] = load_register(
            activation + stream_index(round, slot, kAStreams, global_thread,
                                      stride, kResidentAVectors, mode));
    }
    #pragma unroll
    for (int slot = 0; slot < kBStreams; ++slot) {
        result.b[slot] = load_register(
            weight + stream_index(round, slot, kBStreams, global_thread,
                                  stride, kResidentBVectors, mode));
    }
    return result;
}

__device__ __forceinline__ void store_stage(
        v4u* shared, int stage, int thread, const OperandRegisters& values) {
    #pragma unroll
    for (int slot = 0; slot < kAStreams; ++slot) {
        shared[(stage * kAStreams + slot) * kThreads + thread] = values.a[slot];
    }
    constexpr int b_base = kStageCount * kAStreams * kThreads;
    #pragma unroll
    for (int slot = 0; slot < kBStreams; ++slot) {
        shared[b_base + (stage * kBStreams + slot) * kThreads + thread]
            = values.b[slot];
    }
}

__device__ __forceinline__ std::uint32_t consume_stage(
        const v4u* shared, int stage, int thread, v4f* accumulator0,
        v4f* accumulator1, v4f* accumulator2, v4f* accumulator3,
        v4f* accumulator4, v4f* accumulator5, v4f* accumulator6,
        v4f* accumulator7) {
    const int peer_a = (thread + kWaveSize) & (kThreads - 1);
    const int peer_b = (thread * 5 + 17) & (kThreads - 1);
    const v4u a0 = shared[(stage * kAStreams) * kThreads + peer_a];
    const v4u a1 = shared[(stage * kAStreams + 1) * kThreads + peer_b];
    constexpr int b_base = kStageCount * kAStreams * kThreads;
    const v4u b0 = shared[b_base + (stage * kBStreams) * kThreads + peer_b];
    const v4u b1 = shared[b_base + (stage * kBStreams + 1) * kThreads + peer_a];
    const v4u b2 = shared[b_base + (stage * kBStreams + 2) * kThreads + peer_b];
    const v4u b3 = shared[b_base + (stage * kBStreams + 3) * kThreads + peer_a];
    const v2i mma_a0 = {
        static_cast<std::int32_t>(a0[0]), static_cast<std::int32_t>(a0[1])};
    const v2i mma_a1 = {
        static_cast<std::int32_t>(a1[2]), static_cast<std::int32_t>(a1[3])};
    const v2i mma_b0 = {
        static_cast<std::int32_t>(b0[0]), static_cast<std::int32_t>(b1[1])};
    const v2i mma_b1 = {
        static_cast<std::int32_t>(b2[2]), static_cast<std::int32_t>(b3[3])};
    *accumulator0 = mma_bf16(mma_a0, mma_b0, *accumulator0);
    *accumulator1 = mma_bf16(mma_a1, mma_b1, *accumulator1);
    *accumulator2 = mma_bf16(mma_a0, mma_b1, *accumulator2);
    *accumulator3 = mma_bf16(mma_a1, mma_b0, *accumulator3);
    *accumulator0 = mma_bf16(mma_a0, mma_b1, *accumulator0);
    *accumulator1 = mma_bf16(mma_a1, mma_b0, *accumulator1);
    *accumulator2 = mma_bf16(mma_a0, mma_b0, *accumulator2);
    *accumulator3 = mma_bf16(mma_a1, mma_b1, *accumulator3);
    *accumulator4 = mma_bf16(mma_a0, mma_b0, *accumulator4);
    *accumulator5 = mma_bf16(mma_a1, mma_b1, *accumulator5);
    *accumulator6 = mma_bf16(mma_a0, mma_b1, *accumulator6);
    *accumulator7 = mma_bf16(mma_a1, mma_b0, *accumulator7);
    *accumulator4 = mma_bf16(mma_a0, mma_b1, *accumulator4);
    *accumulator5 = mma_bf16(mma_a1, mma_b0, *accumulator5);
    *accumulator6 = mma_bf16(mma_a0, mma_b1, *accumulator6);
    *accumulator7 = mma_bf16(mma_a1, mma_b0, *accumulator7);
    return a0[0] ^ a1[1] ^ b0[2] ^ b1[3] ^ b2[0] ^ b3[1];
}

extern "C" __global__ void c500_bm32_round_cadence(
        const v4u* activation, const v4u* weight, long long rounds,
        int mode, v4u* sink) {
    extern __shared__ unsigned char shared_bytes[];
    v4u* const shared = reinterpret_cast<v4u*>(shared_bytes);
    const int thread = static_cast<int>(threadIdx.x);
    const long long global_thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + thread);
    const long long stride = (
        static_cast<long long>(gridDim.x) * blockDim.x);
    v4f accumulator0 = {0.125f, 0.25f, 0.5f, 1.0f};
    v4f accumulator1 = {0.25f, 0.5f, 1.0f, 0.125f};
    v4f accumulator2 = {0.5f, 1.0f, 0.125f, 0.25f};
    v4f accumulator3 = {1.0f, 0.125f, 0.25f, 0.5f};
    v4f accumulator4 = {0.375f, 0.625f, 0.875f, 0.125f};
    v4f accumulator5 = {0.625f, 0.875f, 0.125f, 0.375f};
    v4f accumulator6 = {0.875f, 0.125f, 0.375f, 0.625f};
    v4f accumulator7 = {0.125f, 0.375f, 0.625f, 0.875f};
    std::uint32_t checksum = static_cast<std::uint32_t>(thread + 1);
    // Compiler-observed prefetch ordering: next A is issued before the CTA
    // barrier, next B after it, then the current stage is consumed.
    OperandRegisters current = load_operands(
        activation, weight, 0, global_thread, stride, mode);
    store_stage(shared, 0, thread, current);
    for (long long round = 0; round < rounds; ++round) {
        const long long next_round = round + 1;
        OperandRegisters next {};
        const bool next_valid = next_round < rounds;
        if (next_valid) {
            #pragma unroll
            for (int slot = 0; slot < kAStreams; ++slot) {
                next.a[slot] = load_register(
                    activation + stream_index(
                        next_round, slot, kAStreams, global_thread, stride,
                        kResidentAVectors, mode));
            }
        }
        __syncthreads();
        if (next_valid) {
            #pragma unroll
            for (int slot = 0; slot < kBStreams; ++slot) {
                next.b[slot] = load_register(
                    weight + stream_index(
                        next_round, slot, kBStreams, global_thread, stride,
                        kResidentBVectors, mode));
            }
        }
        const int current_stage = static_cast<int>(round & 1);
        checksum ^= consume_stage(
            shared, current_stage, thread, &accumulator0, &accumulator1,
            &accumulator2, &accumulator3, &accumulator4, &accumulator5,
            &accumulator6, &accumulator7);
        const int next_stage = static_cast<int>(next_round & 1);
        store_stage(shared, next_stage, thread, next);
    }
    __syncthreads();
    sink[static_cast<long long>(blockIdx.x) * blockDim.x + thread] = {
        checksum + static_cast<std::uint32_t>(accumulator0.x + accumulator4.x),
        checksum + static_cast<std::uint32_t>(accumulator1.y + accumulator5.y),
        checksum + static_cast<std::uint32_t>(accumulator2.z + accumulator6.z),
        checksum + static_cast<std::uint32_t>(accumulator3.w + accumulator7.w)};
}

extern "C" __global__ void c500_bm32_early_exit(
        const std::int32_t* control, v4u* sink) {
    // One dependent scalar load, then an immediate exit: the no-work CTA
    // serial cost at this launch geometry.
    const int thread = static_cast<int>(threadIdx.x);
    const int value = control[0];
    if (value == -1) {
        sink[static_cast<long long>(blockIdx.x) * blockDim.x + thread] = {
            1u, 2u, 3u, 4u};
    }
}

static double measure_round_cadence(
        const v4u* activation, const v4u* weight, long long rounds, int grid,
        int mode, v4u* sink) {
    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    CHECK(mcEventRecord(start));
    c500_bm32_round_cadence<<<grid, kThreads, kDynamicSharedBytes>>>(
        activation, weight, rounds, mode, sink);
    CHECK(mcGetLastError());
    CHECK(mcEventRecord(end));
    CHECK(mcDeviceSynchronize());
    float elapsed_ms = 0.0F;
    CHECK(mcEventElapsedTime(&elapsed_ms, start, end));
    CHECK(mcEventDestroy(start));
    CHECK(mcEventDestroy(end));
    if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0F) {
        std::fprintf(stderr, "device event duration is invalid\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<double>(elapsed_ms);
}

static double measure_early_exit(
        const std::int32_t* control, int grid, v4u* sink) {
    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    CHECK(mcEventRecord(start));
    c500_bm32_early_exit<<<grid, kThreads, kDynamicSharedBytes>>>(
        control, sink);
    CHECK(mcGetLastError());
    CHECK(mcEventRecord(end));
    CHECK(mcDeviceSynchronize());
    float elapsed_ms = 0.0F;
    CHECK(mcEventElapsedTime(&elapsed_ms, start, end));
    CHECK(mcEventDestroy(start));
    CHECK(mcEventDestroy(end));
    if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0F) {
        std::fprintf(stderr, "early-exit device event duration is invalid\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<double>(elapsed_ms);
}

static const char* mode_name(int mode) {
    if (mode == kModeStream) return "stream";
    if (mode == kModeResident) return "resident";
    if (mode == kModeMixed) return "mixed";
    return "unknown";
}

static void run_round_cell(int mode, int grid, long long rounds,
                           const v4u* activation, const v4u* weight,
                           long long stream_a_vectors,
                           long long stream_b_vectors, v4u* sink) {
    if (mode != kModeResident) {
        const long long stride = static_cast<long long>(grid) * kThreads;
        const long long b_last = (rounds * kBStreams) * stride;
        if (b_last > stream_b_vectors) {
            std::fprintf(stderr,
                         "stream B footprint exceeded: grid=%d rounds=%lld\n",
                         grid, rounds);
            std::exit(EXIT_FAILURE);
        }
        const long long a_last = (rounds * kAStreams) * stride;
        if (mode == kModeStream && a_last > stream_a_vectors) {
            std::fprintf(stderr,
                         "stream A footprint exceeded: grid=%d rounds=%lld\n",
                         grid, rounds);
            std::exit(EXIT_FAILURE);
        }
    }
    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        c500_bm32_round_cadence<<<grid, kThreads, kDynamicSharedBytes>>>(
            activation, weight, rounds, mode, sink);
        CHECK(mcGetLastError());
    }
    CHECK(mcDeviceSynchronize());
    const double bytes_per_launch = static_cast<double>(rounds) * grid
        * kThreads * (kAStreams + kBStreams) * kVectorBytes;
    for (int trial = 0; trial < kTrials; ++trial) {
        const double elapsed_ms = measure_round_cadence(
            activation, weight, rounds, grid, mode, sink);
        const double bandwidth_gbs = bytes_per_launch / elapsed_ms / 1.0e6;
        std::printf("sample,%s,%d,%lld,%d,%d,%.9f,%.6f,%.0f\n",
                    mode_name(mode), grid, rounds, kThreads,
                    kDynamicSharedBytes, elapsed_ms, bandwidth_gbs,
                    bytes_per_launch);
    }
}

static void run_early_exit_cell(int grid, const std::int32_t* control,
                                v4u* sink) {
    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        c500_bm32_early_exit<<<grid, kThreads, kDynamicSharedBytes>>>(
            control, sink);
        CHECK(mcGetLastError());
    }
    CHECK(mcDeviceSynchronize());
    for (int trial = 0; trial < kTrials; ++trial) {
        const double elapsed_ms = measure_early_exit(control, grid, sink);
        std::printf("sample,early_exit,%d,0,%d,%d,%.9f,0.000000,0\n",
                    grid, kThreads, kDynamicSharedBytes, elapsed_ms);
    }
}

template <typename Kernel>
static void validate_resources(const char* mode, Kernel kernel) {
    mcFuncAttributes attributes {};
    const void* const function = reinterpret_cast<const void*>(kernel);
    CHECK(mcFuncGetAttributes(&attributes, function));
    int resident_ctas = 0;
    CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_ctas, function, kThreads, kDynamicSharedBytes));
    if (attributes.sharedSizeBytes != 0 || resident_ctas != 1) {
        std::fprintf(stderr,
                     "resource contract failed mode=%s registers=%d "
                     "static_shared=%zu dynamic_shared=%d "
                     "resident_ctas_per_ap=%d\n",
                     mode, attributes.numRegs,
                     static_cast<std::size_t>(attributes.sharedSizeBytes),
                     kDynamicSharedBytes, resident_ctas);
        std::exit(EXIT_FAILURE);
    }
    std::fprintf(stderr,
                 "resource,mode=%s,registers=%d,static_shared=%zu,"
                 "dynamic_shared=%d,resident_ctas_per_ap=%d\n",
                 mode, attributes.numRegs,
                 static_cast<std::size_t>(attributes.sharedSizeBytes),
                 kDynamicSharedBytes, resident_ctas);
}

}  // namespace

int main() {
    mcDeviceProp_t properties {};
    CHECK(mcGetDeviceProperties(&properties, 0));
    if (std::strstr(properties.name, "C500") == nullptr
            || properties.multiProcessorCount != kAps
            || properties.warpSize != kWaveSize) {
        std::fprintf(stderr, "unexpected device %s aps=%d warp=%d\n",
                     properties.name, properties.multiProcessorCount,
                     properties.warpSize);
        return EXIT_FAILURE;
    }
    validate_resources("round_cadence", c500_bm32_round_cadence);
    validate_resources("early_exit", c500_bm32_early_exit);

    v4u* activation = nullptr;
    v4u* weight = nullptr;
    v4u* sink = nullptr;
    std::int32_t* control = nullptr;
    const int maximum_grid = kAps * 31;
    CHECK(mcMalloc(&activation, kStreamABytes));
    CHECK(mcMalloc(&weight, kStreamBBytes));
    CHECK(mcMalloc(&sink, static_cast<std::size_t>(maximum_grid) * kThreads
                       * sizeof(v4u)));
    CHECK(mcMalloc(&control, sizeof(std::int32_t)));
    CHECK(mcMemset(activation, 1, kStreamABytes));
    CHECK(mcMemset(weight, 1, kStreamBBytes));
    CHECK(mcMemset(sink, 0, static_cast<std::size_t>(maximum_grid) * kThreads
                       * sizeof(v4u)));
    CHECK(mcMemset(control, 0, sizeof(std::int32_t)));

    std::printf("sample_header,mode,grid,rounds,block,dynamic_shared_bytes,"
                "event_ms,logical_operand_bandwidth_gbs,logical_bytes\n");

    // Axis 1: K-round sweep at one resident wave per AP.  The intercept is
    // the launch plus per-CTA fill/drain span; the slope is the round
    // cadence for each operand-residency mode.
    const long long round_sweep[] = {2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96};
    for (int mode : {kModeStream, kModeResident, kModeMixed}) {
        for (long long rounds : round_sweep) {
            run_round_cell(mode, kAps, rounds, activation, weight,
                           kStreamABytes / sizeof(v4u),
                           kStreamBBytes / sizeof(v4u), sink);
        }
    }
    // Axis 2: serial-CTA stacking at the two production K depths (6 and 16
    // rounds), sweeping the number of queued CTAs per AP.
    const int grid_sweep[] = {kAps, kAps * 2, kAps * 4, kAps * 10, kAps * 31};
    for (int mode : {kModeStream, kModeResident, kModeMixed}) {
        for (long long rounds : {6LL, 16LL}) {
            for (int grid : grid_sweep) {
                run_round_cell(mode, grid, rounds, activation, weight,
                               kStreamABytes / sizeof(v4u),
                               kStreamBBytes / sizeof(v4u), sink);
            }
        }
    }
    // Axis 3: no-work CTA serial cost at the same geometry, including the
    // two production grid sizes 3192 and 4256.
    for (int grid : {kAps, kAps * 10, kAps * 31, 3192, 4256}) {
        run_early_exit_cell(grid, control, sink);
    }

    v4u sample {};
    CHECK(mcMemcpy(&sample, sink, sizeof(sample), mcMemcpyDeviceToHost));
    if ((sample[0] | sample[1] | sample[2] | sample[3]) == 0) {
        std::fprintf(stderr, "round-cadence sink validation failed\n");
        return EXIT_FAILURE;
    }
    CHECK(mcFree(activation));
    CHECK(mcFree(weight));
    CHECK(mcFree(control));
    CHECK(mcFree(sink));
    std::fprintf(stderr, "validation=pass\n");
    return EXIT_SUCCESS;
}
