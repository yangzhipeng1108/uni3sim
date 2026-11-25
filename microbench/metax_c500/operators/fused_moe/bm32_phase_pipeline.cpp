/*
 * MetaX C500 BM32 operand-pipeline phase probe.
 *
 * This is an independent mechanism microbenchmark, not a FusedMoE
 * implementation.  It contains neither routing, expert selection, an output
 * tile, production tensor dimensions, nor a projection epilogue.  It only
 * compares static operand schedules using the deployed BM32 compiler
 * resource envelope: six 128-bit loads per 256-thread CTA, two shared-memory
 * stages, a CTA shared barrier, shared reads, and register-only BF16 MMA.
 *
 * ``prefetched`` follows the phase ordering seen in the pinned AOT TTGIR:
 * prefetch the next A vectors, CTA barrier, prefetch the next B vectors,
 * consume the current shared stage, then publish the next stage.  ``bulk``
 * keeps the same bytes, resources, shared reads, and MMA count but loads both
 * operands before its barrier.  ``gather_prefetched`` adds only a synthetic
 * sorted-route index indirection for A; it still has no output tile,
 * production tensor, expert selection, or epilogue.  The comparisons measure
 * operand-addressing and synchronization mechanisms only; no production or
 * evaluation duration is read.
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
constexpr int kGatherTileM = 32;
constexpr int kGatherNFanout = 32;
constexpr int kGatherSourceRows = 2'048;
constexpr int kAVectorsPerStage = 16;
constexpr std::size_t kSourceBytes = 256ULL * 1024ULL * 1024ULL;
constexpr int kIterations = 7;
constexpr int kTrials = 7;
constexpr int kWarmups = 3;

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
static_assert(kAStreams * kThreads
                  == kGatherTileM * kAVectorsPerStage,
              "the gather map must cover one BM32 A stage exactly");

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

__device__ __forceinline__ OperandRegisters load_operands(
        const v4u* activation, const v4u* weight, long long round,
        long long global_thread, long long stride, bool valid) {
    OperandRegisters result {};
    if (!valid) return result;
    #pragma unroll
    for (int slot = 0; slot < kAStreams; ++slot) {
        result.a[slot] = load_register(
            activation + (round * kAStreams + slot) * stride + global_thread);
    }
    #pragma unroll
    for (int slot = 0; slot < kBStreams; ++slot) {
        result.b[slot] = load_register(
            weight + (round * kBStreams + slot) * stride + global_thread);
    }
    return result;
}

__device__ __forceinline__ void load_gather_a(
        const v4u* activation, const v4u* weight,
        const std::int32_t* route_indices, long long round,
        long long rounds, int thread, int cta, v4u* values) {
    (void)weight;
    const int group = cta / kGatherNFanout;
    #pragma unroll
    for (int slot = 0; slot < kAStreams; ++slot) {
        const int linear_vector = slot * kThreads + thread;
        const int row = linear_vector / kAVectorsPerStage;
        const int vector_in_stage = linear_vector % kAVectorsPerStage;
        const std::int32_t source_row = route_indices[
            group * kGatherTileM + row];
        values[slot] = load_register(
            activation + (static_cast<long long>(source_row) * rounds
                          + round) * kAVectorsPerStage + vector_in_stage);
    }
}

__device__ __forceinline__ void load_gather_b(
        const v4u* weight, long long round, int thread, int cta, int grid,
        v4u* values) {
    const long long global_thread = (
        static_cast<long long>(cta) * kThreads + thread);
    const long long stride = static_cast<long long>(grid) * kThreads;
    #pragma unroll
    for (int slot = 0; slot < kBStreams; ++slot) {
        values[slot] = load_register(
            weight + (round * kBStreams + slot) * stride + global_thread);
    }
}

__device__ __forceinline__ OperandRegisters load_gather_operands(
        const v4u* activation, const v4u* weight,
        const std::int32_t* route_indices, long long round,
        long long rounds, int thread, int cta, int grid) {
    OperandRegisters result {};
    load_gather_a(activation, weight, route_indices, round, rounds, thread,
                  cta, result.a);
    load_gather_b(weight, round, thread, cta, grid, result.b);
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
    *accumulator6 = mma_bf16(mma_a0, mma_b0, *accumulator6);
    *accumulator7 = mma_bf16(mma_a1, mma_b1, *accumulator7);
    return a0[0] ^ a1[1] ^ b0[2] ^ b1[3] ^ b2[0] ^ b3[1];
}

extern "C" __global__ void c500_bm32_phase_bulk(
        const v4u* activation, const v4u* weight, long long rounds,
        int iterations, v4u* sink) {
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
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (long long round = 0; round < rounds; ++round) {
            const OperandRegisters values = load_operands(
                activation, weight, round, global_thread, stride, true);
            const int stage = static_cast<int>(round & 1);
            store_stage(shared, stage, thread, values);
            __syncthreads();
            checksum ^= consume_stage(
                shared, stage, thread, &accumulator0, &accumulator1,
                &accumulator2, &accumulator3, &accumulator4, &accumulator5,
                &accumulator6, &accumulator7);
        }
    }
    sink[static_cast<long long>(blockIdx.x) * blockDim.x + thread] = {
        checksum + static_cast<std::uint32_t>(accumulator0.x + accumulator4.x),
        checksum + static_cast<std::uint32_t>(accumulator1.y + accumulator5.y),
        checksum + static_cast<std::uint32_t>(accumulator2.z + accumulator6.z),
        checksum + static_cast<std::uint32_t>(accumulator3.w + accumulator7.w)};
}

extern "C" __global__ void c500_bm32_phase_prefetched(
        const v4u* activation, const v4u* weight, long long rounds,
        int iterations, v4u* sink) {
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
    for (int iteration = 0; iteration < iterations; ++iteration) {
        OperandRegisters current = load_operands(
            activation, weight, 0, global_thread, stride, true);
        store_stage(shared, 0, thread, current);
        for (long long round = 0; round < rounds; ++round) {
            const long long next_round = round + 1;
            // Preserve the compiler's dependency placement: next A is issued
            // before the CTA barrier, while next B is issued after it.
            OperandRegisters next {};
            const bool next_valid = next_round < rounds;
            if (next_valid) {
                #pragma unroll
                for (int slot = 0; slot < kAStreams; ++slot) {
                    next.a[slot] = load_register(
                        activation + (next_round * kAStreams + slot) * stride
                        + global_thread);
                }
            }
            __syncthreads();
            if (next_valid) {
                #pragma unroll
                for (int slot = 0; slot < kBStreams; ++slot) {
                    next.b[slot] = load_register(
                        weight + (next_round * kBStreams + slot) * stride
                        + global_thread);
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
    }
    __syncthreads();
    sink[static_cast<long long>(blockIdx.x) * blockDim.x + thread] = {
        checksum + static_cast<std::uint32_t>(accumulator0.x + accumulator4.x),
        checksum + static_cast<std::uint32_t>(accumulator1.y + accumulator5.y),
        checksum + static_cast<std::uint32_t>(accumulator2.z + accumulator6.z),
        checksum + static_cast<std::uint32_t>(accumulator3.w + accumulator7.w)};
}

extern "C" __global__ void c500_bm32_phase_gather_prefetched(
        const v4u* activation, const v4u* weight,
        const std::int32_t* route_indices, long long rounds,
        int iterations, v4u* sink) {
    extern __shared__ unsigned char shared_bytes[];
    v4u* const shared = reinterpret_cast<v4u*>(shared_bytes);
    const int thread = static_cast<int>(threadIdx.x);
    const int cta = static_cast<int>(blockIdx.x);
    const int grid = static_cast<int>(gridDim.x);
    v4f accumulator0 = {0.125f, 0.25f, 0.5f, 1.0f};
    v4f accumulator1 = {0.25f, 0.5f, 1.0f, 0.125f};
    v4f accumulator2 = {0.5f, 1.0f, 0.125f, 0.25f};
    v4f accumulator3 = {1.0f, 0.125f, 0.25f, 0.5f};
    v4f accumulator4 = {0.375f, 0.625f, 0.875f, 0.125f};
    v4f accumulator5 = {0.625f, 0.875f, 0.125f, 0.375f};
    v4f accumulator6 = {0.875f, 0.125f, 0.375f, 0.625f};
    v4f accumulator7 = {0.125f, 0.375f, 0.625f, 0.875f};
    std::uint32_t checksum = static_cast<std::uint32_t>(thread + 1);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        OperandRegisters current = load_gather_operands(
            activation, weight, route_indices, 0, rounds, thread, cta, grid);
        store_stage(shared, 0, thread, current);
        for (long long round = 0; round < rounds; ++round) {
            const long long next_round = round + 1;
            OperandRegisters next {};
            if (next_round < rounds) {
                load_gather_a(activation, weight, route_indices, next_round,
                              rounds, thread, cta, next.a);
            }
            __syncthreads();
            if (next_round < rounds) {
                load_gather_b(weight, next_round, thread, cta, grid, next.b);
            }
            const int current_stage = static_cast<int>(round & 1);
            checksum ^= consume_stage(
                shared, current_stage, thread, &accumulator0, &accumulator1,
                &accumulator2, &accumulator3, &accumulator4, &accumulator5,
                &accumulator6, &accumulator7);
            const int next_stage = static_cast<int>(next_round & 1);
            store_stage(shared, next_stage, thread, next);
        }
    }
    __syncthreads();
    sink[static_cast<long long>(cta) * blockDim.x + thread] = {
        checksum + static_cast<std::uint32_t>(accumulator0.x + accumulator4.x),
        checksum + static_cast<std::uint32_t>(accumulator1.y + accumulator5.y),
        checksum + static_cast<std::uint32_t>(accumulator2.z + accumulator6.z),
        checksum + static_cast<std::uint32_t>(accumulator3.w + accumulator7.w)};
}

template <typename Kernel>
static double measure(Kernel kernel, const v4u* activation,
                      const v4u* weight, long long rounds, int grid,
                      v4u* sink) {
    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    CHECK(mcEventRecord(start));
    kernel<<<grid, kThreads, kDynamicSharedBytes>>>(
        activation, weight, rounds, kIterations, sink);
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

template <typename Kernel>
static void run_mode(const char* mode, Kernel kernel, const v4u* activation,
                     const v4u* weight, int grid, v4u* sink,
                     long long source_vectors, long long rounds_override) {
    const long long vectors_per_round = (
        static_cast<long long>(grid) * kThreads * kBStreams);
    const long long maximum_rounds = source_vectors / vectors_per_round;
    const long long rounds = (
        rounds_override == 0 ? maximum_rounds : rounds_override);
    if (rounds < 2 || rounds > maximum_rounds) {
        std::fprintf(stderr,
                     "rounds must be in [2, %lld], got %lld\n",
                     maximum_rounds, rounds);
        std::exit(EXIT_FAILURE);
    }
    const double bytes_per_iteration = static_cast<double>(rounds) * grid
        * kThreads * (kAStreams + kBStreams) * kVectorBytes;
    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        kernel<<<grid, kThreads, kDynamicSharedBytes>>>(
            activation, weight, rounds, 1, sink);
        CHECK(mcGetLastError());
    }
    CHECK(mcDeviceSynchronize());
    for (int trial = 0; trial < kTrials; ++trial) {
        const double elapsed_ms = measure(
            kernel, activation, weight, rounds, grid, sink);
        const double bandwidth_gbs = (
            bytes_per_iteration * kIterations / elapsed_ms / 1.0e6);
        std::printf("sample,%s,%d,%lld,%d,%d,%.9f,%.6f\n", mode, grid,
                    rounds, kThreads, kDynamicSharedBytes, elapsed_ms,
                    bandwidth_gbs);
    }
}

static long long parse_rounds_override(int argc, char** argv) {
    if (argc == 1) return 0;
    if (argc != 3 || std::strcmp(argv[1], "--rounds") != 0) {
        std::fprintf(stderr, "usage: %s [--rounds N]\n", argv[0]);
        std::exit(EXIT_FAILURE);
    }
    char* end = nullptr;
    const long long rounds = std::strtoll(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0' || rounds < 2) {
        std::fprintf(stderr, "--rounds must be an integer of at least two\n");
        std::exit(EXIT_FAILURE);
    }
    return rounds;
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

static double measure_gather(
        const v4u* activation, const v4u* weight,
        const std::int32_t* route_indices, long long rounds, int grid,
        v4u* sink) {
    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    CHECK(mcEventRecord(start));
    c500_bm32_phase_gather_prefetched<<<grid, kThreads, kDynamicSharedBytes>>>(
        activation, weight, route_indices, rounds, kIterations, sink);
    CHECK(mcGetLastError());
    CHECK(mcEventRecord(end));
    CHECK(mcDeviceSynchronize());
    float elapsed_ms = 0.0F;
    CHECK(mcEventElapsedTime(&elapsed_ms, start, end));
    CHECK(mcEventDestroy(start));
    CHECK(mcEventDestroy(end));
    if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0F) {
        std::fprintf(stderr, "gather device event duration is invalid\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<double>(elapsed_ms);
}

static void run_gather_mode(
        const v4u* activation, const v4u* weight,
        const std::int32_t* route_indices, int grid, v4u* sink,
        long long source_vectors, long long rounds_override) {
    if (grid % kGatherNFanout) {
        std::fprintf(stderr, "gather grid must divide the N fanout\n");
        std::exit(EXIT_FAILURE);
    }
    const long long vectors_per_round = (
        static_cast<long long>(grid) * kThreads * kBStreams);
    const long long maximum_rounds = source_vectors / vectors_per_round;
    const long long rounds = (
        rounds_override == 0 ? maximum_rounds : rounds_override);
    if (rounds < 2 || rounds > maximum_rounds) {
        std::fprintf(stderr,
                     "gather rounds must be in [2, %lld], got %lld\n",
                     maximum_rounds, rounds);
        std::exit(EXIT_FAILURE);
    }
    const double bytes_per_iteration = static_cast<double>(rounds) * grid
        * kThreads * (kAStreams + kBStreams) * kVectorBytes;
    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        c500_bm32_phase_gather_prefetched<<<
            grid, kThreads, kDynamicSharedBytes>>>(
                activation, weight, route_indices, rounds, 1, sink);
        CHECK(mcGetLastError());
    }
    CHECK(mcDeviceSynchronize());
    for (int trial = 0; trial < kTrials; ++trial) {
        const double elapsed_ms = measure_gather(
            activation, weight, route_indices, rounds, grid, sink);
        const double bandwidth_gbs = (
            bytes_per_iteration * kIterations / elapsed_ms / 1.0e6);
        std::printf("sample,gather_prefetched,%d,%lld,%d,%d,%.9f,%.6f\n",
                    grid, rounds, kThreads, kDynamicSharedBytes, elapsed_ms,
                    bandwidth_gbs);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const long long rounds_override = parse_rounds_override(argc, argv);
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
    validate_resources("bulk", c500_bm32_phase_bulk);
    validate_resources("prefetched", c500_bm32_phase_prefetched);
    validate_resources("gather_prefetched", c500_bm32_phase_gather_prefetched);

    v4u* activation = nullptr;
    v4u* weight = nullptr;
    v4u* sink = nullptr;
    std::int32_t* indices = nullptr;
    const long long source_vectors = kSourceBytes / sizeof(v4u);
    const int maximum_grid = kAps * 16;
    CHECK(mcMalloc(&activation, kSourceBytes));
    CHECK(mcMalloc(&weight, kSourceBytes));
    CHECK(mcMalloc(&sink, static_cast<std::size_t>(maximum_grid) * kThreads
                       * sizeof(v4u)));
    CHECK(mcMalloc(&indices, static_cast<std::size_t>(maximum_grid)
                       / kGatherNFanout * kGatherTileM
                       * sizeof(std::int32_t)));
    CHECK(mcMemset(activation, 1, kSourceBytes));
    CHECK(mcMemset(weight, 1, kSourceBytes));
    CHECK(mcMemset(sink, 0, static_cast<std::size_t>(maximum_grid) * kThreads
                       * sizeof(v4u)));
    std::int32_t host_indices[maximum_grid / kGatherNFanout * kGatherTileM];
    for (int row = 0;
         row < maximum_grid / kGatherNFanout * kGatherTileM; ++row) {
        host_indices[row] = static_cast<std::int32_t>(
            (static_cast<std::uint32_t>(row) * 2654435761U + 17U)
            & (kGatherSourceRows - 1));
    }
    CHECK(mcMemcpy(indices, host_indices, sizeof(host_indices),
                   mcMemcpyHostToDevice));

    std::printf("sample_header,mode,grid,rounds,block,dynamic_shared_bytes,"
                "event_ms,logical_operand_bandwidth_gbs\n");
    for (int grid : {kAps, maximum_grid}) {
        run_mode("bulk", c500_bm32_phase_bulk, activation, weight, grid,
                 sink, source_vectors, rounds_override);
        run_mode("prefetched", c500_bm32_phase_prefetched, activation,
                 weight, grid, sink, source_vectors, rounds_override);
    }
    run_gather_mode(activation, weight, indices, maximum_grid, sink,
                    source_vectors, rounds_override);
    v4u sample {};
    CHECK(mcMemcpy(&sample, sink, sizeof(sample), mcMemcpyDeviceToHost));
    if ((sample[0] | sample[1] | sample[2] | sample[3]) == 0) {
        std::fprintf(stderr, "phase-pipeline sink validation failed\n");
        return EXIT_FAILURE;
    }
    CHECK(mcFree(activation));
    CHECK(mcFree(weight));
    CHECK(mcFree(indices));
    CHECK(mcFree(sink));
    std::fprintf(stderr, "validation=pass\n");
    return EXIT_SUCCESS;
}
