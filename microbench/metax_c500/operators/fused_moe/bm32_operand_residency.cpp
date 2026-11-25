/*
 * Independent C500 BM32 operand-admission microbenchmark.
 *
 * This is deliberately not a FusedMoE implementation.  It emits the static
 * load-stage footprint of the deployed BF16 BM32/BN64/BK128 Triton program:
 * two 128-bit A vectors and four 128-bit B vectors per 256-thread CTA per K
 * stage, directly into two shared-memory stage buffers.  Eight register-only
 * BF16 MMA instructions per stage retain matrix issue pressure but no routing,
 * expert selection, activation, output tile, or production tensor shape is
 * present.  The 49,152-B dynamic allocation is the compiled program's static
 * resource fact; runtime occupancy must prove one resident CTA/AP.
 *
 * Normal device-event duration is used only to establish this independent
 * operand-service mechanism.  No production or Serving duration is read.
 */

#include <mc_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(call) do {                                                   \
    mcError_t error = call;                                                \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n", __FILE__,   \
                     __LINE__, mcGetErrorString(error));                   \
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
constexpr std::size_t kSourceBytes = 256ULL * 1024ULL * 1024ULL;
constexpr int kIterations = 7;
constexpr int kTrials = 5;
constexpr int kWarmups = 3;

typedef __NATIVE_VECTOR__(4, std::uint32_t) v4u;
typedef __NATIVE_VECTOR__(2, std::int32_t) v2i;
typedef __NATIVE_VECTOR__(4, float) v4f;

static_assert(sizeof(v4u) == kVectorBytes, "operand load must be 128 bits");
static_assert(kThreads % kWaveSize == 0, "CTA must use full wave64 groups");
static_assert(kAStreams * kThreads * kVectorBytes * kStageCount
                  + kBStreams * kThreads * kVectorBytes * kStageCount
                  == kDynamicSharedBytes,
              "shared allocation must exactly encode two BM32 operand stages");

__device__ __forceinline__ void direct_load(void* destination,
                                             const v4u* source) {
    __builtin_mxc_ldg_b128_bsm(
        destination, const_cast<v4u*>(source), 0, -1, true, true, false,
        true);
}

__device__ __forceinline__ v4f mma_bf16(v2i a, v2i b, v4f c) {
    return __builtin_mxc_mma_16x16x16bf16(a, b, c);
}

extern "C" __global__ void c500_bm32_operand_residency(
        const v4u* activation, const v4u* weight, long long rounds,
        int iterations, v4u* sink) {
    extern __shared__ unsigned char shared_bytes[];
    const int thread = static_cast<int>(threadIdx.x);
    const long long global_thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + thread);
    const long long stride = (
        static_cast<long long>(gridDim.x) * blockDim.x);
    const int b_stage_base_vectors = kStageCount * kAStreams * kThreads;
    v2i mma_a = {0x3b803c00, 0x3a803b00};
    v2i mma_b = {0x3b003a80, 0x3c003b80};
    v4f accumulator0 = {0.125f, 0.25f, 0.5f, 1.0f};
    v4f accumulator1 = {0.25f, 0.5f, 1.0f, 0.125f};
    v4f accumulator2 = {0.5f, 1.0f, 0.125f, 0.25f};
    v4f accumulator3 = {1.0f, 0.125f, 0.25f, 0.5f};

    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (long long round = 0; round < rounds; ++round) {
            const int stage = static_cast<int>(round % kStageCount);
            #pragma unroll
            for (int slot = 0; slot < kAStreams; ++slot) {
                const long long source_index = (
                    (round * kAStreams + slot) * stride + global_thread);
                const int destination_index = (
                    (stage * kAStreams + slot) * kThreads + thread);
                direct_load(
                    shared_bytes + destination_index * kVectorBytes,
                    activation + source_index);
            }
            #pragma unroll
            for (int slot = 0; slot < kBStreams; ++slot) {
                const long long source_index = (
                    (round * kBStreams + slot) * stride + global_thread);
                const int destination_index = b_stage_base_vectors + (
                    (stage * kBStreams + slot) * kThreads + thread);
                direct_load(
                    shared_bytes + destination_index * kVectorBytes,
                    weight + source_index);
            }
            __builtin_mxc_arrive(kWaveSize);
            __builtin_mxc_barrier_inst();
            // 32x64 output tile = eight 16x16 MMA fragments per BK128 stage.
            accumulator0 = mma_bf16(mma_a, mma_b, accumulator0);
            accumulator1 = mma_bf16(mma_a, mma_b, accumulator1);
            accumulator2 = mma_bf16(mma_a, mma_b, accumulator2);
            accumulator3 = mma_bf16(mma_a, mma_b, accumulator3);
            accumulator0 = mma_bf16(mma_a, mma_b, accumulator0);
            accumulator1 = mma_bf16(mma_a, mma_b, accumulator1);
            accumulator2 = mma_bf16(mma_a, mma_b, accumulator2);
            accumulator3 = mma_bf16(mma_a, mma_b, accumulator3);
        }
    }
    const int final_stage = static_cast<int>((rounds - 1) % kStageCount);
    const int final_index = b_stage_base_vectors + (
        (final_stage * kBStreams + (kBStreams - 1)) * kThreads + thread);
    const v4u observed = reinterpret_cast<v4u*>(shared_bytes)[final_index];
    // The addition makes the MMA chain live without creating a production
    // output tensor or reproducing a projection epilogue.
    sink[static_cast<long long>(blockIdx.x) * blockDim.x + thread] = {
        observed[0] + static_cast<std::uint32_t>(accumulator0.x),
        observed[1] + static_cast<std::uint32_t>(accumulator1.y),
        observed[2] + static_cast<std::uint32_t>(accumulator2.z),
        observed[3] + static_cast<std::uint32_t>(accumulator3.w)};
}

static double measure(const v4u* activation, const v4u* weight,
                      long long rounds, int grid, v4u* sink) {
    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    CHECK(mcEventRecord(start));
    c500_bm32_operand_residency<<<grid, kThreads, kDynamicSharedBytes>>>(
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

static void run_grid(const v4u* activation, const v4u* weight, int grid,
                     v4u* sink, long long source_vectors) {
    const long long vectors_per_round = (
        static_cast<long long>(grid) * kThreads * kBStreams);
    const long long rounds = source_vectors / vectors_per_round;
    if (rounds <= 0) {
        std::fprintf(stderr, "source allocation cannot cover one B stage\n");
        std::exit(EXIT_FAILURE);
    }
    const double bytes_per_iteration = static_cast<double>(rounds) * grid
        * kThreads * (kAStreams + kBStreams) * kVectorBytes;
    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        c500_bm32_operand_residency<<<grid, kThreads, kDynamicSharedBytes>>>(
            activation, weight, rounds, 1, sink);
        CHECK(mcGetLastError());
    }
    CHECK(mcDeviceSynchronize());
    for (int trial = 0; trial < kTrials; ++trial) {
        const double elapsed_ms = measure(activation, weight, rounds, grid, sink);
        const double bandwidth_gbs = (
            bytes_per_iteration * kIterations / elapsed_ms / 1.0e6);
        std::printf("sample,%d,%lld,%d,%d,%d,%.9f,%.6f\n", grid, rounds,
                    kThreads, kDynamicSharedBytes, trial, elapsed_ms,
                    bandwidth_gbs);
    }
    v4u first = {};
    v4u last = {};
    CHECK(mcMemcpy(&first, sink, sizeof(first), mcMemcpyDeviceToHost));
    CHECK(mcMemcpy(&last, sink + static_cast<long long>(grid) * kThreads - 1,
                   sizeof(last), mcMemcpyDeviceToHost));
    if ((first[0] | first[1] | first[2] | first[3]) == 0
            || (last[0] | last[1] | last[2] | last[3]) == 0) {
        std::fprintf(stderr, "operand sink validation failed\n");
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main() {
    mcDeviceProp_t properties;
    CHECK(mcGetDeviceProperties(&properties, 0));
    if (std::strstr(properties.name, "C500") == nullptr
            || properties.multiProcessorCount != kAps
            || properties.warpSize != kWaveSize) {
        std::fprintf(stderr, "unexpected device %s aps=%d warp=%d\n",
                     properties.name, properties.multiProcessorCount,
                     properties.warpSize);
        return EXIT_FAILURE;
    }
    mcFuncAttributes attributes;
    const void* kernel = reinterpret_cast<const void*>(
        c500_bm32_operand_residency);
    CHECK(mcFuncGetAttributes(&attributes, kernel));
    int resident_ctas = 0;
    CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_ctas, kernel, kThreads, kDynamicSharedBytes));
    if (attributes.sharedSizeBytes != 0 || resident_ctas != 1) {
        std::fprintf(stderr,
                     "resource contract failed registers=%d static_shared=%zu "
                     "dynamic_shared=%d resident_ctas_per_ap=%d\n",
                     attributes.numRegs,
                     static_cast<std::size_t>(attributes.sharedSizeBytes),
                     kDynamicSharedBytes, resident_ctas);
        return EXIT_FAILURE;
    }

    v4u* activation = nullptr;
    v4u* weight = nullptr;
    v4u* sink = nullptr;
    const long long source_vectors = kSourceBytes / sizeof(v4u);
    const int maximum_grid = kAps * 16;
    CHECK(mcMalloc(&activation, kSourceBytes));
    CHECK(mcMalloc(&weight, kSourceBytes));
    CHECK(mcMalloc(&sink, static_cast<std::size_t>(maximum_grid) * kThreads
                       * sizeof(v4u)));
    CHECK(mcMemset(activation, 1, kSourceBytes));
    CHECK(mcMemset(weight, 1, kSourceBytes));
    CHECK(mcMemset(sink, 0, static_cast<std::size_t>(maximum_grid) * kThreads
                       * sizeof(v4u)));

    std::fprintf(stderr,
                 "resource,kernel=c500_bm32_operand_residency,registers=%d,"
                 "static_shared=%zu,dynamic_shared=%d,resident_ctas_per_ap=%d,"
                 "source_bytes=%zu\n",
                 attributes.numRegs,
                 static_cast<std::size_t>(attributes.sharedSizeBytes),
                 kDynamicSharedBytes, resident_ctas, kSourceBytes);
    std::printf("sample_header,grid,rounds,block,dynamic_shared_bytes,trial,"
                "event_ms,logical_operand_bandwidth_gbs\n");
    run_grid(activation, weight, kAps, sink, source_vectors);
    run_grid(activation, weight, maximum_grid, sink, source_vectors);

    CHECK(mcFree(activation));
    CHECK(mcFree(weight));
    CHECK(mcFree(sink));
    std::fprintf(stderr, "validation=pass\n");
    return EXIT_SUCCESS;
}
