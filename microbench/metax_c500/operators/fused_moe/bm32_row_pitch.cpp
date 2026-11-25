/*
 * MetaX C500 BM32 B-operand row-pitch mechanism probe.
 *
 * Independent mechanism microbenchmark, not a FusedMoE implementation.  The
 * pinned BF16 BM32/BN64/BK128 Triton program reads its B operand as 128
 * rows x 128 bytes per K round with a row pitch equal to the projection N
 * extent (source-visible stride_bk).  This probe keeps the deployed
 * compiler resource envelope (256-thread CTA, two 49,152-B shared stages,
 * one CTA barrier per round, shared reads, register-only BF16 MMA) and the
 * compiler-observed prefetch ordering, and varies only the B row pitch --
 * a continuous address-stream property, not a shape bucket.  A-operand
 * loads keep the flat stream pattern of bm32_round_cadence.cpp.
 *
 * No routing, expert selection, output tile, production tensor shape, or
 * production/Serving duration is present.
 *
 * Build in the prescribed C500 profiler container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch=native \
 *       --maca-path=/opt/maca -O3 -std=c++17 \
 *       bm32_row_pitch.cpp -o bm32_row_pitch
 *
 * Run (idle device, one GPU):
 *   MACA_VISIBLE_DEVICES=1 ./bm32_row_pitch
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
constexpr int kStageCount = 2;
constexpr int kVectorBytes = 16;
constexpr int kBRowsPerRound = 128;         // BK rows of one B stage
constexpr int kBVectorsPerRow = 8;          // 128 B per BN64 row
constexpr int kBLoadsPerThread = 4;         // 1024 vectors / 256 threads
constexpr int kDynamicSharedBytes = 49'152;
constexpr int kTrials = 7;
constexpr int kWarmups = 3;

constexpr std::size_t kStreamABytes = 512ULL * 1024ULL * 1024ULL;
constexpr std::size_t kStreamBBytes = 1ULL * 1024ULL * 1024ULL * 1024ULL;

typedef __NATIVE_VECTOR__(4, std::uint32_t) v4u;
typedef __NATIVE_VECTOR__(2, std::int32_t) v2i;
typedef __NATIVE_VECTOR__(4, float) v4f;

static_assert(sizeof(v4u) == kVectorBytes,
              "each operand request must be 128 bits");
static_assert(kBRowsPerRound * kBVectorsPerRow
                  == kThreads * kBLoadsPerThread,
              "one B stage must distribute exactly over the CTA");
static_assert(kStageCount * (kAStreams + kBLoadsPerThread) * kThreads
                  * kVectorBytes == kDynamicSharedBytes,
              "dynamic shared allocation must be exactly two BM32 stages");

struct OperandRegisters {
    v4u a[kAStreams];
    v4u b[kBLoadsPerThread];
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
        long long global_thread, long long stride, long long pitch_bytes,
        long long b_wrap_vectors) {
    OperandRegisters result {};
    const int thread = static_cast<int>(global_thread % kThreads);
    #pragma unroll
    for (int slot = 0; slot < kAStreams; ++slot) {
        result.a[slot] = load_register(
            activation + (round * kAStreams + slot) * stride + global_thread);
    }
    // Production B-stage pattern: kBRowsPerRound rows of 128 bytes at the
    // source-visible row pitch; rounds advance one full K panel slice.
    const long long pitch_vectors = pitch_bytes / kVectorBytes;
    const long long round_advance = (
        static_cast<long long>(blockIdx.x) * kBRowsPerRound * pitch_vectors
        + round * kBRowsPerRound * pitch_vectors * gridDim.x) % b_wrap_vectors;
    #pragma unroll
    for (int slot = 0; slot < kBLoadsPerThread; ++slot) {
        const int linear = thread + slot * kThreads;
        const int row = linear / kBVectorsPerRow;
        const int in_row = linear % kBVectorsPerRow;
        result.b[slot] = load_register(
            weight + (round_advance + row * pitch_vectors + in_row)
            % b_wrap_vectors);
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
    for (int slot = 0; slot < kBLoadsPerThread; ++slot) {
        shared[b_base + (stage * kBLoadsPerThread + slot) * kThreads + thread]
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
    const v4u b0 = shared[b_base + (stage * kBLoadsPerThread) * kThreads + peer_b];
    const v4u b1 = shared[b_base + (stage * kBLoadsPerThread + 1) * kThreads + peer_a];
    const v4u b2 = shared[b_base + (stage * kBLoadsPerThread + 2) * kThreads + peer_b];
    const v4u b3 = shared[b_base + (stage * kBLoadsPerThread + 3) * kThreads + peer_a];
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

extern "C" __global__ void c500_bm32_row_pitch(
        const v4u* activation, const v4u* weight, long long rounds,
        long long pitch_bytes, long long b_wrap_vectors, v4u* sink) {
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
    OperandRegisters current = load_operands(
        activation, weight, 0, global_thread, stride, pitch_bytes,
        b_wrap_vectors);
    store_stage(shared, 0, thread, current);
    for (long long round = 0; round < rounds; ++round) {
        const long long next_round = round + 1;
        OperandRegisters next {};
        const bool next_valid = next_round < rounds;
        if (next_valid) {
            #pragma unroll
            for (int slot = 0; slot < kAStreams; ++slot) {
                next.a[slot] = load_register(
                    activation
                    + (next_round * kAStreams + slot) * stride + global_thread);
            }
        }
        __syncthreads();
        if (next_valid) {
            #pragma unroll
            for (int slot = 0; slot < kBLoadsPerThread; ++slot) {
                const int linear = thread + slot * kThreads;
                const int row = linear / kBVectorsPerRow;
                const int in_row = linear % kBVectorsPerRow;
                const long long pitch_vectors = pitch_bytes / kVectorBytes;
                const long long round_advance = (
                    static_cast<long long>(blockIdx.x)
                        * kBRowsPerRound * pitch_vectors
                    + next_round * kBRowsPerRound * pitch_vectors * gridDim.x)
                    % b_wrap_vectors;
                next.b[slot] = load_register(
                    weight + (round_advance + row * pitch_vectors + in_row)
                    % b_wrap_vectors);
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

static double measure_cell(const v4u* activation, const v4u* weight,
                           long long rounds, long long pitch_bytes,
                           long long b_wrap_vectors, int grid, v4u* sink) {
    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    CHECK(mcEventRecord(start));
    c500_bm32_row_pitch<<<grid, kThreads, kDynamicSharedBytes>>>(
        activation, weight, rounds, pitch_bytes, b_wrap_vectors, sink);
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

static void validate_resources() {
    mcFuncAttributes attributes {};
    const void* const function =
        reinterpret_cast<const void*>(c500_bm32_row_pitch);
    CHECK(mcFuncGetAttributes(&attributes, function));
    int resident_ctas = 0;
    CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_ctas, function, kThreads, kDynamicSharedBytes));
    if (attributes.sharedSizeBytes != 0 || resident_ctas != 1) {
        std::fprintf(stderr,
                     "resource contract failed registers=%d "
                     "static_shared=%zu resident_ctas_per_ap=%d\n",
                     attributes.numRegs,
                     static_cast<std::size_t>(attributes.sharedSizeBytes),
                     resident_ctas);
        std::exit(EXIT_FAILURE);
    }
    std::fprintf(stderr,
                 "resource,mode=row_pitch,registers=%d,static_shared=%zu,"
                 "dynamic_shared=%d,resident_ctas_per_ap=%d\n",
                 attributes.numRegs,
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
    validate_resources();

    v4u* activation = nullptr;
    v4u* weight = nullptr;
    v4u* sink = nullptr;
    const int maximum_grid = kAps * 31;
    CHECK(mcMalloc(&activation, kStreamABytes));
    CHECK(mcMalloc(&weight, kStreamBBytes));
    CHECK(mcMalloc(&sink, static_cast<std::size_t>(maximum_grid) * kThreads
                       * sizeof(v4u)));
    CHECK(mcMemset(activation, 1, kStreamABytes));
    CHECK(mcMemset(weight, 1, kStreamBBytes));
    CHECK(mcMemset(sink, 0, static_cast<std::size_t>(maximum_grid) * kThreads
                       * sizeof(v4u)));
    const long long b_wrap_vectors = kStreamBBytes / sizeof(v4u);

    std::printf("sample_header,grid,rounds,pitch_bytes,block,"
                "dynamic_shared_bytes,event_ms,b_logical_gbs\n");
    // Row-pitch sweep at the two production K depths.  3072 B is the W1
    // (N=1536) source pitch and 4096 B is the W2 (N=2048) source pitch for
    // the deployed qwen3-30b-a3b configuration; the surrounding points
    // isolate the continuous pitch mechanism from any one shape.
    const long long pitches[] = {1536, 2048, 3072, 4096, 6144, 8192};
    for (int grid : {kAps, kAps * 31}) {
        for (long long rounds : {6LL, 16LL}) {
            for (long long pitch : pitches) {
                for (int warmup = 0; warmup < kWarmups; ++warmup) {
                    c500_bm32_row_pitch<<<
                        grid, kThreads, kDynamicSharedBytes>>>(
                            activation, weight, rounds, pitch,
                            b_wrap_vectors, sink);
                    CHECK(mcGetLastError());
                }
                CHECK(mcDeviceSynchronize());
                const double b_bytes = static_cast<double>(rounds) * grid
                    * kBRowsPerRound * kBVectorsPerRow * kVectorBytes;
                for (int trial = 0; trial < kTrials; ++trial) {
                    const double elapsed_ms = measure_cell(
                        activation, weight, rounds, pitch, b_wrap_vectors,
                        grid, sink);
                    std::printf("sample,%d,%lld,%lld,%d,%d,%.9f,%.6f\n",
                                grid, rounds, pitch, kThreads,
                                kDynamicSharedBytes, elapsed_ms,
                                b_bytes / elapsed_ms / 1.0e6);
                }
            }
        }
    }

    v4u sample {};
    CHECK(mcMemcpy(&sample, sink, sizeof(sample), mcMemcpyDeviceToHost));
    if ((sample[0] | sample[1] | sample[2] | sample[3]) == 0) {
        std::fprintf(stderr, "row-pitch sink validation failed\n");
        return EXIT_FAILURE;
    }
    CHECK(mcFree(activation));
    CHECK(mcFree(weight));
    CHECK(mcFree(sink));
    std::fprintf(stderr, "validation=pass\n");
    return EXIT_SUCCESS;
}
