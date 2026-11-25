/*
 * MetaX C500 register-only BF16 MMA accumulator-independence sweep.
 *
 * The existing mma_throughput.cpp fixes four independent accumulator chains.
 * This probe keeps the same 16x16x16 BF16 instruction, full-chip wave counts,
 * event timing, and register-only data path while varying the number of live
 * accumulator chains.  It does not reproduce a production GEMM workload.
 *
 * Build in the prescribed C500 profiler container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch=native \
 *       --maca-path=/opt/maca -O3 mma_accumulator_sweep.cpp \
 *       -o mma_accumulator_sweep
 */

#include <mc_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do {                                                   \
    mcError_t error = (call);                                              \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",              \
                     __FILE__, __LINE__, mcGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

typedef __NATIVE_VECTOR__(2, int32_t) v2i;
typedef __NATIVE_VECTOR__(4, float) v4f;

__device__ __forceinline__ v4f apply_bf16_mma(v2i a, v2i b, v4f c) {
    return __builtin_mxc_mma_16x16x16bf16(a, b, c);
}

template <int Chains>
__global__ void mma_accumulator_kernel(
        v4f* sink, int iterations) {
    const v2i a = {0x3b803c00, 0x3a803b00};
    const v2i b = {0x3b003a80, 0x3c003b80};
    v4f accumulators[Chains];
    #pragma unroll
    for (int chain = 0; chain < Chains; ++chain) {
        const float seed = 0.125f * static_cast<float>(chain + 1);
        accumulators[chain] = {seed, seed + 0.125f,
                               seed + 0.25f, seed + 0.375f};
    }
    #pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        #pragma unroll
        for (int chain = 0; chain < Chains; ++chain) {
            accumulators[chain] = apply_bf16_mma(
                a, b, accumulators[chain]);
        }
    }
    v4f sum = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int chain = 0; chain < Chains; ++chain) {
        sum += accumulators[chain];
    }
    const long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    sink[thread] = sum;
}

template <int Chains>
static double median_elapsed_ms(
        int grid, int block, v4f* sink, int iterations) {
    constexpr int trials = 7;
    std::vector<double> samples;
    samples.reserve(trials);
    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    for (int warmup = 0; warmup < 3; ++warmup) {
        mma_accumulator_kernel<Chains><<<grid, block>>>(
            sink, std::max(1, iterations / 32));
    }
    CHECK(mcDeviceSynchronize());
    for (int trial = 0; trial < trials; ++trial) {
        CHECK(mcEventRecord(start));
        mma_accumulator_kernel<Chains><<<grid, block>>>(sink, iterations);
        CHECK(mcEventRecord(end));
        CHECK(mcDeviceSynchronize());
        float elapsed_ms = 0.0f;
        CHECK(mcEventElapsedTime(&elapsed_ms, start, end));
        if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f) {
            std::fprintf(stderr, "invalid elapsed time %.9g ms\n", elapsed_ms);
            std::exit(EXIT_FAILURE);
        }
        samples.push_back(elapsed_ms);
    }
    CHECK(mcEventDestroy(start));
    CHECK(mcEventDestroy(end));
    std::sort(samples.begin(), samples.end());
    return samples[trials / 2];
}

template <int Chains>
static void run_cell(
        int ap_count, int waves_per_ap, int iterations,
        double xcore_clock_hz, v4f* sink) {
    constexpr int block = 64;
    constexpr double flops_per_mma = 8192.0;
    const int grid = ap_count * waves_per_ap;
    const double elapsed_ms = median_elapsed_ms<Chains>(
        grid, block, sink, iterations);
    const double wave_mma_instructions = (
        static_cast<double>(grid) * Chains * iterations);
    const double device_cycles = elapsed_ms * 1.0e-3 * xcore_clock_hz;
    const double mma_per_ap_cycle = (
        wave_mma_instructions / ap_count / device_cycles);
    const double flops_per_ap_cycle = mma_per_ap_cycle * flops_per_mma;
    std::printf(
        "%d,%d,%d,%d,%d,%.6f,%.9f,%.6f\n",
        grid, block, waves_per_ap, Chains, iterations, elapsed_ms,
        mma_per_ap_cycle, flops_per_ap_cycle);
}

int main() {
    mcDeviceProp_t properties;
    CHECK(mcGetDeviceProperties(&properties, 0));
    if (properties.multiProcessorCount != 104 || properties.warpSize != 64) {
        std::fprintf(
            stderr, "expected C500 104 AP/wave64, got %d AP/wave%d\n",
            properties.multiProcessorCount, properties.warpSize);
        return EXIT_FAILURE;
    }
    constexpr int maximum_waves_per_ap = 32;
    constexpr int iterations = 16384;
    // The single-wave, two-chain cell completes too quickly for the C500
    // event-clock quantum to give a stable rate.  Keep its instruction and
    // data path unchanged, but lengthen only that independent mechanism
    // trial so every occupancy point has a multi-millisecond timing span.
    constexpr int two_chain_iterations = 65536;
    constexpr double xcore_clock_hz = 1125000000.0;
    v4f* sink = nullptr;
    CHECK(mcMalloc(
        &sink,
        static_cast<size_t>(properties.multiProcessorCount)
            * maximum_waves_per_ap * properties.warpSize * sizeof(v4f)));

    std::printf("device,%s\n", properties.name);
    std::printf("ap_count,%d\n", properties.multiProcessorCount);
    std::printf("wave_size,%d\n", properties.warpSize);
    std::printf("xcore_clock_hz,%.0f\n", xcore_clock_hz);
    std::printf("mma_input_dtype,bf16\n");
    std::printf("mma_accumulator_dtype,fp32\n");
    std::printf("mma_shape,16x16x16\n");
    std::printf("flops_per_mma,8192\n");
    std::printf(
        "grid,block,waves_per_ap,accumulator_chains,iterations,median_ms,"
        "mma_per_ap_cycle,flops_per_ap_cycle\n");
    for (const int waves_per_ap : {1, 2, 4, 8, 16, 32}) {
        run_cell<2>(properties.multiProcessorCount, waves_per_ap,
                    two_chain_iterations, xcore_clock_hz, sink);
        run_cell<4>(properties.multiProcessorCount, waves_per_ap,
                    iterations, xcore_clock_hz, sink);
        run_cell<8>(properties.multiProcessorCount, waves_per_ap,
                    iterations, xcore_clock_hz, sink);
        run_cell<16>(properties.multiProcessorCount, waves_per_ap,
                     iterations, xcore_clock_hz, sink);
        run_cell<32>(properties.multiProcessorCount, waves_per_ap,
                     iterations, xcore_clock_hz, sink);
    }

    CHECK(mcFree(sink));
    return EXIT_SUCCESS;
}
