/*
 * MetaX C500 FP16/BF16 16x16x16 MMA latency/throughput benchmark.
 *
 * One wave-level __builtin_mxc_mma_16x16x16{f16,bf16} instruction performs
 * 16*16*16 FMAs = 8192 FLOPs.  Operands and accumulators stay in registers;
 * the timed loop contains only one selected MMA dtype.  A dependent
 * accumulator measures latency and four independent accumulators measure
 * issue throughput.  Keeping both dtypes in one source prevents the BF16
 * production path from inheriting an FP16 capacity without direct evidence.
 *
 * Build in the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch=native \
 *       --maca-path=/opt/maca -O3 mma_throughput.cpp -o mma_throughput
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

typedef __NATIVE_VECTOR__(4, _Float16) v4h;
typedef __NATIVE_VECTOR__(2, int32_t) v2i;
typedef __NATIVE_VECTOR__(4, float) v4f;

struct Fp16Mma {
    using Operand = v4h;

    __device__ __forceinline__ static Operand operand_a() {
        return {
            static_cast<_Float16>(0.0078125f),
            static_cast<_Float16>(0.00390625f),
            static_cast<_Float16>(0.001953125f),
            static_cast<_Float16>(0.0009765625f),
        };
    }

    __device__ __forceinline__ static Operand operand_b() {
        return {
            static_cast<_Float16>(0.0009765625f),
            static_cast<_Float16>(0.001953125f),
            static_cast<_Float16>(0.00390625f),
            static_cast<_Float16>(0.0078125f),
        };
    }

    __device__ __forceinline__ static v4f apply(
            Operand a, Operand b, v4f c) {
        return __builtin_mxc_mma_16x16x16f16(a, b, c);
    }
};

struct Bf16Mma {
    using Operand = v2i;

    __device__ __forceinline__ static Operand operand_a() {
        // Four packed BF16 values: 0.0078125, 0.00390625, 0.001953125,
        // and 0.0009765625.  Raw packing keeps the timed path free of
        // conversion instructions.
        return {0x3b803c00, 0x3a803b00};
    }

    __device__ __forceinline__ static Operand operand_b() {
        return {0x3b003a80, 0x3c003b80};
    }

    __device__ __forceinline__ static v4f apply(
            Operand a, Operand b, v4f c) {
        return __builtin_mxc_mma_16x16x16bf16(a, b, c);
    }
};

template <typename Mma>
__global__ void mma_dependent_latency_kernel(
        v4f* sink, unsigned long long* elapsed_cycles, int iterations) {
    typename Mma::Operand a = Mma::operand_a();
    typename Mma::Operand b = Mma::operand_b();
    v4f c = {0.125f, 0.25f, 0.5f, 1.0f};
    unsigned long long start = clock64();
    #pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        c = Mma::apply(a, b, c);
    }
    unsigned long long end = clock64();
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    sink[thread] = c;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        elapsed_cycles[0] = end - start;
    }
}

template <typename Mma>
__global__ void mma_independent_throughput_kernel(
        v4f* sink, int iterations) {
    typename Mma::Operand a = Mma::operand_a();
    typename Mma::Operand b = Mma::operand_b();
    v4f c0 = {0.125f, 0.25f, 0.5f, 1.0f};
    v4f c1 = {0.25f, 0.5f, 1.0f, 0.125f};
    v4f c2 = {0.5f, 1.0f, 0.125f, 0.25f};
    v4f c3 = {1.0f, 0.125f, 0.25f, 0.5f};
    #pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        c0 = Mma::apply(a, b, c0);
        c1 = Mma::apply(a, b, c1);
        c2 = Mma::apply(a, b, c2);
        c3 = Mma::apply(a, b, c3);
    }
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    sink[thread] = {
        c0.x + c1.x + c2.x + c3.x,
        c0.y + c1.y + c2.y + c3.y,
        c0.z + c1.z + c2.z + c3.z,
        c0.w + c1.w + c2.w + c3.w,
    };
}

template <typename Mma>
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
        mma_independent_throughput_kernel<Mma><<<grid, block>>>(
            sink, std::max(1, iterations / 32));
    }
    CHECK(mcDeviceSynchronize());
    for (int trial = 0; trial < trials; ++trial) {
        CHECK(mcEventRecord(start));
        mma_independent_throughput_kernel<Mma><<<grid, block>>>(
            sink, iterations);
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

template <typename Mma>
static void run_dtype(
        const char* dtype, int ap_count, int block, int max_waves_per_ap,
        int independent_chains, int latency_iterations,
        int throughput_iterations, double flops_per_mma,
        double xcore_clock_hz, const int* wave_multipliers,
        int multiplier_count, v4f* sink, unsigned long long* cycles) {
    mma_dependent_latency_kernel<Mma><<<1, block>>>(
        sink, cycles, latency_iterations);
    CHECK(mcDeviceSynchronize());
    unsigned long long elapsed_cycles = 0;
    CHECK(mcMemcpy(
        &elapsed_cycles, cycles, sizeof(elapsed_cycles),
        mcMemcpyDeviceToHost));

    std::printf("mma_input_dtype,%s\n", dtype);
    std::printf("dependent_cycles_per_mma,%.6f\n",
                static_cast<double>(elapsed_cycles) / latency_iterations);
    std::printf(
        "grid,block,waves_per_ap,iterations,chains,median_ms,"
        "mma_per_ap_cycle,flops_per_ap_cycle,flops_per_peu_cycle\n");
    for (int index = 0; index < multiplier_count; ++index) {
        int waves_per_ap = wave_multipliers[index];
        if (waves_per_ap <= 0 || waves_per_ap > max_waves_per_ap) {
            std::fprintf(stderr, "invalid waves/AP %d\n", waves_per_ap);
            std::exit(EXIT_FAILURE);
        }
        int grid = ap_count * waves_per_ap;
        double elapsed_ms = median_elapsed_ms<Mma>(
            grid, block, sink, throughput_iterations);
        double wave_mma_instructions = (
            static_cast<double>(grid) * independent_chains
            * throughput_iterations);
        double device_cycles = elapsed_ms * 1.0e-3 * xcore_clock_hz;
        double mma_per_ap_cycle = (
            wave_mma_instructions / ap_count / device_cycles);
        double flops_per_ap_cycle = mma_per_ap_cycle * flops_per_mma;
        std::printf(
            "%d,%d,%d,%d,%d,%.6f,%.9f,%.6f,%.6f\n",
            grid, block, waves_per_ap, throughput_iterations,
            independent_chains, elapsed_ms, mma_per_ap_cycle,
            flops_per_ap_cycle, flops_per_ap_cycle / 4.0);
    }
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
    constexpr int block = 64;
    constexpr int max_waves_per_ap = 32;
    constexpr int independent_chains = 4;
    constexpr int latency_iterations = 262144;
    constexpr int throughput_iterations = 16384;
    constexpr double flops_per_mma = 8192.0;
    constexpr double xcore_clock_hz = 1125000000.0;
    const int wave_multipliers[] = {1, 2, 4, 8, 16, max_waves_per_ap};

    v4f* sink = nullptr;
    unsigned long long* cycles = nullptr;
    CHECK(mcMalloc(
        &sink,
        static_cast<size_t>(properties.multiProcessorCount)
            * max_waves_per_ap * block
            * sizeof(v4f)));
    CHECK(mcMalloc(&cycles, sizeof(unsigned long long)));

    std::printf("device,%s\n", properties.name);
    std::printf("ap_count,%d\n", properties.multiProcessorCount);
    std::printf("wave_size,%d\n", properties.warpSize);
    std::printf("xcore_clock_hz,%.0f\n", xcore_clock_hz);
    std::printf("mma_shape,16x16x16\n");
    std::printf("mma_accumulator_dtype,fp32\n");
    std::printf("flops_per_mma,%.0f\n", flops_per_mma);
    constexpr int multiplier_count = (
        sizeof(wave_multipliers) / sizeof(wave_multipliers[0]));
    run_dtype<Fp16Mma>(
        "fp16", properties.multiProcessorCount, block, max_waves_per_ap,
        independent_chains, latency_iterations, throughput_iterations,
        flops_per_mma, xcore_clock_hz, wave_multipliers, multiplier_count,
        sink, cycles);
    run_dtype<Bf16Mma>(
        "bf16", properties.multiProcessorCount, block, max_waves_per_ap,
        independent_chains, latency_iterations, throughput_iterations,
        flops_per_mma, xcore_clock_hz, wave_multipliers, multiplier_count,
        sink, cycles);

    CHECK(mcFree(cycles));
    CHECK(mcFree(sink));
    return EXIT_SUCCESS;
}
