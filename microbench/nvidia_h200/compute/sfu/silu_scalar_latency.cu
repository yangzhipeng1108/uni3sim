/*
 * H200 owner copy of microbench/nvidia_a800/compute/sfu/silu_scalar_latency.cu.
 * Kernels, thread counts, iteration counts, trials and printed fields are
 * unchanged.  Hardware coupling only: built for sm_90 (was sm_80) by
 * silu_scalar_latency_runner.py, which pins the process to the owned H200 GPU
 * and clock-gates the capture.  The source has no device-identity check.
 *
 * A800 scalar-SiLU dependency microbenchmark (original description).
 *
 * This deliberately excludes production-kernel launch geometry and global
 * memory traffic.  It measures only register-resident scalar mechanisms used
 * by SiLU: expf, reciprocal/division, and their dependent arithmetic chain.
 * The independent-chain kernel exposes instruction throughput with multiple
 * accumulators; the dependent-chain kernel exposes the critical-path latency.
 *
 * Build:
 *   nvcc -O3 -arch=sm_90 -o silu_scalar_latency silu_scalar_latency.cu
 * Run on an idle GPU:
 *   CUDA_VISIBLE_DEVICES=<gpu> ./silu_scalar_latency
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>


namespace {

constexpr int kTrials = 1000;
constexpr int kWarmup = 100;


void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::fprintf(
            stderr, "%s failed: %s\n", expression,
            cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}


#define CHECK_CUDA(call) check_cuda((call), #call)


__device__ __forceinline__ float scalar_silu(float value) {
    return value / (1.0f + expf(-value));
}


__global__ void dependent_silu_chain(
        std::uint64_t* __restrict__ cycles, float* __restrict__ output,
        int iterations, int trial) {
    float value = 0.25f + static_cast<float>(threadIdx.x & 7) * 0.015625f;
    __syncthreads();
    const std::uint64_t begin = clock64();
    for (int index = 0; index < iterations; ++index) {
        value = scalar_silu(value) + 0.125f;
    }
    __syncthreads();
    const std::uint64_t end = clock64();
    output[threadIdx.x] = value;
    if (threadIdx.x == 0) {
        cycles[trial] = end - begin;
    }
}


__global__ void independent_silu_chains(
        std::uint64_t* __restrict__ cycles, float* __restrict__ output,
        int iterations, int trial) {
    float value0 = 0.125f + static_cast<float>(threadIdx.x & 7) * 0.0078125f;
    float value1 = value0 + 0.125f;
    float value2 = value0 + 0.250f;
    float value3 = value0 + 0.375f;
    float value4 = value0 + 0.500f;
    float value5 = value0 + 0.625f;
    float value6 = value0 + 0.750f;
    float value7 = value0 + 0.875f;
    __syncthreads();
    const std::uint64_t begin = clock64();
    for (int index = 0; index < iterations; ++index) {
        value0 = scalar_silu(value0) + 0.0625f;
        value1 = scalar_silu(value1) + 0.0625f;
        value2 = scalar_silu(value2) + 0.0625f;
        value3 = scalar_silu(value3) + 0.0625f;
        value4 = scalar_silu(value4) + 0.0625f;
        value5 = scalar_silu(value5) + 0.0625f;
        value6 = scalar_silu(value6) + 0.0625f;
        value7 = scalar_silu(value7) + 0.0625f;
    }
    __syncthreads();
    const std::uint64_t end = clock64();
    output[threadIdx.x] = (
        value0 + value1 + value2 + value3
        + value4 + value5 + value6 + value7);
    if (threadIdx.x == 0) {
        cycles[trial] = end - begin;
    }
}


std::uint64_t median(std::vector<std::uint64_t> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}


template <typename Kernel>
std::uint64_t measure(
        Kernel kernel, int threads, int iterations,
        std::uint64_t* cycles_device, float* output_device) {
    for (int index = 0; index < kWarmup; ++index) {
        kernel<<<1, threads>>>(
            cycles_device, output_device, iterations, 0);
    }
    for (int trial = 0; trial < kTrials; ++trial) {
        kernel<<<1, threads>>>(
            cycles_device, output_device, iterations, trial);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<std::uint64_t> cycles(kTrials);
    CHECK_CUDA(
        cudaMemcpy(
            cycles.data(), cycles_device,
            cycles.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
    return median(cycles);
}

}  // namespace


int main() {
    int device = -1;
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));

    std::uint64_t* cycles_device = nullptr;
    float* output_device = nullptr;
    CHECK_CUDA(cudaMalloc(
        &cycles_device, kTrials * sizeof(std::uint64_t)));
    CHECK_CUDA(cudaMalloc(
        &output_device, 1024 * sizeof(float)));

    std::printf(
        "device=%d name=%s clock_khz=%d trials=%d\n",
        device, properties.name, properties.clockRate, kTrials);
    for (int threads : {32, 128, 512, 1024}) {
        const std::uint64_t dependent_baseline = measure(
            dependent_silu_chain, threads, 0,
            cycles_device, output_device);
        const std::uint64_t independent_baseline = measure(
            independent_silu_chains, threads, 0,
            cycles_device, output_device);
        for (int iterations : {1, 2, 4, 8, 16, 32, 64}) {
            const std::uint64_t dependent = measure(
                dependent_silu_chain, threads, iterations,
                cycles_device, output_device);
            const std::uint64_t independent = measure(
                independent_silu_chains, threads, iterations,
                cycles_device, output_device);
            const double dependent_per_scalar = static_cast<double>(
                dependent - dependent_baseline) / iterations;
            const double independent_per_scalar = static_cast<double>(
                independent - independent_baseline) / (8 * iterations);
            std::printf(
                "threads=%d iterations=%d dependent_cycles=%llu "
                "dependent_cycles_per_scalar=%.3f "
                "independent_cycles=%llu "
                "independent_cycles_per_scalar=%.3f\n",
                threads, iterations,
                static_cast<unsigned long long>(dependent),
                dependent_per_scalar,
                static_cast<unsigned long long>(independent),
                independent_per_scalar);
        }
    }

    CHECK_CUDA(cudaFree(output_device));
    CHECK_CUDA(cudaFree(cycles_device));
    return EXIT_SUCCESS;
}
