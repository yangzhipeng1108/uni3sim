// Measure a minimal warm default-cache global-memory dependency task.  One lane per
// warp executes a single load -> integer dependency -> store chain.  The body
// is operator-independent and measures the complete task floor for a kernel
// that must issue and drain global memory, rather than an empty/register-only
// scheduler floor.  Ordinary LDG.E/STG.E plus repeated warm-up does not prove
// exclusive L2 residency, so the report deliberately retains the weaker cache
// contract.

#include <cuda_runtime.h>
#include <nvToolsExt.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>


namespace {

constexpr int kWarmupLaunches = 200;
constexpr int kSweepLaunches = 200;
constexpr int kMaximumGrid = 4096;
constexpr int kMaximumBlock = 256;
constexpr int kWarpWidth = 32;
constexpr unsigned long long kClockWarmupCycles = 1140000000ULL;


__global__ void memory_dependency_task(
        const std::uint32_t* input, std::uint32_t* output) {
    const int lane = threadIdx.x & (kWarpWidth - 1);
    if (lane != 0) {
        return;
    }
    const int warps_per_block = (blockDim.x + kWarpWidth - 1) / kWarpWidth;
    const int warp = threadIdx.x / kWarpWidth;
    const int index = blockIdx.x * warps_per_block + warp;
    output[index] = input[index] + 1U;
}


__global__ void clock_warmup_kernel(float* sink) {
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    float x0 = 1.0009765625f + static_cast<float>(thread) * 0x1p-24f;
    float x1 = x0 + 0x1p-11f;
    float x2 = x0 + 0x1p-10f;
    float x3 = x0 + 0x1.8p-10f;
    const unsigned long long start = clock64();
    do {
        x0 = x0 * 0.9999847412109375f + 0.0000152587890625f;
        x1 = x1 * 0.9999847412109375f + 0.0000152587890625f;
        x2 = x2 * 0.9999847412109375f + 0.0000152587890625f;
        x3 = x3 * 0.9999847412109375f + 0.0000152587890625f;
    } while (clock64() - start < kClockWarmupCycles);
    sink[thread] = x0 + x1 + x2 + x3;
}


void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::fprintf(
            stderr, "%s failed: %s\n", expression,
            cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}


#define CHECK_CUDA(call) check_cuda((call), #call)


void emit_measurement_marker(const char* name) {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto nanoseconds =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    std::printf("%s=%lld\n", name, static_cast<long long>(nanoseconds));
    std::fflush(stdout);
}

}  // namespace


int main(int argc, char** argv) {
    if (argc != 2 || std::strcmp(argv[1], "--sweep") != 0) {
        std::fprintf(stderr, "usage: %s --sweep\n", argv[0]);
        return EXIT_FAILURE;
    }

    constexpr int maximum_warps =
        kMaximumGrid * ((kMaximumBlock + kWarpWidth - 1) / kWarpWidth);
    std::uint32_t* input = nullptr;
    std::uint32_t* output = nullptr;
    float* warmup_sink = nullptr;
    CHECK_CUDA(cudaMalloc(&input, maximum_warps * sizeof(*input)));
    CHECK_CUDA(cudaMalloc(&output, maximum_warps * sizeof(*output)));
    constexpr int warmup_grid = 864;
    constexpr int warmup_block = 256;
    CHECK_CUDA(cudaMalloc(
        &warmup_sink, warmup_grid * warmup_block * sizeof(*warmup_sink)));
    CHECK_CUDA(cudaMemset(input, 1, maximum_warps * sizeof(*input)));
    CHECK_CUDA(cudaMemset(output, 0, maximum_warps * sizeof(*output)));
    clock_warmup_kernel<<<warmup_grid, warmup_block>>>(warmup_sink);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    for (int index = 0; index < kWarmupLaunches; ++index) {
        memory_dependency_task<<<kMaximumGrid, kMaximumBlock>>>(
            input, output);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    emit_measurement_marker("measurement_started_unix_ns");
    // Keep an already-warmed GPU busy long enough for the independent 100 ms
    // nvidia-smi sampler to observe the clock inside the measurement window.
    // This kernel has no NVTX range and is excluded from task durations.
    clock_warmup_kernel<<<warmup_grid, warmup_block>>>(warmup_sink);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    char label[112];
    for (int grid : {256, 512, 1024, 2048, 4096}) {
        for (int block : {128, 256}) {
            std::snprintf(
                label, sizeof(label),
                "memory_dependency_grid_%d_block_%d", grid, block);
            for (int index = 0; index < kSweepLaunches; ++index) {
                nvtxRangePushA(label);
                memory_dependency_task<<<grid, block>>>(input, output);
                nvtxRangePop();
            }
        }
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    emit_measurement_marker("measurement_completed_unix_ns");
    CHECK_CUDA(cudaFree(input));
    CHECK_CUDA(cudaFree(output));
    CHECK_CUDA(cudaFree(warmup_sink));
    return EXIT_SUCCESS;
}
