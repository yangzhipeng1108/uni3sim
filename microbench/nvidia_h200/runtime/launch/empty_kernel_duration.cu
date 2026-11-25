// H200 owner copy of microbench/nvidia_a800/runtime/launch/empty_kernel_duration.cu.
// Kernel bodies, warm-up, launch counts and NVTX labelling are the A800 ones.
// Hardware / toolchain coupling changed (and nothing else):
//   - build: nvcc -O3 -arch=sm_90 -cudart static (CUDA 12.2), run on H200;
//   - NVTX: the header-only NVTX3 include <nvtx3/nvToolsExt.h> replaces the
//     legacy <nvToolsExt.h> + libnvToolsExt link (the H200 host's CUDA 12.9
//     ships no legacy libnvToolsExt); the API calls are identical;
//   - clock warm-up grid 864 = 8 x 108 becomes 8 x 132 = 1056;
//   - sweep grids: the A800 list plus the H200 full-chip multiples 132, 264
//     and 528 (1 x, 2 x, 4 x 132 SMs).
// Measure the GPU-device duration floor with the same nsys NVTX-kernel
// boundary used by the A800 Vector evaluation observations.  A register-only
// FMA warmup runs before all labeled launches so DVFS transition points cannot
// move through the geometry sweep; the warmup is outside every measured range.

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>


namespace {

constexpr int kWarmupLaunches = 200;
constexpr int kMeasuredLaunches = 2000;
constexpr unsigned long long kClockWarmupCycles = 1140000000ULL;


__global__ void empty_kernel() {}


__global__ void clock_warmup_kernel(float* sink) {
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    float x0 = 1.0009765625f + static_cast<float>(thread) * 0x1p-24f;
    float x1 = x0 + 0x1p-11f;
    float x2 = x0 + 0x1p-10f;
    float x3 = x0 + 0x1.8p-10f;
    float x4 = x0 + 0x1p-9f;
    float x5 = x0 + 0x1.4p-9f;
    float x6 = x0 + 0x1.8p-9f;
    float x7 = x0 + 0x1.cp-9f;
    const unsigned long long start = clock64();
    do {
        x0 = x0 * 0.9999847412109375f + 0.0000152587890625f;
        x1 = x1 * 0.9999847412109375f + 0.0000152587890625f;
        x2 = x2 * 0.9999847412109375f + 0.0000152587890625f;
        x3 = x3 * 0.9999847412109375f + 0.0000152587890625f;
        x4 = x4 * 0.9999847412109375f + 0.0000152587890625f;
        x5 = x5 * 0.9999847412109375f + 0.0000152587890625f;
        x6 = x6 * 0.9999847412109375f + 0.0000152587890625f;
        x7 = x7 * 0.9999847412109375f + 0.0000152587890625f;
    } while (clock64() - start < kClockWarmupCycles);
    sink[thread] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}


void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::fprintf(
            stderr, "%s failed: %s\n", expression, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}


#define CHECK_CUDA(call) check_cuda((call), #call)

}  // namespace


int main(int argc, char** argv) {
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--sweep") != 0)) {
        std::fprintf(stderr, "usage: %s [--sweep]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const bool sweep = argc == 2;
    float* warmup_sink = nullptr;
    constexpr int warmup_grid = 1056;
    constexpr int warmup_block = 256;
    CHECK_CUDA(cudaMalloc(
        &warmup_sink, warmup_grid * warmup_block * sizeof(float)));
    clock_warmup_kernel<<<warmup_grid, warmup_block>>>(warmup_sink);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    for (int index = 0; index < kWarmupLaunches; ++index) {
        empty_kernel<<<1, 1>>>();
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    if (sweep) {
        constexpr int kSweepLaunches = 200;
        char label[96];
        for (int grid : {
                1, 4, 16, 64, 108, 132, 216, 264, 432, 528, 1024, 4096,
                8192, 16384, 32768, 65536, 131072, 262144, 524288,
                1048576}) {
            for (int block : {1, 32, 64, 128, 256, 512, 1024}) {
                std::snprintf(label, sizeof(label),
                              "empty_kernel_grid_%d_block_%d", grid, block);
                for (int index = 0; index < kSweepLaunches; ++index) {
                    nvtxRangePushA(label);
                    empty_kernel<<<grid, block>>>();
                    nvtxRangePop();
                }
            }
        }
    } else {
        for (int index = 0; index < kMeasuredLaunches; ++index) {
            nvtxRangePushA("empty_kernel_device_duration");
            empty_kernel<<<1, 1>>>();
            nvtxRangePop();
        }
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(warmup_sink));
    return EXIT_SUCCESS;
}
