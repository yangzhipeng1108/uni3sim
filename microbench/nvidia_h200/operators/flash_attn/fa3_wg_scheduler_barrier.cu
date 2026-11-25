// Independent H200 FA3 two-warpgroup scheduler-barrier service probe.
//
// vLLM 0.30 pinned FA3 source 506341a, mainloop_fwd_sm90_tma_gmma_ws.hpp
// warp_scheduler_barrier_sync/arrive: each MMA warpgroup waits at its named
// barrier (participant count 256) and signals the other.  This probe repeats
// that handshake around a register-only hold, with no attention, tensor
// operands, WGMMA, TMA, global input or production shape.  The third
// warpgroup only reserves the 384-thread launch; 166912 B shared memory
// reserves the observed one-CTA-per-SM residency.
//
// mode 0: same register hold without barriers; mode 1: alternating handshake.
// Both are clock64 timed per warpgroup.  rounds 0/1/8/64 separate launch-local
// startup from the steady handshake; hold 0/64/256 probes overlap sensitivity.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, \
                 cudaGetErrorString(e)); std::exit(2); } } while (0)

struct Reading { unsigned long long cycles; unsigned int sink; };

__device__ __forceinline__ void arrive_one() {
    asm volatile("bar.arrive 1, 256;" ::: "memory");
}

__device__ __forceinline__ void arrive_two() {
    asm volatile("bar.arrive 2, 256;" ::: "memory");
}

__device__ __forceinline__ void wait_one() {
    asm volatile("bar.sync 1, 256;" ::: "memory");
}

__device__ __forceinline__ void wait_two() {
    asm volatile("bar.sync 2, 256;" ::: "memory");
}

__global__ void __launch_bounds__(384, 1)
probe(int mode, int rounds, int hold, Reading* readings) {
    extern __shared__ unsigned char reserve[];
    const int wg = threadIdx.x / 128;
    if (wg == 2) return;
    unsigned int value = unsigned(threadIdx.x + blockIdx.x + 1);
    // Keep the shared reservation live without adding an input-memory path.
    if (threadIdx.x == 0) reserve[0] = 0;
    __syncwarp();
    unsigned long long start = clock64();
    if (mode == 1 && rounds > 0 && wg == 0) arrive_one();
    for (int round = 0; round < rounds; ++round) {
        if (mode == 1) {
            if (wg == 0) wait_one(); else wait_two();
        }
        #pragma unroll 1
        for (int step = 0; step < hold; ++step) {
            asm volatile("add.u32 %0, %0, 1;" : "+r"(value));
        }
        if (mode == 1) {
            if (wg == 0) arrive_two(); else arrive_one();
        }
    }
    unsigned long long elapsed = clock64() - start;
    if (threadIdx.x % 128 == 0) {
        readings[blockIdx.x * 2 + wg] = {elapsed, value};
    }
}

int main() {
    Reading* device = nullptr;
    CHECK(cudaFuncSetAttribute(
        probe, cudaFuncAttributeMaxDynamicSharedMemorySize, 166912));
    for (int grid : {1, 132}) {
        CHECK(cudaMalloc(&device, sizeof(Reading) * grid * 2));
        std::vector<Reading> host(grid * 2);
        for (int mode : {0, 1}) for (int hold : {0, 64, 256})
        for (int rounds : {0, 1, 8, 64}) {
            for (int warm = 0; warm < 3; ++warm) {
                probe<<<grid, 384, 166912>>>(mode, rounds, hold, device);
                CHECK(cudaGetLastError());
            }
            CHECK(cudaDeviceSynchronize());
            probe<<<grid, 384, 166912>>>(mode, rounds, hold, device);
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(host.data(), device, sizeof(Reading) * grid * 2,
                             cudaMemcpyDeviceToHost));
            for (int cta = 0; cta < grid; ++cta)
                for (int wg = 0; wg < 2; ++wg)
                    std::printf("%d,%d,%d,%d,%d,%d,%llu,%u\n",
                                grid, mode, hold, rounds, cta, wg,
                                host[cta * 2 + wg].cycles,
                                host[cta * 2 + wg].sink);
        }
        CHECK(cudaFree(device));
    }
    return 0;
}
