// Independent, staggered SS WGMMA and FFMA/EX2 bursts in separate WGs.
// No Attention, max/reduction, KV stream or production workload.
#include "phase_wgmma_helper.cuh"
#include <cmath>

constexpr int TraceWords = 6;

__device__ __forceinline__ unsigned long long global_ns() {
    unsigned long long value;
    asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(value) :: "memory");
    return value;
}

template<int Matrices, int Values>
__global__ void __launch_bounds__(256, 1)
phase_pair(int mode, int offset, unsigned long long* trace, float* sink) {
    extern __shared__ __align__(1024) unsigned char shared[];
    __shared__ unsigned long long epoch;
    for (int i = threadIdx.x; i < kSmemBytes / 2; i += blockDim.x)
        reinterpret_cast<unsigned short*>(shared)[i] = 0x3f00u + ((i * 17) & 127);
    unsigned base = static_cast<unsigned>(__cvta_generic_to_shared(shared));
    uint64_t a = make_desc(base, 16, 1024), b = make_desc(base + 65536, 16, 1024);
    float accum[64], values[Values];
    #pragma unroll
    for (int i = 0; i < 64; ++i) accum[i] = 0;
    #pragma unroll
    for (int i = 0; i < Values; ++i) values[i] = -.5f - .0001f * i;
    int wg = threadIdx.x / 128;
    __syncthreads();
    if (threadIdx.x == 0) epoch = clock64();
    __syncthreads();
    // This control exists in every mode, including isolated Tensor. Record
    // actual entry times; the requested offset is not assumed exact.
    if (wg == 1)
        while (clock64() - epoch < static_cast<unsigned>(offset)) {}
    unsigned long long c0 = epoch, c1 = epoch, n0 = 0, n1 = 0, enqueued = epoch;
    if (wg == 0 && (mode == 1 || mode == 3)) {
        // Force all physical accumulator initializations before the inner
        // timestamp. The empty asm preserves values but makes each input
        // explicit to the compiler; private setup is not a Tensor service.
        #pragma unroll
        for (int i = 0; i < 64; ++i) asm volatile("" : "+f"(accum[i]));
        wg_fence();
        c0 = clock64();
        n0 = global_ns();
        #pragma unroll
        for (int m = 0; m < Matrices; ++m) wgmma_ss_n128_kmajor(accum, a, b, 1);
        wg_commit();
        enqueued = clock64();
        wg_wait0();
        n1 = global_ns();
        c1 = clock64();
    } else if (wg == 1 && (mode == 2 || mode == 3)) {
        float bias = -.75f - .00001f * (threadIdx.x & 31);
        #pragma unroll
        for (int i = 0; i < Values; ++i) asm volatile("" : "+f"(values[i]));
        asm volatile("" : "+f"(bias));
        c0 = clock64();
        n0 = global_ns();
        #pragma unroll
        for (int i = 0; i < Values; ++i) {
            float shifted;
            asm volatile("fma.rn.ftz.f32 %0,%1,%2,%3;" : "=f"(shifted) :
                "f"(values[i]), "f"(values[(i + 17) % Values]), "f"(bias));
            asm volatile("ex2.approx.ftz.f32 %0,%1;" : "=f"(values[i]) : "f"(shifted));
        }
        n1 = global_ns();
        c1 = clock64();
        enqueued = c0;
    }
    // Keep all accumulator and scalar registers live across the timers.
    float sum = 0;
    #pragma unroll
    for (int i = 0; i < 64; ++i) sum += accum[i];
    #pragma unroll
    for (int i = 0; i < Values; ++i) sum += values[i];
    // Trace stores cannot interfere with an unfinished peer burst. Timers
    // precede this barrier; inactive WGs are excluded during aggregation.
    __syncthreads();
    if (threadIdx.x % 128 == 0) {
        int index = blockIdx.x * 2 + wg;
        trace[index * TraceWords] = c0 - epoch;
        trace[index * TraceWords + 1] = c1 - epoch;
        trace[index * TraceWords + 2] = c1 - c0;
        trace[index * TraceWords + 3] = n1 - n0;
        trace[index * TraceWords + 4] = n0;
        trace[index * TraceWords + 5] = enqueued - epoch;
        sink[index] = sum;
    }
}

template<int Matrices, int Values>
void run(int grid, int offset, unsigned long long* trace, float* sink) {
    auto kernel = phase_pair<Matrices, Values>;
    CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    for (int mode = 0; mode < 4; ++mode) for (int trial = -2; trial < 9; ++trial) {
        kernel<<<grid, 256, kSmemBytes>>>(mode, offset, trace, sink);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        std::vector<unsigned long long> times(grid * 2 * TraceWords);
        std::vector<float> values(grid * 2);
        CHECK(cudaMemcpy(times.data(), trace, times.size() * sizeof(*trace), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(values.data(), sink, values.size() * sizeof(*sink), cudaMemcpyDeviceToHost));
        for (int cta = 0; cta < grid; ++cta) for (int wg = 0; wg < 2; ++wg) {
            int i = (cta * 2 + wg) * TraceWords;
            bool active = (wg == 0 && (mode == 1 || mode == 3))
                       || (wg == 1 && (mode == 2 || mode == 3));
            if ((active && (!times[i + 2] || !times[i + 3])) || !std::isfinite(values[cta * 2 + wg])) {
                std::fprintf(stderr, "invalid independent phase result\n");
                std::exit(3);
            }
            if (wg == 0 && (mode == 1 || mode == 3) && times[i + 2] < Matrices * 32ull) {
                std::fprintf(stderr, "Tensor window below conservative instruction-work floor\n");
                std::exit(4);
            }
            if (trial >= 0)
                std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%llu\n",
                    Matrices, Values, grid, offset, mode, trial, cta, wg,
                    times[i], times[i + 1], times[i + 2], times[i + 3], times[i + 4], times[i + 5]);
        }
    }
}

int main() {
    unsigned long long* trace;
    float* sink;
    CHECK(cudaMalloc(&trace, 132 * 2 * TraceWords * sizeof(*trace)));
    CHECK(cudaMalloc(&sink, 132 * 2 * sizeof(*sink)));
    std::printf("matrices,values,grid,requested_offset_cycles,mode,trial,cta,wg,entry_cycles,exit_cycles,elapsed_cycles,elapsed_ns,entry_global_ns,enqueue_cycles\n");
    for (int grid : {1, 132}) for (int offset : {0, 128, 256, 512, 1024, 2048}) {
        run<4, 64>(grid, offset, trace, sink);
        run<8, 64>(grid, offset, trace, sink);
        run<16, 64>(grid, offset, trace, sink);
        run<4, 128>(grid, offset, trace, sink);
        run<8, 128>(grid, offset, trace, sink);
        run<16, 128>(grid, offset, trace, sink);
    }
    CHECK(cudaFree(trace));
    CHECK(cudaFree(sink));
}
