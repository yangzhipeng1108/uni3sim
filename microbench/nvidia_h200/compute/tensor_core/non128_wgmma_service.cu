// Independent BF16 WGMMA batches and optional bounded FFMA/EX2 pressure.
// No global operand stream, max, normalization, KV ring or Attention operator.
#include "non128_wgmma_helper.cuh"
#include <cmath>

__device__ __forceinline__ unsigned long long global_ns() {
    unsigned long long value;
    asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(value) :: "memory");
    return value;
}

template<int Kind, int N, int Steps, bool RS, int Values>
__device__ __forceinline__ float primitive(uint64_t a, uint64_t b, int repeats) {
    float accum[N / 2], values[Values];
    uint32_t fragments[Steps][4];
    #pragma unroll
    for (int i = 0; i < N / 2; ++i) accum[i] = 0;
    #pragma unroll
    for (int i = 0; i < Values; ++i) values[i] = -.5f - .0001f * i;
    #pragma unroll
    for (int k = 0; k < Steps; ++k) {
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            fragments[k][i] = 0x3e003e00u + ((threadIdx.x * 17 + k * 4 + i) & 127);
    }
    #pragma unroll 1
    for (int r = 0; r < repeats; ++r) {
        if constexpr (Kind != 1) {
            wg_fence();
            #pragma unroll
            for (int k = 0; k < Steps; ++k) {
                if constexpr (RS && N == 64) wgmma_rs_n64_mnmajor(accum, fragments[k], b, 1);
                if constexpr (RS && N == 128) wgmma_rs_n128_mnmajor(accum, fragments[k], b, 1);
                if constexpr (!RS && N == 128) wgmma_ss_n128_kmajor(accum, a, b, 1);
                if constexpr (!RS && N == 176) wgmma_ss_n176_kmajor(accum, a, b, 1);
                if constexpr (!RS && N == 192) wgmma_ss_n192_kmajor(accum, a, b, 1);
            }
            wg_commit();
        }
        if constexpr (Kind != 0) {
            unsigned bits = unsigned(r) * 9973u + threadIdx.x;
            float bias = -__uint_as_float((bits & 0x007fffffu) | 0x3f800000u);
            #pragma unroll
            for (int i = 0; i < Values; ++i) {
                float shifted;
                asm volatile("fma.rn.ftz.f32 %0,%1,%2,%3;" : "=f"(shifted) :
                    "f"(values[i]), "f"(values[(i + 17) % Values]), "f"(bias));
                asm volatile("ex2.approx.ftz.f32 %0,%1;" : "=f"(values[i]) : "f"(shifted));
            }
        }
        if constexpr (Kind != 1) wg_wait0();
    }
    float sum = 0;
    #pragma unroll
    for (int i = 0; i < N / 2; ++i) sum += accum[i];
    #pragma unroll
    for (int i = 0; i < Values; ++i) sum += values[i];
    return sum;
}

template<int N, int Steps, bool RS, int Values>
__global__ void __launch_bounds__(256, 1)
shape_service(int mode, int active, int repeats, unsigned long long* trace, float* sink) {
    extern __shared__ __align__(1024) unsigned char shared[];
    for (int i = threadIdx.x; i < kSmemBytes / 2; i += blockDim.x)
        reinterpret_cast<unsigned short*>(shared)[i] = 0x3f00u + ((i * 17) & 127);
    unsigned base = static_cast<unsigned>(__cvta_generic_to_shared(shared));
    uint64_t a = make_desc(base, 16, 1024), b = make_desc(base + 65536, 16, 1024);
    int wg = threadIdx.x / 128;
    __syncthreads();
    auto c0 = clock64();
    auto n0 = global_ns();
    float value = 0;
    if (wg < active) {
        if (mode == 0) value = primitive<0, N, Steps, RS, Values>(a, b, repeats);
        else if (mode == 1) value = primitive<1, N, Steps, RS, Values>(a, b, repeats);
        else value = primitive<2, N, Steps, RS, Values>(a, b, repeats);
    }
    auto n1 = global_ns();
    auto c1 = clock64();
    __syncthreads();
    if (threadIdx.x % 128 == 0) {
        int i = blockIdx.x * 2 + wg;
        trace[i * 2] = c1 - c0;
        trace[i * 2 + 1] = n1 - n0;
        sink[i] = value;
    }
}

template<int N, int Steps, bool RS, int Values>
void run(int grid, int active, int repeats, unsigned long long* trace, float* sink) {
    auto kernel = shape_service<N, Steps, RS, Values>;
    CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    for (int mode = 0; mode < 3; ++mode) for (int trial = -1; trial < 3; ++trial) {
        kernel<<<grid, 256, kSmemBytes>>>(mode, active, repeats, trace, sink);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        std::vector<unsigned long long> times(grid * 4);
        std::vector<float> values(grid * 2);
        CHECK(cudaMemcpy(times.data(), trace, times.size() * sizeof(*trace), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(values.data(), sink, values.size() * sizeof(*sink), cudaMemcpyDeviceToHost));
        for (int cta = 0; cta < grid; ++cta) for (int wg = 0; wg < active; ++wg) {
            int i = cta * 2 + wg;
            if (!times[i * 2] || !times[i * 2 + 1] || !std::isfinite(values[i])) std::exit(3);
            if (mode != 1 && times[i * 2] < repeats * Steps * 16ull) std::exit(4);
            if (trial >= 0) std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu\n",
                N, Steps, int(RS), Values, grid, active, repeats, mode, trial, cta, wg,
                times[i * 2], times[i * 2 + 1]);
        }
    }
}

int main() {
    unsigned long long* trace; float* sink;
    CHECK(cudaMalloc(&trace, 132 * 4 * sizeof(*trace)));
    CHECK(cudaMalloc(&sink, 132 * 2 * sizeof(*sink)));
    std::printf("n,steps,register_a,scalar_values,grid,active_wgs,repeats,mode,trial,cta,wg,cycles,global_ns\n");
    for (int grid : {1, 32, 132}) for (int active : {1, 2}) for (int repeats : {128, 256, 512}) {
        run<176, 8, false, 88>(grid, active, repeats, trace, sink);
        run<192, 4, false, 96>(grid, active, repeats, trace, sink);
        run<128, 8, false, 64>(grid, active, repeats, trace, sink);
        run<128, 11, true, 88>(grid, active, repeats, trace, sink);
        run<64, 12, true, 96>(grid, active, repeats, trace, sink);
        run<128, 8, true, 64>(grid, active, repeats, trace, sink);
    }
    CHECK(cudaFree(trace)); CHECK(cudaFree(sink));
}
