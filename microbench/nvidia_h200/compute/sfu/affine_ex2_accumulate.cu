// Independent bounded register FFMA -> EX2 -> dependent-ADD bundles.
// No attention, normalization, reduction across lanes, matrix or KV traffic.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(2); } } while (0)

struct Window { unsigned long long begin, end, ns_begin, ns_end; unsigned sm, nsm; };

__device__ __forceinline__ unsigned long long timer_ns() {
    unsigned long long x; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(x)); return x;
}

template<int Mode, int Chains, int Width>
__global__ __launch_bounds__(256, 1)
void affine_ex2(int repeats, int active_wgs, float scale, Window* windows, float* output) {
    float x[Chains], sums[Chains];
    #pragma unroll
    for (int c = 0; c < Chains; ++c) { x[c] = float((threadIdx.x * 31 + c * 13) % 3); sums[c] = 0.f; }
    const bool active = threadIdx.x < active_wgs * 128;
    __syncthreads();
    const unsigned long long begin = clock64(); const auto ns_begin = timer_ns();
    if (active) {
        #pragma unroll 1
        for (int repeat = 0; repeat < repeats; ++repeat) {
            if constexpr (Mode == 0) { asm volatile("" ::: "memory"); }
            else {
                float inputs[Chains];
                #pragma unroll
                for (int c = 0; c < Chains; ++c) inputs[c] = __fmaf_rn(float(repeat), 0x1p-20f, x[c]);
                #pragma unroll
                for (int i = 0; i < Width; ++i) {
                    #pragma unroll
                    for (int c = 0; c < Chains; ++c) {
                        if constexpr (Mode == 1) {
                            asm volatile("add.rn.f32 %0, %0, %1;" : "+f"(sums[c]) : "f"(x[c]));
                        } else {
                            float affine, exponential;
                            asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(affine) : "f"(inputs[c]), "f"(scale), "f"(-1.f + float(i) * 0x1p-10f));
                            asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(exponential) : "f"(affine));
                            asm volatile("add.rn.f32 %0, %0, %1;" : "+f"(sums[c]) : "f"(exponential));
                        }
                    }
                }
            }
        }
    }
    const auto ns_end = timer_ns(); const unsigned long long end = clock64();
    if ((threadIdx.x & 31) == 0 && active) {
        unsigned sm, nsm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
        asm volatile("mov.u32 %0, %%nsmid;" : "=r"(nsm));
        windows[blockIdx.x * 8 + threadIdx.x / 32] = {begin, end, ns_begin, ns_end, sm, nsm};
    }
    #pragma unroll
    for (int c = 0; c < Chains; ++c) output[(blockIdx.x * 256 + threadIdx.x) * Chains + c] = sums[c];
}

template<int Mode, int Chains, int Width>
void capture(int trial, int grid, int active_wgs, int repeats) {
    Window* windows; float* output;
    CHECK(cudaMalloc(&windows, grid * 8 * sizeof(Window)));
    CHECK(cudaMalloc(&output, grid * 256 * Chains * sizeof(float)));
    CHECK(cudaFuncSetAttribute(affine_ex2<Mode,Chains,Width>, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024));
    for (int warm = 0; warm < 2; ++warm) {
        affine_ex2<Mode,Chains,Width><<<grid, 256, 200 * 1024>>>(repeats, active_wgs, 1.f, windows, output);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    }
    std::vector<Window> host(grid * 8); std::vector<float> values(grid * 256 * Chains);
    CHECK(cudaMemcpy(host.data(), windows, host.size() * sizeof(Window), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(values.data(), output, values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    float reference[3] = {};
    for (int input = 0; input < 3; ++input) for (int r = 0; r < repeats; ++r) for (int i = 0; i < Width; ++i) {
        float current = std::fma(float(r), 0x1p-20f, float(input));
        float affine = std::fma(current, 1.f, -1.f + float(i) * 0x1p-10f);
        reference[input] += std::exp2(affine);
    }
    for (int b = 0; b < grid; ++b) {
        for (int t = 0; t < 256; ++t) for (int c = 0; c < Chains; ++c) {
            const int input = (t * 31 + c * 13) % 3;
            const float expected = Mode == 0 || t >= active_wgs * 128 ? 0.f
                : Mode == 2 ? reference[input] : float(repeats * Width * input);
            const float actual = values[(b * 256 + t) * Chains + c];
            if (!std::isfinite(actual) || std::fabs(actual - expected) > 0.00001f * std::fmax(1.f, expected)) {
                std::fprintf(stderr, "numeric %d/%d/%d/%d/%d/%d\n",Mode,Chains,Width,b,t,c); std::exit(3);
            }
        }
        for (int w = 0; w < active_wgs * 4; ++w) {
            const auto& r = host[b * 8 + w];
            if (r.end <= r.begin || r.ns_end < r.ns_begin || r.sm >= r.nsm) std::exit(4);
            std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%u,%u,%llu,%llu,%llu,%llu\n",
                trial,Mode,Chains,Width,grid,active_wgs,repeats,b,w,r.sm,r.nsm,r.begin,r.end,r.ns_begin,r.ns_end);
        }
    }
    CHECK(cudaFree(output)); CHECK(cudaFree(windows));
}

template<int Mode, int Chains>
void widths(int trial, int grid, int active, int repeats) {
    if (trial % 2) { capture<Mode,Chains,64>(trial,grid,active,repeats); capture<Mode,Chains,32>(trial,grid,active,repeats); capture<Mode,Chains,16>(trial,grid,active,repeats); }
    else { capture<Mode,Chains,16>(trial,grid,active,repeats); capture<Mode,Chains,32>(trial,grid,active,repeats); capture<Mode,Chains,64>(trial,grid,active,repeats); }
}

int main() {
    cudaDeviceProp p; CHECK(cudaGetDeviceProperties(&p,0));
    if (p.major != 9 || p.multiProcessorCount != 132) return 5;
    std::puts("trial,mode,chains,width,grid,wgs,repeats,cta,warp,sm,nsm,begin,end,ns_begin,ns_end");
    for (int trial = 0; trial < 3; ++trial) for (int grid : {1,132}) for (int wgs : {1,2}) for (int repeats : {64,128,256}) {
        widths<0,2>(trial,grid,wgs,repeats); widths<1,2>(trial,grid,wgs,repeats); widths<2,2>(trial,grid,wgs,repeats);
        widths<0,8>(trial,grid,wgs,repeats); widths<1,8>(trial,grid,wgs,repeats); widths<2,8>(trial,grid,wgs,repeats);
    }
}
