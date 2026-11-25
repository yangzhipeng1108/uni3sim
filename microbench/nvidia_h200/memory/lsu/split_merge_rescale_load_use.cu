/*
 * H200 split-KV merge chunk: load-use interval of an LSE + partial-O pair
 * consumed by an online-softmax rescale chain.
 *
 * Mechanism, not an operator.  A split-KV attention merge (the FA2 combine,
 * flashinfer merge_states / the persistent MLA merge loop) walks the KV
 * chunks of one output row serially: load the chunk's log-sum-exp (4 B) and
 * its partial output (16 B per lane), rescale, accumulate, next chunk.  With
 * in-order issue the next chunk's loads wait for the previous chunk's
 * consumers, so the per-chunk cost is one load round trip plus the dependent
 * consumer chain.  The existing fa2_oaccum_ldg128_load_use probe measures the
 * round trip with four FMA consumers on lines the reader warmed itself; this
 * probe varies (a) who put the lines in L2 and (b) the consumer chain.
 *
 * One cooperative launch of GRID x 128 threads (128 KiB dynamic shared
 * memory, one CTA per SM).  Phase 1: every chunk k (LSE float + 2 KiB of
 * BF16 partial output) is written either by CTA 1 + k % (GRID - 1)
 * ("cross": lines other SMs just wrote, as after a split-KV grid barrier) or
 * read once by CTA 0 with ld.global.cg ("self_warm").  grid.sync().  Phase 2:
 * CTA 0 alone walks CHUNKS chunks; every statement is volatile PTX, so the
 * order load(k) -> consumers(k) -> load(k+1) is preserved.  Consumers:
 *   int_min : one xor of the loaded words (load-to-use only);
 *   rescale : m' = max(m, lse); a = ex2(m - m'); b = ex2(lse - m');
 *             8 BF16 values unpacked; acc = acc * a + v * b (FMUL + FFMA);
 *   chain   : the rescale chain on register values, no load.
 * Per-chunk interval = (cycles(CHUNKS=43) - cycles(CHUNKS=11)) / 32, clock64
 * envelope over the CTA's 128 lanes.  No production kernel, input or
 * duration is used.
 *
 * Build: nvcc -O3 -std=c++17 -arch=sm_90 -cudart static -o split_merge_rescale_load_use split_merge_rescale_load_use.cu
 * Usage: split_merge_rescale_load_use <samples>
 * Output CSV: writer,consumer,chunks,sample,cycles
 */

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;

namespace {

constexpr int kThreads = 128;
constexpr int kGrid = 132;
constexpr int kMaxChunks = 64;
constexpr int kSmem = 128 * 1024;

void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", expression,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(call) check_cuda((call), #call)

__device__ __forceinline__ float ld_f32(const float* p) {
    float v;
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint4 ld_v4(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint4 ld_v4_cg(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void rescale(float lse, uint4 o, float* m,
                                        float* acc) {
    float mn, a, b, d0, d1;
    asm volatile("max.ftz.f32 %0, %1, %2;" : "=f"(mn) : "f"(*m), "f"(lse));
    asm volatile("sub.ftz.f32 %0, %1, %2;" : "=f"(d0) : "f"(*m), "f"(mn));
    asm volatile("sub.ftz.f32 %0, %1, %2;" : "=f"(d1) : "f"(lse), "f"(mn));
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(a) : "f"(d0));
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(d1));
    const std::uint32_t w[4] = {o.x, o.y, o.z, o.w};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float lo, hi, t0, t1;
        asm volatile("shl.b32 %0, %1, 16;" : "=r"(*(std::uint32_t*)&lo)
                     : "r"(w[i]));
        asm volatile("and.b32 %0, %1, 0xffff0000;"
                     : "=r"(*(std::uint32_t*)&hi) : "r"(w[i]));
        asm volatile("mul.ftz.f32 %0, %1, %2;" : "=f"(t0) : "f"(lo), "f"(b));
        asm volatile("mul.ftz.f32 %0, %1, %2;" : "=f"(t1) : "f"(hi), "f"(b));
        asm volatile("fma.rn.ftz.f32 %0, %0, %1, %2;"
                     : "+f"(acc[2 * i]) : "f"(a), "f"(t0));
        asm volatile("fma.rn.ftz.f32 %0, %0, %1, %2;"
                     : "+f"(acc[2 * i + 1]) : "f"(a), "f"(t1));
    }
    *m = mn;
}

// consumer: 0 int_min, 1 rescale, 2 chain (no load)
__global__ void __launch_bounds__(kThreads) merge_walk(
        float* lse, uint4* part, int writer_cross, int consumer, int chunks,
        unsigned long long* cycles, float* sink) {
    extern __shared__ unsigned char guard[];
    __shared__ unsigned long long begin_ticks[kThreads];
    __shared__ unsigned long long end_ticks[kThreads];
    if (threadIdx.x == 0) guard[0] = 1;
    const int lane = threadIdx.x;
    cg::grid_group grid = cg::this_grid();
    if (writer_cross) {
        for (int k = 0; k < chunks; ++k) {
            if (blockIdx.x == 1 + k % (gridDim.x - 1)) {
                part[k * kThreads + lane] = make_uint4(
                    0x3f803f80u + k, 0x3f803f81u, 0x3f803f82u, 0x3f803f83u);
                if (lane == 0) lse[k] = 0.001f * k;
            }
        }
        __threadfence();
    } else if (blockIdx.x == 0) {
        for (int k = 0; k < chunks; ++k) {
            uint4 v = ld_v4_cg(part + k * kThreads + lane);
            asm volatile("" : : "r"(v.x), "r"(v.w) : "memory");
        }
    }
    grid.sync();
    if (blockIdx.x != 0) return;
    float m = -1e30f;
    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
    std::uint32_t x = 0;
    const unsigned long long begin = clock64();
    #pragma unroll 1
    for (int k = 0; k < chunks; ++k) {
        if (consumer == 2) {
            const uint4 o = make_uint4(0x3f803f80u + k, x, 7u, 9u);
            rescale(0.001f * k, o, &m, acc);
        } else {
            const float l = ld_f32(lse + k);
            const uint4 o = ld_v4(part + k * kThreads + lane);
            if (consumer == 0) {
                asm volatile("xor.b32 %0, %0, %1;" : "+r"(x) : "r"(o.x));
                asm volatile("xor.b32 %0, %0, %1;"
                             : "+r"(x) : "r"(__float_as_uint(l)));
            } else {
                rescale(l, o, &m, acc);
            }
        }
    }
    const unsigned long long end = clock64();
    begin_ticks[lane] = begin;
    end_ticks[lane] = end;
    float s = m + static_cast<float>(x);
    #pragma unroll
    for (int i = 0; i < 8; ++i) s += acc[i];
    sink[lane] = s;
    __syncthreads();
    if (lane == 0) {
        unsigned long long first = begin_ticks[0], last = end_ticks[0];
        for (int i = 1; i < kThreads; ++i) {
            first = min(first, begin_ticks[i]);
            last = max(last, end_ticks[i]);
        }
        *cycles = last - first;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <samples>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int samples = std::atoi(argv[1]);
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 9 || prop.multiProcessorCount != kGrid) {
        std::fprintf(stderr, "expected sm_90 with %d SMs\n", kGrid);
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaFuncSetAttribute(
        merge_walk, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    float* lse = nullptr;
    uint4* part = nullptr;
    unsigned long long* cycles = nullptr;
    float* sink = nullptr;
    CHECK_CUDA(cudaMalloc(&lse, sizeof(float) * kMaxChunks));
    CHECK_CUDA(cudaMalloc(&part, sizeof(uint4) * kThreads * kMaxChunks));
    CHECK_CUDA(cudaMalloc(&cycles, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&sink, sizeof(float) * kThreads));
    CHECK_CUDA(cudaMemset(lse, 0, sizeof(float) * kMaxChunks));
    CHECK_CUDA(cudaMemset(part, 0, sizeof(uint4) * kThreads * kMaxChunks));
    std::printf("writer,consumer,chunks,sample,cycles\n");
    const char* writers[] = {"self_warm", "cross"};
    const char* consumers[] = {"int_min", "rescale", "chain"};
    for (int writer = 0; writer < 2; ++writer) {
        for (int consumer = 0; consumer < 3; ++consumer) {
            for (int chunks : {11, 43}) {
                for (int sample = -3; sample < samples; ++sample) {
                    void* args[] = {&lse, &part, &writer, &consumer, &chunks,
                                    &cycles, &sink};
                    CHECK_CUDA(cudaLaunchCooperativeKernel(
                        reinterpret_cast<void*>(merge_walk), dim3(kGrid),
                        dim3(kThreads), args, kSmem, 0));
                    CHECK_CUDA(cudaDeviceSynchronize());
                    unsigned long long host = 0;
                    CHECK_CUDA(cudaMemcpy(&host, cycles, sizeof(host),
                                          cudaMemcpyDeviceToHost));
                    if (sample >= 0) {
                        std::printf("%s,%s,%d,%d,%llu\n", writers[writer],
                                    consumers[consumer], chunks, sample,
                                    host);
                    }
                }
            }
        }
    }
    return EXIT_SUCCESS;
}
