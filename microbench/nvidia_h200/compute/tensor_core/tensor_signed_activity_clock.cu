// Independent signed-data and sustained-activity clock control. No Attention or global operand stream.
#include "signed_activity_wgmma_helper.cuh"
#include <cmath>
constexpr int Chunks = 4;
constexpr int SharedBytes = 204800;
struct Stamp { unsigned long long c0, c1, n0, n1; };
__device__ __forceinline__ unsigned long long global_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(t) :: "memory");
    return t;
}
__host__ __device__ unsigned hash_word(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
    x *= 0x846ca68bu; return x ^ (x >> 16);
}
template<bool Mixed>
__global__ void __launch_bounds__(256, 1) signed_activity_clock(
        int activity, int scale, int repeats, int active, int conditioning, Stamp* trace, float* output) {
    extern __shared__ __align__(1024) unsigned char shared[];
    for (int i = threadIdx.x; i < SharedBytes / 2; i += blockDim.x) {
        unsigned h = hash_word(unsigned(i) ^ 0x19a47bcdu);
        unsigned bits = activity < 2 ? 0x3e80u + (h & 255)
            : 0x3c80u + (((h >> 8) & 7u) << 7) + (h & 127u);
        if (activity) bits |= ((h >> 16) & 1u) << 15;
        // Finite signed BF16 power-of-two A scaling commutes with FP32
        // accumulation when no overflow/underflow occurs;every output is checked.
        if (scale == 2 && i < 65536 / 2) bits += 0x80u;
        reinterpret_cast<unsigned short*>(shared)[i] = bits;
    }
    unsigned base = unsigned(__cvta_generic_to_shared(shared));
    uint64_t a = make_desc(base, 16, 1024);
    uint64_t b = make_desc(base + 65536, 16, 1024);
    float accum[64];
    #pragma unroll
    for (int i = 0; i < 64; ++i) accum[i] = 0.f;
    float seed = -.5f, sums[4] = {0.f, 0.f, 0.f, 0.f};
    __syncthreads();
    if (threadIdx.x / 128 >= active) return;
    // Reference cycles define only a device-conditioning duration. No timing
    // price is extracted. Each window resets D to avoid growth saturation.
    int warm_windows = conditioning ? 1 + 100663296 / (16 * 64 * repeats * active) : 0;
    #pragma unroll 1
    for (int window = 0; window < warm_windows + Chunks; ++window) {
        int chunk = window - warm_windows;
        Stamp s{};
        if (chunk >= 0) {s.n0 = global_ns(); s.c0 = clock64();}
        #pragma unroll 1
        for (int rep = 0; rep < repeats; ++rep) {
            wg_fence();
            #pragma unroll
            for (int k = 0; k < 16; ++k)
                wgmma_ss_n128_kmajor(accum, a, b, int((rep | k) != 0));
            wg_commit(); wg_wait0();
            if constexpr (Mixed) {
                #pragma unroll
                for (int i = 0; i < 64; ++i) {
                    float x, p;
                    asm volatile("fma.rn.ftz.f32 %0,%1,%2,%3;"
                        : "=f"(x) : "f"(seed), "f"(.5f), "f"(-.0001f * i));
                    asm volatile("ex2.approx.ftz.f32 %0,%1;" : "=f"(p) : "f"(x));
                    asm volatile("add.rn.ftz.f32 %0,%0,%1;" : "+f"(sums[i % 4]) : "f"(p));
                }
                seed = -__uint_as_float((__float_as_uint(sums[0]) & 0x007fffffu) | 0x3f000000u);
            }
        }
        if (chunk >= 0) {s.c1 = clock64(); s.n1 = global_ns();}
        if (chunk >= 0 && threadIdx.x % 128 == 0)
            trace[(blockIdx.x * 2 + threadIdx.x / 128) * Chunks + chunk] = s;
    }
    #pragma unroll
    for (int i = 0; i < 64; ++i)
        output[(blockIdx.x * 256 + threadIdx.x) * 64 + i] = accum[i];
    // Keep every mixed arithmetic result live, outside the clock windows.
    if constexpr (Mixed) {
        if (!std::isfinite(seed) || !std::isfinite(sums[0] + sums[1] + sums[2] + sums[3]))
            output[(blockIdx.x * 256 + threadIdx.x) * 64] = NAN;
    }
}
__global__ void validate_pair(const float* one, const float* two, unsigned* bad, int count, int active, int groups) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || (i / (128 * 64)) % 2 >= active) return;
    if (!std::isfinite(one[i]) || !std::isfinite(two[i]) ||
        __float_as_uint(2.f * one[i]) != __float_as_uint(two[i]))
        atomicAdd(bad + i / (128 * 64), 1);
    if (one[i] != 0.f) atomicAdd(bad + groups + i / (128 * 64), 1);
}
template<bool Mixed> void run(int activity, int grid, int repeats, int trial, int active, int conditioning,
        Stamp* trace, float* one, float* two, unsigned* bad) {
    auto kernel = signed_activity_clock<Mixed>;
    CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedBytes));
    std::vector<Stamp> stamps[2];
    // Rotate power-of-two control order; never select a favorable order.
    for (int j = 0; j < 2; ++j) {
        int index = (j + trial) % 2, scale = index + 1;
        kernel<<<grid, 256, SharedBytes>>>(activity, scale, repeats, active, conditioning, trace, index ? two : one);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        stamps[index].resize(grid * 2 * Chunks);
        CHECK(cudaMemcpy(stamps[index].data(), trace, stamps[index].size() * sizeof(Stamp), cudaMemcpyDeviceToHost));
    }
    CHECK(cudaMemset(bad, 0, grid * 4 * sizeof(unsigned)));
    int count = grid * 256 * 64;
    validate_pair<<<(count + 255) / 256, 256>>>(one, two, bad, count, active, grid * 2);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    std::vector<unsigned> errors(grid * 4);
    CHECK(cudaMemcpy(errors.data(), bad, errors.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
    for (int scale = 1; scale <= 2; ++scale)
        for (int cta = 0; cta < grid; ++cta) for (int wg = 0; wg < active; ++wg) {
            unsigned error = errors[cta * 2 + wg];
            if (error || errors[grid * 2 + cta * 2 + wg] == 0) { std::fprintf(stderr, "numerical scaling gate failed:%u\n", error); std::exit(3); }
            for (int chunk = 0; chunk < Chunks; ++chunk) {
                auto s = stamps[scale - 1][(cta * 2 + wg) * Chunks + chunk];
                if (!(s.c0 < s.c1 && s.n0 < s.n1)) std::exit(4);
                std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,8192,%u\n",
                    activity, int(Mixed), active, conditioning, grid, repeats, trial, scale, cta, wg, chunk,
                    s.c0, s.c1, s.n0, s.n1, error);
            }
        }
}
int main() {
    int sms; CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    if (sms != 132) return 5;
    Stamp* trace; float *one, *two; unsigned* bad;
    CHECK(cudaMalloc(&trace, 132 * 2 * Chunks * sizeof(Stamp)));
    CHECK(cudaMalloc(&one, 132 * 256 * 64 * sizeof(float)));
    CHECK(cudaMalloc(&two, 132 * 256 * 64 * sizeof(float)));
    CHECK(cudaMalloc(&bad, 132 * 4 * sizeof(unsigned)));
    std::puts("activity,mixed,active,conditioning,grid,repeats,trial,scale,cta,wg,chunk,c0,c1,n0,n1,checked_values,bad_values");
    for (int grid : {1, 132}) for (int active : {1, 2})
        for (int conditioning : {0, 1}) for (int repeats : {16, 64, 256})
        for (int trial = 0; trial < 3; ++trial) for (int offset = 0; offset < 3; ++offset) {
            int activity = (offset + trial) % 3;
            run<false>(activity, grid, repeats, trial, active, conditioning, trace, one, two, bad);
            run<true>(activity, grid, repeats, trial, active, conditioning, trace, one, two, bad);
        }
    CHECK(cudaFree(trace)); CHECK(cudaFree(one)); CHECK(cudaFree(two)); CHECK(cudaFree(bad));
}
