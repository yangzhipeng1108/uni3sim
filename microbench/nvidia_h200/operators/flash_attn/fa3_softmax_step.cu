// H200 (sm_90) FlashAttention-3 online-softmax step service probe.
//
// Mechanism: the register-resident softmax work one MMA warpgroup performs
// per KV block between "S accumulator ready" and "P ready" in the pinned
// FA3 source (vllm-flash-attention 803020a8, hopper/softmax.h):
//   max_get_scale<Is_first=false>: thread-local max over the fragment rows,
//     quad (4-lane) shuffle all-reduce, exp2f of the running-max delta,
//     row_sum rescale;
//   online_softmax: exp2f(s * scale - max_scaled) for every score value
//     (FFMA + MUFU.EX2) and a thread-local row sum;
//   convert_type_out: FP32 -> BF16 pack of P;
//   rescale_o: multiply the O accumulator fragment by the row scales.
// A WGMMA m64nN accumulator gives every thread 2 rows x N/4 values, so the
// fragment is ROWS=2 x VALUES=N/4 (prefill QK n128 -> 32, decode QK n64 ->
// 16) and the O fragment is 2 x DV/4 values per warpgroup (prefill dv 128
// -> 32 per row; decode dv 256 per warpgroup -> 64 per row).
// Each step's scores depend on the previous step's P and row sum so steps
// cannot overlap; mode 0 runs only that dependency update (baseline).
//
// usage: fa3_softmax_step MODE VALUES OVALUES ACTIVE_WG GRID REPS
//   MODE 0 = baseline dependency update, 1 = full softmax step
// prints: mode,values,ovalues,active_wg,grid,reps,cta,wg,cycles
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    std::exit(2); } } while (0)

template <int MODE, int V, int VO>
__global__ void __launch_bounds__(256, 1)
probe(int reps, int active_wg, float seed, unsigned long long* out, float* sink) {
    const int wg = threadIdx.x / 128;
    float s[2][V], o[2][VO], row_max[2], row_sum[2];
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        row_max[r] = -1e30f; row_sum[r] = 0.f;
        #pragma unroll
        for (int i = 0; i < V; ++i) s[r][i] = seed * (threadIdx.x + i + r);
        #pragma unroll
        for (int i = 0; i < VO; ++i) o[r][i] = seed * (i - r);
    }
    const float scale_log2 = 0.1275f;
    uint32_t p_pack[2][V / 2];
    unsigned long long start = 0, stop = 0;
    if (wg < active_wg) {
        start = clock64();
        for (int it = 0; it < reps; ++it) {
            if constexpr (MODE == 1) {
                float scale[2];
                #pragma unroll
                for (int r = 0; r < 2; ++r) {
                    float m = row_max[r];
                    #pragma unroll
                    for (int i = 0; i < V; ++i) m = fmaxf(m, s[r][i]);
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                    scale[r] = exp2f((row_max[r] - m) * scale_log2);
                    row_sum[r] *= scale[r];
                    row_max[r] = m;
                    const float max_scaled = m * scale_log2;
                    float sum = 0.f;
                    #pragma unroll
                    for (int i = 0; i < V; ++i) { s[r][i] = exp2f(s[r][i] * scale_log2 - max_scaled); sum += s[r][i]; }
                    row_sum[r] += sum;
                    #pragma unroll
                    for (int i = 0; i < V / 2; ++i) {
                        __nv_bfloat162 packed = __floats2bfloat162_rn(s[r][2 * i], s[r][2 * i + 1]);
                        p_pack[r][i] = *reinterpret_cast<uint32_t*>(&packed);
                    }
                    #pragma unroll
                    for (int i = 0; i < VO; ++i) o[r][i] *= scale[r];
                }
            } else {
                #pragma unroll
                for (int r = 0; r < 2; ++r)
                    #pragma unroll
                    for (int i = 0; i < V / 2; ++i) p_pack[r][i] = __float_as_uint(s[r][2 * i]) ^ it;
            }
            // Next step's scores depend on this step's P (stands in for the next QK accumulator).
            #pragma unroll
            for (int r = 0; r < 2; ++r)
                #pragma unroll
                for (int i = 0; i < V; ++i)
                    s[r][i] = __uint_as_float((p_pack[r][i / 2] & 0x007fffffu) | 0x40000000u) + row_sum[r] * 1e-30f;
        }
        stop = clock64();
    }
    float acc = row_max[0] + row_sum[1] + s[0][0] + s[1][V - 1];
    #pragma unroll
    for (int r = 0; r < 2; ++r)
        #pragma unroll
        for (int i = 0; i < VO; ++i) acc += o[r][i];
    if (acc == 1234.5f) sink[threadIdx.x] = acc;
    if (wg < active_wg && threadIdx.x % 128 == 0) {
        out[(blockIdx.x * 2 + wg) * 2] = start;
        out[(blockIdx.x * 2 + wg) * 2 + 1] = stop;
    }
}

template <typename Kernel>
static void run(Kernel kernel, int mode, int v, int vo, int active_wg, int grid, int reps) {
    unsigned long long* out = nullptr; float* sink = nullptr;
    CHECK(cudaMalloc(&out, sizeof(unsigned long long) * grid * 4));
    CHECK(cudaMalloc(&sink, sizeof(float) * 256));
    CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024));
    for (int pass = 0; pass < 2; ++pass) {
        kernel<<<grid, 256, 200 * 1024>>>(reps, active_wg, 0.001f, out, sink);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    }
    std::vector<unsigned long long> host(grid * 4);
    CHECK(cudaMemcpy(host.data(), out, sizeof(unsigned long long) * grid * 4, cudaMemcpyDeviceToHost));
    std::printf("mode,values,ovalues,active_wg,grid,reps,cta,wg,cycles\n");
    for (int c = 0; c < grid; ++c)
        for (int w = 0; w < active_wg; ++w)
            std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%llu\n", mode, v, vo, active_wg, grid, reps, c, w,
                        host[(c * 2 + w) * 2 + 1] - host[(c * 2 + w) * 2]);
}

int main(int argc, char** argv) {
    if (argc != 7) { std::fprintf(stderr, "usage: %s MODE VALUES OVALUES ACTIVE_WG GRID REPS\n", argv[0]); return 1; }
    const int mode = std::atoi(argv[1]), v = std::atoi(argv[2]), vo = std::atoi(argv[3]);
    const int wg = std::atoi(argv[4]), grid = std::atoi(argv[5]), reps = std::atoi(argv[6]);
    if (wg < 1 || wg > 2 || grid < 1 || reps < 1) { std::fprintf(stderr, "bad args\n"); return 1; }
#define CASE(M, VV, VVO) if (mode == M && v == VV && vo == VVO) { run(probe<M, VV, VVO>, mode, v, vo, wg, grid, reps); return 0; }
    CASE(0, 16, 64) CASE(1, 16, 64) CASE(0, 32, 32) CASE(1, 32, 32)
    std::fprintf(stderr, "unsupported mode/values/ovalues\n");
    return 1;
}
