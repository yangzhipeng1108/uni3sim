/*
 * L2 streaming read-BW vs access column-width microbenchmark (P58, controlled).
 *
 * Uses a single hand-written streaming kernel and only varies the *access width*
 * (how many distinct columns of an L2-resident matrix each wave touches),
 * which is the physical proxy for the model's nm_ratio: wide-N output tiles
 * make CTAs touch more distinct N-columns of B, spreading L2 requests across
 * more partitions.
 *
 * Buffer is a [ROWS x COLS] fp32 matrix sized to stay fully L2-resident
 * (< 40 MB). All CTAs repeatedly stream-read the whole buffer column-major in
 * tiles of WIDTH columns, so the steady-state access set is L2 (hit rate
 * ~100%) and we measure achieved L2 read BW vs WIDTH directly.
 *
 * The kernel is launched once per WIDTH; NCU profiles each. WIDTH sweeps
 * powers of two so the column footprint scans narrow -> wide.
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -o l2_bw_width l2_bw_width.cu
 * Profile:
 *   ncu --kernel-name regex:"stream_cols" \
 *       --metrics lts__t_bytes.sum.per_second,lts__t_sector_hit_rate.pct,\
 *  sm__cycles_elapsed.avg.per_second --csv ./l2_bw_width
 */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1);} } while (0)

// Each block streams the whole L2-resident buffer, but restricts its column
// accesses to a contiguous window of `width` columns starting at a
// block-dependent offset, looping over all row-tiles. This makes the number
// of distinct L2 partitions touched scale with `width`.
__global__ void stream_cols(const float* __restrict__ buf,
                            float* __restrict__ out,
                            int rows, int cols, int width, int iters) {
    // Vectorized float4 loads + per-thread accumulator unrolling to expose
    // enough memory-level parallelism to saturate L2 read BW (the scalar
    // version was latency-bound and measured ILP, not BW).
    int w4 = width >> 2;                       // width in float4 units
    int c4 = cols >> 2;
    int col0_4 = ((blockIdx.x * width) % cols) >> 2;
    const float4* buf4 = reinterpret_cast<const float4*>(buf);
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int it = 0; it < iters; it++) {
        for (int r = 0; r < rows; r++) {
            const float4* rowp = buf4 + (long)r * c4 + col0_4;
            for (int c = threadIdx.x; c < w4; c += blockDim.x) {
                float4 v = rowp[c];
                acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
            }
        }
    }
    if (threadIdx.x == 0) out[blockIdx.x] = acc.x + acc.y + acc.z + acc.w;
}

int main() {
    const int ROWS = 2048;
    const int COLS = 4096;                       // 2048*4096*4 = 32 MB < 40 MB L2
    const long N = (long)ROWS * COLS;
    const int N_BLOCKS = 108 * 2;                // ~2 waves over 108 SMs
    const int BLOCK = 256;

    const int widths[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    const int NW = sizeof(widths) / sizeof(widths[0]);

    float *d_buf, *d_out;
    CHECK_CUDA(cudaMalloc(&d_buf, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, N_BLOCKS * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_buf, 0, N * sizeof(float)));

    for (int w = 0; w < NW; w++) {
        int width = widths[w];
        // Hold total bytes touched roughly constant: more iters for narrow width.
        int iters = 4096 / width; if (iters < 1) iters = 1;

        // warmup (untimed) to load L2
        stream_cols<<<N_BLOCKS, BLOCK>>>(d_buf, d_out, ROWS, COLS, width, 1);
        CHECK_CUDA(cudaDeviceSynchronize());

        printf("WIDTH width=%d iters=%d\n", width, iters);
        fflush(stdout);

        stream_cols<<<N_BLOCKS, BLOCK>>>(d_buf, d_out, ROWS, COLS, width, iters);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaFree(d_out));
    return 0;
}
