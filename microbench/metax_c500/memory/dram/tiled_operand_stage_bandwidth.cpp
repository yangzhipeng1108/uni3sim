/* MetaX C500 operand bandwidth of a shared-memory-staged tiled stream.
 *
 * Companion to thread_parallel_stream_bandwidth.cpp, which measures a straight
 * contiguous stream and finds achieved bandwidth proportional to a launch's
 * total thread count at 0.0704 GB/s per thread until it saturates near
 * 1,432 GB/s.  A decode GEMM does not reach that: its QKV and output
 * projections sustain 0.0286 and 0.0289 GB/s per thread at 20,480 and 16,384
 * threads, both 0.41x the stream's per-thread rate.  The launch geometry is
 * not the difference -- the model already emits the deployed grid and block --
 * so the difference has to be in how the operand is moved.
 *
 * This probe isolates that.  Same bytes, same coalesced global access, same
 * grid and block knobs; the only change is that each round stages its tile
 * through shared memory and crosses a barrier before the tile is consumed,
 * which is the structural difference between a tiled operand pipeline and a
 * copy.  PROBE_STAGE selects it: 0 consumes straight from global registers,
 * 1 stages through shared memory with one barrier per round.  Running both at
 * identical geometry gives the derate as a ratio of two measurements taken the
 * same way, so the launch cost and the DRAM path cancel.
 *
 * It is a memory-path mechanism: no matrix instruction, no operator, no
 * production shape, no production duration.  The swept K/N extents are powers
 * of two chosen to bracket the deployed geometry without reproducing it.
 *
 * Timing is back-to-back launches bounded by MACA events, with each launch
 * moving hundreds of MiB so the ~6.5 us eager dispatch this deployment charges
 * per launch stays under one percent of the measured span.
 *
 * Compile (MetaX C500 / MACA):
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -o tiled_operand_stage_bandwidth \
 *       tiled_operand_stage_bandwidth.cpp
 *
 * Run:
 *   ./tiled_operand_stage_bandwidth <device> <mib> <replays>
 */
#include <mc_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef PROBE_TILE_BYTES
#define PROBE_TILE_BYTES 8192
#endif

typedef uint4 vec_t;
constexpr int kTileBytes = PROBE_TILE_BYTES;
constexpr int kTileVecs = kTileBytes / sizeof(vec_t);

extern "C" __global__ void stream_direct(const vec_t* __restrict__ src,
                                         unsigned int* __restrict__ sink,
                                         long long n) {
    const long long stride =
        static_cast<long long>(gridDim.x) * blockDim.x;
    unsigned int acc = 0u;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x
                       + threadIdx.x;
         i < n; i += stride) {
        vec_t v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xffffffffu) sink[blockIdx.x] = acc;
}

extern "C" __global__ void stream_staged(const vec_t* __restrict__ src,
                                         unsigned int* __restrict__ sink,
                                         long long n) {
    __shared__ vec_t tile[kTileVecs];
    const long long stride =
        static_cast<long long>(gridDim.x) * blockDim.x;
    unsigned int acc = 0u;
    for (long long base = static_cast<long long>(blockIdx.x) * blockDim.x;
         base < n; base += stride) {
        // One round: coalesced global load into shared, barrier, then consume.
        for (int t = threadIdx.x; t < kTileVecs; t += blockDim.x) {
            const long long g = base + t;
            if (g < n) tile[t] = src[g];
        }
        __syncthreads();
        for (int t = threadIdx.x; t < kTileVecs; t += blockDim.x) {
            vec_t v = tile[t];
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
        __syncthreads();
    }
    if (acc == 0xffffffffu) sink[blockIdx.x] = acc;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("usage: %s device mib replays\n", argv[0]);
        return 2;
    }
    const int device = std::atoi(argv[1]);
    const long long mib = std::atoll(argv[2]);
    const int replays = std::atoi(argv[3]);
    if (mib <= 0 || replays <= 0) return 2;
    if (mcSetDevice(device) != mcSuccess) return 3;

    const long long bytes = mib << 20;
    const long long n = bytes / static_cast<long long>(sizeof(vec_t));
    vec_t* src = nullptr;
    unsigned int* sink = nullptr;
    if (mcMalloc(&src, bytes) != mcSuccess) return 3;
    if (mcMalloc(&sink, 4096 * sizeof(*sink)) != mcSuccess) return 3;
    mcMemset(src, 1, bytes);

    const int grids[] = {64, 80, 104, 208, 416};
    const int blocks[] = {128, 256, 512};
    std::printf("mib=%lld replays=%d tile_bytes=%d read_bytes_per_launch=%lld\n",
                mib, replays, kTileBytes, bytes);
    std::printf("%-6s %-6s %-9s %-10s %-12s %-12s %-9s\n",
                "grid", "block", "threads", "mode", "us", "GB_s",
                "GB_s_thread");
    for (int gi = 0; gi < 5; ++gi) {
        for (int bi = 0; bi < 3; ++bi) {
            const int grid = grids[gi];
            const int block = blocks[bi];
            for (int mode = 0; mode < 2; ++mode) {
                for (int i = 0; i < 3; ++i) {
                    if (mode) stream_staged<<<grid, block>>>(src, sink, n);
                    else stream_direct<<<grid, block>>>(src, sink, n);
                }
                if (mcDeviceSynchronize() != mcSuccess) return 4;
                mcEvent_t a, b;
                mcEventCreate(&a);
                mcEventCreate(&b);
                mcEventRecord(a, 0);
                for (int i = 0; i < replays; ++i) {
                    if (mode) stream_staged<<<grid, block>>>(src, sink, n);
                    else stream_direct<<<grid, block>>>(src, sink, n);
                }
                mcEventRecord(b, 0);
                if (mcDeviceSynchronize() != mcSuccess) return 4;
                float ms = 0.0f;
                mcEventElapsedTime(&ms, a, b);
                mcEventDestroy(a);
                mcEventDestroy(b);
                const double us = ms * 1000.0 / replays;
                const double gbs =
                    static_cast<double>(bytes) / (us * 1e-6) / 1e9;
                std::printf("%-6d %-6d %-9lld %-10s %-12.3f %-12.2f %-9.5f\n",
                            grid, block,
                            static_cast<long long>(grid) * block,
                            mode ? "staged" : "direct", us, gbs,
                            gbs / (static_cast<double>(grid) * block));
                std::fflush(stdout);
            }
        }
    }
    mcFree(src);
    mcFree(sink);
    return 0;
}
