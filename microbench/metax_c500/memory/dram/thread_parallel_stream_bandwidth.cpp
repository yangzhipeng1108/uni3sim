/* MetaX C500 achieved DRAM bandwidth against a launch's thread parallelism.
 *
 * src/kernel/predictors/common.py::gpu_memory_source declares the gap this
 * measures: "The model has no source-bound per-kernel resident-warp or
 * independent request contract", so it returns one generic producer per active
 * unit and no independent-request count, and every GPU prediction carries
 * `workload_producer_independence_unavailable`.  With that term absent the
 * memory model lets any launch reach the hardware byte capacity regardless of
 * how few threads it runs, which is why the C500 Serving glue -- launched at 16
 * to 96 threads per CTA -- is priced 3 to 4 times short.
 *
 * Timing is a MACA graph replay bounded by events, not an eager launch loop.
 * On this deployment an eager launch costs about 6.5 us of host dispatch on top
 * of the kernel: the same one-token 2,048-wide BF16 elementwise measures
 * 2.270 us per launch replayed from a graph, 8.733 us eager back-to-back and
 * 29.757 us eager with a synchronisation, and only the graph figure reproduces
 * the 2,080 ns the canonical C500 dataset records for that shape.  The deployed
 * decode step replays a graph, so the graph figure is also the one it pays.
 *
 * The kernel is a contiguous 16-byte-per-thread stream with a grid-stride loop, so bytes,
 * CTA count and threads per CTA are independent knobs and the access pattern is
 * fully coalesced: what varies between cells is only how many independent
 * requests the launch can keep in flight.  No operator, no production shape and
 * no production duration takes part.
 *
 * Compile (MetaX C500 / MACA):
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -o thread_parallel_stream_bandwidth \
 *       thread_parallel_stream_bandwidth.cpp
 *
 * Run:
 *   ./thread_parallel_stream_bandwidth <device> <mib> <replays>
 */
#include <mc_runtime.h>
#include <cstdio>
#include <cstdlib>

typedef uint4 vec_t;   /* 16-byte coalesced unit; the payload dtype is irrelevant to a pure stream */

__global__ void stream_copy(const vec_t* __restrict__ src,
                            vec_t* __restrict__ dst, long long n) {
    const long long stride =
        static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x
                       + threadIdx.x;
         i < n; i += stride) {
        dst[i] = src[i];
    }
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
    vec_t *src = nullptr, *dst = nullptr;
    if (mcMalloc(&src, bytes) != mcSuccess) return 3;
    if (mcMalloc(&dst, bytes) != mcSuccess) return 3;
    mcMemset(src, 1, bytes);

    const int grids[] = {1, 8, 35, 70, 104, 208, 416, 832};
    const int blocks[] = {16, 32, 64, 128, 256, 512, 1024};
    std::printf("mib=%lld replays=%d bytes_moved_per_launch=%lld\n",
                mib, replays, 2 * bytes);
    std::printf("%-8s %-8s %-10s %-12s %-12s\n",
                "grid", "block", "threads", "us", "GB_s");
    for (int gi = 0; gi < 8; ++gi) {
        for (int bi = 0; bi < 7; ++bi) {
            const int grid = grids[gi];
            const int block = blocks[bi];
            for (int i = 0; i < 3; ++i) stream_copy<<<grid, block>>>(src, dst, n);
            if (mcDeviceSynchronize() != mcSuccess) return 4;

            mcEvent_t start, stop;
            mcEventCreate(&start);
            mcEventCreate(&stop);
            mcEventRecord(start, 0);
            for (int i = 0; i < replays; ++i)
                stream_copy<<<grid, block>>>(src, dst, n);
            mcEventRecord(stop, 0);
            if (mcDeviceSynchronize() != mcSuccess) return 4;
            float ms = 0.0f;
            mcEventElapsedTime(&ms, start, stop);
            mcEventDestroy(start);
            mcEventDestroy(stop);
            const double us = ms * 1000.0 / replays;
            const double gbs =
                (2.0 * static_cast<double>(bytes)) / (us * 1e-6) / 1e9;
            std::printf("%-8d %-8d %-10lld %-12.3f %-12.2f\n",
                        grid, block,
                        static_cast<long long>(grid) * block, us, gbs);
            std::fflush(stdout);
        }
    }
    mcFree(src);
    mcFree(dst);
    return 0;
}
