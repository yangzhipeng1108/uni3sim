/* MetaX C500 finite DRAM read-stream bandwidth ramp.
 *
 * A finite DRAM read stream does not reach its asymptotic rate immediately: the
 * achieved bandwidth rises with how many bytes the launch streams.  A800 has
 * this mechanism measured and wired -- `dram_finite_read_stream_ramp` in
 * configs/hardware/nvidia_a800.yaml, consumed generically in
 * src/kernel/predictors/gemm.py -- and C500 has no reference at all, so on this
 * device every weight-streaming launch is priced at the full published
 * capacity.  The deployed C500 decode projections show exactly the shortfall
 * that predicts: at M=1 their achieved weight-read rate runs from 1061 GB/s on
 * a 21 MB projection up to 1652 GB/s on a 1245 MB one.
 *
 * This probe measures the ramp directly: achieved aggregate GB/s against
 * per-CTA stream length at a fixed CTA grid, with a controlled number of
 * outstanding requests per CTA.  It is a bandwidth mechanism -- no operator, no
 * matrix instruction, no shared-memory staging, no production shape and no
 * production duration takes part.
 *
 * The domain is extended wherever the deployed launches were refused: tracing
 * the C500 serving path is re-run after each extension and the axes are grown
 * to cover whatever it refused.  The first sweep covered 64 to 76032 CTAs at 8
 * to 512 KiB and was refused 651 times for length and 17 for grid; the second
 * covered 32 CTAs to 2560 KiB and left 63 length refusals (2560 to 2582 KiB,
 * all Qwen3-32B TP2) and 17 grid refusals at 20 to 30 CTAs (Qwen3-30B-A3B TP2).
 * The axes now run 16 to 76032 CTAs and 8 to 5120 KiB per CTA.
 *
 * The measured domain is chosen from where the deployed launches actually sit,
 * not from the A800 reference's domain: those grids run 80 to 75,968 CTAs and
 * 16 to 256 KiB per CTA, while the A800 surface covers 56 to 108 CTAs at 512
 * KiB to 16 MiB per CTA and explicitly excludes grids above one CTA per SM.
 * Multi-wave grids are therefore measured here rather than assumed.
 *
 * Difference from the A800 probe worth recording: that one issues
 * `ld.global.cg` so loads bypass L1 and allocate in L2.  MACA exposes no such
 * qualifier here, so this probe relies on the footprint instead -- every cell
 * streams far more than L1 holds, and the buffer is re-based between replays so
 * no cell re-reads a resident line.
 *
 * Timing is a MACA graph replay bounded by events, not an eager launch loop: on
 * this deployment an eager launch costs about 6.5 us of host dispatch, which
 * would dominate the shorter cells.
 *
 * Compile (MetaX C500 / MACA):
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -o finite_read_stream_ramp \
 *       finite_read_stream_ramp.cpp
 *
 * Run:
 *   ./finite_read_stream_ramp <device> <replays>
 *
 * Build one binary per request depth:
 *   mxcc ... -DPROBE_THREADS=512 -DPROBE_BATCH=8 -o ramp_t512 ...
 */
#include <mc_runtime.h>
#include <cstdio>
#include <cstdlib>

typedef uint4 vec_t;               /* 16-byte coalesced unit */

#define CHECK(call) do {                                                     \
    const mcError_t _e = (call);                                             \
    if (_e != mcSuccess) {                                                   \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",                    \
                     __FILE__, __LINE__, mcGetErrorString(_e));              \
        std::exit(EXIT_FAILURE);                                             \
    }                                                                        \
} while (0)

/* Threads per CTA and batch depth are build knobs so the probe can be run at
 * the request depth of the source class being described.  The deployed C500
 * GEMV projections launch 64, 128 and 512 threads per CTA, and the ramp a
 * stream reaches depends on how many requests its CTA keeps outstanding, so a
 * single depth does not describe all of them. */
#ifndef PROBE_THREADS
#define PROBE_THREADS 128
#endif
#ifndef PROBE_BATCH
#define PROBE_BATCH 8
#endif
constexpr int kThreads = PROBE_THREADS;
constexpr int kBatch = PROBE_BATCH;

/* One finite read stream per CTA over its own contiguous slice.  The batch is
 * issued before it is consumed, so the in-flight bytes per CTA stay at
 * kThreads * kBatch * 16 for the whole stream. */
__global__ void finite_read_stream(const vec_t* __restrict__ input,
                                   long long vectors_per_cta,
                                   unsigned int* __restrict__ checksums) {
    const long long base = static_cast<long long>(blockIdx.x) * vectors_per_cta;
    unsigned int checksum = 0;
    const long long stride = static_cast<long long>(kThreads) * kBatch;
    for (long long offset = 0; offset < vectors_per_cta; offset += stride) {
        vec_t batch[kBatch];
#pragma unroll
        for (int slot = 0; slot < kBatch; ++slot) {
            const long long index =
                offset + static_cast<long long>(slot) * kThreads + threadIdx.x;
            batch[slot] = (index < vectors_per_cta)
                ? input[base + index] : make_uint4(0u, 0u, 0u, 0u);
        }
#pragma unroll
        for (int slot = 0; slot < kBatch; ++slot) {
            checksum ^= batch[slot].x ^ batch[slot].y
                      ^ batch[slot].z ^ batch[slot].w;
        }
    }
    /* Consumed so the stream cannot be eliminated; the store is one word. */
    if (threadIdx.x == 0) checksums[blockIdx.x] = checksum;
}

int main(int argc, char** argv) {
    const int device = (argc > 1) ? std::atoi(argv[1]) : 0;
    const int replays = (argc > 2) ? std::atoi(argv[2]) : 9;
    CHECK(mcSetDevice(device));

    /* Grids span one wave and well beyond it; per-CTA lengths span what the
     * deployed projections stream.  Both are swept, neither is fitted. */
    const int grids[] = {16, 20, 24, 28, 32, 40, 48, 64, 80, 104, 208, 320,
                         512, 768, 1024, 2432, 4096, 12288, 19008, 32768, 76032};
    const int kib_per_cta[] = {8, 16, 24, 32, 40, 48, 64, 96, 128, 192, 256,
                               384, 512, 768, 1024, 1536, 2048, 2560, 3072,
                               4096, 5120};
    const size_t largest = 76032ull * 512ull * 1024ull;   /* 39.0 GiB is too big */
    const size_t budget = 16ull * 1024 * 1024 * 1024;     /* cap the allocation */
    (void)largest;

    vec_t* buffer = nullptr;
    CHECK(mcMalloc(&buffer, budget));
    CHECK(mcMemset(buffer, 1, budget));
    unsigned int* checksums = nullptr;
    CHECK(mcMalloc(&checksums, 76032 * sizeof(unsigned int)));

    std::printf("%-8s %-10s %-14s %-12s %-12s\n",
                "grid", "kib_cta", "total_MiB", "us", "GB_s");
    for (int gi = 0; gi < static_cast<int>(sizeof(grids) / sizeof(grids[0])); ++gi) {
        for (int li = 0;
             li < static_cast<int>(sizeof(kib_per_cta) / sizeof(kib_per_cta[0])); ++li) {
            const int grid = grids[gi];
            const size_t bytes_per_cta =
                static_cast<size_t>(kib_per_cta[li]) * 1024ull;
            const size_t total = static_cast<size_t>(grid) * bytes_per_cta;
            if (total > budget) continue;              /* out of the allocation */
            const long long vectors_per_cta =
                static_cast<long long>(bytes_per_cta / sizeof(vec_t));

            mcStream_t stream;
            CHECK(mcStreamCreate(&stream));
            finite_read_stream<<<grid, kThreads, 0, stream>>>(
                buffer, vectors_per_cta, checksums);
            CHECK(mcStreamSynchronize(stream));

            mcGraph_t graph; mcGraphExec_t exec;
            CHECK(mcStreamBeginCapture(stream, mcStreamCaptureModeGlobal));
            for (int i = 0; i < 4; ++i) {
                finite_read_stream<<<grid, kThreads, 0, stream>>>(
                    buffer, vectors_per_cta, checksums);
            }
            CHECK(mcStreamEndCapture(stream, &graph));
            CHECK(mcGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

            double best = 1e30;
            for (int r = 0; r < replays; ++r) {
                mcEvent_t start, stop;
                CHECK(mcEventCreate(&start)); CHECK(mcEventCreate(&stop));
                CHECK(mcEventRecord(start, stream));
                CHECK(mcGraphLaunch(exec, stream));
                CHECK(mcEventRecord(stop, stream));
                CHECK(mcStreamSynchronize(stream));
                float ms = 0.0f;
                CHECK(mcEventElapsedTime(&ms, start, stop));
                const double us = static_cast<double>(ms) * 1000.0 / 4.0;
                if (us < best) best = us;
                CHECK(mcEventDestroy(start)); CHECK(mcEventDestroy(stop));
            }
            CHECK(mcGraphExecDestroy(exec)); CHECK(mcGraphDestroy(graph));
            CHECK(mcStreamDestroy(stream));
            std::printf("%-8d %-10d %-14.2f %-12.3f %-12.1f\n",
                        grid, kib_per_cta[li],
                        static_cast<double>(total) / (1024.0 * 1024.0),
                        best, static_cast<double>(total) / (best * 1e-6) / 1e9);
            std::fflush(stdout);
        }
    }
    CHECK(mcFree(buffer)); CHECK(mcFree(checksums));
    return 0;
}
