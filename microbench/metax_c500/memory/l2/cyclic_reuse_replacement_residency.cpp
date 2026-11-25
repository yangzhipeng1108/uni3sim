/*
 * MetaX C500 cyclic-reuse L2 replacement residency benchmark.
 *
 * A faithful replay of the production grouped-MoE reference stream through a
 * fully associative LRU at this target's L2 size predicts about twice the
 * operand reuse the hardware realises, and that gap survives every knob the
 * replay has: the capacity (8 MiB down to 1 MiB moves it by at most 0.07),
 * the concurrency width, the co-resident phase alignment, and the operand
 * double-read phase.  What has never been measured on this target is how far
 * the real L2 falls short of an ideal LRU in the first place -- the 910B has
 * such a measurement (shared_l2_replacement_capacity_cann_8_5_1.json), this
 * one does not, and its `l2_replacement_capacity_bytes` is a fallback to the
 * nominal size with no evidence behind it.
 *
 * This probe measures exactly that.  Every worker cyclically re-reads a
 * private working set of `bytes_per_worker`, `passes` times, in address
 * order.  Under a fully associative LRU the answer is a step function: a
 * total footprint at or below the replacement capacity costs one compulsory
 * pass, anything above costs every pass.  The measured curve's position gives
 * the replacement capacity and its shape gives the realised hit rate against
 * that ideal.
 *
 * It carries no routing, no expert blocking, no tiling and no MMA; it is a
 * cyclic re-read, which is the one mechanism in question.
 *
 * Traffic counters only.  No duration here may price a Forward path, and the
 * footprint-to-traffic relation must not be fitted or interpolated.
 *
 * Build inside the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -o cyclic_reuse_replacement_residency \
 *       cyclic_reuse_replacement_residency.cpp
 */

#include <mc_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do {                                                   \
    mcError_t error = call;                                                \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,             \
                     mcGetErrorString(error));                             \
        std::exit(1);                                                      \
    }                                                                      \
} while (0)

static const int kBlockThreads = 256;
static const int kPasses = 4;

__global__ __launch_bounds__(kBlockThreads) void cyclic_reuse(
    const uint4* __restrict__ src,
    uint32_t* __restrict__ out,
    int vectors_per_worker
) {
    const int worker = blockIdx.x;
    const int lane = threadIdx.x;
    const size_t base = (size_t)worker * vectors_per_worker;
    uint32_t checksum = 0u;
    for (int pass = 0; pass < kPasses; pass++) {
        for (int i = lane; i < vectors_per_worker; i += kBlockThreads) {
            const uint4 value = src[base + i];
            checksum += value.x ^ value.w;
        }
    }
    out[worker] = checksum;
}

int main() {
    /* Total footprint = workers * vectors_per_worker * 16 B.  Sweep it
     * across the 8 MiB nominal L2 with a fixed worker count. */
    const int workers = 104;
    const int footprint_kib[] = {
        512, 1024, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 32768};
    const int points = (int)(sizeof(footprint_kib) / sizeof(int));

    std::vector<int> vectors(points);
    size_t max_vectors = 0;
    for (int i = 0; i < points; i++) {
        /* Round down to a whole number of 16 B vectors per worker. */
        vectors[i] = (int)(((size_t)footprint_kib[i] * 1024)
                           / ((size_t)workers * 16));
        if (vectors[i] < 1) vectors[i] = 1;
        max_vectors = std::max(max_vectors,
                               (size_t)workers * vectors[i]);
    }

    uint4* src = nullptr;
    uint32_t* out = nullptr;
    CHECK(mcMalloc(&src, max_vectors * sizeof(uint4)));
    CHECK(mcMalloc(&out, (size_t)workers * sizeof(uint32_t)));
    CHECK(mcMemset(src, 0x5a, max_vectors * sizeof(uint4)));
    CHECK(mcMemset(out, 0, (size_t)workers * sizeof(uint32_t)));
    CHECK(mcDeviceSynchronize());

    std::printf("{\"schema\":\"cyclic_reuse_replacement_residency_v1\",");
    std::printf("\"workers\":%d,\"passes\":%d,\"block_threads\":%d,",
                workers, kPasses, kBlockThreads);
    std::printf("\"ordinals\":[");
    for (int i = 0; i < points; i++) {
        const double footprint = (double)workers * vectors[i] * 16.0;
        if (i != 0) std::printf(",");
        std::printf("{\"ordinal\":%d,\"footprint_bytes\":%.0f,"
                    "\"vectors_per_worker\":%d,\"logical_read_bytes\":%.0f}",
                    i, footprint, vectors[i], footprint * kPasses);
    }
    std::printf("]}\n");
    std::fflush(stdout);

    CHECK(mcProfilerStart());
    for (int i = 0; i < points; i++) {
        cyclic_reuse<<<workers, kBlockThreads>>>(src, out, vectors[i]);
        CHECK(mcGetLastError());
        CHECK(mcDeviceSynchronize());
    }
    CHECK(mcProfilerStop());

    CHECK(mcFree(src));
    CHECK(mcFree(out));
    return 0;
}
