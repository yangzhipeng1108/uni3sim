/*
 * MetaX C500 L2 set-conflict and associativity probe.
 *
 * The shared miss core is a fully associative byte-capacity LRU: an object
 * survives whenever the distinct bytes touched since its last reference fit
 * the capacity.  A real L2 is set associative, so a working set that fits by
 * capacity still thrashes when more lines than the associativity map to one
 * set.  Which lines collide is decided by the set-index function, and a
 * grouped GEMM walks its operands at large regular strides -- a weight tile's
 * rows are one projection row apart, an activation row one K extent apart --
 * which is exactly the address pattern that concentrates on few sets.
 *
 * Sweep the number of lines a single workgroup cycles over, at a stride that
 * is varied independently.  The footprint is `lines` cache lines, kilobytes
 * at most, far inside any capacity, so a fully associative cache of this size
 * would serve every pass after the first from cache at every stride.  Whatever
 * collapse appears is set conflict, and the line count at which it appears is
 * the associativity for that stride.
 *
 * One workgroup, so nothing else on the chip displaces the working set.  The
 * counter is the downstream read, which L1 hits never reach, so the vector
 * cache cannot mask the effect.
 *
 * Traffic counters only.  No duration here may price a Forward path.
 *
 * Build inside the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -o l2_set_conflict_associativity \
 *       l2_set_conflict_associativity.cpp
 */

#include <mc_runtime.h>

#include <algorithm>
#include <cstring>
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
static const int kLineBytes = 128;          /* C500 downstream line */
static const int kVectorsPerLine = kLineBytes / 16;
static const int kPasses = 8;

/* `lines` cache lines spaced `stride_bytes` apart, cycled `kPasses` times.
 * Eight threads cover one line, so one wavefront of the workgroup walks
 * thirty-two lines and loops for more. */
__global__ __launch_bounds__(kBlockThreads) void set_conflict(
    const uint4* __restrict__ buf,
    uint32_t* __restrict__ out,
    int lines,
    int stride_vectors
) {
    const int lane = threadIdx.x;
    const int line = lane / kVectorsPerLine;
    const int slot = lane % kVectorsPerLine;
    const int lines_per_sweep = kBlockThreads / kVectorsPerLine;
    uint32_t checksum = 0u;
    for (int pass = 0; pass < kPasses; pass++) {
        for (int i = line; i < lines; i += lines_per_sweep) {
            const uint4 v = buf[(size_t)i * stride_vectors + slot];
            checksum += v.x ^ v.w;
        }
    }
    out[lane] = checksum;
}

struct Config { int lines, stride_bytes; };

int main() {
    /* mcProfiler stops emitting per-kernel reports after about fifty kernels
     * in one marker window, so the sweep is split into parts and the part is
     * selected by MIDLAYER_PROBE_PART.  Every part is self-contained and
     * carries the contiguous control, which fixes the per-launch constant
     * that has to be removed before the passes can be read. */
    const char* part_text = std::getenv("MIDLAYER_PROBE_PART");
    const int part = part_text ? std::atoi(part_text) : 0;

    std::vector<Config> configs;
    /* A contiguous control, in every part: consecutive lines cannot collide
     * on any sane set index, so this reads the compulsory pass and nothing
     * more, and it fixes the per-launch constant that has to be removed
     * before the passes can be read. */
    for (int n : {8, 16, 32, 64, 128}) {
        configs.push_back({n, kLineBytes});
    }
    if (part == 0) {
        /* Power-of-two strides, which concentrate a low-bit set index.  The
         * line count where a collapse starts would be the associativity at
         * that stride. */
        for (int stride : {4096, 16384, 65536, 262144, 1048576}) {
            for (int n : {2, 4, 6, 8, 10, 12, 16, 24, 32}) {
                configs.push_back({n, stride});
            }
        }
    } else if (part == 1) {
        /* Push the line count far past any plausible associativity at the
         * two strides that concentrate hardest.  If retention still holds,
         * either the associativity exceeds every one of these counts or the
         * set index is hashed; either way a regular stride cannot concentrate
         * on a set. */
        for (int stride : {65536, 1048576}) {
            for (int n : {48, 64, 96, 128, 192, 256, 384, 512}) {
                configs.push_back({n, stride});
            }
        }
    } else {
        /* Strides that are not powers of two, separating a stride effect
         * from a plain footprint effect.  5760 is the production activation
         * row stride, the K extent times the element size. */
        for (int stride : {5760, 196608}) {
            for (int n : {8, 16, 32, 64, 128}) {
                configs.push_back({n, stride});
            }
        }
    }

    size_t max_vectors = 0;
    for (const Config& c : configs) {
        max_vectors = std::max(
            max_vectors,
            (size_t)c.lines * (c.stride_bytes / 16) + kVectorsPerLine);
    }
    uint4* buf = nullptr;
    uint32_t* out = nullptr;
    CHECK(mcMalloc(&buf, max_vectors * sizeof(uint4)));
    CHECK(mcMalloc(&out, (size_t)kBlockThreads * sizeof(uint32_t)));
    CHECK(mcMemset(buf, 0x5a, max_vectors * sizeof(uint4)));
    CHECK(mcMemset(out, 0, (size_t)kBlockThreads * sizeof(uint32_t)));
    CHECK(mcDeviceSynchronize());

    std::printf("{\"schema\":\"l2_set_conflict_associativity_v1\",");
    std::printf("\"block_threads\":%d,\"line_bytes\":%d,\"passes\":%d,",
                kBlockThreads, kLineBytes, kPasses);
    std::printf("\"ordinals\":[");
    for (size_t i = 0; i < configs.size(); i++) {
        const Config& c = configs[i];
        if (i != 0) std::printf(",");
        std::printf("{\"ordinal\":%zu,\"lines\":%d,\"stride_bytes\":%d,"
                    "\"footprint_bytes\":%d,\"compulsory_bytes\":%d,"
                    "\"logical_bytes\":%d}",
                    i, c.lines, c.stride_bytes,
                    c.lines * kLineBytes, c.lines * kLineBytes,
                    c.lines * kLineBytes * kPasses);
    }
    std::printf("]}\n");
    std::fflush(stdout);

    CHECK(mcProfilerStart());
    for (const Config& c : configs) {
        set_conflict<<<1, kBlockThreads>>>(
            buf, out, c.lines, c.stride_bytes / 16);
        CHECK(mcGetLastError());
        CHECK(mcDeviceSynchronize());
    }
    CHECK(mcProfilerStop());

    CHECK(mcFree(buf));
    CHECK(mcFree(out));
    return 0;
}
