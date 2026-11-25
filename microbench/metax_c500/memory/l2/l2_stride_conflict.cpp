/*
 * MetaX C500 L2 power-of-two stride conflict probe.
 *
 * The production-address replica (routed_stream_phase.cpp) replays the
 * canonical E8 W1 case's single streams far below their counters even in the
 * measured execution order, while the E32/E128 streams and the E8 W2 streams
 * replay within a few percent.  E8 W1 reads activation rows 4 tokens x 8 KiB
 * = 32 KiB apart and weight rows 8 KiB apart, both powers of two; the earlier
 * l2_set_conflict_associativity probe never tested 8/32/128 KiB strides with
 * more than 512 lines.
 *
 * One workgroup cycles `lines` 128-B lines spaced `stride` bytes apart, eight
 * passes.  Far inside the replacement capacity, a set-index function that
 * spreads the stride keeps every pass after the first in L2; realised passes
 * above one expose a set concentration for that stride.  Traffic counters
 * only; no duration here may price a Forward path.
 *
 * Build: /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *   --maca-path=/opt/maca -O3 -o l2_stride_conflict l2_stride_conflict.cpp
 */

#include <mc_runtime.h>

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

static const int kThreads = 256;
static const int kPasses = 8;

__global__ __launch_bounds__(kThreads) void cycle_lines(
    const uint4* __restrict__ buf, uint32_t* __restrict__ out, int lines,
    long long stride_vec
) {
    const int lane = threadIdx.x;
    const int part = lane & 7;
    uint32_t checksum = 0u;
    for (int p = 0; p < kPasses; p++) {
        for (int i = lane >> 3; i < lines; i += kThreads / 8) {
            const uint4 v = buf[(size_t)i * stride_vec + part];
            checksum += v.x ^ v.w;
        }
        __syncthreads();
    }
    out[lane] = checksum;
}

int main() {
    const long long strides[] = {8192, 32768, 131072, 46080};
    const int lines[] = {64, 256, 512, 1024, 2048, 4096};
    size_t max_bytes = 0;
    for (long long s : strides) for (int l : lines)
        if ((size_t)s * l > max_bytes) max_bytes = (size_t)s * l;
    uint4* buf = nullptr; uint32_t* out = nullptr;
    CHECK(mcMalloc(&buf, max_bytes + 4096));
    CHECK(mcMemset(buf, 0x3c, max_bytes + 4096));
    CHECK(mcMalloc(&out, kThreads * 4));
    std::printf("{\"schema\":\"l2_stride_conflict_v1\",\"passes\":%d,\"ordinals\":[", kPasses);
    int n = 0;
    for (long long s : strides) for (int l : lines)
        std::printf("%s{\"ordinal\":%d,\"stride_bytes\":%lld,\"lines\":%d}", n++ ? "," : "", n, s, l);
    std::printf("]}\n");
    std::fflush(stdout);
    CHECK(mcProfilerStart());
    for (long long s : strides) for (int l : lines) {
        cycle_lines<<<1, kThreads>>>(buf, out, l, s / 16);
        CHECK(mcGetLastError());
        CHECK(mcDeviceSynchronize());
    }
    CHECK(mcProfilerStop());
    CHECK(mcFree(buf)); CHECK(mcFree(out));
    return 0;
}
