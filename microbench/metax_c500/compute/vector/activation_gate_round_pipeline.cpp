/*
 * MetaX C500 activation-gate round-pipeline mechanism benchmark.
 *
 * STATUS: the deployed activation cannot be priced from this probe, and the
 * reason is recorded here so the next attempt does not repeat it.  At the
 * exact deployed geometry -- 280 rows, 96 threads, a 768-element output half,
 * one round per thread -- this probe costs 5,177 ns while the same kernel
 * measures 12,100 ns in the retained serving step.  Four candidate causes for
 * that 2.3x were measured and rejected:
 *
 *   - host launch-submission jitter: timing moved to paired differential
 *     events around a graph replay, which cut fresh-capture spread from 1.64x
 *     to 1.004x median and left the level unchanged;
 *   - clock ramp: a full-load preamble removed a bimodal small-cell result
 *     (the same cell at 6.3 or 13.1 us) and left the level unchanged;
 *   - L2 residency: rotating each launch over its own slab until the rotation
 *     exceeds L2 four times over moved the deployed geometry from 4,168 to
 *     4,913 ns, 18%, not 2.3x;
 *   - activation precision: the deployed silu_kernel uses full-precision
 *     expf rather than the fast intrinsic, which is worth 5%.
 *
 * The residual is therefore not a property of this kernel that an isolated
 * launch can reach.  Whatever it is lives in the deployed execution context
 * -- a 1,173-kernel captured graph on a rank that is also servicing
 * collectives -- and pricing the kernel from an isolated probe would put that
 * context's cost into a per-kernel coefficient.  No coefficient from this
 * probe is eligible for Forward.
 *
 * The sweep body, the cold rotation, the graph timing and the clock preamble
 * are all reusable and correct; what is missing is a measurement that
 * reproduces the deployed graph's structure.
 *
 * Build in the prescribed C500 profiler container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch=native \
 *       --maca-path=/opt/maca -O3 -std=c++17 \
 *       activation_gate_round_pipeline.cpp -o activation_gate_round_pipeline
 *
 * Run (idle device, one GPU):
 *   MACA_VISIBLE_DEVICES=3 ./activation_gate_round_pipeline
 */

#include <mc_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do {                                                   \
    const mcError_t error = (call);                                        \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",                  \
                     __FILE__, __LINE__, mcGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

namespace {

// csrc/libtorch_stable/cuda_vec_utils.cuh VecTraits<false>: a 16-byte vector,
// which for BF16 is eight scalars carried as four packed pairs.
constexpr int kVectorBytes = 16;
constexpr int kScalarsPerVector = kVectorBytes / 2;
constexpr int kTrials = 9;
constexpr int kRepetitions = 100;
constexpr int kWarmups = 10;
// The deployed kernel reads a buffer the preceding grouped GEMM just wrote,
// after that GEMM streamed hundreds of megabytes of expert weights through
// L2.  A probe that replays one slab would measure an L2-resident input
// instead.  Rotate each launch over its own slab until the rotation exceeds
// L2 several times over.
constexpr size_t kL2Bytes = 8ull << 20;
constexpr size_t kColdWorkingSetBytes = 4 * kL2Bytes;

struct alignas(16) Vec16 { uint32_t w[4]; };

// BF16 carried as raw bits, exactly as the source's packed pairs are: the
// probe must not depend on a toolchain BF16 header that this image lacks.
__device__ __forceinline__ float bf16_to_float(uint32_t bits) {
    const uint32_t widened = bits << 16;
    return __uint_as_float(widened);
}

__device__ __forceinline__ uint32_t float_to_bf16(float value) {
    const uint32_t bits = __float_as_uint(value);
    return (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
}

__device__ __forceinline__ float silu(float value) {
    // The deployed silu_kernel uses full-precision expf, not the fast
    // intrinsic; the two differ by an order of magnitude in issued work.
    return value / (1.0f + expf(-value));
}

// Mirrors act_and_mul_kernel's vector branch: x_ptr and y_ptr are the two
// halves of one row, and each thread walks the row in blockDim strides.
__global__ void activation_gate_kernel(
        uint16_t* __restrict__ out, const uint16_t* __restrict__ input,
        const int d, const int num_vecs) {
    const uint16_t* x_ptr = input + (size_t)blockIdx.x * 2 * d;
    const uint16_t* y_ptr = x_ptr + d;
    uint16_t* out_ptr = out + (size_t)blockIdx.x * d;
    const Vec16* x_vec = reinterpret_cast<const Vec16*>(x_ptr);
    const Vec16* y_vec = reinterpret_cast<const Vec16*>(y_ptr);
    Vec16* out_vec = reinterpret_cast<Vec16*>(out_ptr);
    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
        Vec16 x = x_vec[i];
        Vec16 y = y_vec[i];
        Vec16 r;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float x_lo = bf16_to_float(x.w[j] & 0xffffu);
            const float x_hi = bf16_to_float(x.w[j] >> 16);
            const float y_lo = bf16_to_float(y.w[j] & 0xffffu);
            const float y_hi = bf16_to_float(y.w[j] >> 16);
            const uint32_t r_lo = float_to_bf16(silu(x_lo) * y_lo);
            const uint32_t r_hi = float_to_bf16(silu(x_hi) * y_hi);
            r.w[j] = (r_hi << 16) | (r_lo & 0xffffu);
        }
        out_vec[i] = r;
    }
}

__global__ void empty_kernel(uint16_t* out) { if (false) out[0] = out[0]; }

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

// Paired differential events around a graph replay.  The deployed runtime
// issues these kernels from a captured Full-Graph, and at four to fifty
// microseconds per launch a host-submitted stream measures its own submission
// jitter rather than the kernel: an empty kernel alone moved between 3,809
// and 9,549 ns that way.  Capturing kRepetitions launches into one graph and
// timing a replay removes host submission from the window entirely.
double time_graph_replay(bool dependent, int rows, int threads, int d,
                         int num_vecs, uint16_t* out, const uint16_t* input,
                         int slabs, size_t in_slab_elems,
                         size_t out_slab_elems) {
    mcStream_t stream;
    CHECK(mcStreamCreate(&stream));
    CHECK(mcStreamBeginCapture(stream, mcStreamCaptureModeGlobal));
    for (int r = 0; r < kRepetitions; r++) {
        const int slab = r % slabs;
        uint16_t* out_slab = out + (size_t)slab * out_slab_elems;
        const uint16_t* in_slab = input + (size_t)slab * in_slab_elems;
        if (dependent) {
            activation_gate_kernel<<<rows, threads, 0, stream>>>(
                out_slab, in_slab, d, num_vecs);
        } else {
            empty_kernel<<<rows, threads, 0, stream>>>(out_slab);
        }
    }
    mcGraph_t graph;
    CHECK(mcStreamEndCapture(stream, &graph));
    mcGraphExec_t graph_exec;
    CHECK(mcGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    mcEvent_t start, stop;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&stop));
    for (int w = 0; w < kWarmups; w++) {
        CHECK(mcGraphLaunch(graph_exec, stream));
    }
    CHECK(mcStreamSynchronize(stream));

    std::vector<double> samples;
    for (int trial = 0; trial < kTrials; trial++) {
        CHECK(mcEventRecord(start, stream));
        CHECK(mcGraphLaunch(graph_exec, stream));
        CHECK(mcEventRecord(stop, stream));
        CHECK(mcEventSynchronize(stop));
        float ms = 0.0f;
        CHECK(mcEventElapsedTime(&ms, start, stop));
        samples.push_back((double)ms * 1e6 / kRepetitions);
    }
    CHECK(mcEventDestroy(start));
    CHECK(mcEventDestroy(stop));
    CHECK(mcGraphExecDestroy(graph_exec));
    CHECK(mcGraphDestroy(graph));
    CHECK(mcStreamDestroy(stream));
    return median(samples);
}

// The device idles between campaigns and its first timed cells would
// otherwise be measured while the clock is still ramping, which shows up as a
// bimodal small-cell result.  Hold the device busy before any timing.
void ramp_clock(uint16_t* out, const uint16_t* input) {
    for (int i = 0; i < 2000; i++) {
        activation_gate_kernel<<<1664, 256>>>(out, input, 5120, 640);
    }
    CHECK(mcDeviceSynchronize());
}

}  // namespace

int main(int argc, char** argv) {
    // One cell per process, so that no cell inherits another's device state.
    // The driver supplies the cell and randomizes the order across captures.
    if (argc != 4) {
        std::fprintf(stderr,
                     "usage: %s <rows> <threads> <output_half_d>\n", argv[0]);
        return 2;
    }
    const int rows = std::atoi(argv[1]);
    const int threads = std::atoi(argv[2]);
    const int d = std::atoi(argv[3]);
    if (rows <= 0 || threads <= 0 || d <= 0 || d % kScalarsPerVector) {
        std::fprintf(stderr, "invalid cell\n");
        return 2;
    }
    const int num_vecs = d / kScalarsPerVector;
    if (threads > num_vecs) {
        std::fprintf(stderr, "threads exceed the row's vector count\n");
        return 2;
    }
    const int rounds = (num_vecs + threads - 1) / threads;

    const size_t in_elems = (size_t)rows * 2 * d;
    const size_t out_elems = (size_t)rows * d;
    const size_t footprint = (in_elems + out_elems) * sizeof(uint16_t);
    int slabs = (int)((kColdWorkingSetBytes + footprint - 1) / footprint);
    if (slabs < 1) slabs = 1;
    if (slabs > kRepetitions) slabs = kRepetitions;
    uint16_t *input = nullptr, *out = nullptr;
    CHECK(mcMalloc(&input, in_elems * slabs * sizeof(uint16_t)));
    CHECK(mcMalloc(&out, out_elems * slabs * sizeof(uint16_t)));
    CHECK(mcMemset(input, 0x3c, in_elems * slabs * sizeof(uint16_t)));

    {
        const int ramp_rows = 1664, ramp_d = 5120;
        uint16_t *ramp_in = nullptr, *ramp_out = nullptr;
        CHECK(mcMalloc(&ramp_in,
                       (size_t)ramp_rows * 2 * ramp_d * sizeof(uint16_t)));
        CHECK(mcMalloc(&ramp_out,
                       (size_t)ramp_rows * ramp_d * sizeof(uint16_t)));
        CHECK(mcMemset(ramp_in, 0x3c,
                       (size_t)ramp_rows * 2 * ramp_d * sizeof(uint16_t)));
        ramp_clock(ramp_out, ramp_in);
        CHECK(mcFree(ramp_in));
        CHECK(mcFree(ramp_out));
    }

    const double dependent_ns = time_graph_replay(
        true, rows, threads, d, num_vecs, out, input, slabs,
        in_elems, out_elems);
    const double empty_ns = time_graph_replay(
        false, rows, threads, d, num_vecs, out, input, slabs,
        in_elems, out_elems);
    std::printf("cell,rows,threads,d,num_vecs,rounds,slabs,"
                "dependent_ns,empty_ns,increment_ns\n");
    std::printf("d%d_rows%d_thr%d,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f\n",
                d, rows, threads, rows, threads, d, num_vecs, rounds, slabs,
                dependent_ns, empty_ns, dependent_ns - empty_ns);
    CHECK(mcFree(input));
    CHECK(mcFree(out));
    return 0;
}
