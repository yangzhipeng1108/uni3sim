// Measure how a wave's CTAs contend for memory on A800.
//
// The CTA dispatch probe holds per-CTA work as a dependent FMA chain, so it
// measures dispatch alone: 222.4 ns for a full chip of 108 CTAs.  NORM's
// plateau is short by 0.4 to 1.2 microseconds, an order of magnitude more, so
// the rest is not dispatch.  The candidate is memory contention: NORM launches
// one CTA per token, each streaming one hidden-size row, and the model treats
// a single wave as duration-invariant in CTA count.
//
// This probe keeps per-CTA bytes fixed while sweeping the grid, so any rise
// beyond the measured dispatch cost is contention among concurrent CTAs.
// Per-CTA byte counts cover the row sizes NORM uses (hidden 2048 to 16384 at
// bf16, so 4 to 32 KiB read plus the same written).  Totals stay inside the
// 40 MiB L2 at every point, matching NORM's own regime, so what is measured is
// on-chip contention rather than a DRAM ceiling.
//
// Timing reuses the CUDA-graph paired differential, which removes per-launch
// host overhead; a plain stream loop cannot resolve effects of this size.
// Synthetic probe: no production operator, kernel or Serving duration.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::fprintf(stderr, "cuda error %s at %d\n", cudaGetErrorString(e), __LINE__); \
  std::exit(1); } } while (0)

// Each CTA streams its own slice: reads `vec4_per_cta` float4s and writes them
// back, which is the read-plus-write shape a norm row has.
extern "C" __global__ void stream_cta(const float4 *__restrict__ src,
                                      float4 *__restrict__ dst,
                                      int vec4_per_cta) {
  const size_t base = static_cast<size_t>(blockIdx.x) * vec4_per_cta;
  for (int i = threadIdx.x; i < vec4_per_cta; i += blockDim.x) {
    float4 v = src[base + i];
    v.x += 1.0f;
    dst[base + i] = v;
  }
}

static float graph_per_launch_ns(int grid, int block, int vec4_per_cta,
                                 const float4 *src, float4 *dst,
                                 int launches, int replays) {
  cudaStream_t stream; CHECK(cudaStreamCreate(&stream));
  cudaGraph_t graph; cudaGraphExec_t exec;
  CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < launches; ++i)
    stream_cta<<<grid, block, 0, stream>>>(src, dst, vec4_per_cta);
  CHECK(cudaStreamEndCapture(stream, &graph));
  CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  for (int i = 0; i < 3; ++i) CHECK(cudaGraphLaunch(exec, stream));
  CHECK(cudaStreamSynchronize(stream));
  cudaEvent_t t0, t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
  CHECK(cudaEventRecord(t0, stream));
  for (int i = 0; i < replays; ++i) CHECK(cudaGraphLaunch(exec, stream));
  CHECK(cudaEventRecord(t1, stream));
  CHECK(cudaStreamSynchronize(stream));
  float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, t0, t1));
  CHECK(cudaEventDestroy(t0)); CHECK(cudaEventDestroy(t1));
  CHECK(cudaGraphExecDestroy(exec)); CHECK(cudaGraphDestroy(graph));
  CHECK(cudaStreamDestroy(stream));
  return ms * 1e6f / (replays * launches);
}

int main(int argc, char **argv) {
  int device = (argc > 1) ? std::atoi(argv[1]) : 0;
  CHECK(cudaSetDevice(device));
  const int grids[]  = {1, 2, 4, 8, 16, 32, 64, 108};
  const int blocks[] = {128, 256, 1024};
  // bytes read per CTA: 4, 8, 16, 32 KiB -> the hidden 2048..16384 bf16 rows
  const int kib[]    = {4, 8, 16, 32};
  size_t max_bytes = 108ull * 32ull * 1024ull;
  float4 *src = nullptr, *dst = nullptr;
  CHECK(cudaMalloc(&src, max_bytes)); CHECK(cudaMalloc(&dst, max_bytes));
  CHECK(cudaMemset(src, 1, max_bytes));
  std::printf("grid,block,kib_per_cta,per_launch_ns\n");
  for (size_t bi = 0; bi < sizeof(blocks)/sizeof(int); ++bi)
   for (size_t ki = 0; ki < sizeof(kib)/sizeof(int); ++ki)
    for (size_t gi = 0; gi < sizeof(grids)/sizeof(int); ++gi) {
      int grid = grids[gi], block = blocks[bi];
      int vec4 = kib[ki] * 1024 / static_cast<int>(sizeof(float4));
      float hi = graph_per_launch_ns(grid, block, vec4, src, dst, 64, 20);
      float lo = graph_per_launch_ns(grid, block, vec4, src, dst, 32, 20);
      std::printf("%d,%d,%d,%.2f\n", grid, block, kib[ki],
                  (hi * 64 - lo * 32) / 32.0f);
      std::fflush(stdout);
    }
  CHECK(cudaFree(src)); CHECK(cudaFree(dst));
  return 0;
}
