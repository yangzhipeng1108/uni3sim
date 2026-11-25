// Measure A800 CTA dispatch cost against CTA footprint and CTA duration.
//
// A800's launch model bills cta_round_ns per full-chip round, so CTA count
// inside a wave is free.  A torch probe already showed that is wrong for
// 128-thread CTAs, but torch picks the block size, so it cannot reach the
// 1024-thread geometry NORM launches.  This probe controls grid, block,
// dynamic shared memory and per-CTA duration independently.
//
// Per-CTA work is a dependent FMA chain, so its duration is set by the chain
// length alone and does not scale with block size or grid: what changes
// between points is only the geometry under test.  Each CTA also performs one
// store so it is not an empty CTA.
//
// Timing uses a CUDA-graph paired differential: a graph holding 2N launches
// against one holding N, divided by N.  That removes graph launch and event
// overhead, leaving per-launch time; a plain back-to-back stream loop cannot
// resolve this because the per-launch host gap is microseconds.
//
// The probe is synthetic.  No production operator, kernel or Serving duration
// is involved.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::fprintf(stderr, "cuda error %s at %d\n", cudaGetErrorString(e), __LINE__); \
  std::exit(1); } } while (0)

extern "C" __global__ void spin_cta(float *sink, int chain) {
  extern __shared__ float scratch[];
  if (threadIdx.x == 0) scratch[0] = 1.0f;
  __syncthreads();
  float acc = scratch[0] + static_cast<float>(threadIdx.x);
  for (int i = 0; i < chain; ++i) acc = __fmaf_rn(acc, 1.000001f, 1.0f);
  if (acc == -1.0f) sink[blockIdx.x] = acc;   // never taken; keeps the chain
}

static float graph_per_launch_ns(int grid, int block, size_t smem, int chain,
                                 float *sink, int launches, int replays) {
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));
  cudaGraph_t graph; cudaGraphExec_t exec;
  CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < launches; ++i)
    spin_cta<<<grid, block, smem, stream>>>(sink, chain);
  CHECK(cudaStreamEndCapture(stream, &graph));
  CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  for (int i = 0; i < 3; ++i) CHECK(cudaGraphLaunch(exec, stream));
  CHECK(cudaStreamSynchronize(stream));
  cudaEvent_t t0, t1;
  CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
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
  float *sink = nullptr; CHECK(cudaMalloc(&sink, 4096 * sizeof(float)));
  const int grids[]  = {1, 2, 4, 8, 16, 32, 64, 108, 216};
  const int blocks[] = {128, 256, 512, 1024};
  const size_t smems[] = {0, 16384};
  const int chains[] = {64, 1024, 16384, 262144};
  std::printf("grid,block,smem,chain,per_launch_ns\n");
  for (size_t bi = 0; bi < sizeof(blocks)/sizeof(int); ++bi)
   for (size_t si = 0; si < sizeof(smems)/sizeof(size_t); ++si)
    for (size_t ci = 0; ci < sizeof(chains)/sizeof(int); ++ci)
     for (size_t gi = 0; gi < sizeof(grids)/sizeof(int); ++gi) {
       int grid = grids[gi], block = blocks[bi], chain = chains[ci];
       size_t smem = smems[si];
       float hi = graph_per_launch_ns(grid, block, smem, chain, sink, 64, 20);
       float lo = graph_per_launch_ns(grid, block, smem, chain, sink, 32, 20);
       float per = (hi * 64 - lo * 32) / 32.0f;
       std::printf("%d,%d,%zu,%d,%.2f\n", grid, block, smem, chain, per);
       std::fflush(stdout);
     }
  CHECK(cudaFree(sink));
  return 0;
}
