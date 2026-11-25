// Independent TMA transport service; reuse the existing MMA-free ring body.
// No matrix multiply, production library, shape list or operator duration.
#define main original_operand_stream_main
#include "fp8_tma_operand_stream.cu"
#undef main
#include <algorithm>
#include <chrono>

int main() {
  uint8_t *stream = nullptr, *shared = nullptr, *output = nullptr, *evict = nullptr;
  constexpr uint64_t row_bytes = 128 * 128;
  constexpr uint64_t stream_rows = 65536;
  CHECK(cudaMalloc(&stream, stream_rows * row_bytes));
  CHECK(cudaMalloc(&shared, 256 * row_bytes));
  CHECK(cudaMalloc(&output, 132 * 65536));
  CHECK(cudaMalloc(&evict, 256ull * 1024 * 1024));
  CHECK(cudaMemset(stream, 1, stream_rows * row_bytes));
  CHECK(cudaMemset(shared, 1, 256 * row_bytes));
  CHECK(cudaFuncSetAttribute(tma_stream_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 232448));
  cudaEvent_t begin, end;
  CHECK(cudaEventCreate(&begin)); CHECK(cudaEventCreate(&end));
  CUtensorMap warm_first = make_map(stream, stream_rows, row_bytes, 64);
  CUtensorMap warm_second = make_map(shared, 256, row_bytes, 16);
  auto warm_begin = std::chrono::steady_clock::now();
  do {
    tma_stream_kernel<<<132,384,41216>>>(warm_first, warm_second, output,
        4,64,16,128,0,0,256,132);
    CHECK(cudaDeviceSynchronize());
  } while (std::chrono::duration<double>(
      std::chrono::steady_clock::now()-warm_begin).count() < 2.0);
  uint64_t cursor = 0;
  // Axes describe transport geometry only, independent of evaluation shapes.
  for (int grid : {72, 128, 132})
  for (int cluster_size : {1, 2, 4, 8})
  for (int rows : {64})
  for (int depth : {4})
  for (int shared_rows : {16})
  for (int rounds : {0, 8})
  for (int cached : {0, 1}) {
    if (grid % cluster_size) continue;
    int bytes = depth * (rows + shared_rows) * 128 + 256;
    if (bytes > 232448) continue;
    CUtensorMap first = make_map(stream, stream_rows, row_bytes, rows);
    CUtensorMap second = make_map(shared, 256, row_bytes, std::max(1, shared_rows));
    std::vector<float> samples;
    constexpr int replays = 32;
    cudaStream_t work_stream;
    CHECK(cudaStreamCreate(&work_stream));
    cudaGraph_t graph;
    CHECK(cudaStreamBeginCapture(work_stream, cudaStreamCaptureModeThreadLocal));
    for (int replay = 0; replay < replays; ++replay) {
      if (cursor + uint64_t(grid * rows) > stream_rows) cursor = 0;
      int base = cached ? 0 : int(cursor);
      cudaLaunchConfig_t lc = {};
      lc.gridDim = dim3(grid, 1, 1); lc.blockDim = dim3(384, 1, 1);
      lc.dynamicSmemBytes = bytes; lc.stream = work_stream;
      cudaLaunchAttribute attribute[1];
      attribute[0].id = cudaLaunchAttributeClusterDimension;
      attribute[0].val.clusterDim.x = cluster_size;
      attribute[0].val.clusterDim.y = 1; attribute[0].val.clusterDim.z = 1;
      lc.attrs = attribute; lc.numAttrs = 1;
      CHECK(cudaLaunchKernelEx(&lc, tma_stream_kernel, first, second, output, depth,
          rows, shared_rows, rounds, base, 0, 256, grid));
      cursor += grid * rows;
    }
    CHECK(cudaStreamEndCapture(work_stream, &graph));
    cudaGraphExec_t executable;
    CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int trial = 0; trial < 19; ++trial) {
      if (!cached) CHECK(cudaMemset(evict, trial, 256ull * 1024 * 1024));
      if (cached) CHECK(cudaGraphLaunch(executable, work_stream));
      CHECK(cudaEventRecord(begin, work_stream));
      CHECK(cudaGraphLaunch(executable, work_stream));
      CHECK(cudaEventRecord(end, work_stream)); CHECK(cudaEventSynchronize(end));
      float ms; CHECK(cudaEventElapsedTime(&ms, begin, end));
      if (trial >= 4) samples.push_back(ms * 1e6f / replays);
    }
    CHECK(cudaGraphExecDestroy(executable)); CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(work_stream));
    std::sort(samples.begin(), samples.end());
    printf("{\"grid\":%d,\"stream_rows\":%d,\"shared_rows\":%d,"
        "\"cluster_size\":%d,\"graph_nodes\":32,\"depth\":%d,\"rounds\":%d,\"same_operand_replay\":%d,"
        "\"median_ns\":%.3f,\"min_ns\":%.3f,\"max_ns\":%.3f}\n",
        grid,rows,shared_rows,cluster_size,depth,rounds,cached,samples[7],samples.front(),samples.back());
  }
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaEventDestroy(begin)); CHECK(cudaEventDestroy(end));
  CHECK(cudaFree(evict)); CHECK(cudaFree(stream)); CHECK(cudaFree(shared)); CHECK(cudaFree(output));
}
