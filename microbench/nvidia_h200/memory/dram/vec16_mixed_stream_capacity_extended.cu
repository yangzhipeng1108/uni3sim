// Independent exact-geometry extension of the two-read/one-write stream probe.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  std::exit(1); } } while (0)

struct Cell { int grid; int block; };
constexpr Cell CELLS[] = {
    {65536, 1024}, {262144, 256}, {32768, 1024}, {262144, 192},
    {131072, 360}, {327680, 128}, {131072, 256}, {16384, 1024},
    {131072, 192}, {65536, 360}, {327680, 64}, {163840, 128},
};

__global__ void two_read_one_write(const int4* a, const int4* b, int4* out) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int4 x = __ldg(a + i);
    int4 y = __ldg(b + i);
    out[i] = make_int4(x.x ^ y.x, x.y ^ y.y, x.z ^ y.z, x.w ^ y.w);
}

int main() {
    cudaDeviceProp props;
    CHECK(cudaGetDeviceProperties(&props, 0));
    if (std::strstr(props.name, "H200") == nullptr || props.multiProcessorCount != 132) return 1;
    constexpr int trials = 7;
    constexpr size_t max_bytes = size_t(524288) * 1024 * sizeof(int4);
    int4 *a, *b, *out;
    CHECK(cudaMalloc(&a, max_bytes));
    CHECK(cudaMalloc(&b, max_bytes));
    CHECK(cudaMalloc(&out, max_bytes));
    CHECK(cudaMemset(a, 0x5a, max_bytes));
    CHECK(cudaMemset(b, 0xa5, max_bytes));
    CHECK(cudaMemset(out, 0, max_bytes));
    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    std::puts("grid,block,trial,read_bytes,write_bytes,event_ms");
    for (Cell cell : CELLS) {
        const size_t elements = size_t(cell.grid) * cell.block;
        const size_t role_bytes = elements * sizeof(int4);
        if (role_bytes <= 62914560) return 2;
        two_read_one_write<<<cell.grid, cell.block>>>(a, b, out);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        for (int trial = 0; trial < trials; ++trial) {
            CHECK(cudaEventRecord(start));
            two_read_one_write<<<cell.grid, cell.block>>>(a, b, out);
            CHECK(cudaEventRecord(end));
            CHECK(cudaEventSynchronize(end));
            CHECK(cudaGetLastError());
            float ms;
            CHECK(cudaEventElapsedTime(&ms, start, end));
            std::printf("%d,%d,%d,%zu,%zu,%.9f\n", cell.grid, cell.block, trial,
                        2 * role_bytes, role_bytes, double(ms));
        }
        int4 sample;
        CHECK(cudaMemcpy(&sample, out + elements - 1, sizeof(sample), cudaMemcpyDeviceToHost));
        if (sample.x != -1 || sample.y != -1 || sample.z != -1 || sample.w != -1) {
            std::fprintf(stderr, "mixed-stream output validation failed\n");
            return 3;
        }
    }
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
    CHECK(cudaFree(a));
    CHECK(cudaFree(b));
    CHECK(cudaFree(out));
    return 0;
}
