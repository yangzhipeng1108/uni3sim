/* A800 L2 reuse across a second pass that also writes.
 *
 * A single-pass stream over distinct buffers and a two-pass stream that re-reads
 * its input while writing an output of the same size have the same unique
 * footprint, but not the same reuse distance: the writes of the second pass can
 * evict the input before it is re-read.  This probe measures the difference
 * directly.  At each footprint it times two launches over the same buffers:
 *
 *   pass_mode 1 -- read A once in a read phase, then write B in a write phase.
 *   pass_mode 0 -- the same 1:1 bytes, but each store is issued beside the load
 *                  that feeds it, so both directions are in flight together.
 *   pass_mode 2 -- read A once; then read A again and write B, interleaved.
 *
 * It reports only achieved logical bytes per second at a footprint, contains no
 * operator, no reduction, no normalisation and no production shape, and consumes
 * no production duration.  The two modes move different byte counts, which the
 * output records, so a rate comparison is a like-for-like rate comparison.
 *
 * Forward consumes nothing from this probe, and the reason is its results.  It
 * was written to test whether the plain-RMSNorm under-prediction comes from its
 * second pass re-reading an input that the first pass's writes have evicted, so
 * that a footprint-based residency fraction is too optimistic.  At the exact
 * T=1024 H=8192 row geometry the two-pass mode is 1.22 to 1.32 times FASTER per
 * byte than the one-pass mode at every footprint measured, so the re-read is an
 * L2 hit and the reuse distance is not the residual.  Sweeping the block size
 * from 128 to 1024 threads, which moves resident CTAs per multiprocessor from
 * 16 to 2, changes the rate by at most 17 percent, so co-resident CTA count is
 * not the residual either.  Kept so neither hypothesis is retried.
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -DPROBE_PASSES=2 -DPROBE_THREADS=1024 \
 *        -o l2_two_pass_reuse_distance l2_two_pass_reuse_distance.cu
 */
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_CUDA(call) do {                                                \
    const cudaError_t error = (call);                                        \
    if (error != cudaSuccess) {                                              \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                     __FILE__, __LINE__, cudaGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                             \
    }                                                                        \
} while (0)

#ifndef PROBE_PASSES
#define PROBE_PASSES 2
#endif
constexpr int kPasses = PROBE_PASSES;
#ifndef PROBE_THREADS
#define PROBE_THREADS 1024
#endif
constexpr int kThreads = PROBE_THREADS;
constexpr const char* kExpectedDevice = "NVIDIA A800-SXM4-80GB";

__device__ __forceinline__ uint4 load_cg(const uint4* address) {
    uint4 value;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(address));
    return value;
}

__device__ __forceinline__ void store_wb(uint4* address, uint4 value) {
    asm volatile("st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(address), "r"(value.x), "r"(value.y),
                   "r"(value.z), "r"(value.w)
                 : "memory");
}

/* One CTA owns one contiguous row of the buffers, exactly as a row-wise
 * two-pass kernel does, so the reuse distance between the two reads of a row is
 * the whole launch rather than a few lines. */
__global__ __launch_bounds__(kThreads)
void two_pass_row_stream(const uint4* __restrict__ input,
                         uint4* __restrict__ output,
                         long long vectors_per_row,
                         std::uint32_t* __restrict__ sink) {
    const long long base =
        static_cast<long long>(blockIdx.x) * vectors_per_row;
    unsigned int checksum = 0;
    for (long long offset = threadIdx.x; offset < vectors_per_row;
            offset += kThreads) {
        const uint4 value = load_cg(input + base + offset);
        checksum ^= value.x ^ value.y ^ value.z ^ value.w;
    }
    if (kPasses == 1) {
        for (long long offset = threadIdx.x; offset < vectors_per_row;
                offset += kThreads) {
            store_wb(output + base + offset, make_uint4(
                checksum, checksum, checksum, checksum));
        }
    } else if (kPasses == 0) {
        // Same 1:1 byte mix as pass_mode 1, but each store is issued next to
        // the load that feeds it instead of after a whole read phase, so the
        // two directions are in flight together.
        for (long long offset = threadIdx.x; offset < vectors_per_row;
                offset += kThreads) {
            const uint4 value = load_cg(input + base + offset);
            store_wb(output + base + offset, value);
        }
    } else {
        __syncthreads();
        for (long long offset = threadIdx.x; offset < vectors_per_row;
                offset += kThreads) {
            const uint4 value = load_cg(input + base + offset);
            store_wb(output + base + offset, make_uint4(
                value.x ^ checksum, value.y, value.z, value.w));
        }
    }
    if (checksum == 0xffffffffu) sink[blockIdx.x] = checksum;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr,
                     "usage: %s DEVICE TRIALS ROWS ROW_KIB\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device = std::atoi(argv[1]);
    const int trials = std::atoi(argv[2]);
    const int rows = std::atoi(argv[3]);
    const long long row_kib = std::atoll(argv[4]);
    if (device < 0 || trials < 3 || trials > 15 || rows <= 0 || row_kib <= 0) {
        return EXIT_FAILURE;
    }

    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (std::strcmp(properties.name, kExpectedDevice) != 0
            || properties.major != 8 || properties.minor != 0
            || properties.multiProcessorCount != 108) {
        std::fprintf(stderr, "unexpected A800 device contract\n");
        return EXIT_FAILURE;
    }

    const long long row_bytes = row_kib * 1024;
    const long long buffer_bytes = row_bytes * rows;
    const long long vectors_per_row = row_bytes / 16;
    if (vectors_per_row * 16 != row_bytes) return EXIT_FAILURE;
    uint4* input = nullptr;
    uint4* output = nullptr;
    std::uint32_t* sink = nullptr;
    CHECK_CUDA(cudaMalloc(&input, buffer_bytes));
    CHECK_CUDA(cudaMalloc(&output, buffer_bytes));
    CHECK_CUDA(cudaMalloc(&sink, rows * sizeof(*sink)));
    CHECK_CUDA(cudaMemset(input, 0x27, buffer_bytes));
    CHECK_CUDA(cudaMemset(output, 0, buffer_bytes));
    CHECK_CUDA(cudaMemset(sink, 0, rows * sizeof(*sink)));

    for (int warmup = 0; warmup < 3; ++warmup) {
        two_pass_row_stream<<<rows, kThreads>>>(
            input, output, vectors_per_row, sink);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    const long long read_bytes =
        buffer_bytes * ((kPasses == 0) ? 1 : kPasses);
    const long long write_bytes = buffer_bytes;
    const long long logical_bytes = read_bytes + write_bytes;
    std::printf("passes,rows,row_kib,unique_footprint_bytes,logical_bytes,"
                "trial,duration_ms\n");
    for (int trial = 0; trial < trials; ++trial) {
        cudaEvent_t start;
        cudaEvent_t stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        two_pass_row_stream<<<rows, kThreads>>>(
            input, output, vectors_per_row, sink);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        std::printf("%d,%d,%lld,%lld,%lld,%d,%.9f\n",
                    kPasses, rows, row_kib, 2 * buffer_bytes, logical_bytes,
                    trial, milliseconds);
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaEventDestroy(start));
    }
    CHECK_CUDA(cudaFree(sink));
    CHECK_CUDA(cudaFree(output));
    CHECK_CUDA(cudaFree(input));
    return EXIT_SUCCESS;
}
