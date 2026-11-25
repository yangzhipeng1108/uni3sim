// NVIDIA H200 stream-serialized memcpy service between kernels.
//
// Mechanism probe (no production operator or duration).  SGLang's EAGLE V2
// iteration issues small eager cudaMemcpyAsync copies (H2D from pinned and
// pageable host memory, D2H into pinned memory, D2D) on the forward stream
// between kernels.  This probe measures what one such copy adds to a stream:
// a chain of CHAIN trivial kernels with one copy between consecutive kernels,
// minus the same kernel chain without copies, divided by CHAIN; per
// direction and payload, median of TRIALS.  CUDA-event timing on a
// non-default stream; nothing else runs on the GPU.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_90 -cudart static -o stream_memcpy_service stream_memcpy_service.cu
// Usage: stream_memcpy_service <chain> <trials>
// Output CSV: direction,payload_bytes,trial,ns_per_copy

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK_CUDA(call) do {                                              \
    const cudaError_t error = (call);                                      \
    if (error != cudaSuccess) {                                            \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                     __FILE__, __LINE__, cudaGetErrorString(error));       \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

__global__ void touch(unsigned int* sink) {
    if (threadIdx.x == 0) sink[blockIdx.x] += 1U;
}

enum class Direction { kH2DPinned, kH2DPageable, kD2HPinned, kD2D };

const char* name(Direction d) {
    switch (d) {
        case Direction::kH2DPinned: return "h2d_pinned";
        case Direction::kH2DPageable: return "h2d_pageable";
        case Direction::kD2HPinned: return "d2h_pinned";
        case Direction::kD2D: return "d2d";
    }
    return "?";
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <chain> <trials>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int chain = std::atoi(argv[1]);
    const int trials = std::atoi(argv[2]);
    const size_t max_bytes = 4 << 20;
    unsigned int* sink = nullptr;
    unsigned char *dev_a = nullptr, *dev_b = nullptr, *pinned = nullptr;
    CHECK_CUDA(cudaMalloc(&sink, 4096));
    CHECK_CUDA(cudaMalloc(&dev_a, max_bytes));
    CHECK_CUDA(cudaMalloc(&dev_b, max_bytes));
    CHECK_CUDA(cudaMallocHost(&pinned, max_bytes));
    std::vector<unsigned char> pageable(max_bytes, 1);
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t begin, end;
    CHECK_CUDA(cudaEventCreate(&begin));
    CHECK_CUDA(cudaEventCreate(&end));
    auto run = [&](bool with_copy, Direction d, size_t bytes) {
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUDA(cudaEventRecord(begin, stream));
        for (int i = 0; i < chain; ++i) {
            touch<<<1, 32, 0, stream>>>(sink);
            if (!with_copy) continue;
            switch (d) {
                case Direction::kH2DPinned:
                    CHECK_CUDA(cudaMemcpyAsync(dev_a, pinned, bytes,
                                               cudaMemcpyHostToDevice, stream));
                    break;
                case Direction::kH2DPageable:
                    CHECK_CUDA(cudaMemcpyAsync(dev_a, pageable.data(), bytes,
                                               cudaMemcpyHostToDevice, stream));
                    break;
                case Direction::kD2HPinned:
                    CHECK_CUDA(cudaMemcpyAsync(pinned, dev_a, bytes,
                                               cudaMemcpyDeviceToHost, stream));
                    break;
                case Direction::kD2D:
                    CHECK_CUDA(cudaMemcpyAsync(dev_b, dev_a, bytes,
                                               cudaMemcpyDeviceToDevice, stream));
                    break;
            }
        }
        touch<<<1, 32, 0, stream>>>(sink);
        CHECK_CUDA(cudaEventRecord(end, stream));
        CHECK_CUDA(cudaEventSynchronize(end));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, begin, end));
        return static_cast<double>(ms) * 1e6;
    };
    std::printf("direction,payload_bytes,trial,ns_per_copy\n");
    for (Direction d : {Direction::kH2DPinned, Direction::kH2DPageable,
                        Direction::kD2HPinned, Direction::kD2D}) {
        for (size_t bytes : {size_t(8), size_t(256), size_t(4096),
                             size_t(65536), size_t(1) << 20, size_t(2) << 20,
                             size_t(4) << 20}) {
            run(true, d, bytes);
            run(false, d, bytes);
            for (int t = 0; t < trials; ++t) {
                const double with_copy = run(true, d, bytes);
                const double without = run(false, d, bytes);
                std::printf("%s,%zu,%d,%.1f\n", name(d), bytes, t,
                            (with_copy - without) / chain);
            }
        }
    }
    return EXIT_SUCCESS;
}
