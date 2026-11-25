/*
 * H200 SM-clock warm-up spinner used by the clock-gated WP5 capture helper.
 *
 * The H200 container cannot lock the SM clock (nvidia-smi -lgc is denied) and
 * the idle clock is 345 MHz.  Before every timed capture the helper runs this
 * binary on the owned GPU so the SM clock reaches the 1980-MHz boost clock.
 * It measures nothing; the spin is outside every timed probe process.
 *
 * usage: gpu_clock_warmup MILLISECONDS
 */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void spin(unsigned long long ns, float* sink) {
    unsigned long long start;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    float value = threadIdx.x;
    for (;;) {
        unsigned long long now;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        if (now - start >= ns) break;
        value = value * 1.0000001f + 0.5f;
    }
    if (value == -1.0f) sink[threadIdx.x] = value;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MILLISECONDS\n", argv[0]);
        return EXIT_FAILURE;
    }
    const long ms = std::strtol(argv[1], nullptr, 10);
    if (ms <= 0) return EXIT_FAILURE;
    float* sink = nullptr;
    if (cudaMalloc(&sink, 1024 * sizeof(float)) != cudaSuccess) return EXIT_FAILURE;
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    spin<<<sms, 256>>>(static_cast<unsigned long long>(ms) * 1000000ULL, sink);
    if (cudaDeviceSynchronize() != cudaSuccess) return EXIT_FAILURE;
    cudaFree(sink);
    return EXIT_SUCCESS;
}
