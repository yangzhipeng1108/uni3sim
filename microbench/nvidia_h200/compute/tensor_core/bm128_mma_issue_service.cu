// Independent BF16 MMA issue probe for one-resident-CTA SM90 service.
// It has no routed rows, weight tensor, global operand stream, or GEMM output.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <initializer_list>

namespace {

constexpr int kThreads = 256;
constexpr int kSharedBytes = 196608;
constexpr int kAccumulatorFragments = 32;
constexpr int kSamples = 7;

void check(cudaError_t result, const char* site) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", site, cudaGetErrorString(result));
        std::exit(1);
    }
}

__device__ __forceinline__ void mma_bf16(float (&acc)[4],
                                           const unsigned (&a)[4],
                                           const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

__global__ void mma_issue_probe(float* output, int k_tiles) {
    extern __shared__ unsigned reserved[];
    if (threadIdx.x == 0) reserved[0] = blockIdx.x & 1;
    __syncthreads();

    // Packed BF16 1.0 fragments. All arithmetic stays in registers; the
    // fixed shared allocation only constrains CTA residency to one per SM.
    const unsigned a[4] = {0x3f803f80u, 0x3f803f80u, 0x3f803f80u, 0x3f803f80u};
    const unsigned b[2] = {0x3f803f80u, 0x3f803f80u};
    float acc[kAccumulatorFragments][4] = {};

#pragma unroll 1
    for (int tile = 0; tile < k_tiles; ++tile) {
        // A BM128/BN256/BK64 CTA has 128 m16n8k16 instructions per warp
        // per K tile. The probe preserves that instruction census, while
        // omitting all production data movement and routing.
#pragma unroll
        for (int phase = 0; phase < 4; ++phase) {
#pragma unroll
            for (int fragment = 0; fragment < kAccumulatorFragments; ++fragment) {
                mma_bf16(acc[fragment], a, b);
            }
        }
    }
    float checksum = static_cast<float>(reserved[0]);
#pragma unroll
    for (int fragment = 0; fragment < kAccumulatorFragments; ++fragment) {
        checksum += acc[fragment][0] + acc[fragment][1]
                  + acc[fragment][2] + acc[fragment][3];
    }
    if ((threadIdx.x & 31) == 0) {
        output[blockIdx.x * (kThreads / 32) + threadIdx.x / 32] = checksum;
    }
}

void run_case(int sm_count, int k_tiles, int waves, float* output,
              cudaEvent_t start, cudaEvent_t end) {
    const int grid = sm_count * waves;
    for (int warmup = 0; warmup < 2; ++warmup) {
        mma_issue_probe<<<grid, kThreads, kSharedBytes>>>(output, k_tiles);
    }
    check(cudaGetLastError(), "warmup launch");
    check(cudaDeviceSynchronize(), "warmup synchronize");

    std::printf("{\"k_tiles\":%d,\"waves\":%d,\"grid\":%d,\"samples_ms\":[",
                k_tiles, waves, grid);
    for (int sample = 0; sample < kSamples; ++sample) {
        check(cudaEventRecord(start), "event start");
        mma_issue_probe<<<grid, kThreads, kSharedBytes>>>(output, k_tiles);
        check(cudaEventRecord(end), "event end");
        check(cudaEventSynchronize(end), "sample synchronize");
        check(cudaGetLastError(), "sample launch");
        float elapsed_ms = 0.0f;
        check(cudaEventElapsedTime(&elapsed_ms, start, end), "event elapsed");
        std::printf("%s%.9f", sample ? "," : "", elapsed_ms);
    }
    std::printf("]}\n");
}

}  // namespace

int main() {
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, 0), "device properties");
    if (props.major != 9 || props.multiProcessorCount != 132) {
        std::fprintf(stderr, "probe requires one 132-SM H200\n");
        return 2;
    }
    check(cudaFuncSetAttribute(mma_issue_probe,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes), "dynamic shared limit");
    int occupancy = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, mma_issue_probe,
        kThreads, kSharedBytes), "CTA occupancy");
    if (occupancy != 1) {
        std::fprintf(stderr, "expected exactly one resident CTA per SM, got %d\n", occupancy);
        return 3;
    }
    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes, mma_issue_probe), "kernel attributes");
    if (attributes.localSizeBytes != 0) {
        std::fprintf(stderr, "probe spilled %zu local bytes per thread\n",
                     attributes.localSizeBytes);
        return 4;
    }
    std::printf("{\"device\":\"%s\",\"sm_count\":%d,\"regs_per_thread\":%d,"
                "\"dynamic_shared_bytes\":%d,\"resident_ctas_per_sm\":%d}\n",
                props.name, props.multiProcessorCount, attributes.numRegs,
                kSharedBytes, occupancy);

    float* output = nullptr;
    check(cudaMalloc(&output, static_cast<size_t>(props.multiProcessorCount) * 32 * 8 * sizeof(float)),
          "output allocation");
    cudaEvent_t start, end;
    check(cudaEventCreate(&start), "start event");
    check(cudaEventCreate(&end), "end event");
    for (int k_tiles : {12, 32}) {
        for (int waves : {16, 32}) run_case(props.multiProcessorCount, k_tiles, waves, output, start, end);
    }
    check(cudaEventDestroy(start), "destroy start event");
    check(cudaEventDestroy(end), "destroy end event");
    check(cudaFree(output), "free output");
    return 0;
}
