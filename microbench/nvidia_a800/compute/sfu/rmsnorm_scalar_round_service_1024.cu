/*
 * Independent SM80 BF16 RMSNorm scalar source-round service probe.
 *
 * This is not an RMSNorm implementation.  The two specializations isolate the
 * scalar work on either side of the block reduction.  There is no reduction,
 * reciprocal square root, epsilon, or production tensor shape.
 *
 * It is the direct analogue of ``add_rmsnorm_round_service_1024_h*.cu`` for
 * the other NORM variant, and exists because A800 has no grid-one CTA-local
 * round service for ``rms_norm_kernel``: the published
 * ``rmsnorm_*_first_1024`` services were captured at grid 64 and report a
 * joint load/compute source-loop envelope, which cannot be separated into a
 * CTA-local memory completion, so Forward refuses them.
 *
 * Source shape, from ``vllm::rms_norm_kernel<c10::BFloat16>`` in the
 * production ``_C.abi3.so`` (REG:24 STACK:0 SHARED:168 LOCAL:0) and its
 * SASS, which issues three ``LDG.E.U16.CONSTANT`` and one ``STG.E.U16``:
 *
 *   pass 1, ``input_square``  one 2-byte load of the row
 *                             ``variance += x * x``
 *   pass 2, ``output_scale``  one 2-byte reread of the row, one 2-byte load
 *                             of the shared weight, ``x * s_variance * w``,
 *                             one 2-byte store to ``out``
 *
 * ``rms_norm`` is not in place: ``out`` is a third tensor, so the store does
 * not alias either load.  Both loads are ``const __restrict__`` in the
 * production signature and keep the read-only encoding here.
 *
 * The production loop is ``for (idx = threadIdx.x; idx < hidden_size;
 * idx += blockDim.x)`` with a 1024-thread block, so a round is 1024 active
 * lanes except for a tail round with the remainder.  Across the A800 NORM
 * cases the only widths that occur are 1024 and 512, which are the two
 * declared specializations; no other width is interpolated.
 *
 * One launch executes exactly one round and times it with ``clock64`` read by
 * lane 0 of each active warp, so the reported service is the intrinsic
 * CTA-local round rather than a steady-state rate amortized over a loop.
 * Larger grids are captured as validation only: they contain launch-wide
 * cache and memory contention, which Forward prices through its separate
 * aggregate envelope.
 *
 * Build:
 *   /usr/local/cuda-12.2/bin/nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo \
 *     -maxrregcount=24 -Xptxas -v \
 *     -o rmsnorm_scalar_round_service_1024 \
 *     rmsnorm_scalar_round_service_1024.cu
 *
 * ``-maxrregcount=24`` caps the probe at the production register footprint.
 * It is a ceiling, not a floor: the specializations compile to 20 and 18
 * registers because one isolated round is less work than the production
 * loop.  That difference does not change the occupancy signature, because a
 * 1024-thread CTA is limited to two per SM by the 2048-thread ceiling long
 * before registers bind -- 24 registers per thread would still leave two.
 * The probe verifies that at runtime through
 * ``cudaOccupancyMaxActiveBlocksPerMultiprocessor`` and aborts on anything
 * other than two.  The build must report 168 bytes of static shared memory
 * and no stack, spill, or local traffic for every specialization.
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK_CUDA(call) do {                                               \
    cudaError_t error = (call);                                             \
    if (error != cudaSuccess) {                                             \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                     __FILE__, __LINE__, cudaGetErrorString(error));        \
        std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
} while (0)

// Production resource signature of vllm::rms_norm_kernel<c10::BFloat16>.
constexpr int kThreads = 1024;
constexpr int kProductionRegisters = 24;
constexpr int kStaticSharedBytes = 168;
constexpr int kResidentCtas = 2;
constexpr int kSmCount = 108;
constexpr const char* kExpectedDevice = "NVIDIA A800-SXM4-80GB";

// The only two active-lane widths a production A800 RMSNorm round issues.
constexpr int kWideRound = 1024;
constexpr int kTailRound = 512;

constexpr int kTrials = 5;
constexpr int kWarmups = 3;

enum class RoundFunction : int { INPUT_SQUARE = 0, OUTPUT_SCALE = 1 };

struct SharedStorage {
    float s_variance;
    unsigned char padding[kStaticSharedBytes - sizeof(float)];
};
static_assert(sizeof(SharedStorage) == kStaticSharedBytes);

__device__ __forceinline__ unsigned int read_smid() {
    unsigned int value;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
    return value;
}

__device__ __forceinline__ std::uint64_t read_clock64() {
    std::uint64_t value;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(value));
    return value;
}

template <RoundFunction function, int ActiveThreads>
__global__ __launch_bounds__(kThreads, kResidentCtas)
void rmsnorm_scalar_round_service(
        const __nv_bfloat16* __restrict__ input,
        const __nv_bfloat16* __restrict__ weight,
        __nv_bfloat16* __restrict__ out,
        float runtime_variance_scale,
        float* __restrict__ sink,
        std::uint64_t* __restrict__ start_cycles,
        std::uint64_t* __restrict__ end_cycles,
        unsigned int* __restrict__ smids) {
    __shared__ __align__(16) SharedStorage shared;
    if (threadIdx.x < kStaticSharedBytes) {
        reinterpret_cast<unsigned char*>(&shared)[threadIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
        shared.s_variance = runtime_variance_scale;
        smids[blockIdx.x] = read_smid();
    }
    __syncthreads();

    std::uint64_t begin_cycles = 0;
    std::uint64_t finish_cycles = 0;
    float variance = 0.0f;
    const int offset = blockIdx.x * ActiveThreads + threadIdx.x;
    if (threadIdx.x < ActiveThreads) {
        const int lane = threadIdx.x % 32;
        if (lane == 0) begin_cycles = read_clock64();
        // Both specializations read the shared cell.  The production kernel
        // holds one ``__shared__ float s_variance`` for the whole function,
        // so its 168-byte static allocation is present in both passes; a
        // specialization that only wrote the cell would have it eliminated
        // and would no longer carry the production occupancy signature.
        const float scale =
            *reinterpret_cast<volatile float*>(&shared.s_variance);
        if constexpr (function == RoundFunction::INPUT_SQUARE) {
            const float x = __bfloat162float(input[offset]);
            variance = fmaf(x, x, scale);
        } else {
            const float x = __bfloat162float(input[offset]);
            const float w = __bfloat162float(weight[threadIdx.x]);
            const float product = x * scale * w;
            out[offset] = __float2bfloat16_rn(product);
            variance = product;
        }
        if (lane == 0) finish_cycles = read_clock64();
        const int warp = threadIdx.x / 32;
        if (lane == 0) {
            const int timer_offset = blockIdx.x * (ActiveThreads / 32) + warp;
            start_cycles[timer_offset] = begin_cycles;
            end_cycles[timer_offset] = finish_cycles;
        }
        sink[offset] = variance;
    }
}

struct Cell { const char* name; int grid_blocks; };

template <RoundFunction function, int ActiveThreads>
int resident_cta_limit() {
    int blocks = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, rmsnorm_scalar_round_service<function, ActiveThreads>,
        kThreads, 0));
    return blocks;
}

template <RoundFunction function, int ActiveThreads>
void run_cell(const Cell& cell, const __nv_bfloat16* input,
              const __nv_bfloat16* weight, __nv_bfloat16* out, float* sink,
              std::uint64_t* starts, std::uint64_t* ends,
              unsigned int* smids) {
    const int limit = resident_cta_limit<function, ActiveThreads>();
    if (limit != kResidentCtas) {
        std::fprintf(stderr,
                     "resident CTA limit %d does not match the production "
                     "signature %d\n", limit, kResidentCtas);
        std::exit(EXIT_FAILURE);
    }
    const int warps = ActiveThreads / 32;
    auto launch = [&]() {
        rmsnorm_scalar_round_service<function, ActiveThreads>
            <<<cell.grid_blocks, kThreads>>>(
                input, weight, out, 0.875f, sink, starts, ends, smids);
    };
    for (int warmup = 0; warmup < kWarmups; ++warmup) launch();
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<std::uint64_t> host_starts(cell.grid_blocks * warps);
    std::vector<std::uint64_t> host_ends(cell.grid_blocks * warps);
    std::vector<unsigned int> host_smids(cell.grid_blocks);
    for (int trial = 0; trial < kTrials; ++trial) {
        launch();
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(host_starts.data(), starts,
                              host_starts.size() * sizeof(std::uint64_t),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(host_ends.data(), ends,
                              host_ends.size() * sizeof(std::uint64_t),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(host_smids.data(), smids,
                              host_smids.size() * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        // The CTA-local round service is the widest warp span inside a CTA:
        // the round is complete when its last active warp has finished.
        for (int block = 0; block < cell.grid_blocks; ++block) {
            std::uint64_t first = ~0ULL, last = 0;
            for (int warp = 0; warp < warps; ++warp) {
                const int index = block * warps + warp;
                first = std::min(first, host_starts[index]);
                last = std::max(last, host_ends[index]);
            }
            std::printf("%d,%s,%s,%d,%d,%d,%d,%u,%llu\n",
                        trial,
                        function == RoundFunction::INPUT_SQUARE
                            ? "input_square" : "output_scale",
                        cell.name, cell.grid_blocks, kThreads, ActiveThreads,
                        block, host_smids[block],
                        static_cast<unsigned long long>(last - first));
        }
    }
}

int main(int argc, char** argv) {
    // grid 1 defines the Forward-eligible CTA-local service; the larger
    // grids are validation cells only.
    const Cell cells[] = {
        {"grid_1", 1}, {"grid_4", 4}, {"grid_16", 16}, {"grid_108", 108},
    };

    if (argc == 2 && std::strcmp(argv[1], "--describe") == 0) {
        std::printf("probe=rmsnorm_scalar_round_service_1024 arch=sm_80\n");
        std::printf("block_threads=%d registers_per_thread=%d "
                    "static_shared_bytes=%d resident_ctas=%d\n",
                    kThreads, kProductionRegisters, kStaticSharedBytes,
                    kResidentCtas);
        std::printf("active_threads_per_source_round=%d,%d\n",
                    kTailRound, kWideRound);
        std::printf("functions=input_square,output_scale rounds_per_launch=1"
                    " timer_domain=clock64_same_cta\n");
        std::printf("forward_eligible_cell=grid_1 "
                    "larger_grids=validation_only trials=%d\n", kTrials);
        return 0;
    }

    cudaDeviceProp properties;
    CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
    if (std::strcmp(properties.name, kExpectedDevice) != 0
            || properties.multiProcessorCount != kSmCount) {
        std::fprintf(stderr, "unexpected device %s with %d SMs\n",
                     properties.name, properties.multiProcessorCount);
        return EXIT_FAILURE;
    }

    const std::size_t elements =
        static_cast<std::size_t>(kSmCount) * 2 * kWideRound;
    __nv_bfloat16 *input = nullptr, *weight = nullptr, *out = nullptr;
    float* sink = nullptr;
    std::uint64_t *starts = nullptr, *ends = nullptr;
    unsigned int* smids = nullptr;
    CHECK_CUDA(cudaMalloc(&input, elements * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&weight, kWideRound * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&out, elements * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&sink, elements * sizeof(float)));
    const std::size_t timers =
        static_cast<std::size_t>(kSmCount) * 2 * (kWideRound / 32);
    CHECK_CUDA(cudaMalloc(&starts, timers * sizeof(std::uint64_t)));
    CHECK_CUDA(cudaMalloc(&ends, timers * sizeof(std::uint64_t)));
    CHECK_CUDA(cudaMalloc(&smids, kSmCount * 2 * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(input, 0, elements * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMemset(weight, 0, kWideRound * sizeof(__nv_bfloat16)));

    std::printf("trial,function,cell,grid_blocks,block_threads,"
                "active_threads,block,smid,round_cycles\n");
    for (const Cell& cell : cells) {
        run_cell<RoundFunction::INPUT_SQUARE, kWideRound>(
            cell, input, weight, out, sink, starts, ends, smids);
        run_cell<RoundFunction::OUTPUT_SCALE, kWideRound>(
            cell, input, weight, out, sink, starts, ends, smids);
        run_cell<RoundFunction::INPUT_SQUARE, kTailRound>(
            cell, input, weight, out, sink, starts, ends, smids);
        run_cell<RoundFunction::OUTPUT_SCALE, kTailRound>(
            cell, input, weight, out, sink, starts, ends, smids);
    }

    CHECK_CUDA(cudaFree(smids));
    CHECK_CUDA(cudaFree(ends));
    CHECK_CUDA(cudaFree(starts));
    CHECK_CUDA(cudaFree(sink));
    CHECK_CUDA(cudaFree(out));
    CHECK_CUDA(cudaFree(weight));
    CHECK_CUDA(cudaFree(input));
    return 0;
}
