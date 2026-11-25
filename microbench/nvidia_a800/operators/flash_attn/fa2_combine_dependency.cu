/* Isolated SM80 primitives from the FA2 split-combine CTA dependency chain. */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

constexpr int kThreads = 128;

struct Result {
    std::uint64_t cycles;
    std::uint32_t sm_id;
    std::uint32_t checksum;
};

void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", expression,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(call) check_cuda((call), #call)

struct MaxOp {
    __device__ __forceinline__ float operator()(float lhs, float rhs) const {
        return fmaxf(lhs, rhs);
    }
};

struct SumOp {
    __device__ __forceinline__ float operator()(float lhs, float rhs) const {
        return lhs + rhs;
    }
};

template <int Threads>
struct Allreduce {
    static_assert(Threads == 32 || Threads == 16 || Threads == 8
                  || Threads == 4);

    template <typename Operator>
    __device__ __forceinline__ static float run(float value, Operator op) {
        constexpr int offset = Threads / 2;
        value = op(value, __shfl_xor_sync(0xffffffffU, value, offset));
        return Allreduce<offset>::run(value, op);
    }
};

template <>
struct Allreduce<2> {
    template <typename Operator>
    __device__ __forceinline__ static float run(float value, Operator op) {
        return op(value, __shfl_xor_sync(0xffffffffU, value, 1));
    }
};

__device__ __forceinline__ std::uint32_t physical_smid() {
    std::uint32_t value;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
    return value;
}

__device__ __forceinline__ void compiler_barrier() {
    asm volatile("" ::: "memory");
}

__device__ __forceinline__ float load_global_cg_32(const float* address) {
    float value;
    asm volatile("ld.global.cg.f32 %0, [%1];"
                 : "=f"(value) : "l"(address) : "memory");
    return value;
}

template <int Mode, int AllreduceWidth, int MaxSplits, int BlockM,
          int HeadDim>
__global__ __launch_bounds__(kThreads, 1)
void combine_dependency_kernel(const float* seeds, Result* results,
                               int repeats, int num_splits) {
    static_assert(AllreduceWidth == 2 || AllreduceWidth == 4
                  || AllreduceWidth == 8 || AllreduceWidth == 16
                  || AllreduceWidth == 32);
    static_assert(MaxSplits == 2 || MaxSplits == 4 || MaxSplits == 8
                  || MaxSplits == 16 || MaxSplits == 32);
    static_assert((BlockM == 4 && HeadDim == 128)
                  || (BlockM == 8 && HeadDim == 64));
    static_assert(BlockM * HeadDim / kThreads == 4);

    __shared__ std::uint64_t begin_ticks[kThreads];
    __shared__ std::uint64_t end_ticks[kThreads];
    __shared__ std::uint32_t checksums[kThreads];
    // This is the exact source layout: the extra column controls the bank
    // mapping of the post-load transpose and of the scale publication.
    __shared__ float s_lse[MaxSplits][BlockM + 1];
    const int lane = threadIdx.x;
    const float seed = seeds[blockIdx.x * kThreads + lane];
    float value = seed + 0.125f;
    constexpr int kNLsePerThread = (
        MaxSplits * BlockM + kThreads - 1) / kThreads;
    constexpr int kRowsPerLoadLSE = kThreads / BlockM;
    constexpr int kRowsPerLoadTranspose = (
        kRowsPerLoadLSE < MaxSplits ? kRowsPerLoadLSE : MaxSplits);
    float lse_accum[kNLsePerThread];
    #pragma unroll
    for (int index = 0; index < kNLsePerThread; ++index) {
        lse_accum[index] = seed + 0.015625f * (index + 1);
    }
    float output[4] = {
        seed + 0.25f, seed + 0.5f, seed + 0.75f, seed + 1.0f};
    __nv_bfloat162 converted[2];
    std::uint32_t packed0 = __float_as_uint(output[0]);
    std::uint32_t packed1 = __float_as_uint(output[2]);

    __syncthreads();
    const std::uint64_t begin = clock64();
    compiler_barrier();
    #pragma unroll 1
    for (int repeat = 0; repeat < repeats; ++repeat) {
        if constexpr (Mode == 0) {
            compiler_barrier();
        } else if constexpr (Mode == 1) {
            value = Allreduce<AllreduceWidth>::run(value, MaxOp{});
        } else if constexpr (Mode == 2) {
            value = Allreduce<AllreduceWidth>::run(value, SumOp{});
        }
        if constexpr (Mode == 3) {
            asm volatile("" : "+r"(packed0), "+r"(packed1));
        } else if constexpr (Mode == 4) {
            converted[0] = __float22bfloat162_rn(make_float2(
                __uint_as_float(packed0), __uint_as_float(packed0)));
            converted[1] = __float22bfloat162_rn(make_float2(
                __uint_as_float(packed1), __uint_as_float(packed1)));
            packed0 = *reinterpret_cast<std::uint32_t*>(&converted[0]);
            packed1 = *reinterpret_cast<std::uint32_t*>(&converted[1]);
            asm volatile("" : "+r"(packed0), "+r"(packed1));
        } else if constexpr (Mode == 5 || Mode == 9) {
            // FA2 combine: registers holding the just-loaded LSE values are
            // published with the padded shared-memory stride, synchronized,
            // then reread in the source's transposed lane layout.
            #pragma unroll
            for (int index = 0; index < kNLsePerThread; ++index) {
                const int row = (
                    index * kRowsPerLoadLSE + lane / BlockM);
                const int column = lane % BlockM;
                // Mode 9 adds the source's coalesced-but-row-strided global
                // LSE load.  The stride is exactly lse_size=grid*kBlockM,
                // matching gLSEaccum(row, col) across consecutive CTAs.
                const std::size_t lse_size =
                    static_cast<std::size_t>(gridDim.x) * BlockM;
                const float lse = row < num_splits
                    ? (Mode == 9
                       ? load_global_cg_32(
                           seeds + static_cast<std::size_t>(row) * lse_size
                           + static_cast<std::size_t>(blockIdx.x) * BlockM
                           + column)
                       : lse_accum[index])
                    : -INFINITY;
                if (row < MaxSplits) {
                    s_lse[row][column] = lse;
                }
            }
            __syncthreads();
            #pragma unroll
            for (int index = 0; index < kNLsePerThread; ++index) {
                const int row = (
                    index * kRowsPerLoadTranspose
                    + lane % kRowsPerLoadTranspose);
                const int column = lane / kRowsPerLoadTranspose;
                lse_accum[index] = (
                    row < MaxSplits && column < BlockM)
                    ? s_lse[row][column] : -INFINITY;
            }
            value = lse_accum[0];
        } else if constexpr (Mode == 6) {
            // This begins after the max Allreduce.  It deliberately includes
            // only expf and the source-local accumulation, never the following
            // sum Allreduce.
            const float lse_max = value;
            float lse_sum = expf(lse_accum[0] - lse_max);
            #pragma unroll
            for (int index = 1; index < kNLsePerThread; ++index) {
                lse_sum += expf(lse_accum[index] - lse_max);
            }
            value = lse_sum + 0.125f;
        } else if constexpr (Mode == 7) {
            // This begins after the sum Allreduce.  Keep the source's NaN/zero
            // guard in the measured fragment so predication is source-shaped.
            const float lse_sum = fabsf(value) + 1.0f;
            const float lse_max = seed + 0.5f;
            value = (lse_sum == 0.0f || lse_sum != lse_sum)
                ? INFINITY : logf(lse_sum) + lse_max;
        } else if constexpr (Mode == 8) {
            // FA2 combine publishes one active split-by-row scale per source
            // predicate, then synchronizes before the Oaccum loop consumes it.
            const float lse_logsum = seed + 0.75f;
            #pragma unroll
            for (int index = 0; index < kNLsePerThread; ++index) {
                const int row = (
                    index * kRowsPerLoadTranspose
                    + lane % kRowsPerLoadTranspose);
                const int column = lane / kRowsPerLoadTranspose;
                if (row < num_splits && column < BlockM) {
                    s_lse[row][column] = expf(
                        lse_accum[index] - lse_logsum);
                }
            }
            __syncthreads();
        }
    }
    compiler_barrier();
    const std::uint64_t end = clock64();
    compiler_barrier();

    // Make the scale publication observable after the timed interval without
    // charging the consumer read to this producer/barrier fragment.
    if constexpr (Mode == 8) {
        const int row = lane % kRowsPerLoadTranspose;
        const int column = lane / kRowsPerLoadTranspose;
        if (row < num_splits && column < BlockM) {
            value = s_lse[row][column];
        }
    }

    std::uint32_t checksum = __float_as_uint(value);
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
        checksum ^= __float_as_uint(output[item]);
    }
    checksum ^= packed0 ^ packed1;
    if constexpr (Mode == 4) {
        checksum ^= *reinterpret_cast<std::uint32_t*>(&converted[0]);
        checksum ^= *reinterpret_cast<std::uint32_t*>(&converted[1]);
    }
    begin_ticks[lane] = begin;
    end_ticks[lane] = end;
    checksums[lane] = checksum;
    __syncthreads();

    if (lane == 0) {
        std::uint64_t first = begin_ticks[0];
        std::uint64_t last = end_ticks[0];
        std::uint32_t combined = 0;
        #pragma unroll
        for (int thread = 0; thread < kThreads; ++thread) {
            first = min(first, begin_ticks[thread]);
            last = max(last, end_ticks[thread]);
            combined ^= checksums[thread]
                + static_cast<std::uint32_t>(thread);
        }
        results[blockIdx.x] = Result{
            last - first, physical_smid(), combined};
    }
}

int parse_positive(const char* text, const char* name) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0
            || value > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "%s must be a positive integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

int parse_nonnegative(const char* text, const char* name) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0
            || value > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "%s must be a non-negative integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

template <int Width, int MaxSplits, int BlockM, int HeadDim>
void launch_mode(int mode, const float* seeds, Result* results,
                 int grid, int repeats, int num_splits) {
    if (mode == 0) {
        combine_dependency_kernel<0, Width, MaxSplits, BlockM, HeadDim>
            <<<grid, kThreads>>>(seeds, results, repeats, num_splits);
    } else if (mode == 1) {
        combine_dependency_kernel<1, Width, MaxSplits, BlockM, HeadDim>
            <<<grid, kThreads>>>(seeds, results, repeats, num_splits);
    } else if (mode == 2) {
        combine_dependency_kernel<2, Width, MaxSplits, BlockM, HeadDim>
            <<<grid, kThreads>>>(seeds, results, repeats, num_splits);
    } else if (mode == 3) {
        combine_dependency_kernel<3, Width, MaxSplits, BlockM, HeadDim>
            <<<grid, kThreads>>>(seeds, results, repeats, num_splits);
    } else if (mode == 4) {
        combine_dependency_kernel<4, Width, MaxSplits, BlockM, HeadDim>
            <<<grid, kThreads>>>(seeds, results, repeats, num_splits);
    } else if (mode == 5) {
        combine_dependency_kernel<5, Width, MaxSplits, BlockM, HeadDim>
            <<<grid, kThreads>>>(seeds, results, repeats, num_splits);
    } else if (mode == 6) {
        combine_dependency_kernel<6, Width, MaxSplits, BlockM, HeadDim>
            <<<grid, kThreads>>>(seeds, results, repeats, num_splits);
    } else if (mode == 7) {
        combine_dependency_kernel<7, Width, MaxSplits, BlockM, HeadDim>
            <<<grid, kThreads>>>(seeds, results, repeats, num_splits);
    } else if (mode == 8) {
        combine_dependency_kernel<8, Width, MaxSplits, BlockM, HeadDim>
            <<<grid, kThreads>>>(seeds, results, repeats, num_splits);
    } else if (mode == 9) {
        combine_dependency_kernel<9, Width, MaxSplits, BlockM, HeadDim>
            <<<grid, kThreads>>>(seeds, results, repeats, num_splits);
    } else {
        std::fprintf(stderr, "mode must be in [0,8]\n");
        std::exit(EXIT_FAILURE);
    }
}

template <int MaxSplits, int BlockM, int HeadDim>
void launch_width(int mode, int width, const float* seeds, Result* results,
                  int grid, int repeats, int num_splits) {
    if (width == 0 || width == 2) {
        launch_mode<2, MaxSplits, BlockM, HeadDim>(
            mode, seeds, results, grid, repeats, num_splits);
    } else if (width == 4) {
        launch_mode<4, MaxSplits, BlockM, HeadDim>(
            mode, seeds, results, grid, repeats, num_splits);
    } else if (width == 8) {
        launch_mode<8, MaxSplits, BlockM, HeadDim>(
            mode, seeds, results, grid, repeats, num_splits);
    } else if (width == 16) {
        launch_mode<16, MaxSplits, BlockM, HeadDim>(
            mode, seeds, results, grid, repeats, num_splits);
    } else if (width == 32) {
        launch_mode<32, MaxSplits, BlockM, HeadDim>(
            mode, seeds, results, grid, repeats, num_splits);
    } else {
        std::fprintf(stderr, "unsupported allreduce width\n");
        std::exit(EXIT_FAILURE);
    }
}

template <int BlockM, int HeadDim>
void launch_max_splits(int mode, int width, int max_splits,
                       const float* seeds, Result* results, int grid,
                       int repeats, int num_splits) {
    if (max_splits == 2) {
        launch_width<2, BlockM, HeadDim>(
            mode, width, seeds, results, grid, repeats, num_splits);
    } else if (max_splits == 4) {
        launch_width<4, BlockM, HeadDim>(
            mode, width, seeds, results, grid, repeats, num_splits);
    } else if (max_splits == 8) {
        launch_width<8, BlockM, HeadDim>(
            mode, width, seeds, results, grid, repeats, num_splits);
    } else if (max_splits == 16) {
        launch_width<16, BlockM, HeadDim>(
            mode, width, seeds, results, grid, repeats, num_splits);
    } else if (max_splits == 32) {
        launch_width<32, BlockM, HeadDim>(
            mode, width, seeds, results, grid, repeats, num_splits);
    } else {
        std::fprintf(stderr, "unsupported maximum split count\n");
        std::exit(EXIT_FAILURE);
    }
}

int ceil_power_of_two(int value) {
    if (value <= 0 || value > 32) {
        std::fprintf(stderr, "split count must be in [1,32]\n");
        std::exit(EXIT_FAILURE);
    }
    int result = 2;
    while (result < value) {
        result *= 2;
    }
    return result;
}

void launch_geometry(int mode, int width, int num_splits, int block_m,
                     int head_dim, const float* seeds, Result* results,
                     int grid, int repeats) {
    const bool allreduce = mode == 1 || mode == 2
        || (mode == 0 && width > 0);
    const bool source_fragment = mode >= 5 || (mode == 0 && width == 0);
    const bool conversion = mode == 3 || mode == 4;
    if (!source_fragment && !allreduce && !conversion) {
        std::fprintf(stderr, "mode must be in [0,8]\n");
        std::exit(EXIT_FAILURE);
    }
    if (allreduce) {
        if (width != num_splits || width < 2) {
            std::fprintf(stderr, "allreduce geometry must use width splits\n");
            std::exit(EXIT_FAILURE);
        }
    } else if (conversion) {
        if (width != 0 || num_splits != 0) {
            std::fprintf(stderr, "conversion geometry has no split state\n");
            std::exit(EXIT_FAILURE);
        }
    } else if (width != 0 || num_splits <= 0) {
        std::fprintf(stderr, "source fragment requires split geometry\n");
        std::exit(EXIT_FAILURE);
    }
    const int max_splits = conversion ? 2 : ceil_power_of_two(num_splits);
    if (block_m == 4 && head_dim == 128) {
        launch_max_splits<4, 128>(mode, width, max_splits, seeds, results,
                                  grid, repeats, num_splits);
    } else if (block_m == 8 && head_dim == 64) {
        launch_max_splits<8, 64>(mode, width, max_splits, seeds, results,
                                 grid, repeats, num_splits);
    } else {
        std::fprintf(stderr, "unsupported output conversion geometry\n");
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 9) {
        std::fprintf(
            stderr,
            "usage: %s MODE GRID REPEATS SAMPLES WIDTH SPLITS BLOCK_M "
            "HEAD_DIM\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int mode = parse_positive(argv[1], "mode+1") - 1;
    const int grid = parse_positive(argv[2], "grid");
    const int repeats = parse_positive(argv[3], "repeats");
    const int samples = parse_positive(argv[4], "samples");
    const int width = parse_nonnegative(argv[5], "allreduce_width");
    const int num_splits = parse_nonnegative(argv[6], "num_splits");
    const int block_m = parse_positive(argv[7], "block_m");
    const int head_dim = parse_positive(argv[8], "head_dim");
    if (mode < 0 || mode > 9) {
        std::fprintf(stderr, "encoded mode must be in [1,10]\n");
        return EXIT_FAILURE;
    }
    const int max_splits = num_splits == 0 ? 0
        : ceil_power_of_two(num_splits);

    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
    if (properties.major != 8 || properties.minor != 0
            || properties.multiProcessorCount != 108) {
        std::fprintf(stderr, "expected a 108-SM sm_80 device\n");
        return EXIT_FAILURE;
    }
    const std::size_t thread_count =
        static_cast<std::size_t>(grid) * kThreads;
    std::vector<float> host_seeds(thread_count);
    for (std::size_t index = 0; index < thread_count; ++index) {
        host_seeds[index] =
            static_cast<float>((index * 17) & 255) * 0.00390625f;
    }
    float* seeds = nullptr;
    Result* results = nullptr;
    CHECK_CUDA(cudaMalloc(&seeds, thread_count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&results, grid * sizeof(Result)));
    CHECK_CUDA(cudaMemcpy(seeds, host_seeds.data(),
                          thread_count * sizeof(float),
                          cudaMemcpyHostToDevice));
    for (int warmup = 0; warmup < 10; ++warmup) {
        launch_geometry(mode, width, num_splits, block_m, head_dim,
                        seeds, results, grid, repeats);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::printf(
        "sample,mode,grid,repeats,allreduce_width,num_splits,max_splits,"
        "block_m,head_dim,output_values_per_lane,block,sm_id,cycles,"
        "checksum\n");
    std::vector<Result> host_results(grid);
    for (int sample = 0; sample < samples; ++sample) {
        launch_geometry(mode, width, num_splits, block_m, head_dim,
                        seeds, results, grid, repeats);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(host_results.data(), results,
                              grid * sizeof(Result), cudaMemcpyDeviceToHost));
        for (int block = 0; block < grid; ++block) {
            const Result result = host_results[block];
            if (result.cycles == 0 || result.sm_id >= 108) {
                std::fprintf(stderr, "invalid result at block %d\n", block);
                return EXIT_FAILURE;
            }
            std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,4,%d,%u,%llu,%u\n",
                        sample, mode, grid, repeats, width, num_splits,
                        max_splits, block_m, head_dim, block, result.sm_id,
                        static_cast<unsigned long long>(result.cycles),
                        result.checksum);
        }
    }
    CHECK_CUDA(cudaFree(results));
    CHECK_CUDA(cudaFree(seeds));
    return EXIT_SUCCESS;
}
