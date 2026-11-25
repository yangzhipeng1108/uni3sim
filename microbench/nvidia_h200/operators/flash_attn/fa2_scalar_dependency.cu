/* H200 owner copy of microbench/nvidia_a800/operators/flash_attn/
 * fa2_scalar_dependency.cu.  Kernel bodies and the measured mechanism are
 * unchanged.  Hardware-coupling changes only:
 *   - device contract sm_80/108 SMs -> sm_90/132 SMs (H200);
 *   - valid %smid bound 108 -> 132.
 * Compiled with -arch=sm_90 by the H200 runner.
 */
/* Isolated SM80 scalar primitives used by the FA2 online-softmax program. */

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

constexpr int kThreads = 128;
constexpr int kMaximumRowsPerThread = 4;

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

__device__ __forceinline__ float fp32_max(float lhs, float rhs) {
    float result;
    asm volatile("max.f32 %0, %1, %2;" : "=f"(result) : "f"(lhs), "f"(rhs));
    return result;
}

__device__ __forceinline__ float fp32_add(float lhs, float rhs) {
    float result;
    asm volatile("add.rn.f32 %0, %1, %2;"
                 : "=f"(result) : "f"(lhs), "f"(rhs));
    return result;
}

__device__ __forceinline__ std::uint32_t physical_smid() {
    std::uint32_t value;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
    return value;
}

__device__ __forceinline__ void compiler_barrier() {
    asm volatile("" ::: "memory");
}

template <int Mode, int RowsPerThread, int ScoreValuesPerRow>
__global__ __launch_bounds__(kThreads, 1)
void scalar_dependency_kernel(const float* seeds, Result* results,
                              int repeats) {
    static_assert(RowsPerThread == 2 || RowsPerThread == 4);
    static_assert(
        ScoreValuesPerRow == 16 || ScoreValuesPerRow == 32
        || ScoreValuesPerRow == 64);
    __shared__ std::uint64_t begin_ticks[kThreads];
    __shared__ std::uint64_t end_ticks[kThreads];
    __shared__ std::uint32_t checksums[kThreads];
    const int lane = threadIdx.x;
  const float seed = seeds[blockIdx.x * kThreads + lane];
    float values[kMaximumRowsPerThread] = {
        seed + 0.125f, seed - 0.25f, seed + 0.5f, seed - 1.0f};
    std::uint32_t checksum = static_cast<std::uint32_t>(lane + 1);
    float score_values[RowsPerThread][ScoreValuesPerRow];
    __nv_bfloat16 converted_values[RowsPerThread][ScoreValuesPerRow];
    float row_sums[RowsPerThread] = {};
    if constexpr (Mode >= 5) {
        #pragma unroll
        for (int row = 0; row < RowsPerThread; ++row) {
            #pragma unroll
            for (int item = 0; item < ScoreValuesPerRow; ++item) {
                score_values[row][item] = (
                    seed + static_cast<float>(row * ScoreValuesPerRow + item)
                    * 0.0009765625f);
            }
        }
    }

    __syncthreads();
    const std::uint64_t begin = clock64();
    compiler_barrier();
    #pragma unroll 1
    for (int repeat = 0; repeat < repeats; ++repeat) {
        if constexpr (Mode == 0) {
            compiler_barrier();
        } else if constexpr (Mode == 1) {
            #pragma unroll
            for (int item = 0; item < ScoreValuesPerRow; ++item) {
                #pragma unroll
                for (int value = 0; value < RowsPerThread; ++value) {
                    values[value] = fp32_max(values[value], seed);
                }
            }
            #pragma unroll
            for (int offset = 2; offset >= 1; offset /= 2) {
                #pragma unroll
                for (int value = 0; value < RowsPerThread; ++value) {
                    values[value] = fp32_max(
                        values[value], __shfl_xor_sync(
                            0xffffffffU, values[value], offset));
                }
            }
        } else if constexpr (Mode == 2) {
            #pragma unroll
            for (int item = 0; item < ScoreValuesPerRow; ++item) {
                #pragma unroll
                for (int value = 0; value < RowsPerThread; ++value) {
                    values[value] = fp32_add(values[value], seed);
                }
            }
        } else if constexpr (Mode == 3) {
            __syncthreads();
        } else if constexpr (Mode == 4) {
            #pragma unroll
            for (int offset = 2; offset >= 1; offset /= 2) {
                #pragma unroll
                for (int value = 0; value < RowsPerThread; ++value) {
                    values[value] = fp32_add(
                    values[value], __shfl_xor_sync(
                            0xffffffffU, values[value], offset));
                }
            }
        } else if constexpr (Mode == 5) {
            #pragma unroll
            for (int row = 0; row < RowsPerThread; ++row) {
                float sum = 0.0f;
                #pragma unroll
                for (int item = 0; item < ScoreValuesPerRow; ++item) {
                    const float scaled = __fmaf_rn(
                        score_values[row][item], 1.4426950408889634f, -seed);
                    const float probability = exp2f(scaled);
                    score_values[row][item] = probability;
                    sum = fp32_add(sum, probability);
                }
                row_sums[row] = sum;
            }
        } else if constexpr (Mode == 6) {
            const int maximum_column = (
                ScoreValuesPerRow * 4 + static_cast<int>(seed));
            #pragma unroll
            for (int row = 0; row < RowsPerThread; ++row) {
                #pragma unroll
                for (int item = 0; item < ScoreValuesPerRow; ++item) {
                    const int column = item * 4 + (lane & 3);
                    if (column >= maximum_column) {
                        score_values[row][item] = -__int_as_float(0x7f800000);
                    }
                }
            }
        } else if constexpr (Mode == 7) {
            #pragma unroll
            for (int row = 0; row < RowsPerThread; ++row) {
                #pragma unroll
                for (int item = 0; item < ScoreValuesPerRow; ++item) {
                    converted_values[row][item] = __float2bfloat16_rn(
                        score_values[row][item]);
                }
            }
        }
    }
    compiler_barrier();
    const std::uint64_t end = clock64();
    begin_ticks[lane] = begin;
    end_ticks[lane] = end;
    #pragma unroll
    for (int value = 0; value < RowsPerThread; ++value) {
        checksum ^= __float_as_uint(values[value]);
        if constexpr (Mode >= 5) {
            checksum ^= __float_as_uint(row_sums[value]);
            #pragma unroll
            for (int item = 0; item < ScoreValuesPerRow; ++item) {
                checksum ^= __float_as_uint(score_values[value][item]);
                if constexpr (Mode == 7) {
                    checksum ^= static_cast<std::uint32_t>(
                        __bfloat16_as_ushort(converted_values[value][item]));
                }
            }
        }
    }
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
            combined ^= checksums[thread] + static_cast<std::uint32_t>(thread);
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

template <int Mode, int RowsPerThread, int ScoreValuesPerRow>
void launch(const float* seeds, Result* results, int grid, int repeats) {
    scalar_dependency_kernel<Mode, RowsPerThread, ScoreValuesPerRow>
        <<<grid, kThreads>>>(
        seeds, results, repeats);
}

template <int RowsPerThread, int ScoreValuesPerRow>
void launch_mode(int mode, const float* seeds, Result* results,
                 int grid, int repeats) {
    if (mode == 0) {
        launch<0, RowsPerThread, ScoreValuesPerRow>(
            seeds, results, grid, repeats);
    } else if (mode == 1) {
        launch<1, RowsPerThread, ScoreValuesPerRow>(
            seeds, results, grid, repeats);
    } else if (mode == 2) {
        launch<2, RowsPerThread, ScoreValuesPerRow>(
            seeds, results, grid, repeats);
    } else if (mode == 3) {
        launch<3, RowsPerThread, ScoreValuesPerRow>(
            seeds, results, grid, repeats);
    } else if (mode == 4) {
        launch<4, RowsPerThread, ScoreValuesPerRow>(
            seeds, results, grid, repeats);
    } else if (mode == 5) {
        launch<5, RowsPerThread, ScoreValuesPerRow>(
            seeds, results, grid, repeats);
    } else if (mode == 6) {
        launch<6, RowsPerThread, ScoreValuesPerRow>(
            seeds, results, grid, repeats);
    } else if (mode == 7) {
        launch<7, RowsPerThread, ScoreValuesPerRow>(
            seeds, results, grid, repeats);
    } else {
        std::fprintf(stderr, "mode must be in [0,4]\n");
        std::exit(EXIT_FAILURE);
    }
}

void launch_geometry(int mode, int rows_per_thread, int score_values_per_row,
                     const float* seeds, Result* results, int grid,
                     int repeats) {
    if (rows_per_thread == 2 && score_values_per_row == 32) {
        launch_mode<2, 32>(mode, seeds, results, grid, repeats);
    } else if (rows_per_thread == 2 && score_values_per_row == 64) {
        launch_mode<2, 64>(mode, seeds, results, grid, repeats);
    } else if (rows_per_thread == 4 && score_values_per_row == 16) {
        launch_mode<4, 16>(mode, seeds, results, grid, repeats);
    } else if (rows_per_thread == 4 && score_values_per_row == 32) {
        launch_mode<4, 32>(mode, seeds, results, grid, repeats);
    } else {
        std::fprintf(stderr, "unsupported scalar dependency geometry\n");
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(
            stderr,
            "usage: %s MODE GRID REPEATS SAMPLES ROWS SCORE_VALUES\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int mode = parse_positive(argv[1], "mode+1") - 1;
    const int grid = parse_positive(argv[2], "grid");
    const int repeats = parse_positive(argv[3], "repeats");
    const int samples = parse_positive(argv[4], "samples");
    const int rows_per_thread = parse_positive(argv[5], "rows_per_thread");
    const int score_values_per_row = parse_positive(
        argv[6], "score_values_per_row");
    if (mode < 0 || mode > 7) {
        std::fprintf(stderr, "encoded mode must be in [1,8]\n");
        return EXIT_FAILURE;
    }

    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
    if (properties.major != 9 || properties.minor != 0
            || properties.multiProcessorCount != 132) {
        std::fprintf(stderr, "expected a 132-SM sm_90 device\n");
        return EXIT_FAILURE;
    }
    const std::size_t thread_count =
        static_cast<std::size_t>(grid) * kThreads;
    std::vector<float> host_seeds(thread_count);
    for (std::size_t index = 0; index < thread_count; ++index) {
        host_seeds[index] = static_cast<float>((index * 17) & 255) * 0.00390625f;
    }
    float* seeds = nullptr;
    Result* results = nullptr;
    CHECK_CUDA(cudaMalloc(&seeds, thread_count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&results, grid * sizeof(Result)));
    CHECK_CUDA(cudaMemcpy(seeds, host_seeds.data(),
                          thread_count * sizeof(float),
                          cudaMemcpyHostToDevice));
    for (int warmup = 0; warmup < 10; ++warmup) {
        launch_geometry(mode, rows_per_thread, score_values_per_row,
                        seeds, results, grid, repeats);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::printf(
        "sample,mode,grid,repeats,lane_local_rows_per_thread,"
        "score_values_per_lane_local_row,block,sm_id,cycles,checksum\n");
    std::vector<Result> host_results(grid);
    for (int sample = 0; sample < samples; ++sample) {
        launch_geometry(mode, rows_per_thread, score_values_per_row,
                        seeds, results, grid, repeats);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(host_results.data(), results,
                              grid * sizeof(Result), cudaMemcpyDeviceToHost));
        for (int block = 0; block < grid; ++block) {
            const Result result = host_results[block];
            if (result.cycles == 0 || result.sm_id >= 132) {
                std::fprintf(stderr, "invalid result at block %d\n", block);
                return EXIT_FAILURE;
            }
            std::printf("%d,%d,%d,%d,%d,%d,%d,%u,%llu,%u\n",
                        sample, mode, grid, repeats, rows_per_thread,
                        score_values_per_row, block, result.sm_id,
                        static_cast<unsigned long long>(result.cycles),
                        result.checksum);
        }
    }
    CHECK_CUDA(cudaFree(results));
    CHECK_CUDA(cudaFree(seeds));
    return EXIT_SUCCESS;
}
