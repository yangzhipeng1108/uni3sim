/* Measure two-resident-CTA overlap for the isolated FA2 scalar fragments. */

#define main fa2_scalar_dependency_original_main
#include "fa2_scalar_dependency.cu"
#undef main

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kResidentDynamicSharedBytes = 77824;
constexpr int kExpectedSms = 108;

template <int Mode, int RowsPerThread, int ScoreValuesPerRow>
void run_resident_cell(int grid, int repeats, int samples) {
    CHECK_CUDA(cudaFuncSetAttribute(
        scalar_dependency_kernel<Mode, RowsPerThread, ScoreValuesPerRow>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kResidentDynamicSharedBytes));
    int resident_blocks = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_blocks,
        scalar_dependency_kernel<Mode, RowsPerThread, ScoreValuesPerRow>,
        kThreads,
        kResidentDynamicSharedBytes));
    if (resident_blocks != 2) {
        std::fprintf(stderr, "expected exactly two resident CTAs, got %d\n",
                     resident_blocks);
        std::exit(EXIT_FAILURE);
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
        scalar_dependency_kernel<Mode, RowsPerThread, ScoreValuesPerRow>
            <<<grid, kThreads, kResidentDynamicSharedBytes>>>(
                seeds, results, repeats);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t begin;
    cudaEvent_t end;
    CHECK_CUDA(cudaEventCreate(&begin));
    CHECK_CUDA(cudaEventCreate(&end));
    std::vector<Result> host_results(grid);
    std::printf(
        "sample,mode,grid,repeats,lane_local_rows_per_thread,"
        "score_values_per_lane_local_row,resident_ctas_per_sm,"
        "dynamic_shared_bytes,duration_ns,median_cta_cycles\n");
    for (int sample = 0; sample < samples; ++sample) {
        CHECK_CUDA(cudaEventRecord(begin));
        scalar_dependency_kernel<Mode, RowsPerThread, ScoreValuesPerRow>
            <<<grid, kThreads, kResidentDynamicSharedBytes>>>(
                seeds, results, repeats);
        CHECK_CUDA(cudaEventRecord(end));
        CHECK_CUDA(cudaEventSynchronize(end));
        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, begin, end));
        CHECK_CUDA(cudaMemcpy(host_results.data(), results,
                              grid * sizeof(Result), cudaMemcpyDeviceToHost));

        std::vector<int> blocks_by_sm(kExpectedSms, 0);
        std::vector<std::uint64_t> cycles;
        cycles.reserve(grid);
        for (const Result& result : host_results) {
            if (result.sm_id >= kExpectedSms || result.cycles == 0) {
                std::fprintf(stderr, "invalid block result\n");
                std::exit(EXIT_FAILURE);
            }
            ++blocks_by_sm[result.sm_id];
            cycles.push_back(result.cycles);
        }
        const int expected_blocks_per_sm = grid / kExpectedSms;
        if (grid % kExpectedSms != 0
                || !std::all_of(
                    blocks_by_sm.begin(), blocks_by_sm.end(),
                    [expected_blocks_per_sm](int count) {
                        return count == expected_blocks_per_sm;
                    })) {
            std::fprintf(stderr,
                         "CTA assignment is not balanced across all SMs\n");
            std::exit(EXIT_FAILURE);
        }
        std::sort(cycles.begin(), cycles.end());
        const std::uint64_t median_cycles = cycles[cycles.size() / 2];
        std::printf(
            "%d,%d,%d,%d,%d,%d,%d,%d,%.3f,%llu\n",
            sample, Mode, grid, repeats, RowsPerThread,
            ScoreValuesPerRow, resident_blocks,
            kResidentDynamicSharedBytes,
            static_cast<double>(elapsed_ms) * 1.0e6,
            static_cast<unsigned long long>(median_cycles));
    }
    CHECK_CUDA(cudaEventDestroy(end));
    CHECK_CUDA(cudaEventDestroy(begin));
    CHECK_CUDA(cudaFree(results));
    CHECK_CUDA(cudaFree(seeds));
}

template <int RowsPerThread, int ScoreValuesPerRow>
void dispatch_mode(int mode, int grid, int repeats, int samples) {
    if (mode == 0) {
        run_resident_cell<0, RowsPerThread, ScoreValuesPerRow>(
            grid, repeats, samples);
    } else if (mode == 1) {
        run_resident_cell<1, RowsPerThread, ScoreValuesPerRow>(
            grid, repeats, samples);
    } else if (mode == 3) {
        run_resident_cell<3, RowsPerThread, ScoreValuesPerRow>(
            grid, repeats, samples);
    } else if (mode == 4) {
        run_resident_cell<4, RowsPerThread, ScoreValuesPerRow>(
            grid, repeats, samples);
    } else if (mode == 5) {
        run_resident_cell<5, RowsPerThread, ScoreValuesPerRow>(
            grid, repeats, samples);
    } else if (mode == 6) {
        run_resident_cell<6, RowsPerThread, ScoreValuesPerRow>(
            grid, repeats, samples);
    } else if (mode == 7) {
        run_resident_cell<7, RowsPerThread, ScoreValuesPerRow>(
            grid, repeats, samples);
    } else {
        std::fprintf(stderr, "mode must be one of 0,1,3,4,5,6,7\n");
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
    const int rows = parse_positive(argv[5], "rows");
    const int score_values = parse_positive(argv[6], "score_values");
    if (grid != 108 && grid != 216) {
        std::fprintf(stderr, "grid must be 108 or 216\n");
        return EXIT_FAILURE;
    }
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
    if (properties.major != 8 || properties.minor != 0
            || properties.multiProcessorCount != kExpectedSms) {
        std::fprintf(stderr, "expected a 108-SM sm_80 device\n");
        return EXIT_FAILURE;
    }
    if (rows == 2 && score_values == 32) {
        dispatch_mode<2, 32>(mode, grid, repeats, samples);
    } else if (rows == 4 && score_values == 16) {
        dispatch_mode<4, 16>(mode, grid, repeats, samples);
    } else if (rows == 4 && score_values == 32) {
        dispatch_mode<4, 32>(mode, grid, repeats, samples);
    } else {
        std::fprintf(stderr, "unsupported scalar geometry\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
