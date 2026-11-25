/*
 * Independent SM80 BF16 staged tile dependency probe.
 *
 * This is not a production matrix multiplication.  Each CTA consumes a
 * private stream of synthetic 64x64x64 BF16 panels through a six-slot
 * cp.async -> shared-memory -> HMMA handoff.  It has no matrix-level output
 * ownership, split-K reduction, epilogue, or production operand layout.  The
 * only retained mechanism is one source-sized tile pipeline at fixed
 * one-CTA-per-SM residency.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>


#define CHECK_CUDA(call) do {                                               \
    cudaError_t error = (call);                                             \
    if (error != cudaSuccess) {                                             \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                     __FILE__, __LINE__, cudaGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
} while (0)


namespace wmma = nvcuda::wmma;

constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kMatrixExtent = 64;
constexpr int kWmmaExtent = 16;
constexpr int kElementsPerOperand = kMatrixExtent * kMatrixExtent;
constexpr int kElementsPerPanel = 2 * kElementsPerOperand;
constexpr int kPanelBytes = kElementsPerPanel * sizeof(__nv_bfloat16);
constexpr int kPipelineStages = 6;
constexpr int kDynamicSharedBytes = kPipelineStages * kPanelBytes;
constexpr int kSmCount = 108;
constexpr int kWarmups = 3;
constexpr int kGrids[] = {1, 64, 80, 108};
constexpr int kRounds[] = {16, 64, 128, 224};
constexpr const char* kExpectedDevice = "NVIDIA A800-SXM4-80GB";


__device__ __forceinline__ unsigned int read_smid() {
    unsigned int value;
    asm volatile("mov.u32 %0, %smid;" : "=r"(value));
    return value;
}


__device__ __forceinline__ void cp_async_16(
        void* shared_pointer, const void* global_pointer) {
    const unsigned int shared_address =
        __cvta_generic_to_shared(shared_pointer);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        :: "r"(shared_address), "l"(global_pointer) : "memory");
}


__device__ __forceinline__ void copy_panel(
        unsigned char* shared_panel,
        const unsigned char* global_panel) {
    for (int offset = threadIdx.x * 16;
         offset < kPanelBytes;
         offset += blockDim.x * 16) {
        cp_async_16(shared_panel + offset, global_panel + offset);
    }
}


__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}


__device__ __forceinline__ void cp_async_wait_for_current(int future_groups) {
    if (future_groups >= 5) {
        asm volatile("cp.async.wait_group 5;" ::: "memory");
    } else if (future_groups == 4) {
        asm volatile("cp.async.wait_group 4;" ::: "memory");
    } else if (future_groups == 3) {
        asm volatile("cp.async.wait_group 3;" ::: "memory");
    } else if (future_groups == 2) {
        asm volatile("cp.async.wait_group 2;" ::: "memory");
    } else if (future_groups == 1) {
        asm volatile("cp.async.wait_group 1;" ::: "memory");
    } else {
        asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
}


__global__ void tile_dependency(
        const __nv_bfloat16* __restrict__ input,
        float* __restrict__ output,
        int rounds,
        std::uint64_t* __restrict__ start_cycles,
        std::uint64_t* __restrict__ end_cycles,
        unsigned int* __restrict__ smids) {
    extern __shared__ __align__(16) unsigned char shared_storage[];
    const int warp = threadIdx.x / 32;
    const std::size_t cta_input =
        static_cast<std::size_t>(blockIdx.x) * rounds * kElementsPerPanel;
    const int preload = min(rounds, kPipelineStages);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulators[4];
    for (int index = 0; index < 4; ++index) {
        wmma::fill_fragment(accumulators[index], 0.0f);
    }

    if (threadIdx.x == 0) {
        smids[blockIdx.x] = read_smid();
        start_cycles[blockIdx.x] = clock64();
    }
    __syncthreads();
    for (int panel = 0; panel < preload; ++panel) {
        copy_panel(
            shared_storage + panel * kPanelBytes,
            reinterpret_cast<const unsigned char*>(
                input + cta_input
                + static_cast<std::size_t>(panel) * kElementsPerPanel));
        cp_async_commit();
    }

#pragma unroll 1
    for (int round = 0; round < rounds; ++round) {
        const int remaining = rounds - round - 1;
        cp_async_wait_for_current(min(remaining, kPipelineStages - 1));
        __syncthreads();
        const __nv_bfloat16* panel =
            reinterpret_cast<const __nv_bfloat16*>(
                shared_storage + (round % kPipelineStages) * kPanelBytes);
        const __nv_bfloat16* operand_a = panel;
        const __nv_bfloat16* operand_b = panel + kElementsPerOperand;
        for (int k_slice = 0; k_slice < 4; ++k_slice) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16,
                           __nv_bfloat16, wmma::row_major> fragment_a;
            wmma::load_matrix_sync(
                fragment_a,
                operand_a + warp * kWmmaExtent * kMatrixExtent
                + k_slice * kWmmaExtent,
                kMatrixExtent);
            for (int n_slice = 0; n_slice < 4; ++n_slice) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16,
                               __nv_bfloat16, wmma::col_major> fragment_b;
                wmma::load_matrix_sync(
                    fragment_b,
                    operand_b + k_slice * kWmmaExtent
                    + n_slice * kWmmaExtent * kMatrixExtent,
                    kMatrixExtent);
                wmma::mma_sync(
                    accumulators[n_slice], fragment_a, fragment_b,
                    accumulators[n_slice]);
            }
        }
        __syncthreads();
        const int next_panel = round + kPipelineStages;
        if (next_panel < rounds) {
            copy_panel(
                shared_storage + (round % kPipelineStages) * kPanelBytes,
                reinterpret_cast<const unsigned char*>(
                    input + cta_input
                    + static_cast<std::size_t>(next_panel)
                      * kElementsPerPanel));
            cp_async_commit();
        }
    }
    cp_async_wait_for_current(0);

    float* cta_output = output
        + static_cast<std::size_t>(blockIdx.x)
          * kMatrixExtent * kMatrixExtent;
    for (int n_slice = 0; n_slice < 4; ++n_slice) {
        wmma::store_matrix_sync(
            cta_output + warp * kWmmaExtent * kMatrixExtent
            + n_slice * kWmmaExtent,
            accumulators[n_slice], kMatrixExtent, wmma::mem_row_major);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        end_cycles[blockIdx.x] = clock64();
    }
}


void launch_cell(
        int grid, int rounds,
        const __nv_bfloat16* input,
        float* output,
        std::uint64_t* starts,
        std::uint64_t* ends,
        unsigned int* smids) {
    tile_dependency<<<grid, kThreads, kDynamicSharedBytes>>>(
        input, output, rounds, starts, ends, smids);
}


int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s DEVICE TRIALS\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device = std::stoi(argv[1]);
    const int trials = std::stoi(argv[2]);
    if (device < 0 || trials < 3) {
        std::fprintf(stderr, "device must be nonnegative and trials >= 3\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (std::string(properties.name) != kExpectedDevice
            || properties.major != 8 || properties.minor != 0
            || properties.multiProcessorCount != kSmCount) {
        std::fprintf(stderr, "unexpected device contract\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaFuncSetAttribute(
        tile_dependency,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kDynamicSharedBytes));
    int resident_ctas = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_ctas, tile_dependency, kThreads, kDynamicSharedBytes));
    if (resident_ctas != 1) {
        std::fprintf(stderr, "expected one resident CTA per SM, got %d\n",
                     resident_ctas);
        return EXIT_FAILURE;
    }

    constexpr int max_grid = kGrids[sizeof(kGrids) / sizeof(kGrids[0]) - 1];
    constexpr int max_rounds =
        kRounds[sizeof(kRounds) / sizeof(kRounds[0]) - 1];
    const std::size_t input_elements =
        static_cast<std::size_t>(max_grid) * max_rounds * kElementsPerPanel;
    const std::size_t output_elements =
        static_cast<std::size_t>(max_grid)
        * kMatrixExtent * kMatrixExtent;
    __nv_bfloat16* input = nullptr;
    float* output = nullptr;
    std::uint64_t* starts = nullptr;
    std::uint64_t* ends = nullptr;
    unsigned int* smids = nullptr;
    CHECK_CUDA(cudaMalloc(&input, input_elements * sizeof(*input)));
    CHECK_CUDA(cudaMalloc(&output, output_elements * sizeof(*output)));
    CHECK_CUDA(cudaMalloc(&starts, max_grid * sizeof(*starts)));
    CHECK_CUDA(cudaMalloc(&ends, max_grid * sizeof(*ends)));
    CHECK_CUDA(cudaMalloc(&smids, max_grid * sizeof(*smids)));
    CHECK_CUDA(cudaMemset(input, 0, input_elements * sizeof(*input)));

    std::printf("meta,schema,nvidia_a800_bf16_staged_tile_dependency_native_v1\n");
    std::printf("meta,device_index,%d\n", device);
    std::printf("meta,device_name,%s\n", properties.name);
    std::printf("meta,compute_capability,sm_%d%d\n",
                properties.major, properties.minor);
    std::printf("meta,sm_count,%d\n", properties.multiProcessorCount);
    std::printf("meta,block_threads,%d\n", kThreads);
    std::printf("meta,warps_per_cta,%d\n", kWarps);
    std::printf("meta,tile_shape,64x64x64\n");
    std::printf("meta,pipeline_stages,%d\n", kPipelineStages);
    std::printf("meta,panel_bytes,%d\n", kPanelBytes);
    std::printf("meta,dynamic_shared_bytes_per_cta,%d\n", kDynamicSharedBytes);
    std::printf("meta,resident_ctas_per_sm,%d\n", resident_ctas);
    std::printf("meta,dependency,ldgsts_shared_hmma_loop_carried_accumulator\n");
    std::printf("meta,timer_domain,clock64_sm_local_cta_cycles\n");
    std::printf("meta,production_or_evaluation_duration_consumed,false\n");
    std::printf("raw,grid,rounds,trial,active_sms,median_cta_cycles,"
                "maximum_cta_cycles,loaded_bytes,stored_bytes,tensor_fma\n");

    std::vector<std::uint64_t> host_starts(max_grid);
    std::vector<std::uint64_t> host_ends(max_grid);
    std::vector<unsigned int> host_smids(max_grid);
    for (int grid : kGrids) {
        for (int rounds : kRounds) {
            for (int warmup = 0; warmup < kWarmups; ++warmup) {
                launch_cell(
                    grid, rounds, input, output, starts, ends, smids);
            }
            CHECK_CUDA(cudaDeviceSynchronize());
            for (int trial = 0; trial < trials; ++trial) {
                launch_cell(
                    grid, rounds, input, output, starts, ends, smids);
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaMemcpy(
                    host_starts.data(), starts,
                    grid * sizeof(*starts), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(
                    host_ends.data(), ends,
                    grid * sizeof(*ends), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(
                    host_smids.data(), smids,
                    grid * sizeof(*smids), cudaMemcpyDeviceToHost));
                std::vector<std::uint64_t> cycles;
                std::vector<bool> active(kSmCount, false);
                for (int block = 0; block < grid; ++block) {
                    if (host_smids[block] >= static_cast<unsigned int>(kSmCount)
                            || host_ends[block] <= host_starts[block]) {
                        std::fprintf(stderr, "invalid CTA timing record\n");
                        return EXIT_FAILURE;
                    }
                    active[host_smids[block]] = true;
                    cycles.push_back(host_ends[block] - host_starts[block]);
                }
                std::sort(cycles.begin(), cycles.end());
                const int active_sms = static_cast<int>(
                    std::count(active.begin(), active.end(), true));
                const std::uint64_t loaded_bytes =
                    static_cast<std::uint64_t>(grid) * rounds * kPanelBytes;
                const std::uint64_t stored_bytes =
                    static_cast<std::uint64_t>(grid)
                    * kMatrixExtent * kMatrixExtent * sizeof(float);
                const std::uint64_t tensor_fma =
                    static_cast<std::uint64_t>(grid) * rounds
                    * kMatrixExtent * kMatrixExtent * kMatrixExtent;
                std::printf("raw,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%llu\n",
                            grid, rounds, trial, active_sms,
                            static_cast<unsigned long long>(
                                cycles[cycles.size() / 2]),
                            static_cast<unsigned long long>(cycles.back()),
                            static_cast<unsigned long long>(loaded_bytes),
                            static_cast<unsigned long long>(stored_bytes),
                            static_cast<unsigned long long>(tensor_fma));
            }
        }
    }

    CHECK_CUDA(cudaFree(smids));
    CHECK_CUDA(cudaFree(ends));
    CHECK_CUDA(cudaFree(starts));
    CHECK_CUDA(cudaFree(output));
    CHECK_CUDA(cudaFree(input));
    return EXIT_SUCCESS;
}
