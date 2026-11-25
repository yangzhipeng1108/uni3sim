/*
 * Independent SM80 local-stack spill-round probe.
 *
 * This is not a FlashAttention implementation. It preserves only the pinned
 * 128-thread specializations' pinned local-stack and scalar load/store site
 * geometry. D128_CAUSAL selects the 176-byte, 51-store, 53-load program;
 * otherwise the original 56-byte, 13-store, 13-load program is retained.
 * REGISTER_CONTROL retains the integer dataflow without local-memory
 * operations so the report can subtract it directly.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t error = (call);                                              \
    if (error != cudaSuccess) {                                              \
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE);                                             \
    }                                                                        \
} while (0)

constexpr int kThreads = 128;
constexpr int kSms = 108;
#ifdef D128_CAUSAL
constexpr int kSlots = 44;
#else
constexpr int kSlots = 13;
#endif

__device__ __forceinline__ unsigned int smid() {
    unsigned int result;
    asm volatile("mov.u32 %0, %smid;" : "=r"(result));
    return result;
}

__global__ __launch_bounds__(kThreads, 2)
void local_spill_round(int rounds, std::uint64_t* starts,
                       std::uint64_t* ends, unsigned int* smids,
                       std::uint32_t* checksums) {
#ifndef REGISTER_CONTROL
    volatile std::uint32_t spill[kSlots];
#endif
    std::uint32_t state = 0x9e3779b9u ^ static_cast<std::uint32_t>(threadIdx.x);
    __syncthreads();
    if (threadIdx.x == 0) {
        smids[blockIdx.x] = smid();
        starts[blockIdx.x] = clock64();
    }
    __syncthreads();
#pragma unroll 1
    for (int round = 0; round < rounds; ++round) {
#ifdef D128_CAUSAL
#ifndef REGISTER_CONTROL
#pragma unroll
        for (int slot = 0; slot < kSlots; ++slot) {
            spill[slot] = state + static_cast<std::uint32_t>(slot * 17 + round);
            asm volatile("" ::: "memory");
        }
#pragma unroll
        for (int slot = 0; slot < 7; ++slot) {
            spill[slot] = state + static_cast<std::uint32_t>(slot * 29 + round);
            asm volatile("" ::: "memory");
        }
#pragma unroll
        for (int slot = 0; slot < kSlots; ++slot) {
            const std::uint32_t value = spill[slot];
            asm volatile("" ::: "memory");
            state = (state << 5) ^ (state >> 2) ^ value;
        }
#pragma unroll
        for (int slot = 0; slot < 9; ++slot) {
            const std::uint32_t value = spill[slot];
            asm volatile("" ::: "memory");
            state = (state << 5) ^ (state >> 2) ^ value;
        }
#else
        std::uint32_t values[kSlots];
#pragma unroll
        for (int slot = 0; slot < kSlots; ++slot) {
            values[slot] = (
                state + static_cast<std::uint32_t>(slot * 17 + round));
            asm volatile("" : "+r"(values[slot]));
        }
#pragma unroll
        for (int slot = 0; slot < 7; ++slot) {
            values[slot] = (
                state + static_cast<std::uint32_t>(slot * 29 + round));
            asm volatile("" : "+r"(values[slot]));
        }
#pragma unroll
        for (int slot = 0; slot < kSlots; ++slot) {
            const std::uint32_t value = values[slot];
            state = (state << 5) ^ (state >> 2) ^ value;
        }
#pragma unroll
        for (int slot = 0; slot < 9; ++slot) {
            const std::uint32_t value = values[slot];
            state = (state << 5) ^ (state >> 2) ^ value;
        }
#endif
#else
#ifndef REGISTER_CONTROL
#pragma unroll
        for (int slot = 0; slot < kSlots; ++slot) {
            spill[slot] = state + static_cast<std::uint32_t>(slot * 17 + round);
            asm volatile("" ::: "memory");
        }
#pragma unroll
        for (int slot = 0; slot < kSlots; ++slot) {
            const std::uint32_t value = spill[slot];
            asm volatile("" ::: "memory");
            state = (state << 5) ^ (state >> 2) ^ value;
        }
#else
        const std::uint32_t v0 = state + round;
        const std::uint32_t v1 = state + 17 + round;
        const std::uint32_t v2 = state + 34 + round;
        const std::uint32_t v3 = state + 51 + round;
        const std::uint32_t v4 = state + 68 + round;
        const std::uint32_t v5 = state + 85 + round;
        const std::uint32_t v6 = state + 102 + round;
        const std::uint32_t v7 = state + 119 + round;
        const std::uint32_t v8 = state + 136 + round;
        const std::uint32_t v9 = state + 153 + round;
        const std::uint32_t v10 = state + 170 + round;
        const std::uint32_t v11 = state + 187 + round;
        const std::uint32_t v12 = state + 204 + round;
#define MIX(value) state = (state << 5) ^ (state >> 2) ^ value
        MIX(v0); MIX(v1); MIX(v2); MIX(v3); MIX(v4); MIX(v5); MIX(v6);
        MIX(v7); MIX(v8); MIX(v9); MIX(v10); MIX(v11); MIX(v12);
#undef MIX
#endif
#endif
    }
    if (threadIdx.x == 0) {
        checksums[blockIdx.x] = state;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        ends[blockIdx.x] = clock64();
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s DEVICE TRIALS\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device = std::stoi(argv[1]);
    const int trials = std::stoi(argv[2]);
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp property{};
    CUDA_CHECK(cudaGetDeviceProperties(&property, device));
    if (std::string(property.name) != "NVIDIA A800-SXM4-80GB"
            || property.multiProcessorCount != kSms) {
        std::fprintf(stderr, "unexpected device contract\n");
        return EXIT_FAILURE;
    }
    int resident = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, local_spill_round, kThreads, 0));
    constexpr int kMaxBlocks = 2 * kSms;
    std::uint64_t *starts, *ends;
    unsigned int* smids;
    std::uint32_t* checksums;
    CUDA_CHECK(cudaMalloc(&starts, kMaxBlocks * sizeof(*starts)));
    CUDA_CHECK(cudaMalloc(&ends, kMaxBlocks * sizeof(*ends)));
    CUDA_CHECK(cudaMalloc(&smids, kMaxBlocks * sizeof(*smids)));
    CUDA_CHECK(cudaMalloc(&checksums, kMaxBlocks * sizeof(*checksums)));
    std::printf("trial,grid,rounds,resident_limit,active_sms,min_ctas,max_ctas,median_block_cycles,max_sm_span_cycles,checksum\n");
    for (int warmup = 0; warmup < 3; ++warmup) {
        local_spill_round<<<kSms, kThreads>>>(64, starts, ends, smids, checksums);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const int grids[] = {kSms, 2 * kSms};
    const int round_values[] = {0, 1, 4, 16, 64, 256};
    for (int trial = 0; trial < trials; ++trial) {
        for (int grid : grids) {
            for (int rounds : round_values) {
                local_spill_round<<<grid, kThreads>>>(
                    rounds, starts, ends, smids, checksums);
                CUDA_CHECK(cudaDeviceSynchronize());
                std::vector<std::uint64_t> hstart(grid), hend(grid);
                std::vector<unsigned int> hsmid(grid);
                std::vector<std::uint32_t> hchecksum(grid);
                CUDA_CHECK(cudaMemcpy(hstart.data(), starts, grid * sizeof(*starts), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hend.data(), ends, grid * sizeof(*ends), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hsmid.data(), smids, grid * sizeof(*smids), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hchecksum.data(), checksums, grid * sizeof(*checksums), cudaMemcpyDeviceToHost));
                std::vector<std::uint64_t> block_cycles;
                std::vector<std::uint64_t> sm_start(kSms, UINT64_MAX), sm_end(kSms, 0);
                std::vector<int> counts(kSms, 0);
                std::uint32_t checksum = 0;
                for (int block = 0; block < grid; ++block) {
                    if (hsmid[block] >= kSms || hend[block] <= hstart[block]) {
                        std::fprintf(stderr, "invalid timing record\n");
                        return EXIT_FAILURE;
                    }
                    const int sm = static_cast<int>(hsmid[block]);
                    ++counts[sm];
                    sm_start[sm] = std::min(sm_start[sm], hstart[block]);
                    sm_end[sm] = std::max(sm_end[sm], hend[block]);
                    block_cycles.push_back(hend[block] - hstart[block]);
                    checksum ^= hchecksum[block];
                }
                const int active = static_cast<int>(std::count_if(
                    counts.begin(), counts.end(), [](int value) { return value > 0; }));
                const int minimum = *std::min_element(counts.begin(), counts.end());
                const int maximum = *std::max_element(counts.begin(), counts.end());
                std::uint64_t max_span = 0;
                for (int sm = 0; sm < kSms; ++sm) {
                    if (counts[sm] > 0) {
                        max_span = std::max(max_span, sm_end[sm] - sm_start[sm]);
                    }
                }
                std::sort(block_cycles.begin(), block_cycles.end());
                const auto med = block_cycles[block_cycles.size() / 2];
                std::printf("%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%u\n",
                    trial, grid, rounds, resident, active, minimum, maximum,
                    static_cast<unsigned long long>(med),
                    static_cast<unsigned long long>(max_span), checksum);
            }
        }
    }
    CUDA_CHECK(cudaFree(checksums));
    CUDA_CHECK(cudaFree(smids));
    CUDA_CHECK(cudaFree(ends));
    CUDA_CHECK(cudaFree(starts));
}
