/* Independent A800 grouped-MoE descriptor-residency probe.
 *
 * Every profiled CTA replays one source-shaped BM64/BN64/BK32 operand
 * descriptor.  Logical program IDs use the Triton GROUP_SIZE_M=8 swizzle:
 * eight M descriptors are visited before advancing N.  The probe performs no
 * tensor-core work, routing, output matrix, or production-operator timing.
 * Its only purpose is to expose L2 request, L2 hit/miss, and DRAM read bytes
 * for one 32-MiB resident geometry and two geometries above the 40-MiB A800
 * L2 capacity.  A fourth cell cold-evicts the resident geometry as a control.
 */
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do {                                             \
    cudaError_t error = (call);                                           \
    if (error != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(error));            \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while (0)

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 32;
constexpr int GROUP_SIZE_M = 8;
constexpr int K = 2048;
constexpr int N = 960;
constexpr int NUM_PID_N = N / BLOCK_N;
constexpr int BLOCK_THREADS = 128;
constexpr size_t EVICTION_BYTES = 256ull * 1024ull * 1024ull;

struct CaseDefinition {
    const char* name;
    int num_pid_m;
    bool cold_control;
};

constexpr CaseDefinition CASES[] = {
    {"source_gsm8_resident_32mib", 8, false},
    {"source_gsm8_over_l2_48mib", 12, false},
    {"source_gsm8_over_l2_64mib", 16, false},
    {"source_gsm8_cold_32mib_control", 8, true},
};

__global__ void initialize_words(uint32_t* words, size_t count) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    uint32_t value = (uint32_t)index * 747796405u + 2891336453u;
    value = ((value >> ((value >> 28u) + 4u)) ^ value) * 277803737u;
    words[index] = (value >> 22u) ^ value;
}

__global__ void evict_l2(const uint4* source, uint32_t* result,
                         size_t word_count) {
    uint32_t checksum = 0;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < word_count;
         index += (size_t)gridDim.x * blockDim.x) {
        const uint4 value = source[index];
        checksum ^= value.x ^ value.y ^ value.z ^ value.w;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        checksum ^= __shfl_xor_sync(0xffffffffu, checksum, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicXor(result, checksum);
    }
}

__device__ __forceinline__ void replay_descriptor(
        const uint4* activation, const uint4* weights, uint32_t* output,
        int num_pid_m) {
    constexpr int A_WORDS_PER_ROW = BLOCK_K * 2 / (int)sizeof(uint4);
    constexpr int A_WORDS = BLOCK_M * A_WORDS_PER_ROW;
    constexpr int B_WORDS_PER_ROW = BLOCK_N * 2 / (int)sizeof(uint4);
    constexpr int B_WORDS = BLOCK_K * B_WORDS_PER_ROW;
    constexpr int WORDS_PER_GROUP = A_WORDS + B_WORDS;
    constexpr int A_WORDS_PER_SOURCE_ROW = K * 2 / (int)sizeof(uint4);
    constexpr int B_WORDS_PER_SOURCE_ROW = N * 2 / (int)sizeof(uint4);
    constexpr int GROUPS_K = K / BLOCK_K;
    __shared__ uint4 tile[WORDS_PER_GROUP];
    __shared__ uint32_t warp_checksums[4];

    const int pid = (int)blockIdx.x;
    const int descriptors_per_full_group = GROUP_SIZE_M * NUM_PID_N;
    const int group_id = pid / descriptors_per_full_group;
    const int first_pid_m = group_id * GROUP_SIZE_M;
    const int remaining_pid_m = num_pid_m - first_pid_m;
    const int group_size_m =
        remaining_pid_m < GROUP_SIZE_M ? remaining_pid_m : GROUP_SIZE_M;
    const int pid_in_group = pid % descriptors_per_full_group;
    const int pid_m = first_pid_m + pid_in_group % group_size_m;
    const int pid_n = pid_in_group / group_size_m;
    const int thread = threadIdx.x;
    uint32_t checksum = 0;

    for (int group_k = 0; group_k < GROUPS_K; ++group_k) {
        for (int word = thread; word < A_WORDS; word += BLOCK_THREADS) {
            const int row = word / A_WORDS_PER_ROW;
            const int column_word = word % A_WORDS_PER_ROW;
            tile[word] = activation[
                ((size_t)pid_m * BLOCK_M + row) * A_WORDS_PER_SOURCE_ROW
                + group_k * A_WORDS_PER_ROW + column_word];
        }
        for (int word = thread; word < B_WORDS; word += BLOCK_THREADS) {
            const int row_k = word / B_WORDS_PER_ROW;
            const int column_word = word % B_WORDS_PER_ROW;
            tile[A_WORDS + word] = weights[
                ((size_t)pid_m * K + group_k * BLOCK_K + row_k)
                    * B_WORDS_PER_SOURCE_ROW
                + pid_n * B_WORDS_PER_ROW + column_word];
        }
        __syncthreads();
        for (int word = thread; word < WORDS_PER_GROUP;
             word += BLOCK_THREADS) {
            const uint4 value = tile[word];
            checksum ^= value.x ^ value.y ^ value.z ^ value.w;
        }
        __syncthreads();
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        checksum ^= __shfl_xor_sync(0xffffffffu, checksum, offset);
    }
    if ((thread & 31) == 0) warp_checksums[thread / 32] = checksum;
    __syncthreads();
    if (thread < 4) {
        checksum = warp_checksums[thread];
        checksum ^= __shfl_xor_sync(0x0000000fu, checksum, 2);
        checksum ^= __shfl_xor_sync(0x0000000fu, checksum, 1);
        if (thread == 0) output[pid] = checksum;
    }
}

extern "C" __global__ void warm_fused_moe_descriptor_replay(
        const uint4* activation, const uint4* weights, uint32_t* output,
        int num_pid_m) {
    replay_descriptor(activation, weights, output, num_pid_m);
}

extern "C" __global__ void profile_fused_moe_descriptor_replay(
        const uint4* activation, const uint4* weights, uint32_t* output,
        int num_pid_m) {
    replay_descriptor(activation, weights, output, num_pid_m);
}

static void run_case(const CaseDefinition& definition) {
    const int grid = definition.num_pid_m * NUM_PID_N;
    const size_t activation_bytes =
        (size_t)definition.num_pid_m * BLOCK_M * K * sizeof(uint16_t);
    const size_t weight_bytes =
        (size_t)definition.num_pid_m * K * N * sizeof(uint16_t);
    const size_t logical_descriptor_bytes =
        (size_t)grid * K * (BLOCK_M + BLOCK_N) * sizeof(uint16_t);
    uint32_t* activation;
    uint32_t* weights;
    uint32_t* output;
    CUDA_CHECK(cudaMalloc(&activation, activation_bytes));
    CUDA_CHECK(cudaMalloc(&weights, weight_bytes));
    CUDA_CHECK(cudaMalloc(&output, (size_t)grid * sizeof(uint32_t)));
    initialize_words<<<(activation_bytes / 4 + 255) / 256, 256>>>(
        activation, activation_bytes / 4);
    initialize_words<<<(weight_bytes / 4 + 255) / 256, 256>>>(
        weights, weight_bytes / 4);
    CUDA_CHECK(cudaGetLastError());

    if (definition.cold_control) {
        uint32_t* eviction;
        CUDA_CHECK(cudaMalloc(&eviction, EVICTION_BYTES));
        initialize_words<<<(EVICTION_BYTES / 4 + 255) / 256, 256>>>(
            eviction, EVICTION_BYTES / 4);
        CUDA_CHECK(cudaMemset(output, 0, sizeof(uint32_t)));
        evict_l2<<<108, 256>>>(
            reinterpret_cast<const uint4*>(eviction), output,
            EVICTION_BYTES / sizeof(uint4));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(eviction));
    } else {
        for (int warmup = 0; warmup < 3; ++warmup) {
            warm_fused_moe_descriptor_replay<<<grid, BLOCK_THREADS>>>(
                reinterpret_cast<const uint4*>(activation),
                reinterpret_cast<const uint4*>(weights), output,
                definition.num_pid_m);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    profile_fused_moe_descriptor_replay<<<grid, BLOCK_THREADS>>>(
        reinterpret_cast<const uint4*>(activation),
        reinterpret_cast<const uint4*>(weights), output,
        definition.num_pid_m);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t checksum = 0;
    CUDA_CHECK(cudaMemcpy(
        &checksum, output, sizeof(checksum), cudaMemcpyDeviceToHost));
    printf("%s,%d,%d,%d,%zu,%zu,%zu,%zu,%s,%u\n",
           definition.name, definition.num_pid_m, NUM_PID_N, grid,
           activation_bytes, weight_bytes, activation_bytes + weight_bytes,
           logical_descriptor_bytes,
           definition.cold_control ? "cold_evicted" : "source_warmed",
           checksum);
    CUDA_CHECK(cudaFree(activation));
    CUDA_CHECK(cudaFree(weights));
    CUDA_CHECK(cudaFree(output));
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s CASE_ID\n", argv[0]);
        return EXIT_FAILURE;
    }
    char* end = nullptr;
    const long case_id_value = strtol(argv[1], &end, 10);
    if (*argv[1] == '\0' || *end != '\0' || case_id_value < 0
            || case_id_value >= (long)(sizeof(CASES) / sizeof(CASES[0]))) {
        fprintf(stderr, "CASE_ID must be between 0 and 3\n");
        return EXIT_FAILURE;
    }
    const int case_id = static_cast<int>(case_id_value);
    run_case(CASES[case_id]);
    return EXIT_SUCCESS;
}
