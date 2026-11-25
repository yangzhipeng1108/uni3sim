/* Independent A800 grouped-MoE B-panel fanout probe.
 *
 * This is not a GEMM and has no tensor-core, activation, routing, or output
 * matrix work.  Every CTA reads one complete BF16 BK32 x BN64 B panel over
 * K=2048.  Logical program IDs use the vLLM Triton GROUP_SIZE_M=8 mapping.
 * Consecutive pid_m values map to expert_id=pid_m/fanout, so the same expert
 * panel is requested by exactly 1, 2, 4, or 8 CTAs while the unique B
 * footprint remains fixed at 32 MiB.  A 256-MiB sweep cold-evicts every cell
 * before the one profiled launch.
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
constexpr int NUM_PID_N = 1;
constexpr int EXPERTS = 128;
constexpr int BLOCK_THREADS = 128;
constexpr size_t B_PANEL_BYTES =
    (size_t)K * BLOCK_N * sizeof(uint16_t);
constexpr size_t UNIQUE_B_BYTES = (size_t)EXPERTS * B_PANEL_BYTES;
constexpr size_t EVICTION_BYTES = 256ull * 1024ull * 1024ull;

constexpr int FANOUTS[] = {1, 2, 4, 8};

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
    if ((threadIdx.x & 31) == 0) atomicXor(result, checksum);
}

extern "C" __global__ void profile_fused_moe_b_panel_fanout(
        const uint4* weights, uint32_t* output, int num_pid_m, int fanout) {
    constexpr int WORDS_PER_K_ROW = BLOCK_N * 2 / (int)sizeof(uint4);
    constexpr int WORDS_PER_GROUP = BLOCK_K * WORDS_PER_K_ROW;
    constexpr int GROUPS_K = K / BLOCK_K;
    __shared__ uint4 panel[WORDS_PER_GROUP];
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
    const int expert_id = pid_m / fanout;
    const int thread = threadIdx.x;
    uint32_t checksum = 0;

    for (int group_k = 0; group_k < GROUPS_K; ++group_k) {
        for (int word = thread; word < WORDS_PER_GROUP;
             word += BLOCK_THREADS) {
            const size_t panel_word =
                (size_t)expert_id * B_PANEL_BYTES / sizeof(uint4);
            panel[word] = weights[
                panel_word + group_k * WORDS_PER_GROUP + word];
        }
        __syncthreads();
        for (int word = thread; word < WORDS_PER_GROUP;
             word += BLOCK_THREADS) {
            const uint4 value = panel[word];
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
        if (thread == 0) output[pid] = checksum ^ (uint32_t)pid_n;
    }
}

static void run_case(int fanout) {
    const int num_pid_m = EXPERTS * fanout;
    const int grid = num_pid_m * NUM_PID_N;
    const size_t logical_b_bytes = (size_t)grid * B_PANEL_BYTES;
    uint32_t* weights;
    uint32_t* eviction;
    uint32_t* output;
    CUDA_CHECK(cudaMalloc(&weights, UNIQUE_B_BYTES));
    CUDA_CHECK(cudaMalloc(&eviction, EVICTION_BYTES));
    CUDA_CHECK(cudaMalloc(&output, (size_t)grid * sizeof(uint32_t)));
    initialize_words<<<(UNIQUE_B_BYTES / 4 + 255) / 256, 256>>>(
        weights, UNIQUE_B_BYTES / 4);
    initialize_words<<<(EVICTION_BYTES / 4 + 255) / 256, 256>>>(
        eviction, EVICTION_BYTES / 4);
    CUDA_CHECK(cudaMemset(output, 0, (size_t)grid * sizeof(uint32_t)));
    evict_l2<<<108, 256>>>(
        reinterpret_cast<const uint4*>(eviction), output,
        EVICTION_BYTES / sizeof(uint4));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(eviction));

    profile_fused_moe_b_panel_fanout<<<grid, BLOCK_THREADS>>>(
        reinterpret_cast<const uint4*>(weights), output, num_pid_m, fanout);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t checksum = 0;
    CUDA_CHECK(cudaMemcpy(
        &checksum, output, sizeof(checksum), cudaMemcpyDeviceToHost));
    printf("b_panel_fanout_%d,%d,%d,%d,%d,%d,%d,%d,%d,%zu,%zu,"
           "cold_evicted,%u\n",
           fanout, fanout, EXPERTS, num_pid_m, grid, BLOCK_M, BLOCK_N,
           BLOCK_K, GROUP_SIZE_M, UNIQUE_B_BYTES, logical_b_bytes, checksum);
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
            || case_id_value >= (long)(sizeof(FANOUTS) / sizeof(FANOUTS[0]))) {
        fprintf(stderr, "CASE_ID must be between 0 and 3\n");
        return EXIT_FAILURE;
    }
    const int case_id = static_cast<int>(case_id_value);
    run_case(FANOUTS[case_id]);
    return EXIT_SUCCESS;
}
