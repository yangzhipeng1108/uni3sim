/* Independent cold B-descriptor replay for A800 grouped FusedMoE.
 *
 * This probe deliberately excludes A, tensor-core work, routing, output, and
 * all production timings.  It preserves only the static Triton pid swizzle,
 * the per-expert M-tile source map, BF16 K7168/BN64 B descriptor, 128-thread
 * CTA, and five-CTA/SM shared-memory admission.  A 256-MiB L2 eviction pass
 * immediately precedes the selected replay kernel, so the NCU counter result
 * describes one cold source B stream rather than a warmed repeat.
 */
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do {                                             \
    cudaError_t error = (call);                                           \
    if (error != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, \
                cudaGetErrorString(error));                              \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while (0)

constexpr int kBlockN = 64;
constexpr int kK = 7168;
constexpr int kGroupSizeM = 8;
constexpr int kThreads = 128;
constexpr int kExperts = 256;
constexpr size_t kEvictionBytes = 256ull * 1024ull * 1024ull;
// nvcc reserves one 1-KiB static shared segment for the checksum epilogue.
// Request 31 KiB dynamically so the allocated 32 KiB block admits exactly
// five CTAs in GA100's 163-KiB per-SM shared-memory budget.
constexpr size_t kDynamicSharedBytes = 31ull * 1024ull;

struct Arguments {
    int n;
    int active_m_tiles;
    int padded_m_tiles;
    int num_pid_n;
    int* expert_by_m;
};

static int parse_positive(const char* text, const char* label) {
    char* end = nullptr;
    const long value = strtol(text, &end, 10);
    if (*text == '\0' || *end != '\0' || value <= 0 || value > INT32_MAX) {
        fprintf(stderr, "%s must be a positive integer\\n", label);
        exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

static int* parse_tile_counts(const char* text, int* active_m_tiles) {
    int* expert_by_m = nullptr;
    CUDA_CHECK(cudaMallocHost(&expert_by_m, (size_t)kExperts * 10 * sizeof(int)));
    int written = 0;
    const char* cursor = text;
    for (int expert = 0; expert < kExperts; ++expert) {
        char* end = nullptr;
        const long count = strtol(cursor, &end, 10);
        if (end == cursor || count <= 0 || count > 10) {
            fprintf(stderr, "tile count %d must be in [1,10]\\n", expert);
            exit(EXIT_FAILURE);
        }
        for (int tile = 0; tile < count; ++tile) expert_by_m[written++] = expert;
        if (expert + 1 == kExperts) {
            if (*end != '\0') {
                fprintf(stderr, "tile count list has trailing input\\n");
                exit(EXIT_FAILURE);
            }
        } else {
            if (*end != ',') {
                fprintf(stderr, "tile count list must have 256 values\\n");
                exit(EXIT_FAILURE);
            }
            cursor = end + 1;
        }
    }
    *active_m_tiles = written;
    return expert_by_m;
}

__global__ void initialize_words(uint4* output, size_t count) {
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < count;
         index += (size_t)gridDim.x * blockDim.x) {
        const unsigned value = (unsigned)index * 747796405u + 2891336453u;
        output[index] = make_uint4(value, value ^ 0x9e3779b9u,
                                   value + 0x7f4a7c15u, ~value);
    }
}

__global__ void evict_l2(const uint4* source, unsigned* sink, size_t count) {
    unsigned checksum = 0;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < count;
         index += (size_t)gridDim.x * blockDim.x) {
        const uint4 value = source[index];
        checksum ^= value.x ^ value.y ^ value.z ^ value.w;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        checksum ^= __shfl_xor_sync(0xffffffffu, checksum, offset);
    }
    if ((threadIdx.x & 31) == 0) atomicXor(sink, checksum);
}

__device__ __forceinline__ uint4 load_l2_only(const uint4* pointer) {
    uint4 value;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(pointer));
    return value;
}

extern "C" __global__ void profile_e256_cold_weight_replay(
        const uint4* weights, const int* expert_by_m, unsigned* sink,
        int n, int active_m_tiles, int padded_m_tiles, int num_pid_n) {
    extern __shared__ uint4 admission_scratch[];
    const int pid = (int)blockIdx.x;
    const int descriptors_per_group = kGroupSizeM * num_pid_n;
    const int group = pid / descriptors_per_group;
    const int first_m = group * kGroupSizeM;
    const int remaining_m = padded_m_tiles - first_m;
    const int group_size_m = remaining_m < kGroupSizeM ? remaining_m : kGroupSizeM;
    const int in_group = pid % descriptors_per_group;
    const int pid_m = first_m + in_group % group_size_m;
    const int pid_n = in_group / group_size_m;
    if (pid_m >= active_m_tiles) return;
    const int expert = expert_by_m[pid_m];
    if (expert < 0 || expert >= kExperts || pid_n < 0 || pid_n >= num_pid_n) return;
    const size_t words_per_tile = (size_t)kK * kBlockN * sizeof(uint16_t) / sizeof(uint4);
    const size_t words_per_expert = (size_t)n * kK * sizeof(uint16_t) / sizeof(uint4);
    const uint4* tile = weights + (size_t)expert * words_per_expert
        + (size_t)pid_n * words_per_tile;
    unsigned checksum = 0;
    for (size_t word = threadIdx.x; word < words_per_tile; word += kThreads) {
        const uint4 value = load_l2_only(tile + word);
        checksum ^= value.x ^ value.y ^ value.z ^ value.w;
    }
    admission_scratch[threadIdx.x] = make_uint4(checksum, checksum ^ 0x5a5a5a5au,
                                                  checksum + 1u, ~checksum);
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint4 value = admission_scratch[0];
        atomicXor(sink, value.x ^ value.y ^ value.z ^ value.w);
    }
}

static Arguments parse_arguments(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s N PADDED_M TILE_COUNTS_CSV\\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    const int n = parse_positive(argv[1], "N");
    if (n % kBlockN != 0 || n == 4096) {
        fprintf(stderr, "N must be a non-production positive multiple of 64\\n");
        exit(EXIT_FAILURE);
    }
    const int padded_m_tiles = parse_positive(argv[2], "PADDED_M");
    if (padded_m_tiles % kGroupSizeM != 4 || padded_m_tiles < kExperts) {
        fprintf(stderr, "PADDED_M must retain the source final GSM8 tail\\n");
        exit(EXIT_FAILURE);
    }
    int active_m_tiles = 0;
    int* host_expert_by_m = parse_tile_counts(argv[3], &active_m_tiles);
    if (active_m_tiles >= padded_m_tiles) {
        fprintf(stderr, "PADDED_M must retain inactive launch rows\\n");
        exit(EXIT_FAILURE);
    }
    int* device_expert_by_m = nullptr;
    CUDA_CHECK(cudaMalloc(&device_expert_by_m, (size_t)active_m_tiles * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(device_expert_by_m, host_expert_by_m,
                           (size_t)active_m_tiles * sizeof(int),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaFreeHost(host_expert_by_m));
    return {n, active_m_tiles, padded_m_tiles, n / kBlockN, device_expert_by_m};
}

int main(int argc, char** argv) {
    Arguments arguments = parse_arguments(argc, argv);
    const size_t weight_bytes = (size_t)kExperts * arguments.n * kK * sizeof(uint16_t);
    uint4* weights = nullptr;
    uint4* eviction = nullptr;
    unsigned* sink = nullptr;
    CUDA_CHECK(cudaMalloc(&weights, weight_bytes));
    CUDA_CHECK(cudaMalloc(&eviction, kEvictionBytes));
    CUDA_CHECK(cudaMalloc(&sink, sizeof(unsigned)));
    initialize_words<<<108, 256>>>(weights, weight_bytes / sizeof(uint4));
    initialize_words<<<108, 256>>>(eviction, kEvictionBytes / sizeof(uint4));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(sink, 0, sizeof(unsigned)));
    evict_l2<<<108, 256>>>(eviction, sink, kEvictionBytes / sizeof(uint4));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFuncSetAttribute(profile_e256_cold_weight_replay,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)kDynamicSharedBytes));
    const int grid = arguments.padded_m_tiles * arguments.num_pid_n;
    profile_e256_cold_weight_replay<<<grid, kThreads, kDynamicSharedBytes>>>(
        weights, arguments.expert_by_m, sink, arguments.n,
        arguments.active_m_tiles, arguments.padded_m_tiles, arguments.num_pid_n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned host_sink = 0;
    CUDA_CHECK(cudaMemcpy(&host_sink, sink, sizeof(host_sink), cudaMemcpyDeviceToHost));
    const size_t logical_b_bytes = (size_t)arguments.active_m_tiles
        * arguments.num_pid_n * kK * kBlockN * sizeof(uint16_t);
    printf("e256_cold_weight_replay,%d,%d,%d,%d,%d,%zu,%u\\n", arguments.n,
           arguments.active_m_tiles, arguments.padded_m_tiles,
           arguments.num_pid_n, grid, logical_b_bytes, host_sink);
    CUDA_CHECK(cudaFree(arguments.expert_by_m));
    CUDA_CHECK(cudaFree(sink));
    CUDA_CHECK(cudaFree(eviction));
    CUDA_CHECK(cudaFree(weights));
    return EXIT_SUCCESS;
}
