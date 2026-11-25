/*
 * Independent A800 BF16 indirect-row gather mechanism benchmark.
 *
 * This does not execute a GEMM or a production Fused-MoE operator.  It
 * isolates the activation operand's indexed global-to-shared transfer for the
 * two frozen launch signatures used by the production kernels.  The timed
 * logical bytes are only the gathered BF16 activation bytes.
 */
#include <cuda_runtime.h>

#include <errno.h>
#include <limits.h>
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

constexpr int K = 2048;
constexpr int ROUTED_ROWS = 262144;
constexpr int BLOCK_THREADS = 128;

static bool parse_int(const char* text, int* output) {
    errno = 0;
    char* end = nullptr;
    const long value = strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
            || value < INT_MIN || value > INT_MAX) {
        return false;
    }
    *output = (int)value;
    return true;
}

__global__ void initialize_words(uint32_t* words, size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    uint32_t value = (uint32_t)index * 747796405u + 2891336453u;
    value = ((value >> ((value >> 28u) + 4u)) ^ value) * 277803737u;
    words[index] = (value >> 22u) ^ value;
}

template<int BM, int BK>
__global__ void gather_to_shared(
        const uint4* __restrict__ activation,
        const int* __restrict__ row_indices,
        uint32_t* __restrict__ output) {
    constexpr int WORDS_PER_ROW = BK * (int)sizeof(uint16_t) / (int)sizeof(uint4);
    constexpr int WORDS_PER_TILE = BM * WORDS_PER_ROW;
    constexpr int ACTIVATION_WORDS_PER_ROW =
        K * (int)sizeof(uint16_t) / (int)sizeof(uint4);
    constexpr int GROUPS = K / BK;
    __shared__ uint4 tile[WORDS_PER_TILE];
    __shared__ uint32_t warp_checksums[4];
    const int thread = threadIdx.x;
    const int routed_row_start = blockIdx.x * BM;
    uint32_t checksum = 0;
    for (int group = 0; group < GROUPS; ++group) {
        for (int word = thread; word < WORDS_PER_TILE;
                word += BLOCK_THREADS) {
            const int row = word / WORDS_PER_ROW;
            const int column_word = word % WORDS_PER_ROW;
            const int source_row = row_indices[routed_row_start + row];
            tile[word] = activation[
                (size_t)source_row * ACTIVATION_WORDS_PER_ROW
                + group * WORDS_PER_ROW + column_word];
        }
        __syncthreads();
        for (int word = thread; word < WORDS_PER_TILE;
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
        if (thread == 0) output[blockIdx.x] = checksum;
    }
}

static void fill_indices(int* indices, int tokens, bool scattered) {
    int* permutation = static_cast<int*>(malloc((size_t)tokens * sizeof(int)));
    if (permutation == nullptr) {
        fprintf(stderr, "host permutation allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (int index = 0; index < tokens; ++index) permutation[index] = index;
    if (scattered) {
        uint32_t state = 0x9e3779b9u;
        for (int index = tokens - 1; index > 0; --index) {
            state = state * 1664525u + 1013904223u;
            int other = (int)(state % (uint32_t)(index + 1));
            int temporary = permutation[index];
            permutation[index] = permutation[other];
            permutation[other] = temporary;
        }
    }
    for (int row = 0; row < ROUTED_ROWS; ++row) {
        indices[row] = permutation[row % tokens];
    }
    free(permutation);
}

template<int BM, int BK>
static void run_case(const char* name, int tokens, bool scattered, int samples) {
    static_assert(K % BK == 0, "K must be divisible by BK");
    static_assert(ROUTED_ROWS % BM == 0, "routed rows must divide the grid");
    constexpr int GRID = ROUTED_ROWS / BM;
    const size_t activation_bytes =
        (size_t)tokens * K * sizeof(uint16_t);
    const size_t activation_words = activation_bytes / sizeof(uint32_t);
    uint32_t* activation;
    int* row_indices;
    uint32_t* output;
    CUDA_CHECK(cudaMalloc(&activation, activation_bytes));
    CUDA_CHECK(cudaMalloc(&row_indices, (size_t)ROUTED_ROWS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&output, (size_t)GRID * sizeof(uint32_t)));
    initialize_words<<<(activation_words + 255) / 256, 256>>>(
        activation, activation_words);
    CUDA_CHECK(cudaGetLastError());
    int* host_indices = static_cast<int*>(
        malloc((size_t)ROUTED_ROWS * sizeof(int)));
    if (host_indices == nullptr) {
        fprintf(stderr, "host index allocation failed\n");
        exit(EXIT_FAILURE);
    }
    fill_indices(host_indices, tokens, scattered);
    CUDA_CHECK(cudaMemcpy(
        row_indices, host_indices, (size_t)ROUTED_ROWS * sizeof(int),
        cudaMemcpyHostToDevice));
    free(host_indices);
    for (int warmup = 0; warmup < 3; ++warmup) {
        gather_to_shared<BM, BK><<<GRID, BLOCK_THREADS>>>(
            reinterpret_cast<const uint4*>(activation), row_indices, output);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start;
    cudaEvent_t end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    const double logical_bytes =
        (double)ROUTED_ROWS * K * sizeof(uint16_t);
    for (int sample = 0; sample < samples; ++sample) {
        CUDA_CHECK(cudaEventRecord(start));
        gather_to_shared<BM, BK><<<GRID, BLOCK_THREADS>>>(
            reinterpret_cast<const uint4*>(activation), row_indices, output);
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
        if (!(milliseconds > 0.0f)) {
            fprintf(stderr, "non-positive event time\n");
            exit(EXIT_FAILURE);
        }
        const double bandwidth = logical_bytes / (milliseconds * 1.0e6);
        printf("%s,%d,%d,%d,%d,%d,%d,%.9g,%.9g\n",
               name, sample, BM, BK, tokens, scattered ? 1 : 0, GRID,
               milliseconds, bandwidth);
    }
    uint32_t checksum = 0;
    CUDA_CHECK(cudaMemcpy(
        &checksum, output, sizeof(checksum), cudaMemcpyDeviceToHost));
    fprintf(stderr, "validated case=%s output0=%u\n", name, checksum);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(activation));
    CUDA_CHECK(cudaFree(row_indices));
    CUDA_CHECK(cudaFree(output));
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s CASE_ID SAMPLES\n", argv[0]);
        return EXIT_FAILURE;
    }
    int case_id = 0;
    int samples = 0;
    if (!parse_int(argv[1], &case_id) || !parse_int(argv[2], &samples)
            || samples <= 0) {
        fprintf(stderr, "CASE_ID and positive SAMPLES must be valid integers\n");
        return EXIT_FAILURE;
    }
    if (case_id == 0) {
        run_case<16, 64>("scatter_bm16_bk64_l2", 4096, true, samples);
    } else if (case_id == 1) {
        run_case<64, 32>("scatter_bm64_bk32_l2", 4096, true, samples);
    } else if (case_id == 2) {
        run_case<16, 64>("contiguous_bm16_bk64_l2", 4096, false, samples);
    } else if (case_id == 3) {
        run_case<16, 64>("scatter_bm16_bk64_gt_l2", 32768, true, samples);
    } else {
        fprintf(stderr, "CASE_ID must be 0, 1, 2, or 3\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
