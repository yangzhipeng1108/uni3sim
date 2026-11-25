/* Independent A800 grouped-MoE operand-read mechanism benchmark.
 *
 * The kernel transfers the exact BF16 A and B tiles of each production launch
 * signature into shared memory and consumes them in a checksum.  The BM64
 * case accepts an active-row count matching the production Triton A-operand
 * token mask: inactive A rows contribute zero without issuing an activation
 * load, while the B tile, CTA launch, shared-memory footprint, barriers, and
 * shared-memory consumption remain unchanged.  It contains no tensor-core
 * operation, output matrix, routing pipeline, or production operator timing.
 * Both unique input footprints total 32 MiB at 64 active rows and therefore
 * isolate the 40-MiB L2-resident operand service.
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
constexpr int TOKENS = 4096;
constexpr int GRID = 4096;
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

template<int BM, int BN, int BK, int WEIGHT_TILES>
__global__ void grouped_operand_read(
        const uint4* __restrict__ activation,
        const uint4* __restrict__ weights,
        const int* __restrict__ row_indices,
        uint32_t* __restrict__ output,
        int active_rows) {
    constexpr int A_WORDS_PER_ROW = BK * 2 / (int)sizeof(uint4);
    constexpr int A_WORDS_PER_TILE = BM * A_WORDS_PER_ROW;
    constexpr int B_WORDS_PER_GROUP = BK * BN * 2 / (int)sizeof(uint4);
    constexpr int WORDS_PER_GROUP = A_WORDS_PER_TILE + B_WORDS_PER_GROUP;
    constexpr int A_WORDS_PER_SOURCE_ROW = K * 2 / (int)sizeof(uint4);
    constexpr int B_WORDS_PER_WEIGHT_TILE = K * BN * 2 / (int)sizeof(uint4);
    constexpr int GROUPS = K / BK;
    __shared__ uint4 tile[WORDS_PER_GROUP];
    __shared__ uint32_t warp_checksums[4];
    const int thread = threadIdx.x;
    const int weight_tile = blockIdx.x % WEIGHT_TILES;
    uint32_t checksum = 0;
    for (int group = 0; group < GROUPS; ++group) {
        for (int word = thread; word < A_WORDS_PER_TILE;
                word += BLOCK_THREADS) {
            const int row = word / A_WORDS_PER_ROW;
            const int column_word = word % A_WORDS_PER_ROW;
            uint4 value = make_uint4(0u, 0u, 0u, 0u);
            if (row < active_rows) {
                const int source_row = row_indices[blockIdx.x * BM + row];
                value = activation[
                    (size_t)source_row * A_WORDS_PER_SOURCE_ROW
                    + group * A_WORDS_PER_ROW + column_word];
            }
            tile[word] = value;
        }
        for (int word = thread; word < B_WORDS_PER_GROUP;
                word += BLOCK_THREADS) {
            tile[A_WORDS_PER_TILE + word] = weights[
                (size_t)weight_tile * B_WORDS_PER_WEIGHT_TILE
                + group * B_WORDS_PER_GROUP + word];
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
        if (thread == 0) output[blockIdx.x] = checksum;
    }
}

static void fill_indices(int* indices, int rows) {
    int* permutation = static_cast<int*>(malloc(TOKENS * sizeof(int)));
    if (permutation == nullptr) {
        fprintf(stderr, "host permutation allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (int index = 0; index < TOKENS; ++index) permutation[index] = index;
    uint32_t state = 0x9e3779b9u;
    for (int index = TOKENS - 1; index > 0; --index) {
        state = state * 1664525u + 1013904223u;
        const int other = (int)(state % (uint32_t)(index + 1));
        const int temporary = permutation[index];
        permutation[index] = permutation[other];
        permutation[other] = temporary;
    }
    for (int row = 0; row < rows; ++row) {
        indices[row] = permutation[row % TOKENS];
    }
    free(permutation);
}

template<int BM, int BN, int BK, int WEIGHT_TILES>
static void run_case(const char* name, int active_rows, int samples) {
    if (active_rows < 1 || active_rows > BM) {
        fprintf(stderr, "active rows must be in [1, %d] for %s\n", BM, name);
        exit(EXIT_FAILURE);
    }
    constexpr size_t ACTIVATION_BYTES = (size_t)TOKENS * K * 2;
    constexpr size_t WEIGHT_BYTES = (size_t)WEIGHT_TILES * K * BN * 2;
    constexpr int ROUTED_ROWS = GRID * BM;
    uint32_t* activation;
    uint32_t* weights;
    int* row_indices;
    uint32_t* output;
    CUDA_CHECK(cudaMalloc(&activation, ACTIVATION_BYTES));
    CUDA_CHECK(cudaMalloc(&weights, WEIGHT_BYTES));
    CUDA_CHECK(cudaMalloc(&row_indices, (size_t)ROUTED_ROWS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&output, GRID * sizeof(uint32_t)));
    initialize_words<<<(ACTIVATION_BYTES / 4 + 255) / 256, 256>>>(
        activation, ACTIVATION_BYTES / 4);
    initialize_words<<<(WEIGHT_BYTES / 4 + 255) / 256, 256>>>(
        weights, WEIGHT_BYTES / 4);
    CUDA_CHECK(cudaGetLastError());
    int* host_indices = static_cast<int*>(
        malloc((size_t)ROUTED_ROWS * sizeof(int)));
    if (host_indices == nullptr) {
        fprintf(stderr, "host index allocation failed\n");
        exit(EXIT_FAILURE);
    }
    fill_indices(host_indices, ROUTED_ROWS);
    CUDA_CHECK(cudaMemcpy(
        row_indices, host_indices, (size_t)ROUTED_ROWS * sizeof(int),
        cudaMemcpyHostToDevice));
    free(host_indices);
    for (int warmup = 0; warmup < 3; ++warmup) {
        grouped_operand_read<BM, BN, BK, WEIGHT_TILES>
            <<<GRID, BLOCK_THREADS>>>(
                reinterpret_cast<const uint4*>(activation),
                reinterpret_cast<const uint4*>(weights), row_indices, output,
                active_rows);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start;
    cudaEvent_t end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    const size_t logical_a_bytes =
        (size_t)GRID * K * active_rows * sizeof(uint16_t);
    constexpr size_t LOGICAL_B_BYTES =
        (size_t)GRID * K * BN * sizeof(uint16_t);
    const double logical_bytes = (double)(logical_a_bytes + LOGICAL_B_BYTES);
    for (int sample = 0; sample < samples; ++sample) {
        CUDA_CHECK(cudaEventRecord(start));
        grouped_operand_read<BM, BN, BK, WEIGHT_TILES>
            <<<GRID, BLOCK_THREADS>>>(
                reinterpret_cast<const uint4*>(activation),
                reinterpret_cast<const uint4*>(weights), row_indices, output,
                active_rows);
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
        const double bandwidth = logical_bytes / (milliseconds * 1.0e6);
        printf("%s,%d,%d,%d,%d,%d,%d,%zu,%zu,%zu,%zu,%.9g,%.9g\n",
               name, sample, BM, BN, BK, GRID, active_rows,
               ACTIVATION_BYTES, WEIGHT_BYTES, logical_a_bytes,
               LOGICAL_B_BYTES, milliseconds, bandwidth);
    }
    uint32_t checksum = 0;
    CUDA_CHECK(cudaMemcpy(
        &checksum, output, sizeof(checksum), cudaMemcpyDeviceToHost));
    fprintf(stderr, "validated case=%s active_rows=%d output0=%u\n",
            name, active_rows, checksum);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(activation));
    CUDA_CHECK(cudaFree(weights));
    CUDA_CHECK(cudaFree(row_indices));
    CUDA_CHECK(cudaFree(output));
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s CASE_ID ACTIVE_ROWS SAMPLES\n", argv[0]);
        return EXIT_FAILURE;
    }
    int case_id = 0;
    int active_rows = 0;
    int samples = 0;
    if (!parse_int(argv[1], &case_id)
            || !parse_int(argv[2], &active_rows)
            || !parse_int(argv[3], &samples)
            || samples <= 0) {
        fprintf(stderr,
                "CASE_ID, ACTIVE_ROWS, and positive SAMPLES must be valid integers\n");
        return EXIT_FAILURE;
    }
    if (case_id == 0) {
        run_case<16, 32, 64, 128>(
            "operand_bm16_bn32_bk64", active_rows, samples);
    } else if (case_id == 1) {
        run_case<64, 64, 32, 64>(
            "operand_bm64_bn64_bk32", active_rows, samples);
    } else {
        fprintf(stderr, "CASE_ID must be 0 or 1\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
