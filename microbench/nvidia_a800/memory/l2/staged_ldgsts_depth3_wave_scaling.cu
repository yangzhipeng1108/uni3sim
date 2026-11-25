/* Independent SM80 depth-three LDGSTS finite-startup wave-scaling probe.
 *
 * This is deliberately a copy-only source mechanism: 256-thread CTAs move
 * one private 16-KiB A-shaped and one 32-KiB B-shaped FP16 byte panel through
 * a three-slot cp.async ring.  It has no MMA, output matrix, split-K
 * publication, or production kernel input.  Comparing one resident CTA wave
 * with longer launches determines whether the finite three-group startup is
 * paid per physical CTA wave.
 */
#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

#define CHECK_CUDA(call) do {                                                \
    const cudaError_t error = (call);                                        \
    if (error != cudaSuccess) {                                              \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                     __FILE__, __LINE__, cudaGetErrorString(error));        \
        std::exit(EXIT_FAILURE);                                             \
    }                                                                        \
} while (0)

constexpr int kThreads = 256;
constexpr int kSmCount = 108;
constexpr int kMaxRounds = 1024;
constexpr int kAPanelBytes = 128 * 64 * 2;
constexpr int kBPanelBytes = 64 * 256 * 2;
constexpr int kPanelBytes = kAPanelBytes + kBPanelBytes;
constexpr int kStages = 3;
constexpr int kSharedBytes = kStages * kPanelBytes;
constexpr int kL2BackingPanels = (32 * 1024 * 1024) / kPanelBytes;
constexpr int kGridCounts[] = {108, 216, 432, 864, 1728, 3456, 6912, 11008};
constexpr int kRounds[] = {0, 32, 256, 1024};
constexpr const char* kExpectedDevice = "NVIDIA A800-SXM4-80GB";

int parse_int(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
            || value < std::numeric_limits<int>::min()
            || value > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "%s must be an integer, got %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

__device__ __forceinline__ void cp_async_16(
        void* shared_destination, const void* global_source) {
    const std::uint32_t shared_address = static_cast<std::uint32_t>(
        __cvta_generic_to_shared(shared_destination));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(shared_address), "l"(global_source) : "memory");
}

__device__ __forceinline__ void issue_panel(
        std::uint8_t* destination, const std::uint8_t* source) {
    constexpr int kVectors = kPanelBytes / 16;
    for (int vector = threadIdx.x; vector < kVectors; vector += blockDim.x) {
        cp_async_16(destination + vector * 16, source + vector * 16);
    }
    asm volatile("cp.async.commit_group;\n" :: : "memory");
}

__device__ __forceinline__ const std::uint8_t* panel_source(
        const std::uint8_t* input, int round) {
    const std::size_t panel = (
        static_cast<std::size_t>(blockIdx.x) * kMaxRounds + round)
        % kL2BackingPanels;
    return input + panel * kPanelBytes;
}

__global__ __launch_bounds__(kThreads, 1)
void staged_ldgsts_depth3_wave_scaling(
        const std::uint8_t* __restrict__ input,
        std::uint32_t* __restrict__ checksums, int rounds) {
    extern __shared__ __align__(16) std::uint8_t shared[];
    const int initial_groups = rounds < kStages ? rounds : kStages;
    for (int round = 0; round < initial_groups; ++round) {
        issue_panel(shared + round * kPanelBytes, panel_source(input, round));
    }
    std::uint32_t checksum = 0;
    #pragma unroll 1
    for (int round = 0; round < rounds; ++round) {
        asm volatile("cp.async.wait_group 1;\n" :: : "memory");
        __syncthreads();
        const int slot = round % kStages;
        checksum ^= reinterpret_cast<volatile std::uint32_t*>(
            shared + slot * kPanelBytes)[threadIdx.x];
        __syncthreads();
        const int next = round + kStages;
        if (next < rounds) {
            issue_panel(shared + slot * kPanelBytes, panel_source(input, next));
        }
    }
    asm volatile("cp.async.wait_group 0;\n" :: : "memory");
    if (threadIdx.x == 0) {
        checksums[blockIdx.x] = checksum;
    }
}

void configure_variant() {
    CHECK_CUDA(cudaFuncSetAttribute(
        staged_ldgsts_depth3_wave_scaling,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes));
    int resident = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, staged_ldgsts_depth3_wave_scaling, kThreads, kSharedBytes));
    if (resident != 1) {
        std::fprintf(stderr,
                     "expected one resident CTA per SM, got %d\n", resident);
        std::exit(EXIT_FAILURE);
    }
}

void launch(const std::uint8_t* input, std::uint32_t* checksums,
            int grid, int rounds) {
    staged_ldgsts_depth3_wave_scaling
        <<<grid, kThreads, kSharedBytes>>>(input, checksums, rounds);
    CHECK_CUDA(cudaGetLastError());
}

void warm_l2(const std::uint8_t* input, std::uint32_t* checksums) {
    launch(input, checksums, kSmCount, kMaxRounds);
    CHECK_CUDA(cudaDeviceSynchronize());
}

void measure(const std::uint8_t* input, std::uint32_t* checksums, int trials) {
    std::printf("grid_ctas,rounds,trial,duration_ms,logical_bytes,"
                "resident_ctas_per_sm\n");
    for (int grid : kGridCounts) {
        for (int rounds : kRounds) {
            for (int trial = 0; trial < trials; ++trial) {
                warm_l2(input, checksums);
                cudaEvent_t start;
                cudaEvent_t stop;
                CHECK_CUDA(cudaEventCreate(&start));
                CHECK_CUDA(cudaEventCreate(&stop));
                CHECK_CUDA(cudaEventRecord(start));
                launch(input, checksums, grid, rounds);
                CHECK_CUDA(cudaEventRecord(stop));
                CHECK_CUDA(cudaEventSynchronize(stop));
                float duration_ms = 0.0F;
                CHECK_CUDA(cudaEventElapsedTime(&duration_ms, start, stop));
                const long long logical_bytes = static_cast<long long>(grid)
                    * rounds * kPanelBytes;
                std::printf("%d,%d,%d,%.9f,%lld,1\n", grid, rounds, trial,
                            duration_ms, logical_bytes);
                CHECK_CUDA(cudaEventDestroy(stop));
                CHECK_CUDA(cudaEventDestroy(start));
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s DEVICE TRIALS\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device = parse_int(argv[1], "device");
    const int trials = parse_int(argv[2], "trials");
    if (device < 0 || trials < 3 || trials > 11) {
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (std::string(properties.name) != kExpectedDevice
            || properties.major != 8 || properties.minor != 0
            || properties.multiProcessorCount != kSmCount) {
        std::fprintf(stderr, "unexpected A800 device contract\n");
        return EXIT_FAILURE;
    }
    configure_variant();

    constexpr std::size_t kL2Bytes =
        static_cast<std::size_t>(kL2BackingPanels) * kPanelBytes;
    constexpr int kMaxGrid = 11008;
    std::uint8_t* input = nullptr;
    std::uint32_t* checksums = nullptr;
    CHECK_CUDA(cudaMalloc(&input, kL2Bytes));
    CHECK_CUDA(cudaMalloc(&checksums, kMaxGrid * sizeof(*checksums)));
    CHECK_CUDA(cudaMemset(input, 0x5a, kL2Bytes));
    CHECK_CUDA(cudaMemset(checksums, 0, kMaxGrid * sizeof(*checksums)));
    warm_l2(input, checksums);
    measure(input, checksums, trials);
    CHECK_CUDA(cudaFree(checksums));
    CHECK_CUDA(cudaFree(input));
    return EXIT_SUCCESS;
}
