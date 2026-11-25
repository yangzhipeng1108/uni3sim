/* NVIDIA H200 owner copy of
 * microbench/nvidia_a800/memory/l2/staged_ldgsts_depth3_curve.cu.
 * Kernel body, 49,152-B panels, three-slot ring with wait_group 1, round set,
 * 32-MiB L2-resident backing set, unique DRAM stream, and the event-timed
 * measurement loop are unchanged.  Hardware coupling changes only:
 *   - kGrid 108 -> 132 (one CTA per physical H200 SM);
 *   - device contract "NVIDIA H200", compute capability 9.0, 132 SMs;
 *   - build for sm_90;
 *   - one discarded event-timed zero-round launch precedes the measured
 *     matrix (the first event-timed launch of an H200 process is ~2 us slow,
 *     which failed the A800 zero-round stability gate in 3/3 captures);
 * The 147,456-B reservation still admits exactly one resident CTA on the
 * 228-KiB H200 SM (checked at runtime).  The DRAM tier backs 132 x 1024
 * panels (6.2 GiB), far above the 60-MiB H200 L2.
 */
/* Independent SM80 three-stage LDGSTS source-service curve.
 *
 * This is not a GEMM.  Each 256-thread CTA copies private 128x64 A-shaped
 * and 64x256 B-shaped FP16 byte panels through a three-slot cp.async ring.
 * There is no matrix multiply, output matrix, split-K reducer, or production
 * shape.  The two cache tiers isolate a 32-MiB L2-resident circular backing
 * set and a unique, L2-capacity-exceeding DRAM stream.  Multiple round counts
 * expose source startup and steady service without consuming production
 * duration.
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
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                     __FILE__, __LINE__, cudaGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                             \
    }                                                                        \
} while (0)

constexpr int kThreads = 256;
constexpr int kGrid = 132;
constexpr int kMaxRounds = 1024;
constexpr int kAPanelBytes = 128 * 64 * 2;
constexpr int kBPanelBytes = 64 * 256 * 2;
constexpr int kPanelBytes = kAPanelBytes + kBPanelBytes;
constexpr int kStages = 3;
constexpr int kSharedBytes = kStages * kPanelBytes;
constexpr int kL2BackingPanels = (32 * 1024 * 1024) / kPanelBytes;
constexpr int kRounds[] = {0, 16, 32, 48, 128, 256, 1024};
constexpr const char* kExpectedDevice = "NVIDIA H200";

enum class Tier : int {
    kL2 = 0,
    kDram = 1,
};

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
    for (int vector = threadIdx.x; vector < kVectors;
         vector += blockDim.x) {
        cp_async_16(destination + vector * 16, source + vector * 16);
    }
    asm volatile("cp.async.commit_group;\n" :: : "memory");
}

template<Tier tier>
__device__ __forceinline__ const std::uint8_t* panel_source(
        const std::uint8_t* input, int round) {
    std::size_t panel = 0;
    if constexpr (tier == Tier::kL2) {
        panel = (static_cast<std::size_t>(blockIdx.x) * kMaxRounds + round)
            % kL2BackingPanels;
    } else {
        panel = static_cast<std::size_t>(blockIdx.x) * kMaxRounds + round;
    }
    return input + panel * kPanelBytes;
}

template<Tier tier>
__global__ __launch_bounds__(kThreads, 1)
void staged_ldgsts_depth3_curve(
        const std::uint8_t* __restrict__ input,
        std::uint32_t* __restrict__ checksums,
        int rounds) {
    extern __shared__ __align__(16) std::uint8_t shared[];
    const int initial_groups = rounds < kStages ? rounds : kStages;
    for (int round = 0; round < initial_groups; ++round) {
        issue_panel(
            shared + round * kPanelBytes,
            panel_source<tier>(input, round));
    }
    std::uint32_t checksum = 0;
    #pragma unroll 1
    for (int round = 0; round < rounds; ++round) {
        // The source-visible three-slot pipeline reaches DEPBAR.LE SB0,1:
        // at most one newer LDGSTS group may remain before this slot is read.
        asm volatile("cp.async.wait_group 1;\n" :: : "memory");
        __syncthreads();
        const int slot = round % kStages;
        checksum ^= reinterpret_cast<volatile std::uint32_t*>(
            shared + slot * kPanelBytes)[threadIdx.x];
        __syncthreads();
        const int next = round + kStages;
        if (next < rounds) {
            issue_panel(
                shared + slot * kPanelBytes,
                panel_source<tier>(input, next));
        }
    }
    asm volatile("cp.async.wait_group 0;\n" :: : "memory");
    if (threadIdx.x == 0) {
        checksums[blockIdx.x] = checksum;
    }
}

template<Tier tier>
void configure_variant() {
    CHECK_CUDA(cudaFuncSetAttribute(
        staged_ldgsts_depth3_curve<tier>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSharedBytes));
    int resident = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, staged_ldgsts_depth3_curve<tier>,
        kThreads, kSharedBytes));
    if (resident != 1) {
        std::fprintf(stderr,
                     "expected one resident CTA per SM, got %d\n", resident);
        std::exit(EXIT_FAILURE);
    }
}

template<Tier tier>
void launch(
        const std::uint8_t* input, std::uint32_t* checksums, int rounds) {
    staged_ldgsts_depth3_curve<tier>
        <<<kGrid, kThreads, kSharedBytes>>>(input, checksums, rounds);
    CHECK_CUDA(cudaGetLastError());
}

template<Tier tier>
void measure(
        const char* tier_name,
        const std::uint8_t* input,
        const std::uint8_t* l2_input,
        std::uint32_t* checksums,
        int trials) {
    for (int rounds : kRounds) {
        for (int trial = 0; trial < trials; ++trial) {
            if constexpr (tier == Tier::kL2) {
                launch<Tier::kL2>(l2_input, checksums, kMaxRounds);
                CHECK_CUDA(cudaDeviceSynchronize());
            }
            cudaEvent_t start;
            cudaEvent_t stop;
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));
            CHECK_CUDA(cudaEventRecord(start));
            launch<tier>(input, checksums, rounds);
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float duration_ms = 0.0F;
            CHECK_CUDA(cudaEventElapsedTime(&duration_ms, start, stop));
            const long long logical_bytes = (
                static_cast<long long>(kGrid) * rounds * kPanelBytes);
            std::printf("%s,%d,%d,%.9f,%lld,1\n",
                        tier_name, rounds, trial, duration_ms, logical_bytes);
            CHECK_CUDA(cudaEventDestroy(stop));
            CHECK_CUDA(cudaEventDestroy(start));
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
            || properties.major != 9 || properties.minor != 0
            || properties.multiProcessorCount != kGrid) {
        std::fprintf(stderr, "unexpected H200 device contract\n");
        return EXIT_FAILURE;
    }
    configure_variant<Tier::kL2>();
    configure_variant<Tier::kDram>();

    constexpr std::size_t kL2Bytes =
        static_cast<std::size_t>(kL2BackingPanels) * kPanelBytes;
    constexpr std::size_t kDramBytes =
        static_cast<std::size_t>(kGrid) * kMaxRounds * kPanelBytes;
    std::uint8_t* l2_input = nullptr;
    std::uint8_t* dram_input = nullptr;
    std::uint32_t* checksums = nullptr;
    CHECK_CUDA(cudaMalloc(&l2_input, kL2Bytes));
    CHECK_CUDA(cudaMalloc(&dram_input, kDramBytes));
    CHECK_CUDA(cudaMalloc(&checksums, kGrid * sizeof(*checksums)));
    CHECK_CUDA(cudaMemset(l2_input, 0x5a, kL2Bytes));
    CHECK_CUDA(cudaMemset(dram_input, 0xa5, kDramBytes));
    CHECK_CUDA(cudaMemset(checksums, 0, kGrid * sizeof(*checksums)));

    launch<Tier::kL2>(l2_input, checksums, kMaxRounds);
    launch<Tier::kDram>(dram_input, checksums, kMaxRounds);
    CHECK_CUDA(cudaDeviceSynchronize());

    // H200 timing-harness coupling: the first cudaEvent-timed launch of a
    // process reads ~2 us slower on this driver (the first zero-round L2
    // trial, 3/3 captures).  One discarded event-timed zero-round launch
    // absorbs it before the measured matrix; no measured cell changes.
    {
        cudaEvent_t start;
        cudaEvent_t stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        launch<Tier::kL2>(l2_input, checksums, 0);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaEventDestroy(start));
    }

    std::printf("tier,rounds,trial,duration_ms,logical_bytes,"
                "resident_ctas_per_sm\n");
    measure<Tier::kL2>(
        "l2_resident", l2_input, l2_input, checksums, trials);
    measure<Tier::kDram>(
        "dram_unique", dram_input, l2_input, checksums, trials);

    CHECK_CUDA(cudaFree(checksums));
    CHECK_CUDA(cudaFree(dram_input));
    CHECK_CUDA(cudaFree(l2_input));
    return EXIT_SUCCESS;
}
