/* NVIDIA H200 owner copy of
 * microbench/nvidia_a800/memory/dram/staged_ldgsts_64x64_depth6_curve.cu.
 * Kernel body, 128-thread CTA, 16-KiB A/B pair panels, three complete-pair
 * slots, 4096 rounds, the fixed 64-CTA decode cohort, 32-MiB L2-resident /
 * unique-DRAM tiers and the event-timed loop are unchanged.  Hardware
 * coupling changes only:
 *   - device contract "NVIDIA H200", compute capability 9.0, 132 SMs;
 *   - the dynamic shared reservation gains an untouched pad so that exactly
 *     one CTA stays resident per SM.  The 228-KiB H200 SM would otherwise
 *     admit two CTAs of the A800 48-KiB static + 48-KiB dynamic class; the
 *     A800 class is one resident CTA per SM, so the pad is sized until two
 *     no longer fit (2 x (49,152 + 69,632 + 1,024 reserved) B > 233,472 B).
 *     The operand ring still uses only its complete-slot bytes, and the
 *     occupancy query still fails closed unless it reports one;
 *   - build for sm_90.
 */
/* Independent SM80 64x64x64 staged-LDGSTS source-service curve.
 *
 * This is deliberately a copy-only mechanism benchmark, not a GEMM: a
 * 128-thread CTA moves private FP16-sized 64x64 A/B panel pairs through the
 * three complete 16-KiB operand slots exposed by CUDA 12.2's six-stage
 * cuBLASLt source program.  A fixed 64-CTA grid models one resident CTA on
 * each active SM of the decode launch.  The L2 and DRAM tiers distinguish a
 * 32-MiB resident circular backing set from a unique backing stream.
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

constexpr int kThreads = 128;
constexpr int kGrid = 64;
constexpr int kRounds = 4096;
constexpr int kAPanelBytes = 64 * 64 * 2;
constexpr int kBPanelBytes = 64 * 64 * 2;
constexpr int kPanelBytes = kAPanelBytes + kBPanelBytes;
constexpr int kCompletePairSlots = 3;
constexpr int kOperandSharedBytes = kCompletePairSlots * kPanelBytes;
constexpr int kStaticOccupancyBytes = kOperandSharedBytes;
constexpr int kH200OccupancyPadBytes = 20 * 1024;
constexpr int kDynamicSharedBytes =
    kOperandSharedBytes + kH200OccupancyPadBytes;
constexpr int kL2BackingBytes = 32 * 1024 * 1024;
constexpr int kL2BackingPanels = kL2BackingBytes / kPanelBytes;
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
        panel = (static_cast<std::size_t>(blockIdx.x) * kRounds + round)
            % kL2BackingPanels;
    } else {
        panel = static_cast<std::size_t>(blockIdx.x) * kRounds + round;
    }
    return input + panel * kPanelBytes;
}

template<Tier tier>
__global__ __launch_bounds__(kThreads, 1)
void staged_ldgsts_64x64_depth6_curve(
        const std::uint8_t* __restrict__ input,
        std::uint32_t* __restrict__ checksums) {
    __shared__ __align__(16) volatile std::uint8_t occupancy_storage[
        kStaticOccupancyBytes];
    extern __shared__ __align__(16) std::uint8_t operand_storage[];
    if (threadIdx.x == 0) {
        occupancy_storage[0] = static_cast<std::uint8_t>(blockIdx.x);
    }
    __syncthreads();
    for (int slot = 0; slot < kCompletePairSlots; ++slot) {
        issue_panel(
            operand_storage + slot * kPanelBytes,
            panel_source<tier>(input, slot));
    }
    std::uint32_t checksum = occupancy_storage[0];
    #pragma unroll 1
    for (int round = 0; round < kRounds; ++round) {
        asm volatile("cp.async.wait_group 1;\n" :: : "memory");
        __syncthreads();
        const int slot = round % kCompletePairSlots;
        checksum ^= reinterpret_cast<volatile std::uint32_t*>(
            operand_storage + slot * kPanelBytes)[threadIdx.x];
        __syncthreads();
        const int next = round + kCompletePairSlots;
        if (next < kRounds) {
            issue_panel(
                operand_storage + slot * kPanelBytes,
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
        staged_ldgsts_64x64_depth6_curve<tier>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kDynamicSharedBytes));
    int resident = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, staged_ldgsts_64x64_depth6_curve<tier>,
        kThreads, kDynamicSharedBytes));
    if (resident != 1) {
        std::fprintf(stderr,
                     "expected one resident CTA per SM, got %d\n", resident);
        std::exit(EXIT_FAILURE);
    }
}

template<Tier tier>
void launch(const std::uint8_t* input, std::uint32_t* checksums) {
    staged_ldgsts_64x64_depth6_curve<tier>
        <<<kGrid, kThreads, kDynamicSharedBytes>>>(input, checksums);
    CHECK_CUDA(cudaGetLastError());
}

template<Tier tier>
void measure(const char* tier_name, const std::uint8_t* input,
             std::uint32_t* checksums, int trials) {
    for (int warmup = 0; warmup < 3; ++warmup) {
        launch<tier>(input, checksums);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    for (int trial = 0; trial < trials; ++trial) {
        cudaEvent_t start;
        cudaEvent_t stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        launch<tier>(input, checksums);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float duration_ms = 0.0F;
        CHECK_CUDA(cudaEventElapsedTime(&duration_ms, start, stop));
        const long long logical_bytes = (
            static_cast<long long>(kGrid) * kRounds * kPanelBytes);
        std::printf("%s,%d,%.9f,%lld,1\n", tier_name, trial, duration_ms,
                    logical_bytes);
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaEventDestroy(start));
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
            || properties.multiProcessorCount != 132) {
        std::fprintf(stderr, "unexpected H200 device contract\n");
        return EXIT_FAILURE;
    }
    configure_variant<Tier::kL2>();
    configure_variant<Tier::kDram>();

    const std::size_t dram_bytes = static_cast<std::size_t>(kGrid)
        * kRounds * kPanelBytes;
    std::uint8_t* l2_input = nullptr;
    std::uint8_t* dram_input = nullptr;
    std::uint32_t* checksums = nullptr;
    CHECK_CUDA(cudaMalloc(&l2_input, kL2BackingBytes));
    CHECK_CUDA(cudaMalloc(&dram_input, dram_bytes));
    CHECK_CUDA(cudaMalloc(&checksums, kGrid * sizeof(*checksums)));
    CHECK_CUDA(cudaMemset(l2_input, 0x5a, kL2BackingBytes));
    CHECK_CUDA(cudaMemset(dram_input, 0xa5, dram_bytes));
    CHECK_CUDA(cudaMemset(checksums, 0, kGrid * sizeof(*checksums)));

    std::printf("tier,trial,duration_ms,logical_bytes,resident_ctas_per_sm\n");
    measure<Tier::kL2>("l2_resident", l2_input, checksums, trials);
    measure<Tier::kDram>("dram_unique", dram_input, checksums, trials);

    CHECK_CUDA(cudaFree(checksums));
    CHECK_CUDA(cudaFree(dram_input));
    CHECK_CUDA(cudaFree(l2_input));
    return EXIT_SUCCESS;
}
