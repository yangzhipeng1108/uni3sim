/* Independent finite three-slot LDGSTS stream with mixed cache tiers.
 *
 * This benchmark contains only asynchronous global-to-shared copies and a
 * checksum consumer.  Every CTA transfers one fixed 16-KiB operand-sized
 * panel per round through three shared-memory slots.  The L2 fraction changes
 * which vectors use a 32-MiB recycling backing store; the remainder uses a
 * unique DRAM backing store.  It deliberately contains no matrix operation,
 * output epilogue, or production-kernel source.
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

#ifndef PROBE_GRID
#define PROBE_GRID 64
#endif
#ifndef PROBE_SLOTS
#define PROBE_SLOTS 3
#endif
#ifndef PROBE_KSYNC
#define PROBE_KSYNC 0
#endif
constexpr int kThreads = 128;
constexpr int kGrid = PROBE_GRID;
constexpr int kPairSlots = PROBE_SLOTS;
constexpr int kPanelBytes = 16 * 1024;
constexpr int kPanelVectors = kPanelBytes / 16;
constexpr int kSharedBytes = kPairSlots * kPanelBytes;
// The static pad keeps exactly one CTA resident per SM for any supported
// slot count; sm_80 caps static shared memory at 48 KiB per CTA.
constexpr int kStaticOccupancyBytes = 48 * 1024;
constexpr int kL2BackingBytes = 32 * 1024 * 1024;
constexpr int kL2BackingPanels = kL2BackingBytes / kPanelBytes;
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

__device__ __forceinline__ void copy_16(void* destination, const void* source) {
    const std::uint32_t shared_address = static_cast<std::uint32_t>(
        __cvta_generic_to_shared(destination));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(shared_address), "l"(source) : "memory");
}

template<int L2Vectors, int Rounds>
__device__ __forceinline__ void issue_panel(
        std::uint8_t* destination, const std::uint8_t* l2_input,
        const std::uint8_t* dram_input, int round) {
#if PROBE_KSYNC
    /* K-synchronized operand order: at every round each CTA reads its panel
     * from the same K position, so the grid-collective DRAM stream advances
     * sequentially, matching the operand traffic order of a K-synchronized
     * staged GEMM mainloop. */
    const std::size_t panel_index = (
        static_cast<std::size_t>(round) * kGrid + blockIdx.x);
#else
    const std::size_t panel_index = (
        static_cast<std::size_t>(blockIdx.x) * Rounds + round);
#endif
    const std::size_t l2_panel = panel_index % kL2BackingPanels;
    const std::size_t dram_panel = panel_index;
    for (int vector = threadIdx.x; vector < kPanelVectors; vector += blockDim.x) {
        const std::uint8_t* source = vector < L2Vectors
            ? l2_input + l2_panel * kPanelBytes + vector * 16
            : dram_input + dram_panel * kPanelBytes + vector * 16;
        copy_16(destination + vector * 16, source);
    }
    asm volatile("cp.async.commit_group;\n" :: : "memory");
}

template<int L2Vectors, int Rounds>
__global__ __launch_bounds__(kThreads, 1)
void finite_mixed_stream(const std::uint8_t* __restrict__ l2_input,
                         const std::uint8_t* __restrict__ dram_input,
                         std::uint32_t* __restrict__ checksums) {
    __shared__ __align__(16) volatile std::uint8_t occupancy_storage[
        kStaticOccupancyBytes];
    extern __shared__ __align__(16) std::uint8_t slots[];
    if (threadIdx.x == 0) {
        occupancy_storage[0] = static_cast<std::uint8_t>(blockIdx.x);
    }
    __syncthreads();
    for (int slot = 0; slot < kPairSlots; ++slot) {
        issue_panel<L2Vectors, Rounds>(
            slots + slot * kPanelBytes, l2_input, dram_input, slot);
    }
    std::uint32_t checksum = occupancy_storage[0];
#pragma unroll 1
    for (int round = 0; round < Rounds; ++round) {
        asm volatile("cp.async.wait_group 1;\n" :: : "memory");
        __syncthreads();
        const int slot = round % kPairSlots;
        checksum ^= reinterpret_cast<volatile std::uint32_t*>(
            slots + slot * kPanelBytes)[threadIdx.x];
        __syncthreads();
        const int next = round + kPairSlots;
        if (next < Rounds) {
            issue_panel<L2Vectors, Rounds>(
                slots + slot * kPanelBytes, l2_input, dram_input, next);
        }
    }
    asm volatile("cp.async.wait_group 0;\n" :: : "memory");
    if (threadIdx.x == 0) checksums[blockIdx.x] = checksum;
}

template<int L2Vectors, int Rounds>
void configure() {
    CHECK_CUDA(cudaFuncSetAttribute(
        finite_mixed_stream<L2Vectors, Rounds>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes));
    int resident = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, finite_mixed_stream<L2Vectors, Rounds>, kThreads,
        kSharedBytes));
    if (resident != 1) {
        std::fprintf(stderr, "expected one resident CTA per SM, got %d\n", resident);
        std::exit(EXIT_FAILURE);
    }
}

template<int L2Vectors, int Rounds>
void launch(const std::uint8_t* l2_input, const std::uint8_t* dram_input,
            std::uint32_t* checksums) {
    finite_mixed_stream<L2Vectors, Rounds><<<kGrid, kThreads, kSharedBytes>>>(
        l2_input, dram_input, checksums);
    CHECK_CUDA(cudaGetLastError());
}

template<int L2Vectors, int Rounds>
void measure(const char* fraction_name, const std::uint8_t* l2_input,
             const std::uint8_t* dram_input, std::uint32_t* checksums,
             int trials) {
    configure<L2Vectors, Rounds>();
    for (int warmup = 0; warmup < 3; ++warmup) {
        launch<L2Vectors, Rounds>(l2_input, dram_input, checksums);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    for (int trial = 0; trial < trials; ++trial) {
        cudaEvent_t start;
        cudaEvent_t stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        launch<L2Vectors, Rounds>(l2_input, dram_input, checksums);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float duration_ms = 0.0F;
        CHECK_CUDA(cudaEventElapsedTime(&duration_ms, start, stop));
        const long long total_bytes = static_cast<long long>(kGrid) * Rounds
            * kPanelBytes;
        const long long l2_bytes = total_bytes * L2Vectors / kPanelVectors;
        std::printf("%s,%d,%d,%.9f,%lld,%lld,%lld,1\n", fraction_name,
                    trial, Rounds, duration_ms, total_bytes, l2_bytes,
                    total_bytes - l2_bytes);
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaEventDestroy(start));
    }
}

template<int Rounds>
void run_fraction(int fraction, const std::uint8_t* l2_input,
                  const std::uint8_t* dram_input, std::uint32_t* checksums,
                  int trials) {
    switch (fraction) {
    case 0: measure<0, Rounds>("l2_0pct", l2_input, dram_input, checksums, trials); break;
    case 25: measure<kPanelVectors / 4, Rounds>("l2_25pct", l2_input, dram_input, checksums, trials); break;
    case 50: measure<kPanelVectors / 2, Rounds>("l2_50pct", l2_input, dram_input, checksums, trials); break;
    case 75: measure<3 * kPanelVectors / 4, Rounds>("l2_75pct", l2_input, dram_input, checksums, trials); break;
    case 100: measure<kPanelVectors, Rounds>("l2_100pct", l2_input, dram_input, checksums, trials); break;
    default:
        std::fprintf(stderr, "unsupported L2 fraction %d\n", fraction);
        std::exit(EXIT_FAILURE);
    }
}

template<int Rounds>
void run_rounds(int trials) {
    // The prologue always stages kPairSlots operand panels, so even an
    // empty (zero-round) launch needs backing for those first panels.
    constexpr int kBackedRounds = Rounds < kPairSlots ? kPairSlots : Rounds;
    const std::size_t dram_bytes = static_cast<std::size_t>(kGrid)
        * kBackedRounds * kPanelBytes;
    std::uint8_t* l2_input = nullptr;
    std::uint8_t* dram_input = nullptr;
    std::uint32_t* checksums = nullptr;
    CHECK_CUDA(cudaMalloc(&l2_input, kL2BackingBytes));
    CHECK_CUDA(cudaMalloc(&dram_input, dram_bytes));
    CHECK_CUDA(cudaMalloc(&checksums, kGrid * sizeof(*checksums)));
    CHECK_CUDA(cudaMemset(l2_input, 0x5a, kL2BackingBytes));
    CHECK_CUDA(cudaMemset(dram_input, 0xa5, dram_bytes));
    CHECK_CUDA(cudaMemset(checksums, 0, kGrid * sizeof(*checksums)));
    for (int fraction : {0, 25, 50, 75, 100}) {
        run_fraction<Rounds>(fraction, l2_input, dram_input, checksums, trials);
    }
    CHECK_CUDA(cudaFree(checksums));
    CHECK_CUDA(cudaFree(dram_input));
    CHECK_CUDA(cudaFree(l2_input));
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s DEVICE TRIALS ROUNDS\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device = parse_int(argv[1], "device");
    const int trials = parse_int(argv[2], "trials");
    const int rounds = parse_int(argv[3], "rounds");
    if (device < 0 || trials < 3 || trials > 11) return EXIT_FAILURE;
    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (std::string(properties.name) != kExpectedDevice
            || properties.major != 8 || properties.minor != 0
            || properties.multiProcessorCount != 108) {
        std::fprintf(stderr, "unexpected A800 device contract\n");
        return EXIT_FAILURE;
    }
    std::printf("l2_fraction,trial,rounds,duration_ms,total_bytes,l2_bytes,dram_bytes,resident_ctas_per_sm\n");
    switch (rounds) {
    case 0: run_rounds<0>(trials); break;
    case 16: run_rounds<16>(trials); break;
    case 32: run_rounds<32>(trials); break;
    case 40: run_rounds<40>(trials); break;
    case 48: run_rounds<48>(trials); break;
    case 64: run_rounds<64>(trials); break;
    case 96: run_rounds<96>(trials); break;
    case 128: run_rounds<128>(trials); break;
    case 152: run_rounds<152>(trials); break;
    case 256: run_rounds<256>(trials); break;
    case 512: run_rounds<512>(trials); break;
    default:
        std::fprintf(stderr,
                     "rounds must be one of 0,16,32,40,48,64,96,128,152,256,512\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
