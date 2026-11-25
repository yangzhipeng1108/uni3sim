/* Independent finite LDGSTS stream on SM80, parameterized by ring topology.
 *
 * Same ring as staged_ldgsts_finite_stream.cu -- no matrix instructions and no
 * output epilogue, a 128-thread CTA moving one operand pair per round through
 * asynchronous shared-memory slots -- with the operand-pair size, slot count,
 * CTA grid and occupancy pad exposed as compile-time parameters.  That sweeps
 * the declared panel geometries the SM80 GEMM source classes actually use
 * (64x64x64 -> 16 KiB, 64x128x64 -> 24 KiB, 64x256x64 -> 40 KiB) instead of
 * only the 16-KiB pair, and confirms whether the per-round service and the
 * finite fill/drain depend on the grid.
 *
 * The occupancy pad keeps exactly one resident CTA per SM for every panel
 * size: the probe fails rather than measure a different residency.  It
 * contains no production shape and no production or evaluation duration.
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -DPROBE_PANEL_BYTES=24576 -DPROBE_SLOTS=3 \
 *        -DPROBE_GRID=64 -DPROBE_PAD_BYTES=24576 \
 *        -o staged_ldgsts_panel_topology_stream \
 *        staged_ldgsts_panel_topology_stream.cu
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

#ifndef PROBE_PANEL_BYTES
#define PROBE_PANEL_BYTES (16 * 1024)
#endif
#ifndef PROBE_SLOTS
#define PROBE_SLOTS 3
#endif
#ifndef PROBE_GRID
#define PROBE_GRID 64
#endif
#ifndef PROBE_ZFILL_BYTES
#define PROBE_ZFILL_BYTES 0
#endif
#ifndef PROBE_PAD_BYTES
#define PROBE_PAD_BYTES (PROBE_SLOTS * PROBE_PANEL_BYTES)
#endif
// Residency is part of the measured contract, not a constant: a production ring
// of 24576-byte panels over four slots admits one CTA per multiprocessor while
// one of 16384-byte panels over five slots admits two, and both are real
// topologies.  The probe verifies the residency it was asked for and refuses any
// other, so a measurement can never silently describe a different one.
#ifndef PROBE_RESIDENT
#define PROBE_RESIDENT 1
#endif
constexpr int kThreads = 128;
constexpr int kGrid = PROBE_GRID;
constexpr int kPairSlots = PROBE_SLOTS;
constexpr int kPanelBytes = PROBE_PANEL_BYTES;
constexpr int kSharedBytes = kPairSlots * kPanelBytes;
constexpr int kStaticOccupancyBytes = PROBE_PAD_BYTES;
constexpr int kResidentCtasPerSm = PROBE_RESIDENT;
constexpr int kZfillBytes = PROBE_ZFILL_BYTES;
static_assert(kZfillBytes % 16 == 0 && kZfillBytes >= 0
              && kZfillBytes <= kPanelBytes,
              "zero-filled prefix must be a whole-copy part of the panel");
static_assert(kPanelBytes % 16 == 0, "panel must hold whole 16-byte copies");
static_assert(kPanelBytes / 16 >= kThreads,
              "panel must cover at least one copy per thread");
constexpr int kL2BackingBytes = 32 * 1024 * 1024;
constexpr int kL2BackingPanels = kL2BackingBytes / kPanelBytes;
constexpr const char* kExpectedDevice = "NVIDIA A800-SXM4-80GB";

enum class Tier : int { kL2 = 0, kDram = 1 };

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

/* Out-of-range operand rows.  A cuBLAS SM80 staged GEMM whose logical M is
 * smaller than its tile issues the same LDGSTS instructions for the missing
 * rows with a zero source size, so shared memory is zero-filled and no global
 * byte is fetched while the instruction and its commit group still execute.
 * Reproducing that exactly is what separates "the round costs its declared
 * panel" from "the round costs only the bytes it moves". */
__device__ __forceinline__ void copy_16_zfill(void* destination,
                                              const void* source) {
    const std::uint32_t shared_address = static_cast<std::uint32_t>(
        __cvta_generic_to_shared(destination));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\n"
                 :: "r"(shared_address), "l"(source) : "memory");
}

__device__ __forceinline__ void issue_panel(
        std::uint8_t* destination, const std::uint8_t* source) {
    constexpr int vectors = kPanelBytes / 16;
    constexpr int zfill_vectors = PROBE_ZFILL_BYTES / 16;
    for (int vector = threadIdx.x; vector < vectors; vector += blockDim.x) {
        if (vector < zfill_vectors) {
            copy_16_zfill(destination + vector * 16, source + vector * 16);
        } else {
            copy_16(destination + vector * 16, source + vector * 16);
        }
    }
    asm volatile("cp.async.commit_group;\n" :: : "memory");
}

template<Tier tier, int Rounds>
__device__ __forceinline__ const std::uint8_t* panel_source(
        const std::uint8_t* input, int round) {
    std::size_t panel = 0;
    if constexpr (tier == Tier::kL2) {
        panel = (static_cast<std::size_t>(blockIdx.x) * Rounds + round)
            % kL2BackingPanels;
    } else {
        panel = static_cast<std::size_t>(blockIdx.x) * Rounds + round;
    }
    return input + panel * kPanelBytes;
}

template<Tier tier, int Rounds>
__global__ __launch_bounds__(kThreads, 1)
void finite_stream(const std::uint8_t* __restrict__ input,
                   std::uint32_t* __restrict__ checksums) {
    __shared__ __align__(16) volatile std::uint8_t occupancy_storage[
        kStaticOccupancyBytes];
    extern __shared__ __align__(16) std::uint8_t slots[];
    // The fixed 48-KiB allocation retains the one-CTA/SM envelope while the
    // dynamic allocation is solely the three rotating operand-pair slots.
    if (threadIdx.x == 0) {
        occupancy_storage[0] = static_cast<std::uint8_t>(blockIdx.x);
    }
    __syncthreads();
    for (int slot = 0; slot < kPairSlots; ++slot) {
        issue_panel(slots + slot * kPanelBytes,
                    panel_source<tier, Rounds>(input, slot));
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
            issue_panel(slots + slot * kPanelBytes,
                        panel_source<tier, Rounds>(input, next));
        }
    }
    asm volatile("cp.async.wait_group 0;\n" :: : "memory");
    if (threadIdx.x == 0) checksums[blockIdx.x] = checksum;
}

template<Tier tier, int Rounds>
void configure() {
    CHECK_CUDA(cudaFuncSetAttribute(
        finite_stream<tier, Rounds>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSharedBytes));
    int resident = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, finite_stream<tier, Rounds>, kThreads, kSharedBytes));
    if (resident != kResidentCtasPerSm) {
        std::fprintf(stderr, "expected %d resident CTA(s) per SM, got %d\n",
                     kResidentCtasPerSm, resident);
        std::exit(EXIT_FAILURE);
    }
}

template<Tier tier, int Rounds>
void launch(const std::uint8_t* input, std::uint32_t* checksums) {
    finite_stream<tier, Rounds><<<kGrid, kThreads, kSharedBytes>>>(
        input, checksums);
    CHECK_CUDA(cudaGetLastError());
}

template<Tier tier, int Rounds>
void measure(const char* tier_name, const std::uint8_t* input,
             std::uint32_t* checksums, int trials) {
    configure<tier, Rounds>();
    for (int warmup = 0; warmup < 3; ++warmup) launch<tier, Rounds>(input, checksums);
    CHECK_CUDA(cudaDeviceSynchronize());
    for (int trial = 0; trial < trials; ++trial) {
        cudaEvent_t start;
        cudaEvent_t stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        launch<tier, Rounds>(input, checksums);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float duration_ms = 0.0F;
        CHECK_CUDA(cudaEventElapsedTime(&duration_ms, start, stop));
        const long long bytes = static_cast<long long>(kGrid) * Rounds * kPanelBytes;
        std::printf("%d,%d,%d,%d,%s,%d,%d,%.9f,%lld,%d\n", kGrid, kPanelBytes,
                    kZfillBytes, kPairSlots, tier_name, trial, Rounds,
                    duration_ms, bytes, kResidentCtasPerSm);
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaEventDestroy(start));
    }
}

template<int Rounds>
void run_rounds(int trials) {
    const std::size_t dram_bytes = static_cast<std::size_t>(kGrid) * Rounds * kPanelBytes;
    std::uint8_t* l2_input = nullptr;
    std::uint8_t* dram_input = nullptr;
    std::uint32_t* checksums = nullptr;
    CHECK_CUDA(cudaMalloc(&l2_input, kL2BackingBytes));
    CHECK_CUDA(cudaMalloc(&dram_input, dram_bytes));
    CHECK_CUDA(cudaMalloc(&checksums, kGrid * sizeof(*checksums)));
    CHECK_CUDA(cudaMemset(l2_input, 0x5a, kL2BackingBytes));
    CHECK_CUDA(cudaMemset(dram_input, 0xa5, dram_bytes));
    CHECK_CUDA(cudaMemset(checksums, 0, kGrid * sizeof(*checksums)));
    measure<Tier::kL2, Rounds>("l2_resident", l2_input, checksums, trials);
    measure<Tier::kDram, Rounds>("dram_unique", dram_input, checksums, trials);
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
    std::printf("grid,panel_bytes,zfill_bytes,slots,tier,trial,rounds,duration_ms,logical_bytes,resident_ctas_per_sm\n");
    switch (rounds) {
    // Production K loops are frequently shorter than sixteen tiles, so the
    // short end of the round axis is where many launches actually live.
    case 4: run_rounds<4>(trials); break;
    case 8: run_rounds<8>(trials); break;
    case 16: run_rounds<16>(trials); break;
    case 32: run_rounds<32>(trials); break;
    case 64: run_rounds<64>(trials); break;
    case 128: run_rounds<128>(trials); break;
    case 256: run_rounds<256>(trials); break;
    case 512: run_rounds<512>(trials); break;
    default:
        std::fprintf(stderr,
                     "rounds must be one of 4,8,16,32,64,128,256,512\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
