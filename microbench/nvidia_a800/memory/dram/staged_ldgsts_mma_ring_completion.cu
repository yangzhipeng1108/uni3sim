/* Launch-level fixed completion of a staged LDGSTS ring with a tensor-core
 * consumer, on SM80.
 *
 * Purpose.  The finite completion of a staged ring -- its launch, ring fill and
 * drain, and the entry into the accumulator epilogue -- is a
 * launch/synchronization mechanism, which is the category a Forward microbench
 * is allowed to measure.  A copy-only ring cannot measure it: retiring a slot
 * with a volatile shared read and two barriers under-measures the fixed cost by
 * an order of magnitude against a ring whose slots are consumed by matrix
 * instructions.  This probe therefore keeps the copy-only ring's exact
 * structure -- 128 threads, three complete operand-pair slots, cp.async with
 * commit groups, and the zero-filled A panel a staged GEMM issues when its
 * logical M covers one tile -- and replaces only the consumer with a
 * tensor-core one.
 *
 * What it reports.  Only the intercept of completion against round count, in
 * nanoseconds, with a physical meaning (launch plus ring fill plus drain plus
 * epilogue entry).  The per-round steady service is deliberately NOT reported:
 * the published typed byte capacities already price the marginal round
 * correctly, so nothing here needs to replace them.
 *
 * Achieved-rate result, added later.  Measured as a fraction of DRAM read peak
 * at a production topology -- panel 24576, four slots, one resident CTA per SM --
 * this ring reads 0.766 and 0.778 at grid 224 with 1536 and 3072 KiB per CTA,
 * against 0.772 and 0.781 for the copy-only ring and 0.818 for the production
 * kernel the dataset's ncu counters describe.  Adding the tensor-core consumer
 * moves the fraction DOWN by 0.006 to 0.008, so the consumer is not why an
 * authored ring falls short of production: no authored ring reaches it, because
 * cuBLAS's ring is hand-tuned SASS whose issue schedule a PTX-authored probe does
 * not match.  That closes the probe route for an operand-path efficiency
 * coefficient by measurement.
 *
 * Known risk, for the reviewer.  Ring plus LDSM-class fragment loads plus MMA
 * plus an accumulator epilogue is structurally the inner loop of a GEMM.  The
 * probe contains no production shape, no production data and no production
 * kernel, and it consumes no production duration, but its structural proximity
 * to the source class is closer than any other probe in this directory.  Treat
 * the derived intercept as a mechanism candidate until that proximity has been
 * reviewed against the Forward zero-fitting contract.
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -DPROBE_PANEL_BYTES=24576 -DPROBE_SLOTS=3 \
 *        -DPROBE_GRID=64 -DPROBE_PAD_BYTES=24576 -DPROBE_ZFILL_BYTES=8192 \
 *        -o staged_ldgsts_mma_ring_completion \
 *        staged_ldgsts_mma_ring_completion.cu
 */
#include <cuda_runtime.h>
#include <mma.h>

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
constexpr int kThreads = 128;
constexpr int kGrid = PROBE_GRID;
constexpr int kPairSlots = PROBE_SLOTS;
constexpr int kPanelBytes = PROBE_PANEL_BYTES;
constexpr int kSharedBytes = kPairSlots * kPanelBytes;
constexpr int kStaticOccupancyBytes = PROBE_PAD_BYTES;
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
    using namespace nvcuda;
    __shared__ __align__(16) volatile std::uint8_t occupancy_storage[
        kStaticOccupancyBytes];
    extern __shared__ __align__(16) std::uint8_t slots[];
    if (threadIdx.x == 0) {
        occupancy_storage[0] = static_cast<std::uint8_t>(blockIdx.x);
    }
    __syncthreads();
    // One accumulator fragment per warp keeps the epilogue real without
    // introducing an output matrix or a production shape.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    for (int slot = 0; slot < kPairSlots; ++slot) {
        issue_panel(slots + slot * kPanelBytes,
                    panel_source<tier, Rounds>(input, slot));
    }
#pragma unroll 1
    for (int round = 0; round < Rounds; ++round) {
        asm volatile("cp.async.wait_group 1;\n" :: : "memory");
        __syncthreads();
        const int slot = round % kPairSlots;
        // Consume the slot through the tensor pipe: fragment loads out of the
        // staged panel followed by one matrix multiply-accumulate per warp.
        const half* panel = reinterpret_cast<const half*>(
            slots + slot * kPanelBytes);
        const int warp = static_cast<int>(threadIdx.x) / 32;
        const half* a_tile = panel + warp * 16 * 16;
        const half* b_tile = panel + (kPanelBytes / sizeof(half)) / 2
                             + warp * 16 * 16;
        wmma::load_matrix_sync(a_frag, a_tile, 16);
        wmma::load_matrix_sync(b_frag, b_tile, 16);
        wmma::mma_sync(accumulator, a_frag, b_frag, accumulator);
        __syncthreads();
        const int next = round + kPairSlots;
        if (next < Rounds) {
            issue_panel(slots + slot * kPanelBytes,
                        panel_source<tier, Rounds>(input, next));
        }
    }
    asm volatile("cp.async.wait_group 0;\n" :: : "memory");
    // Accumulator epilogue: reduce the fragment so the matrix work cannot be
    // eliminated, then publish one value per CTA.
    float total = 0.0f;
    for (int element = 0; element < accumulator.num_elements; ++element) {
        total += accumulator.x[element];
    }
    total += static_cast<float>(occupancy_storage[0]);
    if (threadIdx.x == 0) {
        checksums[blockIdx.x] = static_cast<std::uint32_t>(total);
    }
}

template<Tier tier, int Rounds>
void configure() {
    CHECK_CUDA(cudaFuncSetAttribute(
        finite_stream<tier, Rounds>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSharedBytes));
    int resident = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, finite_stream<tier, Rounds>, kThreads, kSharedBytes));
    if (resident != 1) {
        std::fprintf(stderr, "expected one resident CTA per SM, got %d\n", resident);
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
        std::printf("%d,%d,%d,%d,%s,%d,%d,%.9f,%lld,1\n", kGrid, kPanelBytes,
                    kZfillBytes, kPairSlots, tier_name, trial, Rounds,
                    duration_ms, bytes);
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
    case 16: run_rounds<16>(trials); break;
    case 32: run_rounds<32>(trials); break;
    case 64: run_rounds<64>(trials); break;
    case 128: run_rounds<128>(trials); break;
    case 256: run_rounds<256>(trials); break;
    case 512: run_rounds<512>(trials); break;
    default:
        std::fprintf(stderr, "rounds must be one of 16,32,64,128,256,512\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
