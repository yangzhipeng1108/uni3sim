/*
 * Independent A800 LDG128-to-FMA operand-readiness mechanism probe.
 *
 * The timed body is deliberately smaller than FlashAttention: a 128-thread
 * CTA performs one aligned 16-byte float4 global load per lane and consumes
 * the four values with FP32 FMA instructions.  The accumulator survives each
 * repeat, which prevents the compiler from replacing the loop with an
 * independent vector operation.  It has no production input, kernel name, or
 * duration dependency.
 *
 * Modes are matched by construction:
 *   0 empty      -- address arithmetic and loop control only;
 *   1 fma_only   -- four register FP32 FMA operations per lane/repeat;
 *   2 ldg128_fma -- an aligned ld.global.cg.v4.f32 followed by those FMAs.
 *
 * The report subtracts fma_only from ldg128_fma.  Therefore the published
 * quantity is load-to-operands-ready cycles, not a complete operator time and
 * not a second charge for the FMA resource.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int kThreadsPerCta = 128;
constexpr int kExpectedSmCount = 108;
constexpr int kExpectedMajor = 8;
constexpr int kExpectedMinor = 0;
constexpr int kTileBytes = kThreadsPerCta * sizeof(float4);
constexpr int kDynamicSharedBytes = 96 * 1024;
constexpr std::size_t kL2WarmBytes = 8ULL * 1024ULL * 1024ULL;
constexpr std::size_t kDramPoolBytes = 256ULL * 1024ULL * 1024ULL;
constexpr std::size_t kEvictionBytes = 80ULL * 1024ULL * 1024ULL;
constexpr int kGrids[] = {1, 32, 108};
constexpr int kRepeatPoints[] = {1, 2, 3, 4, 6, 8, 11, 16, 22};

enum class Mode : int {
    kEmpty = 0,
    kFmaOnly = 1,
    kLdg128Fma = 2,
};

enum class Tier : int {
    kL2WarmCg = 0,
    kDramCg = 1,
};

struct Result {
    std::uint64_t cycles;
    std::uint32_t sm_id;
    std::uint32_t checksum;
};

static_assert(sizeof(Result) == 16, "result ABI must remain fixed");
static_assert(kTileBytes == 2048, "one CTA must read a 2 KiB float4 tile");

void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", expression,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(call) check_cuda((call), #call)

__device__ __forceinline__ std::uint32_t physical_smid() {
    std::uint32_t value;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
    return value;
}

__device__ __forceinline__ float4 load_global_cg_128(const float4* address) {
    float x;
    float y;
    float z;
    float w;
    asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                 : "l"(address)
                 : "memory");
    return make_float4(x, y, z, w);
}

__device__ __forceinline__ void four_dependent_fma(
        float4 values, float scale, float4* accumulator) {
    // Inline PTX fixes the intended four FP32 FMA consumers.  The values and
    // accumulator operands are live across the repeated loop, so the next
    // vector load cannot overwrite the prior values before they are consumed.
    asm volatile("fma.rn.f32 %0, %1, %2, %0;"
                 : "+f"(accumulator->x) : "f"(scale), "f"(values.x));
    asm volatile("fma.rn.f32 %0, %1, %2, %0;"
                 : "+f"(accumulator->y) : "f"(scale), "f"(values.y));
    asm volatile("fma.rn.f32 %0, %1, %2, %0;"
                 : "+f"(accumulator->z) : "f"(scale), "f"(values.z));
    asm volatile("fma.rn.f32 %0, %1, %2, %0;"
                 : "+f"(accumulator->w) : "f"(scale), "f"(values.w));
}

template <Tier tier>
__device__ __forceinline__ std::size_t tile_index(
        int repeat, int grid, int block, std::size_t tile_count) {
    const std::size_t ordinal =
        static_cast<std::size_t>(repeat) * grid + block;
    if constexpr (tier == Tier::kL2WarmCg) {
        return ordinal;
    } else {
        // 131071 is odd and the pool has 131072 2-KiB tiles.  This visits a
        // distinct, widely separated tile for every source-loop iteration;
        // the runner evicts L2 before each timed DRAM sample.
        return (ordinal * 131071ULL) % tile_count;
    }
}

template <Mode mode, Tier tier>
__global__ __launch_bounds__(kThreadsPerCta)
void fa2_oaccum_ldg128_load_use_kernel(
        const float4* data, Result* results, int repeats,
        std::size_t tile_count) {
    extern __shared__ std::uint8_t occupancy_guard[];
    __shared__ std::uint64_t begin_ticks[kThreadsPerCta];
    __shared__ std::uint64_t end_ticks[kThreadsPerCta];
    __shared__ std::uint32_t checksums[kThreadsPerCta];
    const int lane = threadIdx.x;
    if (lane == 0) {
        occupancy_guard[0] = static_cast<std::uint8_t>(blockIdx.x);
    }
    float4 accumulator = make_float4(
        0.03125f * (lane + 1), 0.0625f * (lane + 1),
        0.09375f * (lane + 1), 0.125f * (lane + 1));

    __syncthreads();
    const std::uint64_t begin = clock64();
    #pragma unroll 1
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const std::size_t tile = tile_index<tier>(
            repeat, gridDim.x, blockIdx.x, tile_count);
        const float4* address = data + tile * kThreadsPerCta + lane;
        const float scale = 1.0f + 0.0009765625f * (
            static_cast<float>((lane + repeat) & 31) + 1.0f);
        if constexpr (mode == Mode::kEmpty) {
            asm volatile("" : : "l"(address), "f"(scale) : "memory");
        } else if constexpr (mode == Mode::kFmaOnly) {
            const float4 values = make_float4(
                scale + 0.25f, scale + 0.5f, scale + 0.75f,
                scale + 1.0f);
            four_dependent_fma(values, scale, &accumulator);
        } else {
            const float4 values = load_global_cg_128(address);
            four_dependent_fma(values, scale, &accumulator);
        }
    }
    const std::uint64_t end = clock64();

    const std::uint32_t checksum =
        __float_as_uint(accumulator.x)
        ^ __float_as_uint(accumulator.y)
        ^ __float_as_uint(accumulator.z)
        ^ __float_as_uint(accumulator.w)
        ^ static_cast<std::uint32_t>(lane);
    begin_ticks[lane] = begin;
    end_ticks[lane] = end;
    checksums[lane] = checksum;
    __syncthreads();
    if (lane == 0) {
        std::uint64_t first = begin_ticks[0];
        std::uint64_t last = end_ticks[0];
        std::uint32_t combined = 0;
        #pragma unroll
        for (int index = 0; index < kThreadsPerCta; ++index) {
            first = min(first, begin_ticks[index]);
            last = max(last, end_ticks[index]);
            combined ^= checksums[index] + static_cast<std::uint32_t>(index);
        }
        results[blockIdx.x] = Result{last - first, physical_smid(), combined};
    }
}

__global__ void fill_noncompressible(float4* data, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (index >= count) {
        return;
    }
    const std::uint32_t base = static_cast<std::uint32_t>(index);
    data[index] = make_float4(
        __uint_as_float(0x3f000001U ^ (base * 747796405U)),
        __uint_as_float(0x3f100001U ^ (base * 2891336453U)),
        __uint_as_float(0x3f200001U ^ (base * 1181783497U)),
        __uint_as_float(0x3f300001U ^ (base * 42595009U)));
}

__global__ void evict_l2(std::uint32_t* eviction, std::size_t words,
                         std::uint32_t salt) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (index < words) {
        eviction[index] = static_cast<std::uint32_t>(index) ^ salt;
    }
}

template <Tier tier>
__global__ __launch_bounds__(kThreadsPerCta)
void warm_l2(const float4* data, std::size_t tile_count, int repeats,
             std::uint32_t* sink) {
    const int lane = threadIdx.x;
    std::uint32_t checksum = 0;
    #pragma unroll 1
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const std::size_t tile = tile_index<tier>(
            repeat, gridDim.x, blockIdx.x, tile_count);
        const float4 values = load_global_cg_128(
            data + tile * kThreadsPerCta + lane);
        checksum ^= __float_as_uint(values.x) ^ __float_as_uint(values.w);
    }
    if (lane == 0) {
        sink[blockIdx.x] = checksum;
    }
}

bool is_allowed(int value, const int* points, int count) {
    for (int index = 0; index < count; ++index) {
        if (value == points[index]) {
            return true;
        }
    }
    return false;
}

int parse_positive(const char* text, const char* name) {
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value <= 0
            || value > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "%s must be a positive integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

Mode parse_mode(const char* text) {
    if (std::strcmp(text, "empty") == 0) {
        return Mode::kEmpty;
    }
    if (std::strcmp(text, "fma_only") == 0) {
        return Mode::kFmaOnly;
    }
    if (std::strcmp(text, "ldg128_fma") == 0) {
        return Mode::kLdg128Fma;
    }
    std::fprintf(stderr, "mode must be empty, fma_only, or ldg128_fma\n");
    std::exit(EXIT_FAILURE);
}

Tier parse_tier(const char* text) {
    if (std::strcmp(text, "l2_warm_cg") == 0) {
        return Tier::kL2WarmCg;
    }
    if (std::strcmp(text, "dram_cg") == 0) {
        return Tier::kDramCg;
    }
    std::fprintf(stderr, "tier must be l2_warm_cg or dram_cg\n");
    std::exit(EXIT_FAILURE);
}

const char* mode_name(Mode mode) {
    if (mode == Mode::kEmpty) {
        return "empty";
    }
    if (mode == Mode::kFmaOnly) {
        return "fma_only";
    }
    if (mode == Mode::kLdg128Fma) {
        return "ldg128_fma";
    }
    std::abort();
}

const char* tier_name(Tier tier) {
    if (tier == Tier::kL2WarmCg) {
        return "l2_warm_cg";
    }
    if (tier == Tier::kDramCg) {
        return "dram_cg";
    }
    std::abort();
}

const char* kernel_name(Mode mode, Tier tier) {
    if (mode == Mode::kLdg128Fma && tier == Tier::kL2WarmCg) {
        return "fa2_oaccum_ldg128_fma_l2_warm_cg";
    }
    if (mode == Mode::kLdg128Fma && tier == Tier::kDramCg) {
        return "fa2_oaccum_ldg128_fma_dram_cg";
    }
    if (mode == Mode::kFmaOnly) {
        return "fa2_oaccum_fma_only";
    }
    return "fa2_oaccum_empty";
}

template <Mode mode, Tier tier>
void launch(const float4* data, Result* results, int grid, int repeats,
            std::size_t tile_count, cudaStream_t stream) {
    fa2_oaccum_ldg128_load_use_kernel<mode, tier>
        <<<grid, kThreadsPerCta, kDynamicSharedBytes, stream>>>(
            data, results, repeats, tile_count);
    CHECK_CUDA(cudaGetLastError());
}

template <Mode mode, Tier tier>
void configure_variant() {
    auto kernel = fa2_oaccum_ldg128_load_use_kernel<mode, tier>;
    CHECK_CUDA(cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kDynamicSharedBytes));
    int active_blocks = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, kernel, kThreadsPerCta, kDynamicSharedBytes));
    if (active_blocks != 1) {
        std::fprintf(stderr,
                     "mechanism probe must admit exactly one CTA/SM; got %d\n",
                     active_blocks);
        std::exit(EXIT_FAILURE);
    }
}

void configure_variants() {
    configure_variant<Mode::kEmpty, Tier::kL2WarmCg>();
    configure_variant<Mode::kFmaOnly, Tier::kL2WarmCg>();
    configure_variant<Mode::kLdg128Fma, Tier::kL2WarmCg>();
    configure_variant<Mode::kEmpty, Tier::kDramCg>();
    configure_variant<Mode::kFmaOnly, Tier::kDramCg>();
    configure_variant<Mode::kLdg128Fma, Tier::kDramCg>();
}

void dispatch(Mode mode, Tier tier, const float4* data, Result* results,
              int grid, int repeats, std::size_t tile_count,
              cudaStream_t stream) {
    if (tier == Tier::kL2WarmCg) {
        if (mode == Mode::kEmpty) {
            launch<Mode::kEmpty, Tier::kL2WarmCg>(
                data, results, grid, repeats, tile_count, stream);
        } else if (mode == Mode::kFmaOnly) {
            launch<Mode::kFmaOnly, Tier::kL2WarmCg>(
                data, results, grid, repeats, tile_count, stream);
        } else {
            launch<Mode::kLdg128Fma, Tier::kL2WarmCg>(
                data, results, grid, repeats, tile_count, stream);
        }
    } else if (tier == Tier::kDramCg) {
        if (mode == Mode::kEmpty) {
            launch<Mode::kEmpty, Tier::kDramCg>(
                data, results, grid, repeats, tile_count, stream);
        } else if (mode == Mode::kFmaOnly) {
            launch<Mode::kFmaOnly, Tier::kDramCg>(
                data, results, grid, repeats, tile_count, stream);
        } else {
            launch<Mode::kLdg128Fma, Tier::kDramCg>(
                data, results, grid, repeats, tile_count, stream);
        }
    } else {
        std::abort();
    }
}

void warm_target(Tier tier, const float4* data, std::size_t tile_count,
                 int grid, int repeats, std::uint32_t* sink) {
    if (tier == Tier::kL2WarmCg) {
        warm_l2<Tier::kL2WarmCg><<<grid, kThreadsPerCta>>>(
            data, tile_count, repeats, sink);
    } else {
        warm_l2<Tier::kDramCg><<<grid, kThreadsPerCta>>>(
            data, tile_count, repeats, sink);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

void validate_results(const std::vector<Result>& results, int grid) {
    bool seen[kExpectedSmCount] = {};
    for (int block = 0; block < grid; ++block) {
        const Result result = results[block];
        if (result.cycles == 0 || result.sm_id >= kExpectedSmCount) {
            std::fprintf(stderr, "invalid clock64 result at CTA %d\n", block);
            std::exit(EXIT_FAILURE);
        }
        if (seen[result.sm_id]) {
            std::fprintf(stderr,
                         "CTA grid did not map one timed CTA per SM: SM %u\n",
                         result.sm_id);
            std::exit(EXIT_FAILURE);
        }
        seen[result.sm_id] = true;
    }
}

void print_usage(const char* executable) {
    std::fprintf(stderr,
                 "usage: %s MODE GRID REPEATS SAMPLES TIER\n"
                 "  MODE: empty|fma_only|ldg128_fma\n"
                 "  GRID: 1|32|108; REPEATS: 1|2|3|4|6|8|11|16|22\n"
                 "  TIER: l2_warm_cg|dram_cg\n",
                 executable);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    const Mode mode = parse_mode(argv[1]);
    const int grid = parse_positive(argv[2], "grid");
    const int repeats = parse_positive(argv[3], "repeats");
    const int samples = parse_positive(argv[4], "samples");
    const Tier tier = parse_tier(argv[5]);
    if (!is_allowed(grid, kGrids, sizeof(kGrids) / sizeof(kGrids[0]))) {
        std::fprintf(stderr, "grid is outside the frozen mechanism matrix\n");
        return EXIT_FAILURE;
    }
    if (!is_allowed(repeats, kRepeatPoints,
                    sizeof(kRepeatPoints) / sizeof(kRepeatPoints[0]))) {
        std::fprintf(stderr, "repeats is outside the frozen mechanism matrix\n");
        return EXIT_FAILURE;
    }

    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
    if (properties.major != kExpectedMajor || properties.minor != kExpectedMinor
            || properties.warpSize != 32
            || properties.multiProcessorCount != kExpectedSmCount) {
        std::fprintf(stderr,
                     "expected a 108-SM sm_80 A800; got sm_%d%d, warp%d, %d SM\n",
                     properties.major, properties.minor, properties.warpSize,
                     properties.multiProcessorCount);
        return EXIT_FAILURE;
    }
    configure_variants();

    const std::size_t data_bytes = tier == Tier::kL2WarmCg
        ? kL2WarmBytes : kDramPoolBytes;
    const std::size_t data_count = data_bytes / sizeof(float4);
    const std::size_t tile_count = data_bytes / kTileBytes;
    if (tile_count < static_cast<std::size_t>(grid) * repeats) {
        std::fprintf(stderr, "data pool does not cover the source loop\n");
        return EXIT_FAILURE;
    }
    float4* data = nullptr;
    Result* device_results = nullptr;
    std::uint32_t* warm_sink = nullptr;
    std::uint32_t* eviction = nullptr;
    CHECK_CUDA(cudaMalloc(&data, data_bytes));
    CHECK_CUDA(cudaMalloc(&device_results, grid * sizeof(Result)));
    CHECK_CUDA(cudaMalloc(&warm_sink, grid * sizeof(std::uint32_t)));
    CHECK_CUDA(cudaMalloc(&eviction, kEvictionBytes));
    const int fill_threads = 256;
    const int fill_blocks = static_cast<int>((data_count + fill_threads - 1)
        / fill_threads);
    fill_noncompressible<<<fill_blocks, fill_threads>>>(data, data_count);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t event_begin;
    cudaEvent_t event_end;
    CHECK_CUDA(cudaEventCreate(&event_begin));
    CHECK_CUDA(cudaEventCreate(&event_end));
    std::vector<Result> host_results(grid);

    // Warm-up is intentionally outside the capture.  It exercises the exact
    // launch geometry but its time never appears in CSV or report parameters.
    for (int warmup = 0; warmup < 5; ++warmup) {
        if (tier == Tier::kDramCg) {
            const std::size_t words = kEvictionBytes / sizeof(std::uint32_t);
            evict_l2<<<static_cast<int>((words + fill_threads - 1) / fill_threads),
                       fill_threads>>>(eviction, words,
                                        static_cast<std::uint32_t>(warmup));
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            warm_target(tier, data, tile_count, grid, repeats, warm_sink);
        }
        dispatch(mode, tier, data, device_results, grid, repeats, tile_count,
                 nullptr);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    std::printf(
        "sample,mode,tier,kernel,grid,repeats,block,sm_id,cycles,checksum,event_ns\n");
    for (int sample = 0; sample < samples; ++sample) {
        if (tier == Tier::kDramCg) {
            const std::size_t words = kEvictionBytes / sizeof(std::uint32_t);
            evict_l2<<<static_cast<int>((words + fill_threads - 1) / fill_threads),
                       fill_threads>>>(eviction, words,
                                        static_cast<std::uint32_t>(sample + 17));
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            warm_target(tier, data, tile_count, grid, repeats, warm_sink);
        }
        CHECK_CUDA(cudaEventRecord(event_begin));
        dispatch(mode, tier, data, device_results, grid, repeats, tile_count,
                 nullptr);
        CHECK_CUDA(cudaEventRecord(event_end));
        CHECK_CUDA(cudaEventSynchronize(event_end));
        float event_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&event_ms, event_begin, event_end));
        CHECK_CUDA(cudaMemcpy(host_results.data(), device_results,
                              grid * sizeof(Result), cudaMemcpyDeviceToHost));
        validate_results(host_results, grid);
        const double event_ns = static_cast<double>(event_ms) * 1.0e6;
        if (!(event_ns > 0.0)) {
            std::fprintf(stderr, "CUDA event did not report a positive interval\n");
            return EXIT_FAILURE;
        }
        for (int block = 0; block < grid; ++block) {
            const Result result = host_results[block];
            std::printf("%d,%s,%s,%s,%d,%d,%d,%u,%llu,%u,%.3f\n",
                        sample, mode_name(mode), tier_name(tier),
                        kernel_name(mode, tier), grid, repeats, block,
                        result.sm_id,
                        static_cast<unsigned long long>(result.cycles),
                        result.checksum, event_ns);
        }
    }

    CHECK_CUDA(cudaEventDestroy(event_end));
    CHECK_CUDA(cudaEventDestroy(event_begin));
    CHECK_CUDA(cudaFree(eviction));
    CHECK_CUDA(cudaFree(warm_sink));
    CHECK_CUDA(cudaFree(device_results));
    CHECK_CUDA(cudaFree(data));
    return EXIT_SUCCESS;
}
