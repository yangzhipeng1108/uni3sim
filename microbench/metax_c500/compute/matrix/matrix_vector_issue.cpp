/*
 * MetaX C500 register-only MMA/Vector-ALU issue-overlap benchmark.
 *
 * The C500 architecture assigns MMA and Vector ALU instructions to one issue
 * slot.  That statement alone does not say whether the slot remains occupied
 * throughout the four-cycle saturated MMA initiation interval or can issue
 * independent Vector ALU work between MMA instructions.  This benchmark
 * distinguishes those two mechanisms with full-chip, register-only streams:
 * MMA-only, add_f32-only, and fixed 4-MMA plus 4/8/12/16-add mixtures.
 *
 * No production operator, global-memory traffic inside the timed loop,
 * profiler counter, or evaluation duration is used.  The final stores only
 * keep every accumulator live and occur after the timed instruction stream.
 *
 * Build in the prescribed MACA 3.7.2 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch=native \
 *       --maca-path=/opt/maca -O3 matrix_vector_issue.cpp \
 *       -o matrix_vector_issue
 */

#include <mc_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(call) do {                                                   \
    mcError_t error = (call);                                              \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",               \
                     __FILE__, __LINE__, mcGetErrorString(error));          \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

namespace {

constexpr int kExpectedApCount = 104;
constexpr int kExpectedWaveSize = 64;
constexpr int kBlockThreads = 64;
constexpr int kWavesPerAp = 32;
constexpr int kIterations = 32768;
constexpr int kWarmups = 3;
constexpr int kTrials = 9;
constexpr double kXcoreClockHz = 1125000000.0;

typedef __NATIVE_VECTOR__(2, int32_t) v2i;
typedef __NATIVE_VECTOR__(4, float) v4f;

struct Sink {
    v4f matrix;
    float vector;
};

__device__ __forceinline__ v4f mma_bf16(v2i a, v2i b, v4f c) {
    return __builtin_mxc_mma_16x16x16bf16(a, b, c);
}

__device__ __forceinline__ float add_f32(float lhs, float rhs) {
    float output;
    asm volatile("add_f32 %0, %1, %2"
                 : "=r"(output) : "r"(lhs), "r"(rhs));
    return output;
}

template <int count>
struct AddBatch {
    __device__ __forceinline__ static void run(
            float* accumulators, float increment) {
        AddBatch<count - 1>::run(accumulators, increment);
        accumulators[count - 1] = add_f32(
            accumulators[count - 1], increment);
    }
};

template <>
struct AddBatch<0> {
    __device__ __forceinline__ static void run(float*, float) {}
};

template <int adds_per_mma>
__global__ void mixed_issue_kernel(Sink* sink, int iterations) {
    static_assert(adds_per_mma >= 1 && adds_per_mma <= 4,
                  "adds_per_mma must be in [1, 4]");
    const v2i a = {0x3b803c00, 0x3a803b00};
    const v2i b = {0x3b003a80, 0x3c003b80};
    v4f c0 = {0.125f, 0.25f, 0.5f, 1.0f};
    v4f c1 = {0.25f, 0.5f, 1.0f, 0.125f};
    v4f c2 = {0.5f, 1.0f, 0.125f, 0.25f};
    v4f c3 = {1.0f, 0.125f, 0.25f, 0.5f};
    float values[16] = {
        1.0001f, 1.0002f, 1.0003f, 1.0004f,
        1.0005f, 1.0006f, 1.0007f, 1.0008f,
        1.0009f, 1.0010f, 1.0011f, 1.0012f,
        1.0013f, 1.0014f, 1.0015f, 1.0016f,
    };
    const float increment = 0.00000011920928955078125f;
    #pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        c0 = mma_bf16(a, b, c0);
        AddBatch<adds_per_mma>::run(values, increment);
        c1 = mma_bf16(a, b, c1);
        AddBatch<adds_per_mma>::run(values + 4, increment);
        c2 = mma_bf16(a, b, c2);
        AddBatch<adds_per_mma>::run(values + 8, increment);
        c3 = mma_bf16(a, b, c3);
        AddBatch<adds_per_mma>::run(values + 12, increment);
    }
    const long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    sink[thread].matrix = {
        c0.x + c1.x + c2.x + c3.x,
        c0.y + c1.y + c2.y + c3.y,
        c0.z + c1.z + c2.z + c3.z,
        c0.w + c1.w + c2.w + c3.w,
    };
    float vector_sum = 0.0f;
    #pragma unroll
    for (int index = 0; index < 16; ++index) {
        vector_sum += values[index];
    }
    sink[thread].vector = vector_sum;
}

__global__ void mma_only_kernel(Sink* sink, int iterations) {
    const v2i a = {0x3b803c00, 0x3a803b00};
    const v2i b = {0x3b003a80, 0x3c003b80};
    v4f c0 = {0.125f, 0.25f, 0.5f, 1.0f};
    v4f c1 = {0.25f, 0.5f, 1.0f, 0.125f};
    v4f c2 = {0.5f, 1.0f, 0.125f, 0.25f};
    v4f c3 = {1.0f, 0.125f, 0.25f, 0.5f};
    #pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        c0 = mma_bf16(a, b, c0);
        c1 = mma_bf16(a, b, c1);
        c2 = mma_bf16(a, b, c2);
        c3 = mma_bf16(a, b, c3);
    }
    const long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    sink[thread].matrix = {
        c0.x + c1.x + c2.x + c3.x,
        c0.y + c1.y + c2.y + c3.y,
        c0.z + c1.z + c2.z + c3.z,
        c0.w + c1.w + c2.w + c3.w,
    };
    sink[thread].vector = c0.x;
}

template <int add_count>
__global__ void add_only_kernel(Sink* sink, int iterations) {
    static_assert(add_count == 4 || add_count == 8
                  || add_count == 12 || add_count == 16,
                  "unsupported add count");
    float values[16] = {
        1.0001f, 1.0002f, 1.0003f, 1.0004f,
        1.0005f, 1.0006f, 1.0007f, 1.0008f,
        1.0009f, 1.0010f, 1.0011f, 1.0012f,
        1.0013f, 1.0014f, 1.0015f, 1.0016f,
    };
    const float increment = 0.00000011920928955078125f;
    #pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        AddBatch<add_count>::run(values, increment);
    }
    const long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    float vector_sum = 0.0f;
    #pragma unroll
    for (int index = 0; index < add_count; ++index) {
        vector_sum += values[index];
    }
    sink[thread].matrix = {vector_sum, vector_sum, vector_sum, vector_sum};
    sink[thread].vector = vector_sum;
}

enum class Mode {
    kMmaOnly,
    kAdd4,
    kAdd8,
    kAdd12,
    kAdd16,
    kMixed4Add4,
    kMixed4Add8,
    kMixed4Add12,
    kMixed4Add16,
};

const char* mode_name(Mode mode) {
    switch (mode) {
        case Mode::kMmaOnly: return "mma4";
        case Mode::kAdd4: return "add4";
        case Mode::kAdd8: return "add8";
        case Mode::kAdd12: return "add12";
        case Mode::kAdd16: return "add16";
        case Mode::kMixed4Add4: return "mixed_mma4_add4";
        case Mode::kMixed4Add8: return "mixed_mma4_add8";
        case Mode::kMixed4Add12: return "mixed_mma4_add12";
        case Mode::kMixed4Add16: return "mixed_mma4_add16";
        default:
            std::fprintf(stderr, "unknown mode\n");
            std::exit(EXIT_FAILURE);
    }
}

int mma_count(Mode mode) {
    return mode == Mode::kMmaOnly ? 4
        : mode >= Mode::kMixed4Add4 ? 4 : 0;
}

int add_count(Mode mode) {
    switch (mode) {
        case Mode::kMmaOnly: return 0;
        case Mode::kAdd4:
        case Mode::kMixed4Add4: return 4;
        case Mode::kAdd8:
        case Mode::kMixed4Add8: return 8;
        case Mode::kAdd12:
        case Mode::kMixed4Add12: return 12;
        case Mode::kAdd16:
        case Mode::kMixed4Add16: return 16;
        default:
            std::fprintf(stderr, "unknown mode\n");
            std::exit(EXIT_FAILURE);
    }
}

void launch(Mode mode, int grid, Sink* sink, int iterations) {
    switch (mode) {
        case Mode::kMmaOnly:
            mma_only_kernel<<<grid, kBlockThreads>>>(sink, iterations); return;
        case Mode::kAdd4:
            add_only_kernel<4><<<grid, kBlockThreads>>>(sink, iterations); return;
        case Mode::kAdd8:
            add_only_kernel<8><<<grid, kBlockThreads>>>(sink, iterations); return;
        case Mode::kAdd12:
            add_only_kernel<12><<<grid, kBlockThreads>>>(sink, iterations); return;
        case Mode::kAdd16:
            add_only_kernel<16><<<grid, kBlockThreads>>>(sink, iterations); return;
        case Mode::kMixed4Add4:
            mixed_issue_kernel<1><<<grid, kBlockThreads>>>(sink, iterations); return;
        case Mode::kMixed4Add8:
            mixed_issue_kernel<2><<<grid, kBlockThreads>>>(sink, iterations); return;
        case Mode::kMixed4Add12:
            mixed_issue_kernel<3><<<grid, kBlockThreads>>>(sink, iterations); return;
        case Mode::kMixed4Add16:
            mixed_issue_kernel<4><<<grid, kBlockThreads>>>(sink, iterations); return;
        default:
            std::fprintf(stderr, "unknown mode\n");
            std::exit(EXIT_FAILURE);
    }
}

double median_elapsed_ms(Mode mode, int grid, Sink* sink) {
    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        launch(mode, grid, sink, std::max(1, kIterations / 64));
    }
    CHECK(mcDeviceSynchronize());
    std::vector<double> samples;
    samples.reserve(kTrials);
    for (int trial = 0; trial < kTrials; ++trial) {
        CHECK(mcEventRecord(start));
        launch(mode, grid, sink, kIterations);
        CHECK(mcEventRecord(end));
        CHECK(mcDeviceSynchronize());
        float elapsed_ms = 0.0f;
        CHECK(mcEventElapsedTime(&elapsed_ms, start, end));
        if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f) {
            std::fprintf(stderr, "invalid elapsed time %.9g ms\n", elapsed_ms);
            std::exit(EXIT_FAILURE);
        }
        samples.push_back(elapsed_ms);
    }
    CHECK(mcEventDestroy(start));
    CHECK(mcEventDestroy(end));
    std::sort(samples.begin(), samples.end());
    return samples[kTrials / 2];
}

int parse_nonnegative_int(const char* text, const char* label) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > 2147483647L) {
        std::fprintf(stderr, "%s must be a non-negative integer\n", label);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 || std::string(argv[1]) != "--device-id") {
        std::fprintf(stderr, "usage: %s --device-id DEVICE_ID\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device_id = parse_nonnegative_int(argv[2], "device-id");
    CHECK(mcSetDevice(device_id));
    mcDeviceProp_t properties {};
    CHECK(mcGetDeviceProperties(&properties, device_id));
    if (properties.multiProcessorCount != kExpectedApCount
            || properties.warpSize != kExpectedWaveSize) {
        std::fprintf(
            stderr, "expected C500 104 AP/wave64, got %d AP/wave%d\n",
            properties.multiProcessorCount, properties.warpSize);
        return EXIT_FAILURE;
    }
    const int grid = kExpectedApCount * kWavesPerAp;
    Sink* sink = nullptr;
    CHECK(mcMalloc(
        &sink, static_cast<std::size_t>(grid) * kBlockThreads * sizeof(Sink)));

    const Mode modes[] = {
        Mode::kMmaOnly,
        Mode::kAdd4, Mode::kAdd8, Mode::kAdd12, Mode::kAdd16,
        Mode::kMixed4Add4, Mode::kMixed4Add8,
        Mode::kMixed4Add12, Mode::kMixed4Add16,
    };
    std::printf(
        "mode,grid,block,waves_per_ap,iterations,mma_per_iteration,"
        "add_per_iteration,median_ms,mma_per_ap_cycle,add_per_ap_cycle,"
        "slot_instructions_per_ap_cycle\n");
    for (Mode mode : modes) {
        const double elapsed_ms = median_elapsed_ms(mode, grid, sink);
        const double device_cycles = elapsed_ms * 1.0e-3 * kXcoreClockHz;
        const double wave_scale = (
            static_cast<double>(grid) * kIterations
            / kExpectedApCount / device_cycles);
        const int matrix_instructions = mma_count(mode);
        const int vector_instructions = add_count(mode);
        std::printf(
            "%s,%d,%d,%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f\n",
            mode_name(mode), grid, kBlockThreads, kWavesPerAp, kIterations,
            matrix_instructions, vector_instructions, elapsed_ms,
            wave_scale * matrix_instructions,
            wave_scale * vector_instructions,
            wave_scale * (matrix_instructions + vector_instructions));
    }
    std::fflush(stdout);
    CHECK(mcFree(sink));
    return EXIT_SUCCESS;
}
