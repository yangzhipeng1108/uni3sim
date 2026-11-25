/*
 * MetaX C500 four-wave, eight-output reduction-completion benchmark.
 *
 * The retained b16gemvt<256, 8, 4, 8> launch uses four wave64 groups and
 * 144 bytes of static shared memory.  Each wave reduces eight independent
 * FP32 accumulators, publishes its 4 x 8 partials through the 128-byte shared
 * payload, and the first wave completes the eight outputs.  The remaining
 * 16 shared bytes preserve the source-visible allocation envelope.
 *
 * This benchmark contains no GEMV operand, matrix multiply, production shape,
 * or production/evaluation duration.  Adjacent runtime repetition slopes of
 * a same-register control remove launch, seed-update, and terminal-store work.
 */

#include <mc_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CHECK(call) do {                                                   \
    mcError_t error = (call);                                              \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",                 \
                     __FILE__, __LINE__, mcGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

namespace {

constexpr int kExpectedApCount = 104;
constexpr int kExpectedWaveSize = 64;
constexpr int kBlockThreads = 256;
constexpr int kWavesPerCta = kBlockThreads / kExpectedWaveSize;
constexpr int kOutputsPerCta = 8;
constexpr int kSharedFloats = 36;
constexpr int kWarmupsPerCell = 4;
constexpr int kSamplesPerCell = 31;
constexpr int kLaunchesPerBatch = 16;
constexpr unsigned long long kWaveMask = 0xffffffffffffffffULL;

static_assert(kWavesPerCta == 4);
static_assert(kSharedFloats * sizeof(float) == 144);

enum class Mode {
    kControl,
    kReduce8,
};

const char* mode_name(Mode mode) {
    switch (mode) {
        case Mode::kControl: return "control";
        case Mode::kReduce8: return "reduce8";
        default:
            std::fprintf(stderr, "unknown mode\n");
            std::exit(EXIT_FAILURE);
    }
}

template <bool reduce>
__global__ void block256_wave8_completion(float* output, int repetitions) {
    __shared__ __align__(16) float shared[kSharedFloats];
    float values[kOutputsPerCta];
    #pragma unroll
    for (int item = 0; item < kOutputsPerCta; ++item) {
        values[item] = (
            1.0F + static_cast<float>(threadIdx.x + item) * 0.0009765625F);
    }

    const int lane = threadIdx.x % kExpectedWaveSize;
    const int wave = threadIdx.x / kExpectedWaveSize;
    #pragma unroll 1
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        #pragma unroll
        for (int item = 0; item < kOutputsPerCta; ++item) {
            values[item] += (
                static_cast<float>(item + 1) * 0.00000011920928955078125F);
        }
        if (reduce) {
            #pragma unroll
            for (int offset = kExpectedWaveSize / 2; offset > 0; offset /= 2) {
                #pragma unroll
                for (int item = 0; item < kOutputsPerCta; ++item) {
                    const float peer = __shfl_down_sync(
                        kWaveMask, values[item], offset, kExpectedWaveSize);
                    if (lane + offset < kExpectedWaveSize) {
                        values[item] += peer;
                    }
                }
            }
            if (lane == 0) {
                #pragma unroll
                for (int item = 0; item < kOutputsPerCta; ++item) {
                    shared[wave * kOutputsPerCta + item] = values[item];
                }
            }
            __syncthreads();
            if (threadIdx.x < kOutputsPerCta) {
                float combined = shared[threadIdx.x];
                #pragma unroll
                for (int source_wave = 1;
                     source_wave < kWavesPerCta; ++source_wave) {
                    combined += shared[
                        source_wave * kOutputsPerCta + threadIdx.x];
                }
                shared[threadIdx.x] = combined;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                #pragma unroll
                for (int item = 0; item < kOutputsPerCta; ++item) {
                    values[item] = shared[item];
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int item = 0; item < kOutputsPerCta; ++item) {
            values[item] *= 0.015625F;
        }
    }

    if (threadIdx.x == 0) {
        const long long base = (
            static_cast<long long>(blockIdx.x) * kOutputsPerCta);
        #pragma unroll
        for (int item = 0; item < kOutputsPerCta; ++item) {
            output[base + item] = values[item];
        }
    }
}

struct Cell {
    Mode mode;
    int grid;
    int repetitions;
    int sample;
};

struct KernelResources {
    int registers_per_thread;
    std::size_t shared_bytes_per_cta;
    int resident_ctas_per_ap;
};

int parse_nonnegative_int(const char* text, const char* label) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > 2147483647L) {
        std::fprintf(stderr, "%s must be a non-negative integer\n", label);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

template <bool reduce>
KernelResources resources() {
    mcFuncAttributes attributes {};
    CHECK(mcFuncGetAttributes(
        &attributes,
        reinterpret_cast<const void*>(block256_wave8_completion<reduce>)));
    int resident = 0;
    CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, block256_wave8_completion<reduce>, kBlockThreads, 0));
    return {
        attributes.numRegs,
        attributes.sharedSizeBytes,
        resident,
    };
}

void launch(Mode mode, int grid, int repetitions, float* output) {
    switch (mode) {
        case Mode::kControl:
            block256_wave8_completion<false><<<grid, kBlockThreads>>>(
                output, repetitions);
            return;
        case Mode::kReduce8:
            block256_wave8_completion<true><<<grid, kBlockThreads>>>(
                output, repetitions);
            return;
        default:
            std::fprintf(stderr, "unknown mode\n");
            std::exit(EXIT_FAILURE);
    }
}

void launch_batch(Mode mode, int grid, int repetitions, float* output) {
    for (int launch_index = 0;
         launch_index < kLaunchesPerBatch; ++launch_index) {
        launch(mode, grid, repetitions, output);
    }
    CHECK(mcGetLastError());
}

float record_batch(
        mcEvent_t start, mcEvent_t end, Mode mode, int grid,
        int repetitions, float* output) {
    launch_batch(mode, grid, repetitions, output);
    CHECK(mcEventRecord(start));
    launch_batch(mode, grid, repetitions, output);
    CHECK(mcEventRecord(end));
    CHECK(mcDeviceSynchronize());
    float elapsed_ms = 0.0F;
    CHECK(mcEventElapsedTime(&elapsed_ms, start, end));
    if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0F) {
        std::fprintf(stderr, "invalid mcEvent duration: %.9g ms\n", elapsed_ms);
        std::exit(EXIT_FAILURE);
    }
    return elapsed_ms;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5 || std::string(argv[1]) != "--device-id"
            || std::string(argv[3]) != "--trial") {
        std::fprintf(
            stderr, "usage: %s --device-id DEVICE_ID --trial TRIAL\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int device_id = parse_nonnegative_int(argv[2], "device-id");
    const int trial = parse_nonnegative_int(argv[4], "trial");
    if (trial < 1 || trial > 3) {
        std::fprintf(stderr, "trial must be one of 1, 2, or 3\n");
        return EXIT_FAILURE;
    }
    CHECK(mcSetDevice(device_id));
    mcDeviceProp_t properties {};
    CHECK(mcGetDeviceProperties(&properties, device_id));
    if (properties.multiProcessorCount != kExpectedApCount
            || properties.warpSize != kExpectedWaveSize
            || properties.clockRate <= 0) {
        std::fprintf(
            stderr, "expected C500 104 AP/wave64 and a positive clock\n");
        return EXIT_FAILURE;
    }

    const KernelResources control = resources<false>();
    const KernelResources reduce8 = resources<true>();
    if (control.resident_ctas_per_ap <= 0
            || reduce8.resident_ctas_per_ap <= 0
            || reduce8.shared_bytes_per_cta != 144) {
        std::fprintf(stderr, "invalid block-256 completion resources\n");
        return EXIT_FAILURE;
    }

    constexpr int grids[] = {
        1, 104, 192, 208, 416, 640, 832, 1664,
    };
    constexpr int repetitions[] = {4, 8, 16, 32};
    constexpr Mode modes[] = {Mode::kControl, Mode::kReduce8};
    constexpr int maximum_grid = 1664;
    float* output = nullptr;
    CHECK(mcMalloc(
        reinterpret_cast<void**>(&output),
        static_cast<std::size_t>(maximum_grid) * kOutputsPerCta
        * sizeof(float)));

    for (Mode mode : modes) {
        for (int grid : grids) {
            for (int repeat : repetitions) {
                for (int warmup = 0; warmup < kWarmupsPerCell; ++warmup) {
                    launch(mode, grid, repeat, output);
                }
            }
        }
    }
    CHECK(mcGetLastError());
    CHECK(mcDeviceSynchronize());

    std::vector<Cell> plan;
    for (Mode mode : modes) {
        for (int grid : grids) {
            for (int repeat : repetitions) {
                for (int sample = 0; sample < kSamplesPerCell; ++sample) {
                    plan.push_back({mode, grid, repeat, sample});
                }
            }
        }
    }
    std::mt19937 generator(511003U + 7919U * static_cast<unsigned>(trial));
    std::shuffle(plan.begin(), plan.end(), generator);

    std::printf(
        "metadata,1,MetaX C500,%d,%d,%d,%d,%d,%d,%d,%d,%d,%zu,%zu,%d,%d\n",
        device_id, properties.multiProcessorCount, properties.warpSize,
        properties.clockRate, trial, kSamplesPerCell, kLaunchesPerBatch,
        control.registers_per_thread, reduce8.registers_per_thread,
        control.shared_bytes_per_cta, reduce8.shared_bytes_per_cta,
        control.resident_ctas_per_ap, reduce8.resident_ctas_per_ap);

    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    for (std::size_t order = 0; order < plan.size(); ++order) {
        const Cell& item = plan[order];
        const float duration_ms = record_batch(
            start, end, item.mode, item.grid, item.repetitions, output);
        std::printf(
            "sample,%zu,%s,%d,%d,%d,%a\n",
            order, mode_name(item.mode), item.grid, item.repetitions,
            item.sample, static_cast<double>(duration_ms));
    }
    CHECK(mcEventDestroy(end));
    CHECK(mcEventDestroy(start));

    launch(Mode::kReduce8, maximum_grid, 4, output);
    CHECK(mcDeviceSynchronize());
    std::vector<float> host_output(
        static_cast<std::size_t>(maximum_grid) * kOutputsPerCta);
    CHECK(mcMemcpy(
        host_output.data(), output,
        host_output.size() * sizeof(float), mcMemcpyDeviceToHost));
    for (std::size_t index = 0; index < host_output.size(); ++index) {
        if (!std::isfinite(host_output[index])) {
            std::fprintf(stderr, "invalid output at index %zu\n", index);
            return EXIT_FAILURE;
        }
    }
    CHECK(mcFree(output));
    return EXIT_SUCCESS;
}
