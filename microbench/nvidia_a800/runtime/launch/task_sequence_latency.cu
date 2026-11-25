/*
 * Operator-independent SM80 task-boundary and kernel-sequence probe.
 *
 * CUDA Events bracket zero or more identical kernels.  The queued condition
 * places a long same-stream blocker before the start Event, giving the host
 * enough time to submit the whole measured sequence before the Event executes.
 * The idle condition starts from a synchronized stream.  Comparing the two
 * conditions separates GPU command processing from host submission exposure.
 */

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

constexpr int kThreads = 128;
constexpr int kSamples = 50;
constexpr int kTrials = 3;
constexpr unsigned long long kClockWarmupCycles = 1140000000ULL;
constexpr unsigned long long kQueueBlockerCycles = 5000000ULL;
constexpr int kFaMainDynamicSharedMemoryBytes = 81920;

void check(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", expression,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(call) check((call), #call)

int parse_int(const char* text, const char* name) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0
            || value > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "%s must be a nonnegative integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

__global__ void empty_task() {}

__global__ void dependent_task(
        const std::uint32_t* input, std::uint32_t* output) {
    const int index = threadIdx.x;
    output[index] = input[index] + 1U;
}

__global__ void fa_main_geometry_task(std::uint32_t* output) {
    extern __shared__ std::uint32_t shared[];
    const int block = blockIdx.x
        + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    if (threadIdx.x == 0) {
        shared[0] = static_cast<std::uint32_t>(block + 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        output[block] = shared[0];
    }
}

__global__ void fa_combine_geometry_task(std::uint32_t* values) {
    if (threadIdx.x == 0) {
        const int begin = blockIdx.x * 4;
        values[64 + blockIdx.x] = values[begin] + values[begin + 1]
            + values[begin + 2] + values[begin + 3];
    }
}

__global__ void clock_spin(
        unsigned long long cycles, float* output) {
    float value = 1.0009765625f;
    const unsigned long long begin = clock64();
    do {
        value = value * 0.9999847412109375f + 0.0000152587890625f;
    } while (clock64() - begin < cycles);
    if (threadIdx.x == 0) {
        output[blockIdx.x] = value;
    }
}

void launch_program(
        int program, const std::uint32_t* input, std::uint32_t* output) {
    if (program == 0) {
        empty_task<<<1, kThreads>>>();
    } else if (program == 1) {
        dependent_task<<<1, kThreads>>>(input, output);
    } else {
        std::fprintf(stderr, "unknown program %d\n", program);
        std::exit(EXIT_FAILURE);
    }
}

void launch_geometry_sequence(int sequence, std::uint32_t* values) {
    if (sequence == 0) {
        return;
    }
    if (sequence == 1 || sequence == 3) {
        fa_main_geometry_task<<<dim3(1, 4, 8), kThreads,
            kFaMainDynamicSharedMemoryBytes>>>(values);
    } else if (sequence == 2 || sequence == 4) {
        fa_combine_geometry_task<<<8, kThreads>>>(values);
    } else {
        std::fprintf(stderr, "unknown geometry sequence %d\n", sequence);
        std::exit(EXIT_FAILURE);
    }
    if (sequence == 3) {
        fa_combine_geometry_task<<<8, kThreads>>>(values);
    } else if (sequence == 4) {
        fa_main_geometry_task<<<dim3(1, 4, 8), kThreads,
            kFaMainDynamicSharedMemoryBytes>>>(values);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s DEVICE\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device = parse_int(argv[1], "device");
    CHECK(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CHECK(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 8 || properties.minor != 0
            || properties.multiProcessorCount != 108) {
        std::fprintf(stderr, "probe requires the 108-SM sm_80 target\n");
        return EXIT_FAILURE;
    }
    CHECK(cudaFuncSetAttribute(
        fa_main_geometry_task,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kFaMainDynamicSharedMemoryBytes));

    std::uint32_t* input = nullptr;
    std::uint32_t* output = nullptr;
    float* warmup_output = nullptr;
    CHECK(cudaMalloc(&input, kThreads * sizeof(*input)));
    CHECK(cudaMalloc(&output, kThreads * sizeof(*output)));
    CHECK(cudaMalloc(&warmup_output, 864 * sizeof(*warmup_output)));
    CHECK(cudaMemset(input, 0x5a, kThreads * sizeof(*input)));
    CHECK(cudaMemset(output, 0, kThreads * sizeof(*output)));

    clock_spin<<<864, 32>>>(kClockWarmupCycles, warmup_output);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    for (int warmup = 0; warmup < 200; ++warmup) {
        empty_task<<<1, kThreads>>>();
        dependent_task<<<1, kThreads>>>(input, output);
    }
    for (int warmup = 0; warmup < 20; ++warmup) {
        launch_geometry_sequence(3, output);
        launch_geometry_sequence(4, output);
    }
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    std::printf("meta,device,%d\n", device);
    std::printf("meta,device_name,%s\n", properties.name);
    std::printf("meta,sm_count,%d\n", properties.multiProcessorCount);
    std::printf("meta,threads,%d\n", kThreads);
    std::printf("meta,samples,%d\n", kSamples);
    std::printf("meta,trials,%d\n", kTrials);
    std::printf("meta,queue_blocker_cycles,%llu\n", kQueueBlockerCycles);
    std::printf(
        "raw,trial,condition,program,kernel_count,sample,"
        "event_duration_ns,host_submission_ns\n");

    constexpr int counts[] = {0, 1, 2, 4, 5, 8};
    for (int trial = 0; trial < kTrials; ++trial) {
        for (int order = 0; order < 24; ++order) {
            const int rotated = (order + trial * 5) % 24;
            const int condition = rotated / 12;
            const int program = (rotated / 6) % 2;
            const int kernel_count = counts[rotated % 6];
            for (int sample = 0; sample < kSamples; ++sample) {
                if (condition == 0) {
                    CHECK(cudaDeviceSynchronize());
                } else {
                    clock_spin<<<1, 32>>>(
                        kQueueBlockerCycles, warmup_output);
                    CHECK(cudaGetLastError());
                }
                const auto host_begin = std::chrono::steady_clock::now();
                CHECK(cudaEventRecord(start));
                const auto submission_begin = std::chrono::steady_clock::now();
                for (int launch = 0; launch < kernel_count; ++launch) {
                    launch_program(program, input, output);
                }
                CHECK(cudaGetLastError());
                const auto submission_end = std::chrono::steady_clock::now();
                CHECK(cudaEventRecord(stop));
                CHECK(cudaEventSynchronize(stop));
                float elapsed_ms = 0.0f;
                CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
                const auto host_end = std::chrono::steady_clock::now();
                const auto submission_ns = std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                        submission_end - submission_begin).count();
                const auto synchronized_ns = std::chrono::duration_cast<
                    std::chrono::nanoseconds>(host_end - host_begin).count();
                if (submission_ns < 0 || synchronized_ns <= 0) {
                    std::fprintf(stderr, "host timer became non-positive\n");
                    return EXIT_FAILURE;
                }
                std::printf(
                    "raw,%d,%s,%s,%d,%d,%.6f,%lld\n",
                    trial, condition == 0 ? "idle" : "queued",
                    program == 0 ? "empty" : "dependent", kernel_count,
                    sample, static_cast<double>(elapsed_ms) * 1000000.0,
                    static_cast<long long>(submission_ns));
            }
        }
    }

    constexpr const char* geometry_names[] = {
        "none", "fa_main", "fa_combine", "fa_main_combine",
        "fa_combine_main",
    };
    std::printf(
        "geometry_raw,trial,condition,sequence,sample,"
        "event_duration_ns,host_submission_ns\n");
    for (int trial = 0; trial < kTrials; ++trial) {
        for (int order = 0; order < 10; ++order) {
            const int rotated = (order + trial * 3) % 10;
            const int condition = rotated / 5;
            const int sequence = rotated % 5;
            for (int sample = 0; sample < kSamples; ++sample) {
                if (condition == 0) {
                    CHECK(cudaDeviceSynchronize());
                } else {
                    clock_spin<<<1, 32>>>(
                        kQueueBlockerCycles, warmup_output);
                    CHECK(cudaGetLastError());
                }
                const auto host_begin = std::chrono::steady_clock::now();
                CHECK(cudaEventRecord(start));
                const auto submission_begin = std::chrono::steady_clock::now();
                launch_geometry_sequence(sequence, output);
                CHECK(cudaGetLastError());
                const auto submission_end = std::chrono::steady_clock::now();
                CHECK(cudaEventRecord(stop));
                CHECK(cudaEventSynchronize(stop));
                float elapsed_ms = 0.0f;
                CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
                const auto host_end = std::chrono::steady_clock::now();
                const auto submission_ns = std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                        submission_end - submission_begin).count();
                const auto synchronized_ns = std::chrono::duration_cast<
                    std::chrono::nanoseconds>(host_end - host_begin).count();
                if (submission_ns < 0 || synchronized_ns <= 0) {
                    std::fprintf(stderr, "host timer became non-positive\n");
                    return EXIT_FAILURE;
                }
                std::printf(
                    "geometry_raw,%d,%s,%s,%d,%.6f,%lld\n",
                    trial, condition == 0 ? "idle" : "queued",
                    geometry_names[sequence], sample,
                    static_cast<double>(elapsed_ms) * 1000000.0,
                    static_cast<long long>(submission_ns));
            }
        }
    }

    CHECK(cudaEventDestroy(stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaFree(warmup_output));
    CHECK(cudaFree(output));
    CHECK(cudaFree(input));
    return EXIT_SUCCESS;
}
