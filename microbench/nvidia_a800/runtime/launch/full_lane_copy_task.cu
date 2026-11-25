/*
 * Operator-independent SM80 full-lane copy dependency probe.
 *
 * Every thread performs one aligned 64-bit global load whose returned value
 * is consumed by one aligned 64-bit global store.  A 128-thread CTA therefore
 * carries the same 1 KiB read + 1 KiB write geometry for both a BF16 vt4 copy
 * and an FP32 vt2 copy, without invoking either production operator.
 */

#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

constexpr int kThreads = 128;
constexpr int kWarmups = 200;
constexpr int kLaunchesPerSample = 400;
constexpr int kMaximumDimensions = 25;
constexpr unsigned long long kClockWarmupCycles = 1140000000ULL;

struct FastDivider {
    unsigned int divisor;
    unsigned int multiplier;
    unsigned int shift;

    FastDivider() : FastDivider(1U) {}

    explicit FastDivider(unsigned int value) : divisor(value) {
        for (shift = 0; shift < 32; ++shift) {
            if ((1U << shift) >= divisor) {
                break;
            }
        }
        const std::uint64_t one = 1;
        const std::uint64_t magic = (
            ((one << 32) * ((one << shift) - divisor)) / divisor + 1);
        multiplier = static_cast<unsigned int>(magic);
    }

    __device__ __forceinline__ unsigned int div(
            unsigned int value) const {
        const unsigned int high = __umulhi(value, multiplier);
        return (high + value) >> shift;
    }
};

struct OffsetProgram {
    int dimensions;
    FastDivider sizes[kMaximumDimensions];
    unsigned int output_strides[kMaximumDimensions];
    unsigned int input_strides[kMaximumDimensions];
};

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

__global__ void full_lane_copy_task(const uint2* input, uint2* output) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint2 value = input[index];
    output[index] = value;
}


__global__ void full_lane_scalar_copy_vt4(
        const std::uint16_t* input, std::uint16_t* output) {
    const int base = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    const std::uint16_t value0 = input[base];
    const std::uint16_t value1 = input[base + blockDim.x];
    const std::uint16_t value2 = input[base + 2 * blockDim.x];
    const std::uint16_t value3 = input[base + 3 * blockDim.x];
    output[base] = value0;
    output[base + blockDim.x] = value1;
    output[base + 2 * blockDim.x] = value2;
    output[base + 3 * blockDim.x] = value3;
}


__global__ void full_lane_scalar_copy_vt2(
        const std::uint32_t* input, std::uint32_t* output) {
    const int base = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    const std::uint32_t value0 = input[base];
    const std::uint32_t value1 = input[base + blockDim.x];
    output[base] = value0;
    output[base + blockDim.x] = value1;
}


template <typename Value, int ValuesPerThread>
__global__ __launch_bounds__(kThreads, 4)
void strided_offset_copy_task(
        int element_count, const Value* input, Value* output,
        OffsetProgram program) {
    int index = blockIdx.x * kThreads * ValuesPerThread + threadIdx.x;
#pragma unroll
    for (int item = 0; item < ValuesPerThread; ++item) {
        if (index < element_count) {
            unsigned int linear = static_cast<unsigned int>(index);
            unsigned int output_offset = 0;
            unsigned int input_offset = 0;
#pragma unroll
            for (int dim = 0; dim < kMaximumDimensions; ++dim) {
                if (dim == program.dimensions) {
                    break;
                }
                const unsigned int quotient = program.sizes[dim].div(linear);
                const unsigned int remainder = (
                    linear - quotient * program.sizes[dim].divisor);
                linear = quotient;
                output_offset += remainder * program.output_strides[dim];
                input_offset += remainder * program.input_strides[dim];
            }
            output[output_offset] = input[input_offset];
            index += kThreads;
        }
    }
}


OffsetProgram rank4_program(int grid) {
    const unsigned int shape[4] = {
        16U, 2U, 4U, static_cast<unsigned int>(4 * grid)};
    OffsetProgram program{};
    program.dimensions = 4;
    const unsigned int output_strides[4] = {1U, 16U, 32U, 128U};
    const unsigned int input_strides[4] = {1U, 64U, 16U, 128U};
    for (int dim = 0; dim < 4; ++dim) {
        program.sizes[dim] = FastDivider(shape[dim]);
        program.output_strides[dim] = output_strides[dim];
        program.input_strides[dim] = input_strides[dim];
    }
    return program;
}


OffsetProgram rank3_program(int grid) {
    const unsigned int shape[3] = {
        8U, 4U, static_cast<unsigned int>(8 * grid)};
    OffsetProgram program{};
    program.dimensions = 3;
    const unsigned int output_strides[3] = {1U, 8U, 32U};
    const unsigned int input_strides[3] = {4U, 1U, 32U};
    for (int dim = 0; dim < 3; ++dim) {
        program.sizes[dim] = FastDivider(shape[dim]);
        program.output_strides[dim] = output_strides[dim];
        program.input_strides[dim] = input_strides[dim];
    }
    return program;
}

__global__ void clock_warmup_kernel(float* sink) {
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    float x0 = 1.0009765625f + static_cast<float>(thread) * 0x1p-24f;
    float x1 = x0 + 0x1p-11f;
    const unsigned long long start = clock64();
    do {
        x0 = x0 * 0.9999847412109375f + 0.0000152587890625f;
        x1 = x1 * 0.9999847412109375f + 0.0000152587890625f;
    } while (clock64() - start < kClockWarmupCycles);
    sink[thread] = x0 + x1;
}

}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    int trials = 5;
    if (argc == 3) {
        device = parse_int(argv[1], "device");
        trials = parse_int(argv[2], "trials");
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [device trials]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (device < 0 || trials <= 0) {
        std::fprintf(stderr, "device must be nonnegative and trials positive\n");
        return EXIT_FAILURE;
    }
    CHECK(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CHECK(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 8 || properties.minor != 0
            || properties.multiProcessorCount != 108) {
        std::fprintf(stderr, "probe requires the 108-SM sm_80 target\n");
        return EXIT_FAILURE;
    }
    int resident_ctas = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_ctas, full_lane_copy_task, kThreads, 0));
    int resident_ctas_vt4 = 0;
    int resident_ctas_vt2 = 0;
    int resident_ctas_rank4 = 0;
    int resident_ctas_rank3 = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_ctas_vt4, full_lane_scalar_copy_vt4, kThreads, 0));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_ctas_vt2, full_lane_scalar_copy_vt2, kThreads, 0));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_ctas_rank4,
        strided_offset_copy_task<std::uint16_t, 4>, kThreads, 0));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_ctas_rank3,
        strided_offset_copy_task<std::uint32_t, 2>, kThreads, 0));

    constexpr int grids[] = {
        1, 2, 4, 8, 16, 32, 64, 108, 216, 432, 864, 1024,
        1728, 2048, 3456, 4096,
    };
    constexpr int maximum_grid = 4096;
    constexpr int elements = maximum_grid * kThreads;
    uint2* input = nullptr;
    uint2* output = nullptr;
    float* warmup_sink = nullptr;
    CHECK(cudaMalloc(&input, elements * sizeof(*input)));
    CHECK(cudaMalloc(&output, elements * sizeof(*output)));
    CHECK(cudaMemset(input, 0x5a, elements * sizeof(*input)));
    CHECK(cudaMemset(output, 0, elements * sizeof(*output)));
    constexpr int warmup_grid = 864;
    constexpr int warmup_block = 256;
    CHECK(cudaMalloc(
        &warmup_sink, warmup_grid * warmup_block * sizeof(*warmup_sink)));
    clock_warmup_kernel<<<warmup_grid, warmup_block>>>(warmup_sink);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        full_lane_copy_task<<<maximum_grid, kThreads>>>(input, output);
        full_lane_scalar_copy_vt4<<<maximum_grid, kThreads>>>(
            reinterpret_cast<const std::uint16_t*>(input),
            reinterpret_cast<std::uint16_t*>(output));
        full_lane_scalar_copy_vt2<<<maximum_grid, kThreads>>>(
            reinterpret_cast<const std::uint32_t*>(input),
            reinterpret_cast<std::uint32_t*>(output));
        const OffsetProgram program4 = rank4_program(maximum_grid);
        strided_offset_copy_task<std::uint16_t, 4>
            <<<maximum_grid, kThreads>>>(
                maximum_grid * kThreads * 4,
                reinterpret_cast<const std::uint16_t*>(input),
                reinterpret_cast<std::uint16_t*>(output), program4);
        const OffsetProgram program3 = rank3_program(maximum_grid);
        strided_offset_copy_task<std::uint32_t, 2>
            <<<maximum_grid, kThreads>>>(
                maximum_grid * kThreads * 2,
                reinterpret_cast<const std::uint32_t*>(input),
                reinterpret_cast<std::uint32_t*>(output), program3);
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
    std::printf("meta,reference_sm_clock_hz,%d000\n", properties.clockRate);
    std::printf("meta,threads,%d\n", kThreads);
    std::printf("meta,resident_ctas_per_sm,%d\n", resident_ctas);
    std::printf("meta,resident_ctas_per_sm_vt4,%d\n", resident_ctas_vt4);
    std::printf("meta,resident_ctas_per_sm_vt2,%d\n", resident_ctas_vt2);
    std::printf("meta,resident_ctas_per_sm_rank4,%d\n", resident_ctas_rank4);
    std::printf("meta,resident_ctas_per_sm_rank3,%d\n", resident_ctas_rank3);
    std::printf("meta,launches_per_sample,%d\n", kLaunchesPerSample);
    std::printf(
        "raw,trial,program,grid,bytes_read,bytes_written,duration_ns\n");
    for (int trial = 0; trial < trials; ++trial) {
        for (int grid : grids) {
            for (int program = 0; program < 5; ++program) {
                CHECK(cudaEventRecord(start));
                for (int launch = 0; launch < kLaunchesPerSample; ++launch) {
                    if (program == 0) {
                        full_lane_copy_task<<<grid, kThreads>>>(input, output);
                    } else if (program == 1) {
                        full_lane_scalar_copy_vt4<<<grid, kThreads>>>(
                            reinterpret_cast<const std::uint16_t*>(input),
                            reinterpret_cast<std::uint16_t*>(output));
                    } else if (program == 2) {
                        full_lane_scalar_copy_vt2<<<grid, kThreads>>>(
                            reinterpret_cast<const std::uint32_t*>(input),
                            reinterpret_cast<std::uint32_t*>(output));
                    } else if (program == 3) {
                        const OffsetProgram offset_program = (
                            rank4_program(grid));
                        strided_offset_copy_task<std::uint16_t, 4>
                            <<<grid, kThreads>>>(
                                grid * kThreads * 4,
                                reinterpret_cast<const std::uint16_t*>(input),
                                reinterpret_cast<std::uint16_t*>(output),
                                offset_program);
                    } else {
                        const OffsetProgram offset_program = (
                            rank3_program(grid));
                        strided_offset_copy_task<std::uint32_t, 2>
                            <<<grid, kThreads>>>(
                                grid * kThreads * 2,
                                reinterpret_cast<const std::uint32_t*>(input),
                                reinterpret_cast<std::uint32_t*>(output),
                                offset_program);
                    }
                }
                CHECK(cudaEventRecord(stop));
                CHECK(cudaEventSynchronize(stop));
                float elapsed_ms = 0.0f;
                CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
                const double duration_ns = (
                    static_cast<double>(elapsed_ms) * 1.0e6
                    / static_cast<double>(kLaunchesPerSample));
                const unsigned long long bytes = (
                    static_cast<unsigned long long>(grid) * kThreads
                    * sizeof(uint2));
                const char* name = (
                    program == 0 ? "vector64"
                    : program == 1 ? "bf16_vt4"
                    : program == 2 ? "fp32_vt2"
                    : program == 3 ? "bf16_rank4_offset"
                    : "fp32_rank3_offset");
                std::printf("raw,%d,%s,%d,%llu,%llu,%.9f\n", trial,
                            name, grid, bytes, bytes, duration_ns);
            }
        }
    }
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaFree(warmup_sink));
    CHECK(cudaFree(output));
    CHECK(cudaFree(input));
    return EXIT_SUCCESS;
}
