/*
 * MetaX C500 register-resident instruction latency/throughput benchmark.
 *
 * The timed loops contain one primitive opcode and no global-memory traffic.
 * A dependent chain measures cycles/instruction.  Eight independent chains
 * and a waves/AP sweep measure the issue envelope in operations/PEU/cycle.
 * Standard-library expf/rsqrtf are deliberately excluded because the MACA
 * device library lowers them to multi-instruction sequences.
 *
 * Build in the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch=native \
 *       --maca-path=/opt/maca -O3 instruction_throughput.cpp \
 *       -o instruction_throughput
 */

#include <mc_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do {                                                   \
    mcError_t error = (call);                                              \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",              \
                     __FILE__, __LINE__, mcGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

enum Primitive {
    FMA_F32 = 0,
    ADD_F32 = 1,
    EXP2_F32 = 2,
    RCP_F32 = 3,
    RSQRT_F32 = 4,
};

template <Primitive primitive>
struct Instruction;

template <>
struct Instruction<FMA_F32> {
    __device__ __forceinline__ static float apply(float value) {
        float output;
        const float multiplier = 0.9999847412109375f;
        const float addend = 0.0000152587890625f;
        asm volatile("fma_f32 %0, %1, %2, %3"
                     : "=r"(output)
                     : "r"(value), "r"(multiplier), "r"(addend));
        return output;
    }
};

template <>
struct Instruction<ADD_F32> {
    __device__ __forceinline__ static float apply(float value) {
        float output;
        const float addend = 0.0000152587890625f;
        asm volatile("add_f32 %0, %1, %2"
                     : "=r"(output) : "r"(value), "r"(addend));
        return output;
    }
};

template <>
struct Instruction<EXP2_F32> {
    __device__ __forceinline__ static float apply(float value) {
        float output;
        asm volatile("exp_f32 %0, neg(%1)"
                     : "=r"(output) : "r"(value));
        return output;
    }
};

template <>
struct Instruction<RCP_F32> {
    __device__ __forceinline__ static float apply(float value) {
        float output;
        asm volatile("rcp_f32 %0, %1" : "=r"(output) : "r"(value));
        return output;
    }
};

template <>
struct Instruction<RSQRT_F32> {
    __device__ __forceinline__ static float apply(float value) {
        float output;
        asm volatile("rsq_f32 %0, %1" : "=r"(output) : "r"(value));
        return output;
    }
};

template <Primitive primitive>
__global__ void dependent_latency_kernel(
        float* sink, unsigned long long* elapsed_cycles, int iterations) {
    float value = 1.0009765625f;
    unsigned long long start = clock64();
    #pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        value = Instruction<primitive>::apply(value);
    }
    unsigned long long end = clock64();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        sink[0] = value;
        elapsed_cycles[0] = end - start;
    }
}

template <Primitive primitive>
__global__ void independent_throughput_kernel(
        float* sink, int iterations) {
    float x0 = 1.0009765625f;
    float x1 = 1.0019531250f;
    float x2 = 1.0029296875f;
    float x3 = 1.0039062500f;
    float x4 = 1.0048828125f;
    float x5 = 1.0058593750f;
    float x6 = 1.0068359375f;
    float x7 = 1.0078125000f;
    #pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        x0 = Instruction<primitive>::apply(x0);
        x1 = Instruction<primitive>::apply(x1);
        x2 = Instruction<primitive>::apply(x2);
        x3 = Instruction<primitive>::apply(x3);
        x4 = Instruction<primitive>::apply(x4);
        x5 = Instruction<primitive>::apply(x5);
        x6 = Instruction<primitive>::apply(x6);
        x7 = Instruction<primitive>::apply(x7);
    }
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    sink[thread] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}

template <Primitive primitive>
static void launch_latency(
        float* sink, unsigned long long* cycles, int iterations) {
    dependent_latency_kernel<primitive><<<1, 64>>>(
        sink, cycles, iterations);
}

template <Primitive primitive>
static void launch_throughput(
        int grid, int block, float* sink, int iterations) {
    independent_throughput_kernel<primitive><<<grid, block>>>(
        sink, iterations);
}

static const char* primitive_name(Primitive primitive) {
    switch (primitive) {
        case FMA_F32: return "fma_f32";
        case ADD_F32: return "add_f32";
        case EXP2_F32: return "exp2_primitive_f32";
        case RCP_F32: return "rcp_f32";
        case RSQRT_F32: return "rsqrt_primitive_f32";
        default:
            std::fprintf(stderr, "unknown primitive %d\n", primitive);
            std::exit(EXIT_FAILURE);
    }
}

static double operations_per_instruction(Primitive primitive) {
    switch (primitive) {
        case FMA_F32:
            return 2.0;
        case ADD_F32:
        case EXP2_F32:
        case RCP_F32:
        case RSQRT_F32:
            return 1.0;
        default:
            std::fprintf(stderr, "unknown primitive %d\n", primitive);
            std::exit(EXIT_FAILURE);
    }
}

static void dispatch_latency(
        Primitive primitive, float* sink, unsigned long long* cycles,
        int iterations) {
    switch (primitive) {
        case FMA_F32:
            launch_latency<FMA_F32>(sink, cycles, iterations); return;
        case ADD_F32:
            launch_latency<ADD_F32>(sink, cycles, iterations); return;
        case EXP2_F32:
            launch_latency<EXP2_F32>(sink, cycles, iterations); return;
        case RCP_F32:
            launch_latency<RCP_F32>(sink, cycles, iterations); return;
        case RSQRT_F32:
            launch_latency<RSQRT_F32>(sink, cycles, iterations); return;
        default:
            std::fprintf(stderr, "unknown primitive %d\n", primitive);
            std::exit(EXIT_FAILURE);
    }
}

static void dispatch_throughput(
        Primitive primitive, int grid, int block, float* sink,
        int iterations) {
    switch (primitive) {
        case FMA_F32:
            launch_throughput<FMA_F32>(grid, block, sink, iterations); return;
        case ADD_F32:
            launch_throughput<ADD_F32>(grid, block, sink, iterations); return;
        case EXP2_F32:
            launch_throughput<EXP2_F32>(grid, block, sink, iterations); return;
        case RCP_F32:
            launch_throughput<RCP_F32>(grid, block, sink, iterations); return;
        case RSQRT_F32:
            launch_throughput<RSQRT_F32>(grid, block, sink, iterations); return;
        default:
            std::fprintf(stderr, "unknown primitive %d\n", primitive);
            std::exit(EXIT_FAILURE);
    }
}

struct TimedSample {
    double elapsed_ms;
    long long host_start_ns;
    long long host_end_ns;
};


static long long host_time_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}


static std::vector<TimedSample> elapsed_samples(
        Primitive primitive, int grid, int block, float* sink,
        int iterations) {
    constexpr int trials = 7;
    std::vector<TimedSample> samples;
    samples.reserve(trials);
    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    for (int warmup = 0; warmup < 3; ++warmup) {
        dispatch_throughput(
            primitive, grid, block, sink, std::max(1, iterations / 32));
    }
    CHECK(mcDeviceSynchronize());
    for (int trial = 0; trial < trials; ++trial) {
        long long host_start_ns = host_time_ns();
        CHECK(mcEventRecord(start));
        dispatch_throughput(primitive, grid, block, sink, iterations);
        CHECK(mcEventRecord(end));
        CHECK(mcDeviceSynchronize());
        long long host_end_ns = host_time_ns();
        float elapsed_ms = 0.0f;
        CHECK(mcEventElapsedTime(&elapsed_ms, start, end));
        if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f) {
            std::fprintf(stderr, "invalid elapsed time %.9g ms\n", elapsed_ms);
            std::exit(EXIT_FAILURE);
        }
        samples.push_back(TimedSample{
            static_cast<double>(elapsed_ms), host_start_ns, host_end_ns});
    }
    CHECK(mcEventDestroy(start));
    CHECK(mcEventDestroy(end));
    return samples;
}


static double median_elapsed_ms(const std::vector<TimedSample>& samples) {
    if (samples.empty()) {
        std::fprintf(stderr, "elapsed sample vector must not be empty\n");
        std::exit(EXIT_FAILURE);
    }
    std::vector<double> elapsed;
    elapsed.reserve(samples.size());
    for (const TimedSample& sample : samples) {
        elapsed.push_back(sample.elapsed_ms);
    }
    std::sort(elapsed.begin(), elapsed.end());
    return elapsed[elapsed.size() / 2];
}

int main() {
    mcDeviceProp_t properties;
    CHECK(mcGetDeviceProperties(&properties, 0));
    if (properties.multiProcessorCount != 104 || properties.warpSize != 64) {
        std::fprintf(
            stderr, "expected C500 104 AP/wave64, got %d AP/wave%d\n",
            properties.multiProcessorCount, properties.warpSize);
        return EXIT_FAILURE;
    }
    constexpr int block = 64;
    constexpr int max_waves_per_ap = 32;
    constexpr int chains = 8;
    constexpr int latency_iterations = 1048576;
    constexpr int throughput_iterations = 32768;
    constexpr int trials = 7;
    const int wave_multipliers[] = {1, 2, 4, 8, 16, max_waves_per_ap};
    const Primitive primitives[] = {
        FMA_F32, ADD_F32, EXP2_F32, RCP_F32, RSQRT_F32};

    float* sink = nullptr;
    unsigned long long* cycles = nullptr;
    CHECK(mcMalloc(
        &sink,
        static_cast<size_t>(properties.multiProcessorCount)
            * max_waves_per_ap * block
            * sizeof(float)));
    CHECK(mcMalloc(&cycles, sizeof(unsigned long long)));

    std::printf("device,%s\n", properties.name);
    std::printf("ap_count,%d\n", properties.multiProcessorCount);
    std::printf("wave_size,%d\n", properties.warpSize);
    std::printf("reported_clock_khz,%d\n", properties.clockRate);
    std::printf("trials,%d\n", trials);
    std::printf(
        "latency_raw,primitive,trial,iterations,elapsed_cycles\n");
    for (Primitive primitive : primitives) {
        std::vector<unsigned long long> samples;
        samples.reserve(trials);
        for (int trial = 0; trial < trials; ++trial) {
            dispatch_latency(
                primitive, sink, cycles, latency_iterations);
            CHECK(mcDeviceSynchronize());
            unsigned long long elapsed_cycles = 0;
            CHECK(mcMemcpy(
                &elapsed_cycles, cycles, sizeof(elapsed_cycles),
                mcMemcpyDeviceToHost));
            samples.push_back(elapsed_cycles);
            std::printf(
                "latency_raw,%s,%d,%d,%llu\n",
                primitive_name(primitive), trial, latency_iterations,
                elapsed_cycles);
        }
        std::sort(samples.begin(), samples.end());
        std::printf(
            "latency_summary,%s,dependent_cycles_per_instruction,%.9f\n",
            primitive_name(primitive),
            static_cast<double>(samples[samples.size() / 2])
                / latency_iterations);
    }

    std::printf(
        "throughput_raw,primitive,grid,block,waves_per_ap,iterations,"
        "chains,trial,host_start_ns,host_end_ns,elapsed_ms,operations\n");
    for (Primitive primitive : primitives) {
        for (int waves_per_ap : wave_multipliers) {
            int grid = properties.multiProcessorCount * waves_per_ap;
            std::vector<TimedSample> samples = elapsed_samples(
                primitive, grid, block, sink, throughput_iterations);
            double lane_instructions = (
                static_cast<double>(grid) * block * chains
                * throughput_iterations);
            double operations = (
                lane_instructions * operations_per_instruction(primitive));
            for (size_t trial = 0; trial < samples.size(); ++trial) {
                const TimedSample& sample = samples[trial];
                std::printf(
                    "throughput_raw,%s,%d,%d,%d,%d,%d,%zu,%lld,%lld,"
                    "%.9f,%.0f\n",
                    primitive_name(primitive), grid, block, waves_per_ap,
                    throughput_iterations, chains, trial,
                    sample.host_start_ns, sample.host_end_ns,
                    sample.elapsed_ms, operations);
            }
            double elapsed_ms = median_elapsed_ms(samples);
            double operations_per_second = operations / (elapsed_ms * 1.0e-3);
            std::printf(
                "throughput_summary,%s,%d,%d,%d,median_ms,%.9f,"
                "lane_operations_per_second,%.9f\n",
                primitive_name(primitive), grid, block, waves_per_ap,
                elapsed_ms, operations_per_second);
        }
    }

    CHECK(mcFree(cycles));
    CHECK(mcFree(sink));
    return EXIT_SUCCESS;
}
