/*
 * H200 owner copy of microbench/nvidia_a800/compute/reduction/
 * block_reduce_latency.cu.  The kernel body, timed region, CUB collective,
 * timer domains and host validation are unchanged.  Hardware coupling only:
 *   - device check: compute capability 9.0 with 132 SMs (was 8.0 / 108);
 *   - valid %smid range: < 132 (was < 108);
 *   - built natively for sm_90 by the H200 runner (A800: sm_80, CUDA 12.8.93 /
 *     CUB 2.7.0; H200: the CUDA 12.9 toolkit available on the host, CUB 2.8.2,
 *     recorded in the capture manifest).
 * The compile-time macro keeps its A800 name (A800_REDUCTION_CLOCK64) so the
 * two timer-domain binaries are selected exactly as on A800.
 */
/*
 * Isolated A800 block-reduction dependency mechanism.
 *
 * The timed region matches the collective used once per row by the pinned
 * BF16 vec8 vLLM Add-RMSNorm source: CUB BlockReduce<float, 1024>, one thread
 * computes the variance normalization and reciprocal square root, and a block
 * barrier publishes that scalar.  It has no global-memory traffic in the
 * timed region and does not call a production operator.  Host validation
 * consumes every block result after timing.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>
#include <cuda_runtime.h>


namespace {

constexpr int kWarmup = 20;
constexpr float kEpsilon = 1.0e-5f;


struct CubAddOp {
    __device__ __forceinline__ float operator()(float lhs, float rhs) const {
        return lhs + rhs;
    }
};


void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::fprintf(
            stderr, "%s failed: %s\n", expression,
            cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}


#define CHECK_CUDA(call) check_cuda((call), #call)


__device__ __forceinline__ std::uint64_t globaltimer_ns() {
    std::uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
    return value;
}


__device__ __forceinline__ std::uint64_t mechanism_timer() {
#if defined(A800_REDUCTION_CLOCK64)
    return clock64();
#else
    return globaltimer_ns();
#endif
}


__device__ __forceinline__ unsigned int physical_smid() {
    unsigned int value;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
    return value;
}


__global__ void reduce_rsqrt_broadcast(
        __nv_bfloat16* __restrict__ input,
        const std::int64_t input_stride,
        __nv_bfloat16* __restrict__ residual,
        const __nv_bfloat16* __restrict__ weight,
        const float epsilon,
        const int num_tokens,
        const int hidden_size,
        std::uint64_t* __restrict__ timer_begin,
        std::uint64_t* __restrict__ timer_end,
        float* __restrict__ outputs,
        unsigned int* __restrict__ sm_ids,
        const int sample) {
    // The leading arguments intentionally match the production Add-RMSNorm
    // ABI.  The isolated mechanism consumes only its scalar normalization
    // operands; the production tensor streams stay outside the timed region.
    static_cast<void>(input);
    static_cast<void>(input_stride);
    static_cast<void>(residual);
    static_cast<void>(weight);
    static_cast<void>(num_tokens);
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;
    __shared__ float inverse_root;

    const float value =
        1.0f + static_cast<float>(threadIdx.x & 7) * 0.125f;
    const std::uint64_t begin_tick = mechanism_timer();
    const float sum = BlockReduce(reduce_storage).Reduce(
        value, CubAddOp{}, blockDim.x);
    if (threadIdx.x == 0) {
        const float variance = sum / static_cast<float>(hidden_size) + epsilon;
        inverse_root = rsqrtf(variance);
    }
    __syncthreads();
    const std::uint64_t end_tick = mechanism_timer();
    if (threadIdx.x == 0) {
        const int offset = sample * gridDim.x + blockIdx.x;
        timer_begin[offset] = begin_tick;
        timer_end[offset] = end_tick;
        outputs[offset] = inverse_root;
        sm_ids[offset] = physical_smid();
    }
}


int parse_positive_integer(const char* text, const char* name) {
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed <= 0
            || parsed > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "%s must be a positive integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(parsed);
}


std::uint64_t median(std::vector<std::uint64_t> values) {
    if (values.empty()) {
        std::fprintf(stderr, "cannot take the median of an empty vector\n");
        std::exit(EXIT_FAILURE);
    }
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}


float expected_inverse_root(int threads) {
    double sum = 0.0;
    for (int thread = 0; thread < threads; ++thread) {
        sum += 1.0 + static_cast<double>(thread & 7) * 0.125;
    }
    const double variance =
        sum / 4096.0 + kEpsilon;
    return static_cast<float>(1.0 / std::sqrt(variance));
}


bool supported_threads(int threads) {
    return threads == 256 || threads == 1024;
}


void launch_cell(
        int grid, int threads, int samples,
        std::uint64_t* timer_begin_device,
        std::uint64_t* timer_end_device,
        float* outputs_device,
        unsigned int* sm_ids_device) {
    for (int warmup = 0; warmup < kWarmup; ++warmup) {
        reduce_rsqrt_broadcast<<<grid, threads>>>(
            nullptr, 0, nullptr, nullptr, kEpsilon, grid, 4096,
            timer_begin_device, timer_end_device, outputs_device,
            sm_ids_device, 0);
    }
    for (int sample = 0; sample < samples; ++sample) {
        reduce_rsqrt_broadcast<<<grid, threads>>>(
            nullptr, 0, nullptr, nullptr, kEpsilon, grid, 4096,
            timer_begin_device, timer_end_device, outputs_device,
            sm_ids_device, sample);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

}  // namespace


int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s THREADS GRID SAMPLES\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int threads = parse_positive_integer(argv[1], "threads");
    const int grid = parse_positive_integer(argv[2], "grid");
    const int samples = parse_positive_integer(argv[3], "samples");
    if (!supported_threads(threads)) {
        std::fprintf(
            stderr,
            "threads must be one of 256,1024\n");
        return EXIT_FAILURE;
    }

    int device = -1;
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 9 || properties.minor != 0
            || properties.multiProcessorCount != 132) {
        std::fprintf(
            stderr,
            "expected H200 sm_90 with 132 SMs, got cc=%d.%d sms=%d\n",
            properties.major, properties.minor,
            properties.multiProcessorCount);
        return EXIT_FAILURE;
    }

    const std::size_t records =
        static_cast<std::size_t>(samples) * static_cast<std::size_t>(grid);
    std::uint64_t* timer_begin_device = nullptr;
    std::uint64_t* timer_end_device = nullptr;
    float* outputs_device = nullptr;
    unsigned int* sm_ids_device = nullptr;
    CHECK_CUDA(cudaMalloc(
        &timer_begin_device, records * sizeof(std::uint64_t)));
    CHECK_CUDA(cudaMalloc(
        &timer_end_device, records * sizeof(std::uint64_t)));
    CHECK_CUDA(cudaMalloc(&outputs_device, records * sizeof(float)));
    CHECK_CUDA(cudaMalloc(
        &sm_ids_device, records * sizeof(unsigned int)));

    launch_cell(
        grid, threads, samples,
        timer_begin_device, timer_end_device, outputs_device, sm_ids_device);

    std::vector<std::uint64_t> timer_begin(records);
    std::vector<std::uint64_t> timer_end(records);
    std::vector<float> outputs(records);
    std::vector<unsigned int> sm_ids(records);
    CHECK_CUDA(cudaMemcpy(
        timer_begin.data(), timer_begin_device,
        records * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(
        timer_end.data(), timer_end_device,
        records * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(
        outputs.data(), outputs_device,
        records * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(
        sm_ids.data(), sm_ids_device,
        records * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    const float expected = expected_inverse_root(threads);
    for (std::size_t index = 0; index < records; ++index) {
#if defined(A800_REDUCTION_CLOCK64)
        const bool invalid_timer = timer_end[index] <= timer_begin[index];
#else
        const bool invalid_timer = timer_end[index] < timer_begin[index];
#endif
        if (invalid_timer
                || sm_ids[index] >= 132
                || !std::isfinite(outputs[index])
                || std::abs(outputs[index] - expected) > 2.0e-6f) {
            std::fprintf(
                stderr,
                "invalid record index=%zu smid=%u timer=%llu..%llu "
                "output=%.9g expected=%.9g\n",
                index,
                sm_ids[index],
                static_cast<unsigned long long>(timer_begin[index]),
                static_cast<unsigned long long>(timer_end[index]),
                outputs[index], expected);
            return EXIT_FAILURE;
        }
    }

#if defined(A800_REDUCTION_CLOCK64)
    std::printf(
        "sample,device_ordinal,sm_count,threads,grid,"
        "block_median_cycles,sm_local_span_cycles,"
        "validation_checksum,validation_status\n");
#else
    std::printf(
        "sample,device_ordinal,sm_count,threads,grid,"
        "block_median_ns,grid_span_ns,"
        "validation_checksum,validation_status\n");
#endif
    for (int sample = 0; sample < samples; ++sample) {
        std::vector<std::uint64_t> block_ns;
        block_ns.reserve(grid);
        std::uint64_t first_ns = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t last_ns = 0;
        std::vector<std::uint64_t> sm_first(
            properties.multiProcessorCount,
            std::numeric_limits<std::uint64_t>::max());
        std::vector<std::uint64_t> sm_last(
            properties.multiProcessorCount, 0);
        double checksum = 0.0;
        for (int block = 0; block < grid; ++block) {
            const std::size_t index =
                static_cast<std::size_t>(sample) * grid + block;
            block_ns.push_back(timer_end[index] - timer_begin[index]);
            first_ns = std::min(first_ns, timer_begin[index]);
            last_ns = std::max(last_ns, timer_end[index]);
            const unsigned int smid = sm_ids[index];
            sm_first[smid] = std::min(sm_first[smid], timer_begin[index]);
            sm_last[smid] = std::max(sm_last[smid], timer_end[index]);
            checksum += outputs[index];
        }
#if defined(A800_REDUCTION_CLOCK64)
        std::vector<std::uint64_t> sm_local_spans;
        for (int smid = 0; smid < properties.multiProcessorCount; ++smid) {
            if (sm_first[smid] != std::numeric_limits<std::uint64_t>::max()) {
                sm_local_spans.push_back(sm_last[smid] - sm_first[smid]);
            }
        }
        if (sm_local_spans.empty()) {
            std::fprintf(stderr, "sample has no active SM\n");
            return EXIT_FAILURE;
        }
#endif
        std::printf(
            "%d,%d,%d,%d,%d,%llu,%llu,%.9g,pass\n",
            sample, device, properties.multiProcessorCount, threads, grid,
            static_cast<unsigned long long>(median(block_ns)),
#if defined(A800_REDUCTION_CLOCK64)
            static_cast<unsigned long long>(median(sm_local_spans)),
#else
            static_cast<unsigned long long>(last_ns - first_ns),
#endif
            checksum);
    }

    CHECK_CUDA(cudaFree(sm_ids_device));
    CHECK_CUDA(cudaFree(outputs_device));
    CHECK_CUDA(cudaFree(timer_end_device));
    CHECK_CUDA(cudaFree(timer_begin_device));
    return EXIT_SUCCESS;
}
