/*
 * Independent source-ordered heterogeneous LDGSTS assignment probe for SM80.
 *
 * This is not an attention kernel.  It contains no MMA, softmax, mask,
 * accumulation, output conversion, or production tensor.  Two 128-thread
 * CTAs reside on each A800 SM.  Source-adjacent pairs execute one through N
 * identical 64-KiB LDGSTS panels, matching the discrete panel-class order of
 * the causal FA2 source program.  Every CTA records its source ordinal, panel
 * class, physical SM, and admission ordinal local to that SM.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace {

constexpr int kThreads = 128;
constexpr int kSmCount = 108;
constexpr int kResidentCtasPerSm = 2;
constexpr int kDynamicSharedBytes = 76 * 1024;
constexpr int kCopiesPerThread = 32;
constexpr int kCopyBytes = 16;
constexpr int kCopyStrideBytes = 2048;
constexpr int kPanelBytes = 64 * 1024;
constexpr int kCtaInputBytes = 2 * kPanelBytes;

static_assert(
    kThreads * kCopiesPerThread * kCopyBytes == kPanelBytes,
    "one panel must carry exactly 64 KiB");

struct Record {
    unsigned int source_ordinal;
    unsigned int panel_count;
    unsigned int sm_id;
    unsigned int resident_ordinal;
    unsigned long long start_ticket;
    unsigned long long begin_ns;
    unsigned long long end_ns;
};

void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", expression,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(call) check_cuda((call), #call)

int positive(const char* text, const char* name) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0 || value > 4096) {
        std::fprintf(stderr, "%s must be in [1, 4096]\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

int non_negative(const char* text, const char* name) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > 4096) {
        std::fprintf(stderr, "%s must be in [0, 4096]\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

__device__ __forceinline__ unsigned int physical_sm_id() {
    unsigned int value;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
    return value;
}

__device__ __forceinline__ unsigned long long global_nanoseconds() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__device__ __forceinline__ void cp_async_16(
        void* shared_address, const void* global_address) {
    const unsigned shared = __cvta_generic_to_shared(shared_address);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        : : "r"(shared), "l"(global_address) : "memory");
}

__global__ __launch_bounds__(kThreads, kResidentCtasPerSm)
void fa2_queued_ldgsts_assignment(
        const unsigned char* input,
        unsigned int* per_sm_admissions,
        unsigned long long* global_admissions,
        std::uint32_t* checksums,
        Record* records,
        int maximum_panels,
        int period_offset) {
    extern __shared__ unsigned char shared[];
    __shared__ unsigned int local_sm;
    __shared__ unsigned int local_resident_ordinal;
    __shared__ unsigned long long local_start_ticket;
    __shared__ unsigned long long local_begin_ns;

    const unsigned int source_ordinal = blockIdx.x;
    const unsigned int period = 2U * static_cast<unsigned>(maximum_panels);
    const unsigned int panel_count =
        ((source_ordinal + static_cast<unsigned>(period_offset)) % period) / 2U
        + 1U;
    if (threadIdx.x == 0) {
        local_sm = physical_sm_id();
        local_resident_ordinal = atomicAdd(per_sm_admissions + local_sm, 1U);
        local_start_ticket = atomicAdd(global_admissions, 1ULL);
        local_begin_ns = global_nanoseconds();
    }
    __syncthreads();

    const int thread_offset = threadIdx.x * kCopyBytes;
    for (unsigned int panel = 0; panel < panel_count; ++panel) {
        const unsigned char* block_input =
            input + source_ordinal * kCtaInputBytes
            + (panel & 1U) * kPanelBytes;
#pragma unroll
        for (int copy = 0; copy < kCopiesPerThread; ++copy) {
            const int offset = thread_offset + copy * kCopyStrideBytes;
            cp_async_16(shared + offset, block_input + offset);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        checksums[source_ordinal] =
            shared[(source_ordinal * 131U) & (kPanelBytes - 1)];
        records[source_ordinal] = Record{
            source_ordinal, panel_count, local_sm, local_resident_ordinal,
            local_start_ticket, local_begin_ns, global_nanoseconds()};
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(
            stderr,
            "usage: %s TRIALS GRID MAXIMUM_PANELS PERIOD_OFFSET DEVICE\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int trials = positive(argv[1], "trials");
    const int grid = positive(argv[2], "grid");
    const int maximum_panels = positive(argv[3], "maximum_panels");
    const int period_offset = non_negative(argv[4], "period_offset");
    const int device = non_negative(argv[5], "device");
    if ((maximum_panels & (maximum_panels - 1)) != 0
            || maximum_panels > 32) {
        std::fprintf(stderr, "maximum_panels must be a power of two <= 32\n");
        return EXIT_FAILURE;
    }
    if (period_offset >= 2 * maximum_panels) {
        std::fprintf(stderr, "period_offset is outside the source period\n");
        return EXIT_FAILURE;
    }

    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 8 || properties.minor != 0
            || properties.multiProcessorCount != kSmCount) {
        std::fprintf(stderr, "probe requires a 108-SM sm_80 device\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaFuncSetAttribute(
        fa2_queued_ldgsts_assignment,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kDynamicSharedBytes));
    int occupancy = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, fa2_queued_ldgsts_assignment,
        kThreads, kDynamicSharedBytes));
    if (occupancy != kResidentCtasPerSm) {
        std::fprintf(stderr, "expected two resident CTAs per SM, got %d\n",
                     occupancy);
        return EXIT_FAILURE;
    }

    const std::size_t input_bytes =
        static_cast<std::size_t>(grid) * kCtaInputBytes;
    unsigned char* input = nullptr;
    unsigned int* per_sm_admissions = nullptr;
    unsigned long long* global_admissions = nullptr;
    std::uint32_t* checksums = nullptr;
    Record* device_records = nullptr;
    CHECK_CUDA(cudaMalloc(&input, input_bytes));
    CHECK_CUDA(cudaMalloc(
        &per_sm_admissions, kSmCount * sizeof(*per_sm_admissions)));
    CHECK_CUDA(cudaMalloc(
        &global_admissions, sizeof(*global_admissions)));
    CHECK_CUDA(cudaMalloc(&checksums, grid * sizeof(*checksums)));
    CHECK_CUDA(cudaMalloc(&device_records, grid * sizeof(*device_records)));
    CHECK_CUDA(cudaMemset(input, 1, input_bytes));

    auto launch = [&]() {
        CHECK_CUDA(cudaMemset(
            per_sm_admissions, 0,
            kSmCount * sizeof(*per_sm_admissions)));
        CHECK_CUDA(cudaMemset(
            global_admissions, 0, sizeof(*global_admissions)));
        fa2_queued_ldgsts_assignment<<<
            grid, kThreads, kDynamicSharedBytes>>>(
                input, per_sm_admissions, global_admissions,
                checksums, device_records, maximum_panels, period_offset);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    };
    for (int warmup = 0; warmup < 3; ++warmup) {
        launch();
    }

    std::vector<Record> records(grid);
    std::vector<std::uint32_t> host_checksums(grid);
    std::printf("meta,device,%d\n", device);
    std::printf("meta,device_name,%s\n", properties.name);
    std::printf("meta,sm_count,%d\n", properties.multiProcessorCount);
    std::printf("meta,resident_ctas_per_sm,%d\n", occupancy);
    std::printf("meta,threads,%d\n", kThreads);
    std::printf("meta,dynamic_shared_bytes,%d\n", kDynamicSharedBytes);
    std::printf("meta,grid,%d\n", grid);
    std::printf("meta,maximum_panels,%d\n", maximum_panels);
    std::printf("meta,period_offset,%d\n", period_offset);
    std::printf(
        "raw,trial,source_ordinal,panel_count,sm_id,resident_ordinal,"
        "start_ticket,begin_ns,end_ns\n");
    for (int trial = 0; trial < trials; ++trial) {
        launch();
        CHECK_CUDA(cudaMemcpy(
            records.data(), device_records, grid * sizeof(*device_records),
            cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(
            host_checksums.data(), checksums, grid * sizeof(*checksums),
            cudaMemcpyDeviceToHost));
        const std::uint64_t checksum = std::accumulate(
            host_checksums.begin(), host_checksums.end(), std::uint64_t{0});
        std::fprintf(stderr, "trial=%d checksum=%llu\n", trial,
                     static_cast<unsigned long long>(checksum));
        for (const Record& record : records) {
            std::printf(
                "raw,%d,%u,%u,%u,%u,%llu,%llu,%llu\n",
                trial, record.source_ordinal, record.panel_count,
                record.sm_id, record.resident_ordinal,
                record.start_ticket, record.begin_ns, record.end_ns);
        }
    }

    CHECK_CUDA(cudaFree(device_records));
    CHECK_CUDA(cudaFree(checksums));
    CHECK_CUDA(cudaFree(global_admissions));
    CHECK_CUDA(cudaFree(per_sm_admissions));
    CHECK_CUDA(cudaFree(input));
    return EXIT_SUCCESS;
}
