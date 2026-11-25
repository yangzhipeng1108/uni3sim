/*
 * Operator-independent two-resident-CTA admission probe for SM80.
 *
 * The launch stream mirrors only the control pattern needed by a scheduler
 * model: a fixed dim3 launch contains immediately completing, short, and long
 * CTAs.  No production operator data or duration is consumed.
 */

#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

constexpr int kThreads = 128;
constexpr int kSmCount = 108;
constexpr int kResidentCtasPerSm = 2;
constexpr int kDynamicSharedBytes = 64 * 1024;
constexpr int kGridX = 32;
constexpr int kGridY = 4;
constexpr int kGridZ = 32;
constexpr int kBaseIterations = 128;

struct Record {
    unsigned long long start_ticket;
    unsigned long long end_ticket;
    unsigned long long start_cycle;
    unsigned long long end_cycle;
    unsigned int sm_id;
    unsigned int linear_block;
    unsigned int work_class;
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

__device__ __forceinline__ unsigned int sm_id() {
    unsigned int value;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
    return value;
}

__global__ __launch_bounds__(kThreads, kResidentCtasPerSm)
void admission_probe(unsigned long long* event_counter, Record* records) {
    extern __shared__ unsigned int sink[];
    __shared__ unsigned long long start_ticket;
    __shared__ unsigned long long start_cycle;
    __shared__ unsigned int resident_sm;

    const unsigned int linear = (
        blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));
    constexpr int q_tile_limits[kGridY] = {8, 8, 16, 32};
    const unsigned int work_class = (
        blockIdx.x >= q_tile_limits[blockIdx.y]
        ? 0U : static_cast<unsigned int>((blockIdx.x + 4) / 4));

    if (threadIdx.x == 0) {
        resident_sm = sm_id();
        start_ticket = atomicAdd(event_counter, 1ULL);
        start_cycle = clock64();
    }
    __syncthreads();

    unsigned int state = (
        0x9e3779b9U ^ linear * 1664525U
        ^ static_cast<unsigned int>(threadIdx.x));
    const int iterations = static_cast<int>(work_class) * kBaseIterations;
    #pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        state = state * 1664525U + 1013904223U;
        state ^= state >> 13;
    }
    sink[threadIdx.x] = state;
    __syncthreads();

    if (threadIdx.x == 0) {
        const unsigned long long end_cycle = clock64();
        const unsigned long long end_ticket = atomicAdd(event_counter, 1ULL);
        records[linear] = Record{
            start_ticket, end_ticket, start_cycle, end_cycle,
            resident_sm, linear, work_class};
    }
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
            || properties.multiProcessorCount != kSmCount) {
        std::fprintf(stderr, "probe requires the 108-SM sm_80 target\n");
        return EXIT_FAILURE;
    }
    CHECK(cudaFuncSetAttribute(
        admission_probe, cudaFuncAttributeMaxDynamicSharedMemorySize,
        kDynamicSharedBytes));
    int occupancy = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, admission_probe, kThreads, kDynamicSharedBytes));
    if (occupancy != kResidentCtasPerSm) {
        std::fprintf(stderr, "expected occupancy 2, got %d\n", occupancy);
        return EXIT_FAILURE;
    }

    constexpr int count = kGridX * kGridY * kGridZ;
    unsigned long long* counter = nullptr;
    Record* device_records = nullptr;
    Record* host_records = static_cast<Record*>(
        std::malloc(sizeof(Record) * count));
    if (host_records == nullptr) {
        std::fprintf(stderr, "host allocation failed\n");
        return EXIT_FAILURE;
    }
    CHECK(cudaMalloc(&counter, sizeof(*counter)));
    CHECK(cudaMalloc(&device_records, sizeof(Record) * count));

    std::printf("meta,device,%d\n", device);
    std::printf("meta,device_name,%s\n", properties.name);
    std::printf("meta,sm_count,%d\n", properties.multiProcessorCount);
    std::printf("meta,resident_ctas_per_sm,%d\n", occupancy);
    std::printf("meta,dynamic_shared_bytes,%d\n", kDynamicSharedBytes);
    std::printf("meta,grid,%d,%d,%d\n", kGridX, kGridY, kGridZ);
    std::printf("raw,trial,linear_block,block_x,block_y,block_z,work_class,sm_id,start_ticket,end_ticket,start_cycle,end_cycle\n");
    for (int warmup = 0; warmup < 3; ++warmup) {
        CHECK(cudaMemset(counter, 0, sizeof(*counter)));
        admission_probe<<<dim3(kGridX, kGridY, kGridZ), kThreads,
                          kDynamicSharedBytes>>>(counter, device_records);
        CHECK(cudaDeviceSynchronize());
    }
    for (int trial = 0; trial < trials; ++trial) {
        CHECK(cudaMemset(counter, 0, sizeof(*counter)));
        admission_probe<<<dim3(kGridX, kGridY, kGridZ), kThreads,
                          kDynamicSharedBytes>>>(counter, device_records);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(host_records, device_records, sizeof(Record) * count,
                         cudaMemcpyDeviceToHost));
        for (int linear = 0; linear < count; ++linear) {
            const Record& record = host_records[linear];
            const int x = linear % kGridX;
            const int yz = linear / kGridX;
            const int y = yz % kGridY;
            const int z = yz / kGridY;
            std::printf(
                "raw,%d,%u,%d,%d,%d,%u,%u,%llu,%llu,%llu,%llu\n",
                trial, record.linear_block, x, y, z, record.work_class,
                record.sm_id, record.start_ticket, record.end_ticket,
                record.start_cycle, record.end_cycle);
        }
    }
    CHECK(cudaFree(device_records));
    CHECK(cudaFree(counter));
    std::free(host_records);
    return EXIT_SUCCESS;
}
