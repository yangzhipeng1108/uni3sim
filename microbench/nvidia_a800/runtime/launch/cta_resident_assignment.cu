/*
 * NVIDIA A800 eight-resident-CTA assignment probe.
 *
 * This operator-independent kernel has no production tensor data or timing
 * target.  A global admission barrier holds the first resident cohort while
 * every 256-thread CTA records its SM ID and global arrival order.  At grid
 * 864, the occupancy contract requires eight resident CTAs on each of 108
 * SMs.  Grid 1024 then exposes the 160 queued tail CTAs after that cohort is
 * released.  Cross-SM clock64 values are recorded but never compared.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>


namespace {

constexpr int kExpectedSmCount = 108;
constexpr int kBlockThreads = 256;
constexpr int kExpectedResidentCtasPerSm = 8;
constexpr int kResidentCohort =
    kExpectedSmCount * kExpectedResidentCtasPerSm;
constexpr int kWarmupLaunches = 3;
constexpr int kDefaultTrials = 3;
constexpr int kGridSizes[] = {1, 108, 864, 1024};


void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::fprintf(
            stderr, "%s failed: %s\n", expression,
            cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}


#define CHECK_CUDA(call) check_cuda((call), #call)


struct alignas(16) CtaRecord {
    std::uint64_t sm_id;
    std::uint64_t arrival_ticket;
    std::uint64_t release_ticket;
    std::uint64_t start_cycle;
    std::uint64_t end_cycle;
};


extern "C" __global__ __launch_bounds__(
        kBlockThreads, kExpectedResidentCtasPerSm)
void cta_resident_assignment_probe(
        unsigned long long* arrival_counter,
        unsigned long long* release_counter,
        unsigned int* release_flag,
        int resident_target,
        CtaRecord* records) {
    __shared__ std::uint64_t sm_id;
    __shared__ std::uint64_t arrival_ticket;
    if (threadIdx.x == 0) {
        std::uint32_t local_sm_id;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(local_sm_id));
        sm_id = local_sm_id;
        arrival_ticket = atomicAdd(arrival_counter, 1ULL);
    }
    __syncthreads();

    if (arrival_ticket < static_cast<std::uint64_t>(resident_target)) {
        if (threadIdx.x == 0) {
            while (atomicAdd(arrival_counter, 0ULL)
                    < static_cast<unsigned long long>(resident_target)) {
                __nanosleep(64);
            }
            __threadfence();
            atomicExch(release_flag, 1U);
        }
    } else if (threadIdx.x == 0) {
        while (atomicAdd(release_flag, 0U) == 0U) {
            __nanosleep(64);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const std::uint64_t start = clock64();
        const std::uint64_t released = atomicAdd(release_counter, 1ULL);
        const std::uint64_t end = clock64();
        CtaRecord& record = records[blockIdx.x];
        record.sm_id = sm_id;
        record.arrival_ticket = arrival_ticket;
        record.release_ticket = released;
        record.start_cycle = start;
        record.end_cycle = end;
    }
}


std::string cuda_uuid_string(const cudaUUID_t& uuid) {
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(
        uuid.bytes);
    char text[64];
    const int written = std::snprintf(
        text, sizeof(text),
        "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-"
        "%02x%02x%02x%02x%02x%02x",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
        bytes[8], bytes[9], bytes[10], bytes[11],
        bytes[12], bytes[13], bytes[14], bytes[15]);
    if (written <= 0 || written >= static_cast<int>(sizeof(text))) {
        std::fprintf(stderr, "failed to format CUDA UUID\n");
        std::exit(EXIT_FAILURE);
    }
    return text;
}


int parse_trials(int argc, char** argv) {
    int trials = kDefaultTrials;
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--trials") != 0
                || index + 1 >= argc) {
            std::fprintf(stderr, "usage: %s [--trials N]\n", argv[0]);
            std::exit(EXIT_FAILURE);
        }
        char* end = nullptr;
        const long parsed = std::strtol(argv[++index], &end, 10);
        if (*argv[index] == '\0' || *end != '\0'
                || parsed < 1 || parsed > 11) {
            std::fprintf(stderr, "trials must be in [1, 11]\n");
            std::exit(EXIT_FAILURE);
        }
        trials = static_cast<int>(parsed);
    }
    return trials;
}


}  // namespace


int main(int argc, char** argv) {
    const int trials = parse_trials(argc, argv);
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count != 1) {
        std::fprintf(
            stderr, "CUDA_VISIBLE_DEVICES must expose exactly one GPU, got %d\n",
            device_count);
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaSetDevice(0));
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
    if (std::strcmp(properties.name, "NVIDIA A800-SXM4-80GB") != 0
            || properties.major != 8 || properties.minor != 0
            || properties.multiProcessorCount != kExpectedSmCount) {
        std::fprintf(
            stderr, "expected A800 sm_80 with 108 SMs, got %s sm_%d%d/%d\n",
            properties.name, properties.major, properties.minor,
            properties.multiProcessorCount);
        return EXIT_FAILURE;
    }
    int resident_ctas = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_ctas, cta_resident_assignment_probe,
        kBlockThreads, 0));
    if (resident_ctas != kExpectedResidentCtasPerSm) {
        std::fprintf(
            stderr, "256-thread probe must admit exactly 8 CTA/SM, got %d\n",
            resident_ctas);
        return EXIT_FAILURE;
    }

    constexpr int maximum_grid = 1024;
    CtaRecord* device_records = nullptr;
    unsigned long long* arrival_counter = nullptr;
    unsigned long long* release_counter = nullptr;
    unsigned int* release_flag = nullptr;
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&device_records),
        maximum_grid * sizeof(CtaRecord)));
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&arrival_counter),
        sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&release_counter),
        sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&release_flag), sizeof(unsigned int)));

    auto launch = [&](int grid) {
        CHECK_CUDA(cudaMemset(
            device_records, 0, grid * sizeof(CtaRecord)));
        CHECK_CUDA(cudaMemset(
            arrival_counter, 0, sizeof(unsigned long long)));
        CHECK_CUDA(cudaMemset(
            release_counter, 0, sizeof(unsigned long long)));
        CHECK_CUDA(cudaMemset(release_flag, 0, sizeof(unsigned int)));
        const int resident_target = std::min(grid, kResidentCohort);
        cta_resident_assignment_probe<<<grid, kBlockThreads>>>(
            arrival_counter, release_counter, release_flag,
            resident_target, device_records);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    };
    for (int warmup = 0; warmup < kWarmupLaunches; ++warmup) {
        launch(kResidentCohort);
    }

    char pci_bus_id[32];
    CHECK_CUDA(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), 0));
    std::printf("meta,schema,nvidia_a800_cta_resident_assignment_native_v1\n");
    std::printf("meta,device_ordinal,0\n");
    std::printf("meta,device_name,%s\n", properties.name);
    std::printf("meta,device_uuid,%s\n", cuda_uuid_string(
        properties.uuid).c_str());
    std::printf("meta,pci_bus_id,%s\n", pci_bus_id);
    std::printf("meta,compute_capability,sm_80\n");
    std::printf("meta,sm_count,%d\n", kExpectedSmCount);
    std::printf("meta,block_threads,%d\n", kBlockThreads);
    std::printf("meta,resident_ctas_per_sm,%d\n", resident_ctas);
    std::printf("meta,warmup_launches,%d\n", kWarmupLaunches);
    std::printf("meta,production_or_evaluation_duration_consumed,false\n");
    std::printf(
        "raw,grid_size,trial,block_index,sm_id,arrival_ticket,"
        "release_ticket,start_cycle,end_cycle\n");

    std::vector<CtaRecord> host_records(maximum_grid);
    for (int trial = 0; trial < trials; ++trial) {
        for (int grid : kGridSizes) {
            launch(grid);
            CHECK_CUDA(cudaMemcpy(
                host_records.data(), device_records,
                grid * sizeof(CtaRecord), cudaMemcpyDeviceToHost));
            for (int block = 0; block < grid; ++block) {
                const CtaRecord& record = host_records[block];
                if (record.sm_id >= static_cast<std::uint64_t>(
                        kExpectedSmCount)
                        || record.arrival_ticket >= static_cast<std::uint64_t>(
                            grid)
                        || record.release_ticket >= static_cast<std::uint64_t>(
                            grid)
                        || record.end_cycle < record.start_cycle) {
                    std::fprintf(
                        stderr, "invalid record grid=%d trial=%d block=%d\n",
                        grid, trial, block);
                    return EXIT_FAILURE;
                }
                std::printf(
                    "raw,%d,%d,%d,%llu,%llu,%llu,%llu,%llu\n",
                    grid, trial, block,
                    static_cast<unsigned long long>(record.sm_id),
                    static_cast<unsigned long long>(record.arrival_ticket),
                    static_cast<unsigned long long>(record.release_ticket),
                    static_cast<unsigned long long>(record.start_cycle),
                    static_cast<unsigned long long>(record.end_cycle));
            }
        }
    }

    CHECK_CUDA(cudaFree(release_flag));
    CHECK_CUDA(cudaFree(release_counter));
    CHECK_CUDA(cudaFree(arrival_counter));
    CHECK_CUDA(cudaFree(device_records));
    return EXIT_SUCCESS;
}
