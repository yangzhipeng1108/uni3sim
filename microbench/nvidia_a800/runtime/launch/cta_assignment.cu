/*
 * NVIDIA A800 CTA-assignment mechanism benchmark.
 *
 * This is an operator-independent scheduler probe.  A 96-KiB dynamic shared
 * memory request restricts the kernel to one resident CTA per SM.  One warm
 * CUDA context executes every warmup, cell, and trial.  Uniform dummy integer
 * work is held identical across every CTA while exact grid sizes exercise
 * sub-wave, full-wave, residual-tail, and multi-wave assignment.  Existing
 * heterogeneous controls remain solely to distinguish queue dispatch from a
 * whole-wave barrier when uniform CTAs complete nearly simultaneously.  They
 * do not establish that scheduler behavior transfers to another CTA program.
 *
 * Every CTA records its SM ID, two same-SM clock64 boundaries, source work,
 * and globally ordered start/end event tickets.  clock64 values from different
 * SMs are deliberately never compared.  The interval and the gap between two
 * local intervals include probe barriers, metadata, CTA teardown, and CTA
 * entry; neither is a bare CTA reissue latency.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <utility>
#include <vector>


namespace {

constexpr int kExpectedSmCount = 108;
constexpr int kDynamicSharedBytes = 96 * 1024;
constexpr int kBlockThreads = 128;
constexpr int kWarmupLaunches = 3;
constexpr int kDefaultTrials = 7;
constexpr std::uint64_t kBaseWorkIterations = 2048;
constexpr int kGridSizes[] = {
    1, 27, 54, 81, 107, 108,
    109, 135, 161, 163, 189, 215, 216,
    217, 243, 269, 323, 324,
    431, 432, 433, 864,
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


struct alignas(16) CtaRecord {
    std::uint64_t sm_id;
    std::uint64_t start_event_ticket;
    std::uint64_t end_event_ticket;
    std::uint64_t start_cycle;
    std::uint64_t end_cycle;
    std::uint64_t work_iterations;
    std::uint64_t work_cycles;
    std::uint64_t checksum;
};


enum class WorkPattern : int {
    kUniformControl = 0,
    kAlternatingEightToOne = 1,
    kHeadStragglersSixteenToOne = 2,
};


struct Cell {
    WorkPattern pattern;
    const char* pattern_name;
    int grid_size;
};


struct Options {
    int trials;
    bool smoke_cell;
};


constexpr Cell kCells[] = {
    {WorkPattern::kUniformControl, "uniform_control", 1},
    {WorkPattern::kUniformControl, "uniform_control", 27},
    {WorkPattern::kUniformControl, "uniform_control", 54},
    {WorkPattern::kUniformControl, "uniform_control", 81},
    {WorkPattern::kUniformControl, "uniform_control", 107},
    {WorkPattern::kUniformControl, "uniform_control", 108},
    {WorkPattern::kUniformControl, "uniform_control", 109},
    {WorkPattern::kUniformControl, "uniform_control", 135},
    {WorkPattern::kUniformControl, "uniform_control", 161},
    {WorkPattern::kUniformControl, "uniform_control", 163},
    {WorkPattern::kUniformControl, "uniform_control", 189},
    {WorkPattern::kUniformControl, "uniform_control", 215},
    {WorkPattern::kUniformControl, "uniform_control", 216},
    {WorkPattern::kUniformControl, "uniform_control", 217},
    {WorkPattern::kUniformControl, "uniform_control", 243},
    {WorkPattern::kUniformControl, "uniform_control", 269},
    {WorkPattern::kUniformControl, "uniform_control", 323},
    {WorkPattern::kUniformControl, "uniform_control", 324},
    {WorkPattern::kUniformControl, "uniform_control", 431},
    {WorkPattern::kUniformControl, "uniform_control", 432},
    {WorkPattern::kUniformControl, "uniform_control", 433},
    {WorkPattern::kUniformControl, "uniform_control", 864},
    {WorkPattern::kAlternatingEightToOne, "alternating_8to1", 216},
    {WorkPattern::kAlternatingEightToOne, "alternating_8to1", 432},
    {WorkPattern::kAlternatingEightToOne, "alternating_8to1", 864},
    {WorkPattern::kHeadStragglersSixteenToOne,
     "head_stragglers_16to1", 216},
    {WorkPattern::kHeadStragglersSixteenToOne,
     "head_stragglers_16to1", 432},
    {WorkPattern::kHeadStragglersSixteenToOne,
     "head_stragglers_16to1", 864},
};


__host__ __device__ std::uint64_t work_iterations_for(
        WorkPattern pattern, int block_index, int sm_count) {
    if (pattern == WorkPattern::kUniformControl) {
        return kBaseWorkIterations;
    }
    if (pattern == WorkPattern::kAlternatingEightToOne) {
        return kBaseWorkIterations * ((block_index & 1) == 0 ? 1 : 8);
    }
    if (pattern == WorkPattern::kHeadStragglersSixteenToOne) {
        const bool is_head_straggler = (
            block_index < sm_count && (block_index % 4) == 0);
        return kBaseWorkIterations * (is_head_straggler ? 16 : 1);
    }
#if defined(__CUDA_ARCH__)
    asm volatile("trap;");
#else
    std::fprintf(stderr, "unknown work pattern\n");
    std::abort();
#endif
    return 0;
}


extern "C" __global__ __launch_bounds__(kBlockThreads, 1)
void cta_assignment_probe(
        WorkPattern pattern, int sm_count,
        unsigned long long* event_counter,
        CtaRecord* records) {
    extern __shared__ std::uint32_t work_sink[];
    __shared__ std::uint64_t sm_id;
    __shared__ std::uint64_t start_event_ticket;
    __shared__ std::uint64_t start_cycle;
    __shared__ std::uint64_t work_iterations;

    if (threadIdx.x == 0) {
        std::uint32_t local_sm_id;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(local_sm_id));
        sm_id = local_sm_id;
        start_event_ticket = atomicAdd(event_counter, 1ULL);
        work_iterations = work_iterations_for(
            pattern, static_cast<int>(blockIdx.x), sm_count);
        start_cycle = clock64();
    }
    __syncthreads();

    std::uint32_t state = (
        0x9e3779b9U ^ static_cast<std::uint32_t>(blockIdx.x * blockDim.x)
        ^ static_cast<std::uint32_t>(threadIdx.x));
    #pragma unroll 1
    for (std::uint64_t iteration = 0;
         iteration < work_iterations; ++iteration) {
        state = state * 1664525U + 1013904223U;
        state ^= state >> 13;
    }
    work_sink[threadIdx.x] = state;
    __syncthreads();

    if (threadIdx.x == 0) {
        std::uint64_t checksum = 0;
        for (int thread = 0; thread < blockDim.x; ++thread) {
            checksum += work_sink[thread];
        }
        const std::uint64_t end_cycle = clock64();
        const std::uint64_t end_event_ticket = atomicAdd(
            event_counter, 1ULL);
        CtaRecord& record = records[blockIdx.x];
        record.sm_id = sm_id;
        record.start_event_ticket = start_event_ticket;
        record.end_event_ticket = end_event_ticket;
        record.start_cycle = start_cycle;
        record.end_cycle = end_cycle;
        record.work_iterations = work_iterations;
        record.work_cycles = end_cycle - start_cycle;
        record.checksum = checksum;
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


Options parse_options(int argc, char** argv) {
    Options options{kDefaultTrials, false};
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--trials") == 0) {
            if (index + 1 >= argc) {
                std::fprintf(stderr, "--trials requires a value\n");
                std::exit(EXIT_FAILURE);
            }
            char* end = nullptr;
            const long parsed = std::strtol(argv[++index], &end, 10);
            if (*argv[index] == '\0' || *end != '\0'
                    || parsed <= 0 || parsed > 100) {
                std::fprintf(stderr, "trials must be in [1, 100]\n");
                std::exit(EXIT_FAILURE);
            }
            options.trials = static_cast<int>(parsed);
        } else if (std::strcmp(argv[index], "--smoke-cell") == 0) {
            options.smoke_cell = true;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[index]);
            std::exit(EXIT_FAILURE);
        }
    }
    if (options.smoke_cell && options.trials != 1) {
        std::fprintf(stderr, "--smoke-cell requires --trials 1\n");
        std::exit(EXIT_FAILURE);
    }
    return options;
}


void validate_records(
        const Cell& cell, int sm_count,
        const std::vector<CtaRecord>& records) {
    std::vector<bool> tickets(records.size() * 2, false);
    for (std::size_t block = 0; block < records.size(); ++block) {
        const CtaRecord& record = records[block];
        const std::uint64_t expected_iterations = work_iterations_for(
            cell.pattern, static_cast<int>(block), sm_count);
        if (record.sm_id >= static_cast<std::uint64_t>(sm_count)
                || record.start_event_ticket >= tickets.size()
                || record.end_event_ticket >= tickets.size()
                || record.start_event_ticket >= record.end_event_ticket
                || tickets[record.start_event_ticket]
                || tickets[record.end_event_ticket]
                || record.start_cycle >= record.end_cycle
                || record.work_cycles != record.end_cycle - record.start_cycle
                || record.work_iterations != expected_iterations) {
            std::fprintf(
                stderr, "invalid CTA record: grid=%zu block=%zu\n",
                records.size(), block);
            std::exit(EXIT_FAILURE);
        }
        tickets[record.start_event_ticket] = true;
        tickets[record.end_event_ticket] = true;
    }
    if (!std::all_of(tickets.begin(), tickets.end(), [](bool value) {
            return value;
        })) {
        std::fprintf(stderr, "CTA event tickets are not a complete ordering\n");
        std::exit(EXIT_FAILURE);
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> starts;
    starts.reserve(records.size());
    for (const CtaRecord& record : records) {
        starts.emplace_back(record.start_event_ticket, record.sm_id);
    }
    std::sort(starts.begin(), starts.end());
    std::vector<bool> initial_sm_ids(sm_count, false);
    const int initial_count = std::min(
        sm_count, static_cast<int>(records.size()));
    for (int index = 0; index < initial_count; ++index) {
        const std::uint64_t sm_id = starts[index].second;
        if (sm_id >= static_cast<std::uint64_t>(sm_count)
                || initial_sm_ids[sm_id]) {
            std::fprintf(
                stderr,
                "initial CTA starts do not cover distinct SMs: grid=%zu "
                "initial_count=%d\n",
                records.size(), initial_count);
            std::exit(EXIT_FAILURE);
        }
        initial_sm_ids[sm_id] = true;
    }
}


}  // namespace


int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
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
            stderr,
            "expected NVIDIA A800-SXM4-80GB sm_80 with 108 SMs, got "
            "%s sm_%d%d with %d SMs\n",
            properties.name, properties.major, properties.minor,
            properties.multiProcessorCount);
        return EXIT_FAILURE;
    }
    int maximum_optin_shared = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(
        &maximum_optin_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (maximum_optin_shared < kDynamicSharedBytes) {
        std::fprintf(stderr, "device cannot opt in to 96-KiB shared memory\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaFuncSetAttribute(
        cta_assignment_probe,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kDynamicSharedBytes));
    int resident_ctas = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_ctas, cta_assignment_probe,
        kBlockThreads, kDynamicSharedBytes));
    if (resident_ctas != 1) {
        std::fprintf(
            stderr, "96-KiB probe must admit exactly one CTA/SM, got %d\n",
            resident_ctas);
        return EXIT_FAILURE;
    }

    const int maximum_grid = *std::max_element(
        std::begin(kGridSizes), std::end(kGridSizes));
    CtaRecord* device_records = nullptr;
    unsigned long long* device_event_counter = nullptr;
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&device_records),
        static_cast<std::size_t>(maximum_grid) * sizeof(CtaRecord)));
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&device_event_counter),
        sizeof(unsigned long long)));

    for (int warmup = 0; warmup < kWarmupLaunches; ++warmup) {
        CHECK_CUDA(cudaMemset(
            device_event_counter, 0, sizeof(unsigned long long)));
        cta_assignment_probe<<<
            kExpectedSmCount, kBlockThreads, kDynamicSharedBytes>>>(
                WorkPattern::kUniformControl, kExpectedSmCount,
                device_event_counter, device_records);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    char pci_bus_id[32];
    CHECK_CUDA(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), 0));
    std::printf("meta,schema,nvidia_a800_cta_assignment_native_v2\n");
    std::printf("meta,device_ordinal,0\n");
    std::printf("meta,device_name,%s\n", properties.name);
    std::printf(
        "meta,device_uuid,%s\n",
        cuda_uuid_string(properties.uuid).c_str());
    std::printf("meta,pci_bus_id,%s\n", pci_bus_id);
    std::printf("meta,compute_capability,sm_80\n");
    std::printf("meta,sm_count,%d\n", kExpectedSmCount);
    std::printf(
        "meta,dynamic_shared_bytes_per_cta,%d\n", kDynamicSharedBytes);
    std::printf("meta,resident_ctas_per_sm,%d\n", resident_ctas);
    std::printf("meta,block_threads,%d\n", kBlockThreads);
    std::printf("meta,warmup_launches,%d\n", kWarmupLaunches);
    std::printf("meta,warmup_complete,true\n");
    std::printf(
        "meta,capture_mode,%s\n",
        options.smoke_cell ? "smoke_cell" : "full_matrix");
    std::printf("meta,timer_domain,clock64_sm_local_cycles\n");
    std::printf("meta,cross_sm_clocks_comparable,false\n");
    std::printf("meta,event_order_domain,global_atomic_ticket\n");
    std::printf("meta,uniform_scenario_cta_program_identical,true\n");
    std::printf("meta,production_or_evaluation_duration_consumed,false\n");
    std::printf("meta,production_parameter_published,false\n");
    std::printf(
        "raw,scenario,grid_size,trial,block_index,sm_id,"
        "start_event_ticket,end_event_ticket,start_cycle,end_cycle,"
        "work_iterations,work_cycles,checksum\n");

    std::vector<CtaRecord> host_records(maximum_grid);
    constexpr int cell_count = sizeof(kCells) / sizeof(kCells[0]);
    for (int trial = 0; trial < options.trials; ++trial) {
        const int rotation = (trial * 2) % cell_count;
        for (int position = 0; position < cell_count; ++position) {
            const Cell& cell = kCells[(position + rotation) % cell_count];
            if (options.smoke_cell
                    && !(cell.pattern == WorkPattern::kUniformControl
                         && cell.grid_size == 109)) {
                continue;
            }
            const int grid = cell.grid_size;
            CHECK_CUDA(cudaMemset(
                device_event_counter, 0, sizeof(unsigned long long)));
            CHECK_CUDA(cudaMemset(
                device_records, 0,
                static_cast<std::size_t>(grid) * sizeof(CtaRecord)));
            cta_assignment_probe<<<
                grid, kBlockThreads, kDynamicSharedBytes>>>(
                    cell.pattern, kExpectedSmCount,
                    device_event_counter, device_records);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(
                host_records.data(), device_records,
                static_cast<std::size_t>(grid) * sizeof(CtaRecord),
                cudaMemcpyDeviceToHost));
            const std::vector<CtaRecord> cell_records(
                host_records.begin(), host_records.begin() + grid);
            validate_records(cell, kExpectedSmCount, cell_records);
            for (int block = 0; block < grid; ++block) {
                const CtaRecord& record = cell_records[block];
                std::printf(
                    "raw,%s,%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%llu,"
                    "%llu,%llu\n",
                    cell.pattern_name, cell.grid_size, trial, block,
                    static_cast<unsigned long long>(record.sm_id),
                    static_cast<unsigned long long>(record.start_event_ticket),
                    static_cast<unsigned long long>(record.end_event_ticket),
                    static_cast<unsigned long long>(record.start_cycle),
                    static_cast<unsigned long long>(record.end_cycle),
                    static_cast<unsigned long long>(record.work_iterations),
                    static_cast<unsigned long long>(record.work_cycles),
                    static_cast<unsigned long long>(record.checksum));
            }
        }
    }

    CHECK_CUDA(cudaFree(device_event_counter));
    CHECK_CUDA(cudaFree(device_records));
    return EXIT_SUCCESS;
}
