/*
 * MetaX C500 empty-task retained-capture executable.
 *
 * The retained mcEvent campaign uses the independent empty kernel (endk after
 * compilation).  The optional ``--body minimal`` path is a register-only
 * non-empty control for the separate mcTracer scheduler-floor campaign.  The
 * ``--body memory`` path adds one coalesced load -> dependent add -> store per
 * hardware wave so that its increment can be compared with the minimal body
 * in the same mcTracer timer domain.  No path contains production operator
 * work.  The Python runner owns the fixed matrix and sends one
 * whitespace-delimited plan row per sample on stdin.
 *
 * Build in the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch=native \
 *       --maca-path=/opt/maca -O3 empty_kernel_duration.cpp \
 *       -o empty_kernel_duration
 */

#include <mc_runtime.h>

#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <time.h>
#include <tuple>
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

constexpr int kGeometryWarmupLaunches = 16;
constexpr int kExpectedApCount = 104;
constexpr int kExpectedWaveSize = 64;


struct PlanRow {
    std::string sample_id;
    int trial;
    int order_position;
    std::string pair_id;
    std::string geometry_id;
    int grid_x;
    int grid_y;
    int grid_z;
    int block_x;
    int block_y;
    int block_z;
    int launch_count;
};


struct GeometryProbeRecord {
    std::uint32_t slot;
    std::uint32_t block_x;
    std::uint32_t block_y;
    std::uint32_t block_z;
    std::uint32_t grid_x;
    std::uint32_t grid_y;
    std::uint32_t grid_z;
    std::uint32_t block_dim_x;
    std::uint32_t block_dim_y;
    std::uint32_t block_dim_z;
};


struct GeometryProbeCase {
    std::uint32_t grid_x;
    std::uint32_t grid_y;
    std::uint32_t grid_z;
    std::uint32_t block_x;
    std::uint32_t block_y;
    std::uint32_t block_z;
};


constexpr GeometryProbeCase kGeometryProbeCases[] = {
    {32, 64, 128, 512, 1, 1},
    {262144, 1, 1, 512, 1, 1},
    {1, 512, 2, 1024, 1, 1},
    {1024, 1, 1, 1024, 1, 1},
};


__global__ void empty_kernel() {}


// Independent non-empty lower-bound control for the mcTracer timer domain.
// The volatile register-only instruction prevents an empty-body fast path
// without introducing global/shared traffic, a barrier, or operator work.
__global__ void minimal_register_kernel() {
    const float input = static_cast<float>(threadIdx.x & 1U);
    float output;
    asm volatile("add_f32 %0, %1, %1"
                 : "=r"(output) : "r"(input));
    asm volatile("" : : "r"(output));
}


__global__ void memory_dependency_kernel(
        const std::uint32_t* input, std::uint32_t* output) {
    const int linear_thread = (
        threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));
    const int lane = linear_thread & (kExpectedWaveSize - 1);
    if (lane != 0) {
        return;
    }
    const int waves_per_block = (
        (blockDim.x * blockDim.y * blockDim.z + kExpectedWaveSize - 1)
        / kExpectedWaveSize);
    const int wave = linear_thread / kExpectedWaveSize;
    const int flattened_block = (
        blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));
    const int index = flattened_block * waves_per_block + wave;
    output[index] = input[index] + 1U;
}


enum class KernelBody {
    kEmpty,
    kMinimalRegister,
    kMemoryDependency,
};


void launch_body(
        KernelBody body, dim3 grid, dim3 block,
        const std::uint32_t* input, std::uint32_t* output) {
    if (body == KernelBody::kEmpty) {
        empty_kernel<<<grid, block>>>();
    } else if (body == KernelBody::kMinimalRegister) {
        minimal_register_kernel<<<grid, block>>>();
    } else if (body == KernelBody::kMemoryDependency) {
        memory_dependency_kernel<<<grid, block>>>(input, output);
    } else {
        std::fprintf(stderr, "unknown kernel body\n");
        std::exit(EXIT_FAILURE);
    }
}


// The atomic slot is deliberately independent of blockIdx.  A flattened,
// swapped, missing, or duplicate 3-D coordinate therefore remains visible to
// the host validator instead of selecting the same faulty output address.
__global__ void geometry_probe_kernel(
        GeometryProbeRecord* records, unsigned int* allocation_count,
        unsigned int capacity) {
    if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {
        return;
    }
    const unsigned int slot = atomicAdd(allocation_count, 1U);
    if (slot >= capacity) {
        return;
    }
    GeometryProbeRecord record {};
    record.slot = slot;
    record.block_x = blockIdx.x;
    record.block_y = blockIdx.y;
    record.block_z = blockIdx.z;
    record.grid_x = gridDim.x;
    record.grid_y = gridDim.y;
    record.grid_z = gridDim.z;
    record.block_dim_x = blockDim.x;
    record.block_dim_y = blockDim.y;
    record.block_dim_z = blockDim.z;
    records[slot] = record;
}


long long monotonic_raw_ns() {
    struct timespec value {};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) {
        std::fprintf(
            stderr, "clock_gettime(CLOCK_MONOTONIC_RAW) failed: %s\n",
            std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    return (
        static_cast<long long>(value.tv_sec) * 1000000000LL
        + static_cast<long long>(value.tv_nsec));
}


int parse_nonnegative_int(const char* text, const char* label) {
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 0
            || value > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "%s must be a non-negative integer: %s\n",
                     label, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}


void validate_token(const std::string& value, const char* label) {
    if (value.empty() || value.size() > 96) {
        std::fprintf(stderr, "%s has invalid length\n", label);
        std::exit(EXIT_FAILURE);
    }
    for (char character : value) {
        const bool valid = (
            (character >= 'a' && character <= 'z')
            || (character >= 'A' && character <= 'Z')
            || (character >= '0' && character <= '9')
            || character == '_' || character == '-' || character == '.');
        if (!valid) {
            std::fprintf(stderr, "%s contains an invalid character\n", label);
            std::exit(EXIT_FAILURE);
        }
    }
}


void validate_plan_row(const PlanRow& row) {
    validate_token(row.sample_id, "sample_id");
    validate_token(row.pair_id, "pair_id");
    validate_token(row.geometry_id, "geometry_id");
    if (row.trial <= 0 || row.order_position < 0) {
        std::fprintf(stderr, "trial/order_position is invalid for %s\n",
                     row.sample_id.c_str());
        std::exit(EXIT_FAILURE);
    }
    if (row.grid_x <= 0 || row.grid_y <= 0 || row.grid_z <= 0
            || row.block_x <= 0 || row.block_y <= 0 || row.block_z <= 0
            || row.launch_count <= 0) {
        std::fprintf(stderr, "non-positive launch geometry for %s\n",
                     row.sample_id.c_str());
        std::exit(EXIT_FAILURE);
    }
    const long long block_threads = (
        static_cast<long long>(row.block_x) * row.block_y * row.block_z);
    if (block_threads > 1024) {
        std::fprintf(stderr, "block has more than 1024 threads for %s\n",
                     row.sample_id.c_str());
        std::exit(EXIT_FAILURE);
    }
}


std::vector<PlanRow> read_plan() {
    std::vector<PlanRow> rows;
    std::set<std::string> sample_ids;
    std::string line;
    int line_number = 0;
    while (std::getline(std::cin, line)) {
        ++line_number;
        if (line.empty()) {
            std::fprintf(stderr, "empty plan line at %d\n", line_number);
            std::exit(EXIT_FAILURE);
        }
        std::istringstream stream(line);
        PlanRow row {};
        if (!(stream >> row.sample_id >> row.trial >> row.order_position
                    >> row.pair_id >> row.geometry_id
                    >> row.grid_x >> row.grid_y >> row.grid_z
                    >> row.block_x >> row.block_y >> row.block_z
                    >> row.launch_count)) {
            std::fprintf(stderr, "invalid plan schema at line %d\n", line_number);
            std::exit(EXIT_FAILURE);
        }
        std::string trailing;
        if (stream >> trailing) {
            std::fprintf(stderr, "trailing plan field at line %d\n", line_number);
            std::exit(EXIT_FAILURE);
        }
        validate_plan_row(row);
        if (!sample_ids.insert(row.sample_id).second) {
            std::fprintf(stderr, "duplicate sample_id: %s\n",
                         row.sample_id.c_str());
            std::exit(EXIT_FAILURE);
        }
        rows.push_back(row);
    }
    if (!std::cin.eof()) {
        std::fprintf(stderr, "failed while reading plan stdin\n");
        std::exit(EXIT_FAILURE);
    }
    if (rows.empty()) {
        std::fprintf(stderr, "plan stdin must contain at least one row\n");
        std::exit(EXIT_FAILURE);
    }
    return rows;
}


void warm_geometries(
        const std::vector<PlanRow>& rows, KernelBody body,
        const std::uint32_t* input, std::uint32_t* output) {
    using Geometry = std::tuple<int, int, int, int, int, int>;
    std::set<Geometry> geometries;
    for (const PlanRow& row : rows) {
        geometries.insert(Geometry(
            row.grid_x, row.grid_y, row.grid_z,
            row.block_x, row.block_y, row.block_z));
    }
    for (const Geometry& geometry : geometries) {
        const dim3 grid(
            std::get<0>(geometry), std::get<1>(geometry),
            std::get<2>(geometry));
        const dim3 block(
            std::get<3>(geometry), std::get<4>(geometry),
            std::get<5>(geometry));
        for (int launch = 0; launch < kGeometryWarmupLaunches; ++launch) {
            launch_body(body, grid, block, input, output);
        }
    }
    CHECK(mcDeviceSynchronize());
}


void write_probe_bytes(const void* data, std::size_t size) {
    if (std::fwrite(data, 1, size, stdout) != size) {
        std::fprintf(stderr, "failed to write geometry probe protocol\n");
        std::exit(EXIT_FAILURE);
    }
}


void write_probe_u32(std::uint32_t value) {
    const unsigned char bytes[4] = {
        static_cast<unsigned char>(value & 0xffU),
        static_cast<unsigned char>((value >> 8) & 0xffU),
        static_cast<unsigned char>((value >> 16) & 0xffU),
        static_cast<unsigned char>((value >> 24) & 0xffU),
    };
    write_probe_bytes(bytes, sizeof(bytes));
}


void run_geometry_probe() {
    constexpr char kMagic[] = "C500GEO1";
    write_probe_bytes(kMagic, sizeof(kMagic) - 1);
    write_probe_u32(static_cast<std::uint32_t>(
        sizeof(kGeometryProbeCases) / sizeof(kGeometryProbeCases[0])));

    for (std::uint32_t probe_index = 0;
            probe_index < sizeof(kGeometryProbeCases)
                / sizeof(kGeometryProbeCases[0]);
            ++probe_index) {
        const GeometryProbeCase& probe = kGeometryProbeCases[probe_index];
        const std::uint32_t expected_count = (
            probe.grid_x * probe.grid_y * probe.grid_z);
        GeometryProbeRecord* device_records = nullptr;
        unsigned int* device_count = nullptr;
        CHECK(mcMalloc(
            reinterpret_cast<void**>(&device_records),
            static_cast<std::size_t>(expected_count)
                * sizeof(GeometryProbeRecord)));
        CHECK(mcMalloc(
            reinterpret_cast<void**>(&device_count), sizeof(*device_count)));
        CHECK(mcMemset(device_count, 0, sizeof(*device_count)));

        geometry_probe_kernel<<<
            dim3(probe.grid_x, probe.grid_y, probe.grid_z),
            dim3(probe.block_x, probe.block_y, probe.block_z)>>>(
                device_records, device_count, expected_count);
        CHECK(mcDeviceSynchronize());

        unsigned int actual_count = 0;
        CHECK(mcMemcpy(
            &actual_count, device_count, sizeof(actual_count),
            mcMemcpyDeviceToHost));
        const std::uint32_t stored_count = (
            actual_count < expected_count ? actual_count : expected_count);
        std::vector<GeometryProbeRecord> records(stored_count);
        if (stored_count != 0) {
            CHECK(mcMemcpy(
                records.data(), device_records,
                static_cast<std::size_t>(stored_count)
                    * sizeof(GeometryProbeRecord),
                mcMemcpyDeviceToHost));
        }
        CHECK(mcFree(device_count));
        CHECK(mcFree(device_records));

        write_probe_u32(probe_index);
        write_probe_u32(probe.grid_x);
        write_probe_u32(probe.grid_y);
        write_probe_u32(probe.grid_z);
        write_probe_u32(probe.block_x);
        write_probe_u32(probe.block_y);
        write_probe_u32(probe.block_z);
        write_probe_u32(expected_count);
        write_probe_u32(actual_count);
        write_probe_u32(stored_count);
        for (const GeometryProbeRecord& record : records) {
            write_probe_u32(record.slot);
            write_probe_u32(record.block_x);
            write_probe_u32(record.block_y);
            write_probe_u32(record.block_z);
            write_probe_u32(record.grid_x);
            write_probe_u32(record.grid_y);
            write_probe_u32(record.grid_z);
            write_probe_u32(record.block_dim_x);
            write_probe_u32(record.block_dim_y);
            write_probe_u32(record.block_dim_z);
        }
    }
    if (std::fflush(stdout) != 0) {
        std::fprintf(stderr, "failed to flush geometry probe protocol\n");
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace


int main(int argc, char** argv) {
    const bool geometry_probe_mode = (
        argc == 4 && std::strcmp(argv[1], "--geometry-probe") == 0
        && std::strcmp(argv[2], "--device-id") == 0);
    const bool timing_mode = (
        argc == 3 && std::strcmp(argv[1], "--device-id") == 0);
    const bool body_timing_mode = (
        argc == 5 && std::strcmp(argv[1], "--body") == 0
        && std::strcmp(argv[3], "--device-id") == 0);
    if (!geometry_probe_mode && !timing_mode && !body_timing_mode) {
        std::fprintf(
            stderr,
            "usage: %s [--geometry-probe] [--body empty|minimal|memory] "
            "--device-id DEVICE_ID < plan\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    KernelBody body = KernelBody::kEmpty;
    if (body_timing_mode) {
        if (std::strcmp(argv[2], "empty") == 0) {
            body = KernelBody::kEmpty;
        } else if (std::strcmp(argv[2], "minimal") == 0) {
            body = KernelBody::kMinimalRegister;
        } else if (std::strcmp(argv[2], "memory") == 0) {
            body = KernelBody::kMemoryDependency;
        } else {
            std::fprintf(stderr, "body must be empty, minimal, or memory\n");
            return EXIT_FAILURE;
        }
    }
    const int device_id = parse_nonnegative_int(
        argv[geometry_probe_mode ? 3 : (body_timing_mode ? 4 : 2)],
        "device-id");
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
    if (properties.clockRate <= 0) {
        std::fprintf(stderr, "device must report a positive clock rate\n");
        return EXIT_FAILURE;
    }
    if (geometry_probe_mode) {
        run_geometry_probe();
        return EXIT_SUCCESS;
    }
    char pci_bus_id[32] = {};
    CHECK(mcDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device_id));
    for (char* character = pci_bus_id; *character != '\0'; ++character) {
        *character = static_cast<char>(std::tolower(
            static_cast<unsigned char>(*character)));
    }
    if (std::strchr(pci_bus_id, ',') != nullptr
            || std::strchr(pci_bus_id, '\n') != nullptr) {
        std::fprintf(stderr, "PCI bus id is not safe for CSV output\n");
        return EXIT_FAILURE;
    }
    if (std::strchr(properties.name, ',') != nullptr
            || std::strchr(properties.name, '\n') != nullptr) {
        std::fprintf(stderr, "device name is not safe for CSV output\n");
        return EXIT_FAILURE;
    }

    const std::vector<PlanRow> plan = read_plan();
    std::uint32_t* input = nullptr;
    std::uint32_t* output = nullptr;
    if (body == KernelBody::kMemoryDependency) {
        std::size_t maximum_words = 0;
        for (const PlanRow& row : plan) {
            const std::size_t grid_size = (
                static_cast<std::size_t>(row.grid_x) * row.grid_y
                * row.grid_z);
            const std::size_t block_size = (
                static_cast<std::size_t>(row.block_x) * row.block_y
                * row.block_z);
            const std::size_t waves = (
                (block_size + kExpectedWaveSize - 1) / kExpectedWaveSize);
            maximum_words = std::max(maximum_words, grid_size * waves);
        }
        CHECK(mcMalloc(reinterpret_cast<void**>(&input),
                       maximum_words * sizeof(*input)));
        CHECK(mcMalloc(reinterpret_cast<void**>(&output),
                       maximum_words * sizeof(*output)));
        CHECK(mcMemset(input, 1, maximum_words * sizeof(*input)));
        CHECK(mcMemset(output, 0, maximum_words * sizeof(*output)));
    }
    warm_geometries(plan, body, input, output);

    std::printf(
        "metadata,6,%s,%d,%s,%d,%d,%d,CLOCK_MONOTONIC_RAW,mcEvent\n",
        properties.name, device_id, pci_bus_id, properties.multiProcessorCount,
        properties.warpSize, properties.clockRate);
    std::printf(
        "sample_header,sample_id,trial,order_position,pair_id,geometry_id,"
        "grid_x,grid_y,grid_z,block_x,block_y,block_z,launch_count,"
        "host_start_ns,host_end_ns,batch_device_ns,per_launch_device_ns,"
        "device_elapsed_ms_hex,event_status\n");
    std::fflush(stdout);

    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    for (const PlanRow& row : plan) {
        const dim3 grid(row.grid_x, row.grid_y, row.grid_z);
        const dim3 block(row.block_x, row.block_y, row.block_z);
        const long long host_start_ns = monotonic_raw_ns();
        CHECK(mcEventRecord(start));
        for (int launch = 0; launch < row.launch_count; ++launch) {
            launch_body(body, grid, block, input, output);
        }
        CHECK(mcEventRecord(end));
        CHECK(mcDeviceSynchronize());
        const long long host_end_ns = monotonic_raw_ns();
        float elapsed_ms = 0.0f;
        CHECK(mcEventElapsedTime(&elapsed_ms, start, end));
        if (!std::isfinite(elapsed_ms) || elapsed_ms < 0.0f) {
            std::fprintf(stderr, "invalid mcEvent sample: %s %.9g ms\n",
                         row.sample_id.c_str(), elapsed_ms);
            return EXIT_FAILURE;
        }
        const double batch_device_ns = (
            static_cast<double>(elapsed_ms) * 1000000.0);
        const long long host_elapsed_ns = host_end_ns - host_start_ns;
        if (host_elapsed_ns <= 0
                || static_cast<double>(host_elapsed_ns) < batch_device_ns) {
            std::fprintf(
                stderr,
                "host elapsed is smaller than device elapsed for %s: "
                "%lld < %.9f ns\n",
                row.sample_id.c_str(), host_elapsed_ns, batch_device_ns);
            return EXIT_FAILURE;
        }
        const char* event_status = (
            elapsed_ms == 0.0f ? "quantized_zero" : "resolved");
        std::printf(
            "sample,%s,%d,%d,%s,%s,%d,%d,%d,%d,%d,%d,%d,"
            "%lld,%lld,%.9f,%.9f,%a,%s\n",
            row.sample_id.c_str(), row.trial, row.order_position,
            row.pair_id.c_str(), row.geometry_id.c_str(),
            row.grid_x, row.grid_y, row.grid_z,
            row.block_x, row.block_y, row.block_z, row.launch_count,
            host_start_ns, host_end_ns, batch_device_ns,
            batch_device_ns / row.launch_count,
            static_cast<double>(elapsed_ms), event_status);
        std::fflush(stdout);
    }
    CHECK(mcEventDestroy(start));
    CHECK(mcEventDestroy(end));
    if (body == KernelBody::kMemoryDependency) {
        CHECK(mcFree(output));
        CHECK(mcFree(input));
    }
    return EXIT_SUCCESS;
}
