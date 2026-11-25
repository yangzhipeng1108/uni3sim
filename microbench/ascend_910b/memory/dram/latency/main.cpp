// Host driver for the AIV dependent GlobalTensor<int32_t>::GetValue benchmark.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <random>
#include <string>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    const int status = static_cast<int>(call);                            \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",    \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

#define CLOCK_CHECK(call) do {                                            \
    if ((call) != 0) {                                                    \
        std::perror(#call);                                               \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_dram_latency(
    uint32_t blockDim, void* stream, void* buffer, void* output,
    int64_t hops);

__attribute__((constructor(101))) static void early_acl_init() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

static int64_t parse_positive_integer(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value <= 0) {
        std::fprintf(stderr, "%s must be a positive integer, got %s\n",
                     name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

static int64_t elapsed_ns(const timespec& start, const timespec& end) {
    return static_cast<int64_t>(end.tv_sec - start.tv_sec) * 1000000000LL
        + static_cast<int64_t>(end.tv_nsec - start.tv_nsec);
}

struct RunResult {
    int64_t cycles;
    int64_t wall_ns;
    int64_t sink;
};

static RunResult run_kernel(
        aclrtStream stream, void* device_buffer, void* device_output,
        int64_t hops) {
    ACL_CHECK(aclrtlaunch_dram_latency(
        1, stream, device_buffer, device_output, hops));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    timespec start{};
    timespec end{};
    CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &start));
    ACL_CHECK(aclrtlaunch_dram_latency(
        1, stream, device_buffer, device_output, hops));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &end));

    int64_t host_output[8] = {0};
    ACL_CHECK(aclrtMemcpy(
        host_output, sizeof(host_output), device_output, sizeof(host_output),
        ACL_MEMCPY_DEVICE_TO_HOST));
    return RunResult{
        host_output[0], elapsed_ns(start, end), host_output[1]};
}

static int32_t expected_sink(
        const std::vector<int32_t>& values, int64_t hops) {
    int32_t index = 0;
    for (int64_t hop = 0; hop < hops + 64; ++hop) {
        if (index < 0 || static_cast<size_t>(index) >= values.size()) {
            std::fprintf(stderr, "pointer chain escaped its buffer\n");
            std::exit(EXIT_FAILURE);
        }
        index = values[static_cast<size_t>(index)];
    }
    return index;
}

static void run_cell(
        aclrtStream stream, const char* working_set, size_t buffer_mb,
        size_t stride_bytes, int64_t short_hops) {
    if (stride_bytes % sizeof(int32_t) != 0) {
        std::fprintf(stderr, "stride_bytes must be a multiple of %zu\n",
                     sizeof(int32_t));
        std::exit(EXIT_FAILURE);
    }
    const size_t buffer_bytes = buffer_mb * 1024 * 1024;
    const size_t elements = buffer_bytes / sizeof(int32_t);
    const size_t stride_elements = stride_bytes / sizeof(int32_t);
    const size_t nodes = elements / stride_elements;
    if (nodes < 2 || (nodes - 1) * stride_elements > INT32_MAX) {
        std::fprintf(stderr, "pointer-chain geometry is outside int32 range\n");
        std::exit(EXIT_FAILURE);
    }

    std::vector<int32_t> order(nodes);
    for (size_t node = 0; node < nodes; ++node) {
        order[node] = static_cast<int32_t>(node);
    }
    const uint64_t seed = 0x910B0000ULL + buffer_mb * 131ULL + stride_bytes;
    std::mt19937_64 random(seed);
    std::shuffle(order.begin(), order.end(), random);
    std::vector<int32_t> host_buffer(elements, 0);
    for (size_t node = 0; node < nodes; ++node) {
        const size_t next = (node + 1) % nodes;
        host_buffer[static_cast<size_t>(order[node]) * stride_elements] =
            order[next] * static_cast<int32_t>(stride_elements);
    }

    void* device_buffer = nullptr;
    void* device_output = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_buffer, buffer_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_output, 8 * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(
        device_buffer, buffer_bytes, host_buffer.data(), buffer_bytes,
        ACL_MEMCPY_HOST_TO_DEVICE));

    const int64_t long_hops = short_hops * 4;
    const RunResult short_run = run_kernel(
        stream, device_buffer, device_output, short_hops);
    const RunResult long_run = run_kernel(
        stream, device_buffer, device_output, long_hops);
    const int32_t expected_short = expected_sink(host_buffer, short_hops);
    const int32_t expected_long = expected_sink(host_buffer, long_hops);
    const bool valid = (
        short_run.cycles > 0 && long_run.cycles > short_run.cycles
        && short_run.wall_ns > 0 && long_run.wall_ns > short_run.wall_ns
        && short_run.sink == expected_short && long_run.sink == expected_long);
    if (!valid) {
        std::fprintf(stderr, "measurement or sink validation failed for %s\n",
                     working_set);
        std::exit(EXIT_FAILURE);
    }

    const double delta_cycles = static_cast<double>(
        long_run.cycles - short_run.cycles);
    const double delta_wall_ns = static_cast<double>(
        long_run.wall_ns - short_run.wall_ns);
    const double extra_hops = static_cast<double>(long_hops - short_hops);
    const double tick_ns = delta_wall_ns / delta_cycles;
    const double latency_cycles = delta_cycles / extra_hops;
    const double latency_ns = delta_wall_ns / extra_hops;
    std::printf(
        "%s,%zu,%zu,%zu,%zu,%lld,%lld,%lld,%lld,%lld,%lld,"
        "%.12f,%.12f,%.12f,%lld,%lld,pass\n",
        working_set, buffer_mb, buffer_bytes, stride_bytes, nodes,
        static_cast<long long>(short_hops),
        static_cast<long long>(long_hops),
        static_cast<long long>(short_run.cycles),
        static_cast<long long>(short_run.wall_ns),
        static_cast<long long>(long_run.cycles),
        static_cast<long long>(long_run.wall_ns),
        tick_ns, latency_cycles, latency_ns,
        static_cast<long long>(short_run.sink),
        static_cast<long long>(long_run.sink));

    ACL_CHECK(aclrtFree(device_buffer));
    ACL_CHECK(aclrtFree(device_output));
}

int main(int argc, char** argv) {
    if (argc < 6 || std::string(argv[1]) != "--batch"
            || (argc - 2) % 4 != 0) {
        std::fprintf(
            stderr,
            "usage: %s --batch [working_set buffer_mb stride_bytes hops]...\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    for (int index = 2; index < argc; index += 4) {
        const char* working_set = argv[index];
        if (std::string(working_set) != "l2"
                && std::string(working_set) != "dram") {
            std::fprintf(stderr, "working_set must be l2 or dram\n");
            return EXIT_FAILURE;
        }
        const size_t buffer_mb = static_cast<size_t>(
            parse_positive_integer(argv[index + 1], "buffer_mb"));
        const size_t stride_bytes = static_cast<size_t>(
            parse_positive_integer(argv[index + 2], "stride_bytes"));
        const int64_t hops = parse_positive_integer(argv[index + 3], "hops");
        run_cell(stream, working_set, buffer_mb, stride_bytes, hops);
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
