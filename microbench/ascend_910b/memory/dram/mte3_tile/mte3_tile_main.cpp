// Host driver for the independent MTE3 UB-to-GM mechanism benchmark.
#include <acl/acl.h>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",   \
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

extern "C" uint32_t aclrtlaunch_mte3_tile(
    uint32_t blockDim, void* stream, void* gm_destination, void* out,
    int64_t tile_bytes, int64_t destination_tiles,
    int64_t iterations, int64_t mode, int64_t source_buffers,
    int64_t barrier_window);

static int64_t parse_integer(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        std::fprintf(stderr, "%s must be an integer, got %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

__attribute__((constructor(101))) static void early_init() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

struct Cell {
    int64_t tile_bytes;
    int64_t mode;
    int64_t source_buffers;
    int64_t barrier_window;
    int64_t iterations;
    int64_t destination_tiles;
    size_t footprint_bytes;
};

int main(int argc, char** argv) {
    if (argc < 8 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 3) % 5 != 0) {
        std::fprintf(
            stderr,
            "usage: %s --batch footprint_MB tile_bytes mode "
            "source_buffers barrier_window iterations "
            "[tile_bytes mode source_buffers barrier_window iterations ...]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t footprint_mb = parse_integer(argv[2], "footprint_MB");
    if (footprint_mb < 384 || footprint_mb > 4096) {
        std::fprintf(stderr, "footprint_MB must be in [384,4096]\n");
        return EXIT_FAILURE;
    }
    const int cell_count = (argc - 3) / 5;
    if (cell_count > 64) {
        std::fprintf(stderr, "a batch cannot contain more than 64 cells\n");
        return EXIT_FAILURE;
    }
    const size_t requested_bytes =
        static_cast<size_t>(footprint_mb) * 1024 * 1024;
    std::vector<Cell> cells;
    cells.reserve(static_cast<size_t>(cell_count));
    for (int argument = 3; argument < argc; argument += 5) {
        const int64_t tile_bytes = parse_integer(
            argv[argument], "tile_bytes");
        const int64_t mode = parse_integer(argv[argument + 1], "mode");
        const int64_t source_buffers = parse_integer(
            argv[argument + 2], "source_buffers");
        const int64_t barrier_window = parse_integer(
            argv[argument + 3], "barrier_window");
        const int64_t iterations = parse_integer(
            argv[argument + 4], "iterations");
        if (tile_bytes <= 0 || tile_bytes > 32 * 1024
                || tile_bytes % 32 != 0) {
            std::fprintf(
                stderr,
                "tile_bytes must be a multiple of 32 in [32,32768]\n");
            return EXIT_FAILURE;
        }
        if (mode != 0 && mode != 1 && mode != 2) {
            std::fprintf(
                stderr,
                "mode must be 0 (completion), 1 (pipeline), or 2 "
                "(dynamic params completion)\n");
            return EXIT_FAILURE;
        }
        if (source_buffers != 1) {
            std::fprintf(stderr, "source_buffers must be exactly 1\n");
            return EXIT_FAILURE;
        }
        const bool valid_completion_event = (
            mode == 0 && barrier_window == 1);
        const bool valid_pipeline_window = mode == 1 && barrier_window == 0;
        const bool valid_dynamic_completion = (
            mode == 2 && barrier_window == 1);
        if (!valid_completion_event && !valid_pipeline_window
                && !valid_dynamic_completion) {
            std::fprintf(
                stderr,
                "completion modes require barrier_window=1; pipeline "
                "requires 0\n");
            return EXIT_FAILURE;
        }
        if (iterations <= 0
                || iterations > std::numeric_limits<int64_t>::max() / 4
                || iterations % source_buffers != 0
                || (mode == 0 && iterations % barrier_window != 0)) {
            std::fprintf(
                stderr,
                "iterations must be positive, divisible by source_buffers "
                "and bounded barrier_window, and safe for 4x\n");
            return EXIT_FAILURE;
        }
        const int64_t destination_tiles = static_cast<int64_t>(
            requested_bytes / static_cast<size_t>(tile_bytes));
        if (destination_tiles <= 0 || iterations < destination_tiles) {
            std::fprintf(
                stderr,
                "iterations=%ld must cover destination_tiles=%ld "
                "for tile=%ld\n",
                iterations, destination_tiles, tile_bytes);
            return EXIT_FAILURE;
        }
        cells.push_back(Cell{
            tile_bytes,
            mode,
            source_buffers,
            barrier_window,
            iterations,
            destination_tiles,
            static_cast<size_t>(destination_tiles) * tile_bytes,
        });
    }

    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    void* device_destination = nullptr;
    void* device_output = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_destination, requested_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_output, 64, ACL_MEM_MALLOC_HUGE_FIRST));
    constexpr uint16_t kFp16One = 0x3c00;

    auto run = [&](const Cell& cell, int64_t iterations,
                   int64_t& cycles, int64_t& wall_ns) {
        const int64_t validation_tile =
            (iterations - 1) % cell.destination_tiles;
        auto* validation_address =
            static_cast<uint8_t*>(device_destination)
            + validation_tile * cell.tile_bytes;
        int64_t sentinel = -1;
        ACL_CHECK(aclrtMemset(
            validation_address, sizeof(uint16_t), 0, sizeof(uint16_t)));
        ACL_CHECK(aclrtMemcpy(
            device_output, 64, &sentinel, sizeof(sentinel),
            ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtlaunch_mte3_tile(
            1, stream, device_destination, device_output,
            cell.tile_bytes, cell.destination_tiles, iterations,
            cell.mode, cell.source_buffers, cell.barrier_window));
        ACL_CHECK(aclrtSynchronizeStream(stream));

        ACL_CHECK(aclrtMemset(
            validation_address, sizeof(uint16_t), 0, sizeof(uint16_t)));
        ACL_CHECK(aclrtMemcpy(
            device_output, 64, &sentinel, sizeof(sentinel),
            ACL_MEMCPY_HOST_TO_DEVICE));
        struct timespec start;
        struct timespec end;
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &start));
        ACL_CHECK(aclrtlaunch_mte3_tile(
            1, stream, device_destination, device_output,
            cell.tile_bytes, cell.destination_tiles, iterations,
            cell.mode, cell.source_buffers, cell.barrier_window));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &end));
        wall_ns = static_cast<int64_t>(end.tv_sec - start.tv_sec)
            * 1000000000LL + static_cast<int64_t>(end.tv_nsec - start.tv_nsec);

        int64_t host_output[8] = {0};
        uint16_t validation_value = 0;
        ACL_CHECK(aclrtMemcpy(
            host_output, 64, device_output, 64,
            ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(
            &validation_value, sizeof(validation_value), validation_address,
            sizeof(validation_value),
            ACL_MEMCPY_DEVICE_TO_HOST));
        if (host_output[0] <= 0) {
            std::fprintf(
                stderr,
                "MTE3 kernel produced invalid cycle count: %ld\n",
                host_output[0]);
            std::exit(EXIT_FAILURE);
        }
        if (validation_value != kFp16One) {
            std::fprintf(
                stderr,
                "MTE3 data validation failed: got 0x%04x expected 0x%04x\n",
                validation_value, kFp16One);
            std::exit(EXIT_FAILURE);
        }
        cycles = host_output[0];
    };

    for (const Cell& cell : cells) {
        int64_t short_cycles = 0;
        int64_t short_wall_ns = 0;
        int64_t long_cycles = 0;
        int64_t long_wall_ns = 0;
        run(cell, cell.iterations, short_cycles, short_wall_ns);
        run(cell, cell.iterations * 4, long_cycles, long_wall_ns);
        if (long_cycles <= short_cycles || long_wall_ns <= short_wall_ns) {
            std::fprintf(
                stderr,
                "MTE3 calibration must increase: cycles %ld -> %ld, "
                "wall_ns %ld -> %ld\n",
                short_cycles, long_cycles, short_wall_ns, long_wall_ns);
            return EXIT_FAILURE;
        }
        const double delta_cycles = static_cast<double>(
            long_cycles - short_cycles);
        const double tick_ns = static_cast<double>(
            long_wall_ns - short_wall_ns) / delta_cycles;
        const double per_tile_cycles = delta_cycles
            / static_cast<double>(cell.iterations * 3);
        const double per_tile_ns = per_tile_cycles * tick_ns;
        const double bandwidth_gbs = cell.tile_bytes / per_tile_ns;
        if (!std::isfinite(tick_ns) || tick_ns <= 0.0
                || !std::isfinite(per_tile_cycles)
                || per_tile_cycles <= 0.0
                || !std::isfinite(per_tile_ns) || per_tile_ns <= 0.0
                || !std::isfinite(bandwidth_gbs)
                || bandwidth_gbs <= 0.0) {
            std::fprintf(
                stderr,
                "MTE3 derived measurement is not positive and finite\n");
            return EXIT_FAILURE;
        }
        const char* mode_name = (
            cell.mode == 0 ? "hard_event_completion"
            : cell.mode == 1 ? "pipeline"
            : "dynamic_params_completion");
        std::printf(
            "%ld,%zu,%s,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%.9f,%.9f,%.9f,%.9f,pass\n",
            cell.tile_bytes, cell.footprint_bytes, mode_name,
            cell.source_buffers, cell.barrier_window, cell.iterations,
            short_cycles, short_wall_ns, long_cycles, long_wall_ns,
            tick_ns, per_tile_cycles, per_tile_ns, bandwidth_gbs);
        std::fflush(stdout);
    }

    ACL_CHECK(aclrtFree(device_destination));
    ACL_CHECK(aclrtFree(device_output));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
