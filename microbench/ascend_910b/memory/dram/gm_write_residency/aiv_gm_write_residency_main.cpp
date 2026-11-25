// ACL host for the independent AIV GM write-back residency probe.
#include <acl/acl.h>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",      \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_aiv_gm_write_residency(
    uint32_t blockDim, void* stream, void* destination, void* sink,
    int64_t elements_per_worker, int64_t passes, int64_t readback);

namespace {

constexpr int64_t kElementBytes = 2;
constexpr int64_t kTileElements = 8192;

int64_t ParseInteger(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        std::fprintf(stderr, "%s must be an integer, got %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

struct Cell {
    int64_t workers;
    int64_t kib_per_worker;
    int64_t passes;
    int64_t readback;
};

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc < 6 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 2) % 4 != 0) {
        std::fprintf(
            stderr,
            "usage: %s --batch workers kib_per_worker passes readback "
            "[cell ...]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    std::vector<Cell> cells;
    int64_t maximum_workers = 0;
    int64_t maximum_elements = 0;
    for (int argument = 2; argument < argc; argument += 4) {
        Cell cell{
            ParseInteger(argv[argument], "workers"),
            ParseInteger(argv[argument + 1], "kib_per_worker"),
            ParseInteger(argv[argument + 2], "passes"),
            ParseInteger(argv[argument + 3], "readback"),
        };
        const int64_t elements = cell.kib_per_worker * 1024 / kElementBytes;
        if (cell.workers <= 0 || cell.workers > 48
                || cell.kib_per_worker <= 0 || cell.passes <= 0
                || cell.readback < 0 || cell.readback > 1
                || elements % kTileElements != 0) {
            std::fprintf(
                stderr, "invalid write-residency cell at argument %d\n",
                argument);
            return EXIT_FAILURE;
        }
        maximum_workers = cell.workers > maximum_workers
            ? cell.workers : maximum_workers;
        const int64_t total = elements * cell.workers;
        maximum_elements = total > maximum_elements ? total : maximum_elements;
        cells.push_back(cell);
    }

    void* destination = nullptr;
    void* sink = nullptr;
    const size_t destination_bytes =
        static_cast<size_t>(maximum_elements) * kElementBytes;
    const size_t sink_bytes =
        static_cast<size_t>(maximum_workers * kTileElements) * kElementBytes;
    ACL_CHECK(aclrtMalloc(
        &destination, destination_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&sink, sink_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(destination, destination_bytes, 0,
                          destination_bytes));
    ACL_CHECK(aclrtMemset(sink, sink_bytes, 0, sink_bytes));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    for (const Cell& cell : cells) {
        const int64_t elements = cell.kib_per_worker * 1024 / kElementBytes;
        ACL_CHECK(aclrtlaunch_aiv_gm_write_residency(
            static_cast<uint32_t>(cell.workers), stream, destination, sink,
            elements, cell.passes, cell.readback));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::printf(
            "workers=%lld kib_per_worker=%lld passes=%lld readback=%lld "
            "written_KiB=%lld\n",
            static_cast<long long>(cell.workers),
            static_cast<long long>(cell.kib_per_worker),
            static_cast<long long>(cell.passes),
            static_cast<long long>(cell.readback),
            static_cast<long long>(
                cell.workers * cell.kib_per_worker * cell.passes));
        std::fflush(stdout);
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(sink));
    ACL_CHECK(aclrtFree(destination));
    return EXIT_SUCCESS;
}
