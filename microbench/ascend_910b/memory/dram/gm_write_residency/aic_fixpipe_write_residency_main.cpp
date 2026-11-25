// ACL host for the independent AIC Fixpipe GM write-back residency probe.
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

extern "C" uint32_t aclrtlaunch_aic_fixpipe_write_residency(
    uint32_t blockDim, void* stream, void* gm_a, void* gm_b,
    void* destination, int64_t slots, int64_t passes);

namespace {

constexpr int64_t kM = 128;
constexpr int64_t kN = 128;
constexpr int64_t kK = 16;
constexpr int64_t kTileElements = kM * kN;

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
    int64_t cores;
    int64_t slots;
    int64_t passes;
};

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc < 5 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 2) % 3 != 0) {
        std::fprintf(
            stderr, "usage: %s --batch cores slots passes [cell ...]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    std::vector<Cell> cells;
    int64_t maximum_elements = 0;
    for (int argument = 2; argument < argc; argument += 3) {
        Cell cell{
            ParseInteger(argv[argument], "cores"),
            ParseInteger(argv[argument + 1], "slots"),
            ParseInteger(argv[argument + 2], "passes"),
        };
        if (cell.cores <= 0 || cell.cores > 24 || cell.slots < 0
                || cell.passes < 0) {
            std::fprintf(
                stderr, "invalid fixpipe cell at argument %d\n", argument);
            return EXIT_FAILURE;
        }
        const int64_t total =
            cell.cores * (cell.slots > 0 ? cell.slots : 1) * kTileElements;
        maximum_elements = total > maximum_elements ? total : maximum_elements;
        cells.push_back(cell);
    }

    void* gm_a = nullptr;
    void* gm_b = nullptr;
    void* destination = nullptr;
    const size_t operand_bytes = static_cast<size_t>(kM * kK) * 2;
    const size_t destination_bytes =
        static_cast<size_t>(maximum_elements) * sizeof(float);
    ACL_CHECK(aclrtMalloc(&gm_a, operand_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&gm_b, operand_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &destination, destination_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(gm_a, operand_bytes, 0, operand_bytes));
    ACL_CHECK(aclrtMemset(gm_b, operand_bytes, 0, operand_bytes));
    ACL_CHECK(aclrtMemset(destination, destination_bytes, 0,
                          destination_bytes));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    for (const Cell& cell : cells) {
        ACL_CHECK(aclrtlaunch_aic_fixpipe_write_residency(
            static_cast<uint32_t>(cell.cores), stream, gm_a, gm_b,
            destination, cell.slots, cell.passes));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        const long long tile_kib = kTileElements * sizeof(float) / 1024;
        std::printf(
            "cores=%lld slots=%lld passes=%lld unique_KiB=%lld "
            "written_KiB=%lld\n",
            static_cast<long long>(cell.cores),
            static_cast<long long>(cell.slots),
            static_cast<long long>(cell.passes),
            static_cast<long long>(cell.cores * cell.slots * tile_kib),
            static_cast<long long>(cell.cores * cell.passes * tile_kib));
        std::fflush(stdout);
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(destination));
    ACL_CHECK(aclrtFree(gm_b));
    ACL_CHECK(aclrtFree(gm_a));
    return EXIT_SUCCESS;
}
