#include <acl/acl.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define ACL_CHECK(call) do {                                               \
    const int status = static_cast<int>(call);                             \
    if (status != 0) {                                                     \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\\n",    \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

extern "C" uint32_t aclrtlaunch_add_rmsnorm_key31_descriptor_counter_control(
    uint32_t blockDim, void* stream, void* source, void* sink, void* status,
    uint32_t mode, uint32_t repeats);

namespace {

constexpr int64_t kMaximumWorkers = 48;
constexpr size_t kBf16Elements = 8192;
constexpr size_t kRstdElements = 64;
constexpr size_t kStatusStride = 8;

int64_t Parse(const char* text, const char* name, int64_t maximum) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value <= 0
            || value > maximum) {
        std::fprintf(stderr, "%s is out of range\\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7 || std::string(argv[1]) != "--mode"
            || std::string(argv[3]) != "--block-dim"
            || std::string(argv[5]) != "--repeats") {
        std::fprintf(stderr, "usage: %s --mode 1..5 --block-dim N --repeats N\\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t mode = Parse(argv[2], "mode", 5) - 1;
    const int64_t block_dim = Parse(argv[4], "block-dim", kMaximumWorkers);
    const int64_t repeats = Parse(argv[6], "repeats", 4096);
    void* source = nullptr;
    void* sink = nullptr;
    void* status_buffer = nullptr;
    aclrtStream stream = nullptr;
    const size_t source_bytes = block_dim * kBf16Elements * sizeof(uint16_t);
    const size_t sink_bytes = block_dim * kRstdElements * sizeof(float);
    const size_t status_bytes = block_dim * kStatusStride * sizeof(uint64_t);
    std::vector<uint16_t> host_source(source_bytes / sizeof(uint16_t), 0x3f80);
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    ACL_CHECK(aclrtMalloc(&source, source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&sink, sink_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &status_buffer, status_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(source, source_bytes, host_source.data(), source_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtCreateStream(&stream));
    ACL_CHECK(aclrtlaunch_add_rmsnorm_key31_descriptor_counter_control(
        static_cast<uint32_t>(block_dim), stream, source, sink, status_buffer,
        static_cast<uint32_t>(mode), static_cast<uint32_t>(repeats)));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::vector<uint64_t> host_status(status_bytes / sizeof(uint64_t), 0);
    ACL_CHECK(aclrtMemcpy(host_status.data(), status_bytes, status_buffer, status_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    for (int64_t worker = 0; worker < block_dim; ++worker) {
        const size_t offset = worker * kStatusStride;
        if (host_status[offset] != static_cast<uint64_t>(repeats)
                || host_status[offset + 1] != static_cast<uint64_t>(mode)
                || host_status[offset + 2] != static_cast<uint64_t>(repeats)
                || host_status[offset + 3] != static_cast<uint64_t>(worker)) {
            std::fprintf(stderr, "validation failed worker=%lld\\n",
                         static_cast<long long>(worker));
            return EXIT_FAILURE;
        }
    }
    std::printf("mode=%lld,block_dim=%lld,repeats=%lld,validation=pass\\n",
                static_cast<long long>(mode), static_cast<long long>(block_dim),
                static_cast<long long>(repeats));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(status_buffer));
    ACL_CHECK(aclrtFree(sink));
    ACL_CHECK(aclrtFree(source));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
