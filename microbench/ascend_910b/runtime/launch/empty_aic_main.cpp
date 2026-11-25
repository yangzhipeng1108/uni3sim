// Launch a truly empty AIC kernel at one or every legal block dimension.
#include <acl/acl.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>


#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",    \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)


extern "C" uint32_t aclrtlaunch_empty_aic(
    uint32_t block_dim, void* stream);


static int64_t parse_nonnegative_integer(
        const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value < 0) {
        std::fprintf(
            stderr, "%s must be a non-negative integer, got %s\n",
            name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}


static int64_t parse_positive_integer(const char* text, const char* name) {
    const int64_t value = parse_nonnegative_integer(text, name);
    if (value == 0) {
        std::fprintf(stderr, "%s must be positive, got 0\n", name);
        std::exit(EXIT_FAILURE);
    }
    return value;
}


static int device_id() {
    const char* text = std::getenv("ASCEND_DEVICE_ID");
    if (text == nullptr) {
        std::fprintf(
            stderr,
            "ASCEND_DEVICE_ID must explicitly select an idle NPU\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(
        parse_nonnegative_integer(text, "ASCEND_DEVICE_ID"));
}


__attribute__((constructor(101))) static void early_init() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(device_id()));
}


static void measure_block_dim(
        aclrtStream stream, int64_t block_dim, int64_t launches) {
    aclrtEvent start;
    aclrtEvent end;
    ACL_CHECK(aclrtCreateEventWithFlag(&start, ACL_EVENT_TIME_LINE));
    ACL_CHECK(aclrtCreateEventWithFlag(&end, ACL_EVENT_TIME_LINE));
    ACL_CHECK(aclrtRecordEvent(start, stream));
    for (int64_t index = 0; index < launches; ++index) {
        ACL_CHECK(aclrtlaunch_empty_aic(
            static_cast<uint32_t>(block_dim), stream));
    }
    ACL_CHECK(aclrtRecordEvent(end, stream));
    ACL_CHECK(aclrtSynchronizeEvent(end));
    float elapsed_ms = 0.0f;
    ACL_CHECK(aclrtEventElapsedTime(&elapsed_ms, start, end));
    if (!(elapsed_ms > 0.0f)) {
        std::fprintf(stderr, "CANN event health check was not positive\n");
        std::exit(EXIT_FAILURE);
    }
    std::printf(
        "health_check_only=true task_class=aic_only block_dim=%ld "
        "mix_block_dim=0 launches=%ld elapsed_ms=%.6f\n",
        static_cast<long>(block_dim), static_cast<long>(launches),
        elapsed_ms);
    ACL_CHECK(aclrtDestroyEvent(end));
    ACL_CHECK(aclrtDestroyEvent(start));
}


int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(
            stderr, "usage: %s <block_dim|--sweep> [launches]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const bool sweep = std::strcmp(argv[1], "--sweep") == 0;
    const int64_t block_dim = sweep
        ? 1 : parse_positive_integer(argv[1], "block_dim");
    const int64_t launches = argc == 3
        ? parse_positive_integer(argv[2], "launches") : 1;
    if (block_dim > 24) {
        std::fprintf(
            stderr, "block_dim cannot exceed the 24 AIC cores, got %ld\n",
            static_cast<long>(block_dim));
        return EXIT_FAILURE;
    }

    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    if (sweep) {
        for (int64_t dimension = 1; dimension <= 24; ++dimension) {
            measure_block_dim(stream, dimension, launches);
        }
    } else {
        measure_block_dim(stream, block_dim, launches);
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(device_id()));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
