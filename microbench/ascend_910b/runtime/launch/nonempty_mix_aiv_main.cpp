// Launch the minimal non-empty 0 AIC : 1 AIV task at a selected AIV width.
#include <acl/acl.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>


#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",     \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)


extern "C" uint32_t aclrtlaunch_nonempty_mix_aiv(
    uint32_t block_dim, void* stream, void* validation);


namespace {

constexpr int64_t kMaximumAivBlocks = 48;
constexpr int64_t kValuesPerCacheLine = 8;
constexpr size_t kValidationBytes = static_cast<size_t>(
    kMaximumAivBlocks * kValuesPerCacheLine * sizeof(int64_t));

int64_t ParseNonnegativeInteger(const char* text, const char* name) {
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

int64_t ParsePositiveInteger(const char* text, const char* name) {
    const int64_t value = ParseNonnegativeInteger(text, name);
    if (value == 0) {
        std::fprintf(stderr, "%s must be positive, got 0\n", name);
        std::exit(EXIT_FAILURE);
    }
    return value;
}

int DeviceId() {
    const char* text = std::getenv("ASCEND_DEVICE_ID");
    if (text == nullptr) {
        std::fprintf(
            stderr,
            "ASCEND_DEVICE_ID must explicitly select an idle NPU\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(ParseNonnegativeInteger(text, "ASCEND_DEVICE_ID"));
}

void ValidateOutput(void* device_validation, int64_t block_dim) {
    std::vector<int64_t> host_validation(
        kMaximumAivBlocks * kValuesPerCacheLine, 0);
    ACL_CHECK(aclrtMemcpy(
        host_validation.data(), kValidationBytes,
        device_validation, kValidationBytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    for (int64_t block = 0; block < block_dim; ++block) {
        const int64_t actual = host_validation[block * kValuesPerCacheLine];
        if (actual != block + 1) {
            std::fprintf(
                stderr,
                "validation failed for block %ld: expected %ld, got %ld\n",
                static_cast<long>(block), static_cast<long>(block + 1),
                static_cast<long>(actual));
            std::exit(EXIT_FAILURE);
        }
    }
}

void MeasureBlockDim(
        aclrtStream stream, void* device_validation,
        int64_t block_dim, int64_t launches) {
    ACL_CHECK(aclrtMemset(device_validation, kValidationBytes, 0,
                         kValidationBytes));
    aclrtEvent start;
    aclrtEvent end;
    ACL_CHECK(aclrtCreateEventWithFlag(&start, ACL_EVENT_TIME_LINE));
    ACL_CHECK(aclrtCreateEventWithFlag(&end, ACL_EVENT_TIME_LINE));
    ACL_CHECK(aclrtRecordEvent(start, stream));
    for (int64_t index = 0; index < launches; ++index) {
        ACL_CHECK(aclrtlaunch_nonempty_mix_aiv(
            static_cast<uint32_t>(block_dim), stream, device_validation));
    }
    ACL_CHECK(aclrtRecordEvent(end, stream));
    ACL_CHECK(aclrtSynchronizeEvent(end));
    float elapsed_ms = 0.0f;
    ACL_CHECK(aclrtEventElapsedTime(&elapsed_ms, start, end));
    if (!(elapsed_ms > 0.0f)) {
        std::fprintf(stderr, "CANN event health check was not positive\n");
        std::exit(EXIT_FAILURE);
    }
    ValidateOutput(device_validation, block_dim);
    std::printf(
        "health_check_only=true task_class=mix_aiv_nonempty block_dim=%ld "
        "mix_block_dim=0 launches=%ld elapsed_ms=%.6f\n",
        static_cast<long>(block_dim), static_cast<long>(launches), elapsed_ms);
    ACL_CHECK(aclrtDestroyEvent(end));
    ACL_CHECK(aclrtDestroyEvent(start));
}

}  // namespace


__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(DeviceId()));
}


int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(
            stderr, "usage: %s <block_dim|--sweep> [launches]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const bool sweep = std::strcmp(argv[1], "--sweep") == 0;
    const int64_t block_dim = sweep
        ? 1 : ParsePositiveInteger(argv[1], "block_dim");
    const int64_t launches = argc == 3
        ? ParsePositiveInteger(argv[2], "launches") : 1;
    if (block_dim > kMaximumAivBlocks) {
        std::fprintf(
            stderr, "block_dim cannot exceed the 48 AIV cores, got %ld\n",
            static_cast<long>(block_dim));
        return EXIT_FAILURE;
    }

    aclrtStream stream;
    void* device_validation = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    ACL_CHECK(aclrtMalloc(
        &device_validation, kValidationBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    if (sweep) {
        for (int64_t dimension = 1; dimension <= 48; ++dimension) {
            MeasureBlockDim(stream, device_validation, dimension, launches);
        }
    } else {
        MeasureBlockDim(stream, device_validation, block_dim, launches);
    }

    ACL_CHECK(aclrtFree(device_validation));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(DeviceId()));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
