// Host driver for the independent MTE2->Vector->MTE3 AIV task-floor bench.
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
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",    \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_aiv_memory_task_floor(
    uint32_t block_dim, void* stream, void* source, void* destination,
    int64_t rounds);

namespace {

constexpr int64_t kMaximumAivBlocks = 48;
constexpr int64_t kElementsPerAiv = 32;
constexpr int64_t kMaximumRounds = 64;
constexpr uint16_t kHalfOne = 0x3c00;
constexpr size_t kBufferBytes = static_cast<size_t>(
    kMaximumAivBlocks * kMaximumRounds * kElementsPerAiv * sizeof(uint16_t));

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
    return static_cast<int>(
        ParseNonnegativeInteger(text, "ASCEND_DEVICE_ID"));
}

void ValidateOutput(
        void* device_destination, int64_t block_dim, int64_t rounds) {
    const size_t active_values = static_cast<size_t>(
        block_dim * kMaximumRounds * kElementsPerAiv);
    const size_t active_bytes = active_values * sizeof(uint16_t);
    std::vector<uint16_t> output(active_values, 0);
    ACL_CHECK(aclrtMemcpy(
        output.data(), active_bytes, device_destination, active_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    for (int64_t worker = 0; worker < block_dim; ++worker) {
        for (int64_t round = 0; round < rounds; ++round) {
            const size_t base = static_cast<size_t>(
                (worker * rounds + round) * kElementsPerAiv);
            for (int64_t element = 0; element < kElementsPerAiv; ++element) {
                const size_t index = base + static_cast<size_t>(element);
                if (output[index] == kHalfOne) {
                    continue;
                }
                std::fprintf(
                    stderr,
                    "validation failed at element %zu: expected 0x%04x, "
                    "got 0x%04x\n",
                    index, kHalfOne, output[index]);
                std::exit(EXIT_FAILURE);
            }
        }
    }
}

void MeasureBlockDim(
        aclrtStream stream, void* device_source, void* device_destination,
        int64_t block_dim, int64_t launches, int64_t rounds) {
    ACL_CHECK(aclrtMemset(
        device_destination, kBufferBytes, 0, kBufferBytes));
    aclrtEvent start;
    aclrtEvent end;
    ACL_CHECK(aclrtCreateEventWithFlag(&start, ACL_EVENT_TIME_LINE));
    ACL_CHECK(aclrtCreateEventWithFlag(&end, ACL_EVENT_TIME_LINE));
    ACL_CHECK(aclrtRecordEvent(start, stream));
    for (int64_t index = 0; index < launches; ++index) {
        ACL_CHECK(aclrtlaunch_aiv_memory_task_floor(
            static_cast<uint32_t>(block_dim), stream,
            device_source, device_destination, rounds));
    }
    ACL_CHECK(aclrtRecordEvent(end, stream));
    ACL_CHECK(aclrtSynchronizeEvent(end));
    float elapsed_ms = 0.0f;
    ACL_CHECK(aclrtEventElapsedTime(&elapsed_ms, start, end));
    if (!(elapsed_ms > 0.0f)) {
        std::fprintf(stderr, "CANN event duration was not positive\n");
        std::exit(EXIT_FAILURE);
    }
    ValidateOutput(device_destination, block_dim, rounds);
    if (rounds == 1) {
        std::printf(
            "health_check_only=true task_class=aiv_memory_task "
            "block_dim=%ld mix_block_dim=0 launches=%ld elapsed_ms=%.6f\n",
            static_cast<long>(block_dim),
            static_cast<long>(launches), elapsed_ms);
    } else {
        const double per_launch_ns = (
            static_cast<double>(elapsed_ms) * 1.0e6
            / static_cast<double>(launches));
        std::printf(
            "mechanism=aiv_memory_rounds block_dim=%ld rounds=%ld "
            "launches=%ld elapsed_ms=%.6f per_launch_ns=%.9f\n",
            static_cast<long>(block_dim), static_cast<long>(rounds),
            static_cast<long>(launches), elapsed_ms, per_launch_ns);
    }
    ACL_CHECK(aclrtDestroyEvent(end));
    ACL_CHECK(aclrtDestroyEvent(start));
}

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(DeviceId()));
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::fprintf(
            stderr,
            "usage: %s <block_dim|--sweep> [launches] [rounds]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const bool sweep = std::strcmp(argv[1], "--sweep") == 0;
    const int64_t block_dim = sweep
        ? 1 : ParsePositiveInteger(argv[1], "block_dim");
    const int64_t launches = argc == 3
        ? ParsePositiveInteger(argv[2], "launches")
        : (argc == 4 ? ParsePositiveInteger(argv[2], "launches") : 1000);
    const int64_t rounds = argc == 4
        ? ParsePositiveInteger(argv[3], "rounds") : 1;
    if (block_dim > kMaximumAivBlocks) {
        std::fprintf(
            stderr, "block_dim cannot exceed the 48 AIV cores, got %ld\n",
            static_cast<long>(block_dim));
        return EXIT_FAILURE;
    }
    if (rounds > kMaximumRounds) {
        std::fprintf(
            stderr, "rounds cannot exceed %ld, got %ld\n",
            static_cast<long>(kMaximumRounds), static_cast<long>(rounds));
        return EXIT_FAILURE;
    }

    aclrtStream stream;
    void* device_source = nullptr;
    void* device_destination = nullptr;
    std::vector<uint16_t> host_source(
        kMaximumAivBlocks * kMaximumRounds * kElementsPerAiv, kHalfOne);
    ACL_CHECK(aclrtCreateStream(&stream));
    ACL_CHECK(aclrtMalloc(
        &device_source, kBufferBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_destination, kBufferBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(
        device_source, kBufferBytes, host_source.data(), kBufferBytes,
        ACL_MEMCPY_HOST_TO_DEVICE));

    if (sweep) {
        for (int64_t dimension = 1;
             dimension <= kMaximumAivBlocks; ++dimension) {
            MeasureBlockDim(
                stream, device_source, device_destination,
                dimension, launches, rounds);
        }
    } else {
        MeasureBlockDim(
            stream, device_source, device_destination,
            block_dim, launches, rounds);
    }

    ACL_CHECK(aclrtFree(device_destination));
    ACL_CHECK(aclrtFree(device_source));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(DeviceId()));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
