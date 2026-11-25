#include <acl/acl.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define ACL_CHECK(call) do {                                             \
    int status = static_cast<int>(call);                                 \
    if (status != 0) {                                                   \
        std::fprintf(stderr, "ACL error: %s returned %d\n", #call, status); \
        std::exit(EXIT_FAILURE);                                         \
    }                                                                    \
} while (0)

extern "C" uint32_t aclrtlaunch_fia_brcb_service(
    uint32_t block_dim, void* stream, void* cycles, void* validation,
    uint32_t row_chunk_round, uint32_t iterations);

uint32_t ParsePositive(const char* text) {
    char* end = nullptr;
    errno = 0;
    unsigned long value = std::strtoul(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
            || value == 0 || value > UINT32_MAX) {
        std::fprintf(stderr, "expected positive uint32: %s\n", text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<uint32_t>(value);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s ROW_CHUNK ITERATIONS\n", argv[0]);
        return EXIT_FAILURE;
    }
    const uint32_t rows = ParsePositive(argv[1]);
    const uint32_t iterations = ParsePositive(argv[2]);
    const char* device_text = std::getenv("ASCEND_DEVICE_ID");
    if (device_text == nullptr) {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must be explicit\n");
        return EXIT_FAILURE;
    }
    if (std::strcmp(device_text, "0") == 0) {
        ACL_CHECK(aclInit(nullptr));
        ACL_CHECK(aclrtSetDevice(0));
    } else {
        std::fprintf(stderr, "this benchmark requires logical device 0\n");
        return EXIT_FAILURE;
    }
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    void* cycles_device = nullptr;
    void* validation_device = nullptr;
    ACL_CHECK(aclrtMalloc(&cycles_device, 64, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&validation_device, 64, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtlaunch_fia_brcb_service(
        1, stream, cycles_device, validation_device, rows, iterations));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    int64_t cycles = 0;
    uint32_t validation = 0;
    ACL_CHECK(aclrtMemcpy(&cycles, sizeof(cycles), cycles_device,
                         sizeof(cycles), ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(&validation, sizeof(validation), validation_device,
                         sizeof(validation), ACL_MEMCPY_DEVICE_TO_HOST));
    std::puts("row_chunk_round,iterations,system_cycles,duration_ns,validation");
    std::printf("%u,%u,%ld,%.9f,%s\n", rows, iterations,
                static_cast<long>(cycles),
                static_cast<double>(cycles) * 20.0 / iterations,
                validation == 0x3f800000U ? "pass" : "fail");
    ACL_CHECK(aclrtFree(validation_device));
    ACL_CHECK(aclrtFree(cycles_device));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return validation == 0x3f800000U ? EXIT_SUCCESS : EXIT_FAILURE;
}
