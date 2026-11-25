// Host driver for the independent key-1003 ProcessAfter mechanism probe.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",   \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_compute_expert_tokens_reduce(
    uint32_t block_dim, void* stream, void* gm_workspace, void* gm_output,
    void* gm_cycles, int64_t experts, int64_t used_before);

namespace {

constexpr int32_t kLogicalDeviceId = 0;
constexpr int64_t kPhysicalAivCores = 48;

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

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    const char* device_text = std::getenv("ASCEND_DEVICE_ID");
    if (device_text == nullptr || *device_text == '\0'
            || ParseInteger(device_text, "ASCEND_DEVICE_ID")
                != kLogicalDeviceId) {
        std::fprintf(
            stderr,
            "ASCEND_DEVICE_ID must be logical device 0 after remapping\n");
        std::exit(EXIT_FAILURE);
    }
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(kLogicalDeviceId));
}

int main(int argc, char** argv) {
    if (argc != 7 || std::strcmp(argv[1], "--experts") != 0
            || std::strcmp(argv[3], "--used-before") != 0
            || std::strcmp(argv[5], "--samples") != 0) {
        std::fprintf(
            stderr,
            "usage: %s --experts E --used-before U --samples N\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t experts = ParseInteger(argv[2], "experts");
    const int64_t used_before = ParseInteger(argv[4], "used-before");
    const int64_t samples = ParseInteger(argv[6], "samples");
    if (experts <= 0 || experts > 2048 || used_before <= 0
            || used_before > kPhysicalAivCores || samples <= 0) {
        std::fprintf(stderr, "arguments are outside the frozen source domain\n");
        return EXIT_FAILURE;
    }

    const int64_t normal_rows =
        (experts + kPhysicalAivCores - 1) / kPhysicalAivCores;
    const int64_t used_after =
        (experts + normal_rows - 1) / normal_rows;
    std::vector<int32_t> workspace(
        static_cast<size_t>(experts * used_before));
    for (int64_t row = 0; row < experts; ++row) {
        for (int64_t column = 0; column < used_before; ++column) {
            workspace[static_cast<size_t>(row * used_before + column)] =
                static_cast<int32_t>(column);
        }
    }
    std::vector<int32_t> output(static_cast<size_t>(experts), -1);
    std::vector<int64_t> cycles(
        static_cast<size_t>(kPhysicalAivCores), 0);

    void* device_workspace = nullptr;
    void* device_output = nullptr;
    void* device_cycles = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_workspace, workspace.size() * sizeof(int32_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_output, output.size() * sizeof(int32_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_cycles, cycles.size() * sizeof(int64_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(
        device_workspace, workspace.size() * sizeof(int32_t),
        workspace.data(), workspace.size() * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    for (int warmup = 0; warmup < 10; ++warmup) {
        ACL_CHECK(aclrtlaunch_compute_expert_tokens_reduce(
            kPhysicalAivCores, stream, device_workspace, device_output,
            device_cycles, experts, used_before));
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::printf(
        "experts,used_before,used_after,normal_rows,sample,max_system_cycles,"
        "validation\n");
    for (int64_t sample = 0; sample < samples; ++sample) {
        ACL_CHECK(aclrtMemset(
            device_cycles, cycles.size() * sizeof(int64_t), 0,
            cycles.size() * sizeof(int64_t)));
        ACL_CHECK(aclrtlaunch_compute_expert_tokens_reduce(
            kPhysicalAivCores, stream, device_workspace, device_output,
            device_cycles, experts, used_before));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        ACL_CHECK(aclrtMemcpy(
            cycles.data(), cycles.size() * sizeof(int64_t), device_cycles,
            cycles.size() * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(
            output.data(), output.size() * sizeof(int32_t), device_output,
            output.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));
        const int64_t max_cycles = *std::max_element(
            cycles.begin(), cycles.begin() + used_after);
        const bool valid = max_cycles > 0 && std::all_of(
            output.begin(), output.end(), [used_before](int32_t value) {
                return value == used_before - 1;
            });
        std::printf(
            "%ld,%ld,%ld,%ld,%ld,%ld,%s\n",
            static_cast<long>(experts), static_cast<long>(used_before),
            static_cast<long>(used_after), static_cast<long>(normal_rows),
            static_cast<long>(sample), static_cast<long>(max_cycles),
            valid ? "pass" : "fail");
        if (!valid) {
            return EXIT_FAILURE;
        }
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(device_cycles));
    ACL_CHECK(aclrtFree(device_output));
    ACL_CHECK(aclrtFree(device_workspace));
    ACL_CHECK(aclrtResetDevice(kLogicalDeviceId));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
