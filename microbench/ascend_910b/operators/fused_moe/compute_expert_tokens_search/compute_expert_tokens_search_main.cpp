// Host driver for the independent key-1003 local-search mechanism probe.
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
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",   \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_compute_expert_tokens_search(
    uint32_t block_dim, void* stream, void* gm_input, void* gm_cycles,
    void* gm_validation, int64_t input_elements, int64_t target,
    int64_t iterations);

namespace {

constexpr int32_t kLogicalDeviceId = 0;
constexpr int64_t kMaximumInputElements = 320;

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

int SearchSteps(const std::vector<int32_t>& values, int32_t target) {
    int low = 0;
    int high = static_cast<int>(values.size()) - 1;
    int steps = 0;
    while (low <= high) {
        ++steps;
        const int middle = (low + high) / 2;
        if (values[static_cast<size_t>(middle)] > target) {
            high = middle - 1;
        } else {
            low = middle + 1;
        }
    }
    return steps;
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
    if (argc < 5 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 2) % 3 != 0) {
        std::fprintf(
            stderr,
            "usage: %s --batch input_elements target iterations "
            "[input_elements target iterations ...]\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    void* device_input = nullptr;
    void* device_cycles = nullptr;
    void* device_validation = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_input, kMaximumInputElements * sizeof(int32_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_cycles, sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_validation, 3 * sizeof(int32_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    std::vector<int32_t> values(kMaximumInputElements);
    for (int64_t index = 0; index < kMaximumInputElements; ++index) {
        values[static_cast<size_t>(index)] = static_cast<int32_t>(index);
    }
    ACL_CHECK(aclrtMemcpy(
        device_input, kMaximumInputElements * sizeof(int32_t), values.data(),
        kMaximumInputElements * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE));

    std::printf(
        "input_elements,target,search_steps,iterations,system_cycles,"
        "validation\n");
    for (int argument = 2; argument < argc; argument += 3) {
        const int64_t input_elements = ParseInteger(
            argv[argument], "input_elements");
        const int64_t target = ParseInteger(argv[argument + 1], "target");
        const int64_t iterations = ParseInteger(
            argv[argument + 2], "iterations");
        if (input_elements <= 0 || input_elements > kMaximumInputElements
                || iterations <= 0 || target < -1
                || target > input_elements) {
            std::fprintf(stderr, "cell is outside the frozen search domain\n");
            return EXIT_FAILURE;
        }

        ACL_CHECK(aclrtlaunch_compute_expert_tokens_search(
            1, stream, device_input, device_cycles, device_validation,
            input_elements, target, 16));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        ACL_CHECK(aclrtlaunch_compute_expert_tokens_search(
            1, stream, device_input, device_cycles, device_validation,
            input_elements, target, iterations));
        ACL_CHECK(aclrtSynchronizeStream(stream));

        int64_t cycles = 0;
        int32_t validation[3] = {-1, -1, -1};
        ACL_CHECK(aclrtMemcpy(
            &cycles, sizeof(cycles), device_cycles, sizeof(cycles),
            ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(
            validation, sizeof(validation), device_validation,
            sizeof(validation), ACL_MEMCPY_DEVICE_TO_HOST));
        const bool valid = cycles > 0 && validation[0] >= 0
            && validation[0] < input_elements
            && validation[1] == validation[2];
        std::printf(
            "%ld,%ld,%d,%ld,%ld,%s\n",
            static_cast<long>(input_elements), static_cast<long>(target),
            SearchSteps(
                std::vector<int32_t>(values.begin(),
                                     values.begin() + input_elements),
                static_cast<int32_t>(target)),
            static_cast<long>(iterations), static_cast<long>(cycles),
            valid ? "pass" : "fail");
        if (!valid) {
            return EXIT_FAILURE;
        }
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(device_validation));
    ACL_CHECK(aclrtFree(device_cycles));
    ACL_CHECK(aclrtFree(device_input));
    ACL_CHECK(aclrtResetDevice(kLogicalDeviceId));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
