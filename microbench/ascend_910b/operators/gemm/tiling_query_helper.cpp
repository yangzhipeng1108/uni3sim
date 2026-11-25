// Invoke aclnnMatmul while a preload interposer suppresses its device launch.

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_matmul.h>

#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <unistd.h>

namespace {

void CheckAcl(aclError status, const char* operation) {
    if (status != ACL_SUCCESS) {
        std::fprintf(stderr, "%s failed: %d\n", operation,
                     static_cast<int>(status));
        std::exit(1);
    }
}

#define ACL_CHECK(call) CheckAcl((call), #call)

struct Resources {
    int32_t device_id = -1;
    bool acl_initialized = false;
    bool device_set = false;
    void* lhs_data = nullptr;
    void* rhs_data = nullptr;
    void* output_data = nullptr;
    void* workspace = nullptr;
    aclrtStream stream = nullptr;
    aclTensor* lhs = nullptr;
    aclTensor* rhs = nullptr;
    aclTensor* output = nullptr;
    aclOpExecutor* executor = nullptr;
    std::size_t lhs_capacity = 0;
    std::size_t rhs_capacity = 0;
    std::size_t output_capacity = 0;
    std::size_t workspace_capacity = 0;

    void ResetOperation() {
        if (executor != nullptr) {
            ACL_CHECK(aclDestroyAclOpExecutor(executor));
            executor = nullptr;
        }
        if (output != nullptr) {
            ACL_CHECK(aclDestroyTensor(output));
            output = nullptr;
        }
        if (rhs != nullptr) {
            ACL_CHECK(aclDestroyTensor(rhs));
            rhs = nullptr;
        }
        if (lhs != nullptr) {
            ACL_CHECK(aclDestroyTensor(lhs));
            lhs = nullptr;
        }
    }

    void Ensure(void** pointer, std::size_t* capacity, std::size_t required) {
        if (*capacity >= required) return;
        if (*pointer != nullptr) ACL_CHECK(aclrtFree(*pointer));
        ACL_CHECK(aclrtMalloc(pointer, required, ACL_MEM_MALLOC_HUGE_FIRST));
        *capacity = required;
    }

    ~Resources() {
        ResetOperation();
        if (stream != nullptr) {
            ACL_CHECK(aclrtDestroyStream(stream));
        }
        if (workspace != nullptr) {
            ACL_CHECK(aclrtFree(workspace));
        }
        if (output_data != nullptr) {
            ACL_CHECK(aclrtFree(output_data));
        }
        if (rhs_data != nullptr) {
            ACL_CHECK(aclrtFree(rhs_data));
        }
        if (lhs_data != nullptr) {
            ACL_CHECK(aclrtFree(lhs_data));
        }
        if (device_set) {
            ACL_CHECK(aclrtResetDevice(device_id));
        }
        if (acl_initialized) {
            ACL_CHECK(aclFinalize());
        }
    }
};

[[noreturn]] void Fail(const char* message) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

std::size_t TensorBytes(int64_t first, int64_t second);
std::size_t BatchedTensorBytes(int64_t batch, int64_t first, int64_t second);
int64_t MatrixStride(int64_t rows, int64_t columns);
void WriteExclusive(const char* path, const void* data, std::size_t size);
template <typename Function>
Function ResolvePreloadSymbol(const char* name);

void RunOperation(Resources* resources, int32_t input_rank, int64_t batch,
                  int64_t m, int64_t n, int64_t k, int32_t cube_math_type,
                  const char* capture_directory, const char* metadata_output) {
    if (input_rank != 2 && input_rank != 3) {
        Fail("input_rank must be 2 or 3");
    }
    if (input_rank == 2 && batch != 1) {
        Fail("rank-two GEMM requires B=1");
    }
    resources->ResetOperation();
    resources->Ensure(
        &resources->lhs_data, &resources->lhs_capacity,
        BatchedTensorBytes(batch, m, k));
    resources->Ensure(
        &resources->rhs_data, &resources->rhs_capacity, TensorBytes(n, k));
    resources->Ensure(
        &resources->output_data, &resources->output_capacity,
        BatchedTensorBytes(batch, m, n));

    const int64_t lhs_rank_three_dims[] = {batch, m, k};
    const int64_t lhs_rank_three_strides[] = {MatrixStride(m, k), k, 1};
    const int64_t lhs_rank_two_dims[] = {m, k};
    const int64_t lhs_rank_two_strides[] = {k, 1};
    const int64_t rhs_view_dims[] = {k, n};
    const int64_t rhs_view_strides[] = {1, k};
    const int64_t rhs_storage_dims[] = {n, k};
    const int64_t output_rank_three_dims[] = {batch, m, n};
    const int64_t output_rank_three_strides[] = {MatrixStride(m, n), n, 1};
    const int64_t output_rank_two_dims[] = {m, n};
    const int64_t output_rank_two_strides[] = {n, 1};
    const int64_t* lhs_dims = input_rank == 2 ? lhs_rank_two_dims
                                               : lhs_rank_three_dims;
    const int64_t* lhs_strides = input_rank == 2 ? lhs_rank_two_strides
                                                  : lhs_rank_three_strides;
    const int64_t* output_dims = input_rank == 2 ? output_rank_two_dims
                                                  : output_rank_three_dims;
    const int64_t* output_strides = input_rank == 2
        ? output_rank_two_strides : output_rank_three_strides;
    resources->lhs = aclCreateTensor(
        lhs_dims, input_rank, ACL_BF16, lhs_strides, 0, ACL_FORMAT_ND,
        lhs_dims, input_rank, resources->lhs_data);
    resources->rhs = aclCreateTensor(
        rhs_view_dims, 2, ACL_BF16, rhs_view_strides, 0, ACL_FORMAT_ND,
        rhs_storage_dims, 2, resources->rhs_data);
    resources->output = aclCreateTensor(
        output_dims, input_rank, ACL_BF16, output_strides, 0, ACL_FORMAT_ND,
        output_dims, input_rank, resources->output_data);
    if (resources->lhs == nullptr || resources->rhs == nullptr
        || resources->output == nullptr) {
        Fail("aclCreateTensor returned null");
    }
    uint64_t workspace_size = 0;
    ACL_CHECK(aclnnMatmulGetWorkspaceSize(
        resources->lhs, resources->rhs, resources->output,
        static_cast<int8_t>(cube_math_type), &workspace_size,
        &resources->executor));
    if (resources->executor == nullptr) {
        Fail("aclnnMatmulGetWorkspaceSize returned a null executor");
    }
    if (workspace_size > 0) {
        resources->Ensure(
            &resources->workspace, &resources->workspace_capacity,
            static_cast<std::size_t>(workspace_size));
    }
    if (capture_directory == nullptr || capture_directory[0] == '\0') {
        Fail("GEMM capture directory is required");
    }
    if (setenv("MIDLAYER_ASCEND_GEMM_DRY_LAUNCH_DIR",
               capture_directory, 1) != 0) {
        Fail("cannot set GEMM capture directory");
    }
    using BeginDryLaunch = int (*)(std::size_t);
    using EndDryLaunch = int (*)();
    using DryLaunchError = const char* (*)();
    const auto begin = ResolvePreloadSymbol<BeginDryLaunch>(
        "midlayer_ascend_gemm_dry_launch_begin");
    const auto end = ResolvePreloadSymbol<EndDryLaunch>(
        "midlayer_ascend_gemm_dry_launch_end");
    const auto dry_launch_error = ResolvePreloadSymbol<DryLaunchError>(
        "midlayer_ascend_gemm_dry_launch_error");
    if (begin(1) != 0) {
        std::fprintf(stderr, "cannot begin dry launch: %s\n", dry_launch_error());
        std::exit(1);
    }
    ACL_CHECK(aclnnMatmul(
        resources->workspace, workspace_size, resources->executor,
        resources->stream));
    if (end() != 0) {
        std::fprintf(stderr, "dry launch failed: %s\n", dry_launch_error());
        std::exit(1);
    }
    // CANN 8.5.1 acl_meta.h exposes repeatable opt-in and explicit destroy.
    // This helper does not opt in: aclnnMatmul routes through
    // CommonOpExecutorRun, whose successful non-repeatable path consumes the
    // executor. Clear the borrowed pointer; destroying it here double-frees.
    resources->executor = nullptr;
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) {
        Fail("cannot read process peak RSS");
    }
    char metadata[768];
    const int metadata_size = std::snprintf(
        metadata, sizeof(metadata),
        "schema_version=6\ninput_rank=%d\nbatch_size=%" PRId64
        "\nworkspace_size_bytes=%" PRIu64
        "\nexecute_api_called=true\ndevice_kernel_submitted=false"
        "\nduration_consumed=false"
        "\nexecutor_lifecycle=consumed_by_aclnnMatmul"
        "\nprocess_peak_rss_kib=%ld\n",
        input_rank, batch, workspace_size, usage.ru_maxrss);
    if (metadata_size <= 0
        || static_cast<std::size_t>(metadata_size) >= sizeof(metadata)) {
        Fail("helper metadata overflow");
    }
    WriteExclusive(
        metadata_output, metadata, static_cast<std::size_t>(metadata_size));
}

int64_t ParsePositive(const char* text, const char* name) {
    errno = 0;
    char* end = nullptr;
    const long long value = std::strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value <= 0) {
        std::fprintf(stderr, "%s must be a positive integer\n", name);
        std::exit(2);
    }
    return static_cast<int64_t>(value);
}

int32_t ParseNonnegative(const char* text, const char* name) {
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 0
        || value > std::numeric_limits<int32_t>::max()) {
        std::fprintf(stderr, "%s must be a non-negative int32\n", name);
        std::exit(2);
    }
    return static_cast<int32_t>(value);
}

int32_t ParseInputRank(const char* text) {
    const int32_t rank = ParseNonnegative(text, "input_rank");
    if (rank != 2 && rank != 3) {
        Fail("input_rank must be 2 or 3");
    }
    return rank;
}

std::size_t TensorBytes(int64_t first, int64_t second) {
    constexpr std::size_t kBfloat16Bytes = 2;
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (static_cast<uint64_t>(first) > maximum / kBfloat16Bytes
        || static_cast<uint64_t>(second)
            > maximum / (static_cast<std::size_t>(first) * kBfloat16Bytes)) {
        Fail("tensor byte size overflows size_t");
    }
    return static_cast<std::size_t>(first)
        * static_cast<std::size_t>(second) * kBfloat16Bytes;
}

std::size_t BatchedTensorBytes(
        int64_t batch, int64_t first, int64_t second) {
    const std::size_t per_batch = TensorBytes(first, second);
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (static_cast<uint64_t>(batch) > maximum / per_batch) {
        Fail("batched tensor byte size overflows size_t");
    }
    return static_cast<std::size_t>(batch) * per_batch;
}

int64_t MatrixStride(int64_t rows, int64_t columns) {
    const auto maximum = std::numeric_limits<int64_t>::max();
    if (rows > maximum / columns) {
        Fail("tensor stride overflows int64");
    }
    return rows * columns;
}

bool WriteAll(int descriptor, const uint8_t* data, std::size_t size) {
    while (size > 0) {
        const ssize_t written = write(descriptor, data, size);
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0) {
            return false;
        }
        data += written;
        size -= static_cast<std::size_t>(written);
    }
    return true;
}

void WriteExclusive(const char* path, const void* data, std::size_t size) {
    const int descriptor = open(
        path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (descriptor < 0) {
        std::fprintf(stderr, "cannot create helper output %s: %s\n", path,
                     std::strerror(errno));
        std::exit(1);
    }
    if (!WriteAll(descriptor, static_cast<const uint8_t*>(data), size)
        || fsync(descriptor) != 0
        || close(descriptor) != 0) {
        Fail("cannot finish helper output");
    }
}

template <typename Function>
Function ResolvePreloadSymbol(const char* name) {
    dlerror();
    void* address = dlsym(RTLD_DEFAULT, name);
    const char* error = dlerror();
    if (address == nullptr || error != nullptr) {
        std::fprintf(stderr, "cannot resolve preload symbol %s: %s\n", name,
                     error == nullptr ? "unknown error" : error);
        std::exit(1);
    }
    return reinterpret_cast<Function>(address);
}

}  // namespace

int main(int argc, char** argv) {
    const bool batch_mode = argc == 3 && std::string(argv[1]) == "--batch";
    if (!batch_mode && argc != 9) {
        std::fprintf(
            stderr,
            "usage: %s input_rank B M N K cube_math_type device_id metadata_output "
            "\n",
            argv[0]);
        return 2;
    }
    const int32_t device_id = ParseNonnegative(
        batch_mode ? argv[2] : argv[6], "device_id");

    Resources resources;
    resources.device_id = device_id;
    ACL_CHECK(aclInit(nullptr));
    resources.acl_initialized = true;
    ACL_CHECK(aclrtSetDevice(device_id));
    resources.device_set = true;
    ACL_CHECK(aclrtCreateStream(&resources.stream));
    if (!batch_mode) {
        const int32_t input_rank = ParseInputRank(argv[1]);
        const int64_t batch = ParsePositive(argv[2], "B");
        const int64_t m = ParsePositive(argv[3], "M");
        const int64_t n = ParsePositive(argv[4], "N");
        const int64_t k = ParsePositive(argv[5], "K");
        const int32_t cube_math_type = ParseNonnegative(
            argv[6], "cube_math_type");
        RunOperation(&resources, input_rank, batch, m, n, k, cube_math_type,
                     std::getenv("MIDLAYER_ASCEND_GEMM_DRY_LAUNCH_DIR"),
                     argv[8]);
        return 0;
    }
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) Fail("batch request line must not be empty");
        std::istringstream input(line);
        std::string operation_id;
        std::string input_rank_text, batch_text, m_text, n_text, k_text, math_text;
        std::string capture_directory, metadata_output, extra;
        if (!(input >> operation_id >> input_rank_text >> batch_text >> m_text >> n_text
                    >> k_text >> math_text >> capture_directory
                    >> metadata_output)
                || (input >> extra)) {
            Fail("batch request must contain nine whitespace-free fields");
        }
        const int32_t math_type = ParseNonnegative(
            math_text.c_str(), "cube_math_type");
        if (math_type > std::numeric_limits<int8_t>::max()) {
            Fail("cube_math_type exceeds int8 range");
        }
        RunOperation(
            &resources, ParseInputRank(input_rank_text.c_str()),
            ParsePositive(batch_text.c_str(), "B"),
            ParsePositive(m_text.c_str(), "M"),
            ParsePositive(n_text.c_str(), "N"),
            ParsePositive(k_text.c_str(), "K"), math_type,
            capture_directory.c_str(), metadata_output.c_str());
        std::printf("%s\n", operation_id.c_str());
        std::fflush(stdout);
    }
    if (!std::cin.eof()) Fail("cannot read batch requests");
    return 0;
}
