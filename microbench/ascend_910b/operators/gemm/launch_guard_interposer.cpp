// Capture one aclnnMatmul launch and return success without submitting it.

#include <acl/acl_rt.h>

#include <cerrno>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits.h>
#include <mutex>
#include <string>
#include <unistd.h>

namespace {

constexpr const char* kCaptureDirectoryEnvironment =
    "MIDLAYER_ASCEND_GEMM_DRY_LAUNCH_DIR";
constexpr std::size_t kMaximumHostArgumentBytes = 1024 * 1024;
constexpr std::size_t kMaximumPlaceholders = 256;
constexpr std::size_t kMaximumConfigurationAttributes = 64;

#define ACL_CHECK(call) ((call) == ACL_SUCCESS)

static_assert(sizeof(aclrtPlaceHolderInfo) == 2 * sizeof(uint32_t));

std::mutex g_mutex;
bool g_marker_active = false;
std::size_t g_expected_launches = 0;
std::size_t g_captured_launches = 0;
std::string g_capture_directory;
std::string g_failure;

void SetFailureLocked(const std::string& message) {
    if (g_failure.empty()) {
        g_failure = message;
    }
}

bool WriteAll(int descriptor, const void* data, std::size_t size) {
    const auto* cursor = static_cast<const uint8_t*>(data);
    while (size > 0) {
        const ssize_t written = write(descriptor, cursor, size);
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0) {
            return false;
        }
        cursor += written;
        size -= static_cast<std::size_t>(written);
    }
    return true;
}

bool BuildPath(char* destination, std::size_t capacity, const char* name) {
    const int length = std::snprintf(
        destination, capacity, "%s/%s", g_capture_directory.c_str(), name);
    return length > 0 && static_cast<std::size_t>(length) < capacity;
}

bool WriteExclusive(const char* name, const void* data, std::size_t size) {
    char path[PATH_MAX];
    if (!BuildPath(path, sizeof(path), name)) {
        SetFailureLocked(std::string("capture path is too long: ") + name);
        return false;
    }
    const int descriptor = open(
        path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (descriptor < 0) {
        SetFailureLocked(
            std::string("cannot create capture file ") + path + ": "
            + std::strerror(errno));
        return false;
    }
    const bool complete = WriteAll(descriptor, data, size);
    const int sync_status = fsync(descriptor);
    const int close_status = close(descriptor);
    if (!complete || sync_status != 0 || close_status != 0) {
        SetFailureLocked(std::string("cannot finish capture file ") + path);
        return false;
    }
    return true;
}

bool SafeKernelName(aclrtFuncHandle function, char* storage,
                    std::size_t storage_size) {
    std::memset(storage, 0, storage_size);
    return ACL_CHECK(aclrtGetFunctionName(
               function, static_cast<uint32_t>(storage_size), storage))
        && storage[0] != '\0'
        && std::strchr(storage, '\n') == nullptr
        && std::strchr(storage, '\r') == nullptr
        && std::strchr(storage, '=') == nullptr;
}

void CaptureHostArgsLaunch(
        aclrtFuncHandle function, uint32_t block_dim, aclrtStream stream,
        aclrtLaunchKernelCfg* configuration, void* host_args,
        std::size_t args_size, aclrtPlaceHolderInfo* placeholders,
        std::size_t placeholder_count) {
    std::lock_guard<std::mutex> guard(g_mutex);
    if (!g_marker_active) {
        std::fprintf(stderr, "GEMM launch occurred outside dry-launch marker\n");
        std::_Exit(86);
    }
    ++g_captured_launches;
    if (g_captured_launches != 1) {
        SetFailureLocked(
            "aclnnMatmul emitted more than one device launch request");
        return;
    }
    if (function == nullptr || block_dim == 0 || host_args == nullptr
        || args_size == 0 || args_size > kMaximumHostArgumentBytes
        || placeholder_count > kMaximumPlaceholders
        || (placeholder_count > 0 && placeholders == nullptr)) {
        SetFailureLocked("captured aclrtLaunchKernelWithHostArgs is invalid");
        return;
    }
    std::size_t attribute_count = 0;
    const aclrtLaunchKernelAttr* attributes = nullptr;
    if (configuration != nullptr) {
        attribute_count = configuration->numAttrs;
        attributes = configuration->attrs;
        if (attribute_count > kMaximumConfigurationAttributes
            || (attribute_count > 0 && attributes == nullptr)) {
            SetFailureLocked("captured launch configuration is invalid");
            return;
        }
    }
    char kernel_name[1024];
    if (!SafeKernelName(function, kernel_name, sizeof(kernel_name))) {
        SetFailureLocked("cannot resolve captured MatMul kernel name");
        return;
    }
    if (!WriteExclusive("host_args.bin", host_args, args_size)
        || !WriteExclusive(
            "placeholders.bin", placeholders,
            placeholder_count * sizeof(aclrtPlaceHolderInfo))
        || !WriteExclusive(
            "launch_config_attrs.bin", attributes,
            attribute_count * sizeof(aclrtLaunchKernelAttr))) {
        return;
    }
    char metadata[4096];
    const int length = std::snprintf(
        metadata, sizeof(metadata),
        "schema_version=2\nrecord_type=launch\n"
        "api=aclrtLaunchKernelWithHostArgs\nexecute_api=aclnnMatmul\n"
        "kernel_stub=%s\nblock_dim=%u\nstream=%" PRIuPTR
        "\nconfiguration_present=%s\nconfiguration_attr_count=%zu\n"
        "configuration_attr_entry_size=%zu\n"
        "configuration_attr_file=launch_config_attrs.bin\n"
        "host_args_size=%zu\nhost_args_file=host_args.bin\n"
        "placeholder_count=%zu\nplaceholder_entry_size=%zu\n"
        "placeholder_file=placeholders.bin\n"
        "device_kernel_submitted=false\nduration_consumed=false\n",
        kernel_name, block_dim, reinterpret_cast<uintptr_t>(stream),
        configuration == nullptr ? "false" : "true", attribute_count,
        sizeof(aclrtLaunchKernelAttr), args_size, placeholder_count,
        sizeof(aclrtPlaceHolderInfo));
    if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(metadata)) {
        SetFailureLocked("dry-launch metadata overflow");
        return;
    }
    WriteExclusive("launch.meta", metadata, static_cast<std::size_t>(length));
}

void RejectUnsupportedLaunch(const char* api) {
    std::lock_guard<std::mutex> guard(g_mutex);
    if (!g_marker_active) {
        std::fprintf(stderr, "%s occurred outside dry-launch marker\n", api);
        std::_Exit(86);
    }
    SetFailureLocked(
        std::string("unsupported launch API during aclnnMatmul: ") + api);
}

}  // namespace

extern "C" int midlayer_ascend_gemm_dry_launch_begin(
        std::size_t expected_launches) {
    std::lock_guard<std::mutex> guard(g_mutex);
    if (g_marker_active) {
        SetFailureLocked("dry-launch marker is already active");
        return -1;
    }
    if (expected_launches != 1) {
        SetFailureLocked("Ascend GEMM PoC requires exactly one launch");
        return -1;
    }
    const char* directory = std::getenv(kCaptureDirectoryEnvironment);
    if (directory == nullptr || directory[0] == '\0') {
        SetFailureLocked("MIDLAYER_ASCEND_GEMM_DRY_LAUNCH_DIR is required");
        return -1;
    }
    g_capture_directory = directory;
    g_expected_launches = expected_launches;
    g_captured_launches = 0;
    g_failure.clear();
    g_marker_active = true;
    return 0;
}

extern "C" int midlayer_ascend_gemm_dry_launch_end() {
    std::lock_guard<std::mutex> guard(g_mutex);
    if (!g_marker_active) {
        SetFailureLocked("dry-launch marker is not active");
        return -1;
    }
    g_marker_active = false;
    if (g_captured_launches != g_expected_launches) {
        SetFailureLocked(
            "captured " + std::to_string(g_captured_launches)
            + " launch(es), expected "
            + std::to_string(g_expected_launches));
    }
    return g_failure.empty() ? 0 : -1;
}

extern "C" const char* midlayer_ascend_gemm_dry_launch_error() {
    std::lock_guard<std::mutex> guard(g_mutex);
    return g_failure.c_str();
}

extern "C" aclError aclrtLaunchKernelWithHostArgs(
    aclrtFuncHandle function, uint32_t block_dim, aclrtStream stream,
    aclrtLaunchKernelCfg* configuration, void* host_args,
    std::size_t args_size, aclrtPlaceHolderInfo* placeholders,
    std::size_t placeholder_count) {
    CaptureHostArgsLaunch(
        function, block_dim, stream, configuration, host_args, args_size,
        placeholders, placeholder_count);
    return ACL_SUCCESS;
}

extern "C" aclError aclrtLaunchKernel(
    aclrtFuncHandle, uint32_t, const void*, std::size_t, aclrtStream) {
    RejectUnsupportedLaunch("aclrtLaunchKernel");
    return ACL_SUCCESS;
}

extern "C" aclError aclrtLaunchKernelV2(
    aclrtFuncHandle, uint32_t, const void*, std::size_t,
    aclrtLaunchKernelCfg*, aclrtStream) {
    RejectUnsupportedLaunch("aclrtLaunchKernelV2");
    return ACL_SUCCESS;
}

extern "C" aclError aclrtLaunchKernelWithConfig(
    aclrtFuncHandle, uint32_t, aclrtStream, aclrtLaunchKernelCfg*,
    aclrtArgsHandle, void*) {
    RejectUnsupportedLaunch("aclrtLaunchKernelWithConfig");
    return ACL_SUCCESS;
}
