// Capture non-duration CANN GroupedMatmul host arguments and placeholders.
//
// The eager OpAPI path launches the selected GroupedMatmul binary through
// aclrtLaunchKernelWithHostArgs.  The serialized tiling payload is embedded in
// that host-argument buffer; this interposer records it without reading kernel
// duration or profiler timing counters.

#include <acl/acl_rt.h>

#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

constexpr const char* kCaptureDirectoryEnvironment =
    "MIDLAYER_ASCEND_GMM_TILING_CAPTURE_DIR";
constexpr std::size_t kMaximumHostArgumentBytes = 8192;
constexpr std::size_t kMaximumPlaceholders = 64;

std::atomic<uint64_t> g_sequence{0};

template <typename Function>
Function ResolveOriginal(const char* symbol) {
    void* handle = dlopen("libacl_rt.so", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    if (handle == nullptr) {
        std::fprintf(stderr, "GMM tiling capture cannot find libacl_rt.so: %s\n",
                     dlerror());
        std::abort();
    }
    dlerror();
    void* address = dlsym(handle, symbol);
    const char* error = dlerror();
    if (address == nullptr || error != nullptr) {
        std::fprintf(stderr, "GMM tiling capture cannot resolve %s: %s\n",
                     symbol, error == nullptr ? "unknown error" : error);
        std::abort();
    }
    return reinterpret_cast<Function>(address);
}

bool WriteAll(int descriptor, const void* data, std::size_t size) {
    const auto* cursor = static_cast<const uint8_t*>(data);
    while (size > 0) {
        const ssize_t written = write(descriptor, cursor, size);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (written == 0) {
            return false;
        }
        cursor += written;
        size -= static_cast<std::size_t>(written);
    }
    return true;
}

bool BuildRecordPath(char* destination, std::size_t capacity,
                     const char* directory, uint64_t sequence,
                     const char* suffix) {
    const int length = std::snprintf(
        destination, capacity, "%s/capture_%d_%ld_%06" PRIu64 ".%s",
        directory, static_cast<int>(getpid()), syscall(SYS_gettid), sequence,
        suffix);
    return length > 0 && static_cast<std::size_t>(length) < capacity;
}

bool WriteExclusiveFile(const char* path, const void* data, std::size_t size) {
    const int descriptor = open(
        path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (descriptor < 0) {
        std::fprintf(stderr, "GMM tiling capture cannot create %s: %s\n",
                     path, std::strerror(errno));
        return false;
    }
    const bool complete = WriteAll(descriptor, data, size);
    const int close_result = close(descriptor);
    if (!complete || close_result != 0) {
        std::fprintf(stderr, "GMM tiling capture cannot finish %s\n", path);
        return false;
    }
    return true;
}

bool ResolveRuntimeFunctionName(aclrtFuncHandle function, char* storage,
                                std::size_t storage_size,
                                aclError* status) {
    using GetFunctionName = decltype(&aclrtGetFunctionName);
    static GetFunctionName get_function_name =
        ResolveOriginal<GetFunctionName>("aclrtGetFunctionName");
    std::memset(storage, 0, storage_size);
    *status = get_function_name(
        function, static_cast<uint32_t>(storage_size), storage);
    return *status == ACL_SUCCESS && storage[0] != '\0'
        && std::strchr(storage, '\n') == nullptr
        && std::strchr(storage, '\r') == nullptr
        && std::strchr(storage, '=') == nullptr;
}

void CaptureLaunch(aclrtFuncHandle function, uint32_t block_dim,
                   aclrtStream stream,
                   const void* args, std::size_t args_size,
                   const aclrtPlaceHolderInfo* placeholders,
                   std::size_t placeholder_count) {
    const char* directory = std::getenv(kCaptureDirectoryEnvironment);
    if (directory == nullptr || *directory == '\0' || args == nullptr
        || args_size == 0 || args_size > kMaximumHostArgumentBytes
        || placeholder_count > kMaximumPlaceholders
        || (placeholder_count > 0 && placeholders == nullptr)) {
        return;
    }
    char runtime_name[512];
    aclError name_status = ACL_SUCCESS;
    const bool has_runtime_name = ResolveRuntimeFunctionName(
        function, runtime_name, sizeof(runtime_name), &name_status);
    const char* selected = has_runtime_name ? runtime_name : "unavailable";

    const uint64_t sequence = g_sequence.fetch_add(1) + 1;
    char args_path[PATH_MAX];
    char placeholders_path[PATH_MAX];
    char metadata_path[PATH_MAX];
    if (!BuildRecordPath(
            args_path, sizeof(args_path), directory, sequence, "bin")
        || !BuildRecordPath(placeholders_path, sizeof(placeholders_path),
                            directory, sequence, "placeholders")
        || !BuildRecordPath(metadata_path, sizeof(metadata_path), directory,
                            sequence, "meta")) {
        std::fprintf(stderr, "GMM tiling capture path is too long\n");
        return;
    }
    if (!WriteExclusiveFile(args_path, args, args_size)
        || !WriteExclusiveFile(
            placeholders_path, placeholders,
            placeholder_count * sizeof(aclrtPlaceHolderInfo))) {
        return;
    }
    char metadata[1536];
    const int length = std::snprintf(
        metadata, sizeof(metadata),
        "schema_version=2\nrecord_type=launch_args\nsequence=%" PRIu64
        "\npid=%d\ntid=%ld\nruntime_function_name_status=%d"
        "\nruntime_function_name=%s\nblock_dim=%u"
        "\nargs_size=%zu\nplaceholder_count=%zu"
        "\nplaceholder_entry_size=%zu\nstream=%" PRIuPTR "\n",
        sequence, static_cast<int>(getpid()), syscall(SYS_gettid),
        static_cast<int>(name_status), selected, block_dim, args_size,
        placeholder_count,
        sizeof(aclrtPlaceHolderInfo), reinterpret_cast<uintptr_t>(stream));
    if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(metadata)) {
        std::fprintf(stderr, "GMM tiling capture metadata overflow\n");
        return;
    }
    WriteExclusiveFile(metadata_path, metadata, static_cast<std::size_t>(length));
}

}  // namespace

extern "C" aclError aclrtLaunchKernelWithHostArgs(
    aclrtFuncHandle function, uint32_t block_dim, aclrtStream stream,
    aclrtLaunchKernelCfg* configuration, void* host_args,
    std::size_t args_size, aclrtPlaceHolderInfo* placeholders,
    std::size_t placeholder_count) {
    using Original = decltype(&aclrtLaunchKernelWithHostArgs);
    static Original original = ResolveOriginal<Original>(
        "aclrtLaunchKernelWithHostArgs");
    CaptureLaunch(function, block_dim, stream, host_args, args_size,
                  placeholders, placeholder_count);
    return original(function, block_dim, stream, configuration, host_args,
                    args_size, placeholders, placeholder_count);
}
