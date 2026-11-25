// Capture non-duration launch arguments for the GatherV3 kernels used by the
// public ATB RoPE entry.  Capture is enabled only while
// MIDLAYER_ASCEND_GATHER_V3_CAPTURE_DIR is set by the application runner.

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
#include <string>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

constexpr const char* kCaptureDirectoryEnvironment =
    "MIDLAYER_ASCEND_GATHER_V3_CAPTURE_DIR";
constexpr std::size_t kMaximumHostArgumentBytes = 64 * 1024;
constexpr std::size_t kMaximumPlaceholders = 64;

static_assert(sizeof(aclrtPlaceHolderInfo) == 2 * sizeof(uint32_t));

std::atomic<uint64_t> g_sequence{0};

template <typename Function>
Function ResolveVersionedSymbol(void* handle, const char* symbol,
                                const char* version) {
    dlerror();
    void* address = dlvsym(handle, symbol, version);
    const char* error = dlerror();
    if (address == nullptr || error != nullptr) {
        return nullptr;
    }
    return reinterpret_cast<Function>(address);
}

using DlsymFunction = void* (*)(void*, const char*);

DlsymFunction RealDlsym() {
    static DlsymFunction function = [] {
        DlsymFunction resolved = ResolveVersionedSymbol<DlsymFunction>(
            RTLD_NEXT, "dlsym", "GLIBC_2.34");
        if (resolved == nullptr) {
            resolved = ResolveVersionedSymbol<DlsymFunction>(
                RTLD_NEXT, "dlsym", "GLIBC_2.17");
        }
        if (resolved == nullptr) {
            std::fprintf(stderr,
                         "GatherV3 launch capture cannot resolve dlsym\n");
            std::abort();
        }
        return resolved;
    }();
    return function;
}

template <typename Function>
Function ResolveLoadedLibrary(const char* library, const char* symbol) {
    void* handle = dlopen(library, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    if (handle == nullptr) {
        std::fprintf(stderr,
                     "GatherV3 launch capture cannot find loaded %s: %s\n",
                     library, dlerror());
        std::abort();
    }
    dlerror();
    void* address = RealDlsym()(handle, symbol);
    const char* error = dlerror();
    if (address == nullptr || error != nullptr) {
        std::fprintf(stderr,
                     "GatherV3 launch capture cannot resolve %s: %s\n",
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

bool WriteExclusiveFile(const char* path, const void* data, std::size_t size) {
    const int descriptor = open(
        path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (descriptor < 0) {
        std::fprintf(stderr, "GatherV3 capture cannot create %s: %s\n",
                     path, std::strerror(errno));
        return false;
    }
    const bool complete = WriteAll(descriptor, data, size);
    const int close_result = close(descriptor);
    return complete && close_result == 0;
}

bool BuildPath(char* destination, std::size_t capacity,
               const char* directory, uint64_t sequence,
               const char* suffix) {
    const int length = std::snprintf(
        destination, capacity, "%s/capture_%d_%ld_%06" PRIu64 ".%s",
        directory, static_cast<int>(getpid()), syscall(SYS_gettid), sequence,
        suffix);
    return length > 0 && static_cast<std::size_t>(length) < capacity;
}

void CaptureLaunch(aclrtFuncHandle function, uint32_t block_dim,
                   aclrtStream stream, const void* args,
                   std::size_t args_size,
                   const aclrtPlaceHolderInfo* placeholders,
                   std::size_t placeholder_count) {
    const char* directory = std::getenv(kCaptureDirectoryEnvironment);
    if (directory == nullptr || *directory == '\0' || args == nullptr
        || args_size == 0 || args_size > kMaximumHostArgumentBytes
        || placeholder_count > kMaximumPlaceholders
        || (placeholder_count > 0 && placeholders == nullptr)) {
        return;
    }
    char runtime_name[512]{};
    using GetFunctionName = decltype(&aclrtGetFunctionName);
    static GetFunctionName get_function_name =
        ResolveLoadedLibrary<GetFunctionName>(
            "libacl_rt.so", "aclrtGetFunctionName");
    const aclError name_status = get_function_name(
        function, static_cast<uint32_t>(sizeof(runtime_name)), runtime_name);
    if (name_status != ACL_SUCCESS
        || std::strstr(runtime_name, "GatherV3") == nullptr) {
        return;
    }

    const uint64_t sequence = g_sequence.fetch_add(1) + 1;
    char args_path[PATH_MAX];
    char placeholders_path[PATH_MAX];
    char metadata_path[PATH_MAX];
    if (!BuildPath(args_path, sizeof(args_path), directory, sequence, "bin")
        || !BuildPath(placeholders_path, sizeof(placeholders_path), directory,
                      sequence, "placeholders")
        || !BuildPath(metadata_path, sizeof(metadata_path), directory,
                      sequence, "meta")) {
        std::fprintf(stderr, "GatherV3 capture path is too long\n");
        std::abort();
    }
    if (!WriteExclusiveFile(args_path, args, args_size)
        || !WriteExclusiveFile(
            placeholders_path, placeholders,
            placeholder_count * sizeof(aclrtPlaceHolderInfo))) {
        std::fprintf(stderr, "GatherV3 capture could not write launch data\n");
        std::abort();
    }
    const char* args_name = std::strrchr(args_path, '/');
    args_name = args_name == nullptr ? args_path : args_name + 1;
    const char* placeholders_name = std::strrchr(placeholders_path, '/');
    placeholders_name = placeholders_name == nullptr
        ? placeholders_path : placeholders_name + 1;
    char metadata[2048];
    const int length = std::snprintf(
        metadata, sizeof(metadata),
        "schema_version=1\nrecord_type=launch_args\nsequence=%" PRIu64
        "\npid=%d\ntid=%ld\nruntime_function_name=%s"
        "\nblock_dim=%u\nargs_size=%zu\nargs_file=%s"
        "\nplaceholder_count=%zu\nplaceholder_entry_size=%zu"
        "\nplaceholder_file=%s\nlaunch_stream=%" PRIuPTR "\n",
        sequence, static_cast<int>(getpid()), syscall(SYS_gettid),
        runtime_name, block_dim, args_size, args_name, placeholder_count,
        sizeof(aclrtPlaceHolderInfo), placeholders_name,
        reinterpret_cast<uintptr_t>(stream));
    if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(metadata)
        || !WriteExclusiveFile(
            metadata_path, metadata, static_cast<std::size_t>(length))) {
        std::fprintf(stderr, "GatherV3 capture could not write metadata\n");
        std::abort();
    }
}

}  // namespace

extern "C" aclError aclrtLaunchKernelWithHostArgs(
    aclrtFuncHandle function, uint32_t block_dim, aclrtStream stream,
    aclrtLaunchKernelCfg* configuration, void* host_args,
    std::size_t args_size, aclrtPlaceHolderInfo* placeholders,
    std::size_t placeholder_count) {
    using Original = decltype(&aclrtLaunchKernelWithHostArgs);
    static Original original = ResolveLoadedLibrary<Original>(
        "libacl_rt.so", "aclrtLaunchKernelWithHostArgs");
    CaptureLaunch(function, block_dim, stream, host_args, args_size,
                  placeholders, placeholder_count);
    return original(function, block_dim, stream, configuration, host_args,
                    args_size, placeholders, placeholder_count);
}

extern "C" void* dlsym(void* handle, const char* symbol) {
    void* original = RealDlsym()(handle, symbol);
    if (std::getenv(kCaptureDirectoryEnvironment) != nullptr
        && original != nullptr
        && std::strcmp(symbol, "aclrtLaunchKernelWithHostArgs") == 0) {
        return reinterpret_cast<void*>(&aclrtLaunchKernelWithHostArgs);
    }
    return original;
}
