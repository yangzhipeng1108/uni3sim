// Capture non-duration AddRmsNorm launch arguments from CANN 8.5.1.
//
// The interposer records the complete aclrtLaunchKernelWithHostArgs buffer and
// placeholder map while aclnnAddRmsNorm is active.  It deliberately does not
// assume a tiling-data offset: the offline report identifies the unique
// placeholder-backed 80-byte AddRmsNormTilingData payload from reflected fields.

#include <acl/acl_rt.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_add_rms_norm.h>

#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <link.h>
#include <limits.h>
#include <string>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

constexpr const char* kCaptureDirectoryEnvironment =
    "MIDLAYER_ASCEND_ADDRMS_CAPTURE_DIR";
constexpr std::size_t kMaximumHostArgumentBytes = 64 * 1024;
constexpr std::size_t kMaximumPlaceholders = 64;

static_assert(sizeof(aclrtPlaceHolderInfo) == 2 * sizeof(uint32_t));

std::atomic<uint64_t> g_sequence{0};
std::atomic<void*> g_original_aclnnAddRmsNorm{nullptr};
thread_local const char* g_active_execute_api = nullptr;
thread_local uintptr_t g_active_execute_stream = 0;

const char* CaptureDirectory() {
    const char* directory = std::getenv(kCaptureDirectoryEnvironment);
    if (directory == nullptr || *directory == '\0') {
        return nullptr;
    }
    return directory;
}

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
                         "AddRmsNorm tiling capture cannot resolve real dlsym\n");
            std::abort();
        }
        return resolved;
    }();
    return function;
}

template <typename Function>
Function ResolveNext(const char* symbol) {
    dlerror();
    void* address = RealDlsym()(RTLD_NEXT, symbol);
    const char* error = dlerror();
    if (address == nullptr || error != nullptr) {
        std::fprintf(stderr, "AddRmsNorm tiling capture cannot resolve %s: %s\n",
                     symbol, error == nullptr ? "unknown error" : error);
        std::abort();
    }
    return reinterpret_cast<Function>(address);
}

template <typename Function>
Function ResolveLoadedLibrary(const char* library, const char* symbol) {
    void* handle = dlopen(library, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    if (handle == nullptr) {
        std::fprintf(stderr,
                     "AddRmsNorm tiling capture cannot find loaded %s: %s\n",
                     library, dlerror());
        std::abort();
    }
    dlerror();
    void* address = RealDlsym()(handle, symbol);
    const char* error = dlerror();
    if (address == nullptr || error != nullptr) {
        std::fprintf(stderr,
                     "AddRmsNorm tiling capture cannot resolve %s from %s: %s\n",
                     symbol, library,
                     error == nullptr ? "unknown error" : error);
        std::abort();
    }
    return reinterpret_cast<Function>(address);
}

void RememberOriginalExecute(void* address) {
    if (address == nullptr) {
        return;
    }
    void* expected = nullptr;
    if (!g_original_aclnnAddRmsNorm.compare_exchange_strong(
            expected, address, std::memory_order_release,
            std::memory_order_acquire)
        && expected != address) {
        std::fprintf(stderr,
                     "AddRmsNorm tiling capture saw conflicting execute APIs\n");
        std::abort();
    }
}

template <typename Function>
Function ResolveOriginalExecute() {
    void* address = g_original_aclnnAddRmsNorm.load(std::memory_order_acquire);
    if (address != nullptr) {
        return reinterpret_cast<Function>(address);
    }
    return ResolveNext<Function>("aclnnAddRmsNorm");
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
        std::fprintf(stderr, "AddRmsNorm tiling capture cannot create %s: %s\n",
                     path, std::strerror(errno));
        return false;
    }
    const bool complete = WriteAll(descriptor, data, size);
    const int close_result = close(descriptor);
    if (!complete || close_result != 0) {
        std::fprintf(stderr, "AddRmsNorm tiling capture cannot finish %s\n", path);
        return false;
    }
    return true;
}

void WriteMetadata(uint64_t sequence, const char* text) {
    const char* directory = CaptureDirectory();
    if (directory == nullptr) {
        return;
    }
    char path[PATH_MAX];
    if (!BuildRecordPath(path, sizeof(path), directory, sequence, "meta")) {
        std::fprintf(stderr, "AddRmsNorm metadata path is too long\n");
        return;
    }
    WriteExclusiveFile(path, text, std::strlen(text));
}

void CaptureWorkspace(const char* api, uint64_t workspace_size,
                      aclrtStream stream) {
    if (CaptureDirectory() == nullptr) {
        return;
    }
    const uint64_t sequence = g_sequence.fetch_add(1) + 1;
    char metadata[768];
    const int length = std::snprintf(
        metadata, sizeof(metadata),
        "schema_version=1\nrecord_type=workspace\nsequence=%" PRIu64
        "\napi=%s\npid=%d\ntid=%ld\nworkspace_size=%" PRIu64
        "\nexecute_stream=%" PRIuPTR "\n",
        sequence, api, static_cast<int>(getpid()), syscall(SYS_gettid),
        workspace_size, reinterpret_cast<uintptr_t>(stream));
    if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(metadata)) {
        std::fprintf(stderr, "AddRmsNorm workspace metadata overflow\n");
        return;
    }
    WriteMetadata(sequence, metadata);
}

bool CopyIdentityText(const char* source, char* destination,
                      std::size_t capacity, bool symbol) {
    if (source == nullptr || *source == '\0') {
        return false;
    }
    const std::string text(source);
    if (text.size() + 1 > capacity || text.find('\n') != std::string::npos
        || text.find('\r') != std::string::npos
        || (symbol && text.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
               != std::string::npos)) {
        return false;
    }
    std::memcpy(destination, text.c_str(), text.size() + 1);
    return true;
}

struct StubIdentity {
    const char* kind = "unresolved";
    aclError runtime_name_status = ACL_SUCCESS;
    char runtime_function_name[512] = "none";
    const char* elf_membership = "not_mapped_elf";
    char elf_object_path[PATH_MAX] = "none";
    uintptr_t elf_object_base = 0;
    uintptr_t elf_object_offset = 0;
    char elf_symbol[512] = "none";
    uintptr_t elf_symbol_address = 0;
    uintptr_t elf_symbol_offset = 0;
};

StubIdentity ResolveStubIdentity(aclrtFuncHandle stub) {
    StubIdentity identity;
    using GetFunctionName = decltype(&aclrtGetFunctionName);
    static GetFunctionName get_function_name =
        ResolveLoadedLibrary<GetFunctionName>(
            "libacl_rt.so", "aclrtGetFunctionName");
    char runtime_name[sizeof(identity.runtime_function_name)]{};
    identity.runtime_name_status = get_function_name(
        stub, static_cast<uint32_t>(sizeof(runtime_name)), runtime_name);
    const bool has_runtime_name =
        identity.runtime_name_status == ACL_SUCCESS
        && std::memchr(runtime_name, '\0', sizeof(runtime_name)) != nullptr
        && CopyIdentityText(
            runtime_name, identity.runtime_function_name,
            sizeof(identity.runtime_function_name), true);

    Dl_info information{};
    void* symbol_information = nullptr;
    const int dladdr_result = dladdr1(
        stub, &information, &symbol_information,
        RTLD_DL_SYMENT);
    const auto* symbol_entry = static_cast<const ElfW(Sym)*>(
        symbol_information);
    if (dladdr_result != 0 && information.dli_fname != nullptr
        && information.dli_fbase != nullptr) {
        char resolved_path[PATH_MAX];
        const char* object_path = realpath(
            information.dli_fname, resolved_path);
        if (object_path != nullptr
            && CopyIdentityText(
                object_path, identity.elf_object_path,
                sizeof(identity.elf_object_path), false)) {
            const uintptr_t address = reinterpret_cast<uintptr_t>(stub);
            identity.elf_object_base = reinterpret_cast<uintptr_t>(
                information.dli_fbase);
            if (address >= identity.elf_object_base) {
                identity.elf_membership = "mapped_elf";
                identity.elf_object_offset =
                    address - identity.elf_object_base;
                if (information.dli_sname != nullptr
                    && information.dli_saddr != nullptr
                    && CopyIdentityText(
                        information.dli_sname, identity.elf_symbol,
                        sizeof(identity.elf_symbol), true)) {
                    identity.elf_symbol_address = reinterpret_cast<uintptr_t>(
                        information.dli_saddr);
                    if (address >= identity.elf_symbol_address
                        && symbol_entry != nullptr) {
                        const uintptr_t symbol_offset =
                            address - identity.elf_symbol_address;
                        if (symbol_entry->st_size == 0
                            || symbol_offset < symbol_entry->st_size) {
                            identity.elf_symbol_offset = symbol_offset;
                        } else {
                            std::memcpy(
                                identity.elf_symbol, "none", sizeof("none"));
                            identity.elf_symbol_address = 0;
                        }
                    } else {
                        std::memcpy(
                            identity.elf_symbol, "none", sizeof("none"));
                        identity.elf_symbol_address = 0;
                    }
                }
            }
        }
    }
    if (has_runtime_name) {
        identity.kind = "runtime_function_name";
    } else if (std::strcmp(identity.elf_symbol, "none") != 0) {
        identity.kind = "elf_symbol";
    } else if (std::strcmp(identity.elf_membership, "mapped_elf") == 0) {
        identity.kind = "elf_object_offset";
    }
    return identity;
}

void CaptureLaunchArguments(const char* launch_api, aclrtFuncHandle stub,
                            uint32_t block_dim, aclrtStream stream,
                            const void* args,
                            std::size_t args_size,
                            const aclrtPlaceHolderInfo* placeholders,
                            std::size_t placeholder_count) {
    const char* directory = CaptureDirectory();
    if (directory == nullptr || args == nullptr || args_size == 0
        || args_size > kMaximumHostArgumentBytes
        || placeholder_count > kMaximumPlaceholders
        || (placeholder_count > 0 && placeholders == nullptr)) {
        return;
    }
    const uint64_t sequence = g_sequence.fetch_add(1) + 1;
    char args_path[PATH_MAX];
    char placeholders_path[PATH_MAX];
    if (!BuildRecordPath(args_path, sizeof(args_path), directory, sequence,
                         "bin")
        || !BuildRecordPath(placeholders_path, sizeof(placeholders_path),
                            directory, sequence, "placeholders")) {
        std::fprintf(stderr, "AddRmsNorm launch record path is too long\n");
        return;
    }
    if (!WriteExclusiveFile(args_path, args, args_size)
        || !WriteExclusiveFile(
            placeholders_path, placeholders,
            placeholder_count * sizeof(aclrtPlaceHolderInfo))) {
        return;
    }
    const char* args_name = std::strrchr(args_path, '/');
    args_name = args_name == nullptr ? args_path : args_name + 1;
    const char* placeholders_name = std::strrchr(placeholders_path, '/');
    placeholders_name = placeholders_name == nullptr
        ? placeholders_path : placeholders_name + 1;
    const StubIdentity stub_identity = ResolveStubIdentity(stub);
    char object_base[32] = "none";
    char object_offset[32] = "none";
    if (std::strcmp(stub_identity.elf_membership, "mapped_elf") == 0) {
        std::snprintf(
            object_base, sizeof(object_base), "%" PRIuPTR,
            stub_identity.elf_object_base);
        std::snprintf(
            object_offset, sizeof(object_offset), "%" PRIuPTR,
            stub_identity.elf_object_offset);
    }
    char symbol_address[32] = "none";
    char symbol_offset[32] = "none";
    if (std::strcmp(stub_identity.elf_symbol, "none") != 0) {
        std::snprintf(
            symbol_address, sizeof(symbol_address), "%" PRIuPTR,
            stub_identity.elf_symbol_address);
        std::snprintf(
            symbol_offset, sizeof(symbol_offset), "%" PRIuPTR,
            stub_identity.elf_symbol_offset);
    }
    char metadata[8192];
    const int length = std::snprintf(
        metadata, sizeof(metadata),
        "schema_version=2\nrecord_type=launch_args\nsequence=%" PRIu64
        "\napi=%s\nexecute_api=%s\npid=%d\ntid=%ld"
        "\nstub_identity_kind=%s\nstub_address=%" PRIuPTR
        "\nstub_runtime_name_status=%d\nstub_runtime_function_name=%s"
        "\nstub_elf_membership=%s\nstub_elf_object_path=%s"
        "\nstub_elf_object_base=%s"
        "\nstub_elf_object_offset=%s"
        "\nstub_elf_symbol=%s\nstub_elf_symbol_address=%s"
        "\nstub_elf_symbol_offset=%s"
        "\nblock_dim=%u\nargs_size=%zu\nargs_file=%s"
        "\nplaceholder_count=%zu\nplaceholder_entry_size=%zu"
        "\nplaceholder_file=%s\nexecute_stream=%" PRIuPTR
        "\nlaunch_stream=%" PRIuPTR "\n",
        sequence, launch_api, g_active_execute_api,
        static_cast<int>(getpid()), syscall(SYS_gettid), stub_identity.kind,
        reinterpret_cast<uintptr_t>(stub),
        static_cast<int>(stub_identity.runtime_name_status),
        stub_identity.runtime_function_name, stub_identity.elf_membership,
        stub_identity.elf_object_path, object_base, object_offset,
        stub_identity.elf_symbol, symbol_address, symbol_offset,
        block_dim, args_size, args_name, placeholder_count,
        sizeof(aclrtPlaceHolderInfo), placeholders_name,
        g_active_execute_stream, reinterpret_cast<uintptr_t>(stream));
    if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(metadata)) {
        std::fprintf(stderr, "AddRmsNorm launch metadata overflow\n");
        return;
    }
    WriteMetadata(sequence, metadata);
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
    if (g_active_execute_api != nullptr) {
        CaptureLaunchArguments(
            "aclrtLaunchKernelWithHostArgs", function, block_dim, stream,
            host_args, args_size, placeholders, placeholder_count);
        return ACL_SUCCESS;
    }
    return original(function, block_dim, stream, configuration, host_args,
                    args_size, placeholders, placeholder_count);
}

extern "C" aclError aclrtLaunchKernel(
    aclrtFuncHandle function, uint32_t block_dim, const void* args,
    std::size_t args_size, aclrtStream stream) {
    using Original = decltype(&aclrtLaunchKernel);
    static Original original = ResolveLoadedLibrary<Original>(
        "libacl_rt.so", "aclrtLaunchKernel");
    if (g_active_execute_api != nullptr) {
        std::fprintf(
            stderr, "AddRmsNorm tiling capture rejects active %s launch\n",
            "aclrtLaunchKernel");
        std::abort();
    }
    return original(function, block_dim, args, args_size, stream);
}

extern "C" aclError aclrtLaunchKernelV2(
    aclrtFuncHandle function, uint32_t block_dim, const void* args,
    std::size_t args_size, aclrtLaunchKernelCfg* configuration,
    aclrtStream stream) {
    using Original = decltype(&aclrtLaunchKernelV2);
    static Original original = ResolveLoadedLibrary<Original>(
        "libacl_rt.so", "aclrtLaunchKernelV2");
    if (g_active_execute_api != nullptr) {
        std::fprintf(
            stderr, "AddRmsNorm tiling capture rejects active %s launch\n",
            "aclrtLaunchKernelV2");
        std::abort();
    }
    return original(
        function, block_dim, args, args_size, configuration, stream);
}

extern "C" aclError aclrtLaunchKernelWithConfig(
    aclrtFuncHandle function, uint32_t block_dim, aclrtStream stream,
    aclrtLaunchKernelCfg* configuration, aclrtArgsHandle args_handle,
    void* reserve) {
    using Original = decltype(&aclrtLaunchKernelWithConfig);
    static Original original = ResolveLoadedLibrary<Original>(
        "libacl_rt.so", "aclrtLaunchKernelWithConfig");
    if (g_active_execute_api != nullptr) {
        std::fprintf(
            stderr, "AddRmsNorm tiling capture rejects active %s launch\n",
            "aclrtLaunchKernelWithConfig");
        std::abort();
    }
    return original(
        function, block_dim, stream, configuration, args_handle, reserve);
}

extern "C" aclnnStatus aclnnAddRmsNorm(
    void* workspace, uint64_t workspace_size, aclOpExecutor* executor,
    aclrtStream stream) {
    using Original = decltype(&aclnnAddRmsNorm);
    static Original original = ResolveOriginalExecute<Original>();
    if (g_active_execute_api != nullptr) {
        std::fprintf(stderr, "AddRmsNorm tiling capture rejects nested execute\n");
        std::abort();
    }
    CaptureWorkspace("aclnnAddRmsNorm", workspace_size, stream);
    g_active_execute_api = "aclnnAddRmsNorm";
    g_active_execute_stream = reinterpret_cast<uintptr_t>(stream);
    const aclnnStatus status = original(
        workspace, workspace_size, executor, stream);
    g_active_execute_api = nullptr;
    g_active_execute_stream = 0;
    return status;
}

extern "C" void* dlsym(void* handle, const char* symbol) {
    void* original = RealDlsym()(handle, symbol);
    if (CaptureDirectory() != nullptr && original != nullptr) {
        if (std::strcmp(symbol, "aclrtLaunchKernelWithHostArgs") == 0) {
            return reinterpret_cast<void*>(&aclrtLaunchKernelWithHostArgs);
        }
        if (std::strcmp(symbol, "aclrtLaunchKernel") == 0) {
            return reinterpret_cast<void*>(&aclrtLaunchKernel);
        }
        if (std::strcmp(symbol, "aclrtLaunchKernelV2") == 0) {
            return reinterpret_cast<void*>(&aclrtLaunchKernelV2);
        }
        if (std::strcmp(symbol, "aclrtLaunchKernelWithConfig") == 0) {
            return reinterpret_cast<void*>(&aclrtLaunchKernelWithConfig);
        }
        if (std::strcmp(symbol, "aclnnAddRmsNorm") == 0) {
            RememberOriginalExecute(original);
            return reinterpret_cast<void*>(&aclnnAddRmsNorm);
        }
        constexpr const char* kLaunchPrefix = "aclrtLaunchKernel";
        if (g_active_execute_api != nullptr
            && std::strncmp(
                symbol, kLaunchPrefix, std::strlen(kLaunchPrefix)) == 0) {
            std::fprintf(
                stderr,
                "AddRmsNorm tiling capture rejects unhandled active launch API %s\n",
                symbol);
            std::abort();
        }
    }
    return original;
}
