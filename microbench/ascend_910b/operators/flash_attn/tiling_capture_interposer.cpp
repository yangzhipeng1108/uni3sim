// Capture non-duration FusedInferAttentionScore launch facts from CANN 8.5.1.
//
// The CANN 8.5.1 OpAPI executor exposes its serialized tiling payload through
// NnopbaseGetTilingData.  This interposer captures that exact payload before a
// FusedInferAttentionScore execute call.  When the eager path also launches
// through aclrtLaunchKernelWithHostArgs, its complete host argument buffer and
// placeholder map are retained as an independent byte-for-byte cross-check.

#include <acl/acl_rt.h>
#include <aclnn/acl_meta.h>

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
    "MIDLAYER_ASCEND_TILING_CAPTURE_DIR";
constexpr const char* kRuntimeVariantEnvironment =
    "MIDLAYER_ASCEND_RUNTIME_VARIANT";
// These offsets are version-bound CANN 8.5.1 host-args candidates.  The live
// aclrtPlaceHolderInfo map must independently contain the exact 280 -> 296
// mapping before the runner accepts the serialized-data start.
constexpr uint32_t kCandidateTilingAddressOffset = 280;
constexpr uint32_t kCandidateTilingDataOffset = 296;
constexpr uint32_t kMaximumMetadataParameterBytes = 3008;
constexpr std::size_t kMaximumPlaceholders = 64;

static_assert(sizeof(aclrtPlaceHolderInfo) == 2 * sizeof(uint32_t));

constexpr const char* kRuntimeVariants[] = {
    "103000000010022221",
    "103000000010222221",
    "103000000010422221",
    "103000000010522221",
    "103000000010622221",
    "105000000000222220",
    "105000000000322220",
    "5000000000000200203",
    "5000000000010200203",
};

std::atomic<uint64_t> g_sequence{0};
std::atomic<void*> g_original_aclnnFusedInferAttentionScore{nullptr};
std::atomic<void*> g_original_aclnnFusedInferAttentionScoreV2{nullptr};
std::atomic<void*> g_original_aclnnFusedInferAttentionScoreV3{nullptr};
std::atomic<void*> g_original_aclnnFusedInferAttentionScoreV4{nullptr};
thread_local const char* g_active_execute_api = nullptr;

bool IsSelectedRuntimeVariant(const char* variant) {
    if (variant == nullptr) {
        return false;
    }
    for (const char* selected : kRuntimeVariants) {
        if (std::strcmp(variant, selected) == 0) {
            return true;
        }
    }
    return false;
}

const char* RuntimeVariant() {
    const char* variant = std::getenv(kRuntimeVariantEnvironment);
    if (!IsSelectedRuntimeVariant(variant)) {
        return nullptr;
    }
    return variant;
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
    // glibc 2.34 moved libdl entry points into libc while retaining the 2.17
    // compatibility symbol used by CANN.  dlvsym is used deliberately here:
    // calling dlsym while interposing dlsym would recurse before the real
    // resolver had been found.
    static DlsymFunction function = [] {
        DlsymFunction resolved = ResolveVersionedSymbol<DlsymFunction>(
            RTLD_NEXT, "dlsym", "GLIBC_2.34");
        if (resolved == nullptr) {
            resolved = ResolveVersionedSymbol<DlsymFunction>(
                RTLD_NEXT, "dlsym", "GLIBC_2.17");
        }
        if (resolved == nullptr) {
            std::fprintf(stderr,
                         "tiling capture cannot resolve the real dlsym\n");
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
        std::fprintf(stderr, "tiling capture cannot resolve %s: %s\n", symbol,
                     error == nullptr ? "unknown error" : error);
        std::abort();
    }
    return reinterpret_cast<Function>(address);
}

template <typename Function>
Function ResolveLoadedLibrary(const char* library, const char* symbol) {
    void* handle = dlopen(library, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    if (handle == nullptr) {
        std::fprintf(stderr,
                     "tiling capture cannot find loaded library %s: %s\n",
                     library, dlerror());
        std::abort();
    }
    dlerror();
    void* address = RealDlsym()(handle, symbol);
    const char* error = dlerror();
    if (address == nullptr || error != nullptr) {
        std::fprintf(stderr,
                     "tiling capture cannot resolve %s from %s: %s\n",
                     symbol, library,
                     error == nullptr ? "unknown error" : error);
        std::abort();
    }
    return reinterpret_cast<Function>(address);
}

void RememberOriginalExecute(std::atomic<void*>& destination,
                             void* address, const char* symbol) {
    if (address == nullptr) {
        return;
    }
    void* expected = nullptr;
    if (!destination.compare_exchange_strong(
            expected, address, std::memory_order_release,
            std::memory_order_acquire)
        && expected != address) {
        std::fprintf(stderr,
                     "tiling capture saw conflicting %s implementations\n",
                     symbol);
        std::abort();
    }
}

template <typename Function>
Function ResolveOriginalExecute(std::atomic<void*>& source,
                                const char* symbol) {
    void* address = source.load(std::memory_order_acquire);
    if (address != nullptr) {
        return reinterpret_cast<Function>(address);
    }
    // Direct ELF calls do not pass through the dlsym hook.  Preserve that
    // valid path while requiring dlsym-loaded RTLD_LOCAL implementations to
    // have been remembered with their original handle semantics.
    return ResolveNext<Function>(symbol);
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
    const long thread_id = syscall(SYS_gettid);
    const int length = std::snprintf(
        destination, capacity, "%s/capture_%d_%ld_%06" PRIu64 ".%s",
        directory, static_cast<int>(getpid()), thread_id, sequence, suffix);
    return length > 0 && static_cast<std::size_t>(length) < capacity;
}

void WriteMetadata(uint64_t sequence, const char* text) {
    const char* directory = std::getenv(kCaptureDirectoryEnvironment);
    if (directory == nullptr || *directory == '\0') {
        return;
    }
    char path[PATH_MAX];
    if (!BuildRecordPath(path, sizeof(path), directory, sequence, "meta")) {
        std::fprintf(stderr, "tiling capture metadata path is too long\n");
        return;
    }
    const int descriptor = open(
        path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (descriptor < 0) {
        std::fprintf(stderr, "tiling capture cannot create %s: %s\n", path,
                     std::strerror(errno));
        return;
    }
    const bool complete = WriteAll(descriptor, text, std::strlen(text));
    const int close_result = close(descriptor);
    if (!complete || close_result != 0) {
        std::fprintf(stderr, "tiling capture cannot finish %s\n", path);
    }
}

void CaptureWorkspace(const char* api, uint64_t workspace_size) {
    const char* variant = RuntimeVariant();
    if (variant == nullptr) {
        return;
    }
    const uint64_t sequence = g_sequence.fetch_add(1) + 1;
    char metadata[1024];
    const int length = std::snprintf(
        metadata, sizeof(metadata),
        "schema_version=2\nrecord_type=workspace\nsequence=%" PRIu64
        "\napi=%s\npid=%d\ntid=%ld\nruntime_variant=%s"
        "\nworkspace_size=%" PRIu64 "\n",
        sequence, api, static_cast<int>(getpid()), syscall(SYS_gettid),
        variant, workspace_size);
    if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(metadata)) {
        std::fprintf(stderr, "tiling capture workspace metadata overflow\n");
        return;
    }
    WriteMetadata(sequence, metadata);
}

void CaptureExecutorTiling(const char* execute_api, aclOpExecutor* executor) {
    const char* directory = std::getenv(kCaptureDirectoryEnvironment);
    const char* variant = RuntimeVariant();
    if (directory == nullptr || *directory == '\0' || variant == nullptr) {
        return;
    }
    if (executor == nullptr) {
        std::fprintf(stderr, "tiling capture received a null executor\n");
        std::abort();
    }
    using GetTilingData = int32_t (*)(
        aclOpExecutor*, uint8_t**, std::size_t*);
    static GetTilingData get_tiling_data = ResolveLoadedLibrary<GetTilingData>(
        "libnnopbase.so", "NnopbaseGetTilingData");
    uint8_t* tiling_data = nullptr;
    std::size_t tiling_size = 0;
    const int32_t status = get_tiling_data(
        executor, &tiling_data, &tiling_size);
    if (status != 0 || tiling_data == nullptr || tiling_size == 0
        || tiling_size > kMaximumMetadataParameterBytes) {
        std::fprintf(
            stderr,
            "NnopbaseGetTilingData returned invalid payload: status=%d, "
            "pointer=%p, size=%zu\n",
            status, static_cast<void*>(tiling_data), tiling_size);
        std::abort();
    }

    const uint64_t sequence = g_sequence.fetch_add(1) + 1;
    char binary_path[PATH_MAX];
    if (!BuildRecordPath(
            binary_path, sizeof(binary_path), directory, sequence, "tiling")) {
        std::fprintf(stderr, "executor tiling path is too long\n");
        std::abort();
    }
    const int descriptor = open(
        binary_path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (descriptor < 0) {
        std::fprintf(stderr, "cannot create %s: %s\n", binary_path,
                     std::strerror(errno));
        std::abort();
    }
    const bool complete = WriteAll(descriptor, tiling_data, tiling_size);
    const int close_result = close(descriptor);
    if (!complete || close_result != 0) {
        std::fprintf(stderr, "cannot finish %s\n", binary_path);
        std::abort();
    }
    const char* binary_name = std::strrchr(binary_path, '/');
    binary_name = binary_name == nullptr ? binary_path : binary_name + 1;
    char metadata[1536];
    const int length = std::snprintf(
        metadata, sizeof(metadata),
        "schema_version=2\nrecord_type=executor_tiling\nsequence=%" PRIu64
        "\napi=NnopbaseGetTilingData\nexecute_api=%s\npid=%d\ntid=%ld"
        "\nruntime_variant=%s\ntiling_size=%zu\ntiling_file=%s\n",
        sequence, execute_api, static_cast<int>(getpid()), syscall(SYS_gettid),
        variant, tiling_size, binary_name);
    if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(metadata)) {
        std::fprintf(stderr, "executor tiling metadata overflow\n");
        std::abort();
    }
    WriteMetadata(sequence, metadata);
}

const char* SafeStubSymbol(const void* stub, char* storage,
                           std::size_t storage_size) {
    Dl_info information{};
    if (dladdr(stub, &information) == 0 || information.dli_sname == nullptr) {
        return "unavailable";
    }
    const std::string symbol(information.dli_sname);
    if (symbol.empty() || symbol.size() + 1 > storage_size
        || symbol.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
            != std::string::npos) {
        return "unavailable";
    }
    std::memcpy(storage, symbol.c_str(), symbol.size() + 1);
    return storage;
}

void CaptureLaunchArguments(const char* launch_api, const char* execute_api,
                            const void* stub,
                            uint32_t block_dim, const void* args,
                            std::size_t args_size,
                            const aclrtPlaceHolderInfo* placeholders,
                            std::size_t placeholder_count) {
    const char* directory = std::getenv(kCaptureDirectoryEnvironment);
    const char* variant = RuntimeVariant();
    if (directory == nullptr || *directory == '\0' || variant == nullptr
        || args == nullptr || args_size <= kCandidateTilingDataOffset
        || placeholder_count > kMaximumPlaceholders
        || (placeholder_count > 0 && placeholders == nullptr)) {
        return;
    }
    const uint64_t sequence = g_sequence.fetch_add(1) + 1;
    char binary_path[PATH_MAX];
    if (!BuildRecordPath(
            binary_path, sizeof(binary_path), directory, sequence, "bin")) {
        std::fprintf(stderr, "tiling capture argument path is too long\n");
        return;
    }
    const int descriptor = open(
        binary_path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (descriptor < 0) {
        std::fprintf(stderr, "tiling capture cannot create %s: %s\n",
                     binary_path, std::strerror(errno));
        return;
    }
    const bool complete = WriteAll(descriptor, args, args_size);
    const int close_result = close(descriptor);
    if (!complete || close_result != 0) {
        std::fprintf(stderr, "tiling capture cannot finish %s\n", binary_path);
        return;
    }
    char placeholder_path[PATH_MAX];
    if (!BuildRecordPath(placeholder_path, sizeof(placeholder_path), directory,
                         sequence, "placeholders")) {
        std::fprintf(stderr, "tiling capture placeholder path is too long\n");
        return;
    }
    const int placeholder_descriptor = open(
        placeholder_path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (placeholder_descriptor < 0) {
        std::fprintf(stderr, "tiling capture cannot create %s: %s\n",
                     placeholder_path, std::strerror(errno));
        return;
    }
    const bool placeholder_complete = WriteAll(
        placeholder_descriptor, placeholders,
        placeholder_count * sizeof(aclrtPlaceHolderInfo));
    const int placeholder_close_result = close(placeholder_descriptor);
    if (!placeholder_complete || placeholder_close_result != 0) {
        std::fprintf(stderr, "tiling capture cannot finish %s\n",
                     placeholder_path);
        return;
    }
    const char* binary_name = std::strrchr(binary_path, '/');
    binary_name = binary_name == nullptr ? binary_path : binary_name + 1;
    const char* placeholder_name = std::strrchr(placeholder_path, '/');
    placeholder_name = placeholder_name == nullptr
        ? placeholder_path : placeholder_name + 1;
    uint64_t candidate_tiling_address = 0;
    std::memcpy(&candidate_tiling_address,
                static_cast<const uint8_t*>(args)
                    + kCandidateTilingAddressOffset,
                sizeof(candidate_tiling_address));
    char symbol_storage[512];
    const char* stub_symbol = SafeStubSymbol(
        stub, symbol_storage, sizeof(symbol_storage));
    char metadata[2048];
    const int length = std::snprintf(
        metadata, sizeof(metadata),
        "schema_version=2\nrecord_type=launch_args\nsequence=%" PRIu64
        "\napi=%s\nexecute_api=%s\npid=%d\ntid=%ld"
        "\nruntime_variant=%s\nmetadata_tiling_key=unavailable"
        "\nstub_symbol=%s\nblock_dim=%u\nargs_size=%zu"
        "\ncandidate_tiling_address_offset=%u"
        "\ncandidate_tiling_data_offset=%u"
        "\ncandidate_tiling_pointer_slot_value=%" PRIu64
        "\nargs_file=%s\nplaceholder_count=%zu"
        "\nplaceholder_entry_size=%zu\nplaceholder_file=%s\n",
        sequence, launch_api, execute_api,
        static_cast<int>(getpid()), syscall(SYS_gettid), variant, stub_symbol,
        block_dim, args_size,
        kCandidateTilingAddressOffset, kCandidateTilingDataOffset,
        candidate_tiling_address, binary_name, placeholder_count,
        sizeof(aclrtPlaceHolderInfo), placeholder_name);
    if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(metadata)) {
        std::fprintf(stderr, "tiling capture argument metadata overflow\n");
        return;
    }
    WriteMetadata(sequence, metadata);
}

template <typename Original>
aclnnStatus ExecuteAndCapture(const char* api, Original original,
                              void* workspace, uint64_t workspace_size,
                              aclOpExecutor* executor, aclrtStream stream) {
    if (g_active_execute_api != nullptr) {
        std::fprintf(stderr, "tiling capture rejects nested FA execution\n");
        std::abort();
    }
    CaptureWorkspace(api, workspace_size);
    CaptureExecutorTiling(api, executor);
    g_active_execute_api = api;
    const aclnnStatus status = original(
        workspace, workspace_size, executor, stream);
    g_active_execute_api = nullptr;
    return status;
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
    const char* execute_api = g_active_execute_api;
    if (execute_api != nullptr) {
        CaptureLaunchArguments(
            "aclrtLaunchKernelWithHostArgs", execute_api, function, block_dim,
            host_args,
            args_size, placeholders, placeholder_count);
    }
    return original(function, block_dim, stream, configuration, host_args,
                    args_size, placeholders, placeholder_count);
}

#define DEFINE_FA_EXECUTE_WRAPPER(name)                                           \
    extern "C" aclnnStatus name(                                                 \
        void* workspace, uint64_t workspace_size, aclOpExecutor* executor,        \
        const aclrtStream stream) {                                                \
        using Original = decltype(&name);                                          \
        static Original original = ResolveOriginalExecute<Original>(               \
            g_original_##name, #name);                                              \
        return ExecuteAndCapture(                                                  \
            #name, original, workspace, workspace_size, executor, stream);         \
    }

DEFINE_FA_EXECUTE_WRAPPER(aclnnFusedInferAttentionScore)
DEFINE_FA_EXECUTE_WRAPPER(aclnnFusedInferAttentionScoreV2)
DEFINE_FA_EXECUTE_WRAPPER(aclnnFusedInferAttentionScoreV3)
DEFINE_FA_EXECUTE_WRAPPER(aclnnFusedInferAttentionScoreV4)

#undef DEFINE_FA_EXECUTE_WRAPPER

extern "C" void* dlsym(void* handle, const char* symbol) {
    // torch_npu's OpAPI bridge obtains ACLNN entry points with dlsym on a
    // libopapi handle, so ordinary ELF preload interposition does not see the
    // execute call.  Redirect only the four audited FA execute entry points;
    // every other lookup retains the caller's original handle semantics.
    void* original = RealDlsym()(handle, symbol);
    if (RuntimeVariant() != nullptr && original != nullptr) {
        if (std::strcmp(symbol, "aclnnFusedInferAttentionScore") == 0) {
            RememberOriginalExecute(
                g_original_aclnnFusedInferAttentionScore, original, symbol);
            return reinterpret_cast<void*>(&aclnnFusedInferAttentionScore);
        }
        if (std::strcmp(symbol, "aclnnFusedInferAttentionScoreV2") == 0) {
            RememberOriginalExecute(
                g_original_aclnnFusedInferAttentionScoreV2, original, symbol);
            return reinterpret_cast<void*>(&aclnnFusedInferAttentionScoreV2);
        }
        if (std::strcmp(symbol, "aclnnFusedInferAttentionScoreV3") == 0) {
            RememberOriginalExecute(
                g_original_aclnnFusedInferAttentionScoreV3, original, symbol);
            return reinterpret_cast<void*>(&aclnnFusedInferAttentionScoreV3);
        }
        if (std::strcmp(symbol, "aclnnFusedInferAttentionScoreV4") == 0) {
            RememberOriginalExecute(
                g_original_aclnnFusedInferAttentionScoreV4, original, symbol);
            return reinterpret_cast<void*>(&aclnnFusedInferAttentionScoreV4);
        }
    }
    return original;
}
