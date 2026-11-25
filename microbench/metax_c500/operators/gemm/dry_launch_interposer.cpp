/*
 * LD_PRELOAD interposer for host-only mcBLAS selector observation.
 *
 * Initialization launches outside the explicit marker are forwarded.  Inside
 * the marker, the two MXMACA launch APIs used by mcBLAS are recorded and
 * return success without submitting device work.  Module function identities
 * are retained from mcModuleGetFunction.  Runtime-created opaque function
 * values are resolved through mcFuncGetName so every launch remains
 * attributable at launch time.
 */

#include <mc_runtime.h>

#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>


namespace {

struct LaunchRecord {
    const char* api;
    std::string identity;
    const char* identity_source;
    unsigned int grid_x;
    unsigned int grid_y;
    unsigned int grid_z;
    unsigned int block_x;
    unsigned int block_y;
    unsigned int block_z;
    unsigned int dynamic_shared_memory_bytes;
};

std::mutex state_mutex;
bool marker_active = false;
std::size_t expected_launches = 0;
std::vector<LaunchRecord> launches;
std::unordered_map<std::uintptr_t, std::string> module_function_names;
std::string failure;
FILE* output = nullptr;

template <typename Function>
Function require_next(const char* name) {
    dlerror();
    void* symbol = dlsym(RTLD_NEXT, name);
    const char* error = dlerror();
    if (error != nullptr || symbol == nullptr) {
        std::fprintf(stderr, "dry-launch interposer cannot resolve %s: %s\n",
                     name, error == nullptr ? "null symbol" : error);
        std::_Exit(EXIT_FAILURE);
    }
    return reinterpret_cast<Function>(symbol);
}

void set_failure_locked(const std::string& message) {
    if (failure.empty()) {
        failure = message;
    }
}

std::string pointer_text(const void* pointer) {
    char text[2 + sizeof(std::uintptr_t) * 2 + 1] {};
    std::snprintf(text, sizeof(text), "0x%llx",
                  static_cast<unsigned long long>(
                      reinterpret_cast<std::uintptr_t>(pointer)));
    return text;
}

std::pair<std::string, const char*> resolve_identity_locked(
        const void* function) {
    const auto key = reinterpret_cast<std::uintptr_t>(function);
    const auto mapped = module_function_names.find(key);
    if (mapped != module_function_names.end()) {
        return {mapped->second, "mcModuleGetFunction"};
    }
    using GetNameFunction = mcError_t (*)(const char**, const void*);
    static const GetNameFunction get_name =
        require_next<GetNameFunction>("mcFuncGetName");
    const char* runtime_name = nullptr;
    if (get_name(&runtime_name, function) == mcSuccess
            && runtime_name != nullptr && runtime_name[0] != '\0') {
        return {runtime_name, "mcFuncGetName"};
    }
    Dl_info info {};
    if (dladdr(function, &info) != 0 && info.dli_sname != nullptr
            && info.dli_sname[0] != '\0') {
        return {info.dli_sname, "dladdr"};
    }
    set_failure_locked(
        "captured launch has no kernel identity from mcModuleGetFunction, "
        "mcFuncGetName, or dladdr: " + pointer_text(function));
    return {pointer_text(function), "unresolved"};
}

bool capture_if_active(
        const char* api, const void* function,
        unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
        unsigned int block_x, unsigned int block_y, unsigned int block_z,
        unsigned int dynamic_shared_memory_bytes) {
    std::lock_guard<std::mutex> guard(state_mutex);
    if (!marker_active) {
        return false;
    }
    try {
        const auto identity = resolve_identity_locked(function);
        launches.push_back({
            api, identity.first, identity.second,
            grid_x, grid_y, grid_z,
            block_x, block_y, block_z,
            dynamic_shared_memory_bytes,
        });
    } catch (const std::exception& error) {
        set_failure_locked(
            std::string("cannot retain captured launch: ") + error.what());
    } catch (...) {
        set_failure_locked("cannot retain captured launch: unknown failure");
    }
    return true;
}

bool reject_if_active(const char* api) {
    std::lock_guard<std::mutex> guard(state_mutex);
    if (!marker_active) {
        return false;
    }
    set_failure_locked(
        std::string("unsupported device execution entry point in active ")
        + "dry-launch marker: " + api);
    return true;
}

void write_json_string(FILE* stream, const std::string& value) {
    std::fputc('"', stream);
    for (const unsigned char character : value) {
        if (character == '"' || character == '\\') {
            std::fputc('\\', stream);
            std::fputc(character, stream);
        } else if (character == '\n') {
            std::fputs("\\n", stream);
        } else if (character == '\r') {
            std::fputs("\\r", stream);
        } else if (character == '\t') {
            std::fputs("\\t", stream);
        } else if (character < 0x20) {
            std::fprintf(stream, "\\u%04x", character);
        } else {
            std::fputc(character, stream);
        }
    }
    std::fputc('"', stream);
}

void write_capture_locked() {
    for (std::size_t ordinal = 0; ordinal < launches.size(); ++ordinal) {
        const auto& launch = launches[ordinal];
        std::fprintf(output, "{\"record_type\":\"launch\",\"ordinal\":%zu,",
                     ordinal + 1);
        std::fputs("\"api\":", output);
        write_json_string(output, launch.api);
        std::fputs(",\"kernel_identity\":", output);
        write_json_string(output, launch.identity);
        std::fputs(",\"identity_source\":", output);
        write_json_string(output, launch.identity_source);
        std::fprintf(
            output,
            ",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],"
            "\"dynamic_shared_memory_bytes\":%u,"
            "\"static_shared_memory_bytes\":null,"
            "\"static_shared_memory_status\":"
            "\"not_exposed_by_launch_api\"}\n",
            launch.grid_x, launch.grid_y, launch.grid_z,
            launch.block_x, launch.block_y, launch.block_z,
            launch.dynamic_shared_memory_bytes);
    }
    std::fprintf(
        output,
        "{\"record_type\":\"summary\",\"expected_launch_count\":%zu,"
        "\"captured_launch_count\":%zu}\n",
        expected_launches, launches.size());
}

}  // namespace


extern "C" int midlayer_dry_launch_begin(std::size_t expected) {
    std::lock_guard<std::mutex> guard(state_mutex);
    if (marker_active) {
        failure = "dry-launch marker is already active";
        return -1;
    }
    if (expected == 0) {
        failure = "expected launch count must be positive";
        return -1;
    }
    const char* path = std::getenv("MIDLAYER_C500_DRY_LAUNCH_CAPTURE");
    if (path == nullptr || path[0] == '\0') {
        failure = "MIDLAYER_C500_DRY_LAUNCH_CAPTURE is required";
        return -1;
    }
    output = std::fopen(path, "w");
    if (output == nullptr) {
        failure = std::string("cannot open dry-launch capture: ") + path;
        return -1;
    }
    launches.clear();
    failure.clear();
    expected_launches = expected;
    marker_active = true;
    return 0;
}


extern "C" int midlayer_dry_launch_end() {
    std::lock_guard<std::mutex> guard(state_mutex);
    if (!marker_active) {
        failure = "dry-launch marker is not active";
        return -1;
    }
    marker_active = false;
    if (launches.size() != expected_launches) {
        set_failure_locked(
            "captured " + std::to_string(launches.size())
            + " launch(es), expected " + std::to_string(expected_launches));
    }
    if (failure.empty()) {
        write_capture_locked();
    }
    const int close_status = std::fclose(output);
    output = nullptr;
    if (close_status != 0) {
        set_failure_locked("cannot close dry-launch capture");
    }
    return failure.empty() ? 0 : -1;
}


extern "C" const char* midlayer_dry_launch_error() {
    std::lock_guard<std::mutex> guard(state_mutex);
    return failure.c_str();
}


extern "C" mcError_t mcModuleGetFunction(
        mcFunction_t* function, mcModule_t module, const char* name) {
    using RealFunction = mcError_t (*)(mcFunction_t*, mcModule_t, const char*);
    static const RealFunction real =
        require_next<RealFunction>("mcModuleGetFunction");
    const mcError_t status = real(function, module, name);
    if (status == mcSuccess && function != nullptr && *function != nullptr
            && name != nullptr && name[0] != '\0') {
        std::lock_guard<std::mutex> guard(state_mutex);
        try {
            module_function_names[reinterpret_cast<std::uintptr_t>(*function)] =
                name;
        } catch (...) {
            if (marker_active) {
                set_failure_locked("cannot retain mcModuleGetFunction mapping");
            }
        }
    }
    return status;
}


extern "C" mcError_t mcLaunchKernel(
        const void* function, dim3 grid, dim3 block, void** arguments,
        std::size_t dynamic_shared_memory_bytes, mcStream_t stream) {
    if (capture_if_active(
            "mcLaunchKernel", function,
            grid.x, grid.y, grid.z, block.x, block.y, block.z,
            static_cast<unsigned int>(dynamic_shared_memory_bytes))) {
        return mcSuccess;
    }
    using RealFunction = mcError_t (*)(
        const void*, dim3, dim3, void**, std::size_t, mcStream_t);
    static const RealFunction real = require_next<RealFunction>("mcLaunchKernel");
    return real(function, grid, block, arguments,
                dynamic_shared_memory_bytes, stream);
}


extern "C" mcError_t mcModuleLaunchKernel(
        mcFunction_t function,
        unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
        unsigned int block_x, unsigned int block_y, unsigned int block_z,
        unsigned int dynamic_shared_memory_bytes, mcStream_t stream,
        void** kernel_parameters, void** extra) {
    if (capture_if_active(
            "mcModuleLaunchKernel", reinterpret_cast<const void*>(function),
            grid_x, grid_y, grid_z, block_x, block_y, block_z,
            dynamic_shared_memory_bytes)) {
        return mcSuccess;
    }
    using RealFunction = mcError_t (*)(
        mcFunction_t, unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int, unsigned int,
        mcStream_t, void**, void**);
    static const RealFunction real =
        require_next<RealFunction>("mcModuleLaunchKernel");
    return real(function, grid_x, grid_y, grid_z,
                block_x, block_y, block_z, dynamic_shared_memory_bytes,
                stream, kernel_parameters, extra);
}


extern "C" mcError_t mcLaunchCooperativeKernel(
        const void* function, dim3 grid, dim3 block, void** arguments,
        unsigned int dynamic_shared_memory_bytes, mcStream_t stream) {
    if (reject_if_active("mcLaunchCooperativeKernel")) {
        return mcSuccess;
    }
    using RealFunction = mcError_t (*)(
        const void*, dim3, dim3, void**, unsigned int, mcStream_t);
    static const RealFunction real =
        require_next<RealFunction>("mcLaunchCooperativeKernel");
    return real(function, grid, block, arguments,
                dynamic_shared_memory_bytes, stream);
}


extern "C" mcError_t mcGraphLaunch(
        mcGraphExec_t graph, mcStream_t stream) {
    if (reject_if_active("mcGraphLaunch")) {
        return mcSuccess;
    }
    using RealFunction = mcError_t (*)(mcGraphExec_t, mcStream_t);
    static const RealFunction real = require_next<RealFunction>("mcGraphLaunch");
    return real(graph, stream);
}
