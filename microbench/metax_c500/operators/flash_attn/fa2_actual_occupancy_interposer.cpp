/*
 * Dry-launch interposer for source-selected MetaX FlashAttention kernels.
 *
 * While the explicit marker is active, it records each selected function's
 * launch geometry plus mcFuncGetAttributes and
 * mcOccupancyMaxActiveBlocksPerMultiprocessor, then returns success without
 * submitting the device kernel.  It therefore captures static resource and
 * runtime-admission facts only; it records and consumes no duration.
 */

#include <mc_runtime.h>

#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct LaunchRecord {
    std::string api;
    std::string identity;
    std::string identity_source;
    unsigned int grid_x;
    unsigned int grid_y;
    unsigned int grid_z;
    unsigned int block_x;
    unsigned int block_y;
    unsigned int block_z;
    unsigned int dynamic_shared_memory_bytes;
    int registers_per_thread;
    std::size_t static_shared_memory_bytes;
    int resident_ctas_per_ap;
};

std::mutex state_mutex;
bool marker_active = false;
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
        std::fprintf(stderr, "FA occupancy interposer cannot resolve %s: %s\\n",
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

std::pair<std::string, std::string> resolve_identity_locked(
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
    set_failure_locked("captured launch has no attributable kernel identity: "
                       + pointer_text(function));
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
        using GetAttributesFunction = mcError_t (*)(mcFuncAttributes*,
                                                    const void*);
        using OccupancyFunction = mcError_t (*)(int*, const void*, int,
                                                std::size_t);
        static const GetAttributesFunction get_attributes =
            require_next<GetAttributesFunction>("mcFuncGetAttributes");
        static const OccupancyFunction occupancy =
            require_next<OccupancyFunction>(
                "mcOccupancyMaxActiveBlocksPerMultiprocessor");
        mcFuncAttributes attributes {};
        const mcError_t attribute_status = get_attributes(&attributes, function);
        int resident = 0;
        const mcError_t occupancy_status = occupancy(
            &resident, function,
            static_cast<int>(block_x * block_y * block_z),
            static_cast<std::size_t>(dynamic_shared_memory_bytes));
        if (attribute_status != mcSuccess || occupancy_status != mcSuccess
                || attributes.numRegs <= 0 || resident <= 0) {
            set_failure_locked(
                "runtime resource query failed for " + identity.first
                + ": attributes=" + std::to_string(attribute_status)
                + ", occupancy=" + std::to_string(occupancy_status)
                + ", registers=" + std::to_string(attributes.numRegs)
                + ", resident=" + std::to_string(resident));
        }
        launches.push_back({
            api, identity.first, identity.second,
            grid_x, grid_y, grid_z, block_x, block_y, block_z,
            dynamic_shared_memory_bytes, attributes.numRegs,
            attributes.sharedSizeBytes, resident,
        });
    } catch (const std::exception& error) {
        set_failure_locked(std::string("cannot record launch: ") + error.what());
    } catch (...) {
        set_failure_locked("cannot record launch: unknown failure");
    }
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
            "\"registers_per_thread\":%d,"
            "\"static_shared_memory_bytes\":%zu,"
            "\"resident_ctas_per_ap\":%d}\n",
            launch.grid_x, launch.grid_y, launch.grid_z,
            launch.block_x, launch.block_y, launch.block_z,
            launch.dynamic_shared_memory_bytes,
            launch.registers_per_thread, launch.static_shared_memory_bytes,
            launch.resident_ctas_per_ap);
    }
    std::fprintf(output, "{\"record_type\":\"summary\","
                 "\"captured_launch_count\":%zu}\n", launches.size());
}

}  // namespace

extern "C" int midlayer_fa_occupancy_begin() {
    std::lock_guard<std::mutex> guard(state_mutex);
    if (marker_active) {
        failure = "occupancy marker is already active";
        return -1;
    }
    const char* path = std::getenv("MIDLAYER_C500_FA_OCCUPANCY_CAPTURE");
    if (path == nullptr || path[0] == '\0') {
        failure = "MIDLAYER_C500_FA_OCCUPANCY_CAPTURE is required";
        return -1;
    }
    output = std::fopen(path, "w");
    if (output == nullptr) {
        failure = std::string("cannot open occupancy capture: ") + path;
        return -1;
    }
    launches.clear();
    failure.clear();
    marker_active = true;
    return 0;
}

extern "C" int midlayer_fa_occupancy_end() {
    std::lock_guard<std::mutex> guard(state_mutex);
    if (!marker_active) {
        failure = "occupancy marker is not active";
        return -1;
    }
    marker_active = false;
    if (failure.empty()) {
        write_capture_locked();
    }
    const int close_status = std::fclose(output);
    output = nullptr;
    if (close_status != 0) {
        set_failure_locked("cannot close occupancy capture");
    }
    return failure.empty() ? 0 : -1;
}

extern "C" const char* midlayer_fa_occupancy_error() {
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
        module_function_names[reinterpret_cast<std::uintptr_t>(*function)] = name;
    }
    return status;
}

extern "C" mcError_t mcLaunchKernel(
        const void* function, dim3 grid, dim3 block, void** arguments,
        std::size_t dynamic_shared_memory_bytes, mcStream_t stream) {
    if (capture_if_active("mcLaunchKernel", function, grid.x, grid.y, grid.z,
                          block.x, block.y, block.z,
                          static_cast<unsigned int>(dynamic_shared_memory_bytes))) {
        return mcSuccess;
    }
    using RealFunction = mcError_t (*)(const void*, dim3, dim3, void**,
                                        std::size_t, mcStream_t);
    static const RealFunction real = require_next<RealFunction>("mcLaunchKernel");
    return real(function, grid, block, arguments, dynamic_shared_memory_bytes,
                stream);
}

extern "C" mcError_t mcModuleLaunchKernel(
        mcFunction_t function,
        unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
        unsigned int block_x, unsigned int block_y, unsigned int block_z,
        unsigned int dynamic_shared_memory_bytes, mcStream_t stream,
        void** kernel_parameters, void** extra) {
    if (capture_if_active("mcModuleLaunchKernel",
                          reinterpret_cast<const void*>(function),
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
    return real(function, grid_x, grid_y, grid_z, block_x, block_y, block_z,
                dynamic_shared_memory_bytes, stream, kernel_parameters, extra);
}
