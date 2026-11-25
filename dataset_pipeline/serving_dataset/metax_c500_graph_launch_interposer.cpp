/* Evaluation-only mcGraphLaunch correlation for MetaX C500 Serving traces. */

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <mc_runtime.h>

namespace {

struct StepContext {
    bool active = false;
    std::string trace_step_id;
    int worker_rank = -1;
    std::uint64_t context_ordinal = 0;
};

std::atomic<std::uint64_t> launch_ordinal {0};
std::atomic<std::uint64_t> kernel_launch_ordinal {0};
std::mutex context_mutex;
std::mutex log_mutex;
StepContext current_context;
pid_t descriptor_process = 0;
int context_descriptor = -1;
int launch_descriptor = -1;
int kernel_launch_descriptor = -1;
int kernel_resource_descriptor = -1;
std::unordered_map<std::uintptr_t, std::string> module_function_names;
std::unordered_set<std::uintptr_t> recorded_resource_functions;

std::string context_json(const StepContext& context);

template <typename Function>
Function require_next(const char* name) {
    dlerror();
    void* symbol = dlsym(RTLD_NEXT, name);
    const char* error = dlerror();
    if (error != nullptr || symbol == nullptr) {
        std::fprintf(stderr, "C500 Graph launch interposer cannot resolve %s: %s\n",
                     name, error == nullptr ? "null symbol" : error);
        std::_Exit(EXIT_FAILURE);
    }
    return reinterpret_cast<Function>(symbol);
}

std::string required_environment(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        std::fprintf(stderr, "C500 Graph launch interposer requires %s\n", name);
        std::_Exit(EXIT_FAILURE);
    }
    return value;
}

std::uint64_t monotonic_ns() {
    timespec value {};
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        std::perror("clock_gettime");
        std::_Exit(EXIT_FAILURE);
    }
    return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL
        + static_cast<std::uint64_t>(value.tv_nsec);
}

std::string pointer_text(const void* pointer) {
    char text[2 + sizeof(std::uintptr_t) * 2 + 1] {};
    std::snprintf(text, sizeof(text), "0x%llx",
                  static_cast<unsigned long long>(
                      reinterpret_cast<std::uintptr_t>(pointer)));
    return text;
}

void append_record(const char* environment_name, const std::string& record) {
    std::lock_guard<std::mutex> guard(log_mutex);
    const pid_t process = getpid();
    if (descriptor_process != process) {
        if (context_descriptor >= 0) {
            close(context_descriptor);
        }
        if (launch_descriptor >= 0) {
            close(launch_descriptor);
        }
        if (kernel_launch_descriptor >= 0) {
            close(kernel_launch_descriptor);
        }
        if (kernel_resource_descriptor >= 0) {
            close(kernel_resource_descriptor);
        }
        context_descriptor = -1;
        launch_descriptor = -1;
        kernel_launch_descriptor = -1;
        kernel_resource_descriptor = -1;
        module_function_names.clear();
        recorded_resource_functions.clear();
        descriptor_process = process;
    }
    int* descriptor_slot = nullptr;
    if (std::strcmp(
            environment_name, "MIDLAYER_C500_GRAPH_LAUNCH_CONTEXT") == 0) {
        descriptor_slot = &context_descriptor;
    } else if (std::strcmp(
                   environment_name, "MIDLAYER_C500_GRAPH_LAUNCH_LOG") == 0) {
        descriptor_slot = &launch_descriptor;
    } else if (std::strcmp(
                   environment_name, "MIDLAYER_C500_KERNEL_LAUNCH_LOG") == 0) {
        descriptor_slot = &kernel_launch_descriptor;
    } else if (std::strcmp(
                   environment_name, "MIDLAYER_C500_KERNEL_RESOURCE_LOG") == 0) {
        descriptor_slot = &kernel_resource_descriptor;
    } else {
        std::fprintf(stderr, "unknown C500 correlation output environment\n");
        std::_Exit(EXIT_FAILURE);
    }
    if (*descriptor_slot < 0) {
        const std::string path = required_environment(environment_name)
            + "." + std::to_string(process) + ".jsonl";
        *descriptor_slot = open(
            path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
        if (*descriptor_slot < 0) {
            std::perror("open C500 Graph launch correlation log");
            std::_Exit(EXIT_FAILURE);
        }
    }
    const int descriptor = *descriptor_slot;
    std::size_t written = 0;
    while (written < record.size()) {
        const ssize_t result = write(
            descriptor, record.data() + written, record.size() - written);
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            std::perror("write C500 Graph launch correlation log");
            std::_Exit(EXIT_FAILURE);
        }
        written += static_cast<std::size_t>(result);
    }
}

std::pair<std::string, std::string> function_identity(const void* function) {
    const auto function_key = reinterpret_cast<std::uintptr_t>(function);
    {
        std::lock_guard<std::mutex> guard(log_mutex);
        const auto named = module_function_names.find(function_key);
        if (named != module_function_names.end()) {
            return {named->second, "mcModuleGetFunction"};
        }
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
    return {pointer_text(function), "unresolved"};
}

bool is_module_function(const void* function) {
    const auto function_key = reinterpret_cast<std::uintptr_t>(function);
    std::lock_guard<std::mutex> guard(log_mutex);
    return module_function_names.find(function_key) != module_function_names.end();
}

void record_kernel_resource(const void* function, const StepContext& context) {
    const char* resource_path = std::getenv("MIDLAYER_C500_KERNEL_RESOURCE_LOG");
    if (resource_path == nullptr || resource_path[0] == '\0' || !context.active) {
        return;
    }
    const auto function_key = reinterpret_cast<std::uintptr_t>(function);
    {
        std::lock_guard<std::mutex> guard(log_mutex);
        if (!recorded_resource_functions.insert(function_key).second) {
            return;
        }
    }
    int registers_per_thread = 0;
    int static_shared_memory_bytes = 0;
    int local_memory_bytes = 0;
    int max_dynamic_shared_memory_bytes = 0;
    int max_threads_per_block = 0;
    std::string query_api;
    mcError_t query_result = mcSuccess;
    if (is_module_function(function)) {
        using GetAttributeFunction = mcError_t (*)(
            int*, mcFunction_attribute, mcFunction_t);
        static const GetAttributeFunction get_attribute =
            require_next<GetAttributeFunction>("mcFuncGetAttribute");
        query_api = "mcFuncGetAttribute";
        const auto query = [&](int* value, mcFunction_attribute attribute) {
            const mcError_t status = get_attribute(
                value, attribute, reinterpret_cast<mcFunction_t>(
                    const_cast<void*>(function)));
            if (query_result == mcSuccess && status != mcSuccess) {
                query_result = status;
            }
        };
        query(&registers_per_thread, MC_FUNC_ATTRIBUTE_NUM_REGS);
        query(&static_shared_memory_bytes, MC_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
        query(&local_memory_bytes, MC_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES);
        query(&max_dynamic_shared_memory_bytes,
              MC_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES);
        query(&max_threads_per_block, MC_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    } else {
        using GetAttributesFunction = mcError_t (*)(mcFuncAttributes*, const void*);
        static const GetAttributesFunction get_attributes =
            require_next<GetAttributesFunction>("mcFuncGetAttributes");
        mcFuncAttributes attributes {};
        query_api = "mcFuncGetAttributes";
        query_result = get_attributes(&attributes, function);
        registers_per_thread = attributes.numRegs;
        static_shared_memory_bytes = static_cast<int>(attributes.sharedSizeBytes);
        local_memory_bytes = static_cast<int>(attributes.localSizeBytes);
        max_dynamic_shared_memory_bytes = attributes.maxDynamicSharedSizeBytes;
        max_threads_per_block = attributes.maxThreadsPerBlock;
    }
    const auto identity = function_identity(function);
    std::ostringstream record;
    record << "{\"schema_version\":1"
           << ",\"source\":\"midlayer_c500_kernel_resource_interposer\""
           << ",\"process_pid\":" << getpid()
           << ",\"thread_id\":" << syscall(SYS_gettid)
           << ",\"function\":\"" << pointer_text(function) << "\""
           << ",\"function_name\":\"" << identity.first << "\""
           << ",\"identity_source\":\"" << identity.second << "\""
           << ",\"query_api\":\"" << query_api << "\""
           << ",\"query_result\":" << query_result
           << ",\"registers_per_thread\":" << registers_per_thread
           << ",\"static_shared_memory_bytes\":" << static_shared_memory_bytes
           << ",\"local_memory_bytes\":" << local_memory_bytes
           << ",\"max_dynamic_shared_memory_bytes\":"
           << max_dynamic_shared_memory_bytes
           << ",\"max_threads_per_block\":" << max_threads_per_block
           << ",\"context\":" << context_json(context) << "}\n";
    append_record("MIDLAYER_C500_KERNEL_RESOURCE_LOG", record.str());
}

mcError_t record_kernel_launch(
        const char* api, const void* function,
        unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
        unsigned int block_x, unsigned int block_y, unsigned int block_z,
        unsigned int dynamic_shared_memory_bytes, mcStream_t stream,
        const std::uint64_t started_ns, const std::uint64_t ended_ns,
        const mcError_t result, const StepContext& context) {
    const std::uint64_t ordinal = kernel_launch_ordinal.fetch_add(1);
    std::ostringstream record;
    record << "{\"schema_version\":1"
           << ",\"source\":\"midlayer_c500_mckernellaunch_interposer\""
           << ",\"process_pid\":" << getpid()
           << ",\"thread_id\":" << syscall(SYS_gettid)
           << ",\"launch_ordinal\":" << ordinal
           << ",\"api\":\"" << api << "\""
           << ",\"host_monotonic_start_ns\":" << started_ns
           << ",\"host_monotonic_end_ns\":" << ended_ns
           << ",\"function\":\"" << pointer_text(function) << "\""
           << ",\"grid\":[" << grid_x << "," << grid_y << "," << grid_z << "]"
           << ",\"block\":[" << block_x << "," << block_y << "," << block_z << "]"
           << ",\"dynamic_shared_memory_bytes\":"
           << dynamic_shared_memory_bytes
           << ",\"stream\":\"" << pointer_text(stream) << "\""
           << ",\"result\":" << result
           << ",\"context\":" << context_json(context) << "}\n";
    append_record("MIDLAYER_C500_KERNEL_LAUNCH_LOG", record.str());
    return result;
}

std::string context_json(const StepContext& context) {
    if (!context.active) {
        return "null";
    }
    std::ostringstream value;
    value << "{\"trace_step_id\":\"" << context.trace_step_id << "\""
          << ",\"worker_rank\":" << context.worker_rank
          << ",\"context_ordinal\":" << context.context_ordinal << "}";
    return value.str();
}

void append_context_event(const char* event, const StepContext& context) {
    std::ostringstream record;
    record << "{\"schema_version\":1"
           << ",\"source\":\"midlayer_c500_scheduler_step_context\""
           << ",\"event\":\"" << event << "\""
           << ",\"process_pid\":" << getpid()
           << ",\"thread_id\":" << syscall(SYS_gettid)
           << ",\"host_monotonic_ns\":" << monotonic_ns()
           << ",\"context\":" << context_json(context) << "}\n";
    append_record("MIDLAYER_C500_GRAPH_LAUNCH_CONTEXT", record.str());
}

}  // namespace

extern "C" void midlayer_c500_set_step_context(
        const char* trace_step_id, int worker_rank,
        std::uint64_t context_ordinal) {
    if (trace_step_id == nullptr || trace_step_id[0] == '\0' || worker_rank < 0) {
        std::fprintf(stderr, "invalid C500 Scheduler step context\n");
        std::_Exit(EXIT_FAILURE);
    }
    std::lock_guard<std::mutex> guard(context_mutex);
    if (current_context.active) {
        std::fprintf(stderr, "nested C500 Scheduler step context\n");
        std::_Exit(EXIT_FAILURE);
    }
    current_context = {true, trace_step_id, worker_rank, context_ordinal};
    append_context_event("begin", current_context);
}

extern "C" void midlayer_c500_clear_step_context() {
    std::lock_guard<std::mutex> guard(context_mutex);
    if (!current_context.active) {
        std::fprintf(stderr, "missing active C500 Scheduler step context\n");
        std::_Exit(EXIT_FAILURE);
    }
    append_context_event("end", current_context);
    current_context = {};
}

extern "C" mcError_t mcGraphLaunch(mcGraphExec_t graph, mcStream_t stream) {
    using RealFunction = mcError_t (*)(mcGraphExec_t, mcStream_t);
    static const RealFunction real = require_next<RealFunction>("mcGraphLaunch");
    const std::uint64_t ordinal = launch_ordinal.fetch_add(1);
    StepContext context;
    {
        std::lock_guard<std::mutex> guard(context_mutex);
        context = current_context;
    }
    const std::uint64_t started_ns = monotonic_ns();
    const mcError_t result = real(graph, stream);
    const std::uint64_t ended_ns = monotonic_ns();
    std::ostringstream record;
    record << "{\"schema_version\":1"
           << ",\"source\":\"midlayer_c500_mcgraphlaunch_interposer\""
           << ",\"process_pid\":" << getpid()
           << ",\"thread_id\":" << syscall(SYS_gettid)
           << ",\"launch_ordinal\":" << ordinal
           << ",\"host_monotonic_start_ns\":" << started_ns
           << ",\"host_monotonic_end_ns\":" << ended_ns
           << ",\"graph_exec\":\"" << pointer_text(graph) << "\""
           << ",\"stream\":\"" << pointer_text(stream) << "\""
           << ",\"result\":" << result
           << ",\"context\":" << context_json(context) << "}\n";
    append_record("MIDLAYER_C500_GRAPH_LAUNCH_LOG", record.str());
    return result;
}

extern "C" mcError_t mcLaunchKernel(
        const void* function, dim3 grid, dim3 block, void** arguments,
        std::size_t dynamic_shared_memory_bytes, mcStream_t stream) {
    using RealFunction = mcError_t (*)(const void*, dim3, dim3, void**,
                                       std::size_t, mcStream_t);
    static const RealFunction real = require_next<RealFunction>("mcLaunchKernel");
    StepContext context;
    {
        std::lock_guard<std::mutex> guard(context_mutex);
        context = current_context;
    }
    const std::uint64_t started_ns = monotonic_ns();
    const mcError_t result = real(
        function, grid, block, arguments, dynamic_shared_memory_bytes, stream);
    const std::uint64_t ended_ns = monotonic_ns();
    record_kernel_resource(function, context);
    return record_kernel_launch(
        "mcLaunchKernel", function, grid.x, grid.y, grid.z,
        block.x, block.y, block.z,
        static_cast<unsigned int>(dynamic_shared_memory_bytes), stream,
        started_ns, ended_ns, result, context);
}

extern "C" mcError_t mcModuleLaunchKernel(
        mcFunction_t function,
        unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
        unsigned int block_x, unsigned int block_y, unsigned int block_z,
        unsigned int dynamic_shared_memory_bytes, mcStream_t stream,
        void** kernel_parameters, void** extra) {
    using RealFunction = mcError_t (*)(
        mcFunction_t, unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int, unsigned int,
        mcStream_t, void**, void**);
    static const RealFunction real =
        require_next<RealFunction>("mcModuleLaunchKernel");
    StepContext context;
    {
        std::lock_guard<std::mutex> guard(context_mutex);
        context = current_context;
    }
    const std::uint64_t started_ns = monotonic_ns();
    const mcError_t result = real(
        function, grid_x, grid_y, grid_z, block_x, block_y, block_z,
        dynamic_shared_memory_bytes, stream, kernel_parameters, extra);
    const std::uint64_t ended_ns = monotonic_ns();
    record_kernel_resource(reinterpret_cast<const void*>(function), context);
    return record_kernel_launch(
        "mcModuleLaunchKernel", reinterpret_cast<const void*>(function),
        grid_x, grid_y, grid_z, block_x, block_y, block_z,
        dynamic_shared_memory_bytes, stream, started_ns, ended_ns, result,
        context);
}

extern "C" mcError_t mcModuleGetFunction(
        mcFunction_t* function, mcModule_t module, const char* name) {
    using RealFunction = mcError_t (*)(mcFunction_t*, mcModule_t, const char*);
    static const RealFunction real =
        require_next<RealFunction>("mcModuleGetFunction");
    const mcError_t result = real(function, module, name);
    if (result == mcSuccess && function != nullptr && *function != nullptr
            && name != nullptr && name[0] != '\0') {
        std::lock_guard<std::mutex> guard(log_mutex);
        module_function_names[reinterpret_cast<std::uintptr_t>(*function)] = name;
    }
    return result;
}
