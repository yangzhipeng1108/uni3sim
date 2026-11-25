// Non-duration observation of the public API used by the synthetic Cast.
// Build against the real CANN headers; no replacement SDK/runtime is supplied.
#include <acl/acl_rt.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace {
constexpr const char* kTarget = "Cast_e87590d11ccda8b259ab6b1ea7212319_high_performance";
std::mutex output_mutex;

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "Cast host-args observer: %s\n", message);
    std::_Exit(86);
}

std::string quoted(const char* value) {
    std::string result = "\"";
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(value); *p; ++p) {
        if (*p == '"' || *p == '\\') { result += '\\'; result += static_cast<char>(*p); }
        else if (*p < 32) {
            char buffer[7]; std::snprintf(buffer, sizeof(buffer), "\\u%04x", *p); result += buffer;
        } else result += static_cast<char>(*p);
    }
    return result + "\"";
}

void capture(aclrtFuncHandle function, uint32_t block_dim, void* host_args,
             size_t args_size, aclrtPlaceHolderInfo* placeholders, size_t count) {
    using NameFunction = decltype(&aclrtGetFunctionName);
    auto get_name = reinterpret_cast<NameFunction>(dlsym(RTLD_NEXT, "aclrtGetFunctionName"));
    if (!get_name) fail("missing original aclrtGetFunctionName");
    char name[1024] = {};
    if (get_name(function, sizeof(name), name) != ACL_SUCCESS) fail("function name query failed");
    if (std::strlen(name) >= sizeof(name) - 1) fail("truncated function name");
    const char* output = std::getenv("MIDLAYER_CAST_HOST_ARGS_OUTPUT");
    const char* baseline = std::getenv("MIDLAYER_CAST_HOST_ARGS_BASELINE");
    if (!output || !*output || !baseline || std::strlen(baseline) != 40)
        fail("required output/baseline capture environment is absent");
    const bool target = std::strncmp(name, kTarget, std::strlen(kTarget)) == 0;
    std::string row = "{\"api\":\"aclrtLaunchKernelWithHostArgs\",\"baseline\":";
    row += quoted(baseline) + ",\"kernel_name\":" + quoted(name);
    row += ",\"block_dim\":" + std::to_string(block_dim);
    row += ",\"args_size\":" + std::to_string(args_size);
    row += ",\"target\":" + std::string(target ? "true" : "false");
    if (target) {
        if (!host_args || args_size == 0 || args_size > 4096 || count > 16
                || (count && !placeholders)) fail("invalid bounded public host-args buffer");
        constexpr char hex[] = "0123456789abcdef";
        row += ",\"host_args_hex\":\"";
        auto bytes = static_cast<const unsigned char*>(host_args);
        for (size_t i = 0; i < args_size; ++i) { row += hex[bytes[i] >> 4]; row += hex[bytes[i] & 15]; }
        row += "\",\"placeholders\":[";
        for (size_t i = 0; i < count; ++i) {
            const auto& entry = placeholders[i];
            if (entry.addrOffset > args_size || args_size - entry.addrOffset < sizeof(void*)
                    || entry.dataOffset >= args_size) fail("placeholder is outside public host-args buffer");
            if (i) row += ',';
            row += "{\"addr_offset\":" + std::to_string(entry.addrOffset)
                + ",\"data_offset\":" + std::to_string(entry.dataOffset) + "}";
        }
        row += ']';
    }
    row += ",\"duration_recorded\":false}\n";
    std::lock_guard<std::mutex> guard(output_mutex);
    const int fd = open(output, O_WRONLY | O_CREAT | O_APPEND, 0600);
    if (fd < 0) fail("cannot open task-local observation output");
    const ssize_t written = write(fd, row.data(), row.size());
    const int closed = close(fd);
    if (written != static_cast<ssize_t>(row.size()) || closed) fail("observation write failed");
}
}

extern "C" aclError aclrtLaunchKernelWithHostArgs(
        aclrtFuncHandle function, uint32_t block_dim, aclrtStream stream,
        aclrtLaunchKernelCfg* config, void* host_args, size_t args_size,
        aclrtPlaceHolderInfo* placeholders, size_t placeholder_count) {
    using LaunchFunction = decltype(&aclrtLaunchKernelWithHostArgs);
    static auto original = reinterpret_cast<LaunchFunction>(dlsym(RTLD_NEXT, "aclrtLaunchKernelWithHostArgs"));
    if (!original) fail("missing original aclrtLaunchKernelWithHostArgs");
    capture(function, block_dim, host_args, args_size, placeholders, placeholder_count);
    return original(function, block_dim, stream, config, host_args, args_size,
                    placeholders, placeholder_count);
}
