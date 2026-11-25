// Untimed, bounded module and launch facts for isolated GEMM calls.
#include <cupti.h>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
constexpr size_t max_modules = 256, max_launches = 512, max_tensor_maps = 4096;
constexpr size_t max_phases = 64, max_bytes = 128 * 1024 * 1024;
std::mutex lock;
std::atomic<int> phase{-1};
std::atomic<int> error_code{0};
size_t modules = 0, launches = 0, tensor_maps = 0, bytes = 0;
std::string directory;
std::ofstream events;
CUpti_SubscriberHandle subscriber{};
bool subscribed = false;

std::string quote(const char *value) {
    if (value == nullptr) throw std::runtime_error("missing callback symbol");
    std::ostringstream out;
    out << '"';
    for (const unsigned char *p = reinterpret_cast<const unsigned char *>(value); *p; ++p) {
        if (*p == '"' || *p == '\\') out << '\\' << char(*p);
        else if (*p < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(*p);
        else out << char(*p);
    }
    out << '"';
    return out.str();
}

void module(const CUpti_ResourceData *resource, int current) {
    if (resource == nullptr || resource->resourceDescriptor == nullptr)
        throw std::runtime_error("missing module descriptor");
    const auto *data = static_cast<const CUpti_ModuleResourceData *>(resource->resourceDescriptor);
    if (modules >= max_modules || data->cubinSize > max_bytes - bytes || !data->cubinSize || data->pCubin == nullptr)
        throw std::runtime_error("module limit or missing cubin");
    std::string filename = "module_" + std::to_string(modules) + ".cubin";
    std::ofstream file(directory + "/" + filename, std::ios::binary);
    file.write(data->pCubin, data->cubinSize);
    file.close();
    if (!file) throw std::runtime_error("cubin write failed");
    events << "{\"kind\":\"module\",\"phase\":" << current << ",\"module_id\":" << data->moduleId
           << ",\"context\":" << uintptr_t(resource->context) << ",\"bytes\":" << data->cubinSize
           << ",\"path\":" << quote(filename.c_str()) << "}\n";
    ++modules;
    bytes += data->cubinSize;
}

template<class Params> CUlaunchConfig ordinary(const void *raw) {
    const auto *p = static_cast<const Params *>(raw);
    CUlaunchConfig config{};
    config.gridDimX=p->gridDimX; config.gridDimY=p->gridDimY; config.gridDimZ=p->gridDimZ;
    config.blockDimX=p->blockDimX; config.blockDimY=p->blockDimY; config.blockDimZ=p->blockDimZ;
    config.sharedMemBytes=p->sharedMemBytes; config.hStream=p->hStream;
    return config;
}

void launch(const CUpti_CallbackData *data, CUpti_CallbackId id, int current) {
    if (data == nullptr) throw std::runtime_error("missing launch data");
    if (data->callbackSite != CUPTI_API_EXIT) return;
    if (launches >= max_launches || data->functionParams == nullptr || data->functionReturnValue == nullptr
            || *static_cast<const CUresult *>(data->functionReturnValue) != CUDA_SUCCESS)
        throw std::runtime_error("launch limit, missing params, or unsuccessful launch");
    CUlaunchConfig config{};
    switch (id) {
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
        config=ordinary<cuLaunchKernel_params>(data->functionParams); break;
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz:
        config=ordinary<cuLaunchKernel_ptsz_params>(data->functionParams); break;
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
        if (static_cast<const cuLaunchKernelEx_params *>(data->functionParams)->config == nullptr)
            throw std::runtime_error("missing extended config");
        config=*static_cast<const cuLaunchKernelEx_params *>(data->functionParams)->config; break;
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz:
        if (static_cast<const cuLaunchKernelEx_ptsz_params *>(data->functionParams)->config == nullptr)
            throw std::runtime_error("missing extended config");
        config=*static_cast<const cuLaunchKernelEx_ptsz_params *>(data->functionParams)->config; break;
    default: throw std::runtime_error("unsubscribed launch API");
    }
    if (config.numAttrs > 32 || (config.numAttrs && config.attrs == nullptr))
        throw std::runtime_error("invalid launch attributes");
    events << "{\"kind\":\"launch\",\"phase\":" << current << ",\"symbol\":" << quote(data->symbolName)
           << ",\"api\":" << quote(data->functionName) << ",\"grid\":[" << config.gridDimX << ',' << config.gridDimY
           << ',' << config.gridDimZ << "],\"block\":[" << config.blockDimX << ',' << config.blockDimY << ','
           << config.blockDimZ << "],\"dynamic_shared_bytes\":" << config.sharedMemBytes << ",\"attributes\":[";
    for (unsigned i=0; i<config.numAttrs; ++i) {
        if (i) events << ',';
        events << "{\"id\":" << int(config.attrs[i].id);
        if (config.attrs[i].id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
            const auto &c=config.attrs[i].value.clusterDim;
            events << ",\"cluster_dimensions\":[" << c.x << ',' << c.y << ',' << c.z << ']';
        }
        events << '}';
    }
    events << "]}\n";
    ++launches;
}

template<class Value> void array(const Value *values, unsigned count) {
    events << '[';
    for (unsigned index=0; index<count; ++index) {
        if (index) events << ',';
        events << values[index];
    }
    events << ']';
}

void tensor_map(const CUpti_CallbackData *data, int current) {
    if (data == nullptr) throw std::runtime_error("missing tensor-map data");
    if (data->callbackSite != CUPTI_API_EXIT) return;
    if (tensor_maps >= max_tensor_maps || data->functionParams == nullptr || data->functionReturnValue == nullptr
            || *static_cast<const CUresult *>(data->functionReturnValue) != CUDA_SUCCESS)
        throw std::runtime_error("tensor-map limit, missing params, or unsuccessful encode");
    const auto *p=static_cast<const cuTensorMapEncodeTiled_params *>(data->functionParams);
    if (p->tensorRank < 1 || p->tensorRank > 5 || p->globalDim == nullptr
            || p->boxDim == nullptr || p->elementStrides == nullptr
            || (p->tensorRank > 1 && p->globalStrides == nullptr))
        throw std::runtime_error("invalid tensor-map rank or missing arrays");
    // Only discrete descriptor fields. Do not dereference the global tensor,
    // serialize its address, decode opaque launch arguments, or record time.
    events << "{\"kind\":\"tensor_map\",\"phase\":" << current
           << ",\"api\":" << quote(data->functionName) << ",\"rank\":" << p->tensorRank
           << ",\"data_type\":" << int(p->tensorDataType) << ",\"global_dimensions\":";
    array(p->globalDim, p->tensorRank);
    events << ",\"global_strides_bytes\":";
    array(p->globalStrides, p->tensorRank-1);
    events << ",\"box_dimensions\":";
    array(p->boxDim, p->tensorRank);
    events << ",\"element_strides\":";
    array(p->elementStrides, p->tensorRank);
    events << ",\"interleave\":" << int(p->interleave) << ",\"swizzle\":" << int(p->swizzle)
           << ",\"l2_promotion\":" << int(p->l2Promotion) << ",\"oob_fill\":" << int(p->oobFill) << "}\n";
    ++tensor_maps;
}

void CUPTIAPI callback(void *, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void *raw) {
    int current=phase.load();
    if (current < 0) return;
    std::lock_guard<std::mutex> guard(lock);
    if (error_code) return;
    try {
        if (domain == CUPTI_CB_DOMAIN_RESOURCE && id == CUPTI_CBID_RESOURCE_MODULE_LOADED)
            module(static_cast<const CUpti_ResourceData *>(raw), current);
        else if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
            if (id == CUPTI_DRIVER_TRACE_CBID_cuTensorMapEncodeTiled)
                tensor_map(static_cast<const CUpti_CallbackData *>(raw), current);
            else launch(static_cast<const CUpti_CallbackData *>(raw), id, current);
        }
        else throw std::runtime_error("unexpected callback domain");
        if (!events) throw std::runtime_error("event write failed");
    } catch (const std::exception &error) {
        error_code=1;
        events << "{\"kind\":\"error\",\"message\":" << quote(error.what()) << "}\n";
    }
}
}

extern "C" int midlayer_gemm_module_init(const char *path, int enable) {
    if (!directory.empty() || path == nullptr) return 2;
    directory=path;
    events.open(directory + "/module_events.jsonl");
    if (!events) return 3;
    if (!enable) return 0;
    CUptiResult status=cuptiSubscribe(&subscriber, callback, nullptr);
    if (status != CUPTI_SUCCESS) return 1000 + int(status);
    subscribed=true;
    status=cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED);
    if (status != CUPTI_SUCCESS) return 2000 + int(status);
    for (const auto id : {CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz,
            CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz,
            CUPTI_DRIVER_TRACE_CBID_cuTensorMapEncodeTiled}) {
        status=cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, id);
        if (status != CUPTI_SUCCESS) return 3000 + int(status);
    }
    return 0;
}

extern "C" int midlayer_gemm_module_phase(int value) {
    if (value < -1 || value >= int(max_phases)) return 4;
    phase=value;
    return error_code;
}

extern "C" int midlayer_gemm_module_finish() {
    phase=-1;
    if (subscribed && cuptiUnsubscribe(subscriber) != CUPTI_SUCCESS) error_code=5;
    subscribed=false;
    std::lock_guard<std::mutex> guard(lock);
    events.flush();
    std::ofstream metadata(directory + "/module_metadata.json");
    metadata << "{\"modules\":" << modules << ",\"launches\":" << launches
             << ",\"tensor_maps\":" << tensor_maps << ",\"cubin_bytes\":" << bytes
             << ",\"error_code\":" << error_code << ",\"duration_recorded\":false}\n";
    return error_code;
}

#ifdef MIDLAYER_GEMM_MODULE_TEST
extern "C" void midlayer_gemm_module_test(int kind) {
    if (kind == 0) {
        const char value[]="\177ELFsynthetic";
        CUpti_ModuleResourceData data{7, sizeof(value), value};
        CUpti_ResourceData resource{};
        resource.resourceDescriptor=&data;
        callback(nullptr, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED, &resource);
    } else if (kind >= 3 && kind <= 5) {
        CUresult status=kind == 5 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
        const cuuint64_t dimensions[]={2048, 729}, strides[]={4096};
        const cuuint32_t box[]={64, 32}, element_strides[]={1, 1};
        cuTensorMapEncodeTiled_params params{};
        params.tensorRank=kind == 4 ? 0 : 2;
        params.tensorDataType=CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        params.globalDim=dimensions; params.globalStrides=strides;
        params.boxDim=box; params.elementStrides=element_strides;
        params.swizzle=CU_TENSOR_MAP_SWIZZLE_128B;
        params.l2Promotion=CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
        CUpti_CallbackData data{};
        data.callbackSite=CUPTI_API_EXIT; data.functionName="cuTensorMapEncodeTiled";
        data.functionParams=&params; data.functionReturnValue=&status;
        callback(nullptr, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuTensorMapEncodeTiled, &data);
    } else {
        CUresult status=kind == 1 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
        cuLaunchKernel_params params{};
        params.gridDimX=2; params.gridDimY=3; params.gridDimZ=1;
        params.blockDimX=128; params.blockDimY=1; params.blockDimZ=1;
        CUpti_CallbackData data{};
        data.callbackSite=CUPTI_API_EXIT; data.functionName="cuLaunchKernel"; data.symbolName="synthetic\"\\\n";
        data.functionParams=&params; data.functionReturnValue=&status;
        callback(nullptr, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, &data);
    }
}
#endif
