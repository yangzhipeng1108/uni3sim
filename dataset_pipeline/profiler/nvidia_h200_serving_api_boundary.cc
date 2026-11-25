// Native Host API spans in the existing ten-int64 boundary-event format.
// No GPU work or production model duration is priced by this observer.
#include <cupti.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <ctime>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/syscall.h>

namespace {
struct Row { int64_t value[10]; };
static_assert(sizeof(Row) == 80, "boundary event ABI");
struct Context { int64_t step=-1, batch=0, tokens=0, qmax=0; };
struct Entry {
    int64_t label, wall, cpu, tid, correlation;
    CUpti_CallbackDomain domain;
    CUpti_CallbackId cbid;
    Context context;
};
thread_local Context context;
thread_local std::vector<Entry> pending;
thread_local std::vector<Entry> markers;
std::unique_ptr<Row[]> rows;
size_t capacity=0;
std::atomic<size_t> count{0}, active{0}, open_spans{0};
std::atomic<bool> recording{false};
std::atomic<int> error_code{0};
std::mutex label_lock;
std::map<std::string,int64_t> label_ids;
std::vector<std::string> labels={"GPUModelRunner.execute_model", "GPUModelRunner.sample_tokens"};
CUpti_SubscriberHandle subscriber{};
bool subscribed=false;
uint32_t cupti_version=0;

int64_t clock_ns(clockid_t clock) {
    timespec value{};
    if(clock_gettime(clock,&value)!=0) { error_code=1; return 0; }
    return int64_t(value.tv_sec)*1000000000+value.tv_nsec;
}
int64_t thread_id() { return int64_t(syscall(SYS_gettid)); }
Entry begin(int64_t label, int64_t correlation, CUpti_CallbackDomain domain, CUpti_CallbackId cbid) {
    return Entry{label,clock_ns(CLOCK_MONOTONIC),clock_ns(CLOCK_THREAD_CPUTIME_ID),
        thread_id(),correlation,domain,cbid,context};
}
void append(const Entry &entry) {
    int64_t end=clock_ns(CLOCK_MONOTONIC), cpu=clock_ns(CLOCK_THREAD_CPUTIME_ID);
    size_t index=count.fetch_add(1);
    if(index>=capacity) { error_code=2; return; }
    Row value{{entry.label,entry.wall,end,cpu-entry.cpu,entry.context.step,
        entry.context.batch,entry.context.tokens,entry.context.qmax,entry.tid,entry.correlation}};
    if(end<entry.wall || cpu<entry.cpu) error_code=3;
    rows[index]=value;
}
int64_t label(CUpti_CallbackDomain domain, CUpti_CallbackId id, const CUpti_CallbackData *data) {
    std::string name="cuda_api:"+std::to_string(int(domain))+":"+std::to_string(id)+":";
    const char *function=data->functionName==nullptr ? "<unknown-function>" : data->functionName;
    name+=function;
    name+=':';
    // CUPTI documents symbolName only for kernel launch callbacks. Other
    // API callback payloads must not be interpreted as kernel identities.
    bool kernel_launch=std::strstr(function,"LaunchKernel")!=nullptr
        || std::strstr(function,"LaunchCooperativeKernel")!=nullptr
        || std::strcmp(function,"cudaLaunch")==0 || std::strcmp(function,"cuLaunch")==0;
    name+=kernel_launch && data->symbolName!=nullptr ? data->symbolName : "<no-symbol>";
    std::lock_guard<std::mutex> guard(label_lock);
    auto found=label_ids.find(name);
    if(found!=label_ids.end()) return found->second;
    int64_t index=int64_t(labels.size())+98;
    labels.push_back(name);
    label_ids.emplace(name,index);
    return index;
}
void CUPTIAPI callback(void *, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void *raw) {
    active.fetch_add(1);
    // Drain an already entered API (including its nested calls) at shutdown.
    // New independent calls after the stop boundary are outside the trace.
    if(!recording.load() && pending.empty()) { active.fetch_sub(1); return; }
    if(domain!=CUPTI_CB_DOMAIN_RUNTIME_API && domain!=CUPTI_CB_DOMAIN_DRIVER_API) {
        error_code=4; active.fetch_sub(1); return;
    }
    const auto *data=static_cast<const CUpti_CallbackData *>(raw);
    try {
        if(data->callbackSite==CUPTI_API_ENTER) {
            pending.push_back(begin(label(domain,id,data),data->correlationId,domain,id));
            open_spans.fetch_add(1);
        } else if(data->callbackSite==CUPTI_API_EXIT) {
            if(pending.empty() || pending.back().domain!=domain || pending.back().cbid!=id
                    || pending.back().correlation!=data->correlationId) error_code=5;
            else { append(pending.back()); pending.pop_back(); open_spans.fetch_sub(1); }
        } else error_code=6;
    } catch(...) { error_code=7; }
    active.fetch_sub(1);
}
std::string escaped(const std::string &value) {
    std::string result="\"";
    const char *hex="0123456789abcdef";
    for(unsigned char ch:value) {
        if(ch=='"' || ch=='\\') { result+='\\'; result+=char(ch); }
        else if(ch<32) { result+="\\u00"; result+=hex[ch>>4]; result+=hex[ch&15]; }
        else result+=char(ch);
    }
    return result+'"';
}
}

extern "C" const char *midlayer_boundary_provider() {
    Dl_info info{};
    return dladdr(reinterpret_cast<void *>(&cuptiSubscribe), &info)!=0 ? info.dli_fname : nullptr;
}
extern "C" int midlayer_boundary_init(size_t maximum_rows, int api_enabled) {
    if(recording.load() || maximum_rows==0 || rows!=nullptr) return -1;
    uint16_t endian=1;
    if(*reinterpret_cast<unsigned char *>(&endian)!=1) return -3;
    try { rows=std::make_unique<Row[]>(maximum_rows); capacity=maximum_rows; }
    catch(...) { return -2; }
    if(api_enabled) {
        CUptiResult result=cuptiGetVersion(&cupti_version);
        if(result!=CUPTI_SUCCESS) return int(result)+1000;
        result=cuptiSubscribe(&subscriber,callback,nullptr);
        if(result!=CUPTI_SUCCESS) return int(result)+1000;
        subscribed=true;
        recording=true;
        result=cuptiEnableDomain(1,subscriber,CUPTI_CB_DOMAIN_RUNTIME_API);
        if(result==CUPTI_SUCCESS) result=cuptiEnableDomain(1,subscriber,CUPTI_CB_DOMAIN_DRIVER_API);
        if(result!=CUPTI_SUCCESS) { recording=false; error_code=8; return int(result)+1000; }
    }
    recording=true;
    return 0;
}
extern "C" int midlayer_boundary_begin(int64_t phase,int64_t step,int64_t batch,int64_t tokens,int64_t qmax) {
    if(!recording.load() || (phase!=0 && phase!=1) || !markers.empty()) return -1;
    context=Context{step,batch,tokens,qmax};
    markers.push_back(begin(phase,0,CUPTI_CB_DOMAIN_INVALID,0));
    open_spans.fetch_add(1);
    return 0;
}
extern "C" int midlayer_boundary_end() {
    if(!recording.load() || markers.empty()) return -1;
    append(markers.back()); markers.pop_back(); open_spans.fetch_sub(1);
    context=Context{};
    return 0;
}
extern "C" int midlayer_boundary_dump(const char *directory) {
    if(directory==nullptr || rows==nullptr) return -1;
    recording=false;
    int shutdown_error=0;
    int64_t deadline=clock_ns(CLOCK_MONOTONIC)+1000000000;
    while(open_spans.load()!=0 && clock_ns(CLOCK_MONOTONIC)<deadline) std::this_thread::yield();
    if(subscribed) {
        if(cuptiEnableDomain(0,subscriber,CUPTI_CB_DOMAIN_RUNTIME_API)!=CUPTI_SUCCESS) shutdown_error=9;
        if(cuptiEnableDomain(0,subscriber,CUPTI_CB_DOMAIN_DRIVER_API)!=CUPTI_SUCCESS) shutdown_error=9;
        if(cuptiUnsubscribe(subscriber)!=CUPTI_SUCCESS) shutdown_error=9;
        subscribed=false;
    }
    while(active.load()!=0) std::this_thread::yield();
    if(open_spans.load()!=0) error_code=10;
    if(shutdown_error) error_code=shutdown_error;
    size_t written=std::min(count.load(),capacity);
    std::ofstream binary(std::string(directory)+"/events.bin",std::ios::binary);
    binary.write(reinterpret_cast<const char *>(rows.get()),std::streamsize(written*sizeof(Row)));
    binary.close();
    if(!binary) error_code=11;
    std::ofstream metadata(std::string(directory)+"/native_metadata.json");
    metadata<<"{\"fields\":[\"label\",\"start_ns\",\"end_ns\",\"thread_cpu_ns\",\"step\",\"batch\",\"tokens\",\"qmax\",\"thread_id\",\"object_id\"],\"itemsize\":8,\"byteorder\":\"little\",\"pid\":"<<getpid()
        <<",\"rows\":"<<written<<",\"attempted_rows\":"<<count.load()
        <<",\"unclosed_profile_frames\":"<<open_spans.load()<<",\"error_code\":"<<error_code.load()
        <<",\"cupti_version\":"<<cupti_version<<",\"clock\":\"CLOCK_MONOTONIC\",\"thread_clock\":\"CLOCK_THREAD_CPUTIME_ID\",\"labels\":{\"0\":"<<escaped(labels[0])<<",\"1\":"<<escaped(labels[1]);
    for(size_t index=2;index<labels.size();++index) metadata<<",\""<<(index+98)<<"\":"<<escaped(labels[index]);
    metadata<<"}}\n";
    metadata.close();
    if(!metadata) return -12;
    return error_code.load();
}
#ifdef MIDLAYER_API_BOUNDARY_TEST
extern "C" void midlayer_boundary_test_event(int domain,int id,int site,uint32_t correlation,const char *name,const char *symbol) {
    CUpti_CallbackData data{};
    data.callbackSite=static_cast<CUpti_ApiCallbackSite>(site);
    data.correlationId=correlation; data.functionName=name; data.symbolName=symbol;
    callback(nullptr,static_cast<CUpti_CallbackDomain>(domain),CUpti_CallbackId(id),&data);
}
#endif
