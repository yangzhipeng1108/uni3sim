// Independent asynchronous copy producer capacity, no model or tensor math.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

#define CHECK(call) do { const auto error = (call); if (error != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); std::exit(2); \
} } while (0)

__device__ unsigned mix(unsigned value) {
    value ^= value >> 16; value *= 0x7feb352dU;
    value ^= value >> 15; value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

__global__ void initialize(uint4* data, size_t count) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += gridDim.x * blockDim.x) {
        const auto value = static_cast<unsigned>(index);
        data[index] = make_uint4(mix(value), mix(value ^ 0x12345678U),
                                mix(value ^ 0x87654321U), mix(value ^ 0xabcd1234U));
    }
}

struct Stamp { unsigned long long cycles, ns; unsigned checksum; };

__device__ unsigned long long global_ns() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

template<int Depth, bool CacheGlobal>
__global__ void producer_copy(const uint4* input, int vectors, bool hot,
                              int iterations, Stamp* stamps) {
    extern __shared__ __align__(16) unsigned char storage[];
    constexpr size_t pool_vectors = size_t(1) << 25;
    if (threadIdx.x < 128) {
        const auto first_ns = global_ns();
        const auto first_cycle = static_cast<unsigned long long>(clock64());
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const int slot = iteration % Depth;
            const size_t base = hot
                ? (size_t(blockIdx.x) * Depth + slot) * vectors
                : ((size_t(iteration) * gridDim.x + blockIdx.x) * vectors) & (pool_vectors - 1);
            for (int index = threadIdx.x; index < vectors; index += 128) {
                const auto shared = static_cast<unsigned>(__cvta_generic_to_shared(
                    storage + (slot * vectors + index) * sizeof(uint4)));
                if constexpr (CacheGlobal) {
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                        :: "r"(shared), "l"(input + base + index) : "memory");
                } else {
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                        :: "r"(shared), "l"(input + base + index) : "memory");
                }
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
            if (iteration >= Depth - 1) {
                asm volatile("cp.async.wait_group %0;" :: "n"(Depth - 1) : "memory");
            }
        }
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        if (threadIdx.x == 0) {
            stamps[blockIdx.x].cycles = static_cast<unsigned long long>(clock64()) - first_cycle;
            stamps[blockIdx.x].ns = global_ns() - first_ns;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) stamps[blockIdx.x].checksum = *reinterpret_cast<unsigned*>(storage);
}

template<int Depth, bool CacheGlobal>
void measure(uint4* input, Stamp* stamps, cudaStream_t stream, bool hot, int bytes) {
    constexpr int blocks = 132, threads = 384, shared_bytes = 200 * 1024, iterations = 256;
    CHECK(cudaFuncSetAttribute(producer_copy<Depth,CacheGlobal>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    int residency;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&residency,
        producer_copy<Depth,CacheGlobal>, threads, shared_bytes));
    if (residency != 1 || bytes * Depth > shared_bytes) std::exit(3);
    cudaGraph_t graph;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    producer_copy<Depth,CacheGlobal><<<blocks,threads,shared_bytes,stream>>>(
        input, bytes / sizeof(uint4), hot, iterations, stamps);
    CHECK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t executable;
    CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int warm = 0; warm < 8; ++warm) CHECK(cudaGraphLaunch(executable, stream));
    CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
    for (int trial = 0; trial < 7; ++trial) {
        CHECK(cudaEventRecord(start, stream));
        CHECK(cudaGraphLaunch(executable, stream));
        CHECK(cudaEventRecord(stop, stream));
        CHECK(cudaEventSynchronize(stop));
        float milliseconds;
        CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        Stamp values[blocks];
        CHECK(cudaMemcpy(values, stamps, sizeof(values), cudaMemcpyDeviceToHost));
        double clock = 0;
        unsigned long long maximum_ns = 0;
        for (const auto& value : values) {
            if (!value.ns || !value.cycles) std::exit(4);
            clock += double(value.cycles) / value.ns;
            if (value.ns > maximum_ns) maximum_ns = value.ns;
        }
        std::printf("{\"kind\":\"sample\",\"cache_modifier\":\"%s\",\"depth\":%d,\"hot\":%s,\"panel_bytes\":%d,\"blocks\":%d,\"threads\":%d,\"producer_threads\":128,\"shared_bytes\":%d,\"resident_blocks_per_sm\":%d,\"iterations\":%d,\"trial\":%d,\"logical_read_bytes\":%lld,\"event_ms\":%.9f,\"maximum_cta_ns\":%llu,\"mean_sm_clock_ghz\":%.9f}\n",
            CacheGlobal ? "cg" : "ca", Depth, hot ? "true" : "false", bytes,
            blocks, threads, shared_bytes, residency, iterations, trial,
            static_cast<long long>(bytes) * blocks * iterations, milliseconds, maximum_ns, clock / blocks);
    }
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    CHECK(cudaGraphExecDestroy(executable)); CHECK(cudaGraphDestroy(graph));
}

int main() {
    cudaDeviceProp properties;
    CHECK(cudaGetDeviceProperties(&properties, 0));
    if (properties.major != 9 || properties.multiProcessorCount != 132) return 5;
    std::printf("{\"kind\":\"identity\",\"name\":\"%s\",\"sms\":%d}\n", properties.name, properties.multiProcessorCount);
    constexpr size_t bytes = size_t(512) << 20;
    uint4* input; Stamp* stamps; cudaStream_t stream;
    CHECK(cudaMalloc(&input, bytes)); CHECK(cudaMalloc(&stamps, 132 * sizeof(Stamp)));
    CHECK(cudaStreamCreate(&stream));
    initialize<<<528,256,0,stream>>>(input, bytes / sizeof(uint4));
    CHECK(cudaStreamSynchronize(stream));
    for (bool hot : {true,false}) for (int panel : {16384,32768,65536}) {
        measure<1,true>(input, stamps, stream, hot, panel);
        measure<2,true>(input, stamps, stream, hot, panel);
        measure<1,false>(input, stamps, stream, hot, panel);
        measure<2,false>(input, stamps, stream, hot, panel);
        if (panel <= 32768) {
            measure<4,true>(input, stamps, stream, hot, panel);
            measure<4,false>(input, stamps, stream, hot, panel);
        }
    }
    CHECK(cudaStreamDestroy(stream)); CHECK(cudaFree(stamps)); CHECK(cudaFree(input));
    return 0;
}
