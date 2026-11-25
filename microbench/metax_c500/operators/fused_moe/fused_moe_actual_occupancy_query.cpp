// Query static C500 resources for a Triton binary already loaded by warmup.
//
// This helper never launches a kernel.  The Python caller compiles with
// JITFunction.warmup(), calls CompiledKernel._init_handles() to load the
// binary, then passes that registered function handle here for the same
// runtime resource queries used by the driver admission API.

#include <mc_runtime.h>

#include <cstddef>

extern "C" int midlayer_c500_fused_moe_query_occupancy(
        const void* function, int block_threads,
        std::size_t dynamic_shared_memory_bytes,
        int* attribute_status, int* occupancy_status,
        int* registers_per_thread, std::size_t* static_shared_memory_bytes,
        int* resident_ctas_per_ap) {
    if (function == nullptr || block_threads <= 0 || attribute_status == nullptr
            || occupancy_status == nullptr || registers_per_thread == nullptr
            || static_shared_memory_bytes == nullptr
            || resident_ctas_per_ap == nullptr) {
        return -1;
    }
    mcFuncAttributes attributes {};
    int resident = 0;
    const mcError_t attributes_result = mcFuncGetAttributes(&attributes, function);
    const mcError_t occupancy_result = mcOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, function, block_threads, dynamic_shared_memory_bytes);
    *attribute_status = static_cast<int>(attributes_result);
    *occupancy_status = static_cast<int>(occupancy_result);
    *registers_per_thread = attributes.numRegs;
    *static_shared_memory_bytes = attributes.sharedSizeBytes;
    *resident_ctas_per_ap = resident;
    return 0;
}
