/*
 * MetaX C500 device-allocation physical mapping probe.
 *
 * The grouped-MoE investigation ends at one un-measured link.  The stride
 * sweep found that the same logical access sequence, at the production row
 * pitch, costs materially different downstream traffic depending on which
 * allocation it runs against -- roughly a 2x swing -- and attributed that to
 * physical allocation.  It never measured the allocation.  Every conclusion
 * that follows from "allocation is not a launch fact" therefore rests on an
 * assertion rather than on a measurement.
 *
 * This probe measures it.  The MetaX kernel-mode driver exposes
 * `mxkwGetPhysicalAddress`, which resolves a page-aligned device-visible
 * address to its physical address.  For buffers the size of one production
 * expert weight matrix it records, per page:
 *
 *   - the page granularity the driver actually hands out
 *   - whether consecutive virtual pages are physically contiguous
 *   - the physical frame, and the low-order bits that any channel, bank or
 *     rank interleave would be indexed by
 *
 * and repeats the allocation within one process so that the assignment
 * policy is visible, and across processes when run repeatedly, so that
 * run-to-run stability is measured rather than assumed.
 *
 * Allocator note: `mxkwOpenMXCD` opens /dev/mxcd, and on this runtime it and
 * the MACA runtime cannot both hold the device -- opening the kernel handle
 * first makes the first `mcMalloc` fail to initialise, and initialising the
 * runtime first makes `mxkwOpenMXCD` fail.  The probe therefore allocates
 * through the driver's own allocator rather than through `mcMalloc`.  What it
 * measures is consequently the driver's virtual-to-physical assignment
 * policy, which is the layer that assigns frames to any device buffer; it is
 * not a claim about a specific `mcMalloc` pointer's frames.
 *
 * This is an addressing observation, not a traffic or duration measurement.
 * Nothing it reports prices a Forward path; it exists to establish whether
 * the quantity that the traffic depends on is knowable to a static model.
 *
 * Build inside the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -I/opt/maca-3.7.2/include \
 *       -o allocation_physical_mapping allocation_physical_mapping.cpp \
 *       -lmxkw
 */

#include <mxkw/mxkw.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* One Qwen3-235B-A22B expert W1 slab: N=1536*2, K=4096, bf16. */
static const size_t kBufferBytes = 2ull * 1536 * 4096 * 2;
static const int kAllocations = 4;

static void probe_one(int ordinal, void* base, size_t bytes, size_t page,
                      uint32_t node) {
    size_t resolved = 0, contiguous = 0, failed = 0;
    uint64_t previous = 0;
    bool have_previous = false;
    std::vector<uint64_t> frames;
    for (size_t offset = 0; offset < bytes; offset += page) {
        uint64_t pa = 0;
        void* address = static_cast<char*>(base) + offset;
        auto status = mxkwGetPhysicalAddress(address, &pa);
        if (status != MXKW_STATUS_SUCCESS || pa == 0) {
            auto ex = mxkwGetPhysicalAddressEx(address, node, &pa);
            if (ex != MXKW_STATUS_SUCCESS || pa == 0) {
                if (failed == 0)
                    std::fprintf(stderr,
                                 "physical resolve refused: status=%d ex=%d\n",
                                 (int)status, (int)ex);
                failed++;
                have_previous = false;
                continue;
            }
        }
        resolved++;
        frames.push_back(pa);
        if (have_previous && pa == previous + page) contiguous++;
        previous = pa;
        have_previous = true;
    }
    std::printf("{\"ordinal\":%d,\"virtual_base\":\"0x%llx\",\"bytes\":%zu,"
                "\"page_bytes\":%zu,\"pages_resolved\":%zu,"
                "\"pages_failed\":%zu,\"contiguous_steps\":%zu,\"frames\":[",
                ordinal, (unsigned long long)(uintptr_t)base, bytes, page,
                resolved, failed, contiguous);
    for (size_t i = 0; i < frames.size(); i++) {
        if (i) std::printf(",");
        std::printf("\"0x%llx\"", (unsigned long long)frames[i]);
    }
    std::printf("]}\n");
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    const int requested = argc > 1 ? std::atoi(argv[1]) : -1;
    if (mxkwOpenMXCD() != MXKW_STATUS_SUCCESS) {
        std::fprintf(stderr, "mxkwOpenMXCD failed\n");
        return 2;
    }
    int driver_page = 0;
    MX_PAGE_SIZE driver_page_index{};
    auto page_status = mxkwGetGpuPageSize(&driver_page, &driver_page_index);
    const size_t page = driver_page > 0 ? (size_t)driver_page : 4096;

    /* Node ids are not a dense range; they must be read from the driver's
     * own topology after system properties are acquired. */
    MxSystemProperties properties{};
    auto acquired = mxkwAcquireSystemProperties(&properties);
    std::printf("{\"schema\":\"allocation_physical_mapping_v1\","
                "\"buffer_bytes\":%zu,\"allocations\":%d,"
                "\"driver_page_status\":%d,\"driver_page_bytes\":%d,"
                "\"acquire_status\":%d,\"num_nodes\":%u,\"visible_nodes\":%u,"
                "\"nodes\":[",
                kBufferBytes, kAllocations, (int)page_status, driver_page,
                (int)acquired, properties.NumNodes,
                properties.NumVisibleNodes);
    std::vector<uint32_t> gpu_nodes;
    bool printed = false;
    for (uint32_t n = 0; n < 64; n++) {
        MxNodeProperties node_properties{};
        if (mxkwGetNodeProperties(n, &node_properties) != MXKW_STATUS_SUCCESS)
            continue;
        if (printed) std::printf(",");
        printed = true;
        std::printf("{\"node\":%u,\"name\":\"%s\",\"gpu_id\":%u}",
                    n, node_properties.MarketingName, node_properties.gpu_id);
        if (node_properties.gpu_id != 0) gpu_nodes.push_back(n);
    }
    std::printf("]}\n");
    std::fflush(stdout);
    if (requested >= 0) {
        gpu_nodes.clear();
        gpu_nodes.push_back((uint32_t)requested);
    }
    if (gpu_nodes.empty()) {
        std::fprintf(stderr, "no GPU node reported by the driver\n");
        mxkwReleaseSystemProperties();
        mxkwCloseMXCD();
        return 4;
    }
    const uint32_t node = gpu_nodes.front();

    MxMemFlags flags{};
    flags.ui32.NonPaged = 1;
    flags.ui32.HostAccess = 1;
    flags.ui32.PageSize = MX_PAGE_SIZE_2MB;

    std::vector<void*> buffers;
    for (int i = 0; i < kAllocations; i++) {
        void* base = nullptr;
        auto status = mxkwAllocMemory(node, kBufferBytes, flags, &base);
        if (status != MXKW_STATUS_SUCCESS) {
            std::fprintf(stderr, "mxkwAllocMemory %d on node %u failed: %d\n",
                         i, node, (int)status);
            mxkwReleaseSystemProperties();
            mxkwCloseMXCD();
            return 3;
        }
        std::memset(base, 0xa5, kBufferBytes);
        buffers.push_back(base);
    }
    for (int i = 0; i < kAllocations; i++)
        probe_one(i, buffers[i], kBufferBytes, page, node);
    for (void* base : buffers) mxkwFreeMemory(base, kBufferBytes);

    mxkwReleaseSystemProperties();
    mxkwCloseMXCD();
    return 0;
}
