/*
 * MetaX C500 co-scheduled group sharing residency benchmark.
 *
 * The shared miss core prices a grouped-MoE program by interleaving a wave of
 * co-scheduled CTAs in lock step: at each k step every CTA of the wave touches
 * its operands, so the CTAs of one group touch the *same* weight tile at
 * adjacent positions in the reference stream and the sharing is granted for
 * free, independent of capacity.  A replay of the production stream built that
 * way predicts about twice the reuse the hardware realises, and that gap
 * survives the capacity (8 MiB down to 1 MiB), the wave width across its whole
 * range, the co-resident phase alignment and the operand double-read phase.
 *
 * Lock step makes concurrency a sharing amplifier.  Physically concurrency is
 * also an interference amplifier: between one CTA's touch of a shared tile and
 * another's, every other resident CTA issues its own traffic.  Which of the
 * two dominates on this target has never been measured.
 *
 * This probe measures it directly.  `workers` CTAs are partitioned into groups
 * of `group` CTAs.  Every CTA in a group reads the same shared slice, and each
 * CTA also reads a private slice of the same width, so the logical bytes per
 * CTA are identical for every group size.  Ideal sharing costs one DRAM fetch
 * of each shared slice per group; no sharing costs one per CTA.  Sweeping the
 * group size at a fixed footprint therefore reads off the realised sharing.
 *
 * It carries no routing, no expert blocking, no tiling and no MMA.
 *
 * Traffic counters only.  No duration here may price a Forward path, and the
 * group-to-traffic relation must not be fitted or interpolated.
 *
 * Build inside the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -o cooperative_group_sharing_residency \
 *       cooperative_group_sharing_residency.cpp
 */

#include <mc_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do {                                                   \
    mcError_t error = call;                                                \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,             \
                     mcGetErrorString(error));                             \
        std::exit(1);                                                      \
    }                                                                      \
} while (0)

static const int kBlockThreads = 256;
static int g_workers = 512;
static const int kMaxWorkers = 32768;
/* One slice is 16 uint4 per thread-block pass = 4 KiB, the BK x BN operand
 * slab the BM64 launch issues per k step. */
static const int kVectorsPerSlice = 256;

/* `steps` slices are walked in order, exactly as a k loop does.  `shared`
 * holds the slices every CTA of a group reads; `priv` holds the per-CTA
 * ones.  Both are walked in the same loop so the two streams interleave the
 * way a real A/B pair does. */
__global__ __launch_bounds__(kBlockThreads) void group_sharing(
    const uint4* __restrict__ shared_buf,
    const uint4* __restrict__ private_buf,
    uint32_t* __restrict__ out,
    int steps,
    int group,
    int strided,
    int row_stride_vectors,
    int stages
) {
    const int worker = blockIdx.x;
    /* `strided` places a group's members `groups` apart in blockIdx instead
     * of consecutively.  Same groups, same bytes, same order within a CTA --
     * only the dispatch distance between two sharers changes, which is the
     * one thing that separates the production weight sharing (adjacent
     * programs of a group) from the production activation sharing (programs
     * one group-width apart). */
    /* `strided` is the dispatch spacing between two members of a group: 1
     * places them consecutively, s places them s apart.  The group count and
     * every byte read are unchanged; only the distance moves. */
    const int spacing = strided < 1 ? 1 : strided;
    /* Members of a group sit `spacing` apart inside a block of
     * `spacing * group` consecutive programs, so the span between a group's
     * first and last member is `spacing * (group - 1)`.  Sweeping the group
     * size at a fixed spacing therefore sweeps the span. */
    const int block = spacing * group;
    const int gid = (worker / block) * spacing + (worker % spacing);
    const int lane = threadIdx.x;
    const size_t shared_base = (size_t)gid * steps
        * (row_stride_vectors == 0 ? kVectorsPerSlice : 32 * row_stride_vectors);
    const size_t private_base = (size_t)worker * steps * kVectorsPerSlice;

    uint32_t checksum = 0u;
    /* `row_stride_vectors` == 0 walks the shared slice contiguously.  A
     * positive value walks it as 32 rows of 8 uint4 (128 B) spaced that many
     * uint4 apart, which is how a grouped GEMM reads a BK x BN weight tile
     * out of a [K, N] matrix.  Same bytes, same count, same order within the
     * CTA -- only the address spacing between the tile's rows changes. */
    const int kRowsPerSlice = 32;
    const int kVectorsPerRow = kVectorsPerSlice / kRowsPerSlice;
    for (int s = 0; s < steps; s++) {
        const size_t so = shared_base + (size_t)s * kVectorsPerSlice;
        const size_t po = private_base + (size_t)s * kVectorsPerSlice;
        if (row_stride_vectors == 0) {
            for (int i = lane; i < kVectorsPerSlice; i += kBlockThreads) {
                const uint4 a = shared_buf[so + i];
                checksum += a.x ^ a.w;
            }
        } else {
            for (int i = lane; i < kVectorsPerSlice; i += kBlockThreads) {
                const int row = i / kVectorsPerRow;
                const int col = i % kVectorsPerRow;
                const size_t off = shared_base
                    + (size_t)(s * kRowsPerSlice + row) * row_stride_vectors
                    + col;
                const uint4 a = shared_buf[off];
                checksum += a.x ^ a.w;
            }
        }
        for (int i = lane; i < kVectorsPerSlice; i += kBlockThreads) {
            const uint4 b = private_buf[po + i];
            checksum += b.y ^ b.z;
        }
    }
    out[worker] = checksum;
}

__global__ __launch_bounds__(kBlockThreads) void group_sharing_staged(
    const uint4* __restrict__ shared_buf,
    const uint4* __restrict__ private_buf,
    uint32_t* __restrict__ out,
    int steps,
    int group,
    int strided,
    int stages
) {
    extern __shared__ uint4 stage[];
    const int worker = blockIdx.x;
    const int spacing = strided < 1 ? 1 : strided;
    const int block = spacing * group;
    const int gid = (worker / block) * spacing + (worker % spacing);
    const int lane = threadIdx.x;
    const size_t shared_base = (size_t)gid * steps * kVectorsPerSlice;
    const size_t private_base = (size_t)worker * steps * kVectorsPerSlice;
    const int per_stage = 2 * kVectorsPerSlice;   /* one shared + one private */

    uint32_t checksum = 0u;
    for (int s = 0; s < steps + stages; s++) {
        /* Issue stage s. */
        if (s < steps) {
            const int slot = (s % stages) * per_stage;
            const size_t so = shared_base + (size_t)s * kVectorsPerSlice;
            const size_t po = private_base + (size_t)s * kVectorsPerSlice;
            for (int i = lane; i < kVectorsPerSlice; i += kBlockThreads) {
                stage[slot + i] = shared_buf[so + i];
            }
            for (int i = lane; i < kVectorsPerSlice; i += kBlockThreads) {
                stage[slot + kVectorsPerSlice + i] = private_buf[po + i];
            }
        }
        /* Consume stage s - (stages - 1), so `stages` slices are in flight. */
        const int c = s - (stages - 1);
        if (c >= 0 && c < steps) {
            __syncthreads();
            const int slot = (c % stages) * per_stage;
            for (int i = lane; i < per_stage; i += kBlockThreads) {
                const uint4 v = stage[slot + i];
                checksum += v.x ^ v.w;
            }
        }
    }
    out[worker] = checksum;
}

/* Which shared object a program reads from one stream.  A group size of 1
 * gives every program its own object, which is the private (no-sharing)
 * control. */
static __host__ __device__ int stream_gid(int worker, int group, int strided) {
    if (group <= 1) return worker;
    const int spacing = strided < 1 ? 1 : strided;
    const int block = spacing * group;
    return (worker / block) * spacing + (worker % spacing);
}

/* Both streams reused, which is what a grouped GEMM actually issues: the
 * weight tile is shared by the GROUP_SIZE_M adjacent programs of a group and
 * the activation tile by the num_pid_n programs one group width apart.  The
 * single-shared-stream kernel above pairs one reused stream with a strictly
 * private one, so it can never show one reused stream evicting the other.
 * The miss core credits the two streams separately and never expresses their
 * mutual eviction either, so this is the one structural difference between
 * the probe and production that the probe has not carried. */
__global__ __launch_bounds__(kBlockThreads) void two_stream_sharing(
    const uint4* __restrict__ buf_a,
    const uint4* __restrict__ buf_b,
    uint32_t* __restrict__ out,
    int steps,
    int group_a,
    int spacing_a,
    int group_b,
    int spacing_b,
    int stages,
    int work
) {
    extern __shared__ uint4 stage[];
    const int worker = blockIdx.x;
    const int lane = threadIdx.x;
    const size_t base_a =
        (size_t)stream_gid(worker, group_a, spacing_a) * steps * kVectorsPerSlice;
    const size_t base_b =
        (size_t)stream_gid(worker, group_b, spacing_b) * steps * kVectorsPerSlice;
    const int per_stage = 2 * kVectorsPerSlice;

    uint32_t checksum = 0u;
    for (int s = 0; s < steps + stages; s++) {
        if (s < steps) {
            const int slot = (s % stages) * per_stage;
            const size_t oa = base_a + (size_t)s * kVectorsPerSlice;
            const size_t ob = base_b + (size_t)s * kVectorsPerSlice;
            for (int i = lane; i < kVectorsPerSlice; i += kBlockThreads) {
                stage[slot + i] = buf_a[oa + i];
            }
            for (int i = lane; i < kVectorsPerSlice; i += kBlockThreads) {
                stage[slot + kVectorsPerSlice + i] = buf_b[ob + i];
            }
        }
        const int c = s - (stages - 1);
        if (c >= 0 && c < steps) {
            __syncthreads();
            const int slot = (c % stages) * per_stage;
            for (int i = lane; i < per_stage; i += kBlockThreads) {
                const uint4 v = stage[slot + i];
                checksum += v.x ^ v.w;
            }
            /* `work` dependent FMAs per thread per k step.  A pure-load CTA
             * races through its k loop, so two co-resident sharers sit on
             * the same step at the same instant and their requests can merge
             * in flight -- which is how this probe realises sharing across a
             * reuse distance many times the cache.  A real GEMM spends most
             * of each k step in the MMA, so two sharers that entered the
             * machine at different times drift out of step and never issue
             * the same address together.  This knob is the only difference
             * left between the probe and the production kernel. */
            float acc = (float)(checksum & 0xffff) + 1.0f;
            for (int i = 0; i < work; i++) {
                acc = acc * 1.0000001f + 0.5f;
            }
            checksum += (uint32_t)acc;
        }
    }
    out[worker] = checksum;
}

/* The activation tile of a grouped MoE GEMM is not a slice.  It is a gather
 * of BLOCK_SIZE_M token rows named by `sorted_token_ids`, and at BK = 32 in
 * bf16 each row contributes 64 B per k step -- half a 128 B line -- while
 * consecutive k steps walk that row contiguously.  The weight tile stays a
 * regular strided slab.  Both streams above are regular slices, so the one
 * production access shape this probe has never carried is the gather.
 *
 * Bytes and lines are held identical to the slice case: 64 rows of
 * `steps` * 64 B each is the same byte count and, because a row's k stream is
 * contiguous, the same 32 * `steps` lines.  Only where those lines sit
 * changes.  `scatter` = 0 lays an object's rows next to each other;
 * `scatter` = 1 transposes the row-to-slot map so an object's rows are one
 * object-count apart, spread over the whole row space.  Both are bijections
 * over the same slots, so the footprint is bit-for-bit the same too. */
__global__ __launch_bounds__(kBlockThreads) void gather_stream_sharing(
    const uint4* __restrict__ buf_a,
    const uint4* __restrict__ buf_b,
    uint32_t* __restrict__ out,
    int steps,
    int group_a,
    int spacing_a,
    int group_b,
    int spacing_b,
    int stages,
    const int32_t* __restrict__ row_slot,
    int objects
) {
    extern __shared__ uint4 stage[];
    const int worker = blockIdx.x;
    const int lane = threadIdx.x;
    const int gid_a = stream_gid(worker, group_a, spacing_a);
    const size_t base_b =
        (size_t)stream_gid(worker, group_b, spacing_b) * steps * kVectorsPerSlice;
    const int per_stage = 2 * kVectorsPerSlice;
    /* 256 threads over 64 rows: four threads per row, one uint4 each, which
     * is the 64 B a row contributes to one k step. */
    const int kRows = 64;
    const int row = lane >> 2;
    const int chunk = lane & 3;
    /* The map is built on the host so a scattered layout can be a genuine
     * permutation rather than an arithmetic progression.  A transposed map
     * spaces an object's rows by a fixed slot stride, and a fixed stride
     * lands them on few L2 sets, which would collapse reuse for a reason
     * that has nothing to do with the gather.  Reading the map through a
     * table costs every configuration the same small lookup. */
    const int slot = row_slot[gid_a * kRows + row];
    const size_t row_base = (size_t)slot * steps * 4;

    uint32_t checksum = 0u;
    for (int s = 0; s < steps + stages; s++) {
        if (s < steps) {
            const int st = (s % stages) * per_stage;
            const uint4 a = buf_a[row_base + (size_t)s * 4 + chunk];
            stage[st + lane] = a;
            const size_t ob = base_b + (size_t)s * kVectorsPerSlice;
            for (int i = lane; i < kVectorsPerSlice; i += kBlockThreads) {
                stage[st + kVectorsPerSlice + i] = buf_b[ob + i];
            }
        }
        const int c = s - (stages - 1);
        if (c >= 0 && c < steps) {
            __syncthreads();
            const int st = (c % stages) * per_stage;
            for (int i = lane; i < per_stage; i += kBlockThreads) {
                const uint4 v = stage[st + i];
                checksum += v.x ^ v.w;
            }
        }
    }
    out[worker] = checksum;
}

struct Config { int steps, group, strided, row_stride, stages; const char* axis; };
struct PairConfig {
    int steps, group_a, spacing_a, group_b, spacing_b, stages, work;
    const char* axis;
    int workers;   /* 0 keeps the run-wide dispatch width */
};
/* 0 lays an object's rows consecutively, 1 transposes them onto a fixed
 * slot stride, 2 uses a shuffled permutation over the same slots.  All
 * three are bijections over an identical slot set, so the footprint, the
 * byte count and the line count are the same. */
struct GatherConfig {
    int steps, group_a, spacing_a, group_b, spacing_b, stages, scatter;
    const char* axis;
};

static std::vector<int32_t> build_row_slot_map(int objects, int scatter) {
    const int kRows = 64;
    std::vector<int32_t> map((size_t)objects * kRows);
    for (int g = 0; g < objects; g++) {
        for (int r = 0; r < kRows; r++) {
            map[(size_t)g * kRows + r] =
                scatter == 0 ? g * kRows + r : r * objects + g;
        }
    }
    if (scatter == 2) {
        /* Shuffle the slots, then hand them out in the same order, so an
         * object's rows are spread without sitting on any stride. */
        std::vector<int32_t> slots((size_t)objects * kRows);
        for (size_t i = 0; i < slots.size(); i++) slots[i] = (int32_t)i;
        uint64_t state = 0x9e3779b97f4a7c15ull;
        for (size_t i = slots.size(); i > 1; i--) {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            std::swap(slots[i - 1], slots[state % i]);
        }
        for (size_t i = 0; i < map.size(); i++) map[i] = slots[i];
    }
    return map;
}

static int distinct_objects(int workers, int group, int strided) {
    int highest = 0;
    for (int w = 0; w < workers; w++) {
        highest = std::max(highest, stream_gid(w, group, strided));
    }
    return highest + 1;
}

int main() {
    /* One dispatch width for every configuration.  The grid axis was swept
     * separately; holding it fixed here keeps the emitted geometry and the
     * executed launches in step, which an earlier revision of this file did
     * not. */
    g_workers = 8192;

    std::vector<Config> configs;
    /* Axis A -- the adjacent-sharing ceiling.  Sharers consecutive, group
     * size swept.  Ideal sharing equals the group size, so a realised value
     * that stops tracking the group size is a ceiling on how many co-
     * scheduled programs can share one object, independent of how many want
     * to.  Never measured: every earlier run of this probe fixed group = 8.
     *
     * A grouped GEMM with GROUP_SIZE_M = 1 makes pid_n the fastest axis, so
     * its activation tile has num_pid_n adjacent sharers -- 90 for the
     * controlled shape -- and that is the production point this axis has to
     * reach. */
    for (int g : {1, 2, 4, 8, 16, 32, 64, 90, 128, 256}) {
        configs.push_back({64, g, 1, 0, 3, "adjacent_group"});
    }
    /* Axis B -- the production activation-sharing point.  The group is the
     * number of N tiles and the spacing is GROUP_SIZE_M, both read off the
     * launch, not off any error. */
    for (int s : {1, 2, 4, 8, 16}) {
        configs.push_back({64, 90, s, 0, 3, "spaced_group90"});
    }
    /* Repeats of the two ends of axis A, to size the run-to-run spread the
     * way the spacing sweep did. */
    configs.push_back({64, 8, 1, 0, 3, "adjacent_group_repeat"});
    configs.push_back({64, 90, 1, 0, 3, "adjacent_group_repeat"});

    /* Axis C -- both streams reused, at the production structure.  The
     * activation tile has num_pid_n = 90 sharers one GROUP_SIZE_M = 8 apart;
     * the weight tile has 8 adjacent sharers.  A group size of 1 is the
     * private control, so the two single-stream controls below reproduce the
     * corresponding axis A and axis B points and tie the two kernels
     * together. */
    std::vector<PairConfig> pairs;
    pairs.push_back({64, 90, 8, 8, 1, 3, 0, "two_stream_production", 0});
    pairs.push_back({64, 90, 8, 1, 1, 3, 0, "control_activation_only", 0});
    pairs.push_back({64, 1, 1, 8, 1, 3, 0, "control_weight_only", 0});
    pairs.push_back({64, 1, 1, 1, 1, 3, 0, "control_neither", 0});
    pairs.push_back({64, 90, 8, 8, 1, 3, 0, "two_stream_production_repeat", 0});
    /* Axis D -- the same production sharing structure, swept from pure load
     * to compute bound.  Everything but the arithmetic per k step is held
     * fixed: same groups, same spacings, same bytes, same order. */
    for (int w : {4, 16, 64, 256, 1024, 4096}) {
        pairs.push_back({64, 90, 8, 8, 1, 3, w, "two_stream_work", 0});
    }
    /* Axis F -- the shared working set, at a fixed sharing structure.  Every
     * axis so far held the shared footprint where one dispatch width put it,
     * and production's error grows with the activation buffer.  Scaling the
     * dispatch width at a fixed group, spacing and per-program byte count
     * scales the number of distinct shared objects in exact proportion and
     * changes nothing else: the sharers of one object stay ninety and stay
     * eight apart.  The production activation buffer reaches 188 MiB. */
    for (int w : {8192, 16384, 32768, 65536}) {
        pairs.push_back({64, 90, 8, 8, 1, 3, 0, "two_stream_working_set", w});
    }

    /* The spaced stream alone, over the same sweep, to read the activation
     * term off directly. */
    for (int w : {64, 1024, 4096}) {
        pairs.push_back({64, 90, 8, 1, 1, 3, w, "activation_only_work", 0});
    }

    /* Axis E -- the activation stream as the gather it actually is.  Rows
     * adjacent against rows scattered, at the two production sharing points:
     * the GROUP_SIZE_M = 8 launch, where the activation sharers sit eight
     * apart, and the GROUP_SIZE_M = 1 launch, where they are adjacent and
     * ninety of them.  Identical bytes, identical line count, identical
     * sharers; only where the lines sit differs. */
    std::vector<GatherConfig> gathers;
    /* Alternate the two layouts and repeat each three times.  A first pass
     * of this axis had one scattered configuration disagree with its own
     * repeat by a factor of 1.8 while every other repeat in this probe
     * agreed to three figures, so the pair has to be interleaved and
     * replicated before either reading can be believed. */
    for (int rep = 0; rep < 3; rep++) {
        gathers.push_back({64, 90, 8, 8, 1, 3, 0, "gather_production_rows_adjacent"});
        gathers.push_back({64, 90, 8, 8, 1, 3, 1, "gather_production_rows_strided"});
        gathers.push_back({64, 90, 8, 8, 1, 3, 2, "gather_production_rows_shuffled"});
    }
    for (int rep = 0; rep < 2; rep++) {
        gathers.push_back({64, 90, 1, 1, 1, 3, 0, "gather_adjacent90_rows_adjacent"});
        gathers.push_back({64, 90, 1, 1, 1, 3, 2, "gather_adjacent90_rows_shuffled"});
    }

    int max_workers = g_workers;
    for (const PairConfig& p : pairs) {
        max_workers = std::max(max_workers, p.workers ? p.workers : g_workers);
    }
    size_t max_shared = 0, max_private = 0;
    for (const PairConfig& p : pairs) {
        const int w = p.workers ? p.workers : g_workers;
        max_shared = std::max(max_shared,
            (size_t)distinct_objects(w, p.group_a, p.spacing_a)
                * p.steps * kVectorsPerSlice);
        max_private = std::max(max_private,
            (size_t)distinct_objects(w, p.group_b, p.spacing_b)
                * p.steps * kVectorsPerSlice);
    }
    for (const GatherConfig& g : gathers) {
        max_shared = std::max(max_shared,
            (size_t)distinct_objects(g_workers, g.group_a, g.spacing_a)
                * 64 * g.steps * 4);
        max_private = std::max(max_private,
            (size_t)distinct_objects(g_workers, g.group_b, g.spacing_b)
                * g.steps * kVectorsPerSlice);
    }
    for (const PairConfig& p : pairs) {
        max_shared = std::max(max_shared,
            (size_t)distinct_objects(g_workers, p.group_a, p.spacing_a)
                * p.steps * kVectorsPerSlice);
        max_private = std::max(max_private,
            (size_t)distinct_objects(g_workers, p.group_b, p.spacing_b)
                * p.steps * kVectorsPerSlice);
    }
    for (const Config& c : configs) {
        max_shared = std::max(max_shared,
            (size_t)distinct_objects(g_workers, c.group, c.strided)
                * c.steps * kVectorsPerSlice);
        max_private = std::max(max_private,
            (size_t)g_workers * c.steps * kVectorsPerSlice);
    }
    /* Build every row-to-slot map up front and upload them once, before the
     * profiler window opens.  Uploading between launches inside the window
     * made mcProfiler drop the whole per-kernel report. */
    std::vector<size_t> map_offset;
    std::vector<int32_t> all_maps;
    for (const GatherConfig& g : gathers) {
        const int objects =
            distinct_objects(g_workers, g.group_a, g.spacing_a);
        const std::vector<int32_t> map = build_row_slot_map(objects, g.scatter);
        map_offset.push_back(all_maps.size());
        all_maps.insert(all_maps.end(), map.begin(), map.end());
    }
    int32_t* row_slot = nullptr;
    CHECK(mcMalloc(&row_slot,
                   std::max(all_maps.size(), (size_t)1) * sizeof(int32_t)));
    if (!all_maps.empty()) {
        CHECK(mcMemcpy(row_slot, all_maps.data(),
                       all_maps.size() * sizeof(int32_t),
                       mcMemcpyHostToDevice));
    }
    CHECK(mcDeviceSynchronize());

    uint4 *shared_buf = nullptr, *private_buf = nullptr;
    uint32_t* out = nullptr;
    CHECK(mcMalloc(&shared_buf, max_shared * sizeof(uint4)));
    CHECK(mcMalloc(&private_buf, max_private * sizeof(uint4)));
    CHECK(mcMalloc(&out, (size_t)max_workers * sizeof(uint32_t)));
    CHECK(mcMemset(shared_buf, 0x5a, max_shared * sizeof(uint4)));
    CHECK(mcMemset(private_buf, 0xa5, max_private * sizeof(uint4)));
    CHECK(mcMemset(out, 0, (size_t)max_workers * sizeof(uint32_t)));
    CHECK(mcDeviceSynchronize());

    std::printf("{\"schema\":\"cooperative_group_sharing_residency_v2\",");
    std::printf("\"workers\":%d,\"block_threads\":%d,\"slice_bytes\":%d,",
                g_workers, kBlockThreads, kVectorsPerSlice * 16);
    std::printf("\"ordinals\":[");
    for (size_t i = 0; i < configs.size(); i++) {
        const Config& c = configs[i];
        const int groups = distinct_objects(g_workers, c.group, c.strided);
        const double slice = (double)kVectorsPerSlice * 16;
        if (i != 0) std::printf(",");
        std::printf("{\"ordinal\":%zu,\"axis\":\"%s\",\"steps\":%d,"
                    "\"group\":%d,\"spacing\":%d,\"stages\":%d,"
                    "\"distinct_groups\":%d,"
                    "\"shared_unique_bytes\":%.0f,"
                    "\"private_unique_bytes\":%.0f,"
                    "\"shared_logical_bytes\":%.0f}",
                    i, c.axis, c.steps, c.group,
                    c.strided < 1 ? 1 : c.strided, c.stages, groups,
                    (double)groups * c.steps * slice,
                    (double)g_workers * c.steps * slice,
                    (double)g_workers * c.steps * slice);
    }
    for (size_t j = 0; j < pairs.size(); j++) {
        const PairConfig& p = pairs[j];
        const int pw = p.workers ? p.workers : g_workers;
        const int da = distinct_objects(pw, p.group_a, p.spacing_a);
        const int db = distinct_objects(pw, p.group_b, p.spacing_b);
        const double slice = (double)kVectorsPerSlice * 16;
        std::printf(",{\"ordinal\":%zu,\"axis\":\"%s\",\"steps\":%d,"
                    "\"group\":%d,\"spacing\":%d,"
                    "\"group_b\":%d,\"spacing_b\":%d,\"stages\":%d,"
                    "\"work\":%d,"
                    "\"distinct_groups\":%d,\"distinct_groups_b\":%d,"
                    "\"shared_unique_bytes\":%.0f,"
                    "\"private_unique_bytes\":%.0f,"
                    "\"shared_logical_bytes\":%.0f}",
                    configs.size() + j, p.axis, p.steps,
                    p.group_a, p.spacing_a, p.group_b, p.spacing_b, p.stages,
                    p.work, da, db,
                    (double)da * p.steps * slice,
                    (double)db * p.steps * slice,
                    (double)pw * p.steps * slice);
    }
    for (size_t j = 0; j < gathers.size(); j++) {
        const GatherConfig& g = gathers[j];
        const int da = distinct_objects(g_workers, g.group_a, g.spacing_a);
        const int db = distinct_objects(g_workers, g.group_b, g.spacing_b);
        const double slice = (double)kVectorsPerSlice * 16;
        std::printf(",{\"ordinal\":%zu,\"axis\":\"%s\",\"steps\":%d,"
                    "\"group\":%d,\"spacing\":%d,"
                    "\"group_b\":%d,\"spacing_b\":%d,\"stages\":%d,"
                    "\"work\":%d,\"scatter\":%d,"
                    "\"distinct_groups\":%d,\"distinct_groups_b\":%d,"
                    "\"shared_unique_bytes\":%.0f,"
                    "\"private_unique_bytes\":%.0f,"
                    "\"shared_logical_bytes\":%.0f}",
                    configs.size() + pairs.size() + j, g.axis, g.steps,
                    g.group_a, g.spacing_a, g.group_b, g.spacing_b, g.stages,
                    0, g.scatter, da, db,
                    (double)da * g.steps * slice,
                    (double)db * g.steps * slice,
                    (double)g_workers * g.steps * slice);
    }
    std::printf("]}\n");
    std::fflush(stdout);

    CHECK(mcProfilerStart());
    for (const Config& c : configs) {
        const size_t smem =
            (size_t)c.stages * 2 * kVectorsPerSlice * sizeof(uint4);
        group_sharing_staged<<<g_workers, kBlockThreads, smem>>>(
            shared_buf, private_buf, out, c.steps, c.group, c.strided,
            c.stages);
        CHECK(mcGetLastError());
        CHECK(mcDeviceSynchronize());
    }
    for (const PairConfig& p : pairs) {
        const size_t smem =
            (size_t)p.stages * 2 * kVectorsPerSlice * sizeof(uint4);
        two_stream_sharing<<<p.workers ? p.workers : g_workers,
                             kBlockThreads, smem>>>(
            shared_buf, private_buf, out, p.steps, p.group_a, p.spacing_a,
            p.group_b, p.spacing_b, p.stages, p.work);
        CHECK(mcGetLastError());
        CHECK(mcDeviceSynchronize());
    }
    for (size_t j = 0; j < gathers.size(); j++) {
        const GatherConfig& g = gathers[j];
        const size_t smem =
            (size_t)g.stages * 2 * kVectorsPerSlice * sizeof(uint4);
        const int objects =
            distinct_objects(g_workers, g.group_a, g.spacing_a);
        gather_stream_sharing<<<g_workers, kBlockThreads, smem>>>(
            shared_buf, private_buf, out, g.steps, g.group_a, g.spacing_a,
            g.group_b, g.spacing_b, g.stages, row_slot + map_offset[j],
            objects);
        CHECK(mcGetLastError());
        CHECK(mcDeviceSynchronize());
    }
    CHECK(mcProfilerStop());

    CHECK(mcFree(row_slot));
    CHECK(mcFree(shared_buf));
    CHECK(mcFree(private_buf));
    CHECK(mcFree(out));
    return 0;
}
