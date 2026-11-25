/*
 * MetaX C500 concurrent-program phase sharing law benchmark.
 *
 * Every sharing probe this repository has run on C500 fixed the per-program
 * stream length (k depth) at 64 steps while the production grouped-MoE launch
 * takes it from K / BLOCK_K, which is 64, 128, 192 or 224 across the canonical
 * cases.  The k depth sets the reuse distance between two programs that share
 * one object and sit `spacing` apart in dispatch: while the machine advances
 * from one sharer to the next, the resident programs together touch about
 * `spacing * per_program_stream_bytes`, so the shared object survives only
 * while that product stays under the L2 replacement capacity.  The lockstep
 * miss core never expresses this distance: it replays a whole wave at one k
 * step, so the distance is one step regardless of the depth.
 *
 * This probe sweeps the depth and the dispatch-physical axes around the
 * production point of the controlled shape: 90 sharers spaced GROUP_SIZE_M = 8
 * apart on the activation stream, 8 adjacent sharers on the weight stream.
 * The activation stream is the one production realises ~2 passes on where the
 * core credits ~16, and the depth axis is the one quantity all earlier probes
 * held small enough to put the reuse distance under the cache.
 *
 * It carries no routing, no expert blocking, no tiling and no MMA.
 *
 * Traffic counters only.  No duration here may price a Forward path, and the
 * axis-to-traffic relation must not be fitted or interpolated.
 *
 * mcProfiler stops emitting per-kernel reports after about fifty kernels in
 * one marker window, so the sweep is split by MIDLAYER_PROBE_PART (0, 1, 2).
 * Every part carries its own controls.
 *
 * Build inside the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -o concurrent_phase_sharing_law \
 *       concurrent_phase_sharing_law.cpp
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
static const int kDefaultWorkers = 8192;
/* 256 uint4 = 4 KiB per stream per step, the production BM64 x BK32 slab. */
static const int kProductionSliceVectors = 256;

/* Which shared object a program reads from one stream.  A group size of 1
 * gives every program its own object, which is the private (no-sharing)
 * control. */
static __host__ __device__ int stream_gid(int worker, int group, int spacing) {
    if (group <= 1) return worker;
    const int block = spacing * group;
    return (worker / block) * spacing + (worker % spacing);
}

/* Both streams reused, which is what a grouped GEMM issues: the weight tile
 * is shared by the GROUP_SIZE_M adjacent programs of a group and the
 * activation tile by the num_pid_n programs one group width apart.  `steps`
 * is the k depth, `sv` the slice width in uint4, `stages` the software
 * pipeline depth, which sets the shared-memory footprint and with it the
 * resident CTA count. */
__global__ __launch_bounds__(kBlockThreads) void phase_sharing(
    const uint4* __restrict__ buf_a,
    const uint4* __restrict__ buf_b,
    uint32_t* __restrict__ out,
    int steps,
    int group_a,
    int spacing_a,
    int group_b,
    int spacing_b,
    int stages,
    int work,
    int sv
) {
    extern __shared__ uint4 stage[];
    const int worker = blockIdx.x;
    const int lane = threadIdx.x;
    const size_t base_a =
        (size_t)stream_gid(worker, group_a, spacing_a) * steps * sv;
    const size_t base_b =
        (size_t)stream_gid(worker, group_b, spacing_b) * steps * sv;
    const int per_stage = 2 * sv;

    uint32_t checksum = 0u;
    for (int s = 0; s < steps + stages; s++) {
        if (s < steps) {
            const int slot = (s % stages) * per_stage;
            const size_t oa = base_a + (size_t)s * sv;
            const size_t ob = base_b + (size_t)s * sv;
            for (int i = lane; i < sv; i += kBlockThreads) {
                stage[slot + i] = buf_a[oa + i];
            }
            for (int i = lane; i < sv; i += kBlockThreads) {
                stage[slot + sv + i] = buf_b[ob + i];
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
             * races through its k loop, so two co-resident sharers sit on the
             * same step at the same instant; a compute-heavy one drifts.  A
             * few points keep this knob pinned at long depth. */
            float acc = (float)(checksum & 0xffff) + 1.0f;
            for (int i = 0; i < work; i++) {
                acc = acc * 1.0000001f + 0.5f;
            }
            checksum += (uint32_t)acc;
        }
    }
    out[worker] = checksum;
}

struct Config {
    int steps, group_a, spacing_a, group_b, spacing_b, stages, work, sv;
    int workers;   /* 0 keeps the run-wide dispatch width */
    const char* axis;
};

static int distinct_objects(int workers, int group, int spacing) {
    int highest = 0;
    for (int w = 0; w < workers; w++) {
        highest = std::max(highest, stream_gid(w, group, spacing));
    }
    return highest + 1;
}

int main() {
    const char* part_text = std::getenv("MIDLAYER_PROBE_PART");
    const int part = part_text ? std::atoi(part_text) : 0;

    /* The production point of the controlled shape: 90 activation sharers
     * spaced GROUP_SIZE_M = 8 apart, 8 adjacent weight sharers, the BM64
     * per-step slab, and the stages = 2 pipeline whose 16 KiB footprint puts
     * 4 blocks per AP -- the wave_size 416 occupancy. */
    const int P_steps = 192;
    const int P_group_a = 90, P_spacing_a = 8;
    const int P_group_b = 8, P_spacing_b = 1;
    const int P_stages = 2, P_sv = kProductionSliceVectors;

    std::vector<Config> configs;
    /* Controls in every part.  `control_neither` makes both streams private
     * and fixes the compulsory, no-sharing read of each stream;
     * `control_activation_only` and `control_weight_only` isolate one reused
     * stream each, so the per-stream DRAM contribution can be differenced
     * out of the production point. */
    for (int rep = 0; rep < 2; rep++) {
        configs.push_back({P_steps, 1, 1, 1, 1, P_stages, 0, P_sv,
                           0, "control_neither"});
        configs.push_back({P_steps, P_group_a, P_spacing_a, 1, 1, P_stages, 0,
                           P_sv, 0, "control_activation_only"});
        configs.push_back({P_steps, 1, 1, P_group_b, P_spacing_b, P_stages, 0,
                           P_sv, 0, "control_weight_only"});
    }

    if (part == 0) {
        /* Axis K -- the k depth, the per-program stream length.  The
         * activation reuse distance at spacing 8 is 8 * steps * 8 KiB:
         * 1 MiB at steps 16, 4 MiB at 64, 8 MiB at 128, 16 MiB at 256, so
         * the sweep crosses the measured 8 MiB L2 replacement point inside
         * its range.  Everything else stays at the production point. */
        for (int st : {16, 32, 48, 64, 96, 128, 160, 192, 224, 256}) {
            configs.push_back({st, P_group_a, P_spacing_a, P_group_b,
                               P_spacing_b, P_stages, 0, P_sv, 0,
                               "k_depth"});
        }
        /* Repeat the two ends of the axis, to size the run-to-run spread. */
        configs.push_back({64, P_group_a, P_spacing_a, P_group_b, P_spacing_b,
                           P_stages, 0, P_sv, 0, "k_depth_repeat"});
        configs.push_back({192, P_group_a, P_spacing_a, P_group_b,
                           P_spacing_b, P_stages, 0, P_sv, 0,
                           "k_depth_repeat"});
    } else if (part == 1) {
        /* Axis S -- the dispatch spacing between two sharers of one object,
         * which is GROUP_SIZE_M on the activation stream.  The reuse distance
         * is linear in it; sweeping it at the production depth reads the
         * sharing law off a second, independent axis. */
        for (int sp : {1, 2, 4, 8, 16, 32}) {
            configs.push_back({P_steps, P_group_a, sp, P_group_b,
                               P_spacing_b, P_stages, 0, P_sv, 0,
                               "spacing"});
        }
        /* Axis G -- the concurrent address streams sharing one object: the
         * number of sharers per group, which is num_pid_n on the activation
         * stream.  Same bytes per CTA, same footprint, only the share count
         * moves. */
        for (int g : {2, 4, 8, 16, 32, 64, 90}) {
            configs.push_back({P_steps, g, P_spacing_a, P_group_b,
                               P_spacing_b, P_stages, 0, P_sv, 0,
                               "sharers"});
        }
        /* Axis R -- the resident CTA count, through the shared-memory
         * footprint: stages 1/2/3 = 8/16/24 KiB = 8/4/2 blocks per AP =
         * 832/416/208 resident CTAs.  Production runs at stages 2. */
        for (int stg : {1, 2, 3}) {
            configs.push_back({P_steps, P_group_a, P_spacing_a, P_group_b,
                               P_spacing_b, stg, 0, P_sv, 0, "stages"});
        }
        /* The production point itself, at the depth of this part. */
        configs.push_back({P_steps, P_group_a, P_spacing_a, P_group_b,
                           P_spacing_b, P_stages, 0, P_sv, 0,
                           "production_point"});
    } else {
        /* Axis W -- the per-step operand bytes, the BM x BK proxy, swept at
         * a constant shared-memory footprint (stages x sv = 512 uint4), so
         * the resident CTA count does not move with it: sv 128/256/512 with
         * stages 4/2/1.  Per-step bytes per CTA are 4/8/16 KiB. */
        configs.push_back({P_steps, P_group_a, P_spacing_a, P_group_b,
                           P_spacing_b, 4, 0, 128, 0, "slice_bytes"});
        configs.push_back({P_steps, P_group_a, P_spacing_a, P_group_b,
                           P_spacing_b, 2, 0, 256, 0, "slice_bytes"});
        configs.push_back({P_steps, P_group_a, P_spacing_a, P_group_b,
                           P_spacing_b, 1, 0, 512, 0, "slice_bytes"});
        /* Axis F -- the dispatch width, which scales the number of distinct
         * shared objects and the total footprint at a fixed sharer count,
         * spacing, depth and per-CTA bytes. */
        for (int w : {4096, 16384}) {
            configs.push_back({P_steps, P_group_a, P_spacing_a, P_group_b,
                               P_spacing_b, P_stages, 0, P_sv, w,
                               "dispatch_width"});
        }
        /* Axis D -- per-k-step arithmetic, pinned at the long depth. */
        for (int wk : {64, 1024}) {
            configs.push_back({P_steps, P_group_a, P_spacing_a, P_group_b,
                               P_spacing_b, P_stages, wk, P_sv, 0,
                               "work"});
        }
        configs.push_back({P_steps, P_group_a, P_spacing_a, P_group_b,
                           P_spacing_b, P_stages, 0, P_sv, 0,
                           "production_point"});
    }

    int max_workers = kDefaultWorkers;
    size_t max_a = 0, max_b = 0;
    for (const Config& c : configs) {
        const int workers = c.workers ? c.workers : kDefaultWorkers;
        max_workers = std::max(max_workers, workers);
        max_a = std::max(max_a,
            (size_t)distinct_objects(workers, c.group_a, c.spacing_a)
                * c.steps * c.sv);
        max_b = std::max(max_b,
            (size_t)distinct_objects(workers, c.group_b, c.spacing_b)
                * c.steps * c.sv);
    }

    uint4 *buf_a = nullptr, *buf_b = nullptr;
    uint32_t* out = nullptr;
    CHECK(mcMalloc(&buf_a, max_a * sizeof(uint4)));
    CHECK(mcMalloc(&buf_b, max_b * sizeof(uint4)));
    CHECK(mcMalloc(&out, (size_t)max_workers * sizeof(uint32_t)));
    CHECK(mcMemset(buf_a, 0x5a, max_a * sizeof(uint4)));
    CHECK(mcMemset(buf_b, 0xa5, max_b * sizeof(uint4)));
    CHECK(mcMemset(out, 0, (size_t)max_workers * sizeof(uint32_t)));
    CHECK(mcDeviceSynchronize());

    std::printf("{\"schema\":\"concurrent_phase_sharing_law_v1\",");
    std::printf("\"part\":%d,\"default_workers\":%d,\"block_threads\":%d,",
                part, kDefaultWorkers, kBlockThreads);
    std::printf("\"l2_replacement_capacity_bytes\":8388608,");
    std::printf("\"ordinals\":[");
    for (size_t i = 0; i < configs.size(); i++) {
        const Config& c = configs[i];
        const int workers = c.workers ? c.workers : kDefaultWorkers;
        const int da = distinct_objects(workers, c.group_a, c.spacing_a);
        const int db = distinct_objects(workers, c.group_b, c.spacing_b);
        const double slice = (double)c.sv * 16;
        const double per_cta_stream = (double)c.steps * 2 * slice;
        if (i != 0) std::printf(",");
        std::printf("{\"ordinal\":%zu,\"axis\":\"%s\",\"steps\":%d,"
                    "\"group_a\":%d,\"spacing_a\":%d,"
                    "\"group_b\":%d,\"spacing_b\":%d,"
                    "\"stages\":%d,\"work\":%d,\"slice_vectors\":%d,"
                    "\"workers\":%d,"
                    "\"smem_bytes\":%d,"
                    "\"distinct_objects_a\":%d,\"distinct_objects_b\":%d,"
                    "\"per_cta_stream_bytes\":%.0f,"
                    "\"stream_a_reuse_distance_bytes\":%.0f,"
                    "\"logical_a_bytes\":%.0f,\"logical_b_bytes\":%.0f,"
                    "\"unique_a_bytes\":%.0f,\"unique_b_bytes\":%.0f}",
                    i, c.axis, c.steps, c.group_a, c.spacing_a, c.group_b,
                    c.spacing_b, c.stages, c.work, c.sv, workers,
                    c.stages * 2 * c.sv * 16, da, db, per_cta_stream,
                    (double)c.spacing_a * per_cta_stream,
                    (double)workers * c.steps * slice,
                    (double)workers * c.steps * slice,
                    (double)da * c.steps * slice,
                    (double)db * c.steps * slice);
    }
    std::printf("]}\n");
    std::fflush(stdout);

    CHECK(mcProfilerStart());
    for (const Config& c : configs) {
        const size_t smem = (size_t)c.stages * 2 * c.sv * sizeof(uint4);
        phase_sharing<<<c.workers ? c.workers : kDefaultWorkers,
                        kBlockThreads, smem>>>(
            buf_a, buf_b, out, c.steps, c.group_a, c.spacing_a, c.group_b,
            c.spacing_b, c.stages, c.work, c.sv);
        CHECK(mcGetLastError());
        CHECK(mcDeviceSynchronize());
    }
    CHECK(mcProfilerStop());

    CHECK(mcFree(buf_a));
    CHECK(mcFree(buf_b));
    CHECK(mcFree(out));
    return 0;
}
