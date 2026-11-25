#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct summary_plan {
    int32_t occurrence_count;
    int32_t call_count;
    int32_t resource_count;
    int32_t digest_count;
    int32_t repeat_count;
    const int32_t *call_index;
    const int32_t *dependency_offsets;
    const int32_t *dependencies;
    const int32_t *resource_offsets;
    const int32_t *resources;
    const int32_t *repeat_slot;
    const int32_t *repeat_first;
    const int32_t *repeat_last;
    const int32_t *digest_slot;
};

struct host_program {
    int32_t lane_count;
    int32_t identity_count;
    int32_t operation_count;
    const int32_t *offsets;
    const int32_t *lanes;
    const int32_t *kinds;
    const int32_t *identities;
    const int32_t *sync_before_replay;
    const int32_t *extra_enqueues;
    const double *services;
};

static void store_big_endian_double(unsigned char *output, double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    for (int index = 0; index < 8; ++index) {
        output[index] = (unsigned char)(bits >> (56 - 8 * index));
    }
}

int midlayer_summary_timeline(const struct summary_plan *plan,
        const double *durations, double origin, int host_timing,
        const double *static_host_ready, const double *launch_gaps,
        const struct host_program *host,
        double *maximum_end, double *maximum_device_end, double *repeat_bounds,
        double *launch_cpu_time_s, double *launch_critical_path_s,
        int32_t *eager_dispatch_count, int32_t *graph_replay_count, int32_t *launch_kinds,
        unsigned char **interval_records, size_t **record_counts) {
    const int32_t occurrences = plan->occurrence_count;
    const int32_t digests = plan->digest_count;
    const int32_t repeats = plan->repeat_count;
    double *resource_end = NULL, *device_resource_end = NULL;
    double *end_by_slot = NULL, *device_end_by_slot = NULL;
    unsigned char *records = NULL;
    size_t *counts = NULL;
    unsigned char *first_seen = NULL, *last_seen = NULL;
    double *host_ready = NULL, *host_service = NULL, *graph_ready = NULL;
    unsigned char *graph_seen = NULL;
    int status = 1;
    *maximum_end = origin;
    *maximum_device_end = origin;
    *launch_cpu_time_s = 0.0;
    *launch_critical_path_s = 0.0;
    *eager_dispatch_count = 0;
    *graph_replay_count = 0;
    *launch_kinds = 0;
    *interval_records = NULL;
    *record_counts = NULL;
    if (occurrences <= 0 || plan->call_count <= 0 || plan->resource_count <= 0
            || digests < 0 || repeats < 0) return 2;
    resource_end = malloc((size_t)plan->resource_count * sizeof(*resource_end));
    end_by_slot = malloc((size_t)occurrences * sizeof(*end_by_slot));
    counts = calloc((size_t)(digests > 0 ? digests : 1), sizeof(*counts));
    records = calloc((size_t)(digests > 0 ? digests : 1) * (size_t)occurrences * 16, 1);
    first_seen = calloc((size_t)(repeats > 0 ? repeats : 1), 1);
    last_seen = calloc((size_t)(repeats > 0 ? repeats : 1), 1);
    if (host_timing) {
        device_resource_end = malloc((size_t)plan->resource_count * sizeof(*device_resource_end));
        device_end_by_slot = malloc((size_t)occurrences * sizeof(*device_end_by_slot));
    }
    if (host != NULL) {
        host_ready = calloc((size_t)(host->lane_count > 0 ? host->lane_count : 1), sizeof(*host_ready));
        host_service = calloc((size_t)(host->lane_count > 0 ? host->lane_count : 1), sizeof(*host_service));
        graph_ready = calloc((size_t)(host->identity_count > 0 ? host->identity_count : 1), sizeof(*graph_ready));
        graph_seen = calloc((size_t)(host->identity_count > 0 ? host->identity_count : 1), 1);
    }
    if (resource_end == NULL || end_by_slot == NULL || counts == NULL || records == NULL
            || first_seen == NULL || last_seen == NULL
            || (host_timing && (device_resource_end == NULL || device_end_by_slot == NULL))
            || (host != NULL && (host_ready == NULL || host_service == NULL
                || graph_ready == NULL || graph_seen == NULL))) {
        status = 3;
        goto done;
    }
    for (int32_t resource = 0; resource < plan->resource_count; ++resource) {
        resource_end[resource] = origin;
        if (host_timing) device_resource_end[resource] = origin;
    }
    for (int32_t lane = 0; host != NULL && lane < host->lane_count; ++lane) {
        host_ready[lane] = origin;
    }
    for (int32_t slot = 0; slot < occurrences; ++slot) {
        const int32_t index = plan->call_index[slot];
        if (index < 0 || index >= plan->call_count) { status = 4; goto done; }
        const double duration = durations[index];
        const int32_t resource_begin = plan->resource_offsets[slot];
        const int32_t resource_stop = plan->resource_offsets[slot + 1];
        const int32_t dependency_begin = plan->dependency_offsets[slot];
        const int32_t dependency_stop = plan->dependency_offsets[slot + 1];
        if (resource_begin < 0 || resource_stop < resource_begin
                || resource_stop - resource_begin < 1
                || dependency_begin < 0 || dependency_stop < dependency_begin) {
            status = 5;
            goto done;
        }
        const int32_t *resources = plan->resources + resource_begin;
        if (resources[0] < 0 || resources[0] >= plan->resource_count) { status = 6; goto done; }
        double ready = resource_end[resources[0]];
        double device_ready = host_timing ? device_resource_end[resources[0]] : 0.0;
        for (int32_t cursor = 1; cursor < resource_stop - resource_begin; ++cursor) {
            const int32_t resource = resources[cursor];
            if (resource < 0 || resource >= plan->resource_count) { status = 6; goto done; }
            if (resource_end[resource] > ready) ready = resource_end[resource];
            if (host_timing && device_resource_end[resource] > device_ready) {
                device_ready = device_resource_end[resource];
            }
        }
        for (int32_t cursor = dependency_begin; cursor < dependency_stop; ++cursor) {
            const int32_t dependency = plan->dependencies[cursor];
            if (dependency < 0 || dependency >= slot) { status = 7; goto done; }
            if (end_by_slot[dependency] > ready) ready = end_by_slot[dependency];
            if (host_timing && device_end_by_slot[dependency] > device_ready) {
                device_ready = device_end_by_slot[dependency];
            }
        }
        if (host_timing) {
            ready += launch_gaps[slot];
            device_ready += launch_gaps[slot];
            const double device_end = device_ready + duration;
            device_end_by_slot[slot] = device_end;
            if (device_end > *maximum_device_end) *maximum_device_end = device_end;
            for (int32_t cursor = 0; cursor < resource_stop - resource_begin; ++cursor) {
                device_resource_end[resources[cursor]] = device_end;
            }
            if (host != NULL) {
                for (int32_t operation = host->offsets[slot]; operation < host->offsets[slot + 1]; ++operation) {
                    const int32_t lane = host->lanes[operation];
                    const int32_t kind = host->kinds[operation];
                    const int32_t identity = host->identities[operation];
                    if (lane < 0 || lane >= host->lane_count || kind < 0 || kind > 2) {
                        status = 10;
                        goto done;
                    }
                    double host_start = host_ready[lane];
                    double host_end;
                    if (kind == 2 && identity >= 0 && graph_seen[identity]) {
                        host_end = graph_ready[identity];
                    } else {
                        if (kind == 2 && host->sync_before_replay[operation]
                                && device_ready > host_start) {
                            host_start = device_ready;
                        }
                        host_end = host_start + host->services[operation];
                        host_ready[lane] = host_end;
                        host_service[lane] += host->services[operation];
                        *launch_cpu_time_s += host->services[operation];
                        if (kind == 2) {
                            if (identity < 0 || identity >= host->identity_count) {
                                status = 11;
                                goto done;
                            }
                            graph_ready[identity] = host_end;
                            graph_seen[identity] = 1;
                            ++*graph_replay_count;
                            *launch_kinds |= 4;
                        } else if (kind == 1) {
                            *eager_dispatch_count += host->extra_enqueues[operation];
                            *launch_kinds |= 2;
                        } else {
                            *launch_kinds |= 1;
                            if (host->extra_enqueues[operation]) {
                                *eager_dispatch_count += host->extra_enqueues[operation];
                                *launch_kinds |= 2;
                            }
                        }
                    }
                    if (kind != 0 && host_end > ready) ready = host_end;
                }
            } else if (static_host_ready[slot] > ready) {
                ready = static_host_ready[slot];
            }
        }
        const double end = ready + duration;
        end_by_slot[slot] = end;
        if (end > *maximum_end) *maximum_end = end;
        for (int32_t cursor = 0; cursor < resource_stop - resource_begin; ++cursor) {
            resource_end[resources[cursor]] = end;
        }
        const int32_t digest = plan->digest_slot[slot];
        if (digest < 0) continue;
        const int32_t repeat = plan->repeat_slot[slot];
        if (digest >= digests || repeat < 0 || repeat >= repeats) { status = 8; goto done; }
        const double start_interval = ready - origin;
        const double end_interval = end - origin;
        if (plan->repeat_first[slot]) {
            if (!first_seen[repeat] || start_interval < repeat_bounds[2 * repeat]) {
                repeat_bounds[2 * repeat] = start_interval;
            }
            first_seen[repeat] = 1;
        }
        if (plan->repeat_last[slot]) {
            if (!last_seen[repeat] || end_interval > repeat_bounds[2 * repeat + 1]) {
                repeat_bounds[2 * repeat + 1] = end_interval;
            }
            last_seen[repeat] = 1;
        }
        const size_t offset = ((size_t)digest * (size_t)occurrences + counts[digest]) * 16;
        store_big_endian_double(records + offset, start_interval);
        store_big_endian_double(records + offset + 8, end_interval);
        ++counts[digest];
    }
    for (int32_t repeat = 0; repeat < repeats; ++repeat) {
        if (!first_seen[repeat] || !last_seen[repeat]) { status = 9; goto done; }
    }
    for (int32_t lane = 0; host != NULL && lane < host->lane_count; ++lane) {
        if (host_service[lane] > *launch_critical_path_s) {
            *launch_critical_path_s = host_service[lane];
        }
    }
    status = 0;
done:
    free(resource_end);
    free(device_resource_end);
    free(end_by_slot);
    free(device_end_by_slot);
    free(first_seen);
    free(last_seen);
    free(host_ready);
    free(host_service);
    free(graph_ready);
    free(graph_seen);
    if (status != 0) {
        free(records);
        free(counts);
        return status;
    }
    *interval_records = records;
    *record_counts = counts;
    return 0;
}

void midlayer_summary_timeline_free(unsigned char *records, size_t *counts) {
    free(records);
    free(counts);
}

/* The same earliest-ready queue recurrence used by the public Python owner.
 * Only integer DAG/resource structure is compiled here; every duration and
 * every added resource predecessor belongs to the current invocation. */
struct order_item { double time; int32_t position, group, version; };
struct order_heap { struct order_item *items; size_t size, capacity; };
static int order_less(struct order_item a, struct order_item b) {
    if (a.time != b.time) return a.time < b.time;
    if (a.position != b.position) return a.position < b.position;
    if (a.group != b.group) return a.group < b.group;
    return a.version < b.version;
}
static int order_push(struct order_heap *heap, struct order_item value) {
    if (heap->size == heap->capacity) {
        size_t capacity = heap->capacity ? heap->capacity * 2 : 8;
        void *items = realloc(heap->items, capacity * sizeof(*heap->items));
        if (items == NULL) return 0;
        heap->items = items; heap->capacity = capacity;
    }
    size_t index = heap->size++;
    while (index) {
        size_t parent = (index - 1) / 2;
        if (!order_less(value, heap->items[parent])) break;
        heap->items[index] = heap->items[parent]; index = parent;
    }
    heap->items[index] = value; return 1;
}
static struct order_item order_pop(struct order_heap *heap) {
    struct order_item first = heap->items[0], value = heap->items[--heap->size];
    size_t index = 0;
    while (index * 2 + 1 < heap->size) {
        size_t child = index * 2 + 1;
        if (child + 1 < heap->size && order_less(heap->items[child + 1], heap->items[child])) ++child;
        if (!order_less(heap->items[child], value)) break;
        heap->items[index] = heap->items[child]; index = child;
    }
    if (heap->size) heap->items[index] = value;
    return first;
}
struct order_state {
    const int32_t *group_offsets, *group_resources;
    double *resource_ready;
    struct order_heap *waiting, *eligible, ready;
    int32_t *versions;
};
static int order_head(struct order_state *s, int32_t group, struct order_item *head, int *eligible) {
    double time = 0.0;
    for (int32_t i = s->group_offsets[group]; i < s->group_offsets[group + 1]; ++i)
        if (s->resource_ready[s->group_resources[i]] > time) time = s->resource_ready[s->group_resources[i]];
    struct order_heap *waiting = &s->waiting[group], *available = &s->eligible[group];
    while (waiting->size && waiting->items[0].time <= time) {
        struct order_item item = order_pop(waiting); item.time = 0.0;
        if (!order_push(available, item)) return -1;
    }
    if (available->size) {
        *head = available->items[0]; head->time = time; *eligible = 1;
    } else if (waiting->size) {
        *head = waiting->items[0]; *eligible = 0;
    } else return 0;
    head->group = group; head->version = s->versions[group]; return 1;
}
static int order_enqueue(struct order_state *s, int32_t group) {
    struct order_item head; int eligible;
    ++s->versions[group];
    int found = order_head(s, group, &head, &eligible);
    return found < 0 ? 0 : found == 0 || order_push(&s->ready, head);
}
int midlayer_earliest_ready_order(int32_t count, int32_t resource_count, int32_t group_count,
        const int32_t *dependency_offsets, const int32_t *dependencies,
        const int32_t *resource_offsets, const int32_t *resources,
        const int32_t *call_groups, const int32_t *group_offsets, const int32_t *group_resources,
        const double *durations, int32_t *resource_predecessors) {
    if (count <= 0 || resource_count <= 0 || group_count <= 0) return 2;
    int status = 1;
    struct order_state s = {0};
    s.group_offsets = group_offsets; s.group_resources = group_resources;
    s.resource_ready = calloc((size_t)resource_count, sizeof(double));
    s.waiting = calloc((size_t)group_count, sizeof(struct order_heap));
    s.eligible = calloc((size_t)group_count, sizeof(struct order_heap));
    s.versions = calloc((size_t)group_count, sizeof(int32_t));
    int32_t *indegree = calloc((size_t)count, sizeof(int32_t));
    int32_t *last_call = malloc((size_t)resource_count * sizeof(int32_t));
    int32_t *successor_offsets = calloc((size_t)count + 1, sizeof(int32_t));
    int32_t *cursor = calloc((size_t)count, sizeof(int32_t));
    int32_t *successors = malloc((size_t)(dependency_offsets[count] + 1) * sizeof(int32_t));
    double *dependency_end = calloc((size_t)count, sizeof(double));
    if (!s.resource_ready || !s.waiting || !s.eligible || !s.versions || !indegree || !last_call
            || !successor_offsets || !cursor || !successors || !dependency_end) goto done;
    for (int32_t r = 0; r < resource_count; ++r) last_call[r] = -1;
    for (int32_t call = 0; call < count; ++call) {
        indegree[call] = dependency_offsets[call + 1] - dependency_offsets[call];
        for (int32_t d = dependency_offsets[call]; d < dependency_offsets[call + 1]; ++d)
            ++successor_offsets[dependencies[d] + 1];
    }
    for (int32_t i = 1; i <= count; ++i) successor_offsets[i] += successor_offsets[i - 1];
    memcpy(cursor, successor_offsets, (size_t)count * sizeof(int32_t));
    for (int32_t call = 0; call < count; ++call) {
        for (int32_t d = dependency_offsets[call]; d < dependency_offsets[call + 1]; ++d)
            successors[cursor[dependencies[d]]++] = call;
        if (!indegree[call] && !order_push(&s.waiting[call_groups[call]],
                (struct order_item){0.0, call, 0, 0})) goto done;
    }
    for (int32_t g = 0; g < group_count; ++g) if (!order_enqueue(&s, g)) goto done;
    int32_t scheduled = 0;
    while (s.ready.size) {
        struct order_item queued = order_pop(&s.ready), head;
        int32_t group = queued.group; int eligible;
        if (queued.version != s.versions[group]) continue;
        int found = order_head(&s, group, &head, &eligible);
        if (found < 0) goto done;
        if (!found) continue;
        if (head.time != queued.time || head.position != queued.position) {
            if (!order_enqueue(&s, group)) goto done;
            continue;
        }
        order_pop(eligible ? &s.eligible[group] : &s.waiting[group]);
        int32_t call = head.position;
        double end = head.time + durations[call];
        /* Durations have already passed the Python public finite boundary. */
        if (end > 1.7976931348623157e308) { status = 3; goto done; }
        for (int32_t i = resource_offsets[call]; i < resource_offsets[call + 1]; ++i) {
            int32_t resource = resources[i];
            resource_predecessors[i] = last_call[resource];
            s.resource_ready[resource] = end; last_call[resource] = call;
        }
        ++scheduled;
        if (!order_enqueue(&s, group)) goto done;
        for (int32_t i = successor_offsets[call]; i < successor_offsets[call + 1]; ++i) {
            int32_t child = successors[i];
            if (end > dependency_end[child]) dependency_end[child] = end;
            if (!--indegree[child]) {
                int32_t child_group = call_groups[child];
                if (!order_push(&s.waiting[child_group],
                        (struct order_item){dependency_end[child], child, 0, 0})
                        || !order_enqueue(&s, child_group)) goto done;
            }
        }
    }
    status = scheduled == count ? 0 : 4;
done:
    if (s.waiting) for (int32_t i = 0; i < group_count; ++i) free(s.waiting[i].items);
    if (s.eligible) for (int32_t i = 0; i < group_count; ++i) free(s.eligible[i].items);
    free(s.waiting); free(s.eligible); free(s.ready.items); free(s.versions); free(s.resource_ready);
    free(indegree); free(last_call); free(successor_offsets); free(cursor); free(successors); free(dependency_end);
    return status;
}
