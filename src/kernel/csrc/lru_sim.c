/*
 * Fast LRU cache simulation for tile-level traffic prediction (v2).
 *
 * Uses a fixed-size slot pool + separate hash index (open addressing with
 * backward-shift deletion). Evicted slots are properly freed so no tombstone
 * buildup occurs. O(1) amortized per access regardless of eviction rate.
 *
 * Build: gcc -O3 -shared -fPIC -o lru_sim.so lru_sim.c
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>

static void fail_loud(const char *message) {
    fprintf(stderr, "lru_sim: %s\n", message);
    abort();
}

static void *checked_malloc(size_t size) {
    void *pointer = malloc(size);
    if (pointer == NULL) fail_loud("memory allocation failed");
    return pointer;
}

static void *checked_calloc(size_t count, size_t size) {
    void *pointer = calloc(count, size);
    if (pointer == NULL) fail_loud("zero-initialized memory allocation failed");
    return pointer;
}

static int32_t expert_for_tile(
    const int32_t *expert_of_m, int32_t tile_index, int32_t tile_count
) {
    if (tile_index < 0 || tile_index >= tile_count) {
        fail_loud("MoE schedule contains an out-of-range M tile");
    }
    return expert_of_m[tile_index];
}

/* ─── Slot pool: doubly-linked LRU list ─── */
typedef struct {
    uint64_t key;
    int32_t  size;
    uint8_t  dirty;
    int32_t  hash_idx;  /* position in hash table (-1 if free) */
    int32_t  hash_next; /* next slot in the fixed hash bucket */
    int32_t  lru_prev;
    int32_t  lru_next;
} Slot;

typedef struct {
    /* Slot pool */
    Slot    *slots;
    int32_t  max_slots;
    int32_t  n_used;
    int32_t  free_head;     /* singly-linked free list through lru_next */

    /* LRU list */
    int32_t  lru_head;      /* most recent */
    int32_t  lru_tail;      /* least recent */

    /* Hash buckets map key → slot index. -1 = empty. */
    int32_t *hash_table;
    int32_t  hash_cap;      /* power of 2 */
    int32_t  hash_mask;

    /* Byte budget */
    int64_t  cur_bytes;
    int64_t  max_bytes;
    int64_t  anonymous_dirty_bytes;
} LRU;

static inline uint32_t mix32(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    return (uint32_t)(k >> 32);
}

static void lru_init(LRU *c, int32_t max_slots, int64_t max_bytes) {
    c->max_slots = max_slots;
    c->n_used = 0;
    c->lru_head = c->lru_tail = -1;
    c->cur_bytes = 0;
    c->max_bytes = max_bytes;
    c->anonymous_dirty_bytes = 0;

    c->slots = (Slot *)checked_malloc((size_t)max_slots * sizeof(Slot));
    /* Build free list */
    for (int32_t i = 0; i < max_slots; i++) {
        c->slots[i].dirty = 0;
        c->slots[i].hash_idx = -1;
        c->slots[i].hash_next = -1;
        c->slots[i].lru_prev = -1;
        c->slots[i].lru_next = (i + 1 < max_slots) ? i + 1 : -1;
    }
    c->free_head = 0;

    /* Hash table: 2× slots for low load factor */
    c->hash_cap = 1;
    while (c->hash_cap < max_slots * 2) c->hash_cap <<= 1;
    c->hash_mask = c->hash_cap - 1;
    c->hash_table = (int32_t *)checked_malloc(
        (size_t)c->hash_cap * sizeof(int32_t));
    memset(c->hash_table, 0xFF, c->hash_cap * sizeof(int32_t));  /* fill with -1 */
}

static void lru_free(LRU *c) {
    free(c->slots);
    free(c->hash_table);
}

/* Reset cache state without reallocating */
static void lru_reset(LRU *c) {
    c->n_used = 0;
    c->lru_head = c->lru_tail = -1;
    c->cur_bytes = 0;
    c->anonymous_dirty_bytes = 0;
    for (int32_t i = 0; i < c->max_slots; i++) {
        c->slots[i].dirty = 0;
        c->slots[i].hash_idx = -1;
        c->slots[i].hash_next = -1;
        c->slots[i].lru_prev = -1;
        c->slots[i].lru_next = (i + 1 < c->max_slots) ? i + 1 : -1;
    }
    c->free_head = 0;
    memset(c->hash_table, 0xFF, c->hash_cap * sizeof(int32_t));
}

/* Remove slot from LRU list (keep in hash) */
static void list_remove(LRU *c, int32_t si) {
    Slot *s = &c->slots[si];
    if (s->lru_prev >= 0) c->slots[s->lru_prev].lru_next = s->lru_next;
    else c->lru_head = s->lru_next;
    if (s->lru_next >= 0) c->slots[s->lru_next].lru_prev = s->lru_prev;
    else c->lru_tail = s->lru_prev;
}

/* Push slot to head (MRU) */
static void list_push_head(LRU *c, int32_t si) {
    Slot *s = &c->slots[si];
    s->lru_prev = -1;
    s->lru_next = c->lru_head;
    if (c->lru_head >= 0) c->slots[c->lru_head].lru_prev = si;
    c->lru_head = si;
    if (c->lru_tail < 0) c->lru_tail = si;
}

/* Hash lookup: returns slot index or -1. */
static int32_t hash_find(LRU *c, uint64_t key) {
    uint32_t h = mix32(key) & (uint32_t)c->hash_mask;
    for (int32_t si = c->hash_table[h]; si >= 0;
            si = c->slots[si].hash_next) {
        if (c->slots[si].key == key) return si;
    }
    return -1;
}

/* Hash insert at the bucket head. */
static void hash_insert(LRU *c, uint64_t key, int32_t slot_idx) {
    uint32_t h = mix32(key) & (uint32_t)c->hash_mask;
    c->slots[slot_idx].hash_idx = (int32_t)h;
    c->slots[slot_idx].hash_next = c->hash_table[h];
    c->hash_table[h] = slot_idx;
}

/* Hash remove from one fixed bucket. */
static void hash_remove(LRU *c, int32_t slot_idx) {
    int32_t hi = c->slots[slot_idx].hash_idx;
    if (hi < 0) return;
    int32_t previous = -1;
    int32_t current = c->hash_table[hi];
    while (current >= 0 && current != slot_idx) {
        previous = current;
        current = c->slots[current].hash_next;
    }
    if (current < 0) fail_loud("LRU hash bucket lost an active slot");
    if (previous < 0) c->hash_table[hi] = c->slots[current].hash_next;
    else c->slots[previous].hash_next = c->slots[current].hash_next;
    c->slots[slot_idx].hash_idx = -1;
    c->slots[slot_idx].hash_next = -1;
}

/* Evict LRU tail until bytes fit and preserve dirty writeback traffic. */
static void evict_to_fit(LRU *c, int64_t *dirty_evicted_bytes) {
    if (c->cur_bytes > c->max_bytes && c->anonymous_dirty_bytes > 0) {
        int64_t excess = c->cur_bytes - c->max_bytes;
        int64_t evicted = (
            excess < c->anonymous_dirty_bytes
            ? excess : c->anonymous_dirty_bytes);
        c->anonymous_dirty_bytes -= evicted;
        c->cur_bytes -= evicted;
        *dirty_evicted_bytes += evicted;
    }
    while (c->cur_bytes > c->max_bytes && c->lru_tail >= 0) {
        int32_t victim = c->lru_tail;
        Slot *v = &c->slots[victim];
        if (v->dirty) *dirty_evicted_bytes += v->size;
        c->cur_bytes -= v->size;
        list_remove(c, victim);
        hash_remove(c, victim);
        /* Return to free list */
        v->lru_next = c->free_head;
        v->lru_prev = -1;
        v->hash_idx = -1;
        v->hash_next = -1;
        c->free_head = victim;
        c->n_used--;
    }
}

/* Access: returns 1=miss, 0=hit and reports dirty victims. */
static int lru_access_stateful(
    LRU *c,
    uint64_t key,
    int32_t size,
    uint8_t is_write,
    int64_t *dirty_evicted_bytes
) {
    int32_t si = hash_find(c, key);
    if (si >= 0) {
        /* Hit: move to head */
        if (is_write) c->slots[si].dirty = 1;
        list_remove(c, si);
        list_push_head(c, si);
        return 0;
    }
    /* Miss: fast path when cache is full and new entry fits in one eviction */
    if (c->anonymous_dirty_bytes == 0
        && c->free_head < 0 && c->lru_tail >= 0 &&
        c->cur_bytes - c->slots[c->lru_tail].size + size <= c->max_bytes) {
        /* Reuse LRU tail slot directly — avoid free list round-trip */
        int32_t victim = c->lru_tail;
        Slot *v = &c->slots[victim];
        if (v->dirty) *dirty_evicted_bytes += v->size;
        c->cur_bytes -= v->size;
        list_remove(c, victim);
        hash_remove(c, victim);
        /* Repurpose slot */
        v->key = key;
        v->size = size;
        v->dirty = is_write;
        c->cur_bytes += size;
        list_push_head(c, victim);
        hash_insert(c, key, victim);
        return 1;
    }
    /* Slow path: general case */
    if (c->free_head < 0) {
        evict_to_fit(c, dirty_evicted_bytes);
        if (c->free_head < 0) return 1;
    }
    si = c->free_head;
    c->free_head = c->slots[si].lru_next;
    c->slots[si].key = key;
    c->slots[si].size = size;
    c->slots[si].dirty = is_write;
    c->n_used++;
    c->cur_bytes += size;
    list_push_head(c, si);
    hash_insert(c, key, si);
    evict_to_fit(c, dirty_evicted_bytes);
    return 1;
}

/* Backward-compatible clean read access used by existing simulations. */
static int lru_access(LRU *c, uint64_t key, int32_t size) {
    int64_t ignored_dirty_evictions = 0;
    return lru_access_stateful(
        c, key, size, 0, &ignored_dirty_evictions);
}

static int64_t resident_dirty_bytes(const LRU *c) {
    int64_t result = c->anonymous_dirty_bytes;
    for (int32_t current = c->lru_head; current >= 0;
            current = c->slots[current].lru_next) {
        if (c->slots[current].dirty) result += c->slots[current].size;
    }
    return result;
}

/* ─── Helper: compute max_slots from tile sizes ─── */
static int32_t compute_max_slots(int32_t a_tile, int32_t b_tile, int32_t c_tile,
                                  int64_t l2_cap) {
    if (a_tile <= 0 || b_tile <= 0 || c_tile <= 0 || l2_cap <= 0) {
        fail_loud("tile sizes and L2 capacity must be positive");
    }
    int32_t min_tile = a_tile < b_tile ? a_tile : b_tile;
    if (c_tile < min_tile) min_tile = c_tile;
    int32_t max_slots = (int32_t)(l2_cap / min_tile) + 1024;
    if (max_slots > 4000000) max_slots = 4000000;
    if (max_slots < 4096) max_slots = 4096;
    return max_slots;
}

/* ─── Reuse-distance primitives and the shared miss core ─── */

typedef struct {
    int64_t *tree;   /* 1-indexed Fenwick over timeline slots */
    int32_t  n;
} Fenwick;

static void fen_init(Fenwick *f, int32_t n) {
    f->n = n;
    f->tree = (int64_t *)checked_calloc((size_t)n + 1, sizeof(int64_t));
}
static void fen_free(Fenwick *f) { free(f->tree); }
static void fen_add(Fenwick *f, int32_t i, int64_t v) {
    for (i += 1; i <= f->n; i += i & (-i)) f->tree[i] += v;
}
/* prefix sum over slots [0, i] (0-indexed inclusive) */
static int64_t fen_prefix(Fenwick *f, int32_t i) {
    int64_t s = 0;
    for (i += 1; i > 0; i -= i & (-i)) s += f->tree[i];
    return s;
}

/* Open-addressing hash key -> previous timeline slot (-1 if first reference). */
typedef struct {
    uint64_t *keys;
    int32_t  *slot;   /* timeline slot of most recent reference, -1 if empty */
    int32_t   cap;
    int32_t   mask;
} SlotMap;

static void sm_init(SlotMap *m, int32_t expected) {
    int32_t cap = 1;
    while (cap < expected * 2) cap <<= 1;
    if (cap < 16) cap = 16;
    m->cap = cap;
    m->mask = cap - 1;
    m->keys = (uint64_t *)checked_malloc((size_t)cap * sizeof(uint64_t));
    m->slot = (int32_t *)checked_malloc((size_t)cap * sizeof(int32_t));
    for (int32_t i = 0; i < cap; i++) m->slot[i] = -1;
}
static void sm_free(SlotMap *m) { free(m->keys); free(m->slot); }
/* Returns the hash bucket index for key (existing or first empty). */
static int32_t sm_bucket(SlotMap *m, uint64_t key) {
    uint32_t h = mix32(key) & (uint32_t)m->mask;
    for (int32_t i = 0; i < m->cap; i++) {
        int32_t idx = (int32_t)((h + (uint32_t)i) & (uint32_t)m->mask);
        if (m->slot[idx] < 0) { m->keys[idx] = key; return idx; }
        if (m->keys[idx] == key) return idx;
    }
    fail_loud("slot map is full despite 2x sizing");
    return -1;
}

void gemm_stack_distance_hit_curve(
    const int32_t *schedule_mn,
    int32_t n_ctas,
    int32_t k_steps,
    int32_t a_tile_bytes,
    int32_t b_tile_bytes,
    int32_t c_tile_bytes,
    int32_t wave_size,
    const int64_t *query_caps,   /* capacities (bytes) to sample hit curve at */
    int32_t n_queries,
    int64_t *hit_bytes_out,      /* out[n_queries]: hit bytes at each capacity */
    int64_t *total_l2_bytes_out  /* out[1]: total access bytes (denominator) */
) {
    /* Total number of references = bound on distinct timeline slots. */
    int32_t n_waves = (n_ctas + wave_size - 1) / wave_size;
    /* A and B accessed k_steps times per CTA; C once per CTA. */
    int64_t total_refs = 0;
    for (int32_t w = 0; w < n_waves; w++) {
        int32_t ws = w * wave_size;
        int32_t we = ws + wave_size; if (we > n_ctas) we = n_ctas;
        int32_t cnt = we - ws;
        total_refs += (int64_t)2 * cnt * k_steps;  /* A + B per k */
        total_refs += cnt;                          /* C */
    }
    if (total_refs > INT32_MAX - 1) {
        fail_loud("stack-distance reference count exceeds int32 capacity");
    }

    Fenwick fen; fen_init(&fen, (int32_t)total_refs + 1);
    SlotMap map; sm_init(&map, (int32_t)total_refs + 1);

    int64_t live_bytes = 0;          /* sum of all distinct live tile bytes */
    int32_t t = 0;                   /* current timeline slot */
    int64_t total_l2 = 0;

    /* Per-access stack distance is binned implicitly: we accumulate hit bytes
     * into each query capacity bucket. For efficiency we keep query_caps
     * sorted ascending (caller contract); a hit at distance d contributes to
     * all caps > d. We accumulate into a temporary "hits at exactly sorted
     * bucket" array then prefix-sum at the end. */
    for (int32_t q = 0; q < n_queries; q++) hit_bytes_out[q] = 0;

    #define PROCESS_ACCESS(KEY, SZ)                                            \
        do {                                                                   \
            total_l2 += (SZ);                                                  \
            int32_t b = sm_bucket(&map, (KEY));                                \
            int32_t prev = map.slot[b];                                        \
            if (prev >= 0) {                                                   \
                int64_t le = fen_prefix(&fen, prev);                           \
                int64_t sd = live_bytes - le; /* bytes more recent than prev */\
                /* hit iff sd < C: contributes (SZ) to every query cap > sd */ \
                for (int32_t q = 0; q < n_queries; q++)                        \
                    if (query_caps[q] > sd) hit_bytes_out[q] += (SZ);          \
                fen_add(&fen, prev, -(int64_t)(SZ));                           \
                live_bytes -= (SZ);                                           \
            }                                                                  \
            map.slot[b] = t;                                                   \
            fen_add(&fen, t, (int64_t)(SZ));                                   \
            live_bytes += (SZ);                                               \
            t++;                                                               \
        } while (0)

    for (int32_t w = 0; w < n_waves; w++) {
        int32_t ws = w * wave_size;
        int32_t we = ws + wave_size; if (we > n_ctas) we = n_ctas;
        for (int32_t k = 0; k < k_steps; k++) {
            for (int32_t i = ws; i < we; i++) {
                int32_t mi = schedule_mn[i * 2];
                int32_t ni = schedule_mn[i * 2 + 1];
                uint64_t ak = ((uint64_t)(mi & 0xFFFF) << 16) | (uint64_t)(k & 0xFFFF);
                PROCESS_ACCESS(ak, a_tile_bytes);
                uint64_t bk = (1ULL << 48) | ((uint64_t)(ni & 0xFFFF) << 16) | (uint64_t)(k & 0xFFFF);
                PROCESS_ACCESS(bk, b_tile_bytes);
            }
        }
        for (int32_t i = ws; i < we; i++) {
            int32_t mi = schedule_mn[i * 2];
            int32_t ni = schedule_mn[i * 2 + 1];
            uint64_t ck = (2ULL << 48) | ((uint64_t)(mi & 0xFFFF) << 16) | (uint64_t)(ni & 0xFFFF);
            PROCESS_ACCESS(ck, c_tile_bytes);
        }
    }
    #undef PROCESS_ACCESS

    fen_free(&fen);
    sm_free(&map);
    *total_l2_bytes_out = total_l2;
}

/* ─── Shared L2-to-DRAM miss core ───
 *
 * One replacement solver for every hardware and every kernel family.  A
 * caller supplies an ordered access sequence built from that family's launch
 * facts; the core answers how many of each access's bytes the downstream
 * memory has to serve.  It owns no kernel, vendor, or hardware knowledge.
 *
 * The decision uses the exact LRU stack distance: the distinct live bytes
 * referenced since the object was last touched, plus the object itself.  A
 * fully associative byte-capacity LRU hits exactly when that occupancy fits,
 * which the core reproduces when the hardware declares no residency profile.
 * A hardware whose measured residency is graded rather than a clean step
 * supplies an ascending (occupancy, dram_served_fraction) profile and the
 * core interpolates it.  Replacement semantics are therefore a per-hardware
 * numeric parameter, never a code branch.
 */
typedef struct {
    int64_t        capacity_bytes;
    const int64_t *profile_occupancy;
    const double  *profile_fraction;
    int32_t        profile_points;
    Fenwick        fen;
    SlotMap        map;
    int64_t        live_bytes;
    int32_t        t;
    /* Write-back state.  An object carries the dirty flag of its last
     * reference, so a reference that misses reports its predecessor as a
     * dirty victim.  ``anonymous_dirty_bytes`` is older cache content at the
     * replacement tail: it can be displaced but never referenced, so it never
     * changes a hit decision and only drains as distinct bytes accumulate. */
    int64_t        anonymous_dirty_bytes;
    int64_t        initial_anonymous_bytes;
    int64_t        distinct_bytes;
    int32_t       *object_size;
    uint8_t       *object_dirty;
    /* Two interchangeable realizations of the one replacement model.  A
     * hardware that declares no residency profile is an exact capacity step,
     * which the occupancy list answers in constant time and constant space
     * per resident object.  A graded profile needs the reference's exact
     * reuse distance, which the Fenwick timeline above supplies at the cost
     * of space in the number of references.  The answer is the same model
     * either way; only the hardware's own numbers select the realization. */
    int            graded;
    LRU            occupancy;
} MissCore;

static double miss_core_dram_fraction(const MissCore *core, int64_t occupancy) {
    if (core->profile_points <= 0) {
        return occupancy <= core->capacity_bytes ? 0.0 : 1.0;
    }
    if (occupancy <= core->profile_occupancy[0]) {
        return core->profile_fraction[0];
    }
    int32_t low = 0;
    int32_t high = core->profile_points - 1;
    if (occupancy >= core->profile_occupancy[high]) {
        return core->profile_fraction[high];
    }
    while (high - low > 1) {
        int32_t middle = low + (high - low) / 2;
        if (core->profile_occupancy[middle] <= occupancy) {
            low = middle;
        } else {
            high = middle;
        }
    }
    int64_t span = core->profile_occupancy[high] - core->profile_occupancy[low];
    if (span <= 0) return core->profile_fraction[high];
    double weight = (double)(occupancy - core->profile_occupancy[low])
        / (double)span;
    return core->profile_fraction[low]
        + weight * (core->profile_fraction[high] - core->profile_fraction[low]);
}

static void miss_core_init_sized(
    MissCore *core, int64_t capacity_bytes,
    const int64_t *profile_occupancy, const double *profile_fraction,
    int32_t profile_points, int64_t expected_references,
    int32_t resident_object_bound
) {
    if (capacity_bytes <= 0) fail_loud("miss core capacity must be positive");
    if (profile_points > 0
            && (expected_references <= 0
                || expected_references > INT32_MAX - 1)) {
        fail_loud(
            "a graded residency profile needs a bounded reference count");
    }
    if (profile_points < 0) fail_loud("miss core profile length is negative");
    for (int32_t i = 0; i < profile_points; i++) {
        if (profile_occupancy[i] <= 0) {
            fail_loud("miss core profile occupancy must be positive");
        }
        if (i > 0 && profile_occupancy[i] <= profile_occupancy[i - 1]) {
            fail_loud("miss core profile occupancy must be ascending");
        }
        if (profile_fraction[i] < 0.0 || profile_fraction[i] > 1.0) {
            fail_loud("miss core profile fraction must be a fraction");
        }
        if (i > 0 && profile_fraction[i] < profile_fraction[i - 1]) {
            fail_loud("miss core profile fraction must be non-decreasing");
        }
    }
    core->capacity_bytes = capacity_bytes;
    core->profile_occupancy = profile_occupancy;
    core->profile_fraction = profile_fraction;
    core->profile_points = profile_points;
    core->live_bytes = 0;
    core->t = 0;
    core->anonymous_dirty_bytes = 0;
    core->initial_anonymous_bytes = 0;
    core->distinct_bytes = 0;
    core->graded = profile_points > 0;
    if (!core->graded) {
        if (resident_object_bound < 1) {
            fail_loud("miss core needs a positive resident object bound");
        }
        lru_init(&core->occupancy, resident_object_bound, capacity_bytes);
        core->object_size = NULL;
        core->object_dirty = NULL;
        return;
    }
    fen_init(&core->fen, (int32_t)expected_references + 1);
    sm_init(&core->map, (int32_t)expected_references + 1);
    core->object_size = (int32_t *)checked_calloc(
        (size_t)core->map.cap, sizeof(int32_t));
    core->object_dirty = (uint8_t *)checked_calloc(
        (size_t)core->map.cap, sizeof(uint8_t));
}

/* Size the step realization from the smallest object the stream can hold. */
static void miss_core_init(
    MissCore *core, int64_t capacity_bytes,
    const int64_t *profile_occupancy, const double *profile_fraction,
    int32_t profile_points, int64_t expected_references
) {
    int64_t bound = expected_references > 0 ? expected_references : 1;
    if (bound > 4000000) bound = 4000000;
    miss_core_init_sized(
        core, capacity_bytes, profile_occupancy, profile_fraction,
        profile_points, expected_references, (int32_t)bound);
}

/* Seed the replacement tail with older dirty content of the given size. */
static void miss_core_seed_anonymous_dirty(MissCore *core, int64_t bytes) {
    if (bytes < 0 || bytes > core->capacity_bytes) {
        fail_loud("miss core anonymous dirty occupancy must fit in capacity");
    }
    core->anonymous_dirty_bytes = bytes;
    core->initial_anonymous_bytes = bytes;
    if (!core->graded) {
        core->occupancy.cur_bytes = bytes;
        core->occupancy.anonymous_dirty_bytes = bytes;
    }
}

static void miss_core_free(MissCore *core) {
    if (!core->graded) {
        lru_free(&core->occupancy);
        return;
    }
    fen_free(&core->fen);
    sm_free(&core->map);
    free(core->object_size);
    free(core->object_dirty);
}

/* Reference one physical cache object and return its DRAM-served bytes.
 *
 * ``dirty_victim_bytes`` accumulates the bytes a write-back cache drains when
 * this reference finds its object gone and that object had been written.
 */
static double miss_core_touch_stateful(
    MissCore *core, uint64_t key, int32_t size, int is_write,
    int64_t *dirty_victim_bytes
) {
    if (size <= 0) fail_loud("miss core object size must be positive");
    if (!core->graded) {
        int64_t evicted = 0;
        int missed_step = lru_access_stateful(
            &core->occupancy, key, size, (uint8_t)(is_write ? 1 : 0),
            &evicted);
        if (dirty_victim_bytes) *dirty_victim_bytes += evicted;
        return missed_step ? (double)size : 0.0;
    }
    if (core->t >= core->fen.n) fail_loud("miss core reference budget exceeded");
    int32_t bucket = sm_bucket(&core->map, key);
    int32_t previous = core->map.slot[bucket];
    double dram_bytes;
    int missed;
    if (previous < 0) {
        dram_bytes = (double)size;
        missed = 1;
    } else {
        int64_t newer = core->live_bytes - fen_prefix(&core->fen, previous);
        double fraction = miss_core_dram_fraction(
            core, newer + (int64_t)size);
        dram_bytes = (double)size * fraction;
        missed = fraction > 0.0;
        if (missed && core->object_dirty[bucket] && dirty_victim_bytes) {
            *dirty_victim_bytes += core->object_size[bucket];
        }
        fen_add(&core->fen, previous, -(int64_t)size);
        core->live_bytes -= size;
    }
    if (missed) {
        core->distinct_bytes += size;
        if (core->anonymous_dirty_bytes > 0) {
            int64_t room = core->capacity_bytes - core->distinct_bytes;
            if (room < 0) room = 0;
            if (room > core->initial_anonymous_bytes) {
                room = core->initial_anonymous_bytes;
            }
            if (room < core->anonymous_dirty_bytes) {
                if (dirty_victim_bytes) {
                    *dirty_victim_bytes +=
                        core->anonymous_dirty_bytes - room;
                }
                core->anonymous_dirty_bytes = room;
            }
        }
        core->object_dirty[bucket] = (uint8_t)(is_write ? 1 : 0);
    } else if (is_write) {
        core->object_dirty[bucket] = 1;
    }
    core->object_size[bucket] = size;
    core->map.slot[bucket] = core->t;
    fen_add(&core->fen, core->t, (int64_t)size);
    core->live_bytes += size;
    core->t++;
    return dram_bytes;
}

static double miss_core_touch(MissCore *core, uint64_t key, int32_t size) {
    return miss_core_touch_stateful(core, key, size, 0, NULL);
}

/* Bytes the replacement tail still holds that never reached memory. */
static int64_t miss_core_anonymous_dirty_bytes(const MissCore *core) {
    return core->graded
        ? core->anonymous_dirty_bytes
        : core->occupancy.anonymous_dirty_bytes;
}

/* Bytes a write-back cache would still hold dirty once the stream ends. */
static int64_t miss_core_resident_dirty_bytes(const MissCore *core) {
    if (!core->graded) {
        return resident_dirty_bytes(&core->occupancy);
    }
    int64_t result = core->anonymous_dirty_bytes;
    if (core->anonymous_dirty_bytes > 0) {
        /* Nothing has been displaced yet, so every object is still held. */
        for (int32_t bucket = 0; bucket < core->map.cap; bucket++) {
            if (core->map.slot[bucket] >= 0 && core->object_dirty[bucket]) {
                result += core->object_size[bucket];
            }
        }
        return result;
    }
    for (int32_t bucket = 0; bucket < core->map.cap; bucket++) {
        int32_t last = core->map.slot[bucket];
        if (last < 0 || !core->object_dirty[bucket]) continue;
        int32_t size = core->object_size[bucket];
        int64_t newer = core->live_bytes
            - fen_prefix((Fenwick *)&core->fen, last);
        if (miss_core_dram_fraction(core, newer + size) <= 0.0) {
            result += size;
        }
    }
    return result;
}


/* ─── Public API ─── */

void simulate_lru_trace(
    const uint64_t *keys,
    const int32_t *sizes,
    int32_t access_count,
    int64_t capacity_bytes,
    int64_t *out
) {
    if (access_count <= 0 || capacity_bytes <= 0) {
        fail_loud("trace length and cache capacity must be positive");
    }
    int32_t minimum_size = INT32_MAX;
    for (int32_t index = 0; index < access_count; index++) {
        if (sizes[index] <= 0) fail_loud("trace access size must be positive");
        if (sizes[index] < minimum_size) minimum_size = sizes[index];
    }
    int64_t resident_bound = capacity_bytes / minimum_size + 1;
    int32_t max_slots = access_count;
    if (resident_bound < max_slots) max_slots = (int32_t)resident_bound;
    if (max_slots < 1) max_slots = 1;

    LRU cache;
    lru_init(&cache, max_slots, capacity_bytes);
    int64_t requested_bytes = 0;
    int64_t miss_bytes = 0;
    for (int32_t index = 0; index < access_count; index++) {
        requested_bytes += sizes[index];
        miss_bytes += (
            (int64_t)lru_access(&cache, keys[index], sizes[index])
            * sizes[index]);
    }
    lru_free(&cache);
    out[0] = requested_bytes;
    out[1] = miss_bytes;
}

void simulate_lru_trace_accounted(
    const uint64_t *keys,
    const int32_t *sizes,
    const uint8_t *accounted,
    int32_t access_count,
    int64_t capacity_bytes,
    int64_t *out
) {
    if (access_count <= 0 || capacity_bytes <= 0) {
        fail_loud("trace length and cache capacity must be positive");
    }
    int32_t minimum_size = INT32_MAX;
    for (int32_t index = 0; index < access_count; index++) {
        if (sizes[index] <= 0) fail_loud("trace access size must be positive");
        if (accounted[index] > 1) fail_loud("trace accounting mask must be boolean");
        if (sizes[index] < minimum_size) minimum_size = sizes[index];
    }
    int64_t resident_bound = capacity_bytes / minimum_size + 1;
    int32_t max_slots = access_count;
    if (resident_bound < max_slots) max_slots = (int32_t)resident_bound;
    if (max_slots < 1) max_slots = 1;

    LRU cache;
    lru_init(&cache, max_slots, capacity_bytes);
    int64_t requested_bytes = 0;
    int64_t miss_bytes = 0;
    for (int32_t index = 0; index < access_count; index++) {
        int32_t miss = lru_access(&cache, keys[index], sizes[index]);
        if (accounted[index]) {
            requested_bytes += sizes[index];
            miss_bytes += (int64_t)miss * sizes[index];
        }
    }
    lru_free(&cache);
    out[0] = requested_bytes;
    out[1] = miss_bytes;
}

void simulate_lru_trace_stateful(
    const uint64_t *keys,
    const int32_t *sizes,
    const uint8_t *access_kinds,
    const uint8_t *accounted,
    int32_t access_count,
    int64_t capacity_bytes,
    int64_t initial_dirty_bytes,
    int64_t *out
) {
    if (access_count <= 0 || capacity_bytes <= 0) {
        fail_loud("trace length and cache capacity must be positive");
    }
    if (initial_dirty_bytes < 0 || initial_dirty_bytes > capacity_bytes) {
        fail_loud("initial dirty bytes must fit within cache capacity");
    }
    int32_t minimum_size = INT32_MAX;
    for (int32_t index = 0; index < access_count; index++) {
        if (sizes[index] <= 0) fail_loud("trace access size must be positive");
        if (access_kinds[index] > 1) {
            fail_loud("trace access kind must be read or write");
        }
        if (accounted[index] > 1) {
            fail_loud("trace accounting mask must be boolean");
        }
        if (sizes[index] < minimum_size) minimum_size = sizes[index];
    }
    int64_t resident_bound = capacity_bytes / minimum_size + 1;
    int32_t max_slots = access_count;
    if (resident_bound < max_slots) max_slots = (int32_t)resident_bound;
    if (max_slots < 1) max_slots = 1;

    LRU cache;
    lru_init(&cache, max_slots, capacity_bytes);
    cache.cur_bytes = initial_dirty_bytes;
    cache.anonymous_dirty_bytes = initial_dirty_bytes;
    int64_t requested_bytes = 0;
    int64_t read_miss_bytes = 0;
    int64_t dirty_evicted_bytes = 0;
    for (int32_t index = 0; index < access_count; index++) {
        int32_t miss = lru_access_stateful(
            &cache, keys[index], sizes[index], access_kinds[index],
            &dirty_evicted_bytes);
        if (accounted[index]) {
            requested_bytes += sizes[index];
            if (!access_kinds[index] && miss) {
                read_miss_bytes += sizes[index];
            }
        }
    }
    out[0] = requested_bytes;
    out[1] = read_miss_bytes;
    out[2] = dirty_evicted_bytes;
    out[3] = resident_dirty_bytes(&cache);
    out[4] = cache.anonymous_dirty_bytes;
    lru_free(&cache);
}

void simulate_lru_trace_stateful_dual_accounted(
    const uint64_t *keys,
    const int32_t *sizes,
    const uint8_t *access_kinds,
    const uint8_t *first_accounted,
    const uint8_t *second_accounted,
    int32_t access_count,
    int64_t capacity_bytes,
    int64_t initial_dirty_bytes,
    int64_t *out
) {
    if (access_count <= 0 || capacity_bytes <= 0) {
        fail_loud("trace length and cache capacity must be positive");
    }
    if (initial_dirty_bytes < 0 || initial_dirty_bytes > capacity_bytes) {
        fail_loud("initial dirty bytes must fit within cache capacity");
    }
    int32_t minimum_size = INT32_MAX;
    for (int32_t index = 0; index < access_count; index++) {
        if (sizes[index] <= 0) fail_loud("trace access size must be positive");
        if (access_kinds[index] > 1) {
            fail_loud("trace access kind must be read or write");
        }
        if (first_accounted[index] > 1 || second_accounted[index] > 1) {
            fail_loud("trace accounting mask must be boolean");
        }
        if (sizes[index] < minimum_size) minimum_size = sizes[index];
    }
    int64_t resident_bound = capacity_bytes / minimum_size + 1;
    int32_t max_slots = access_count;
    if (resident_bound < max_slots) max_slots = (int32_t)resident_bound;
    if (max_slots < 1) max_slots = 1;

    LRU cache;
    lru_init(&cache, max_slots, capacity_bytes);
    cache.cur_bytes = initial_dirty_bytes;
    cache.anonymous_dirty_bytes = initial_dirty_bytes;
    int64_t first_requested_bytes = 0;
    int64_t first_read_miss_bytes = 0;
    int64_t second_requested_bytes = 0;
    int64_t second_read_miss_bytes = 0;
    int64_t dirty_evicted_bytes = 0;
    for (int32_t index = 0; index < access_count; index++) {
        int32_t miss = lru_access_stateful(
            &cache, keys[index], sizes[index], access_kinds[index],
            &dirty_evicted_bytes);
        if (first_accounted[index]) {
            first_requested_bytes += sizes[index];
            if (!access_kinds[index] && miss) {
                first_read_miss_bytes += sizes[index];
            }
        }
        if (second_accounted[index]) {
            second_requested_bytes += sizes[index];
            if (!access_kinds[index] && miss) {
                second_read_miss_bytes += sizes[index];
            }
        }
    }
    out[0] = first_requested_bytes;
    out[1] = first_read_miss_bytes;
    out[2] = second_requested_bytes;
    out[3] = second_read_miss_bytes;
    out[4] = dirty_evicted_bytes;
    out[5] = resident_dirty_bytes(&cache);
    out[6] = cache.anonymous_dirty_bytes;
    lru_free(&cache);
}

static int64_t align_up_i64(int64_t value, int64_t alignment) {
    if (value <= 0 || alignment <= 0) {
        fail_loud("alignment inputs must be positive");
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

void simulate_cann_gemm_persistent_stateful_dual_accounted(
    int32_t aic_workers,
    int32_t worker_grid_m,
    int32_t worker_grid_n,
    int32_t worker_grid_k,
    int32_t single_core_m,
    int32_t single_core_n,
    int32_t single_core_k,
    int32_t base_m,
    int32_t base_n,
    int32_t base_k,
    int32_t a_l1_k_tiles,
    int32_t b_l1_k_tiles,
    int32_t step_m,
    int32_t step_n,
    int64_t m,
    int64_t n,
    int64_t k,
    int32_t element_bytes,
    int64_t capacity_bytes,
    int64_t initial_dirty_bytes,
    int64_t *out
) {
    int32_t values[] = {
        aic_workers, worker_grid_m, worker_grid_n, worker_grid_k,
        single_core_m, single_core_n, single_core_k,
        base_m, base_n, base_k,
        a_l1_k_tiles, b_l1_k_tiles, step_m, step_n, element_bytes,
    };
    for (size_t index = 0; index < sizeof(values) / sizeof(values[0]); index++) {
        if (values[index] <= 0) {
            fail_loud("CANN persistent program values must be positive");
        }
    }
    if (m <= 0 || n <= 0 || k <= 0 || capacity_bytes <= 0) {
        fail_loud("CANN persistent dimensions and capacity must be positive");
    }
    if (initial_dirty_bytes < 0 || initial_dirty_bytes > capacity_bytes) {
        fail_loud("initial dirty bytes must fit within cache capacity");
    }
    if ((int64_t)worker_grid_m * worker_grid_n * worker_grid_k
            != aic_workers) {
        fail_loud("CANN persistent worker grid differs from workers");
    }
    if (single_core_m % base_m || single_core_n % base_n
            || single_core_k % base_k) {
        fail_loud("CANN persistent core extents must be base-tile aligned");
    }
    if (a_l1_k_tiles * base_k > single_core_k
            || b_l1_k_tiles * base_k > single_core_k) {
        fail_loud("CANN persistent L1 spans exceed the core K extent");
    }

    const int64_t cube_edge = 16;
    const int64_t cache_line_bytes = 512;
    const uint64_t a_tag = 0;
    const uint64_t b_tag = 1ULL << 61;
    const uint64_t output_tag = 2ULL << 61;
    int64_t aligned_m = align_up_i64(m, cube_edge);
    int64_t aligned_n = align_up_i64(n, cube_edge);
    int64_t aligned_k = align_up_i64(k, cube_edge);
    if ((int64_t)single_core_m * worker_grid_m < aligned_m
            || (int64_t)single_core_n * worker_grid_n < aligned_n
            || (int64_t)single_core_k * worker_grid_k < aligned_k) {
        fail_loud("CANN persistent worker extents do not cover dimensions");
    }

    int64_t unique_a_bytes = aligned_m * aligned_k * element_bytes;
    int64_t unique_b_bytes = aligned_n * aligned_k * element_bytes;
    int64_t output_bytes = m * n * element_bytes;
    int64_t unique_output_bytes = (
        (output_bytes + cache_line_bytes - 1) / cache_line_bytes)
        * cache_line_bytes;
    int no_eviction = (
        initial_dirty_bytes <= capacity_bytes - unique_a_bytes
        && unique_b_bytes <= capacity_bytes - initial_dirty_bytes - unique_a_bytes
        && unique_output_bytes <= capacity_bytes - initial_dirty_bytes
            - unique_a_bytes - unique_b_bytes);
    MissCore core;
    if (!no_eviction) {
        int64_t resident_bound = capacity_bytes / cache_line_bytes + 1;
        if (resident_bound > INT32_MAX) {
            fail_loud("CANN persistent cache has too many cache-line slots");
        }
        int32_t max_slots = (int32_t)resident_bound;
        if (max_slots < 1) max_slots = 1;
        /* This program replays cache-line granular output writes, whose
         * reference count is not bounded ahead of the walk.  It therefore
         * serves only the exact capacity step; a hardware that measured a
         * graded residency would have to bound its references first. */
        miss_core_init_sized(
            &core, capacity_bytes, NULL, NULL, 0, 0, max_slots);
        miss_core_seed_anonymous_dirty(&core, initial_dirty_bytes);
    }

    int64_t *worker_m_starts = (int64_t *)checked_malloc(
        (size_t)aic_workers * sizeof(int64_t));
    int64_t *worker_n_starts = (int64_t *)checked_malloc(
        (size_t)aic_workers * sizeof(int64_t));
    int64_t *worker_k_starts = (int64_t *)checked_malloc(
        (size_t)aic_workers * sizeof(int64_t));
    int32_t mn_workers = worker_grid_m * worker_grid_n;
    for (int32_t worker = 0; worker < aic_workers; worker++) {
        int32_t k_worker = worker / mn_workers;
        int32_t mn_worker = worker % mn_workers;
        int32_t n_worker = mn_worker / worker_grid_m;
        int32_t m_worker = mn_worker % worker_grid_m;
        int64_t m_start = (int64_t)m_worker * single_core_m;
        int64_t n_start = (int64_t)n_worker * single_core_n;
        int64_t k_start = (int64_t)k_worker * single_core_k;
        if (m_start > aligned_m - single_core_m) {
            m_start = aligned_m - single_core_m;
        }
        if (n_start > aligned_n - single_core_n) {
            n_start = aligned_n - single_core_n;
        }
        if (k_start > aligned_k - single_core_k) {
            k_start = aligned_k - single_core_k;
        }
        worker_m_starts[worker] = m_start < 0 ? 0 : m_start;
        worker_n_starts[worker] = n_start < 0 ? 0 : n_start;
        worker_k_starts[worker] = k_start < 0 ? 0 : k_start;
    }

    int64_t a_requested_bytes = 0;
    int64_t a_miss_bytes = 0;
    int64_t b_requested_bytes = 0;
    int64_t b_miss_bytes = 0;
    int64_t dirty_evicted_bytes = 0;
    int64_t m_tiles_per_worker = single_core_m / base_m;
    int64_t n_tiles_per_worker = single_core_n / base_n;
    int64_t output_count = m_tiles_per_worker * n_tiles_per_worker;
    int64_t a_k_extent = (int64_t)a_l1_k_tiles * base_k;
    int64_t b_k_extent = (int64_t)b_l1_k_tiles * base_k;
    int full_a_resident = a_k_extent == single_core_k;

    for (int64_t output_index = 0;
            output_index < output_count; output_index++) {
        int64_t local_n = (output_index / m_tiles_per_worker) * base_n;
        int64_t local_m = (output_index % m_tiles_per_worker) * base_m;
        for (int64_t relative_k = 0;
                relative_k < single_core_k; relative_k += base_k) {
            for (int32_t worker = 0; worker < aic_workers; worker++) {
                int64_t k_start = worker_k_starts[worker] + relative_k;
                if ((!full_a_resident || local_n == 0)
                        && local_m % ((int64_t)base_m * step_m) == 0
                        && relative_k % a_k_extent == 0) {
                    int64_t outer_start = worker_m_starts[worker] + local_m;
                    int64_t outer_extent = (int64_t)base_m * step_m;
                    if (outer_extent > aligned_m - outer_start) {
                        outer_extent = aligned_m - outer_start;
                    }
                    int64_t k_extent = a_k_extent;
                    if (k_extent > aligned_k - k_start) {
                        k_extent = aligned_k - k_start;
                    }
                    if (no_eviction) {
                        a_requested_bytes += (
                            outer_extent * k_extent * element_bytes);
                    } else {
                        for (int64_t outer = outer_start;
                                outer < outer_start + outer_extent;
                                outer += cube_edge) {
                            for (int64_t inner_k = k_start;
                                    inner_k < k_start + k_extent;
                                    inner_k += base_k) {
                                int64_t current_k = base_k;
                                if (current_k > aligned_k - inner_k) {
                                    current_k = aligned_k - inner_k;
                                }
                                int32_t size = (int32_t)(
                                    cube_edge * current_k * element_bytes);
                                uint64_t key = a_tag
                                    | ((uint64_t)(outer / cube_edge) << 32)
                                    | (uint64_t)(inner_k / base_k);
                                int miss = miss_core_touch_stateful(
                                    &core, key, size, 0,
                                    &dirty_evicted_bytes) > 0.0;
                                a_requested_bytes += size;
                                if (miss) a_miss_bytes += size;
                            }
                        }
                    }
                }
                if (local_n % ((int64_t)base_n * step_n) == 0
                        && relative_k % b_k_extent == 0) {
                    int64_t outer_start = worker_n_starts[worker] + local_n;
                    int64_t outer_extent = (int64_t)base_n * step_n;
                    if (outer_extent > aligned_n - outer_start) {
                        outer_extent = aligned_n - outer_start;
                    }
                    int64_t k_extent = b_k_extent;
                    if (k_extent > aligned_k - k_start) {
                        k_extent = aligned_k - k_start;
                    }
                    if (no_eviction) {
                        b_requested_bytes += (
                            outer_extent * k_extent * element_bytes);
                    } else {
                        for (int64_t outer = outer_start;
                                outer < outer_start + outer_extent;
                                outer += cube_edge) {
                            for (int64_t inner_k = k_start;
                                    inner_k < k_start + k_extent;
                                    inner_k += base_k) {
                                int64_t current_k = base_k;
                                if (current_k > aligned_k - inner_k) {
                                    current_k = aligned_k - inner_k;
                                }
                                int32_t size = (int32_t)(
                                    cube_edge * current_k * element_bytes);
                                uint64_t key = b_tag
                                    | ((uint64_t)(outer / cube_edge) << 32)
                                    | (uint64_t)(inner_k / base_k);
                                int miss = miss_core_touch_stateful(
                                    &core, key, size, 0,
                                    &dirty_evicted_bytes) > 0.0;
                                b_requested_bytes += size;
                                if (miss) b_miss_bytes += size;
                            }
                        }
                    }
                }
            }
        }
        if (!no_eviction) {
            for (int32_t worker = 0; worker < aic_workers; worker++) {
                int64_t m_start = worker_m_starts[worker] + local_m;
                int64_t n_start = worker_n_starts[worker] + local_n;
                int64_t current_m = base_m;
                int64_t current_n = base_n;
                if (current_m > m - m_start) current_m = m - m_start;
                if (current_n > n - n_start) current_n = n - n_start;
                if (current_m <= 0 || current_n <= 0) continue;
                for (int64_t row = m_start; row < m_start + current_m; row++) {
                    int64_t byte_start = (row * n + n_start) * element_bytes;
                    int64_t byte_end = byte_start + current_n * element_bytes;
                    int64_t first_line = byte_start / cache_line_bytes;
                    int64_t final_line = (byte_end - 1) / cache_line_bytes;
                    for (int64_t line = first_line; line <= final_line; line++) {
                        (void)miss_core_touch_stateful(
                            &core, output_tag | (uint64_t)line,
                            (int32_t)cache_line_bytes, 1,
                            &dirty_evicted_bytes);
                    }
                }
            }
        }
    }
    if (no_eviction) {
        a_miss_bytes = unique_a_bytes;
        b_miss_bytes = unique_b_bytes;
    }
    out[0] = a_requested_bytes;
    out[1] = a_miss_bytes;
    out[2] = b_requested_bytes;
    out[3] = b_miss_bytes;
    out[4] = dirty_evicted_bytes;
    out[5] = (no_eviction
              ? initial_dirty_bytes + unique_output_bytes
              : miss_core_resident_dirty_bytes(&core));
    out[6] = (no_eviction
              ? initial_dirty_bytes
              : miss_core_anonymous_dirty_bytes(&core));
    free(worker_m_starts);
    free(worker_n_starts);
    free(worker_k_starts);
    if (!no_eviction) miss_core_free(&core);
}

void simulate_moe_lru(
    const int32_t *schedule_mn,
    int32_t n_ctas,
    const int32_t *expert_of_m,
    int32_t mt,
    int32_t k_steps,
    int32_t a_tile_bytes,
    int32_t b_tile_bytes,
    int32_t c_tile_bytes,
    int32_t wave_size,
    int64_t l2_capacity_bytes,
    int64_t *out
) {
    int32_t max_slots = compute_max_slots(a_tile_bytes, b_tile_bytes, c_tile_bytes,
                                           l2_capacity_bytes);

    LRU cache;
    lru_init(&cache, max_slots, l2_capacity_bytes);

    int64_t l2_total = 0, dram_total = 0;
    int32_t n_waves = (n_ctas + wave_size - 1) / wave_size;

    for (int32_t w = 0; w < n_waves; w++) {
        int32_t ws = w * wave_size;
        int32_t we = ws + wave_size;
        if (we > n_ctas) we = n_ctas;

        for (int32_t k = 0; k < k_steps; k++) {
            for (int32_t i = ws; i < we; i++) {
                int32_t mi = schedule_mn[i * 2];
                int32_t ni = schedule_mn[i * 2 + 1];
                int32_t e = expert_for_tile(expert_of_m, mi, mt);

                uint64_t ak = ((uint64_t)e << 32) | ((uint64_t)(mi & 0xFFFF) << 16) | (uint64_t)(k & 0xFFFF);
                int a_miss = lru_access(&cache, ak, a_tile_bytes);
                dram_total += (int64_t)a_miss * a_tile_bytes;
                l2_total += (int64_t)a_miss * a_tile_bytes;

                uint64_t bk = (1ULL << 48) | ((uint64_t)e << 32) | ((uint64_t)(ni & 0xFFFF) << 16) | (uint64_t)(k & 0xFFFF);
                l2_total += b_tile_bytes;
                dram_total += (int64_t)lru_access(&cache, bk, b_tile_bytes) * b_tile_bytes;
            }
        }
        for (int32_t i = ws; i < we; i++) {
            int32_t mi = schedule_mn[i * 2];
            int32_t ni = schedule_mn[i * 2 + 1];
            int32_t e = expert_for_tile(expert_of_m, mi, mt);
            uint64_t ck = (2ULL << 48) | ((uint64_t)e << 32) | ((uint64_t)(mi & 0xFFFF) << 16) | (uint64_t)(ni & 0xFFFF);
            l2_total += c_tile_bytes;
            dram_total += (int64_t)lru_access(&cache, ck, c_tile_bytes) * c_tile_bytes;
        }
    }
    lru_free(&cache);
    out[0] = l2_total;
    out[1] = dram_total;
}

void simulate_gemm_lru(
    const int32_t *schedule_mn,
    int32_t n_ctas,
    int32_t k_steps,
    int32_t a_tile_bytes,
    int32_t b_tile_bytes,
    int32_t c_tile_bytes,
    int32_t wave_size,
    int64_t l2_capacity_bytes,
    int64_t *out
) {
    int32_t max_slots = compute_max_slots(a_tile_bytes, b_tile_bytes, c_tile_bytes,
                                           l2_capacity_bytes);

    LRU cache;
    lru_init(&cache, max_slots, l2_capacity_bytes);

    int64_t l2_total = 0, dram_total = 0;
    int32_t n_waves = (n_ctas + wave_size - 1) / wave_size;

    for (int32_t w = 0; w < n_waves; w++) {
        int32_t ws = w * wave_size;
        int32_t we = ws + wave_size;
        if (we > n_ctas) we = n_ctas;

        for (int32_t k = 0; k < k_steps; k++) {
            for (int32_t i = ws; i < we; i++) {
                int32_t mi = schedule_mn[i * 2];
                int32_t ni = schedule_mn[i * 2 + 1];
                uint64_t ak = ((uint64_t)(mi & 0xFFFF) << 16) | (uint64_t)(k & 0xFFFF);
                l2_total += a_tile_bytes;
                dram_total += (int64_t)lru_access(&cache, ak, a_tile_bytes) * a_tile_bytes;
                uint64_t bk = (1ULL << 48) | ((uint64_t)(ni & 0xFFFF) << 16) | (uint64_t)(k & 0xFFFF);
                l2_total += b_tile_bytes;
                dram_total += (int64_t)lru_access(&cache, bk, b_tile_bytes) * b_tile_bytes;
            }
        }
        for (int32_t i = ws; i < we; i++) {
            int32_t mi = schedule_mn[i * 2];
            int32_t ni = schedule_mn[i * 2 + 1];
            uint64_t ck = (2ULL << 48) | ((uint64_t)(mi & 0xFFFF) << 16) | (uint64_t)(ni & 0xFFFF);
            l2_total += c_tile_bytes;
            dram_total += (int64_t)lru_access(&cache, ck, c_tile_bytes) * c_tile_bytes;
        }
    }
    lru_free(&cache);
    out[0] = l2_total;
    out[1] = dram_total;
}

/* ─── N-major GEMM LRU replay ─── */

/*
 * ``simulate_gemm_lru`` accepts arbitrary source orders, so it needs a hash
 * table for cache lookup.  The mcBLAS N-major caller has a stricter contract:
 * its stream is exactly ``(m = cta % mt, n = cta / mt)``.  Keep the identical
 * LRU state transition, but use the dense identities already implied by that
 * stream instead of materializing ``[mt * nt, 2]`` coordinates and hashing
 * every object on every access.
 *
 * The direct maps cover every physical A, B and C object that this replay can
 * address.  They are cache membership indices, not traffic shortcuts: every
 * original access still performs the same MRU promotion or LRU eviction.
 */

static int32_t *new_n_major_index_map(int64_t count) {
    if (count <= 0 || (uint64_t)count > SIZE_MAX / sizeof(int32_t)) {
        fail_loud("N-major GEMM index map size is invalid");
    }
    int32_t *map = (int32_t *)checked_malloc((size_t)count * sizeof(int32_t));
    memset(map, 0xFF, (size_t)count * sizeof(int32_t));
    return map;
}

static void clear_n_major_slot_index(
    LRU *c, int32_t slot_index,
    int32_t *a_index, int32_t *b_index, int32_t *c_index
) {
    Slot *slot = &c->slots[slot_index];
    if (slot->hash_idx == 0) a_index[slot->hash_next] = -1;
    else if (slot->hash_idx == 1) b_index[slot->hash_next] = -1;
    else if (slot->hash_idx == 2) c_index[slot->hash_next] = -1;
    else fail_loud("N-major GEMM slot has no direct index class");
}

static void evict_n_major_to_fit(
    LRU *c, int32_t *a_index, int32_t *b_index, int32_t *c_index
) {
    while (c->cur_bytes > c->max_bytes && c->lru_tail >= 0) {
        int32_t victim = c->lru_tail;
        Slot *slot = &c->slots[victim];
        c->cur_bytes -= slot->size;
        list_remove(c, victim);
        clear_n_major_slot_index(c, victim, a_index, b_index, c_index);
        slot->hash_idx = -1;
        slot->hash_next = -1;
        slot->lru_prev = -1;
        slot->lru_next = c->free_head;
        c->free_head = victim;
        c->n_used--;
    }
}

static int n_major_lru_access(
    LRU *c, int32_t *index_map, int32_t object_index,
    int32_t index_class, uint64_t key, int32_t size,
    int32_t *a_index, int32_t *b_index, int32_t *c_index
) {
    int32_t slot_index = index_map[object_index];
    if (slot_index >= 0) {
        list_remove(c, slot_index);
        list_push_head(c, slot_index);
        return 0;
    }
    if (c->free_head < 0) {
        /* ``compute_max_slots`` leaves enough slots for all byte-resident
         * entries plus slack.  Reaching this branch would violate that shared
         * cache invariant instead of silently dropping an access. */
        fail_loud("N-major GEMM LRU slot pool exhausted");
    }
    slot_index = c->free_head;
    Slot *slot = &c->slots[slot_index];
    c->free_head = slot->lru_next;
    slot->key = key;
    slot->size = size;
    slot->dirty = 0;
    slot->hash_idx = index_class;
    slot->hash_next = object_index;
    c->n_used++;
    c->cur_bytes += size;
    list_push_head(c, slot_index);
    index_map[object_index] = slot_index;
    evict_n_major_to_fit(c, a_index, b_index, c_index);
    return 1;
}

/*
 * In a sufficiently large N-major GEMM, each A(m, k) object is revisited
 * only after every other A object has been touched.  When that distinct A
 * working set exceeds L2, every A access is therefore a miss.  C(m, n) is
 * single-use by GEMM semantics.  The compressed replay below records those
 * one-use A/C streams as an exact byte distance between B accesses, then
 * evaluates B reuse with an order-statistics LRU stack.  It is not an
 * approximation: a B hit is precisely the variable-size LRU reuse-distance
 * predicate, including all distinct A/C bytes and newer B objects.
 */
typedef struct {
    int32_t left;
    int32_t right;
    int32_t parent;
    int32_t subtree_count;
    int64_t last_anonymous_bytes;
    uint32_t priority;
    uint8_t active;
} NMajorBRecencyNode;

static int32_t n_major_b_subtree_count(
    const NMajorBRecencyNode *nodes, int32_t node
) {
    return node < 0 ? 0 : nodes[node].subtree_count;
}

static void n_major_b_update(NMajorBRecencyNode *nodes, int32_t node) {
    nodes[node].subtree_count = 1
        + n_major_b_subtree_count(nodes, nodes[node].left)
        + n_major_b_subtree_count(nodes, nodes[node].right);
}

static int32_t n_major_b_merge(
    NMajorBRecencyNode *nodes, int32_t left, int32_t right
) {
    if (left < 0) return right;
    if (right < 0) return left;
    if (nodes[left].priority < nodes[right].priority) {
        int32_t child = n_major_b_merge(nodes, nodes[left].right, right);
        nodes[left].right = child;
        if (child >= 0) nodes[child].parent = left;
        n_major_b_update(nodes, left);
        return left;
    }
    int32_t child = n_major_b_merge(nodes, left, nodes[right].left);
    nodes[right].left = child;
    if (child >= 0) nodes[child].parent = right;
    n_major_b_update(nodes, right);
    return right;
}

static void n_major_b_split(
    NMajorBRecencyNode *nodes, int32_t root, int32_t first_count,
    int32_t *left_out, int32_t *right_out
) {
    if (root < 0) {
        *left_out = -1;
        *right_out = -1;
        return;
    }
    int32_t left_count = n_major_b_subtree_count(nodes, nodes[root].left);
    if (first_count <= left_count) {
        int32_t left, right;
        n_major_b_split(nodes, nodes[root].left, first_count, &left, &right);
        nodes[root].left = right;
        if (right >= 0) nodes[right].parent = root;
        n_major_b_update(nodes, root);
        if (left >= 0) nodes[left].parent = -1;
        nodes[root].parent = -1;
        *left_out = left;
        *right_out = root;
        return;
    }
    int32_t left, right;
    n_major_b_split(
        nodes, nodes[root].right, first_count - left_count - 1,
        &left, &right);
    nodes[root].right = left;
    if (left >= 0) nodes[left].parent = root;
    n_major_b_update(nodes, root);
    if (right >= 0) nodes[right].parent = -1;
    nodes[root].parent = -1;
    *left_out = root;
    *right_out = right;
}

static int32_t n_major_b_rank(
    const NMajorBRecencyNode *nodes, int32_t node
) {
    int32_t rank = n_major_b_subtree_count(nodes, nodes[node].left);
    int32_t current = node;
    while (nodes[current].parent >= 0) {
        int32_t parent = nodes[current].parent;
        if (nodes[parent].right == current) {
            rank += 1 + n_major_b_subtree_count(nodes, nodes[parent].left);
        }
        current = parent;
    }
    return rank;
}

static int32_t n_major_b_remove_at_rank(
    NMajorBRecencyNode *nodes, int32_t root, int32_t rank,
    int32_t *removed_out
) {
    int32_t left, middle_and_right, middle, right;
    n_major_b_split(nodes, root, rank, &left, &middle_and_right);
    n_major_b_split(nodes, middle_and_right, 1, &middle, &right);
    if (middle < 0 || middle != *removed_out) {
        fail_loud("N-major GEMM B recency tree lost its node");
    }
    nodes[middle].left = -1;
    nodes[middle].right = -1;
    nodes[middle].parent = -1;
    nodes[middle].subtree_count = 1;
    int32_t merged = n_major_b_merge(nodes, left, right);
    if (merged >= 0) nodes[merged].parent = -1;
    return merged;
}

static int n_major_b_access(
    NMajorBRecencyNode *nodes, int32_t *root, int32_t b_object,
    int64_t anonymous_bytes, int64_t capacity_bytes, int32_t b_tile_bytes,
    int force_hit
) {
    NMajorBRecencyNode *node = &nodes[b_object];
    int miss = 1;
    if (node->active) {
        int32_t rank = n_major_b_rank(nodes, b_object);
        if (!force_hit) {
            int64_t reuse_distance = anonymous_bytes
                - node->last_anonymous_bytes
                + (int64_t)rank * b_tile_bytes;
            miss = reuse_distance + b_tile_bytes > capacity_bytes;
        } else {
            miss = 0;
        }
        *root = n_major_b_remove_at_rank(nodes, *root, rank, &b_object);
    } else if (force_hit) {
        fail_loud("N-major GEMM B final touch has no prior access");
    }
    node->active = 1;
    node->last_anonymous_bytes = anonymous_bytes;
    node->left = -1;
    node->right = -1;
    node->parent = -1;
    node->subtree_count = 1;
    *root = n_major_b_merge(nodes, b_object, *root);
    if (*root >= 0) nodes[*root].parent = -1;
    return miss;
}

static int64_t n_major_a_stream_position(
    int32_t cta, int32_t k, int32_t n_ctas, int32_t wave_size,
    int32_t k_steps
) {
    int32_t wave = cta / wave_size;
    int32_t wave_start = wave * wave_size;
    int32_t wave_end = wave_start + wave_size;
    if (wave_end > n_ctas) wave_end = n_ctas;
    int32_t wave_length = wave_end - wave_start;
    return (int64_t)wave_start * k_steps + (int64_t)k * wave_length
        + cta - wave_start;
}

static int64_t n_major_minimum_intervening_a_accesses(
    int32_t mt, int32_t n_ctas, int32_t k_steps, int32_t wave_size
) {
    int64_t minimum = INT64_MAX;
    for (int32_t cta = 0; cta + mt < n_ctas; cta++) {
        for (int32_t k = 0; k < k_steps; k += k_steps - 1) {
            int64_t distance = n_major_a_stream_position(
                cta + mt, k, n_ctas, wave_size, k_steps)
                - n_major_a_stream_position(
                    cta, k, n_ctas, wave_size, k_steps);
            if (distance <= 0) {
                fail_loud("N-major GEMM A stream order is invalid");
            }
            if (distance - 1 < minimum) minimum = distance - 1;
            if (k_steps == 1) break;
        }
    }
    return minimum;
}

static void simulate_gemm_n_major_compressed_lru(
    int32_t mt, int32_t nt, int32_t k_steps,
    int32_t a_tile_bytes, int32_t b_tile_bytes, int32_t c_tile_bytes,
    int32_t wave_size, int64_t l2_capacity_bytes,
    int64_t n_ctas64, int64_t b_count, int64_t *out
) {
    NMajorBRecencyNode *nodes = (NMajorBRecencyNode *)checked_calloc(
        (size_t)b_count, sizeof(NMajorBRecencyNode));
    for (int32_t index = 0; index < (int32_t)b_count; index++) {
        nodes[index].left = -1;
        nodes[index].right = -1;
        nodes[index].parent = -1;
        nodes[index].priority = mix32((uint64_t)index + 1);
    }
    int32_t n_ctas = (int32_t)n_ctas64;
    int32_t n_waves = (n_ctas + wave_size - 1) / wave_size;
    int32_t b_root = -1;
    int64_t anonymous_bytes = 0;
    int64_t b_dram_bytes = 0;
    for (int32_t wave = 0; wave < n_waves; wave++) {
        int32_t start = wave * wave_size;
        int32_t end = start + wave_size;
        if (end > n_ctas) end = n_ctas;
        for (int32_t k = 0; k < k_steps; k++) {
            int32_t mi = start % mt;
            int32_t ni = start / mt;
            int32_t cta = start;
            while (cta < end) {
                int32_t run_length = mt - mi;
                if (run_length > end - cta) run_length = end - cta;
                int32_t b_object = ni * k_steps + k;
                anonymous_bytes += a_tile_bytes;
                b_dram_bytes += (int64_t)n_major_b_access(
                    nodes, &b_root, b_object, anonymous_bytes,
                    l2_capacity_bytes, b_tile_bytes, 0) * b_tile_bytes;
                if (run_length > 1) {
                    anonymous_bytes += (int64_t)(run_length - 1)
                        * a_tile_bytes;
                    (void)n_major_b_access(
                        nodes, &b_root, b_object, anonymous_bytes,
                        l2_capacity_bytes, b_tile_bytes, 1);
                }
                cta += run_length;
                mi += run_length;
                if (mi == mt) {
                    mi = 0;
                    ni++;
                }
            }
        }
        anonymous_bytes += (int64_t)(end - start) * c_tile_bytes;
    }
    int64_t k_ctas = n_ctas64 * k_steps;
    out[0] = k_ctas * ((int64_t)a_tile_bytes + b_tile_bytes)
        + n_ctas64 * c_tile_bytes;
    out[1] = k_ctas * a_tile_bytes + n_ctas64 * c_tile_bytes
        + b_dram_bytes;
    free(nodes);
}

static int simulate_gemm_n_major_all_b_miss_lru(
    int32_t mt, int32_t nt, int32_t k_steps,
    int32_t a_tile_bytes, int32_t b_tile_bytes, int32_t c_tile_bytes,
    int32_t wave_size, int64_t l2_capacity_bytes,
    int64_t n_ctas64, int64_t b_count, int64_t *out
) {
    /* The B touch after each A run makes that B tile most-recent.  Record the
     * anonymous A/C byte position at that final touch.  If the one-use bytes
     * alone exceed the remaining capacity before B is next requested, the
     * next B access must miss regardless of all other B stack state. */
    int64_t *last_b_anonymous = (int64_t *)checked_malloc(
        (size_t)b_count * sizeof(int64_t));
    for (int32_t index = 0; index < (int32_t)b_count; index++) {
        last_b_anonymous[index] = -1;
    }
    int32_t n_ctas = (int32_t)n_ctas64;
    int32_t n_waves = (n_ctas + wave_size - 1) / wave_size;
    int64_t anonymous_bytes = 0;
    int64_t b_miss_count = 0;
    for (int32_t wave = 0; wave < n_waves; wave++) {
        int32_t start = wave * wave_size;
        int32_t end = start + wave_size;
        if (end > n_ctas) end = n_ctas;
        for (int32_t k = 0; k < k_steps; k++) {
            int32_t mi = start % mt;
            int32_t ni = start / mt;
            int32_t cta = start;
            while (cta < end) {
                int32_t run_length = mt - mi;
                if (run_length > end - cta) run_length = end - cta;
                int32_t b_object = ni * k_steps + k;
                anonymous_bytes += a_tile_bytes;
                if (last_b_anonymous[b_object] >= 0
                        && anonymous_bytes - last_b_anonymous[b_object]
                            + b_tile_bytes <= l2_capacity_bytes) {
                    free(last_b_anonymous);
                    return 0;
                }
                b_miss_count++;
                anonymous_bytes += (int64_t)(run_length - 1) * a_tile_bytes;
                last_b_anonymous[b_object] = anonymous_bytes;
                cta += run_length;
                mi += run_length;
                if (mi == mt) {
                    mi = 0;
                    ni++;
                }
            }
        }
        anonymous_bytes += (int64_t)(end - start) * c_tile_bytes;
    }
    int64_t k_ctas = n_ctas64 * k_steps;
    out[0] = k_ctas * ((int64_t)a_tile_bytes + b_tile_bytes)
        + n_ctas64 * c_tile_bytes;
    out[1] = k_ctas * a_tile_bytes + n_ctas64 * c_tile_bytes
        + b_miss_count * b_tile_bytes;
    free(last_b_anonymous);
    return 1;
}

void simulate_gemm_n_major_lru(
    int32_t mt,
    int32_t nt,
    int32_t k_steps,
    int32_t a_tile_bytes,
    int32_t b_tile_bytes,
    int32_t c_tile_bytes,
    int32_t wave_size,
    int64_t l2_capacity_bytes,
    int64_t *out
) {
    int64_t n_ctas64 = (int64_t)mt * nt;
    int64_t a_count = (int64_t)mt * k_steps;
    int64_t b_count = (int64_t)nt * k_steps;
    if (mt <= 0 || nt <= 0 || k_steps <= 0 || wave_size <= 0
            || n_ctas64 > INT32_MAX || a_count > INT32_MAX
            || b_count > INT32_MAX || a_tile_bytes <= 0
            || b_tile_bytes <= 0 || c_tile_bytes <= 0
            || l2_capacity_bytes <= 0) {
        fail_loud("N-major GEMM dimensions are invalid");
    }
    int32_t n_ctas = (int32_t)n_ctas64;
    /* The compressed stream is exact only when each repeated B can remain
     * resident between adjacent A tiles and a full A identity sweep is too
     * large for L2.  All other inputs retain the generic direct replay. */
    int64_t minimum_intervening_a_accesses = (
        n_major_minimum_intervening_a_accesses(
            mt, n_ctas, k_steps, wave_size));
    if (a_tile_bytes <= l2_capacity_bytes - b_tile_bytes
            && minimum_intervening_a_accesses
                > l2_capacity_bytes / a_tile_bytes) {
        if (simulate_gemm_n_major_all_b_miss_lru(
                mt, nt, k_steps, a_tile_bytes, b_tile_bytes, c_tile_bytes,
                wave_size, l2_capacity_bytes, n_ctas64, b_count, out)) {
            return;
        }
        simulate_gemm_n_major_compressed_lru(
            mt, nt, k_steps, a_tile_bytes, b_tile_bytes, c_tile_bytes,
            wave_size, l2_capacity_bytes, n_ctas64, b_count, out);
        return;
    }
    int32_t max_slots = compute_max_slots(
        a_tile_bytes, b_tile_bytes, c_tile_bytes, l2_capacity_bytes);
    int32_t *a_index = new_n_major_index_map(a_count);
    int32_t *b_index = new_n_major_index_map(b_count);
    int32_t *c_index = new_n_major_index_map(n_ctas64);
    LRU cache;
    lru_init(&cache, max_slots, l2_capacity_bytes);
    int64_t l2_total = 0, dram_total = 0;
    int32_t n_waves = (n_ctas + wave_size - 1) / wave_size;

    for (int32_t wave = 0; wave < n_waves; wave++) {
        int32_t start = wave * wave_size;
        int32_t end = start + wave_size;
        if (end > n_ctas) end = n_ctas;
        for (int32_t k = 0; k < k_steps; k++) {
            int32_t mi = start % mt;
            int32_t ni = start / mt;
            int32_t cta = start;
            while (cta < end) {
                int32_t run_length = mt - mi;
                if (run_length > end - cta) run_length = end - cta;
                int32_t b_object = ni * k_steps + k;
                uint64_t bk = (1ULL << 48)
                    | ((uint64_t)(ni & 0xFFFF) << 16)
                    | (uint64_t)(k & 0xFFFF);
                /* A B tile is touched after every A tile in this N-major
                 * run.  Once it is accessed it cannot be evicted before the
                 * run ends when all possible intervening A tiles plus B fit
                 * together.  In that case the repeated B touches have one
                 * net state transition: promote B after the final A.  This
                 * preserves the exact final LRU order while avoiding the
                 * same remove/push pair for every intermediate B hit. */
                int defer_repeated_b = (
                    run_length > 1
                    && (int64_t)run_length * a_tile_bytes + b_tile_bytes
                        <= l2_capacity_bytes);
                for (int32_t offset = 0; offset < run_length; offset++) {
                    int32_t a_object = mi * k_steps + k;
                    uint64_t ak = ((uint64_t)(mi & 0xFFFF) << 16)
                        | (uint64_t)(k & 0xFFFF);
                    l2_total += a_tile_bytes + b_tile_bytes;
                    dram_total += (int64_t)n_major_lru_access(
                        &cache, a_index, a_object, 0, ak, a_tile_bytes,
                        a_index, b_index, c_index) * a_tile_bytes;
                    if (!defer_repeated_b || offset == 0) {
                        dram_total += (int64_t)n_major_lru_access(
                            &cache, b_index, b_object, 1, bk, b_tile_bytes,
                            a_index, b_index, c_index) * b_tile_bytes;
                    }
                    mi++;
                }
                if (defer_repeated_b) {
                    int32_t b_slot = b_index[b_object];
                    if (b_slot < 0) {
                        fail_loud("N-major GEMM deferred B tile was evicted");
                    }
                    list_remove(&cache, b_slot);
                    list_push_head(&cache, b_slot);
                }
                cta += run_length;
                if (mi == mt) {
                    mi = 0;
                    ni++;
                }
            }
        }
        int32_t mi = start % mt;
        int32_t ni = start / mt;
        for (int32_t cta = start; cta < end; cta++) {
            int32_t c_object = ni * mt + mi;
            uint64_t ck = (2ULL << 48)
                | ((uint64_t)(mi & 0xFFFF) << 16)
                | (uint64_t)(ni & 0xFFFF);
            l2_total += c_tile_bytes;
            dram_total += (int64_t)n_major_lru_access(
                &cache, c_index, c_object, 2, ck, c_tile_bytes,
                a_index, b_index, c_index) * c_tile_bytes;
            mi++;
            if (mi == mt) {
                mi = 0;
                ni++;
            }
        }
    }
    lru_free(&cache);
    free(a_index);
    free(b_index);
    free(c_index);
    out[0] = l2_total;
    out[1] = dram_total;
}

void simulate_gemm_lru_tails(
    const int32_t *schedule_mn,
    int32_t n_ctas,
    int32_t m_extent,
    int32_t n_extent,
    int32_t k_extent,
    int32_t tile_m,
    int32_t tile_n,
    int32_t tile_k,
    int32_t element_bytes,
    int32_t wave_size,
    int64_t l2_capacity_bytes,
    const int64_t *profile_occupancy,
    const double *profile_fraction,
    int32_t profile_points,
    int64_t *out
) {
    int32_t maximum_a = tile_m * tile_k * element_bytes;
    int32_t maximum_b = tile_k * tile_n * element_bytes;
    int32_t maximum_c = tile_m * tile_n * element_bytes;
    int32_t max_slots = compute_max_slots(
        maximum_a, maximum_b, maximum_c, l2_capacity_bytes);
    int32_t k_steps = (k_extent + tile_k - 1) / tile_k;
    MissCore core;
    miss_core_init_sized(
        &core, l2_capacity_bytes, profile_occupancy, profile_fraction,
        profile_points, (int64_t)n_ctas * (2 * (int64_t)k_steps + 1),
        max_slots);
    int64_t l2_total = 0, dram_total = 0;
    int64_t a_l2_bytes = 0, a_dram_bytes = 0;
    int64_t b_l2_bytes = 0, b_dram_bytes = 0;
    int64_t output_l2_bytes = 0;
    int32_t n_waves = (n_ctas + wave_size - 1) / wave_size;

    for (int32_t w = 0; w < n_waves; w++) {
        int32_t ws = w * wave_size;
        int32_t we = ws + wave_size;
        if (we > n_ctas) we = n_ctas;
        for (int32_t ki = 0; ki < k_steps; ki++) {
            int32_t current_k = tile_k;
            if ((ki + 1) * tile_k > k_extent) {
                current_k = k_extent - ki * tile_k;
            }
            for (int32_t i = ws; i < we; i++) {
                int32_t mi = schedule_mn[i * 2];
                int32_t ni = schedule_mn[i * 2 + 1];
                int32_t current_m = tile_m;
                int32_t current_n = tile_n;
                if ((mi + 1) * tile_m > m_extent) {
                    current_m = m_extent - mi * tile_m;
                }
                if ((ni + 1) * tile_n > n_extent) {
                    current_n = n_extent - ni * tile_n;
                }
                int32_t a_bytes = current_m * current_k * element_bytes;
                int32_t b_bytes = current_k * current_n * element_bytes;
                uint64_t ak = ((uint64_t)(mi & 0xFFFF) << 16)
                    | (uint64_t)(ki & 0xFFFF);
                uint64_t bk = (1ULL << 48)
                    | ((uint64_t)(ni & 0xFFFF) << 16)
                    | (uint64_t)(ki & 0xFFFF);
                int64_t a_miss_bytes = (int64_t)(
                    miss_core_touch(&core, ak, a_bytes) + 0.5);
                int64_t b_miss_bytes = (int64_t)(
                    miss_core_touch(&core, bk, b_bytes) + 0.5);
                a_l2_bytes += a_bytes;
                b_l2_bytes += b_bytes;
                a_dram_bytes += a_miss_bytes;
                b_dram_bytes += b_miss_bytes;
                l2_total += a_bytes + b_bytes;
                dram_total += a_miss_bytes + b_miss_bytes;
            }
        }
        for (int32_t i = ws; i < we; i++) {
            int32_t mi = schedule_mn[i * 2];
            int32_t ni = schedule_mn[i * 2 + 1];
            int32_t current_m = tile_m;
            int32_t current_n = tile_n;
            if ((mi + 1) * tile_m > m_extent) {
                current_m = m_extent - mi * tile_m;
            }
            if ((ni + 1) * tile_n > n_extent) {
                current_n = n_extent - ni * tile_n;
            }
            int32_t c_bytes = current_m * current_n * element_bytes;
            uint64_t ck = (2ULL << 48)
                | ((uint64_t)(mi & 0xFFFF) << 16)
                | (uint64_t)(ni & 0xFFFF);
            l2_total += c_bytes;
            output_l2_bytes += c_bytes;
            dram_total += (int64_t)(
                miss_core_touch(&core, ck, c_bytes) + 0.5);
        }
    }
    miss_core_free(&core);
    out[0] = l2_total;
    out[1] = dram_total;
    out[2] = a_l2_bytes;
    out[3] = a_dram_bytes;
    out[4] = b_l2_bytes;
    out[5] = b_dram_bytes;
    out[6] = output_l2_bytes;
}

/* ─── GEMM swizzle search: build schedule + LRU in one C call ─── */

/*
 * Build swizzled schedule into pre-allocated buffer.
 * Returns number of valid CTAs, or -1 if count != mt*nt (invalid schedule).
 * conv: 0=M, 1=N
 */
static int32_t build_swizzled_schedule(
    int32_t *buf,       /* output: [max_entries, 2] flattened */
    int32_t gx, int32_t gy,
    int32_t conv, int32_t W,
    int32_t mt, int32_t nt,
    int32_t swizzle_log2
) {
    int32_t tile_width = gx < (1 << swizzle_log2) ? gx : (1 << swizzle_log2);
    int32_t n_tiles_sw = (gx + tile_width - 1) / tile_width;
    int32_t total = n_tiles_sw * tile_width * gy;
    int32_t count = 0;
    int32_t expected = mt * nt;

    for (int32_t lid = 0; lid < total; lid++) {
        int32_t ti = lid / (tile_width * gy);
        int32_t res = lid % (tile_width * gy);
        int32_t gy_idx = res / tile_width;
        int32_t gx_idx = res % tile_width + ti * tile_width;
        if (gx_idx >= gx || gy_idx >= gy) continue;

        int32_t mi, ni;
        if (conv == 0) {        /* M convention */
            mi = gx_idx / W;
            ni = gy_idx * W + gx_idx % W;
        } else if (conv == 1) { /* N convention */
            ni = gx_idx / W;
            mi = gy_idx * W + gx_idx % W;
        } else {
            return -1;
        }
        if (mi >= 0 && mi < mt && ni >= 0 && ni < nt) {
            buf[count * 2] = mi;
            buf[count * 2 + 1] = ni;
            count++;
            if (count > expected) return -1;  /* too many */
        }
    }
    return (count == expected) ? count : -1;
}

void simulate_gemm_swizzle_search(
    int32_t grid_x, int32_t grid_y,
    int32_t conv,       /* 0=M, 1=N */
    int32_t W,
    int32_t mt, int32_t nt,
    int32_t max_sw,
    int32_t k_steps,
    int32_t a_tile_bytes, int32_t b_tile_bytes, int32_t c_tile_bytes,
    int32_t wave_size,
    int64_t l2_capacity_bytes,
    int64_t *out        /* out[0]=best_l2, out[1]=best_dram */
) {
    int32_t n_ctas = mt * nt;
    int32_t *sched_buf = (int32_t *)checked_malloc(
        (size_t)n_ctas * 2 * sizeof(int32_t));

    int32_t max_slots = compute_max_slots(a_tile_bytes, b_tile_bytes, c_tile_bytes,
                                           l2_capacity_bytes);
    LRU cache;
    lru_init(&cache, max_slots, l2_capacity_bytes);

    int64_t best_l2 = 0, best_dram = INT64_MAX;

    for (int32_t sw = 0; sw <= max_sw; sw++) {
        int32_t count = build_swizzled_schedule(
            sched_buf, grid_x, grid_y, conv, W, mt, nt, sw);
        if (count < 0) continue;  /* invalid schedule */

        /* Reset cache for this swizzle attempt */
        lru_reset(&cache);

        int64_t l2_total = 0, dram_total = 0;
        int32_t n_waves = (count + wave_size - 1) / wave_size;

        for (int32_t w = 0; w < n_waves; w++) {
            int32_t ws = w * wave_size;
            int32_t we = ws + wave_size;
            if (we > count) we = count;

            for (int32_t k = 0; k < k_steps; k++) {
                for (int32_t i = ws; i < we; i++) {
                    int32_t mi = sched_buf[i * 2];
                    int32_t ni = sched_buf[i * 2 + 1];
                    uint64_t ak = ((uint64_t)(mi & 0xFFFF) << 16) | (uint64_t)(k & 0xFFFF);
                    l2_total += a_tile_bytes;
                    dram_total += (int64_t)lru_access(&cache, ak, a_tile_bytes) * a_tile_bytes;
                    uint64_t bk = (1ULL << 48) | ((uint64_t)(ni & 0xFFFF) << 16) | (uint64_t)(k & 0xFFFF);
                    l2_total += b_tile_bytes;
                    dram_total += (int64_t)lru_access(&cache, bk, b_tile_bytes) * b_tile_bytes;
                }
            }
            for (int32_t i = ws; i < we; i++) {
                int32_t mi = sched_buf[i * 2];
                int32_t ni = sched_buf[i * 2 + 1];
                uint64_t ck = (2ULL << 48) | ((uint64_t)(mi & 0xFFFF) << 16) | (uint64_t)(ni & 0xFFFF);
                l2_total += c_tile_bytes;
                dram_total += (int64_t)lru_access(&cache, ck, c_tile_bytes) * c_tile_bytes;
            }
        }

        if (dram_total < best_dram) {
            best_dram = dram_total;
            best_l2 = l2_total;
        }
    }

    lru_free(&cache);
    free(sched_buf);
    out[0] = best_l2;
    out[1] = best_dram;
}

/* Run the swizzle search at l2_capacity_bytes and emit the best swizzle's
 * schedule into out_sched [mt*nt, 2]. Returns the CTA count, or -1 if no valid
 * swizzle exists. Lets Python compute the stack-distance hit curve on exactly
 * the schedule the GEMM traffic predictor selected (no swizzle logic in Python).
 */
int32_t gemm_swizzle_best_schedule(
    int32_t grid_x, int32_t grid_y,
    int32_t conv, int32_t W,
    int32_t mt, int32_t nt,
    int32_t max_sw,
    int32_t k_steps,
    int32_t a_tile_bytes, int32_t b_tile_bytes, int32_t c_tile_bytes,
    int32_t wave_size,
    int64_t l2_capacity_bytes,
    int32_t *out_sched   /* [mt*nt, 2] flattened */
) {
    int32_t n_ctas = mt * nt;
    int32_t *sched_buf = (int32_t *)checked_malloc(
        (size_t)n_ctas * 2 * sizeof(int32_t));

    int32_t max_slots = compute_max_slots(a_tile_bytes, b_tile_bytes, c_tile_bytes,
                                           l2_capacity_bytes);
    LRU cache;
    lru_init(&cache, max_slots, l2_capacity_bytes);

    int64_t best_dram = INT64_MAX;
    int32_t best_count = -1;

    for (int32_t sw = 0; sw <= max_sw; sw++) {
        int32_t count = build_swizzled_schedule(
            sched_buf, grid_x, grid_y, conv, W, mt, nt, sw);
        if (count < 0) continue;

        lru_reset(&cache);
        int64_t dram_total = 0;
        int32_t n_waves = (count + wave_size - 1) / wave_size;
        for (int32_t w = 0; w < n_waves; w++) {
            int32_t ws = w * wave_size;
            int32_t we = ws + wave_size; if (we > count) we = count;
            for (int32_t k = 0; k < k_steps; k++) {
                for (int32_t i = ws; i < we; i++) {
                    int32_t mi = sched_buf[i * 2];
                    int32_t ni = sched_buf[i * 2 + 1];
                    uint64_t ak = ((uint64_t)(mi & 0xFFFF) << 16) | (uint64_t)(k & 0xFFFF);
                    dram_total += (int64_t)lru_access(&cache, ak, a_tile_bytes) * a_tile_bytes;
                    uint64_t bk = (1ULL << 48) | ((uint64_t)(ni & 0xFFFF) << 16) | (uint64_t)(k & 0xFFFF);
                    dram_total += (int64_t)lru_access(&cache, bk, b_tile_bytes) * b_tile_bytes;
                }
            }
            for (int32_t i = ws; i < we; i++) {
                int32_t mi = sched_buf[i * 2];
                int32_t ni = sched_buf[i * 2 + 1];
                uint64_t ck = (2ULL << 48) | ((uint64_t)(mi & 0xFFFF) << 16) | (uint64_t)(ni & 0xFFFF);
                dram_total += (int64_t)lru_access(&cache, ck, c_tile_bytes) * c_tile_bytes;
            }
        }
        if (dram_total < best_dram) {
            best_dram = dram_total;
            best_count = count;
            memcpy(out_sched, sched_buf, (size_t)count * 2 * sizeof(int32_t));
        }
    }

    lru_free(&cache);
    free(sched_buf);
    return best_count;
}

/* ─── MoE A/B operand latency classification ───
 *
 * Runs the exact Triton grouped-GEMM schedule through TWO nested LRU caches:
 *   - full cache of capacity l2_capacity_bytes  (near-half + far-half of L2)
 *   - near cache of capacity near_bytes         (near-half only, = L2/2)
 * Each operand access is classified by where it is served:
 *   near : hit in the near cache               -> tau_local
 *   far  : miss near but hit full              -> tau_remote
 *   miss : miss full (evicted / first touch)   -> DRAM
 * A (activation) and B (weight) streams are classified separately because A is
 * indirect-addressed (pointer-chase, latency exposed) while B is prefetchable.
 *
 * Two nested LRU caches keep this O(1) per access (no stack-distance scan). The
 * near/full nesting is exact for LRU: a key resident in the near (smaller)
 * cache is necessarily resident in the full cache, so "near hit" ⊂ "full hit".
 *
 * out[6] = {A_near, A_far, A_miss, B_near, B_far, B_miss}
 */
void simulate_moe_ab(
    int32_t mt, int32_t nt, int32_t gsm,
    int32_t k_steps,
    int32_t a_tile_bytes, int32_t b_tile_bytes,
    int32_t wave_size,
    int64_t l2_capacity_bytes,
    int64_t near_bytes,
    const int32_t *tok_per_expert,
    int32_t E,
    int32_t BM,
    int64_t *out
) {
    /* ── 1. expert_of_m ── */
    int32_t *expert_of_m = (int32_t *)checked_calloc(
        (size_t)mt, sizeof(int32_t));
    int32_t pos = 0;
    for (int32_t e_id = 0; e_id < E; e_id++) {
        int32_t count = tok_per_expert[e_id];
        if (count < 0) fail_loud("tokens_per_expert cannot be negative");
        if (count == 0) continue;
        int32_t n_tiles = (count + BM - 1) / BM;
        int32_t end = pos + n_tiles;
        if (end > mt) end = mt;
        for (int32_t t = pos; t < end; t++) expert_of_m[t] = e_id;
        pos = end;
    }

    /* ── 2. grouped schedule ── */
    int32_t total_ctas = mt * nt;
    int32_t *schedule = (int32_t *)checked_malloc(
        (size_t)total_ctas * 2 * sizeof(int32_t));
    int32_t num_pid_in_group = gsm * nt;
    int32_t sched_count = 0;
    for (int32_t pid = 0; pid < total_ctas; pid++) {
        int32_t group_id = pid / num_pid_in_group;
        int32_t first_pid_m = group_id * gsm;
        int32_t group_size_m = mt - first_pid_m;
        if (group_size_m > gsm) group_size_m = gsm;
        if (group_size_m <= 0) continue;
        int32_t remainder = pid % num_pid_in_group;
        int32_t pid_m = first_pid_m + remainder % group_size_m;
        int32_t pid_n = remainder / group_size_m;
        if (pid_m < mt && pid_n < nt) {
            schedule[sched_count * 2] = pid_m;
            schedule[sched_count * 2 + 1] = pid_n;
            sched_count++;
        }
    }

    /* ── 3. dual nested LRU ── */
    int32_t min_tile = a_tile_bytes < b_tile_bytes ? a_tile_bytes : b_tile_bytes;
    int32_t max_slots = compute_max_slots(min_tile, min_tile, min_tile,
                                          l2_capacity_bytes);
    LRU full, near;
    lru_init(&full, max_slots, l2_capacity_bytes);
    lru_init(&near, max_slots, near_bytes);

    int64_t a_near=0, a_far=0, a_miss=0, b_near=0, b_far=0, b_miss=0;
    int32_t n_waves = (sched_count + wave_size - 1) / wave_size;

    for (int32_t w = 0; w < n_waves; w++) {
        int32_t ws = w * wave_size;
        int32_t we = ws + wave_size;
        if (we > sched_count) we = sched_count;
        for (int32_t k = 0; k < k_steps; k++) {
            for (int32_t i = ws; i < we; i++) {
                int32_t mi = schedule[i * 2];
                int32_t ni = schedule[i * 2 + 1];
                int32_t e = expert_for_tile(expert_of_m, mi, mt);

                uint64_t ak = ((uint64_t)e << 32) | ((uint64_t)(mi & 0xFFFF) << 16) | (uint64_t)(k & 0xFFFF);
                int a_full_miss = lru_access(&full, ak, a_tile_bytes);
                int a_near_miss = lru_access(&near, ak, a_tile_bytes);
                if (a_full_miss) a_miss++;
                else if (a_near_miss) a_far++;
                else a_near++;

                uint64_t bk = (1ULL << 48) | ((uint64_t)e << 32) | ((uint64_t)(ni & 0xFFFF) << 16) | (uint64_t)(k & 0xFFFF);
                int b_full_miss = lru_access(&full, bk, b_tile_bytes);
                int b_near_miss = lru_access(&near, bk, b_tile_bytes);
                if (b_full_miss) b_miss++;
                else if (b_near_miss) b_far++;
                else b_near++;
            }
        }
    }

    lru_free(&full);
    lru_free(&near);
    free(schedule);
    free(expert_of_m);
    out[0]=a_near; out[1]=a_far; out[2]=a_miss;
    out[3]=b_near; out[4]=b_far; out[5]=b_miss;
}

void simulate_moe_full(
    int32_t mt, int32_t nt, int32_t gsm,
    int32_t k_steps,
    int32_t a_tile_bytes, int32_t b_tile_bytes, int32_t c_tile_bytes,
    int32_t wave_size,
    int64_t l2_capacity_bytes,
    const int32_t *tok_per_expert,
    int32_t E,
    int32_t BM,
    int32_t EM,
    int64_t *out
) {
    if (BM <= 0 || EM <= 0 || mt <= 0 || (EM + BM - 1) / BM != mt) {
        fail_loud("MoE padded rows disagree with the M-tile geometry");
    }
    if (a_tile_bytes % BM != 0 || c_tile_bytes % BM != 0) {
        fail_loud("MoE A/C tile bytes must be divisible by BM");
    }

    /* ── 1. Build expert and valid-row metadata for populated M tiles ── */
    int32_t *expert_of_m = (int32_t *)checked_calloc(
        (size_t)mt, sizeof(int32_t));
    int32_t *valid_rows_of_m = (int32_t *)checked_calloc(
        (size_t)mt, sizeof(int32_t));
    int32_t pos = 0;
    for (int32_t e_id = 0; e_id < E; e_id++) {
        int32_t count = tok_per_expert[e_id];
        if (count < 0) fail_loud("tokens_per_expert cannot be negative");
        if (count == 0) continue;
        int32_t n_tiles = (count + BM - 1) / BM;
        int32_t end = pos + n_tiles;
        if (end > mt) {
            fail_loud("MoE populated M tiles exceed launch geometry");
        }
        for (int32_t t = pos; t < end; t++) {
            expert_of_m[t] = e_id;
            int32_t local_tile = t - pos;
            int32_t remaining = count - local_tile * BM;
            valid_rows_of_m[t] = remaining < BM ? remaining : BM;
        }
        pos = end;
    }
    int32_t populated_mt = pos;
    if (populated_mt <= 0) {
        fail_loud("MoE routing has no populated M tile");
    }

    /* ── 2. Build grouped schedule (Triton fused_moe order) ── */
    int32_t total_ctas = mt * nt;
    int32_t *schedule = (int32_t *)checked_malloc(
        (size_t)total_ctas * 2 * sizeof(int32_t));
    int32_t num_pid_in_group = gsm * nt;
    int32_t sched_count = 0;

    for (int32_t pid = 0; pid < total_ctas; pid++) {
        int32_t group_id = pid / num_pid_in_group;
        int32_t first_pid_m = group_id * gsm;
        int32_t group_size_m = mt - first_pid_m;
        if (group_size_m > gsm) group_size_m = gsm;
        if (group_size_m <= 0) continue;

        int32_t remainder = pid % num_pid_in_group;
        int32_t pid_m = first_pid_m + remainder % group_size_m;
        int32_t pid_n = remainder / group_size_m;

        if (pid_m < mt && pid_n < nt) {
            schedule[sched_count * 2] = pid_m;
            schedule[sched_count * 2 + 1] = pid_n;
            sched_count++;
        }
    }

    /* ── 3. Run LRU simulation (same logic as simulate_moe_lru) ── */
    int32_t max_slots = compute_max_slots(a_tile_bytes, b_tile_bytes, c_tile_bytes,
                                           l2_capacity_bytes);
    LRU cache;
    lru_init(&cache, max_slots, l2_capacity_bytes);

    int64_t l2_total = 0, dram_total = 0;
    int32_t n_waves = (sched_count + wave_size - 1) / wave_size;

    for (int32_t w = 0; w < n_waves; w++) {
        int32_t ws = w * wave_size;
        int32_t we = ws + wave_size;
        if (we > sched_count) we = sched_count;

        for (int32_t k = 0; k < k_steps; k++) {
            for (int32_t i = ws; i < we; i++) {
                int32_t mi = schedule[i * 2];
                int32_t ni = schedule[i * 2 + 1];
                if (mi >= populated_mt) continue;
                int32_t e = expert_for_tile(expert_of_m, mi, mt);
                int32_t valid_rows = valid_rows_of_m[mi];
                int32_t a_valid_bytes =
                    (a_tile_bytes / BM) * valid_rows;

                uint64_t ak = ((uint64_t)e << 32) | ((uint64_t)(mi & 0xFFFF) << 16) | (uint64_t)(k & 0xFFFF);
                int a_miss = lru_access(&cache, ak, a_valid_bytes);
                dram_total += (int64_t)a_miss * a_valid_bytes;
                l2_total += (int64_t)a_miss * a_valid_bytes;

                uint64_t bk = (1ULL << 48) | ((uint64_t)e << 32) | ((uint64_t)(ni & 0xFFFF) << 16) | (uint64_t)(k & 0xFFFF);
                l2_total += b_tile_bytes;
                dram_total += (int64_t)lru_access(&cache, bk, b_tile_bytes) * b_tile_bytes;
            }
        }
        for (int32_t i = ws; i < we; i++) {
            int32_t mi = schedule[i * 2];
            int32_t ni = schedule[i * 2 + 1];
            if (mi >= populated_mt) continue;
            int32_t e = expert_for_tile(expert_of_m, mi, mt);
            int32_t valid_rows = valid_rows_of_m[mi];
            int32_t c_valid_bytes =
                (c_tile_bytes / BM) * valid_rows;
            uint64_t ck = (2ULL << 48) | ((uint64_t)e << 32) | ((uint64_t)(mi & 0xFFFF) << 16) | (uint64_t)(ni & 0xFFFF);
            l2_total += c_valid_bytes;
            dram_total += (int64_t)lru_access(&cache, ck, c_valid_bytes) * c_valid_bytes;
        }
    }

    lru_free(&cache);
    free(schedule);
    free(valid_rows_of_m);
    free(expert_of_m);
    out[0] = l2_total;
    out[1] = dram_total;
}

/* ─── CANN GroupedMatmul operand residency ───
 *
 * CANN 8.5.1 assigns each expert's output tiles to the AIC ring in the
 * order produced by MNBlockIdxCompute.  Every AIC then walks K with A followed
 * by B.  This simulation preserves that source order, groups simultaneously
 * resident AIC work into waves, and uses the published shared-L2 capacity to
 * classify exact full-tile A/B requests as resident or DRAM-missing.  Tail
 * requests and output writes still participate in LRU replacement even though
 * they are not eligible for the exact full-tile timing service.
 *
 * out has worker_count * 4 + 2 entries.  Per-worker fields are
 * {A_hit, A_miss, B_hit, B_miss}; the final two fields are total L2 operand
 * read bytes and DRAM-missing operand read bytes.
 */
typedef struct {
    int32_t expert;
    int32_t m_index;
    int32_t n_index;
    int32_t current_m;
    int32_t current_n;
    int32_t worker;
} CannGroupedTile;

static int32_t positive_lcm(int32_t left, int32_t right) {
    int32_t a = left;
    int32_t b = right;
    while (b != 0) {
        int32_t remainder = a % b;
        a = b;
        b = remainder;
    }
    return left / a * right;
}

static void cann_mn_block_index(
    int32_t relative_block, int32_t block_dim_m, int32_t block_dim_n,
    int32_t *m_index, int32_t *n_index
) {
    const int32_t threshold_block_num = 8;
    const int32_t threshold_dim_m = 5;
    if (block_dim_m <= threshold_dim_m) {
        *m_index = relative_block / block_dim_n;
        *n_index = relative_block % block_dim_n;
        return;
    }
    int32_t threshold_m_dim_n = threshold_block_num * block_dim_n;
    int32_t full_threshold_extent =
        block_dim_m * block_dim_n / threshold_m_dim_n
        * threshold_m_dim_n;
    int32_t current_threshold_m = relative_block >= full_threshold_extent
        ? block_dim_m % threshold_block_num : threshold_block_num;
    if (current_threshold_m <= 0) {
        fail_loud("CANN GroupedMatmul M threshold is empty");
    }
    int32_t current_threshold_mn =
        current_threshold_m * threshold_block_num;
    int32_t local_n_extent =
        current_threshold_m * block_dim_n / current_threshold_mn
        * current_threshold_mn;
    int32_t current_threshold_n =
        relative_block % threshold_m_dim_n >= local_n_extent
        ? block_dim_n % threshold_block_num : threshold_block_num;
    if (current_threshold_n <= 0) {
        fail_loud("CANN GroupedMatmul N threshold is empty");
    }
    int32_t local_relative =
        relative_block % threshold_m_dim_n % current_threshold_mn;
    *m_index =
        local_relative % current_threshold_m
        + relative_block / threshold_m_dim_n * threshold_block_num;
    *n_index =
        (local_relative
         + local_relative / positive_lcm(
             current_threshold_m, current_threshold_n))
        % current_threshold_n
        + relative_block % threshold_m_dim_n
        / current_threshold_mn * threshold_block_num;
}

void simulate_cann_grouped_moe_operands(
    int32_t projection_k, int32_t projection_n,
    int32_t tile_m, int32_t tile_n, int32_t tile_k,
    int32_t element_bytes, int32_t worker_count,
    int64_t l2_capacity_bytes,
    int32_t b_l2_cache_disabled,
    const int32_t *tokens_per_expert, int32_t expert_count,
    const int64_t *profile_occupancy,
    const double *profile_fraction,
    int32_t profile_points,
    int64_t *out
) {
    if (projection_k <= 0 || projection_n <= 0
            || tile_m <= 0 || tile_n <= 0 || tile_k <= 0
            || element_bytes <= 0 || worker_count <= 0
            || l2_capacity_bytes <= 0 || expert_count <= 0) {
        fail_loud("CANN GroupedMatmul residency inputs must be positive");
    }
    if (b_l2_cache_disabled != 0 && b_l2_cache_disabled != 1) {
        fail_loud("CANN GroupedMatmul B cache policy must be zero or one");
    }
    int32_t n_tiles = (projection_n + tile_n - 1) / tile_n;
    int32_t k_tiles = (projection_k + tile_k - 1) / tile_k;
    int64_t total_tiles_64 = 0;
    int64_t all_resident_object_bytes = 0;
    int64_t source_object_bound = 0;
    int64_t small_object_bound = 0;
    for (int32_t expert = 0; expert < expert_count; expert++) {
        int32_t rows = tokens_per_expert[expert];
        if (rows < 0) fail_loud("tokens_per_expert cannot be negative");
        if (rows == 0) continue;
        int32_t m_tiles = (rows + tile_m - 1) / tile_m;
        total_tiles_64 += (int64_t)m_tiles * n_tiles;
        all_resident_object_bytes += (
            (int64_t)rows * projection_k
            + (int64_t)projection_n * projection_k
            + (int64_t)rows * projection_n) * element_bytes;
        source_object_bound += (
            (int64_t)(m_tiles + n_tiles) * k_tiles
            + (int64_t)m_tiles * n_tiles);
        /* All C objects and the one M/N tail per expert are conservatively
         * treated as arbitrarily small.  Every remaining A/B object has at
         * least one complete M or N tile, so it is bounded below separately
         * when sizing the slot pool below. */
        small_object_bound += (int64_t)m_tiles * n_tiles
            + (int64_t)2 * k_tiles;
        if (projection_k % tile_k) {
            small_object_bound += m_tiles + n_tiles;
        }
    }
    if (total_tiles_64 <= 0 || total_tiles_64 > INT32_MAX) {
        fail_loud("CANN GroupedMatmul output-tile count is invalid");
    }
    int32_t total_tiles = (int32_t)total_tiles_64;
    CannGroupedTile *schedule = (CannGroupedTile *)checked_malloc(
        (size_t)total_tiles * sizeof(CannGroupedTile));
    int32_t schedule_index = 0;
    int32_t ring_offset = 0;
    for (int32_t expert = 0; expert < expert_count; expert++) {
        int32_t rows = tokens_per_expert[expert];
        if (rows == 0) continue;
        int32_t m_tiles = (rows + tile_m - 1) / tile_m;
        int32_t expert_tiles = m_tiles * n_tiles;
        for (int32_t relative = 0; relative < expert_tiles; relative++) {
            int32_t m_index, n_index;
            cann_mn_block_index(
                relative, m_tiles, n_tiles, &m_index, &n_index);
            if (m_index < 0 || m_index >= m_tiles
                    || n_index < 0 || n_index >= n_tiles) {
                fail_loud("CANN GroupedMatmul produced an invalid tile");
            }
            CannGroupedTile *tile = &schedule[schedule_index++];
            tile->expert = expert;
            tile->m_index = m_index;
            tile->n_index = n_index;
            int32_t remaining_m = rows - m_index * tile_m;
            int32_t remaining_n = projection_n - n_index * tile_n;
            tile->current_m = remaining_m < tile_m ? remaining_m : tile_m;
            tile->current_n = remaining_n < tile_n ? remaining_n : tile_n;
            tile->worker = (ring_offset + relative) % worker_count;
        }
        ring_offset = (ring_offset + expert_tiles) % worker_count;
    }
    if (schedule_index != total_tiles) {
        fail_loud("CANN GroupedMatmul tile schedule is incomplete");
    }

    /* In addition to the legacy eight residency counters per AIC, publish a
     * compact M/N shape histogram.  N has at most one tail shape, so two
     * N classes times tile_m M classes are sufficient; this avoids emitting
     * the CTA schedule to Python solely to rebuild a per-AIC histogram. */
    int32_t tile_class_stride = 2 * tile_m;
    int32_t legacy_values = worker_count * 8;
    int32_t class_values = worker_count * tile_class_stride;
    int32_t total_offset = legacy_values + class_values;
    memset(out, 0, (size_t)(total_offset + 2) * sizeof(int64_t));
    int64_t *l2_read_bytes = &out[total_offset];
    int64_t *dram_read_bytes = &out[total_offset + 1];

    /* When the complete source object set fits in shared L2, LRU has an
     * exact closed form: an A or B descriptor misses exactly at its first
     * output-tile use and every later use hits.  C objects still participate
     * in the proof of residency, but need not be replayed because their
     * misses are not reported as operand traffic.  This replaces the O(tile
     * count * K-tile count) cache replay with one source-order tile census. */
    if (!b_l2_cache_disabled && all_resident_object_bytes <= l2_capacity_bytes) {
        int32_t *m_offsets = (int32_t *)checked_malloc(
            (size_t)expert_count * sizeof(int32_t));
        int32_t total_m_tiles = 0;
        for (int32_t expert = 0; expert < expert_count; expert++) {
            m_offsets[expert] = total_m_tiles;
            int32_t rows = tokens_per_expert[expert];
            if (rows > 0) {
                total_m_tiles += (rows + tile_m - 1) / tile_m;
            }
        }
        uint8_t *a_seen = (uint8_t *)checked_calloc(
            (size_t)total_m_tiles, sizeof(uint8_t));
        uint8_t *b_seen = (uint8_t *)checked_calloc(
            (size_t)expert_count * n_tiles, sizeof(uint8_t));
        int32_t full_k_tiles = projection_k / tile_k;
        for (int32_t index = 0; index < total_tiles; index++) {
            CannGroupedTile *tile = &schedule[index];
            int32_t a_seen_index = (
                m_offsets[tile->expert] + tile->m_index);
            int32_t b_seen_index = tile->expert * n_tiles + tile->n_index;
            int a_miss = a_seen[a_seen_index] == 0;
            int b_miss = b_seen[b_seen_index] == 0;
            a_seen[a_seen_index] = 1;
            b_seen[b_seen_index] = 1;
            int64_t a_total_bytes = (
                (int64_t)tile->current_m * projection_k * element_bytes);
            int64_t b_total_bytes = (
                (int64_t)tile->current_n * projection_k * element_bytes);
            *l2_read_bytes += a_total_bytes + b_total_bytes;
            if (a_miss) *dram_read_bytes += a_total_bytes;
            if (b_miss) *dram_read_bytes += b_total_bytes;
            int64_t *worker = &out[tile->worker * 8];
            if (tile->current_m == tile_m) {
                worker[a_miss ? 1 : 0] += full_k_tiles;
            }
            if (tile->current_n == tile_n) {
                worker[b_miss ? 3 : 2] += full_k_tiles;
            }
            if (tile->current_m == tile_m && tile->current_n == tile_n) {
                worker[4 + (a_miss ? 2 : 0) + (b_miss ? 1 : 0)] += (
                    full_k_tiles);
            }
            int32_t n_class = tile->current_n == tile_n ? 0 : 1;
            int32_t class_index = (
                legacy_values + tile->worker * tile_class_stride
                + n_class * tile_m + tile->current_m - 1);
            out[class_index]++;
        }
        free(b_seen);
        free(a_seen);
        free(m_offsets);
        free(schedule);
        return;
    }

    int32_t min_object_bytes = tile_k * element_bytes;
    int32_t capacity_slots = compute_max_slots(
        min_object_bytes, min_object_bytes, min_object_bytes,
        l2_capacity_bytes);
    /* No cache invocation can retain more objects than this source stream
     * exposes.  Using the capacity-only bound can allocate millions of empty
     * slots for small grouped launches; constraining it by the exact stream
     * cardinality preserves LRU semantics and avoids that expansion. */
    int32_t full_object_min_bytes = (
        tile_m * tile_k < tile_n * tile_k
        ? tile_m * tile_k : tile_n * tile_k) * element_bytes;
    int64_t mixed_size_bound = small_object_bound
        + l2_capacity_bytes / full_object_min_bytes;
    if (mixed_size_bound < source_object_bound) {
        source_object_bound = mixed_size_bound;
    }
    int32_t max_slots = source_object_bound < capacity_slots
        ? (int32_t)source_object_bound : capacity_slots;
    if (max_slots <= 0) fail_loud("CANN GroupedMatmul has no cache objects");
    MissCore core;
    miss_core_init_sized(
        &core, l2_capacity_bytes, profile_occupancy, profile_fraction,
        profile_points,
        (int64_t)total_tiles * (2 * (int64_t)k_tiles + 1), max_slots);
    int32_t wave_count = (total_tiles + worker_count - 1) / worker_count;
    for (int32_t wave = 0; wave < wave_count; wave++) {
        int32_t wave_start = wave * worker_count;
        int32_t wave_end = wave_start + worker_count;
        if (wave_end > total_tiles) wave_end = total_tiles;
        for (int32_t k_index = 0; k_index < k_tiles; k_index++) {
            int32_t remaining_k = projection_k - k_index * tile_k;
            int32_t current_k =
                remaining_k < tile_k ? remaining_k : tile_k;
            for (int32_t index = wave_start; index < wave_end; index++) {
                CannGroupedTile *tile = &schedule[index];
                int32_t a_bytes =
                    tile->current_m * current_k * element_bytes;
                int32_t b_bytes =
                    tile->current_n * current_k * element_bytes;
                uint64_t a_key =
                    ((uint64_t)tile->expert << 40)
                    | ((uint64_t)tile->m_index << 20)
                    | (uint64_t)k_index;
                uint64_t b_key =
                    (1ULL << 63)
                    | ((uint64_t)tile->expert << 40)
                    | ((uint64_t)tile->n_index << 20)
                    | (uint64_t)k_index;
                int a_miss =
                    miss_core_touch(&core, a_key, a_bytes) > 0.0;
                int b_miss = b_l2_cache_disabled
                    ? 1 : miss_core_touch(&core, b_key, b_bytes) > 0.0;
                *l2_read_bytes += (int64_t)a_bytes + b_bytes;
                if (a_miss) *dram_read_bytes += a_bytes;
                if (b_miss) *dram_read_bytes += b_bytes;
                int64_t *worker = &out[tile->worker * 8];
                if (tile->current_m == tile_m && current_k == tile_k) {
                    worker[a_miss ? 1 : 0]++;
                }
                if (tile->current_n == tile_n && current_k == tile_k) {
                    worker[b_miss ? 3 : 2]++;
                }
                if (tile->current_m == tile_m
                        && tile->current_n == tile_n
                        && current_k == tile_k) {
                    worker[4 + (a_miss ? 2 : 0) + (b_miss ? 1 : 0)]++;
                }
            }
        }
        for (int32_t index = wave_start; index < wave_end; index++) {
            CannGroupedTile *tile = &schedule[index];
            int32_t n_class = tile->current_n == tile_n ? 0 : 1;
            int32_t class_index = (
                legacy_values + tile->worker * tile_class_stride
                + n_class * tile_m + tile->current_m - 1);
            out[class_index]++;
            int32_t c_bytes =
                tile->current_m * tile->current_n * element_bytes;
            uint64_t c_key =
                (1ULL << 62)
                | ((uint64_t)tile->expert << 40)
                | ((uint64_t)tile->m_index << 20)
                | (uint64_t)tile->n_index;
            (void)miss_core_touch(&core, c_key, c_bytes);
        }
    }
    miss_core_free(&core);
    free(schedule);
}

/* ─── Byte-stack-distance hit-curve for L2-size differentiability ───
 *
 * Mattson's stack-distance theorem: under LRU, a reference hits in a cache of
 * capacity C iff the *stack distance* of that reference (the amount of
 * distinct data referenced since its previous access to the same key) is < C.
 * Measuring stack distance ONCE per access yields the exact hit count for
 * EVERY capacity C simultaneously — no per-C re-simulation, no fitted model.
 *
 * Here stack distance is measured in BYTES (sum of bytes of the distinct tiles
 * more-recently-used than the key's previous reference), because A/B/C tiles
 * have different byte sizes and the L2 capacity budget is a byte budget. This
 * makes capacity-in-bytes map directly onto stack-distance-in-bytes.
 *
 * Implementation: a Fenwick (binary-indexed) tree over per-reference timeline
 * slots holds each live tile's byte size at the slot of its most recent
 * reference. Stack distance for a repeat reference = (sum of all live bytes) -
 * (prefix sum of bytes at slots <= the key's previous slot). The previous
 * slot's contribution is then zeroed and the tile re-inserted at the current
 * slot. First references have infinite stack distance (compulsory miss).
 *
 * Output: hit_bytes_out[q] = total bytes that HIT for capacity query_caps[q].
 * The caller forms hit_rate(C) = hit_bytes(C)/l2_total and the analytic
 * d(dram)/dC = -slope of hit_bytes between adjacent query capacities (the
 * piecewise-linear CDF derivative — exact within a bin, no tunable parameter).
 * total_l2_bytes_out receives the denominator (sum of all access bytes).
 *
 * This function replays the SAME access stream as simulate_gemm_lru() so its
 * cumulative hits match that simulator byte-for-byte at any capacity.
 */


/* ─── Dense GEMM access sequence ───
 *
 * One family generator: it turns a dense tile program -- the CTA-to-tile
 * schedule a launch package derives from its own vendor raster facts, the
 * retained tile, and the K partition -- into the ordered object references
 * the shared miss core prices.  Every hardware difference reaches this
 * function as one of those discrete launch facts or as a numeric parameter.
 *
 * Concurrency: the resident CTA window advances in waves of wave_size, and
 * the CTAs inside one wave walk their K loop together.  Each CTA reads one A
 * and one B tile per K step and publishes its C tile when the wave retires.
 * A K-split slice past the first also reads the partial C tile it updates
 * when the launch accumulates in place.
 */
static int32_t gemm_tile_slice_edge(
    int32_t index, int32_t steps, int32_t splits, int32_t uniform
) {
    int64_t edge = uniform
        ? (int64_t)index * ((steps + splits - 1) / splits)
        : (int64_t)index * steps / splits;
    return edge < steps ? (int32_t)edge : steps;
}

void simulate_gemm_tile_program(
    const int32_t *schedule_mnz,
    int32_t n_ctas,
    int32_t m_extent,
    int32_t n_extent,
    int32_t k_extent,
    int32_t tile_m,
    int32_t tile_n,
    int32_t tile_k,
    int32_t a_element_bytes,
    int32_t b_element_bytes,
    int32_t c_element_bytes,
    int32_t split_k,
    int32_t uniform_split_k_tiles,
    int32_t accumulate_in_place,
    int32_t wave_size,
    int64_t capacity_bytes,
    const int64_t *profile_occupancy,
    const double *profile_fraction,
    int32_t profile_points,
    int64_t *out
) {
    if (n_ctas <= 0 || m_extent <= 0 || n_extent <= 0 || k_extent <= 0
            || tile_m <= 0 || tile_n <= 0 || tile_k <= 0
            || a_element_bytes <= 0 || b_element_bytes <= 0
            || c_element_bytes <= 0 || split_k <= 0 || wave_size <= 0
            || (uniform_split_k_tiles != 0 && uniform_split_k_tiles != 1)) {
        fail_loud("GEMM tile program dimensions are invalid");
    }
    int32_t total_k_steps = (k_extent + tile_k - 1) / tile_k;
    if (total_k_steps < split_k) {
        fail_loud("GEMM tile program K split exceeds its K steps");
    }
    if (total_k_steps > (1 << 24) || split_k > (1 << 18)) {
        fail_loud("GEMM tile program exceeds the object key encoding");
    }
    for (int32_t index = 0; index < n_ctas; index++) {
        if (schedule_mnz[index * 3] < 0 || schedule_mnz[index * 3] >= (1 << 22)
                || schedule_mnz[index * 3 + 1] < 0
                || schedule_mnz[index * 3 + 1] >= (1 << 22)) {
            fail_loud("GEMM tile program tile index exceeds the key encoding");
        }
    }
    int32_t n_waves = (n_ctas + wave_size - 1) / wave_size;
    int64_t references = 0;
    for (int32_t index = 0; index < n_ctas; index++) {
        int32_t slice = schedule_mnz[index * 3 + 2];
        if (slice < 0 || slice >= split_k) {
            fail_loud("GEMM tile program slice is out of range");
        }
        int32_t begin = gemm_tile_slice_edge(
            slice, total_k_steps, split_k, uniform_split_k_tiles);
        int32_t end = gemm_tile_slice_edge(
            slice + 1, total_k_steps, split_k, uniform_split_k_tiles);
        references += 2 * (int64_t)(end - begin);
        references += (slice > 0 && accumulate_in_place) ? 2 : 1;
    }
    int64_t m_tile_count = 0, n_tile_count = 0;
    for (int32_t index = 0; index < n_ctas; index++) {
        if (schedule_mnz[index * 3] + 1 > m_tile_count) {
            m_tile_count = schedule_mnz[index * 3] + 1;
        }
        if (schedule_mnz[index * 3 + 1] + 1 > n_tile_count) {
            n_tile_count = schedule_mnz[index * 3 + 1] + 1;
        }
    }
    int64_t object_bound = (m_tile_count + n_tile_count) * total_k_steps
        + (int64_t)n_ctas;
    if (object_bound > 4000000) object_bound = 4000000;
    MissCore core;
    miss_core_init_sized(
        &core, capacity_bytes, profile_occupancy, profile_fraction,
        profile_points, references, (int32_t)object_bound);
    double a_dram = 0.0, b_dram = 0.0, c_dram = 0.0;
    int64_t a_l2 = 0, b_l2 = 0, c_l2 = 0;
    for (int32_t wave = 0; wave < n_waves; wave++) {
        int32_t start = wave * wave_size;
        int32_t end_cta = start + wave_size;
        if (end_cta > n_ctas) end_cta = n_ctas;
        int32_t longest = 0;
        for (int32_t index = start; index < end_cta; index++) {
            int32_t slice = schedule_mnz[index * 3 + 2];
            int32_t begin = gemm_tile_slice_edge(
                slice, total_k_steps, split_k, uniform_split_k_tiles);
            int32_t stop = gemm_tile_slice_edge(
                slice + 1, total_k_steps, split_k, uniform_split_k_tiles);
            if (stop - begin > longest) longest = stop - begin;
        }
        for (int32_t step = 0; step < longest; step++) {
            for (int32_t index = start; index < end_cta; index++) {
                int32_t mi = schedule_mnz[index * 3];
                int32_t ni = schedule_mnz[index * 3 + 1];
                int32_t slice = schedule_mnz[index * 3 + 2];
                int32_t begin = gemm_tile_slice_edge(
                    slice, total_k_steps, split_k, uniform_split_k_tiles);
                int32_t stop = gemm_tile_slice_edge(
                    slice + 1, total_k_steps, split_k, uniform_split_k_tiles);
                if (begin + step >= stop) continue;
                int32_t k_index = begin + step;
                int32_t logical_m = m_extent - mi * tile_m;
                if (logical_m > tile_m) logical_m = tile_m;
                int32_t logical_n = n_extent - ni * tile_n;
                if (logical_n > tile_n) logical_n = tile_n;
                int32_t logical_k = k_extent - k_index * tile_k;
                if (logical_k > tile_k) logical_k = tile_k;
                if (logical_m <= 0 || logical_n <= 0 || logical_k <= 0) {
                    fail_loud("GEMM tile program addresses an empty tile");
                }
                int32_t a_bytes = logical_m * logical_k * a_element_bytes;
                int32_t b_bytes = logical_k * logical_n * b_element_bytes;
                uint64_t a_key = ((uint64_t)mi << 24) | (uint64_t)k_index;
                uint64_t b_key = (1ULL << 62)
                    | ((uint64_t)ni << 24) | (uint64_t)k_index;
                a_l2 += a_bytes;
                b_l2 += b_bytes;
                a_dram += miss_core_touch(&core, a_key, a_bytes);
                b_dram += miss_core_touch(&core, b_key, b_bytes);
            }
        }
        for (int32_t index = start; index < end_cta; index++) {
            int32_t mi = schedule_mnz[index * 3];
            int32_t ni = schedule_mnz[index * 3 + 1];
            int32_t slice = schedule_mnz[index * 3 + 2];
            int32_t logical_m = m_extent - mi * tile_m;
            if (logical_m > tile_m) logical_m = tile_m;
            int32_t logical_n = n_extent - ni * tile_n;
            if (logical_n > tile_n) logical_n = tile_n;
            int32_t c_bytes = logical_m * logical_n * c_element_bytes;
            /* A K-split slice that accumulates in place updates the shared
             * output tile; an external reducer gives every slice its own
             * partial buffer, so the slice index joins the object identity. */
            uint64_t c_key = (2ULL << 62) | ((uint64_t)mi << 22)
                | (uint64_t)ni;
            if (!accumulate_in_place) {
                c_key |= (uint64_t)slice << 44;
            }
            if (slice > 0 && accumulate_in_place) {
                c_l2 += c_bytes;
                c_dram += miss_core_touch(&core, c_key, c_bytes);
            }
            c_l2 += c_bytes;
            c_dram += miss_core_touch(&core, c_key, c_bytes);
        }
    }
    miss_core_free(&core);
    out[0] = a_l2;
    out[1] = (int64_t)(a_dram + 0.5);
    out[2] = b_l2;
    out[3] = (int64_t)(b_dram + 0.5);
    out[4] = c_l2;
    out[5] = (int64_t)(c_dram + 0.5);
}

/* ─── Grouped MoE access sequence ───
 *
 * The second family generator over the shared miss core.  A fused grouped
 * GEMM launches one program per padded M tile and N tile, and its source
 * assigns programs to tiles through a group-of-M swizzle.  Each program walks
 * K loading the routed activation rows its M tile owns and the expert weight
 * tile its N coordinate owns, then publishes the valid rows of its output
 * tile.  Padding rows never reach memory, so a tile's activation and output
 * objects carry only the rows the routing histogram populated.
 */
void simulate_moe_tile_program(
    int32_t m_tiles,
    int32_t n_tiles,
    int32_t group_size_m,
    int32_t k_extent,
    int32_t block_k,
    int32_t n_extent,
    int32_t block_n,
    int32_t element_bytes,
    int32_t c_row_bytes,
    int32_t block_m,
    int32_t wave_size,
    const int32_t *tokens_per_expert,
    int32_t expert_count,
    int64_t capacity_bytes,
    const int64_t *profile_occupancy,
    const double *profile_fraction,
    int32_t profile_points,
    int64_t *out
) {
    if (m_tiles <= 0 || n_tiles <= 0 || group_size_m <= 0 || k_extent <= 0
            || block_k <= 0 || n_extent <= 0 || block_n <= 0
            || element_bytes <= 0
            || c_row_bytes <= 0 || block_m <= 0 || wave_size <= 0
            || expert_count <= 0) {
        fail_loud("MoE tile program dimensions are invalid");
    }
    int32_t k_steps = (k_extent + block_k - 1) / block_k;
    int32_t *expert_of_tile = (int32_t *)checked_calloc(
        (size_t)m_tiles, sizeof(int32_t));
    int32_t *rows_of_tile = (int32_t *)checked_calloc(
        (size_t)m_tiles, sizeof(int32_t));
    int32_t populated = 0;
    for (int32_t expert = 0; expert < expert_count; expert++) {
        int32_t count = tokens_per_expert[expert];
        if (count < 0) fail_loud("MoE routing count cannot be negative");
        if (count == 0) continue;
        int32_t tiles = (count + block_m - 1) / block_m;
        if (populated + tiles > m_tiles) {
            fail_loud("MoE populated M tiles exceed the launch geometry");
        }
        for (int32_t index = 0; index < tiles; index++) {
            int32_t remaining = count - index * block_m;
            expert_of_tile[populated] = expert;
            rows_of_tile[populated] =
                remaining < block_m ? remaining : block_m;
            populated++;
        }
    }
    if (populated <= 0) fail_loud("MoE routing has no populated M tile");

    int32_t total = m_tiles * n_tiles;
    int32_t *schedule = (int32_t *)checked_malloc(
        (size_t)total * 2 * sizeof(int32_t));
    int32_t scheduled = 0;
    int32_t group_programs = group_size_m * n_tiles;
    for (int32_t program = 0; program < total; program++) {
        int32_t group = program / group_programs;
        int32_t first_m = group * group_size_m;
        int32_t width = m_tiles - first_m;
        if (width > group_size_m) width = group_size_m;
        if (width <= 0) continue;
        int32_t remainder = program % group_programs;
        int32_t tile_m = first_m + remainder % width;
        int32_t tile_n = remainder / width;
        if (tile_m >= m_tiles || tile_n >= n_tiles) continue;
        /* Programs past the populated routing return before any load. */
        if (tile_m >= populated) continue;
        schedule[scheduled * 2] = tile_m;
        schedule[scheduled * 2 + 1] = tile_n;
        scheduled++;
    }
    if (scheduled <= 0) fail_loud("MoE tile program schedules no program");

    int32_t tail_columns = n_extent % block_n;
    int64_t references = (int64_t)scheduled * (2 * (int64_t)k_steps + 1);
    int64_t object_bound =
        ((int64_t)populated + (int64_t)expert_count * n_tiles)
            * (int64_t)k_steps
        + (int64_t)populated * n_tiles;
    if (tail_columns > 0 && n_tiles > 1) {
        references += 2 * (int64_t)populated * k_steps;
        object_bound += (int64_t)expert_count * k_steps;
    }
    if (object_bound > 4000000) object_bound = 4000000;
    MissCore core;
    miss_core_init_sized(
        &core, capacity_bytes, profile_occupancy, profile_fraction,
        profile_points, references, (int32_t)object_bound);
    double a_dram = 0.0, b_dram = 0.0, c_dram = 0.0;
    int64_t a_l2 = 0, b_l2 = 0, c_l2 = 0;
    int32_t waves = (scheduled + wave_size - 1) / wave_size;
    for (int32_t wave = 0; wave < waves; wave++) {
        int32_t start = wave * wave_size;
        int32_t end = start + wave_size;
        if (end > scheduled) end = scheduled;
        for (int32_t step = 0; step < k_steps; step++) {
            for (int32_t index = start; index < end; index++) {
                int32_t tile_m = schedule[index * 2];
                int32_t tile_n = schedule[index * 2 + 1];
                int32_t expert = expert_of_tile[tile_m];
                int32_t logical_k = k_extent - step * block_k;
                if (logical_k > block_k) logical_k = block_k;
                int32_t logical_n = n_extent - tile_n * block_n;
                if (logical_n > block_n) logical_n = block_n;
                int32_t tail_columns = n_extent % block_n;
                int32_t a_bytes =
                    element_bytes * logical_k * rows_of_tile[tile_m];
                int32_t b_bytes = element_bytes * logical_k * logical_n;
                uint64_t a_key =
                    ((uint64_t)tile_m << 24) | (uint64_t)step;
                uint64_t b_key = (1ULL << 62) | ((uint64_t)expert << 44)
                    | ((uint64_t)tile_n << 24) | (uint64_t)step;
                a_l2 += a_bytes;
                a_dram += miss_core_touch(&core, a_key, a_bytes);
                if (tail_columns > 0 && n_tiles > 1 && tile_n == 0) {
                    int32_t prefix_columns = block_n - tail_columns;
                    int32_t prefix_bytes = element_bytes * logical_k * prefix_columns;
                    int32_t suffix_bytes = element_bytes * logical_k * tail_columns;
                    b_l2 += prefix_bytes + suffix_bytes;
                    b_dram += miss_core_touch(&core, b_key, prefix_bytes);
                    b_dram += miss_core_touch(&core, b_key | (1ULL << 61), suffix_bytes);
                } else if (tail_columns > 0 && n_tiles > 1 && tile_n == n_tiles - 1) {
                    int32_t wrapped_columns = block_n - logical_n;
                    int32_t wrapped_bytes = element_bytes * logical_k * wrapped_columns;
                    uint64_t first_key = (1ULL << 62) | ((uint64_t)expert << 44)
                        | (uint64_t)step;
                    b_l2 += b_bytes + wrapped_bytes;
                    b_dram += miss_core_touch(&core, b_key, b_bytes);
                    b_dram += miss_core_touch(&core, first_key, wrapped_bytes);
                } else {
                    b_l2 += b_bytes;
                    b_dram += miss_core_touch(&core, b_key, b_bytes);
                }
            }
        }
        for (int32_t index = start; index < end; index++) {
            int32_t tile_m = schedule[index * 2];
            int32_t tile_n = schedule[index * 2 + 1];
            int32_t logical_n = n_extent - tile_n * block_n;
            if (logical_n > block_n) logical_n = block_n;
            __int128 c_bytes_wide = (__int128)c_row_bytes
                * rows_of_tile[tile_m] * logical_n / block_n;
            if (c_bytes_wide > INT32_MAX) {
                fail_loud("MoE output tile bytes exceed the C ABI range");
            }
            int32_t c_bytes = (int32_t)c_bytes_wide;
            uint64_t c_key = (2ULL << 62)
                | ((uint64_t)tile_m << 24) | (uint64_t)tile_n;
            c_l2 += c_bytes;
            c_dram += miss_core_touch(&core, c_key, c_bytes);
        }
    }
    miss_core_free(&core);
    free(schedule);
    free(rows_of_tile);
    free(expert_of_tile);
    out[0] = a_l2;
    out[1] = (int64_t)(a_dram + 0.5);
    out[2] = b_l2;
    out[3] = (int64_t)(b_dram + 0.5);
    out[4] = c_l2;
    out[5] = (int64_t)(c_dram + 0.5);
}
