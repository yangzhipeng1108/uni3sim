"""Tiered serving-trace comparison with a pre-registered exclusions registry.

This module upgrades the trace-vs-nsys comparison from a single per-category
count check to a three-tier contract (AC-2), used to validate a synthesized
kernel trace (from the plugin/operator-recipe path or the analytic mirror)
against nsys ground truth:

  Tier 0 — scheduler-sidecar correctness: the per-step scheduler sidecar is
           complete, correctly ordered (monotone iter_id, no gaps), schema-valid,
           and reproducible. (Validated by `tier0_sidecar_ok`.)
  Tier 1 — kernel category sequence: the synthesized trace matches nsys on the
           per-category kernel counts over the focus categories (the existing
           "7/7" bar), GEMV+TC coalesced into GEMM.
  Tier 2 — ordered per-launch exactness after pre-registered exclusions: AFTER
           removing only the launches listed in the exclusions registry (each
           counted and attributed to a registry category), the remaining synth
           and nsys launches are compared element-by-element IN ORDER on
           canonical kernel identity + grid + block. Any identity mismatch,
           length delta, or grid/block mismatch at a matched position fails. The
           synthesized side may emit only modeled (FOCUS) or excluded kernels
           (any other synth kernel is a recipe defect); unresolved/unknown nsys
           launches fail closed. The non-derivable autotuner launch VARIANTS
           (cuBLAS GEMM tile selection, flash splitkv num_splits) are registered
           with scope=tier2: their category is still counted by Tier 1, but
           their specific name+grid+block is excluded from this exact gate.

Zero-fitting discipline (AC-2.4): the exclusions registry
(`configs/validation/serving/nsys_exclusions.yaml`) is a run INPUT fixed before
measurement. It is never mutated per result. Every excluded launch is counted
and reported, so an exclusion can never silently hide a real modeling miss.

Both inputs are ordered launch records with explicit ``grid`` and ``block``.
Name-only inputs are rejected because they would silently disable Tier 2 launch
geometry validation.
"""

from collections import Counter
from pathlib import Path

def _tiered_gate_category(name):
    """Classify captured runtime names for the historical fidelity contract."""
    lowered = name.lower()
    if any(token in lowered for token in (
            "nccl", "allreduce", "all_reduce", "alltoall", "all_to_all",
            "cross_device_reduce")):
        return "COLLECTIVE"
    if any(token in lowered for token in ("fused_moe", "grouped_gemm", "moe_gemm")):
        return "FUSEDMOE"
    if "topkgatingsoftmax" in lowered or "moe_router" in lowered:
        return "REDUCE"
    if "moe_align" in lowered or "count_and_sort" in lowered:
        return "GATHER"
    if "reduce_kernel_moe" in lowered:
        return "REDUCE"
    if "flash_fwd" in lowered:
        return "FLASH_ATTN"
    if "rms_norm" in lowered or "fused_add_rms_norm" in lowered:
        return "NORM"
    if "gemvx" in lowered or "gemv2t" in lowered:
        return "GEMV"
    if any(token in lowered for token in ("s16816gemm", "xmma", "cutlass")):
        return "GEMM"
    if "splitk" in lowered:
        return "SPLITK"
    if "rotary" in lowered:
        return "ROTARY"
    if "act_and_mul" in lowered or "silu" in lowered:
        return "ACTIVATION"
    if "reshape_and_cache" in lowered:
        return "KV_CACHE"
    if "elementwise" in lowered or "vectorized" in lowered:
        return "ELEMENTWISE"
    return "OTHER"


FOCUS = {
    "NORM", "GEMM", "FLASH_ATTN", "ROTARY", "KV_CACHE",
    "ACTIVATION", "REDUCE", "GATHER", "FUSEDMOE",
}

_TIERED_GATE_CATEGORIES = FOCUS | {
    "COLLECTIVE", "ELEMENTWISE", "GEMV", "OTHER", "SPLITK",
}


def coalesce_gemm(counts):
    result = Counter()
    gemm = 0
    for category, count in counts.items():
        if category in ("GEMV", "GEMM"):
            gemm += count
        elif category in _TIERED_GATE_CATEGORIES - {"GEMV", "GEMM"}:
            result[category] += count
        else:
            raise ValueError(f"unknown tiered-gate category: {category!r}")
    if gemm or any(category in counts for category in ("GEMV", "GEMM")):
        result["GEMM"] = gemm
    return result

_DEFAULT_REGISTRY = (
    Path(__file__).resolve().parents[3]
    / "configs" / "validation" / "serving" / "nsys_exclusions.yaml"
)


# ---------------------------------------------------------------------------
# Exclusions registry
# ---------------------------------------------------------------------------

class ExclusionRegistry:
    """Pre-registered, name-pattern kernel exclusions (AC-2.4).

    Loaded from a YAML file with `exclusions: [{category, match: [substr...]}]`.
    `classify(name)` returns the excluded category for a kernel name, or None if
    the kernel is in scope for the per-kernel (Tier 2) comparison.
    """

    def __init__(self, entries):
        # entries: list of (category, [lowercased substrings], scope)
        # scope: "all" -> excluded from BOTH Tier 1 counts and the Tier 2 gate
        #        "tier2" -> kept in Tier 1 category counts, excluded only from the
        #                   Tier 2 exact per-launch gate (the launch's category is
        #                   modeled/counted, but its specific name+grid+block is a
        #                   non-derivable autotuner/occupancy variant).
        self._entries = entries

    @classmethod
    def load(cls, path=None):
        path = Path(path) if path is not None else _DEFAULT_REGISTRY
        if not path.exists():
            raise FileNotFoundError(f"exclusions registry not found: {path}")
        import yaml
        with open(path) as f:
            doc = yaml.safe_load(f)
        if not isinstance(doc, dict):
            raise ValueError(f"malformed exclusions registry root: {path}")
        if set(doc) != {"version", "exclusions"}:
            raise ValueError(
                f"malformed exclusions registry fields {sorted(doc)}: {path}")
        if doc["version"] != 1:
            raise ValueError(
                f"unsupported exclusions registry version {doc['version']!r}: {path}")
        if not isinstance(doc["exclusions"], list) or not doc["exclusions"]:
            raise ValueError(f"exclusions must be a non-empty list: {path}")
        entries = []
        categories = set()
        patterns = set()
        for item in doc["exclusions"]:
            if not isinstance(item, dict):
                raise ValueError(f"exclusion entry must be an object: {item!r}")
            expected = {"category", "scope", "rationale", "match"}
            if set(item) != expected:
                raise ValueError(
                    f"exclusion entry schema mismatch: expected {sorted(expected)}, "
                    f"got {sorted(item)}")
            category = item["category"]
            if not isinstance(category, str) or not category:
                raise ValueError(f"invalid exclusion category: {category!r}")
            if category in categories:
                raise ValueError(
                    f"duplicate exclusion category: {category!r}")
            categories.add(category)
            rationale = item["rationale"]
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError(
                    f"exclusion category {category} requires a rationale")
            if not isinstance(item["match"], list) or not all(
                    isinstance(value, str) and value for value in item["match"]):
                raise ValueError(
                    f"exclusion category {category} match must be a non-empty string list")
            match = [m.lower() for m in item["match"]]
            scope = item["scope"]
            if scope not in ("all", "tier2"):
                raise ValueError(
                    f"exclusion category {category} has invalid scope {scope!r} "
                    f"(expected 'all' or 'tier2')")
            if not match:
                raise ValueError(
                    f"exclusion category {category} has an empty match list")
            duplicates = patterns.intersection(match)
            if duplicates:
                raise ValueError(
                    "exclusion match patterns must be globally unique: "
                    f"{sorted(duplicates)}")
            if len(match) != len(set(match)):
                raise ValueError(
                    f"exclusion category {category} contains duplicate patterns")
            patterns.update(match)
            entries.append((category, match, scope))
        return cls(entries)

    def classify(self, name, scope="all"):
        """Return the excluded category for `name`, or None if in scope.

        scope="all": match entries that apply to Tier 1+2 (the broad exclusions:
            framework noise, NCCL decomposition, cuBLAS splitKreduce, flash
            split-reduce). Used by Tier 1 so genuinely off-model launches do not
            count, while modeled-but-autotuner-variant launches still count.
        scope="tier2": match ALL entries (both "all" and "tier2"-scoped). Used by
            the Tier 2 gate, which additionally excludes the non-derivable
            autotuner/occupancy launch variants (cuBLAS GEMM tile, flash splitkv
            num_splits) from the exact per-launch comparison.
        """
        if scope not in ("all", "tier2"):
            raise ValueError(f"unknown exclusion scope: {scope!r}")
        if not isinstance(name, str) or not name:
            raise ValueError(f"kernel name must be a non-empty string: {name!r}")
        nl = name.lower()
        for category, patterns, escope in self._entries:
            if scope == "all" and escope != "all":
                continue
            for sub in patterns:
                if sub in nl:
                    return category
        return None

    @property
    def categories(self):
        return [c for c, _, _ in self._entries]

    @property
    def tier2_categories(self):
        """Categories that are excluded ONLY from the Tier 2 exact gate."""
        return [c for c, _, s in self._entries if s == "tier2"]


# ---------------------------------------------------------------------------
# nsys side adapters
# ---------------------------------------------------------------------------



def nsys_ordered_launches_from_sqlite(sqlite_path, device_id=0):
    """Return nsys kernel launches IN START ORDER for one device (rank).

    Tier 2 needs the ORDERED launch sequence (the `kernels` count-map in
    nsys_data.json is unordered). Reads the demangled name + grid/block in
    `k.start` order from the CUPTI kernel table for the given deviceId (rank 0 by
    default, matching the rank-0 comparison convention). Returns a list of
    {"name", "grid":[x,y,z], "block":[x,y,z]}.
    """
    import sqlite3
    conn = sqlite3.connect(str(sqlite_path))
    try:
        rows = conn.execute(
            """SELECT s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ
               FROM CUPTI_ACTIVITY_KIND_KERNEL k
               JOIN StringIds s ON k.demangledName = s.id
               WHERE k.deviceId = ?
               ORDER BY k.start""",
            (device_id,)).fetchall()
    finally:
        conn.close()
    return [{"name": r[0], "grid": [r[1], r[2], r[3]], "block": [r[4], r[5], r[6]]}
            for r in rows]


# ---------------------------------------------------------------------------
# Tier 0 — scheduler sidecar correctness
# ---------------------------------------------------------------------------

def tier0_sidecar_ok(sidecar_records):
    """Validate scheduler-sidecar structural correctness (AC-2.1).

    Returns (ok, problems). Checks: non-empty, each record has the required
    fields, iter_id is strictly monotone from its first value with no gaps,
    and per-step total_tokens equals the sum of scheduled token counts.
    """
    problems = []
    if not sidecar_records:
        return False, ["sidecar is empty"]
    required = {"iter_id", "scheduled", "total_tokens"}
    prev = None
    for i, rec in enumerate(sidecar_records):
        missing = required - set(rec)
        if missing:
            problems.append(f"record {i}: missing fields {sorted(missing)}")
            continue
        it = rec["iter_id"]
        if prev is not None and it != prev + 1:
            problems.append(
                f"record {i}: iter_id {it} not contiguous after {prev}")
        prev = it
        sched = rec["scheduled"]
        if isinstance(sched, dict):
            sched_sum = sum(sched.values())
            if sched_sum != rec["total_tokens"]:
                problems.append(
                    f"record {i} (iter {it}): scheduled sum {sched_sum} != "
                    f"total_tokens {rec['total_tokens']}")
    return (len(problems) == 0), problems


# ---------------------------------------------------------------------------
# Tier 1 — kernel category sequence
# ---------------------------------------------------------------------------

def tier1_category_match(synth_names, nsys_names, registry):
    """Compare focus-category counts (AC-2.2). GEMV+TC coalesced into GEMM.

    Registry-excluded kernels (cuBLAS splitKreduce, flash split-KV combine, NCCL
    decomposition, framework noise) are removed from BOTH sides before counting,
    exactly as Tier 2 excludes them — otherwise an excluded-but-FOCUS-classified
    kernel (e.g. flash_fwd_splitkv_combine -> FLASH_ATTN) would leak into the
    FOCUS counts and fail Tier 1 even though it is a registered non-derivable
    artifact. The exclusions registry is mandatory so the comparison cannot
    silently change scope between callers.

    Returns (ok, table) where table maps category -> {synth, nsys, diff} for the
    union of categories, and ok is True iff all FOCUS categories match exactly.
    """
    def _kept(names):
        return [n for n in names if registry.classify(n) is None]

    sc = coalesce_gemm(Counter(_tiered_gate_category(n) for n in _kept(synth_names)))
    nc = coalesce_gemm(Counter(_tiered_gate_category(n) for n in _kept(nsys_names)))
    table = {}
    ok = True
    for cat in sorted(set(sc) | set(nc)):
        s, n = sc[cat], nc[cat]
        table[cat] = {"synth": s, "nsys": n, "diff": s - n}
        if cat in FOCUS and s != n:
            ok = False
    return ok, table


# ---------------------------------------------------------------------------
# Tier 2 — per-kernel after pre-registered exclusions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tier 2 — ordered per-launch comparison after pre-registered exclusions
# ---------------------------------------------------------------------------

# Categories that are framework/autotuner noise on the nsys side: not emitted by
# the recipe and not modeled. They are dropped (and counted) before the ordered
# comparison, exactly like registry exclusions, but tracked separately so the
# report distinguishes "registry-excluded" from "non-modeled framework noise".
# UNKNOWN/unclassified kernels are deliberately NOT in this set: they fail closed
# (Codex R0 requirement 1) so a new/unrecognized kernel can never silently vanish.
_NSYS_NOISE_CATEGORIES = {"ELEMENTWISE", "INDEX_SELECT", "SOFTMAX",
                          "CONCAT", "SORT", "MASKED_FILL", "ARANGE", "RANDOM"}


def _canonical_category(name):
    """Canonical category, with GEMV/TC coalesced to GEMM."""
    cat = _tiered_gate_category(name)
    if cat in ("GEMV", "GEMM"):
        return "GEMM"
    elif cat in _TIERED_GATE_CATEGORIES:
        return cat
    else:
        raise AssertionError(
            f"tiered gate classifier returned an unknown category: {cat!r}")


# Canonical per-launch IDENTITY (finer than category): a stable token derived from
# the kernel name that is the same for the recipe's synthetic name and nsys's
# mangled name of the SAME vLLM op. This lets Tier 2 gate on kernel identity +
# order (stronger than category) for the launches whose identity IS derivable
# (vLLM kernels). For cuBLAS/cutlass GEMM the specific tile variant is a
# non-derivable autotuner choice, so all GEMM collapses to a single "GEMM"
# identity (its variant split stays out of the gate, like the registry exclusions).
_IDENTITY_PATTERNS = [
    ("fused_add_rms_norm", "vllm.fused_add_rms_norm"),
    ("rms_norm", "vllm.rms_norm"),
    ("rotary_embedding", "vllm.rotary"),
    ("reshape_and_cache", "vllm.reshape_and_cache"),
    ("act_and_mul", "vllm.act_and_mul"),
    ("silu", "vllm.act_and_mul"),
    ("flash_fwd_splitkv_combine", "flash.splitkv_combine"),
    ("flash_fwd_splitkv", "flash.splitkv"),
    ("flash_fwd", "flash.fwd"),
    ("topkgatingsoftmax", "moe.router"),
    ("moe_align_block_size", "moe.align"),
    ("count_and_sort_expert", "moe.count_and_sort"),
    ("fused_moe", "moe.fused"),
    ("moe::act_and_mul", "moe.act"),
    ("nccldevkernel", "nccl"),
    ("cross_device_reduce", "nccl"),
    ("all2all", "nccl"),
    ("all_to_all", "nccl"),
]


def _canonical_kernel_identity(name):
    """Return the canonical identity token for a kernel name, else its category.

    vLLM kernels map to a stable token (the same token for the recipe's synthetic
    name and nsys's mangled name of the SAME op), so the Tier 2 gate distinguishes
    e.g. rms_norm from rotary even within one category. cuBLAS/cutlass GEMM is a
    Tier-2 registry exclusion (its tile variant is non-derivable), so it does not
    normally reach this function from the kept set; if it does it collapses to
    "GEMM". Everything else falls back to its category.
    """
    nl = name.lower()
    cat = _canonical_category(name)
    if cat == "GEMM":
        return "GEMM"
    for sub, ident in _IDENTITY_PATTERNS:
        if sub in nl:
            return ident
    return cat


def _as_launch_records(items):
    """Validate ordered launch records and normalize their name field."""
    out = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(
                f"launch record {index} must be a mapping with geometry")
        name_fields = {
            field for field in ("kernel_name", "name") if field in item}
        if len(name_fields) != 1:
            raise ValueError(
                "launch record must contain exactly one of kernel_name/name: "
                f"{item!r}")
        name = item[next(iter(name_fields))]
        if not isinstance(name, str) or not name:
            raise ValueError(f"kernel name must be non-empty: {item!r}")
        geometry = {}
        for field in ("grid", "block"):
            if field not in item:
                raise ValueError(f"launch record missing {field!r}: {item!r}")
            value = item[field]
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                raise ValueError(
                    f"launch {field} must contain exactly three dimensions: "
                    f"{item!r}")
            if any(isinstance(part, bool) or not isinstance(part, int)
                   for part in value):
                raise TypeError(
                    f"launch {field} must contain integers: {item!r}")
            if any(part < 0 for part in value):
                raise ValueError(
                    f"launch {field} cannot contain negative values: {item!r}")
            geometry[field] = list(value)
        out.append({
            "name": name,
            "grid": geometry["grid"],
            "block": geometry["block"],
        })
    return out


def tier2_per_kernel_match(synth_launches, nsys_launches, registry):
    """Tier 2 — ORDERED structured per-launch comparison after exclusions (AC-2.3).

    Inputs are ORDERED launch records for synth and nsys.
    The GATE is, for every KEPT launch (after pre-registered exclusions), exact
    equality of canonical kernel IDENTITY + launch order + grid + block. A grid or
    block mismatch at a matched position FAILS, reported as the first structured
    mismatch — this is the original AC-2.3 contract.

    Exclusions are pre-registered and applied with the Tier 2 scope (registry
    `classify(name, scope="tier2")`), which removes from the exact gate both:
      - the broad off-model launches (NCCL decomposition, framework noise, cuBLAS
        splitKreduce / flash split-reduce epilogues — scope "all"), and
      - the non-derivable autotuner/occupancy launch VARIANTS whose category is
        still modeled and counted by Tier 1 (cuBLAS GEMM tile selection, flash
        splitkv num_splits — scope "tier2").
    Every excluded launch is counted and attributed to its registry category, so
    the exclusion is auditable (AC-2.4) and can never silently hide a modeling miss.

    Algorithm:
      1. Partition each side IN ORDER into registry-excluded (counted by category
         & side), nsys framework-noise (counted, dropped), and the kept ordered
         list of launch records tagged with canonical identity.
      2. Fail closed on raw-unresolved (0x/so:) and any synth kernel that is
         neither a modeled identity nor excluded.
      3. Compare kept sequences element-by-element on (identity, grid, block). Any
         length delta, identity mismatch, or grid/block mismatch fails (concise
         diff w/ index + context).

    Known blind spot: a swap of two adjacent identical (identity+grid+block)
    launches is not detectable by equality alone.
    """
    synth_recs = _as_launch_records(synth_launches)
    nsys_recs = _as_launch_records(nsys_launches)

    def partition(recs, side):
        kept = []                 # list of (identity, record)
        excluded = Counter()
        noise = Counter()
        unexplained = Counter()
        for rec in recs:
            name = rec["name"]
            excl = registry.classify(name, scope="tier2")
            if excl is not None:
                excluded[excl] += 1
                continue
            nl = name.strip().lower()
            if nl.startswith("0x") or nl.startswith("kernel_0x") or nl.startswith("so:"):
                unexplained["UNRESOLVED"] += 1
                continue
            cat = _canonical_category(name)
            if cat in FOCUS:
                kept.append((_canonical_kernel_identity(name), rec))
            elif side == "nsys" and cat in _NSYS_NOISE_CATEGORIES:
                noise[cat] += 1
            else:
                unexplained[cat] += 1
        return kept, excluded, noise, unexplained

    synth_kept, synth_excl, _sn, synth_unexpl = partition(synth_recs, "synth")
    nsys_kept, nsys_excl, nsys_noise, nsys_unexpl = partition(nsys_recs, "nsys")

    synth_seq = [k[0] for k in synth_kept]
    nsys_seq = [k[0] for k in nsys_kept]

    # Exact per-launch gate (AC-2.3): identity + grid + block at each position,
    # over the ordered kept sequences. The first identity OR grid/block mismatch
    # fails; a length delta fails.
    diff = None
    gridblock_compared = 0
    gridblock_diffs = 0
    n = min(len(synth_seq), len(nsys_seq))
    for i in range(n):
        si, sr = synth_kept[i]
        ni, nr = nsys_kept[i]
        if si != ni:
            diff = {"index": i, "synth": si, "nsys": ni, "kind": "identity",
                    "synth_context": synth_seq[max(0, i - 2):i + 3],
                    "nsys_context": nsys_seq[max(0, i - 2):i + 3]}
            break
        # identity matches; gate grid/block when both sides carry launch dims.
        if sr["grid"] is not None and nr["grid"] is not None:
            gridblock_compared += 1
            if sr["grid"] != nr["grid"] or sr["block"] != nr["block"]:
                gridblock_diffs += 1
                if diff is None:
                    diff = {"index": i, "synth": si, "nsys": ni,
                            "kind": "grid/block",
                            "synth_grid": sr["grid"], "synth_block": sr["block"],
                            "nsys_grid": nr["grid"], "nsys_block": nr["block"],
                            "synth_context": synth_seq[max(0, i - 2):i + 3],
                            "nsys_context": nsys_seq[max(0, i - 2):i + 3]}
                break
    if diff is None and len(synth_seq) != len(nsys_seq):
        diff = {"index": n, "kind": "length",
                "synth": synth_seq[n] if n < len(synth_seq) else "<end>",
                "nsys": nsys_seq[n] if n < len(nsys_seq) else "<end>",
                "synth_context": synth_seq[max(0, n - 2):n + 3],
                "nsys_context": nsys_seq[max(0, n - 2):n + 3]}

    ok = (diff is None and not synth_unexpl and not nsys_unexpl)
    return {
        "ok": ok,
        "len_synth": len(synth_seq),
        "len_nsys": len(nsys_seq),
        "first_mismatch": diff,
        "excluded_synth": synth_excl,
        "excluded_nsys": nsys_excl,
        "framework_nsys": nsys_noise,
        "unexplained_synth": synth_unexpl,
        "unexplained_nsys": nsys_unexpl,
        "gridblock_compared": gridblock_compared,
        "gridblock_diffs": gridblock_diffs,  # part of `ok`: any diff -> first_mismatch
    }


# ---------------------------------------------------------------------------
# Public-communication presence gate (AC-3, fail-closed)
# ---------------------------------------------------------------------------

# Modeled PUBLIC communication operations the recipe emits as first-class trace
# entries (distinct from the NCCL *internal* device-kernel decomposition that the
# NCCL_DECOMP registry entry excludes on the nsys side). These are matched by the
# synthesized kernel_name and carry their message volume in block[0]+(block[1]<<32)
# (the historical nccl_send/nccl_recv encoding in the archived kv_transfer.py)
# or in all-reduce / all-to-all entries emitted by the archived analytic mirror.
_PUBLIC_COMM_MATCHERS = {
    # requirement-key -> predicate over a lowercased synth kernel name
    "nccl_send": lambda nl: nl == "nccl_send",
    "nccl_recv": lambda nl: nl == "nccl_recv",
    "all_reduce": lambda nl: "allreduce" in nl or "all_reduce" in nl,
    "all_to_all": lambda nl: "alltoall" in nl or "all_to_all" in nl,
}


def _launch_msg_bytes(rec):
    """Decode message bytes from a launch record's block[0]+(block[1]<<32)."""
    block = rec["block"]
    low, high = block[:2]
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in (low, high)):
        raise TypeError(
            "communication launch byte words must be integers")
    if low < 0 or high < 0:
        raise ValueError(
            "communication launch byte words cannot be negative")
    return low + (high << 32)


def public_comm_gate(synth_launches, comm_requirements):
    """Fail-closed presence/count/byte gate for modeled PUBLIC communication ops.

    `comm_requirements` maps each operation to an explicit `min`, `exact`, and
    `require_bytes` policy. `exact` may be None; the other two fields are required.
    For each required op the SYNTH trace must contain at least `min` (and exactly
    `exact`, if given) entries whose kernel_name matches, and — when bytes are
    required — every matched entry must encode nonzero bytes.

    This is independent of the registry exclusions: the nsys side excludes the
    NCCL internal kernel decomposition (NCCL_DECOMP), but a synthesized trace that
    is MISSING its public collective/KV-transfer ops must still FAIL, satisfying
    AC-3's negative tests ("a PD run whose KV-transfer ops are absent fails").

    Returns {"ok": bool, "checks": [{op, found, bytes_ok, min, exact, ok, ...}]}.
    A schema-defined null or empty object is a no-op (ok=True, no checks).
    """
    if comm_requirements is None:
        return {"ok": True, "checks": []}
    if not isinstance(comm_requirements, dict):
        raise TypeError("public-comm requirements must be an object or null")
    if not comm_requirements:
        return {"ok": True, "checks": []}
    recs = _as_launch_records(synth_launches)
    checks = []
    overall = True
    for op, spec in comm_requirements.items():
        if op not in _PUBLIC_COMM_MATCHERS:
            raise ValueError(f"unknown public-comm requirement {op!r}")
        if not isinstance(spec, dict):
            raise ValueError(f"public-comm requirement {op!r} must be an object")
        required = {"min", "exact", "require_bytes"}
        if set(spec) != required:
            raise ValueError(
                f"public-comm requirement {op!r} must contain exactly "
                f"{sorted(required)}, got {sorted(spec)}")
        matcher = _PUBLIC_COMM_MATCHERS[op]
        min_count = spec["min"]
        exact = spec["exact"]
        require_bytes = spec["require_bytes"]
        if (not isinstance(min_count, int) or isinstance(min_count, bool)
                or min_count < 0):
            raise ValueError(f"public-comm {op!r} min must be a non-negative integer")
        if (exact is not None and (not isinstance(exact, int)
                                   or isinstance(exact, bool) or exact < 0)):
            raise ValueError(f"public-comm {op!r} exact must be a non-negative integer")
        if not isinstance(require_bytes, bool):
            raise ValueError(f"public-comm {op!r} require_bytes must be boolean")
        matched = [r for r in recs if matcher(r["name"].strip().lower())]
        found = len(matched)
        nonzero = sum(1 for r in matched if _launch_msg_bytes(r) > 0)
        bytes_ok = (not require_bytes) or (found > 0 and nonzero == found)
        count_ok = (found >= min_count) and (exact is None or found == exact)
        ok = bool(count_ok and bytes_ok)
        checks.append({"op": op, "found": found, "nonzero_bytes": nonzero,
                       "min": min_count, "exact": exact,
                       "require_bytes": require_bytes, "bytes_ok": bytes_ok,
                       "count_ok": count_ok, "ok": ok})
        overall = overall and ok
    return {"ok": overall, "checks": checks}


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def compare(synth_trace, nsys_launches, sidecar_records=None, registry=None,
            comm_requirements=None):
    """Run the full tiered comparison.

    Args:
        synth_trace: synthesized trace entries (dicts with kernel_name/grid/block),
            in emission order. Used structured for Tier 2.
        nsys_launches: ordered nsys launch records with name/grid/block.
        sidecar_records: optional scheduler-sidecar records for Tier 0.
        registry: ExclusionRegistry; loaded from default path if None.
        comm_requirements: optional dict of required PUBLIC communication ops
            (see `public_comm_gate`). When given, the synthesized trace must
            contain those ops (with nonzero bytes for send/recv); a missing public
            op FAILS overall even though the nsys NCCL internals are registry-
            excluded. This closes AC-3's communication-presence negative test.

    Returns a dict with per-tier results and an overall `passed` flag (Tier 1 +
    Tier 2 + the comm gate when requested; Tier 0 included when sidecar_records is
    provided).
    """
    if registry is None:
        registry = ExclusionRegistry.load()
    normalized_synth = _as_launch_records(synth_trace)
    normalized_nsys = _as_launch_records(nsys_launches)
    synth_names = [record["name"] for record in normalized_synth]
    nsys_names = [record["name"] for record in normalized_nsys]

    result = {}
    if sidecar_records is not None:
        t0_ok, t0_problems = tier0_sidecar_ok(sidecar_records)
        result["tier0"] = {"ok": t0_ok, "problems": t0_problems}
    t1_ok, t1_table = tier1_category_match(synth_names, nsys_names, registry)
    result["tier1"] = {"ok": t1_ok, "table": t1_table}
    t2 = tier2_per_kernel_match(normalized_synth, normalized_nsys, registry)
    result["tier2"] = t2

    passed = t1_ok and t2["ok"]
    if "tier0" in result:
        passed = passed and result["tier0"]["ok"]
    if comm_requirements:
        cg = public_comm_gate(synth_trace, comm_requirements)
        result["comm_gate"] = cg
        passed = passed and cg["ok"]
    result["passed"] = passed
    return result


def format_report(result):
    """Render a human-readable report of a `compare()` result."""
    lines = []
    if "tier0" in result:
        t0 = result["tier0"]
        lines.append(f"Tier 0 (sidecar correctness): {'PASS' if t0['ok'] else 'FAIL'}")
        for p in t0["problems"]:
            lines.append(f"    - {p}")
    t1 = result["tier1"]
    lines.append(f"Tier 1 (category sequence):   {'PASS' if t1['ok'] else 'FAIL'}")
    lines.append(f"    {'category':14}{'synth':>7}{'nsys':>7}{'diff':>7}  scope")
    for cat in sorted(t1["table"]):
        row = t1["table"][cat]
        scope = "FOCUS" if cat in FOCUS else "excluded"
        flag = " MISMATCH" if (cat in FOCUS and row["diff"] != 0) else ""
        lines.append(f"    {cat:14}{row['synth']:>7}{row['nsys']:>7}"
                     f"{row['diff']:>+7}  {scope}{flag}")
    t2 = result["tier2"]
    lines.append(f"Tier 2 (ordered per-launch identity after exclusions): {'PASS' if t2['ok'] else 'FAIL'}")
    lines.append(f"    kept ordered launches: synth={t2['len_synth']} nsys={t2['len_nsys']}")
    lines.append("    excluded launches (counted, not matched):")
    for cat in sorted(set(t2["excluded_synth"]) | set(t2["excluded_nsys"])):
        lines.append(f"      {cat:18} synth={t2['excluded_synth'][cat]:>6}"
                     f"  nsys={t2['excluded_nsys'][cat]:>6}")
    if t2["first_mismatch"]:
        d = t2["first_mismatch"]
        kind = d["kind"]
        lines.append(f"    FIRST MISMATCH ({kind}) at index {d['index']}: "
                     f"synth={d['synth']} vs nsys={d['nsys']}")
        if kind == "grid/block":
            lines.append(f"      synth grid={d['synth_grid']} block={d['synth_block']}")
            lines.append(f"      nsys  grid={d['nsys_grid']} block={d['nsys_block']}")
        lines.append(f"      synth context: {d['synth_context']}")
        lines.append(f"      nsys  context: {d['nsys_context']}")
    # grid/block is GATED for kept launches (AC-2.3 exact gate); any diff above
    # appears as the first mismatch. cuBLAS GEMM tile / flash num_splits launch
    # variants are pre-registered (scope=tier2) exclusions, counted above.
    lines.append(f"    grid/block exact-gate: "
                 f"{t2['gridblock_diffs']}/{t2['gridblock_compared']} "
                 f"matched-position launches differ (0 required to pass)")
    if t2["framework_nsys"]:
        lines.append("    framework/autotuner noise on nsys side (counted, dropped before order compare):")
        for cat, c in t2["framework_nsys"].most_common(20):
            lines.append(f"      {c:>6}  {cat}")
    if t2["unexplained_synth"]:
        lines.append("    UNEXPLAINED synth launches (recipe defect — must be empty):")
        for cat, c in t2["unexplained_synth"].most_common(20):
            lines.append(f"      {c:>6}  {cat}")
    if t2["unexplained_nsys"]:
        lines.append("    UNKNOWN nsys launches (fail-closed — not classified, not excluded):")
        for cat, c in t2["unexplained_nsys"].most_common(20):
            lines.append(f"      {c:>6}  {cat}")
    if "comm_gate" in result:
        cg = result["comm_gate"]
        lines.append(f"Public-communication gate: {'PASS' if cg['ok'] else 'FAIL'}")
        for chk in cg["checks"]:
            if "error" in chk:
                lines.append(f"    {chk['op']}: ERROR {chk['error']}")
                continue
            exact = f" exact={chk['exact']}" if chk["exact"] is not None else ""
            lines.append(
                f"    {chk['op']}: found={chk['found']} (min={chk['min']}{exact}), "
                f"nonzero_bytes={chk['nonzero_bytes']} "
                f"require_bytes={chk['require_bytes']} -> "
                f"{'OK' if chk['ok'] else 'FAIL'}")
    lines.append(f"OVERALL: {'PASS' if result['passed'] else 'FAIL'}")
    return "\n".join(lines)
