#!/usr/bin/env python3
"""Freeze and audit the C500 grouped-gather cache experiment."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


SCHEMA_VERSION = 5
STDOUT_SCHEMA_VERSION = 5
HARDWARE_ID = "metax_c500"
MECHANISM = "grouped_gather_l2_residency_and_downstream_fill"
BASELINE_COMMIT = "8778cc80b2e589063180deb5f18e74ba850a235c"
PRODUCTION_SOURCE_RELATIVE_PATHS = (
    "scripts/kernel_dataset/metax_c500/fused_moe_runner.py",
    "dataset_pipeline/kernel_dataset/metax_c500_fused_moe_kernels.py",
)
PRODUCTION_SOURCE_SHA256S = {
    "scripts/kernel_dataset/metax_c500/fused_moe_runner.py": (
        "436bee88cb5fe444ec0aa5204e7df158edb51119891424ed15237e9dc9c13474"),
    "dataset_pipeline/kernel_dataset/metax_c500_fused_moe_kernels.py": (
        "40e0cd0e9439a6e060cb73ebe36f344ae25d0ba51efddd1ad38d154ece5f064b"),
}
PRODUCTION_SOURCE_BUNDLE_SHA256 = (
    "91b2278fb593e47c8b71965e1b4564969b484326f8ec77559721263fd4eda2a3")
RUNNER_RELATIVE_PATH = (
    "microbench/metax_c500/operators/fused_moe/"
    "fused_moe_gather_cache_runner.py")
L2_CAPACITY_BYTES = 8 * 1024 * 1024
BLOCK_THREADS = 256
WAVE_SIZE = 64
TIMED_ITERATIONS = 3
ATTACH_WAIT_SECONDS = 5
MAX_PROFILE_ATTEMPTS_PER_SAMPLE = 3
RUN_ORDER_SEED = 20260823
EVENT_CACHE_STATE = "steady_state"
FULL_TRIALS = (1, 2, 3)
SMOKE_TRIALS = (1,)
PRODUCTION_SCALE_HOLDOUT_TRIALS = (1, 2, 3)
MIXED_MMA_TRIALS = (1, 2, 3)
EXPERT_ROUTE_MMA_TRIALS = (1, 2, 3)
COUNTER_RELATIVE_SPAN_LIMIT = 0.05
EVENT_RELATIVE_SPAN_LIMIT = 0.05
HOLDOUT_MEAN_APE_LIMIT_PERCENT = 20.0
HOLDOUT_MAX_APE_LIMIT_PERCENT = 40.0
DYNAMIC_SHARED_BYTES = (16 * 1024, 32 * 1024, 64 * 1024)
SIGNATURES = {
    "bm16_bk64_bn32_gsm1": {
        "bm": 16, "bk": 64, "bn": 32, "group_size_m": 1,
        "operand_replay_a": 1, "operand_replay_b": 1,
        "polluter_load_multiplicity": 1,
    },
    "bm64_bk32_bn64_gsm8": {
        "bm": 64, "bk": 32, "bn": 64, "group_size_m": 8,
        "operand_replay_a": 2, "operand_replay_b": 2,
        "polluter_load_multiplicity": 2,
    },
}
HOLDOUT_LABELS = {
    "bm64_k7168_rows512_prewarm_nob",
    "bm64_k7168_rows4096_prewarm_nob",
    "bm64_k7168_fanout8_prewarm_b",
    "bm64_k7168_fanout64_prewarm_b",
    "bm64_k7168_fanout8_prewarm_nob",
    "bm64_k7168_smem65536_prewarm_b",
    "bm64_k7168_weightshare2",
    "bm64_k2048_rows4096_prewarm_b",
    "bm16_k7168_rows512_prewarm_nob",
    "bm16_k7168_fanout32_prewarm_b",
    "bm16_k7168_fanout32_prewarm_nob",
    "bm16_k7168_smem65536_prewarm_b",
}
PRODUCTION_SCALE_HOLDOUT_LABELS = {
    "bm64_w1_topk8_production_scale_b",
    "bm64_w2_routes_production_scale_b",
}
STDOUT_FIELDS = (
    "signature", "bm", "bk", "bn", "group_size_m", "k",
    "source_rows", "group_count", "n_fanout", "weight_share_groups",
    "grid", "block", "dynamic_shared_bytes", "include_polluter",
    "polluter_load_multiplicity", "include_mma", "route_pattern",
    "timed_iterations", "event_cache_state",
    "profile_setup", "logical_a_read_bytes", "logical_b_base_read_bytes",
    "logical_b_read_bytes", "unique_a_bytes", "unique_b_bytes",
    "index_read_bytes", "sink_write_bytes", "event_ms_per_iteration",
    "first_sink", "last_sink",
)
INTEGER_STDOUT_FIELDS = (
    "bm", "bk", "bn", "group_size_m", "k", "source_rows",
    "group_count", "n_fanout", "weight_share_groups", "grid", "block",
    "dynamic_shared_bytes", "include_polluter",
    "polluter_load_multiplicity", "include_mma", "timed_iterations",
    "logical_a_read_bytes", "logical_b_base_read_bytes",
    "logical_b_read_bytes", "unique_a_bytes", "unique_b_bytes",
    "index_read_bytes", "sink_write_bytes",
)
PROVENANCE_KEYS = {
    "schema_version", "hardware_id", "mechanism", "baseline_commit",
    "software_revision", "container_image", "physical_device_id",
    "production_or_evaluation_duration_consumed", "plan_path",
    "plan_sha256", "source_path", "source_sha256", "binary_path",
    "binary_sha256", "device_preflight_path", "device_preflight_sha256",
    "production_source_files", "production_source_bundle_sha256",
    "runner_path", "runner_sha256", "raw_stdout_policy",
}
PRODUCTION_SOURCE_RECORD_KEYS = {
    "repository_path", "snapshot_path", "sha256",
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _production_source_bundle_sha256(source_sha256s):
    if set(source_sha256s) != set(PRODUCTION_SOURCE_RELATIVE_PATHS):
        raise ValueError("production FusedMoE source bundle is incomplete")
    digest = hashlib.sha256(
        b"midlayer-c500-fused-moe-production-source-v1\0")
    for relative_path in PRODUCTION_SOURCE_RELATIVE_PATHS:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(source_sha256s[relative_path]))
    return digest.hexdigest()


def _positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"{label} must be a positive integer")
    return value


def _parse_integer(text, label, *, positive=False):
    if not text or not text.isascii() or not text.isdigit():
        raise ValueError(f"{label} must be a canonical non-negative integer")
    if len(text) > 1 and text.startswith("0"):
        raise ValueError(f"{label} must not contain leading zeroes")
    value = int(text)
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _parse_hex_integer(text, label):
    if (len(text) != 8 or not text.isascii()
            or any(character not in "0123456789abcdef" for character in text)):
        raise ValueError(f"{label} must be eight lowercase hexadecimal digits")
    return int(text, 16)


def _positive_float(text, label):
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def _mcprofiler_number(value, label):
    if isinstance(value, bool):
        raise TypeError(f"{label} cannot be boolean")
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        text = value.strip().replace(",", "")
        for suffix in ("%", "byte", "bytes"):
            if text.endswith(suffix):
                text = text[:-len(suffix)]
        text = text.split("(", 1)[0].strip()
        try:
            result = float(text)
        except ValueError as error:
            raise ValueError(f"{label} is not numeric: {value!r}") from error
    else:
        raise TypeError(f"{label} has an invalid numeric type")
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _signature(name):
    if name not in SIGNATURES:
        raise ValueError(f"unknown gather-cache signature: {name!r}")
    return SIGNATURES[name]


def audit_production_source():
    repository_root = Path(__file__).resolve().parents[4]
    paths = []
    source_sha256s = {}
    sources = []
    for relative_path in PRODUCTION_SOURCE_RELATIVE_PATHS:
        path = repository_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"production FusedMoE source is missing: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != PRODUCTION_SOURCE_SHA256S[relative_path]:
            raise ValueError(
                "production FusedMoE source differs from the frozen "
                f"baseline: {relative_path}")
        paths.append(path)
        source_sha256s[relative_path] = actual_sha256
        sources.append(path.read_text(encoding="utf-8"))
    if (_production_source_bundle_sha256(source_sha256s)
            != PRODUCTION_SOURCE_BUNDLE_SHA256):
        raise ValueError(
            "production FusedMoE source bundle differs from the frozen "
            "baseline")
    source = "\n".join(sources)
    required_fragments = (
        "num_pid_in_group = GROUP_SIZE_M * num_pid_n",
        "pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)",
        "pid_n = (pid % num_pid_in_group) // group_size_m",
        "offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)",
        "offs_token[:, None] // top_k * stride_am",
        "off_experts * stride_be",
        "pid_n * BLOCK_SIZE_N",
    )
    if any(fragment not in source for fragment in required_fragments):
        raise ValueError(
            "production FusedMoE address/order contract changed")
    return tuple(paths)


def _round_up(value, multiple):
    return math.ceil(value / multiple) * multiple


def _groups_covering_source(signature_name, source_rows, share_groups):
    signature = _signature(signature_name)
    rows_per_group = signature["bm"] * signature["group_size_m"]
    return max(share_groups, _round_up(
        math.ceil(source_rows / rows_per_group), share_groups))


def _derived(config):
    signature = _signature(config["signature"])
    bm = signature["bm"]
    bn = signature["bn"]
    group_size_m = signature["group_size_m"]
    k = config["k"]
    source_rows = config["source_rows"]
    group_count = config["group_count"]
    n_fanout = config["n_fanout"]
    share_groups = config["weight_share_groups"]
    include_polluter = config["include_polluter"]
    replay_a = signature["operand_replay_a"]
    replay_b = signature["operand_replay_b"]
    polluter_load_multiplicity = signature["polluter_load_multiplicity"]
    route_rows = group_count * group_size_m * bm
    grid = group_count * group_size_m * n_fanout
    polluter_panels = group_count // share_groups * n_fanout
    element_bytes = 2
    logical_a = grid * bm * k * element_bytes
    logical_b_base = grid * k * bn * element_bytes if include_polluter else 0
    logical_b = logical_b_base * polluter_load_multiplicity
    unique_a = min(route_rows, source_rows) * k * element_bytes
    unique_b = (
        polluter_panels * k * bn * element_bytes if include_polluter else 0)
    unique_index = route_rows * 4
    cache_working_set = unique_a + unique_b + unique_index
    expected_a_l2 = logical_a * replay_a
    expected_b_l2 = logical_b_base * replay_b
    return {
        "bm": bm,
        "bk": signature["bk"],
        "bn": bn,
        "group_size_m": group_size_m,
        "route_rows": route_rows,
        "grid": grid,
        "block": BLOCK_THREADS,
        "activation_allocation_bytes": source_rows * k * element_bytes,
        "polluter_allocation_bytes": max(1, unique_b),
        "operand_replay_a": replay_a,
        "operand_replay_b": replay_b,
        "polluter_load_multiplicity": polluter_load_multiplicity,
        "logical_a_read_bytes": logical_a,
        "logical_b_base_read_bytes": logical_b_base,
        "logical_b_read_bytes": logical_b,
        "expected_a_l2_read_bytes": expected_a_l2,
        "expected_b_l2_read_bytes": expected_b_l2,
        "expected_production_l2_read_bytes": (
            expected_a_l2 + expected_b_l2 + grid * bm * 4),
        "unique_a_bytes": unique_a,
        "unique_b_bytes": unique_b,
        "logical_index_read_bytes": grid * bm * 4,
        "unique_index_bytes": unique_index,
        "cache_working_set_bytes": cache_working_set,
        "cache_regime": (
            "l2_resident" if cache_working_set <= L2_CAPACITY_BYTES
            else "streaming"),
        "sink_write_bytes": grid * BLOCK_THREADS * 4,
    }


def _evidence_role(label, profile_setup):
    if profile_setup == "prelaunch_flush_diagnostic":
        return "diagnostic"
    if profile_setup != "prewarm":
        raise ValueError(f"unknown profile setup: {profile_setup!r}")
    if label in HOLDOUT_LABELS | PRODUCTION_SCALE_HOLDOUT_LABELS:
        return "holdout"
    return "calibration"


def _config(
        axis, label, signature, k, source_rows, group_count, n_fanout,
        weight_share_groups, dynamic_shared_bytes, include_polluter,
        profile_setup, *, include_mma=False, route_pattern="periodic"):
    _signature(signature)
    for value, name in (
            (k, "k"), (source_rows, "source_rows"),
            (group_count, "group_count"), (n_fanout, "n_fanout"),
            (weight_share_groups, "weight_share_groups"),
            (dynamic_shared_bytes, "dynamic_shared_bytes")):
        _positive_integer(value, name)
    if source_rows & (source_rows - 1):
        raise ValueError("source_rows must be a power of two")
    if group_count % weight_share_groups:
        raise ValueError("group_count must be divisible by weight_share_groups")
    if dynamic_shared_bytes not in DYNAMIC_SHARED_BYTES:
        raise ValueError("dynamic_shared_bytes is outside the frozen matrix")
    if not isinstance(include_polluter, bool):
        raise TypeError("include_polluter must be boolean")
    if not isinstance(include_mma, bool):
        raise TypeError("include_mma must be boolean")
    if route_pattern not in {"periodic", "expert_contiguous"}:
        raise ValueError("route_pattern must be periodic or expert_contiguous")
    if profile_setup not in {"prewarm", "prelaunch_flush_diagnostic"}:
        raise ValueError(
            "profile_setup must be prewarm or prelaunch_flush_diagnostic")
    signature_fields = _signature(signature)
    if k % signature_fields["bk"]:
        raise ValueError("k must be divisible by the signature BK")
    if not axis or not label:
        raise ValueError("axis and label must be non-empty")
    return {
        "axis": axis,
        "label": label,
        "signature": signature,
        "k": k,
        "source_rows": source_rows,
        "group_count": group_count,
        "n_fanout": n_fanout,
        "weight_share_groups": weight_share_groups,
        "dynamic_shared_bytes": dynamic_shared_bytes,
        "include_polluter": include_polluter,
        "include_mma": include_mma,
        "route_pattern": route_pattern,
        "profile_setup": profile_setup,
        "evidence_role": _evidence_role(label, profile_setup),
    }


def _full_configs():
    bm64 = "bm64_bk32_bn64_gsm8"
    bm16 = "bm16_bk64_bn32_gsm1"
    configs = []

    for source_rows in (256, 512, 1024, 4096, 32768):
        configs.append(_config(
            "a_working_set", f"bm64_k7168_rows{source_rows}_prewarm_nob",
            bm64, 7168, source_rows,
            _groups_covering_source(bm64, source_rows, 1), 64, 1,
            16 * 1024, False, "prewarm"))
    for source_rows in (512, 1024):
        configs.append(_config(
            "profile_setup_diagnostic",
            f"bm64_k7168_rows{source_rows}_preflush_diag_nob", bm64, 7168,
            source_rows, _groups_covering_source(bm64, source_rows, 1),
            64, 1, 16 * 1024, False, "prelaunch_flush_diagnostic"))

    for n_fanout in (1, 8, 32, 64, 112):
        configs.append(_config(
            "n_fanout_with_b", f"bm64_k7168_fanout{n_fanout}_prewarm_b",
            bm64, 7168, 1024, 2, n_fanout, 2, 16 * 1024, True,
            "prewarm"))
    for n_fanout in (1, 8, 32, 112):
        configs.append(_config(
            "n_fanout_without_b",
            f"bm64_k7168_fanout{n_fanout}_prewarm_nob", bm64, 7168,
            1024, 2, n_fanout, 1, 16 * 1024, False, "prewarm"))
    configs.append(_config(
        "profile_setup_diagnostic", "bm64_k7168_rows1024_preflush_diag_b",
        bm64, 7168, 1024, 2, 64, 2, 16 * 1024, True,
        "prelaunch_flush_diagnostic"))

    for shared_bytes in (32 * 1024, 64 * 1024):
        configs.append(_config(
            "cta_residency", f"bm64_k7168_smem{shared_bytes}_prewarm_b",
            bm64, 7168, 1024, 2, 64, 2, shared_bytes, True, "prewarm"))
    for share_groups in (1, 2, 4, 8):
        configs.append(_config(
            "weight_reuse", f"bm64_k7168_weightshare{share_groups}",
            bm64, 7168, 4096, 8, 64, share_groups, 16 * 1024, True,
            "prewarm"))

    for source_rows in (2048, 4096):
        groups = _groups_covering_source(bm64, source_rows, 2)
        for profile_setup in ("prewarm", "prelaunch_flush_diagnostic"):
            if profile_setup == "prewarm":
                setup_label = "prewarm"
                axis = "w2_signature"
            elif profile_setup == "prelaunch_flush_diagnostic":
                setup_label = "preflush_diag"
                axis = "profile_setup_diagnostic"
            else:
                raise AssertionError(
                    f"unexpected profile setup: {profile_setup!r}")
            configs.append(_config(
                axis,
                f"bm64_k2048_rows{source_rows}_{setup_label}_b", bm64,
                2048, source_rows, groups, 112, 2, 16 * 1024, True,
                profile_setup))

    for source_rows in (256, 512, 1024):
        configs.append(_config(
            "a_working_set", f"bm16_k7168_rows{source_rows}_prewarm_nob",
            bm16, 7168, source_rows,
            _groups_covering_source(bm16, source_rows, 1), 128, 1,
            16 * 1024, False, "prewarm"))
    for n_fanout in (1, 32, 128):
        configs.append(_config(
            "n_fanout_with_b", f"bm16_k7168_fanout{n_fanout}_prewarm_b",
            bm16, 7168, 1024, 64, n_fanout, 1, 16 * 1024, True,
            "prewarm"))
    for n_fanout in (1, 32):
        configs.append(_config(
            "n_fanout_without_b",
            f"bm16_k7168_fanout{n_fanout}_prewarm_nob", bm16, 7168,
            1024, 64, n_fanout, 1, 16 * 1024, False, "prewarm"))
    configs.append(_config(
        "profile_setup_diagnostic", "bm16_k7168_rows1024_preflush_diag_b",
        bm16, 7168, 1024, 64, 128, 1, 16 * 1024, True,
        "prelaunch_flush_diagnostic"))
    for shared_bytes in (32 * 1024, 64 * 1024):
        configs.append(_config(
            "cta_residency", f"bm16_k7168_smem{shared_bytes}_prewarm_b",
            bm16, 7168, 1024, 64, 128, 1, shared_bytes, True, "prewarm"))

    labels = tuple(config["label"] for config in configs)
    if len(labels) != len(set(labels)):
        raise AssertionError("gather-cache matrix contains duplicate labels")
    missing_holdouts = HOLDOUT_LABELS - set(labels)
    if missing_holdouts:
        raise AssertionError(
            f"gather-cache holdout labels are missing: {sorted(missing_holdouts)}")
    roles_by_group = {}
    for config in configs:
        if config["evidence_role"] == "diagnostic":
            continue
        derived = _derived(config)
        group = (
            config["signature"], config["include_polluter"],
            derived["cache_regime"])
        if group not in roles_by_group:
            roles_by_group[group] = set()
        roles_by_group[group].add(config["evidence_role"])
    incomplete_groups = {
        group: roles for group, roles in roles_by_group.items()
        if roles != {"calibration", "holdout"}}
    if incomplete_groups:
        raise AssertionError(
            f"gather-cache parameter groups lack calibration/holdout: "
            f"{incomplete_groups}")
    return tuple(configs)


def _production_scale_holdout_configs():
    """Return the two counter-only large-shape mechanism holdouts.

    Both cases retain the production BM64/BK32/BN64/GROUP_SIZE_M=8 load
    order while omitting MMA and matrix output.  W1 cycles 262,144 routed
    rows over 32,768 source rows exactly eight times, matching TOPK=8 input
    reuse.  W2 gives every routed row a distinct source row.  Two adjacent M
    groups share each weight panel, matching the dominant 16-M-tile expert
    run in the frozen production routing histogram.  These are holdouts for
    the smaller full matrix and never publish a parameter by themselves.
    """

    signature = "bm64_bk32_bn64_gsm8"
    configs = (
        _config(
            "production_scale_holdout",
            "bm64_w1_topk8_production_scale_b",
            signature,
            7168,
            32768,
            512,
            64,
            2,
            16 * 1024,
            True,
            "prewarm",
        ),
        _config(
            "production_scale_holdout",
            "bm64_w2_routes_production_scale_b",
            signature,
            2048,
            262144,
            512,
            112,
            2,
            16 * 1024,
            True,
            "prewarm",
        ),
    )
    if {config["label"] for config in configs} != (
            PRODUCTION_SCALE_HOLDOUT_LABELS):
        raise AssertionError("production-scale holdout labels drifted")
    if any(config["evidence_role"] != "holdout" for config in configs):
        raise AssertionError("production-scale cases must remain holdouts")
    return configs


def _mixed_mma_configs():
    """Return two unchanged address streams with tile-count BF16 MMA.

    Matching load-only cells already exist in the full campaign.  These two
    cells change only the eight wave-level 16x16x16 instructions per K step,
    so they isolate matrix-issue effects without reproducing FusedMoE.
    """

    signature = "bm64_bk32_bn64_gsm8"
    return (
        _config(
            "mixed_mma", "bm64_k7168_weightshare2_mma",
            signature, 7168, 4096, 8, 64, 2, 16 * 1024, True,
            "prewarm", include_mma=True),
        _config(
            "mixed_mma", "bm64_k2048_rows4096_mma",
            signature, 2048, 4096, 8, 112, 2, 16 * 1024, True,
            "prewarm", include_mma=True),
    )


def _expert_route_mma_configs():
    """Return the two production-like expert-contiguous route endpoints."""

    signature = "bm64_bk32_bn64_gsm8"
    return (
        _config(
            "expert_route_mma", "bm64_k7168_topk8_expert_route_mma",
            signature, 7168, 4096, 64, 64, 2, 16 * 1024, True,
            "prewarm", include_mma=True, route_pattern="expert_contiguous"),
        _config(
            "expert_route_mma", "bm64_k2048_routes_expert_route_mma",
            signature, 2048, 32768, 64, 112, 2, 16 * 1024, True,
            "prewarm", include_mma=True, route_pattern="expert_contiguous"),
    )


SMOKE_LABELS = {
    "bm64_k7168_rows512_prewarm_nob",
    "bm64_k7168_rows1024_prewarm_nob",
    "bm64_k7168_fanout1_prewarm_b",
    "bm64_k7168_fanout64_prewarm_b",
    "bm64_k7168_rows1024_preflush_diag_b",
    "bm16_k7168_fanout128_prewarm_b",
}


def build_plan(campaign):
    audit_production_source()
    if campaign == "full":
        configs = _full_configs()
        trials = FULL_TRIALS
    elif campaign == "smoke":
        configs = tuple(
            config for config in _full_configs()
            if config["label"] in SMOKE_LABELS)
        trials = SMOKE_TRIALS
        if {config["label"] for config in configs} != SMOKE_LABELS:
            raise AssertionError("smoke labels do not match the full matrix")
    elif campaign == "production_scale_holdout":
        configs = _production_scale_holdout_configs()
        trials = PRODUCTION_SCALE_HOLDOUT_TRIALS
    elif campaign == "mixed_mma":
        configs = _mixed_mma_configs()
        trials = MIXED_MMA_TRIALS
    elif campaign == "expert_route_mma":
        configs = _expert_route_mma_configs()
        trials = EXPERT_ROUTE_MMA_TRIALS
    else:
        raise ValueError(f"unknown campaign: {campaign!r}")

    samples = []
    for trial in trials:
        ordered_configs = list(configs)
        random.Random(RUN_ORDER_SEED + trial).shuffle(ordered_configs)
        for order_position, config in enumerate(ordered_configs, start=1):
            derived = _derived(config)
            sample_id = f"{config['label']}_trial{trial}"
            arguments = [
                config["signature"], str(config["k"]),
                str(config["source_rows"]), str(config["group_count"]),
                str(config["n_fanout"]),
                str(config["weight_share_groups"]),
                str(config["dynamic_shared_bytes"]),
                "1" if config["include_polluter"] else "0",
                str(derived["polluter_load_multiplicity"]),
                "1" if config["include_mma"] else "0",
                config["route_pattern"],
                str(TIMED_ITERATIONS), str(ATTACH_WAIT_SECONDS), "1",
                config["profile_setup"],
            ]
            samples.append({
                "sample_id": sample_id,
                "trial": trial,
                "order_position": order_position,
                **config,
                **derived,
                "timed_iterations": TIMED_ITERATIONS,
                "attach_wait_seconds": ATTACH_WAIT_SECONDS,
                "profile": True,
                "binary_arguments": arguments,
            })
    release_contract = {
        "calibration_partition": "explicit_label_allowlist_complement",
        "holdout_labels": sorted(HOLDOUT_LABELS),
        "diagnostic_profile_setup": "prelaunch_flush_diagnostic",
        "parameter_group_fields": [
            "signature", "include_polluter", "cache_regime"],
        "parameter_value": (
            "median(expected_a_l2_read_bytes/event_seconds) over "
            "calibration cells"),
        "holdout_prediction": (
            "expected_a_l2_read_bytes/parameter_capacity_bytes_per_s"),
        "counter_relative_span_limit": COUNTER_RELATIVE_SPAN_LIMIT,
        "event_relative_span_limit": EVENT_RELATIVE_SPAN_LIMIT,
        "holdout_mean_ape_limit_percent": HOLDOUT_MEAN_APE_LIMIT_PERCENT,
        "holdout_max_ape_limit_percent": HOLDOUT_MAX_APE_LIMIT_PERCENT,
        "l2_contract_relative_error_limit": 0.05,
        "cold_or_first_touch_parameter_published": False,
    }
    questions = [
        (
            "Does A-only downstream traffic increase when its unique "
            "working set crosses the documented 8-MiB L2 capacity?"),
        (
            "At fixed A work, does increasing N-fanout with independent "
            "B panels increase downstream traffic?"),
        (
            "At fixed work, does reducing resident CTAs with dynamic "
            "shared memory reduce the downstream miss fraction?"),
        (
            "Does a one-time pre-marker flush change counters? This is "
            "diagnostic only because profiler replay prevents a cold-state "
            "claim."),
    ]
    if campaign == "production_scale_holdout":
        release_contract = {
            "calibration_source": "independent_full_matrix",
            "holdout_labels": sorted(PRODUCTION_SCALE_HOLDOUT_LABELS),
            "counter_relative_span_limit": COUNTER_RELATIVE_SPAN_LIMIT,
            "event_relative_span_limit": EVENT_RELATIVE_SPAN_LIMIT,
            "l2_contract_relative_error_limit": 0.05,
            "parameter_publication": False,
        }
        questions = [
            (
                "Do the full-matrix cache and service mechanisms remain valid "
                "when W1 revisits every source row eight times and W1/W2 carry "
                "production-scale weight working sets?"),
        ]
    elif campaign == "mixed_mma":
        release_contract = {
            "paired_load_only_source": "independent_full_matrix",
            "counter_relative_span_limit": COUNTER_RELATIVE_SPAN_LIMIT,
            "event_relative_span_limit": EVENT_RELATIVE_SPAN_LIMIT,
            "l2_contract_relative_error_limit": 0.05,
            "parameter_publication": False,
        }
        questions = [
            (
                "At an unchanged grouped A/B address stream, do eight "
                "wave-level BF16 MMA instructions per K step change L2 "
                "residency or combined service capacity?"),
        ]
    elif campaign == "expert_route_mma":
        release_contract = {
            "paired_periodic_route_source": "mixed_mma",
            "counter_relative_span_limit": COUNTER_RELATIVE_SPAN_LIMIT,
            "event_relative_span_limit": EVENT_RELATIVE_SPAN_LIMIT,
            "l2_contract_relative_error_limit": 0.05,
            "parameter_publication": False,
        }
        questions = [
            (
                "With MMA fixed on, does an expert-contiguous balanced route "
                "stream change grouped-operand L2 residency or service?"),
        ]
    elif campaign in ("smoke", "full"):
        pass
    else:
        raise ValueError(f"unknown campaign: {campaign!r}")
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": MECHANISM,
        "campaign": campaign,
        "baseline_commit": BASELINE_COMMIT,
        "production_source": {
            "files": [
                {
                    "path": relative_path,
                    "sha256": PRODUCTION_SOURCE_SHA256S[relative_path],
                }
                for relative_path in PRODUCTION_SOURCE_RELATIVE_PATHS
            ],
            "bundle_sha256": PRODUCTION_SOURCE_BUNDLE_SHA256,
            "program_order": "group_size_m_then_n_fanout",
            "activation_address": "sorted_token_id_div_top_k",
            "weight_address": "expert_id_and_n_tile",
        },
        "production_or_evaluation_duration_consumed": False,
        "l2_capacity_bytes": L2_CAPACITY_BYTES,
        "run_order_seed": RUN_ORDER_SEED,
        "profile_invocations_per_sample": 1,
        "mcprofiler_contract": {
            "mode": "custom_per_kernel",
            "marker_kernel_invocations": 1,
            "isolated_report_ordinal": 1,
            "maximum_attempts_per_sample": MAX_PROFILE_ATTEMPTS_PER_SAMPLE,
            "instrumented_duration_consumed": False,
            "stdout_policy": (
                "preserve_raw_then_extract_one_consecutive_csv_triplet"),
        },
        "release_contract": release_contract,
        "evidence_contract": {
            "event_ms_per_iteration": (
                "independent microbenchmark service time in steady cache "
                "state; never a production target"),
            "l2_source_bytes": "mcProfiler RoofLine l2c_memacs",
            "downstream_bytes": (
                "mcProfiler Global Memory Read/Write bytes, checked against "
                "RoofLine all_memacs and DNoC request counts"),
            "production_duration": "not consumed",
            "prelaunch_flush_diagnostic": (
                "one flush before the marker only; mcProfiler may replay the "
                "profiled kernel, so this cannot establish cold or compulsory "
                "traffic and is excluded from parameter release"),
        },
        "pre_registered_questions": questions,
        "samples": samples,
    }


def write_plan(path, campaign):
    if path.exists():
        raise FileExistsError(f"plan path already exists: {path}")
    plan = build_plan(campaign)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return plan


def read_plan(path, campaign):
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = build_plan(campaign)
    if payload != expected:
        raise ValueError("gather-cache plan differs from the frozen generator")
    return expected


def parse_stdout(path, planned):
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    if len(rows) != 3:
        raise ValueError(f"{path} must contain metadata, header, and sample")
    if tuple(rows[0]) != (
            "metadata", str(STDOUT_SCHEMA_VERSION), HARDWARE_ID,
            "grouped_gather_cache",
            "production_or_evaluation_duration_consumed=0"):
        raise ValueError(f"{path} metadata is invalid")
    if tuple(rows[1]) != ("sample_header",) + STDOUT_FIELDS:
        raise ValueError(f"{path} sample header is invalid")
    if len(rows[2]) != len(STDOUT_FIELDS) + 1 or rows[2][0] != "sample":
        raise ValueError(f"{path} sample row is invalid")
    values = dict(zip(STDOUT_FIELDS, rows[2][1:]))
    parsed = {
        "signature": values["signature"],
        "route_pattern": values["route_pattern"],
        "event_cache_state": values["event_cache_state"],
        "profile_setup": values["profile_setup"],
        "event_ms_per_iteration": _positive_float(
            values["event_ms_per_iteration"], "event_ms_per_iteration"),
        "first_sink": _parse_hex_integer(values["first_sink"], "first_sink"),
        "last_sink": _parse_hex_integer(values["last_sink"], "last_sink"),
    }
    for field in INTEGER_STDOUT_FIELDS:
        parsed[field] = _parse_integer(values[field], field)
    if parsed["first_sink"] == 0 or parsed["last_sink"] == 0:
        raise ValueError("gather-cache sink values must be non-zero")

    expected = {
        "signature": planned["signature"],
        "bm": planned["bm"],
        "bk": planned["bk"],
        "bn": planned["bn"],
        "group_size_m": planned["group_size_m"],
        "k": planned["k"],
        "source_rows": planned["source_rows"],
        "group_count": planned["group_count"],
        "n_fanout": planned["n_fanout"],
        "weight_share_groups": planned["weight_share_groups"],
        "grid": planned["grid"],
        "block": planned["block"],
        "dynamic_shared_bytes": planned["dynamic_shared_bytes"],
        "include_polluter": 1 if planned["include_polluter"] else 0,
        "polluter_load_multiplicity": planned["polluter_load_multiplicity"],
        "include_mma": 1 if planned["include_mma"] else 0,
        "route_pattern": planned["route_pattern"],
        "timed_iterations": planned["timed_iterations"],
        "event_cache_state": EVENT_CACHE_STATE,
        "profile_setup": planned["profile_setup"],
        "logical_a_read_bytes": planned["logical_a_read_bytes"],
        "logical_b_base_read_bytes": planned["logical_b_base_read_bytes"],
        "logical_b_read_bytes": planned["logical_b_read_bytes"],
        "unique_a_bytes": planned["unique_a_bytes"],
        "unique_b_bytes": planned["unique_b_bytes"],
        "index_read_bytes": planned["logical_index_read_bytes"],
        "sink_write_bytes": planned["sink_write_bytes"],
    }
    for field, expected_value in expected.items():
        if parsed[field] != expected_value:
            raise ValueError(
                f"{path} field {field!r} differs from the frozen plan")
    return parsed


def normalize_profile_stdout(raw_path, output_path, planned):
    if output_path.exists():
        raise FileExistsError(
            f"normalized stdout path already exists: {output_path}")
    lines = raw_path.read_text(encoding="utf-8").splitlines()
    prefixes = ("metadata,", "sample_header,", "sample,")
    positions = []
    selected = []
    for prefix in prefixes:
        matches = [
            (index, line) for index, line in enumerate(lines)
            if line.startswith(prefix)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{raw_path} must contain exactly one {prefix[:-1]!r} row")
        positions.append(matches[0][0])
        selected.append(matches[0][1])
    if positions != list(range(positions[0], positions[0] + 3)):
        raise ValueError(
            f"{raw_path} profiler CSV rows must be consecutive and ordered")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
    try:
        return parse_stdout(output_path, planned)
    except Exception:
        output_path.unlink()
        raise


def _profiler_scalars(payload, path):
    if not isinstance(payload, dict) or not payload:
        raise TypeError(f"{path} must contain a non-empty JSON object")
    scalars = {}
    for section, entries in payload.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                raise TypeError(f"{path} section {section!r} has a non-object")
            name = entry["name"]
            value = entry["value"]
            if not isinstance(name, str) or not name:
                raise ValueError(f"{path} contains an invalid metric name")
            if not isinstance(value, dict):
                scalars[name] = value
    internal = payload["Internal"]
    if not isinstance(internal, list):
        raise TypeError(f"{path} Internal section must be a list")
    rooflines = [
        entry for entry in internal
        if isinstance(entry, dict) and entry["name"] == "RoofLine"
    ]
    if len(rooflines) != 1:
        raise ValueError(f"{path} must contain exactly one RoofLine entry")
    value = rooflines[0]["value"]
    if not isinstance(value, dict):
        raise TypeError(f"{path} RoofLine value must be an object")
    data = value["data"]
    processed = value["processed_data"]
    if not isinstance(data, dict) or not isinstance(processed, dict):
        raise TypeError(f"{path} RoofLine payload must contain two objects")
    return scalars, {**processed, **data}


def _required_number(mapping, name, path):
    if name not in mapping:
        raise KeyError(f"{path} is missing mcProfiler metric {name!r}")
    return _mcprofiler_number(mapping[name], f"{path}:{name}")


def parse_profiler_report(path, planned):
    payload = json.loads(path.read_text(encoding="utf-8"))
    scalars, roofline = _profiler_scalars(payload, path)
    workgroups = _required_number(scalars, "WORKGROUPS", path)
    waves = _required_number(scalars, "WAVES", path)
    global_read = _required_number(scalars, "Global Memory Read bytes", path)
    global_write = _required_number(
        scalars, "Global Memory Write bytes", path)
    dnoc_read = _required_number(scalars, "Dnoc Read Req", path)
    dnoc_write = _required_number(scalars, "Dnoc Write Req", path)
    global_read_instructions = _required_number(
        scalars, "Global Read Instructions", path)
    global_write_instructions = _required_number(
        scalars, "Global Write Instructions", path)
    vl1_read_instructions = _required_number(
        scalars, "VL1 Read Instructions", path)
    l2c_read_instructions = _required_number(
        scalars, "L2C Read Instructions", path)
    l2c_write_instructions = _required_number(
        scalars, "L2C Write Instructions", path)
    vl1_hit_rate = _required_number(scalars, "VL1 Hit Rate", path)
    l2_hit_rate = _required_number(scalars, "L2C Hit Rate", path)
    l2_bytes = _required_number(roofline, "l2c_memacs", path)
    downstream_bytes = _required_number(roofline, "all_memacs", path)
    expected_waves = planned["grid"] * planned["block"] / WAVE_SIZE
    if not math.isclose(workgroups, planned["grid"], abs_tol=0.1):
        raise ValueError(f"{path} WORKGROUPS differs from the planned grid")
    if not math.isclose(waves, expected_waves, abs_tol=0.1):
        raise ValueError(f"{path} WAVES differs from the planned launch")
    for value, label in (
            (global_read, "Global Memory Read bytes"),
            (global_write, "Global Memory Write bytes"),
            (dnoc_read, "Dnoc Read Req"),
            (dnoc_write, "Dnoc Write Req"),
            (global_read_instructions, "Global Read Instructions"),
            (global_write_instructions, "Global Write Instructions"),
            (vl1_read_instructions, "VL1 Read Instructions"),
            (l2c_read_instructions, "L2C Read Instructions"),
            (l2c_write_instructions, "L2C Write Instructions"),
            (l2_bytes, "l2c_memacs"),
            (downstream_bytes, "all_memacs")):
        if value < 0:
            raise ValueError(f"{path} {label} cannot be negative")
    for value, label in (
            (vl1_hit_rate, "VL1 Hit Rate"),
            (l2_hit_rate, "L2C Hit Rate")):
        if not 0 <= value <= 100:
            raise ValueError(f"{path} {label} must be within [0, 100]")
    if l2_bytes <= 0 or global_read <= 0:
        raise ValueError(
            f"{path} gather-cache read traffic must be positive")
    if not math.isclose(
            downstream_bytes, global_read + global_write,
            rel_tol=1e-9, abs_tol=256.0):
        raise ValueError(
            f"{path} all_memacs disagrees with global read plus write")
    for byte_count, requests, direction in (
            (global_read, dnoc_read, "read"),
            (global_write, dnoc_write, "write")):
        if byte_count > 0 and requests <= 0:
            raise ValueError(f"{path} has {direction} bytes without DNoC requests")
        if requests > 0 and byte_count / requests > 256.0 + 1e-9:
            raise ValueError(
                f"{path} exceeds two 128-byte lines per DNoC {direction} request")
    return {
        "workgroups": workgroups,
        "waves": waves,
        "l2_source_bytes": l2_bytes,
        "downstream_total_bytes": downstream_bytes,
        "downstream_read_bytes": global_read,
        "downstream_write_bytes": global_write,
        "dnoc_read_requests": dnoc_read,
        "dnoc_write_requests": dnoc_write,
        "global_read_instructions": global_read_instructions,
        "global_write_instructions": global_write_instructions,
        "vl1_read_instructions": vl1_read_instructions,
        "l2c_read_instructions": l2c_read_instructions,
        "l2c_write_instructions": l2c_write_instructions,
        "vl1_hit_rate_percent": vl1_hit_rate,
        "l2_hit_rate_percent": l2_hit_rate,
    }


def _safe_artifact_path(root, relative, label):
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must stay under the artifact root")
    resolved = root / path
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def read_provenance(artifact_root, plan_path):
    path = artifact_root / "provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != PROVENANCE_KEYS:
        raise ValueError("gather-cache provenance schema is invalid")
    if (payload["schema_version"] != SCHEMA_VERSION
            or payload["hardware_id"] != HARDWARE_ID
            or payload["mechanism"] != MECHANISM
            or payload["baseline_commit"] != BASELINE_COMMIT
            or payload["production_or_evaluation_duration_consumed"] is not False):
        raise ValueError("gather-cache provenance identity is invalid")
    expected_plan_hash = _sha256(plan_path)
    if payload["plan_sha256"] != expected_plan_hash:
        raise ValueError("gather-cache plan SHA-256 does not match provenance")
    plan_snapshot = _safe_artifact_path(
        artifact_root, payload["plan_path"], "plan snapshot")
    if _sha256(plan_snapshot) != expected_plan_hash:
        raise ValueError("gather-cache plan snapshot differs from the input plan")
    for path_key, hash_key, label in (
            ("source_path", "source_sha256", "source snapshot"),
            ("runner_path", "runner_sha256", "runner snapshot"),
            ("binary_path", "binary_sha256", "compiled binary"),
            ("device_preflight_path", "device_preflight_sha256",
             "device preflight")):
        artifact = _safe_artifact_path(artifact_root, payload[path_key], label)
        if _sha256(artifact) != payload[hash_key]:
            raise ValueError(f"gather-cache {label} SHA-256 mismatch")
    production_source_files = payload["production_source_files"]
    if (not isinstance(production_source_files, list)
            or len(production_source_files)
            != len(PRODUCTION_SOURCE_RELATIVE_PATHS)):
        raise ValueError(
            "gather-cache production source file set is incomplete")
    source_sha256s = {}
    for expected_relative_path, record in zip(
            PRODUCTION_SOURCE_RELATIVE_PATHS, production_source_files):
        if (not isinstance(record, dict)
                or set(record) != PRODUCTION_SOURCE_RECORD_KEYS):
            raise ValueError(
                "gather-cache production source record is invalid")
        if record["repository_path"] != expected_relative_path:
            raise ValueError(
                "gather-cache production source order or path is invalid")
        if record["sha256"] != PRODUCTION_SOURCE_SHA256S[
                expected_relative_path]:
            raise ValueError(
                "gather-cache production source is not the frozen baseline")
        artifact = _safe_artifact_path(
            artifact_root, record["snapshot_path"],
            "production source snapshot")
        if _sha256(artifact) != record["sha256"]:
            raise ValueError(
                "gather-cache production source snapshot SHA-256 mismatch")
        source_sha256s[expected_relative_path] = record["sha256"]
    if (payload["production_source_bundle_sha256"]
            != PRODUCTION_SOURCE_BUNDLE_SHA256
            or _production_source_bundle_sha256(source_sha256s)
            != PRODUCTION_SOURCE_BUNDLE_SHA256):
        raise ValueError(
            "gather-cache production source bundle is not the frozen baseline")
    for field in (
            "software_revision", "container_image", "physical_device_id"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"gather-cache provenance {field} is invalid")
    if payload["raw_stdout_policy"] != (
            "preserve_raw_then_extract_one_consecutive_csv_triplet"):
        raise ValueError("gather-cache raw stdout policy is invalid")
    return payload


def _isolated_report(sample_dir):
    allk_dir = sample_dir / "mcprofiler_out" / "_allk"
    if not allk_dir.is_dir():
        raise FileNotFoundError(f"mcProfiler _allk directory is missing: {allk_dir}")
    candidates = []
    for path in allk_dir.glob("*.txt.json"):
        name = path.name
        if name == "report.txt.json" or "period" in name:
            continue
        prefix, separator, _remainder = name.partition("_")
        if not separator or not prefix.isdigit():
            continue
        candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(
            f"{sample_dir} must contain exactly one isolated mcProfiler report")
    prefix = candidates[0].name.partition("_")[0]
    if prefix != "1":
        raise ValueError(
            f"{sample_dir} isolated report ordinal must be one")
    return candidates[0]


def _relative_span(values):
    if not values:
        raise ValueError("relative span requires at least one value")
    median = statistics.median(values)
    if median == 0:
        return 0.0 if all(value == 0 for value in values) else math.inf
    return (max(values) - min(values)) / abs(median)


def _parameter_group(item):
    return (
        item["signature"], item["include_polluter"], item["cache_regime"])


def _release_gate(cells, campaign):
    release_cells = [
        cell for cell in cells if cell["evidence_role"] != "diagnostic"]
    l2_failures = [
        cell["label"] for cell in release_cells
        if not cell["l2_contract_pass"]]
    if campaign == "smoke":
        return {
            "status": "smoke_pass" if not l2_failures else "smoke_fail",
            "eligible_for_forward_parameter_publication": False,
            "l2_contract_failures": l2_failures,
            "stability_failures": [],
            "parameter_candidates": [],
            "holdout_predictions": [],
            "reason": (
                "smoke validates execution and L2 request multiplicity only; "
                "it cannot publish Forward parameters"),
        }
    if campaign == "production_scale_holdout":
        stability_failures = [
            cell["label"] for cell in release_cells
            if not cell["stability_pass"]]
        passed = not l2_failures and not stability_failures
        return {
            "status": (
                "holdout_counter_pass" if passed
                else "holdout_counter_fail"),
            "eligible_for_forward_parameter_publication": False,
            "l2_contract_failures": l2_failures,
            "stability_failures": stability_failures,
            "parameter_candidates": [],
            "holdout_predictions": [],
            "reason": (
                "production-scale cases validate counter extrapolation only; "
                "all parameters must come from the independent full matrix"),
        }
    if campaign in {"mixed_mma", "expert_route_mma"}:
        stability_failures = [
            cell["label"] for cell in release_cells
            if not cell["stability_pass"]]
        passed = not l2_failures and not stability_failures
        return {
            "status": f"{campaign}_counter_{'pass' if passed else 'fail'}",
            "eligible_for_forward_parameter_publication": False,
            "l2_contract_failures": l2_failures,
            "stability_failures": stability_failures,
            "parameter_candidates": [],
            "holdout_predictions": [],
            "reason": (
                "paired counter probes isolate one cache-residency mechanism; "
                "a Forward formula still requires comparison with the "
                "registered control cells"),
        }
    if campaign != "full":
        raise AssertionError(f"unexpected campaign: {campaign!r}")

    stability_failures = [
        cell["label"] for cell in release_cells
        if not cell["stability_pass"]]
    calibration = [
        cell for cell in release_cells
        if cell["evidence_role"] == "calibration"]
    holdout = [
        cell for cell in release_cells if cell["evidence_role"] == "holdout"]
    grouped_capacities = {}
    for cell in calibration:
        group = _parameter_group(cell)
        if group not in grouped_capacities:
            grouped_capacities[group] = []
        grouped_capacities[group].append(
            cell["median"]["a_gather_capacity_bytes_per_s"])
    parameters = []
    for group in sorted(grouped_capacities):
        capacities = grouped_capacities[group]
        parameters.append({
            "signature": group[0],
            "include_polluter": group[1],
            "cache_regime": group[2],
            "capacity_bytes_per_s": statistics.median(capacities),
            "calibration_cell_count": len(capacities),
        })
    parameter_by_group = {
        (entry["signature"], entry["include_polluter"],
         entry["cache_regime"]): entry["capacity_bytes_per_s"]
        for entry in parameters}
    missing_groups = sorted({
        _parameter_group(cell) for cell in holdout
        if _parameter_group(cell) not in parameter_by_group})
    predictions = []
    for cell in holdout:
        group = _parameter_group(cell)
        if group not in parameter_by_group:
            continue
        predicted_ms = (
            cell["expected_a_l2_read_bytes"]
            / parameter_by_group[group] * 1000.0)
        actual_ms = cell["median"]["event_ms_per_iteration"]
        ape = abs(predicted_ms - actual_ms) / actual_ms * 100.0
        predictions.append({
            "label": cell["label"],
            "signature": group[0],
            "include_polluter": group[1],
            "cache_regime": group[2],
            "predicted_event_ms": predicted_ms,
            "observed_event_ms": actual_ms,
            "ape_percent": ape,
        })
    holdout_mean_ape = (
        statistics.mean(item["ape_percent"] for item in predictions)
        if predictions else math.inf)
    holdout_max_ape = (
        max(item["ape_percent"] for item in predictions)
        if predictions else math.inf)
    holdout_pass = (
        not missing_groups
        and len(predictions) == len(holdout)
        and holdout_mean_ape <= HOLDOUT_MEAN_APE_LIMIT_PERCENT
        and holdout_max_ape <= HOLDOUT_MAX_APE_LIMIT_PERCENT)
    passed = not l2_failures and not stability_failures and holdout_pass
    return {
        "status": "pass" if passed else "fail",
        "eligible_for_forward_parameter_publication": passed,
        "l2_contract_failures": l2_failures,
        "stability_failures": stability_failures,
        "missing_parameter_groups": [list(group) for group in missing_groups],
        "parameter_candidates": parameters if passed else [],
        "holdout_predictions": predictions,
        "holdout_mean_ape_percent": holdout_mean_ape,
        "holdout_max_ape_percent": holdout_max_ape,
    }


def build_report(plan_path, campaign, artifact_root):
    plan = read_plan(plan_path, campaign)
    provenance = read_provenance(artifact_root, plan_path)
    samples = []
    for planned in plan["samples"]:
        sample_dir = artifact_root / "samples" / planned["sample_id"]
        if not sample_dir.is_dir():
            raise FileNotFoundError(f"sample directory is missing: {sample_dir}")
        stdout_path = sample_dir / "stdout.csv"
        if not stdout_path.is_file():
            raise FileNotFoundError(f"sample stdout is missing: {stdout_path}")
        raw_stdout_path = sample_dir / "raw_child_stdout.log"
        if not raw_stdout_path.is_file():
            raise FileNotFoundError(
                f"sample raw child stdout is missing: {raw_stdout_path}")
        stdout = parse_stdout(stdout_path, planned)
        profiler_path = _isolated_report(sample_dir)
        profiler = parse_profiler_report(profiler_path, planned)
        operand_l2_read = (
            planned["expected_a_l2_read_bytes"]
            + planned["expected_b_l2_read_bytes"])
        operand_unique_read = (
            planned["unique_a_bytes"] + planned["unique_b_bytes"])
        index_l2_upper = planned["logical_index_read_bytes"]
        operand_downstream_read_lower = max(
            0.0, profiler["downstream_read_bytes"] - index_l2_upper)
        operand_downstream_read_upper = profiler["downstream_read_bytes"]
        logical_read = (
            planned["logical_a_read_bytes"]
            + planned["logical_b_read_bytes"]
            + planned["logical_index_read_bytes"])
        unique_read = (
            planned["unique_a_bytes"] + planned["unique_b_bytes"]
            + planned["unique_index_bytes"])
        samples.append({
            "sample_id": planned["sample_id"],
            "trial": planned["trial"],
            "axis": planned["axis"],
            "label": planned["label"],
            "signature": planned["signature"],
            "k": planned["k"],
            "source_rows": planned["source_rows"],
            "group_count": planned["group_count"],
            "n_fanout": planned["n_fanout"],
            "weight_share_groups": planned["weight_share_groups"],
            "dynamic_shared_bytes": planned["dynamic_shared_bytes"],
            "include_polluter": planned["include_polluter"],
            "include_mma": planned["include_mma"],
            "route_pattern": planned["route_pattern"],
            "polluter_load_multiplicity": (
                planned["polluter_load_multiplicity"]),
            "profile_setup": planned["profile_setup"],
            "evidence_role": planned["evidence_role"],
            "event_cache_state": stdout["event_cache_state"],
            "grid": planned["grid"],
            "cache_regime": planned["cache_regime"],
            "cache_working_set_bytes": planned["cache_working_set_bytes"],
            "expected_a_l2_read_bytes": planned["expected_a_l2_read_bytes"],
            "expected_b_l2_read_bytes": planned["expected_b_l2_read_bytes"],
            "combined_operand_l2_read_bytes": operand_l2_read,
            "combined_operand_unique_read_bytes": operand_unique_read,
            "index_l2_read_upper_bytes": index_l2_upper,
            "expected_production_l2_read_bytes": (
                planned["expected_production_l2_read_bytes"]),
            "logical_read_bytes": logical_read,
            "unique_read_bytes": unique_read,
            "event_ms_per_iteration": stdout["event_ms_per_iteration"],
            **profiler,
            "downstream_to_l2_ratio": (
                profiler["downstream_total_bytes"]
                / profiler["l2_source_bytes"]
                if profiler["l2_source_bytes"] > 0 else None),
            "downstream_read_to_unique_read_ratio": (
                profiler["downstream_read_bytes"] / unique_read),
            "l2_source_to_logical_read_ratio": (
                profiler["l2_source_bytes"] / logical_read),
            "l2_contract_ratio": (
                profiler["l2_source_bytes"]
                / planned["expected_production_l2_read_bytes"]),
            "a_gather_capacity_bytes_per_s": (
                planned["expected_a_l2_read_bytes"]
                / (stdout["event_ms_per_iteration"] / 1000.0)),
            "combined_operand_capacity_bytes_per_s": (
                operand_l2_read
                / (stdout["event_ms_per_iteration"] / 1000.0)),
            "combined_operand_downstream_read_lower_bytes": (
                operand_downstream_read_lower),
            "combined_operand_downstream_read_upper_bytes": (
                operand_downstream_read_upper),
            "combined_operand_downstream_fraction_lower": (
                operand_downstream_read_lower / operand_l2_read),
            "combined_operand_downstream_fraction_upper": (
                operand_downstream_read_upper / operand_l2_read),
            "control_write_to_operand_l2_ratio": (
                profiler["downstream_write_bytes"] / operand_l2_read),
            "stdout_path": str(stdout_path.relative_to(artifact_root)),
            "stdout_sha256": _sha256(stdout_path),
            "raw_child_stdout_path": str(
                raw_stdout_path.relative_to(artifact_root)),
            "raw_child_stdout_sha256": _sha256(raw_stdout_path),
            "profiler_report_path": str(
                profiler_path.relative_to(artifact_root)),
            "profiler_report_sha256": _sha256(profiler_path),
        })

    if campaign == "full":
        configs = _full_configs()
        expected_count = len(FULL_TRIALS)
    elif campaign == "smoke":
        configs = tuple(
            config for config in _full_configs()
            if config["label"] in SMOKE_LABELS)
        expected_count = len(SMOKE_TRIALS)
    elif campaign == "production_scale_holdout":
        configs = _production_scale_holdout_configs()
        expected_count = len(PRODUCTION_SCALE_HOLDOUT_TRIALS)
    elif campaign == "mixed_mma":
        configs = _mixed_mma_configs()
        expected_count = len(MIXED_MMA_TRIALS)
    elif campaign == "expert_route_mma":
        configs = _expert_route_mma_configs()
        expected_count = len(EXPERT_ROUTE_MMA_TRIALS)
    else:
        raise AssertionError(f"unexpected campaign: {campaign!r}")
    cells = []
    for config in configs:
        selected = [
            sample for sample in samples if sample["label"] == config["label"]]
        if len(selected) != expected_count:
            raise ValueError(f"cell {config['label']!r} is incomplete")
        medians = {}
        for field in (
                "event_ms_per_iteration", "l2_source_bytes",
                "downstream_total_bytes", "downstream_read_bytes",
                "downstream_write_bytes", "l2_hit_rate_percent",
                "vl1_hit_rate_percent", "global_read_instructions",
                "global_write_instructions", "vl1_read_instructions",
                "l2c_read_instructions", "l2c_write_instructions",
                "dnoc_read_requests", "dnoc_write_requests",
                "downstream_to_l2_ratio",
                "downstream_read_to_unique_read_ratio",
                "l2_source_to_logical_read_ratio", "l2_contract_ratio",
                "a_gather_capacity_bytes_per_s",
                "combined_operand_capacity_bytes_per_s",
                "combined_operand_downstream_read_lower_bytes",
                "combined_operand_downstream_read_upper_bytes",
                "combined_operand_downstream_fraction_lower",
                "combined_operand_downstream_fraction_upper",
                "control_write_to_operand_l2_ratio"):
            values = [sample[field] for sample in selected]
            if any(value is None for value in values):
                medians[field] = None
            else:
                medians[field] = statistics.median(values)
        counter_spans = {
            field: _relative_span([sample[field] for sample in selected])
            for field in (
                "l2_source_bytes", "downstream_read_bytes",
                "downstream_write_bytes")}
        event_span = _relative_span([
            sample["event_ms_per_iteration"] for sample in selected])
        stability_pass = (
            campaign in {
                "full", "production_scale_holdout", "mixed_mma",
                "expert_route_mma"}
            and all(span <= COUNTER_RELATIVE_SPAN_LIMIT
                    for span in counter_spans.values())
            and event_span <= EVENT_RELATIVE_SPAN_LIMIT)
        l2_contract_relative_error = abs(
            medians["l2_contract_ratio"] - 1.0)
        cells.append({
            **config,
            **_derived(config),
            "trial_count": len(selected),
            "median": medians,
            "counter_relative_spans": counter_spans,
            "event_relative_span": event_span,
            "stability_pass": stability_pass,
            "l2_contract_relative_error": l2_contract_relative_error,
            "l2_contract_pass": l2_contract_relative_error <= 0.05,
        })
    release_gate = _release_gate(cells, campaign)
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": MECHANISM,
        "campaign": campaign,
        "baseline_commit": BASELINE_COMMIT,
        "production_or_evaluation_duration_consumed": False,
        "plan_sha256": _sha256(plan_path),
        "provenance": provenance,
        "sample_count": len(samples),
        "samples": samples,
        "cells": cells,
        "release_gate": release_gate,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    plan_parser = subparsers.add_parser("plan")
    campaign_choices = (
        "smoke", "full", "production_scale_holdout", "mixed_mma",
        "expert_route_mma")
    plan_parser.add_argument(
        "--campaign", choices=campaign_choices, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument(
        "--campaign", choices=campaign_choices, required=True)
    report_parser.add_argument("--plan", type=Path, required=True)
    report_parser.add_argument("--artifact-root", type=Path, required=True)
    report_parser.add_argument("--output", type=Path, required=True)
    normalize_parser = subparsers.add_parser("normalize-stdout")
    normalize_parser.add_argument(
        "--campaign", choices=campaign_choices, required=True)
    normalize_parser.add_argument("--plan", type=Path, required=True)
    normalize_parser.add_argument("--sample-id", required=True)
    normalize_parser.add_argument("--input", type=Path, required=True)
    normalize_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "plan":
        plan = write_plan(args.output, args.campaign)
        print(json.dumps({
            "campaign": args.campaign,
            "sample_count": len(plan["samples"]),
            "output": str(args.output),
        }, sort_keys=True))
        return 0
    if args.mode == "normalize-stdout":
        plan = read_plan(args.plan, args.campaign)
        matches = [
            sample for sample in plan["samples"]
            if sample["sample_id"] == args.sample_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"sample ID must identify exactly one planned sample: "
                f"{args.sample_id!r}")
        normalize_profile_stdout(args.input, args.output, matches[0])
        print(json.dumps({
            "input": str(args.input),
            "output": str(args.output),
            "sample_id": args.sample_id,
        }, sort_keys=True))
        return 0
    if args.mode != "report":
        raise AssertionError(f"unexpected mode: {args.mode!r}")
    if args.output.exists():
        raise FileExistsError(f"report path already exists: {args.output}")
    report = build_report(args.plan, args.campaign, args.artifact_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "campaign": args.campaign,
        "sample_count": report["sample_count"],
        "cell_count": len(report["cells"]),
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
