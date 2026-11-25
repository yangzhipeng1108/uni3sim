"""NVIDIA H200 kernel-dataset plans and campaign candidate assembly.

H200 candidates use the A800 canonical timing contract
``nsys_nvtx_kernel_median`` (three independent NSYS process windows per case,
median whole-trace window, 5% three-window reproducibility limit; see
``experiment_logs/kernel_golden/nvidia_a800_device_golden_20260912``).
``build_nvidia_h200_plans`` turns the A800 canonical case list into H200
runner plans (the A800 job commands with only the runner path switched to
the H200 owner copy and the NVTX marker prefix switched to ``H200_KERNEL``).
``publish_campaign_candidates`` merges the evaluation-supported
candidates of an assembled root into the canonical per-family datasets.
``assemble_campaign_plan`` assembles any kcampaign plan run -- including
H200-only cases and families -- with the plan as case list and
``op_info.Attr`` rebuilt from runner arguments plus launch evidence.  H200 has no Nsight Compute, so the per-kernel records
carry NSYS launch facts (name, grid, block, registers, shared memory) in the
A800 NCU ``Metadata`` / ``Launch Statistics`` field names; NCU counter
sections are absent.
"""

from __future__ import annotations

import collections
import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
from typing import Any, Mapping, Sequence

from dataset_pipeline.kernel_dataset.config import (
    get_kernel_dataset_path,
    load_pipeline_config,
)
from dataset_pipeline.kernel_dataset.dataset_writer import (
    build_kernel_evaluation_timing,
    publish_complete_dataset,
)
from dataset_pipeline.kernel_dataset.nvidia_a800 import (
    _apply_canonical_gemm_timing_dtypes,
    _build_jobs_with_executables,
    _resolve_kernel_cfg,
)
from dataset_pipeline.kernel_dataset.nvidia_a800_event_capture import (
    build_nvidia_a800_moe_topk_ids,
)
from dataset_pipeline.kernel_dataset.nvidia_h200_capture import (
    H200_HARDWARE_ID,
    H200_REQUIRED_SM_CLOCK_MHZ,
    H200_VLLM_STACKS,
)
from dataset_pipeline.kernel_dataset.workload_shapes import (
    build_vector_shapes,
)
from src.kernel.dataset_storage import (
    KERNEL_DATASET_SUFFIX,
    compress_kernel_dataset_content,
    load_kernel_dataset_document,
)


H200_FAMILIES = ("fused_moe", "gemm", "vector", "flash_attn")
H200_RUNNER_RELPATHS = {
    family: f"scripts/kernel_dataset/nvidia_h200/{family}_runner.py"
    for family in H200_FAMILIES
}
H200_WINDOWS_PER_CASE = 3
H200_MAXIMUM_WINDOW_RELATIVE_DIFFERENCE = 0.05
H200_TIMING_SOURCE = "nsys_nvtx_kernel_median"
_PLACEHOLDER_PYTHON = "__python__"


def _canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(
        payload, separators=(",", ":"), sort_keys=True,
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def h200_timing_marker(family: str, case_id: str) -> str:
    return f"H200_KERNEL::{family}::{case_id}"


def _option_value(command: Sequence[str], option: str) -> str:
    indexes = [index for index, value in enumerate(command) if value == option]
    if len(indexes) != 1:
        raise ValueError(f"command must contain exactly one {option}")
    return command[indexes[0] + 1]


def _without_option(command: Sequence[str], option: str) -> list[str]:
    indexes = [index for index, value in enumerate(command) if value == option]
    if not indexes:
        return list(command)
    if len(indexes) != 1:
        raise ValueError(f"command repeats {option}")
    index = indexes[0]
    return list(command[:index]) + list(command[index + 2:])


def _flash_attn_serving_command(attributes: Mapping[str, Any]) -> list[str]:
    if attributes["split_policy"] != "serving" or attributes["SP_REQUESTED"] != 1:
        raise ValueError("FA_SERVING case must be serving policy with SP 1")
    return [
        _PLACEHOLDER_PYTHON, "scripts/kernel_dataset/nvidia_a800/flash_attn_runner.py",
        "--batch", str(attributes["B"]),
        "--seq_len_cache", str(attributes["KV_LEN"]),
        "--num_heads_q", str(attributes["HQ"]),
        "--num_heads_kv", str(attributes["HKV"]),
        "--head_dim", str(attributes["D"]),
        "--dtype", attributes["dtype"],
        "--kv_layout", attributes["kv_layout"],
        "--kv_dist", attributes["kv_dist"],
        "--seq_len_q", str(attributes["Q_LEN"]),
        "--q-lens-json", json.dumps(attributes["Q_LENS"]),
        "--kv-lens-json", json.dumps(attributes["KV_LENS"]),
        "--causal", str(int(attributes["causal"])),
        "--num_splits", "1",
        "--nvtx_range_name", "profile_region",
    ]


def build_nvidia_h200_plans(
        *, a800_config_path: Path, family: str) -> Mapping[str, Any]:
    """Return H200 runner plans for every A800 canonical case of a family."""

    if family not in H200_FAMILIES:
        raise ValueError(f"no H200 NSYS plan builder for family {family!r}")
    config = load_pipeline_config(str(a800_config_path))
    if config["hardware_id"] != "nvidia_a800":
        raise ValueError("H200 plans are built from the A800 case contract")
    kernel_config = _resolve_kernel_cfg(config, family)
    jobs = _build_jobs_with_executables(
        config, family, kernel_config, "ncu-not-used", _PLACEHOLDER_PYTHON)
    dataset_path = Path(get_kernel_dataset_path(config, family))
    dataset = load_kernel_dataset_document(dataset_path)
    jobs = _apply_canonical_gemm_timing_dtypes(family, jobs, dataset)
    job_index = {job["filename"]: job for job in jobs}
    if len(job_index) != len(jobs):
        raise ValueError(f"A800 {family} jobs contain duplicate case IDs")
    missing = sorted(set(dataset) - set(job_index))
    if family == "flash_attn":
        # FA_SERVING cases come from the frozen serving-capture manifest, not
        # the YAML grid; their runner arguments are the same flash_attn
        # runner CLI rebuilt from the canonical Attr (explicit Q/KV lengths).
        for case_id in missing:
            if not case_id.startswith("FA_SERVING_"):
                raise KeyError(f"A800 FA case has no runner job: {case_id}")
            job_index[case_id] = {"runner_cmd": _flash_attn_serving_command(
                dataset[case_id]["op_info"]["Attr"])}
        missing = []
    if missing:
        raise KeyError(
            f"A800 canonical {family} cases have no runner job: {missing[:5]}")
    plans = []
    for case_id in dataset:
        command = list(job_index[case_id]["runner_cmd"])
        if command[0] != _PLACEHOLDER_PYTHON:
            raise ValueError(f"unexpected A800 runner command: {command!r}")
        a800_runner = Path(command[1])
        if a800_runner.name != f"{family}_runner.py":
            raise ValueError(f"unexpected A800 runner path: {a800_runner}")
        arguments = _without_option(command[2:], "--tiling_output")
        marker = h200_timing_marker(family, case_id)
        _option_value(arguments, "--nvtx_range_name")
        index = arguments.index("--nvtx_range_name")
        arguments[index + 1] = marker
        attributes = dataset[case_id]["op_info"]["Attr"]
        if family == "fused_moe":
            build_nvidia_a800_moe_topk_ids(
                token_counts=attributes["tok_per_expert"],
                num_tokens=attributes["TOK"], num_experts=attributes["E"],
                top_k=attributes["TOPK"],
                context=f"A800 canonical case {case_id}")
            arguments += [
                "--expected-tok-per-expert-json",
                json.dumps(attributes["tok_per_expert"],
                           separators=(",", ":")),
            ]
        plans.append({
            "family": family,
            "case_id": case_id,
            "marker": marker,
            "runner_relpath": H200_RUNNER_RELPATHS[family],
            "a800_runner_relpath": (
                f"scripts/kernel_dataset/nvidia_a800/{family}_runner.py"),
            "runner_args": arguments,
            "canonical_attributes_sha256": _canonical_json_sha256(attributes),
        })
    return {
        "schema_version": 1,
        "hardware_id": H200_HARDWARE_ID,
        "family": family,
        "case_source": str(dataset_path),
        "case_count": len(plans),
        "plans": plans,
    }


def _dimensions_text(values: Sequence[int]) -> str:
    return "(" + ", ".join(str(int(value)) for value in values) + ")"


def _kernel_record(launch: Mapping[str, Any],
                   compute_capability: str) -> Mapping[str, Any]:
    grid = launch["grid"]
    block = launch["block"]
    grid_total = grid[0] * grid[1] * grid[2]
    block_total = block[0] * block[1] * block[2]
    return {
        "Metadata": {
            "Block Size": _dimensions_text(block),
            "CC": compute_capability,
            "Grid Size": _dimensions_text(grid),
            "Kernel Name": launch["name"],
            "Source": "nsys_cupti_kernel_activity",
        },
        "Sections": {
            "Launch Statistics": {
                "Header": {
                    "Block Size": str(block_total),
                    "Dynamic Shared Memory Per Block [byte/block]": str(
                        launch["dynamic_shared_memory_bytes"]),
                    "Grid Size": str(grid_total),
                    "Registers Per Thread [register/thread]": str(
                        launch["registers_per_thread"]),
                    "Shared Memory Configuration Size [byte]": str(
                        launch["shared_memory_executed_bytes"]),
                    "Static Shared Memory Per Block [byte/block]": str(
                        launch["static_shared_memory_bytes"]),
                    "Threads [thread]": str(grid_total * block_total),
                },
            },
        },
    }


def _launch_signature(launches: Sequence[Mapping[str, Any]]) -> list:
    return [
        (item["name"], tuple(item["grid"]), tuple(item["block"]),
         item["registers_per_thread"], item["static_shared_memory_bytes"],
         item["dynamic_shared_memory_bytes"])
        for item in launches
    ]


# ── Campaign candidate assembly ──────────────────────────────────────────
#
# The case list of a candidate is the kcampaign plan JSON itself
# (``microbench/nvidia_h200/kcampaign``), so H200-only cases need no A800
# counterpart.  ``op_info.Attr`` is rebuilt from the plan's runner arguments
# plus the runner's launch evidence (telemetry), never copied from A800.

# Windows written before the one-process-per-GPU switch (2026-09-26 03:14:29
# CST, host clock) were captured with concurrent CUDA contexts on the same
# GPU and were purged; any such window that still appears is rejected.
H200_SINGLE_PROCESS_CAPTURE_START_EPOCH_NS = int(datetime.datetime(
    2026, 9, 26, 3, 14, 29,
    tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()) * (
        10 ** 9)

# Families whose op Type is a kernel-evaluation op type with a predictor
# KernelKind (evaluation/kernel/evaluation_runner._OP_TYPE_PREDICTOR_KINDS).
H200_EVALUATION_FAMILIES = (
    "gemm", "vector", "flash_attn", "fused_moe",
    "wq_fused_moe", "v4attn")

# Evaluation families whose Attr is the plan runner arguments plus the
# runner's launch evidence (consumed by
# evaluation/kernel/observation_normalizer.build_*_case_specs) instead of a
# per-op typed attribute record.  Value: (candidate op Type, required
# telemetry launch-evidence field).
H200_EVALUATION_EVIDENCE_FAMILIES = {
    "wq_fused_moe": ("WQ_FUSEDMOE", "wq_moe_routing"),
    "v4attn": ("V4ATTN", "v4attn_case"),
}

# Families assembled as candidates but without a matching evaluation op type.
# Value: (candidate op Type, required telemetry launch-evidence field, reason).
H200_EVALUATION_UNSUPPORTED_FAMILIES = {
    "kda": (
        "KDA", "kda_case",
        "no KernelKind or evaluation op type for Kimi delta attention"),
    "k3_mla": (
        "K3_MLA", "k3_mla_case",
        "Kimi-K3 gated-MLA decode branch (bmm + FA3 + gate) has no "
        "evaluation op type"),
    "dsa_sparse_mla": (
        "DSA_SPARSE_MLA", "dsa_sparse_mla",
        "src has KernelKind SPARSE_ATTN but kernel evaluation has no op type "
        "for it"),
    "dsa_indexer": (
        "DSA_INDEXER", "dsa_indexer",
        "src has KernelKind SPARSE_ATTN_INDEXER but kernel evaluation has no "
        "op type for it"),
    "dsa_cache_write": (
        "DSA_CACHE_WRITE", "dsa_cache_write",
        "fp8_ds_mla / indexer cache inserts are not the KV_CACHE "
        "reshape_and_cache op of the evaluation"),
}

H200_CAMPAIGN_RUNNER_RELPATHS = {
    "gemm": "scripts/kernel_dataset/nvidia_h200/gemm_runner.py",
    "vector": "scripts/kernel_dataset/nvidia_h200/vector_runner.py",
    "flash_attn": "scripts/kernel_dataset/nvidia_h200/flash_attn_runner.py",
    "fused_moe": "scripts/kernel_dataset/nvidia_h200/fused_moe_runner.py",
    "wq_fused_moe": "scripts/kernel_dataset/nvidia_h200/kc_wq_moe_runner.py",
    "kda": "scripts/kernel_dataset/nvidia_h200/kc_kda_runner.py",
    "k3_mla": "scripts/kernel_dataset/nvidia_h200/kc_k3_mla_runner.py",
    "v4attn": "scripts/kernel_dataset/nvidia_h200/kc_v4attn_runner.py",
    "dsa_sparse_mla": (
        "scripts/kernel_dataset/nvidia_h200/kc_dsa_sparse_mla_runner.py"),
    "dsa_indexer": "scripts/kernel_dataset/nvidia_h200/kc_dsa_indexer_runner.py",
    "dsa_cache_write": (
        "scripts/kernel_dataset/nvidia_h200/kc_dsa_cache_write_runner.py"),
}
if set(H200_CAMPAIGN_RUNNER_RELPATHS) != (
        set(H200_EVALUATION_FAMILIES)
        | set(H200_EVALUATION_UNSUPPORTED_FAMILIES)):
    raise AssertionError("H200 campaign family tables disagree")
if not set(H200_EVALUATION_EVIDENCE_FAMILIES) <= set(H200_EVALUATION_FAMILIES):
    raise AssertionError(
        "H200 evidence families must be evaluation families")

_INTEGER_TEXT = re.compile(r"-?[0-9]+")
_FLOAT_TEXT = re.compile(r"-?[0-9]+(\.[0-9]*)?([eE][-+]?[0-9]+)?")


def load_case_windows(case_dir: Path) -> list[Mapping[str, Any]]:
    windows = []
    for path in sorted(case_dir.glob("window_*.launch.json")):
        windows.append(json.loads(path.read_text(encoding="utf-8")))
    return windows


def load_case_failures(case_dir: Path) -> list[Mapping[str, Any]]:
    failures = []
    for path in sorted(case_dir.glob("window_*.failure.json")):
        failures.append(json.loads(path.read_text(encoding="utf-8")))
    return failures


def window_vllm_stack(window: Mapping[str, Any]) -> str:
    """Return the H200 stack id (``vllm024``/``vllm030``) of one window."""

    environment = window["telemetry"]["environment"]
    version = environment["vllm_version"]
    if not isinstance(version, str) or not version:
        raise ValueError("H200 window environment.vllm_version is missing")
    release = version.split("+")[0]
    if release not in H200_VLLM_STACKS:
        raise ValueError(
            f"H200 window vLLM {version!r} is not a supported capture stack "
            f"{sorted(H200_VLLM_STACKS)}")
    stack = H200_VLLM_STACKS[release]
    if "vllm_stack" in environment and environment["vllm_stack"] != stack:
        raise ValueError(
            f"H200 window vllm_stack {environment['vllm_stack']!r} "
            f"disagrees with vllm_version {version!r}")
    return stack


def select_case_timing(
        case_id: str, windows: Sequence[Mapping[str, Any]],
        failed_window_ids: Sequence[int]) -> Mapping[str, Any]:
    """Apply the A800 ``nsys_nvtx_kernel_median`` rule to H200 windows.

    A window is accepted only when its telemetry ``clock_accepted`` is true
    (SM clock >= 1980 MHz before the NVTX push and after the pop) and it was
    captured after the one-process-per-GPU switch.  The newest three accepted
    windows are retained; their whole-trace totals must agree within 5% and
    the median-total window is selected.
    """

    accepted = []
    clock_rejected = []
    concurrent_rejected = []
    for window in windows:
        telemetry = window["telemetry"]
        before = telemetry["clock_before_nvtx"]["sm_clock_mhz"]
        after = telemetry["clock_after_nvtx"]["sm_clock_mhz"]
        clocks_ok = (before >= H200_REQUIRED_SM_CLOCK_MHZ
                     and after >= H200_REQUIRED_SM_CLOCK_MHZ)
        if telemetry["clock_accepted"] is not clocks_ok:
            raise ValueError(
                f"{case_id} window {window['window']}: clock_accepted "
                f"{telemetry['clock_accepted']!r} contradicts clocks "
                f"{before}/{after} MHz")
        if (telemetry["process_start_epoch_ns"]
                < H200_SINGLE_PROCESS_CAPTURE_START_EPOCH_NS):
            concurrent_rejected.append(window["window"])
        elif clocks_ok:
            accepted.append(window)
        else:
            clock_rejected.append({"window": window["window"],
                                   "clock_before_mhz": before,
                                   "clock_after_mhz": after})
    result = {
        "case_id": case_id,
        "window_count": len(windows),
        "clock_rejected_windows": clock_rejected,
        "concurrent_capture_rejected_windows": concurrent_rejected,
        "failed_windows": list(failed_window_ids),
    }
    if len(accepted) < H200_WINDOWS_PER_CASE:
        if len(windows) < H200_WINDOWS_PER_CASE:
            result["status"] = ("runner_failure" if failed_window_ids
                                else "insufficient_windows")
        elif concurrent_rejected:
            result["status"] = "concurrent_capture_rejected"
        else:
            result["status"] = "insufficient_clock_accepted_windows"
        return result
    # Retries append higher window IDs: the newest three accepted windows
    # form the retained set, as for the A800 canonical selection.
    retained = sorted(accepted, key=lambda item: item["window"])[
        -H200_WINDOWS_PER_CASE:]
    signatures = {json.dumps(_launch_signature(window["launches"]))
                  for window in retained}
    if len(signatures) != 1:
        result["status"] = "launch_identity_differs_between_windows"
        return result
    stacks = {window_vllm_stack(window) for window in retained}
    if len(stacks) != 1:
        result["status"] = "mixed_vllm_versions_between_windows"
        return result
    totals = [sum(item["duration_ns"] for item in window["launches"])
              for window in retained]
    spread = (max(totals) - min(totals)) / max(totals)
    median_total = statistics.median(totals)
    selected = retained[totals.index(median_total)]
    result.update({
        "retained": retained,
        "retained_windows": [window["window"] for window in retained],
        "window_totals_ns": totals,
        "maximum_relative_difference": spread,
        "selected_window": selected["window"],
        "selected": selected,
        "status": ("pass" if spread <= H200_MAXIMUM_WINDOW_RELATIVE_DIFFERENCE
                   else "reproducibility_exceeds_limit"),
    })
    return result


def parse_runner_options(arguments: Sequence[str]) -> dict[str, str]:
    """Return ``{option: text}`` for a strict ``--option value`` list."""

    if len(arguments) % 2:
        raise ValueError(f"runner arguments are not option pairs: {arguments}")
    options = {}
    for index in range(0, len(arguments), 2):
        option, value = arguments[index], arguments[index + 1]
        if not option.startswith("--") or value.startswith("--"):
            raise ValueError(f"invalid runner option pair {option!r} {value!r}")
        if option in options:
            raise ValueError(f"runner option repeated: {option}")
        options[option] = value
    return options


def _require_options(options: Mapping[str, str], required: set[str],
                     optional: set[str], context: str) -> None:
    missing = required - set(options)
    unexpected = set(options) - required - optional
    if missing or unexpected:
        raise ValueError(
            f"{context} runner options invalid: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}")


def _typed_option(value: str) -> Any:
    if _INTEGER_TEXT.fullmatch(value):
        return int(value)
    if _FLOAT_TEXT.fullmatch(value):
        return float(value)
    if value.startswith("["):
        return json.loads(value)
    return value


def _positive(options: Mapping[str, str], option: str) -> int:
    value = int(options[option])
    if value <= 0:
        raise ValueError(f"runner option {option} must be positive: {value}")
    return value


def _gemm_attributes(options: Mapping[str, str]) -> Mapping[str, Any]:
    _require_options(options, {"--bs", "--sl", "--in_feat", "--out_feat",
                               "--dtype"}, set(), "gemm")
    return {
        "B": _positive(options, "--bs"),
        "M": _positive(options, "--sl"),
        "K": _positive(options, "--in_feat"),
        "N": _positive(options, "--out_feat"),
        "dtype": options["--dtype"],
    }


def _vector_type_and_attributes(
        options: Mapping[str, str]) -> tuple[str, Mapping[str, Any]]:
    required = {"--op", "--tok", "--dtype", "--batch", "--op2", "--hidden",
                "--num_heads", "--head_dim", "--intermediate",
                "--num_kv_heads"}
    _require_options(options, required, set(), "vector")
    # One-workload expansion through the canonical hardware-independent
    # expander; its script arguments must reproduce the plan exactly.
    layer_name = "h200_runner_args"
    kernel_config = {
        "workload_configs": [{
            "op": options["--op"], "tok": _positive(options, "--tok"),
            "batch": _positive(options, "--batch"),
            "op2": options["--op2"], "layer": layer_name}],
        "layers": [{
            "name": layer_name, "override_only": False,
            "dtype": options["--dtype"],
            "hidden": _positive(options, "--hidden"),
            "num_heads": _positive(options, "--num_heads"),
            "head_dim": _positive(options, "--head_dim"),
            "intermediate": _positive(options, "--intermediate"),
            "num_kv_heads": _positive(options, "--num_kv_heads")}],
    }
    shapes = build_vector_shapes(kernel_config)
    if len(shapes) != 1:
        raise AssertionError("one vector workload expanded to several shapes")
    shape = shapes[0]
    if parse_runner_options(shape["script_args"]) != dict(options):
        raise ValueError(
            "vector runner options differ from the canonical expansion: "
            f"{shape['script_args']} vs {dict(options)}")
    attributes = {name: value for name, value in shape["attr"].items()
                  if name != "layer"}
    return shape["type"], attributes


def _flash_attn_attributes(options: Mapping[str, str]) -> Mapping[str, Any]:
    required = {"--batch", "--seq_len_cache", "--num_heads_q",
                "--num_heads_kv", "--head_dim", "--dtype", "--kv_layout",
                "--kv_dist", "--seq_len_q", "--q-lens-json", "--kv-lens-json",
                "--causal", "--num_splits"}
    _require_options(options, required, set(), "flash_attn")
    batch = _positive(options, "--batch")
    query_lengths = json.loads(options["--q-lens-json"])
    kv_lengths = json.loads(options["--kv-lens-json"])
    for name, values in (("Q", query_lengths), ("KV", kv_lengths)):
        if (not isinstance(values, list) or len(values) != batch
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value <= 0 for value in values)):
            raise ValueError(f"flash_attn {name} lengths invalid: {values!r}")
    query_length = _positive(options, "--seq_len_q")
    kv_length = _positive(options, "--seq_len_cache")
    if max(query_lengths) != query_length or max(kv_lengths) != kv_length:
        raise ValueError("flash_attn length lists disagree with the maxima")
    causal = int(options["--causal"])
    if causal not in (0, 1):
        raise ValueError(f"flash_attn --causal must be 0/1, got {causal}")
    kv_layout = options["--kv_layout"]
    splits = _positive(options, "--num_splits")
    # H200 runner: only contiguous decode forwards num_splits to FA3 as an
    # executed input; every other layout runs FA3's own heuristic
    # (num_splits=0), i.e. the serving policy.
    forced = kv_layout == "contiguous" and query_length == 1
    if kv_layout == "chunked_prefill":
        # The runner arguments carry the executed per-request chunks only;
        # their sum is the scheduler tokens executed by this case.
        requested_query = sum(query_lengths)
    else:
        requested_query = query_length
    return {
        "B": batch,
        "S": kv_length,
        "Q_LEN": query_length,
        "Q_REQUESTED": requested_query,
        "KV_LEN": kv_length,
        "KV_REQUESTED": kv_length,
        "HQ": _positive(options, "--num_heads_q"),
        "HKV": _positive(options, "--num_heads_kv"),
        "D": _positive(options, "--head_dim"),
        "SP_REQUESTED": splits,
        "Q_LENS": query_lengths,
        "KV_LENS": kv_lengths,
        "split_policy": "forced" if forced else "serving",
        "kv_layout": kv_layout,
        "kv_dist": options["--kv_dist"],
        "causal": bool(causal),
        "dtype": options["--dtype"],
    }


def fa3_split_attributes(evidence: Mapping[str, Any],
                         batch: int) -> Mapping[str, Any]:
    """Decode one ``fa3_scheduler_metadata`` record into split attributes.

    Layout of ``mha_fwd_get_scheduler_metadata`` (pinned FA3 source of both
    H200 stacks): int32 blocks of ``b_rounded = round_up(batch, 4)`` --
    ``prepare_seqlen_q``, ``num_splits_dynamic`` (only when the static split
    bound is > 1), ``num_nheads_in_l2`` (only when the normalized call is
    causal or local) -- followed by one tile semaphore (SM90).  Without the
    dynamic block every request runs one split and no combine kernel is
    launched.
    """

    arguments = evidence["get_scheduler_metadata_arguments"]
    values = evidence["metadata_int32"]
    if arguments["batch_size"] != batch:
        raise ValueError("FA3 scheduler metadata batch differs from the case")
    if arguments["window_size"] != [-1, -1] or arguments["has_softcap"]:
        raise ValueError("FA3 split decoding supports no window / softcap")
    # mha_fwd_get_scheduler_metadata: a one-token query without a window is
    # non-causal unless hdim is in (64, 128] with a paged KV cache.
    causal = arguments["causal"]
    if arguments["max_seqlen_q"] == 1 and (
            arguments["headdim"] <= 64 or arguments["headdim"] > 128
            or arguments["page_size"] is None):
        causal = False
    rounded = (batch + 3) // 4 * 4
    blocks, remainder = divmod(len(values) - 1, rounded)
    dynamic_blocks = blocks - 1 - int(causal)
    if remainder or dynamic_blocks not in (0, 1):
        raise ValueError(
            f"FA3 scheduler metadata of {len(values)} int32 does not match "
            f"batch {batch} (causal={causal})")
    if dynamic_blocks:
        per_request = list(values[rounded:rounded + batch])
    else:
        per_request = [1] * batch
    if any(isinstance(value, bool) or not isinstance(value, int)
           or value < 1 for value in per_request):
        raise ValueError(f"FA3 dynamic split counts invalid: {per_request}")
    return {
        "SP": max(per_request),
        "SP_PER_REQUEST": per_request,
        "FA3_SPLIT_COMBINE": bool(dynamic_blocks),
    }


def _flash_attn_split_attributes(
        retained: Sequence[Mapping[str, Any]],
        batch: int) -> Mapping[str, Any] | None:
    """Return the split attributes the retained windows agree on, if any."""

    decoded = [fa3_split_attributes(
        window["telemetry"]["fa3_scheduler_metadata"], batch)
        for window in retained
        if "fa3_scheduler_metadata" in window["telemetry"]]
    if not decoded:
        return None
    if any(item != decoded[0] for item in decoded):
        raise ValueError(
            f"retained FA windows disagree on the FA3 split decision: "
            f"{decoded}")
    return decoded[0]


def _fused_moe_attributes(options: Mapping[str, str],
                          telemetry: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_options(
        options, {"--num_tokens", "--num_experts", "--top_k", "--hidden_size",
                  "--intermediate_size", "--dtype"},
        {"--expected-tok-per-expert-json"}, "fused_moe")
    tiling = telemetry["moe_tiling"]
    attributes = {
        "TOK": _positive(options, "--num_tokens"),
        "E": _positive(options, "--num_experts"),
        "TOPK": _positive(options, "--top_k"),
        "H": _positive(options, "--hidden_size"),
        "INT": _positive(options, "--intermediate_size"),
        "dtype": options["--dtype"],
    }
    for name, evidence_name in (("TOK", "num_tokens"), ("E", "num_experts"),
                                ("TOPK", "top_k"), ("H", "hidden_size"),
                                ("INT", "intermediate_size")):
        if tiling[evidence_name] != attributes[name]:
            raise ValueError(
                f"fused_moe launch evidence {evidence_name}="
                f"{tiling[evidence_name]} differs from runner {name}")
    tokens_per_expert = tiling["tok_per_expert"]
    if (len(tokens_per_expert) != attributes["E"]
            or sum(tokens_per_expert) != attributes["TOK"] * attributes["TOPK"]):
        raise ValueError("fused_moe launch evidence tok_per_expert invalid")
    if "--expected-tok-per-expert-json" in options and json.loads(
            options["--expected-tok-per-expert-json"]) != tokens_per_expert:
        raise ValueError("fused_moe routing differs from the planned routing")
    attributes["tok_per_expert"] = tokens_per_expert
    attributes["moe_tiling"] = {
        projection: {
            "N": tiling[projection]["N"],
            "K": tiling[projection]["K"],
            "BM": tiling[projection]["BLOCK_SIZE_M"],
            "BN": tiling[projection]["BLOCK_SIZE_N"],
            "BK": tiling[projection]["BLOCK_SIZE_K"],
            "GSM": tiling[projection]["GROUP_SIZE_M"],
            "EM": tiling[projection]["EM"],
            "grid": tiling[projection]["grid"],
        }
        for projection in ("w1", "w2")
    }
    return attributes


def campaign_case_op_info(
        family: str, runner_args: Sequence[str],
        selected: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    """Return ``(Type, Attr)`` from plan runner arguments + launch evidence."""

    options = parse_runner_options(
        _without_option(runner_args, "--nvtx_range_name"))
    telemetry = selected["telemetry"]
    if family == "gemm":
        return "GEMM", _gemm_attributes(options)
    if family == "vector":
        return _vector_type_and_attributes(options)
    if family == "flash_attn":
        return "FLASHATTN", _flash_attn_attributes(options)
    if family == "fused_moe":
        return "FUSEDMOE", _fused_moe_attributes(options, telemetry)
    if family in H200_EVALUATION_EVIDENCE_FAMILIES:
        op_type, evidence_name = H200_EVALUATION_EVIDENCE_FAMILIES[family]
        return op_type, {
            "runner_args": {option[2:]: _typed_option(value)
                            for option, value in options.items()},
            "launch_evidence": telemetry[evidence_name],
        }
    if family in H200_EVALUATION_UNSUPPORTED_FAMILIES:
        op_type, evidence_name, _ = H200_EVALUATION_UNSUPPORTED_FAMILIES[
            family]
        return op_type, {
            "runner_args": {option[2:]: _typed_option(value)
                            for option, value in options.items()},
            "launch_evidence": telemetry[evidence_name],
        }
    raise ValueError(f"no H200 campaign family {family!r}")


def load_campaign_plan(plan_path: Path) -> Mapping[str, Any]:
    """Load one kcampaign plan and validate its case contract."""

    document = json.loads(plan_path.read_text(encoding="utf-8"))
    if "hardware_id" in document and document["hardware_id"] != (
            H200_HARDWARE_ID):
        raise ValueError(f"plan {plan_path} is not an H200 plan")
    plans = document["plans"]
    if not isinstance(plans, list) or not plans:
        raise ValueError(f"plan {plan_path} has no cases")
    case_ids = [plan["case_id"] for plan in plans]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError(f"plan {plan_path} repeats case IDs")
    for plan in plans:
        family = plan["family"]
        if family not in H200_CAMPAIGN_RUNNER_RELPATHS:
            raise ValueError(
                f"plan {plan_path} case {plan['case_id']}: no H200 campaign "
                f"family {family!r}")
        if plan["runner_relpath"] != H200_CAMPAIGN_RUNNER_RELPATHS[family]:
            raise ValueError(
                f"plan {plan_path} case {plan['case_id']}: runner "
                f"{plan['runner_relpath']} is not the {family} runner")
        if plan["marker"] != h200_timing_marker(family, plan["case_id"]):
            raise ValueError(f"plan case {plan['case_id']} marker mismatch")
        if _option_value(plan["runner_args"], "--nvtx_range_name") != (
                plan["marker"]):
            raise ValueError(f"plan case {plan['case_id']} NVTX mismatch")
    return document


def _load_run_provenance(results_root: Path, plan_path: Path) -> list:
    records = json.loads(
        (results_root / "_provenance.json").read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"{results_root} has no launch provenance records")
    plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    for record in records:
        if record["plan_sha256"] != plan_sha256:
            raise ValueError(
                f"{results_root} was launched with plan sha256 "
                f"{record['plan_sha256']}, local plan {plan_path} is "
                f"{plan_sha256}")
    return records


def _entry(family: str, plan: Mapping[str, Any],
           decision: Mapping[str, Any], campaign: str,
           stack: str) -> Mapping[str, Any]:
    selected = decision["selected"]
    op_type, attributes = campaign_case_op_info(
        family, plan["runner_args"], selected)
    if family == "flash_attn":
        attributes = {**attributes, **decision["fa3_split"]}
    compute_capability = selected["telemetry"]["device"]["compute_capability"]
    kernels = [_kernel_record(launch, compute_capability)
               for launch in selected["launches"]]
    durations = [launch["duration_ns"] for launch in selected["launches"]]
    return {
        "op_info": {
            "Type": op_type,
            "Attr": attributes,
            "evaluation": build_kernel_evaluation_timing(
                H200_HARDWARE_ID, kernels, durations),
        },
        "kernels": kernels,
        "h200_capture": {
            "campaign": campaign,
            "vllm_stack": stack,
            "vllm_version": selected["telemetry"]["environment"][
                "vllm_version"],
            "environment": selected["telemetry"]["environment"],
            "selected_window": decision["selected_window"],
            "retained_windows": decision["retained_windows"],
            "window_totals_ns": decision["window_totals_ns"],
            "maximum_relative_difference": decision[
                "maximum_relative_difference"],
        },
    }


def assemble_campaign_plan(
        *, plan_path: Path, results_root: Path,
        output_dir: Path) -> Mapping[str, Any]:
    """Assemble one candidate per family of a kcampaign plan run.

    ``results_root`` is ``runs/<plan>`` (``_provenance.json`` plus
    ``cases/<case>/window_NN.launch.json``).  Every window of the run must
    come from one vLLM stack, so a candidate never mixes versions.
    """

    plan_document = load_campaign_plan(plan_path)
    provenance = _load_run_provenance(results_root, plan_path)
    campaign = plan_path.name[:-len(".json")]
    plans_by_family: dict[str, list] = {}
    for plan in plan_document["plans"]:
        if plan["family"] not in plans_by_family:
            plans_by_family[plan["family"]] = []
        plans_by_family[plan["family"]].append(plan)
    run_stacks = set()
    families = {}
    for family, plans in plans_by_family.items():
        document = {}
        case_reports = []
        clock_values = []
        for plan in plans:
            case_id = plan["case_id"]
            case_dir = results_root / "cases" / case_id
            windows = load_case_windows(case_dir) if case_dir.is_dir() else []
            failures = load_case_failures(case_dir) if case_dir.is_dir() else []
            window_ids = {window["window"] for window in windows}
            failed_ids = sorted(failure["window"] for failure in failures
                                if failure["window"] not in window_ids)
            for window in windows:
                for name in ("case_id", "family", "marker", "runner_args"):
                    if window[name] != plan[name]:
                        raise ValueError(
                            f"{case_dir} window {window['window']} {name} "
                            "differs from the plan")
                run_stacks.add(window_vllm_stack(window))
            if not windows and not failed_ids:
                case_reports.append({"case_id": case_id,
                                     "status": "not_captured"})
                continue
            decision = select_case_timing(case_id, windows, failed_ids)
            if decision["status"] == "pass" and family == "flash_attn":
                split = _flash_attn_split_attributes(
                    decision["retained"],
                    int(_option_value(plan["runner_args"], "--batch")))
                if split is None:
                    decision["status"] = "missing_fa3_split_evidence"
                else:
                    decision["fa3_split"] = split
            case_reports.append({name: value for name, value in
                                 decision.items()
                                 if name not in ("selected", "retained")})
            if decision["status"] != "pass":
                continue
            for window in windows:
                if window["window"] in decision["retained_windows"]:
                    telemetry = window["telemetry"]
                    clock_values.append(
                        telemetry["clock_before_nvtx"]["sm_clock_mhz"])
                    clock_values.append(
                        telemetry["clock_after_nvtx"]["sm_clock_mhz"])
            document[case_id] = decision
        families[family] = (plans, document, case_reports, clock_values)
    if len(run_stacks) > 1:
        raise ValueError(
            f"{results_root} mixes vLLM stacks {sorted(run_stacks)}")
    stack = next(iter(run_stacks)) if run_stacks else None
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for family, (plans, decisions, case_reports, clock_values) in (
            families.items()):
        plan_index = {plan["case_id"]: plan for plan in plans}
        document = {case_id: _entry(family, plan_index[case_id], decision,
                                    campaign, stack)
                    for case_id, decision in decisions.items()}
        candidate_path = None
        stale_path = output_dir / f"{family}{KERNEL_DATASET_SUFFIX}"
        if not document and stale_path.exists():
            # A rerun whose family no longer passes must not leave the
            # previous candidate behind for evaluation to read.
            stale_path.unlink()
        if document:
            candidate_path = stale_path
            payload = json.dumps(document, indent=2, ensure_ascii=False,
                                 allow_nan=False).encode("utf-8")
            candidate_path.write_bytes(
                compress_kernel_dataset_content(payload))
        status_counts = dict(collections.Counter(
            report["status"] for report in case_reports))
        supported = family in H200_EVALUATION_FAMILIES
        manifest = {
            "schema_version": 2,
            "artifact_kind": "nvidia_h200_kcampaign_kernel_dataset_candidate",
            "hardware_id": H200_HARDWARE_ID,
            "campaign": campaign,
            "family": family,
            "plan_path": str(plan_path),
            "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "run_provenance": provenance,
            "vllm_stack": stack,
            "evaluation_ground_truth_only": True,
            "production_or_evaluation_duration_consumed": False,
            "evaluation_supported": supported,
            "evaluation_unsupported_reason": (
                None if supported
                else H200_EVALUATION_UNSUPPORTED_FAMILIES[family][2]),
            "attribute_source": "plan runner_args + runner launch evidence",
            "candidate_path": (None if candidate_path is None
                               else str(candidate_path)),
            "plan_case_count": len(plans),
            "measured_case_count": len(document),
            "complete": len(document) == len(plans),
            "case_status_counts": status_counts,
            "protocol": {
                "timing_source": H200_TIMING_SOURCE,
                "windows_per_retained_case": H200_WINDOWS_PER_CASE,
                "selection": "median_whole_trace_of_three_independent_windows",
                "maximum_relative_difference_limit": (
                    H200_MAXIMUM_WINDOW_RELATIVE_DIFFERENCE),
                "clock_rule": (
                    "telemetry clock_accepted: SM clock before NVTX push and "
                    f"after NVTX pop >= {H200_REQUIRED_SM_CLOCK_MHZ} MHz"),
                "single_process_capture_start_epoch_ns": (
                    H200_SINGLE_PROCESS_CAPTURE_START_EPOCH_NS),
                "power_warmup": "bf16_gemm_4096_x1000_before_runner_warmup",
            },
            "retained_window_sm_clock_mhz": (
                {"min": min(clock_values), "max": max(clock_values)}
                if clock_values else None),
            "cases": case_reports,
        }
        manifest_path = output_dir / f"{family}.manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=1), encoding="utf-8")
        manifests[family] = manifest
    return manifests


def _launch_identity(entry: Mapping[str, Any]) -> list:
    return [(kernel["Metadata"]["Kernel Name"], kernel["Metadata"]["Grid Size"],
             kernel["Metadata"]["Block Size"]) for kernel in entry["kernels"]]


def publish_campaign_candidates(
        *, candidate_root: Path, publication_dir: Path,
        canonical_dir: Path) -> Mapping[str, Any]:
    """Publish the evaluation-supported families of an assembled root.

    A case measured by two plans is one physical case: both observations
    must carry the same launch identity, and the one whose retained windows
    agree more closely (smaller ``maximum_relative_difference``; ties by
    campaign name) is published.  The choice uses measurement agreement
    only, never prediction error.  A published entry keeps ``op_info`` and
    ``kernels`` only, the entry layout of the other NVIDIA kernel dataset;
    the per-case ``h200_capture`` record stays in the candidates.  Every
    family's candidate and a publication record are written under
    ``publication_dir``.
    """

    observations: dict[str, dict[str, list]] = {
        family: {} for family in H200_EVALUATION_FAMILIES}
    sources = []
    stacks = set()
    for manifest_path in sorted(candidate_root.glob(f"*/*.manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        family = manifest["family"]
        campaign = manifest["campaign"]
        if (manifest["artifact_kind"]
                != "nvidia_h200_kcampaign_kernel_dataset_candidate"
                or manifest["hardware_id"] != H200_HARDWARE_ID
                or manifest_path.parent.name != campaign
                or manifest_path.name != f"{family}.manifest.json"):
            raise ValueError(f"{manifest_path} is not an H200 candidate "
                             "manifest for its own plan and family")
        if manifest["evaluation_supported"] is not True:
            continue
        if manifest["candidate_path"] is None:
            continue
        payload_path = manifest_path.parent / f"{family}{KERNEL_DATASET_SUFFIX}"
        document = load_kernel_dataset_document(payload_path)
        if len(document) != manifest["measured_case_count"]:
            raise ValueError(f"{payload_path} disagrees with its manifest")
        stacks.add(manifest["vllm_stack"])
        sources.append({
            "campaign": campaign, "family": family,
            "manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()).hexdigest(),
            "candidate_sha256": hashlib.sha256(
                payload_path.read_bytes()).hexdigest(),
            "measured_case_count": len(document),
        })
        for case_id, entry in document.items():
            if set(entry) != {"op_info", "kernels", "h200_capture"}:
                raise ValueError(f"{payload_path}:{case_id} entry fields are "
                                 f"{sorted(entry)}")
            if entry["h200_capture"]["campaign"] != campaign:
                raise ValueError(f"{payload_path}:{case_id} names another "
                                 "campaign")
            if case_id not in observations[family]:
                observations[family][case_id] = []
            observations[family][case_id].append(entry)
    if len(stacks) != 1:
        raise ValueError(f"{candidate_root} holds vLLM stacks {sorted(stacks)}")

    deduplicated = []
    published = {}
    for family in H200_EVALUATION_FAMILIES:
        dataset = {}
        for case_id in sorted(observations[family]):
            entries = observations[family][case_id]
            if len({json.dumps(_launch_identity(entry)) for entry in entries}) != 1:
                raise ValueError(
                    f"{family}/{case_id} was measured with different launches")
            chosen = min(entries, key=lambda entry: (
                entry["h200_capture"]["maximum_relative_difference"],
                entry["h200_capture"]["campaign"]))
            if len(entries) > 1:
                deduplicated.append({
                    "family": family, "case_id": case_id,
                    "published_campaign": chosen["h200_capture"]["campaign"],
                    "observations": [{
                        "campaign": entry["h200_capture"]["campaign"],
                        "maximum_relative_difference": entry["h200_capture"][
                            "maximum_relative_difference"],
                    } for entry in entries],
                })
            dataset[case_id] = {"op_info": chosen["op_info"],
                                "kernels": chosen["kernels"]}
        publish_complete_dataset(
            dataset, tuple(dataset),
            str(publication_dir / f"{family}.json"),
            str(canonical_dir / f"{family}{KERNEL_DATASET_SUFFIX}"))
        published[family] = len(dataset)

    record = {
        "schema_version": 1,
        "artifact_kind": "nvidia_h200_kernel_dataset_publication",
        "hardware_id": H200_HARDWARE_ID,
        "vllm_stack": next(iter(stacks)),
        "candidate_root": str(candidate_root),
        "sources": sources,
        "published_case_counts": published,
        "duplicate_rule": (
            "same launch identity required; smaller retained-window "
            "maximum_relative_difference, ties by campaign name"),
        "deduplicated_cases": deduplicated,
        "canonical_sha256": {
            family: hashlib.sha256((canonical_dir / (
                f"{family}{KERNEL_DATASET_SUFFIX}")).read_bytes()).hexdigest()
            for family in H200_EVALUATION_FAMILIES},
    }
    (publication_dir / "publication.json").write_text(
        json.dumps(record, indent=1), encoding="utf-8")
    return record
