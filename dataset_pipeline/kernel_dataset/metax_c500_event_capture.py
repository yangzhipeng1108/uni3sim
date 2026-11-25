"""No-profiler device-Event capture plans for MetaX C500 runners."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from dataset_pipeline.kernel_dataset.config import (
    canonical_json_sha256,
    load_kernel_dataset_mapping,
    get_kernel_config_path,
    get_kernel_dataset_path,
    get_kernel_script_path,
    load_pipeline_config,
    load_workload_config,
)
from dataset_pipeline.kernel_dataset.metax_c500 import build_shapes


C500_L2_BYTES = 8 * 1024 * 1024
MINIMUM_DISTINCT_INPUT_MIB = 64


def build_vector_buffer_contract(
        attributes: Mapping[str, Any]) -> Mapping[str, int]:
    dtype_bytes = attributes["dtype_bytes"]
    tok = attributes["tok"]
    op = attributes["op"]
    if op == "rmsnorm":
        tensor_bytes = tok * attributes["hidden"] * dtype_bytes
        input_bytes = tensor_bytes + attributes["hidden"] * dtype_bytes
        output_bytes = tensor_bytes
        working_set_bytes = input_bytes + output_bytes
    elif op == "add_rmsnorm":
        tensor_bytes = tok * attributes["hidden"] * dtype_bytes
        input_bytes = 2 * tensor_bytes + attributes["hidden"] * dtype_bytes
        output_bytes = 2 * tensor_bytes
        working_set_bytes = input_bytes
    elif op == "rope":
        qk_bytes = (
            tok * (attributes["num_heads"] + attributes["num_kv_heads"])
            * attributes["head_dim"] * dtype_bytes)
        input_bytes = (
            qk_bytes + tok * 8
            + tok * attributes["head_dim"] * dtype_bytes)
        output_bytes = qk_bytes
        working_set_bytes = input_bytes
    elif op == "silu":
        output_bytes = tok * attributes["intermediate"] * dtype_bytes
        input_bytes = 2 * output_bytes
        working_set_bytes = input_bytes + output_bytes
    elif op == "elementwise":
        output_bytes = tok * attributes["hidden"] * dtype_bytes
        input_bytes = 2 * output_bytes
        working_set_bytes = input_bytes + output_bytes
    elif op == "kvcache":
        key_value_bytes = (
            2 * tok * attributes["num_kv_heads"] * attributes["head_dim"]
            * dtype_bytes)
        input_bytes = key_value_bytes + tok * 8 + 2 * 4
        output_bytes = key_value_bytes
        working_set_bytes = input_bytes + output_bytes
    else:
        raise ValueError(f"unsupported Vector op for distinct inputs: {op!r}")
    return {
        "per_call_distinct_input_bytes": input_bytes,
        "per_call_distinct_output_storage_bytes": output_bytes,
        "per_call_distinct_working_set_bytes": working_set_bytes,
    }


def build_vector_distinct_input_contract(
        attributes: Mapping[str, Any], minimum_distinct_input_mib: int,
        ) -> Mapping[str, Any]:
    if (isinstance(minimum_distinct_input_mib, bool)
            or not isinstance(minimum_distinct_input_mib, int)):
        raise TypeError("minimum_distinct_input_mib must be an integer")
    if minimum_distinct_input_mib < MINIMUM_DISTINCT_INPUT_MIB:
        raise ValueError(
            "minimum_distinct_input_mib must be at least 64 (8x C500 L2)")
    if minimum_distinct_input_mib % 8 != 0:
        raise ValueError(
            "minimum_distinct_input_mib must be an integer multiple of the "
            "8 MiB C500 L2")
    minimum_bytes = minimum_distinct_input_mib * 1024 * 1024
    per_call_bytes = build_vector_buffer_contract(attributes)[
        "per_call_distinct_input_bytes"]
    lower_count = math.ceil(minimum_bytes / per_call_bytes)
    higher_count = 2 * lower_count
    return {
        "hardware_l2_bytes": C500_L2_BYTES,
        "required_minimum_l2_multiple": 8,
        "configured_minimum_l2_multiple": minimum_distinct_input_mib // 8,
        "configured_minimum_bytes": minimum_bytes,
        "per_call_bytes": per_call_bytes,
        "lower_replay_bytes": lower_count * per_call_bytes,
        "higher_replay_bytes": higher_count * per_call_bytes,
        "delta_replay_bytes": (higher_count - lower_count) * per_call_bytes,
        "repetition_selection_rule": (
            "lower=ceil(configured_minimum_bytes/per_call_bytes);"
            "higher=2*lower"),
        "repetition_counts": (lower_count, higher_count),
    }






def _validate_attributes(family, case_id, shape_attributes,
                         canonical_attributes):
    if family in {"gemm", "vector"}:
        if shape_attributes != canonical_attributes:
            raise ValueError(
                f"C500 workload expansion disagrees with Attr: {case_id}")
        return
    if family == "flash_attn":
        expected = dict(canonical_attributes)
        del expected["SP"]
        actual = dict(shape_attributes)
        del actual["SP"]
        if actual != expected:
            raise ValueError(
                f"C500 FlashAttention expansion disagrees with Attr: {case_id}")
        return
    if family == "fused_moe":
        expected = dict(canonical_attributes)
        del expected["moe_tiling"]
        token_counts = expected.pop("tok_per_expert")
        if shape_attributes != expected:
            raise ValueError(
                f"C500 FusedMoE expansion disagrees with Attr: {case_id}")
        if (not isinstance(token_counts, list) or not token_counts
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value < 0 for value in token_counts)):
            raise ValueError(
                f"C500 FusedMoE routing counts are invalid: {case_id}")
        return
    raise ValueError(f"unsupported C500 family: {family!r}")


def build_metax_c500_event_plans(
        *, config_path: Path,
        selected_case_ids: Mapping[str, Sequence[str]]) -> list[Mapping[str, Any]]:
    """Build exact no-profiler runner commands for the fixed C500 sample."""

    config = load_pipeline_config(str(config_path))
    if config["hardware_id"] != "metax_c500":
        raise ValueError("C500 Event capture requires metax_c500 config")
    families = ("gemm", "flash_attn", "fused_moe", "vector")
    if set(selected_case_ids) != set(families):
        raise ValueError("C500 Event selection must contain all four families")
    plans = []
    for family in families:
        requested = list(selected_case_ids[family])
        if (not requested or len(requested) != len(set(requested))
                or any(not isinstance(case_id, str) or not case_id
                       for case_id in requested)):
            raise ValueError(
                f"C500 selected {family} case IDs must be unique and non-empty")
        workload_config = load_workload_config(
            get_kernel_config_path(config, family), family)
        shapes = build_shapes(family, workload_config)
        shape_index = {shape["filename"]: shape for shape in shapes}
        if len(shape_index) != len(shapes):
            raise ValueError(f"C500 {family} workload grid has duplicate IDs")
        dataset = load_kernel_dataset_mapping(Path(get_kernel_dataset_path(config, family)))
        runner_path = Path(get_kernel_script_path(config, family)).resolve()
        for case_id in requested:
            if case_id not in shape_index:
                raise KeyError(
                    f"selected case is absent from C500 workload grid: {case_id}")
            if case_id not in dataset:
                raise KeyError(
                    f"selected case is absent from C500 dataset: {case_id}")
            shape = shape_index[case_id]
            attributes = dataset[case_id]["op_info"]["Attr"]
            _validate_attributes(family, case_id, shape["attr"], attributes)
            command = [sys.executable, str(runner_path), *shape["script_args"]]
            if family == "flash_attn":
                requested_splits = shape["attr"]["SP_REQUESTED"]
                if requested_splits is not None:
                    command += ["--num_splits", str(requested_splits)]
                command += ["--profiler_wait", "0"]
            elif family in {"fused_moe", "vector"}:
                command += ["--profiler_wait", "0"]
            elif family == "gemm":
                pass
            else:
                raise AssertionError(f"unhandled C500 family: {family}")
            plans.append({
                "hardware_id": "metax_c500",
                "family": family,
                "case_id": case_id,
                "runner_path": runner_path,
                "command": command,
                "canonical_attributes": attributes,
                "canonical_attributes_sha256": canonical_json_sha256(
                    attributes),
            })
    return plans


def build_metax_c500_family_event_plans(
        *, config_path: Path, family: str,
        selected_case_ids: Sequence[str]) -> list[Mapping[str, Any]]:
    """Build exact no-profiler runner commands for one C500 family."""

    if family not in {"gemm", "flash_attn", "fused_moe", "vector"}:
        raise ValueError(f"unsupported C500 family: {family!r}")
    config = load_pipeline_config(str(config_path))
    if config["hardware_id"] != "metax_c500":
        raise ValueError("C500 Event capture requires metax_c500 config")
    requested = list(selected_case_ids)
    if (not requested or len(requested) != len(set(requested))
            or any(not isinstance(case_id, str) or not case_id
                   for case_id in requested)):
        raise ValueError(
            f"C500 selected {family} case IDs must be unique and non-empty")
    workload_config = load_workload_config(
        get_kernel_config_path(config, family), family)
    shapes = build_shapes(family, workload_config)
    shape_index = {shape["filename"]: shape for shape in shapes}
    if len(shape_index) != len(shapes):
        raise ValueError(f"C500 {family} workload grid has duplicate IDs")
    dataset = load_kernel_dataset_mapping(Path(get_kernel_dataset_path(config, family)))
    runner_path = Path(get_kernel_script_path(config, family)).resolve()
    plans = []
    for case_id in requested:
        if case_id not in shape_index:
            raise KeyError(
                f"selected case is absent from C500 workload grid: {case_id}")
        if case_id not in dataset:
            raise KeyError(
                f"selected case is absent from C500 dataset: {case_id}")
        shape = shape_index[case_id]
        attributes = dataset[case_id]["op_info"]["Attr"]
        _validate_attributes(family, case_id, shape["attr"], attributes)
        command = [sys.executable, str(runner_path), *shape["script_args"]]
        if family == "flash_attn":
            requested_splits = shape["attr"]["SP_REQUESTED"]
            if requested_splits is not None:
                command += ["--num_splits", str(requested_splits)]
            command += ["--profiler_wait", "0"]
        elif family in {"fused_moe", "vector"}:
            command += ["--profiler_wait", "0"]
        elif family == "gemm":
            pass
        else:
            raise AssertionError(f"unhandled C500 family: {family}")
        plans.append({
            "hardware_id": "metax_c500",
            "family": family,
            "case_id": case_id,
            "runner_path": runner_path,
            "command": command,
            "canonical_attributes": attributes,
            "canonical_attributes_sha256": canonical_json_sha256(
                attributes),
            "canonical_duration_ns": dataset[case_id]["op_info"]
            ["evaluation"]["timing"]["total_duration_ns"],
        })
    return plans


def capture_metax_c500_case(
        *, plan: Mapping[str, Any], output: Path, baseline_commit: str,
        source_snapshot_sha256: str, timer_sanity_sha256: str,
        warmup_iterations: int, sample_count: int,
        graph_repetitions: tuple[int, int] | None = None,
        buffer_count: int = 1, cache_eviction_mib: int = 0,
        minimum_distinct_input_mib: int | None = None,
        ) -> Mapping[str, Any]:
    """Execute one exact C500 case and require a measured Event artifact."""

    if plan["family"] == "fused_moe" and graph_repetitions is None:
        raise ValueError(
            "C500 FusedMoE Event capture requires graph mode with the "
            "preprepared GPU sequence")
    if isinstance(buffer_count, bool) or not isinstance(buffer_count, int):
        raise TypeError("buffer_count must be an integer")
    if buffer_count <= 0:
        raise ValueError("buffer_count must be positive")
    if buffer_count != 1:
        if plan["family"] != "vector":
            raise ValueError("rotating buffers are supported only for vector")
        if graph_repetitions is None:
            raise ValueError("rotating buffers require graph repetitions")
        if (cache_eviction_mib == 0
                and min(graph_repetitions) < buffer_count):
            raise ValueError(
                "every rotating graph must traverse the complete buffer "
                "cycle: each repetition count must be at least buffer_count")
        if (cache_eviction_mib != 0
                and max(graph_repetitions) > buffer_count):
            raise ValueError(
                "sweep-assisted rotating graphs must not reuse an operator "
                "buffer within one replay: buffer_count must be at least "
                "each repetition count")
    if minimum_distinct_input_mib is not None:
        if (isinstance(minimum_distinct_input_mib, bool)
                or not isinstance(minimum_distinct_input_mib, int)):
            raise TypeError("minimum_distinct_input_mib must be an integer")
        if minimum_distinct_input_mib < 64:
            raise ValueError(
                "minimum_distinct_input_mib must be at least 64 "
                "(8x C500 L2)")
        if minimum_distinct_input_mib % 8 != 0:
            raise ValueError(
                "minimum_distinct_input_mib must be an integer multiple of "
                "the 8 MiB C500 L2")
        if (plan["family"] != "vector" or graph_repetitions is None
                or buffer_count != graph_repetitions[1]):
            raise ValueError(
                "minimum distinct input requires Vector graph mode with one "
                "unique buffer per higher-replay call")
        if cache_eviction_mib != 256:
            raise ValueError(
                "C500 cold Vector differential capture requires a 256 MiB "
                "queued sweep")
    command = list(plan["command"])
    command += [
        "--event-output", str(output),
        "--event-case-id", plan["case_id"],
        "--event-baseline-commit", baseline_commit,
        "--event-source-snapshot-sha256", source_snapshot_sha256,
        "--event-timer-sanity-sha256", timer_sanity_sha256,
        "--event-canonical-attributes-sha256",
        plan["canonical_attributes_sha256"],
        "--event-warmup", str(warmup_iterations),
        "--event-samples", str(sample_count),
    ]
    if graph_repetitions is not None:
        if plan["family"] not in {
                "gemm", "flash_attn", "fused_moe", "vector"}:
            raise ValueError(
                "graph_repetitions is unsupported for this family: "
                f"{plan['family']}")
        command += [
            "--event-graph-repetitions",
            str(graph_repetitions[0]), str(graph_repetitions[1]),
        ]
        if plan["family"] == "fused_moe":
            command += [
                "--tok-per-expert-json",
                json.dumps(plan["canonical_attributes"]["tok_per_expert"],
                           separators=(",", ":")),
            ]
        if buffer_count != 1:
            command += ["--event-buffer-count", str(buffer_count)]
        if minimum_distinct_input_mib is not None:
            command += [
                "--event-minimum-distinct-input-mib",
                str(minimum_distinct_input_mib),
            ]
    if cache_eviction_mib:
        command += ["--event-cache-eviction-mib", str(cache_eviction_mib)]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"C500 Event runner failed for {plan['case_id']}: "
            f"stderr={completed.stderr!r}")
    if not output.is_file():
        raise RuntimeError(
            f"C500 Event runner returned success without output: "
            f"{plan['case_id']}")
    return {
        "case_id": plan["case_id"],
        "status": "measured",
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output": str(output),
    }
