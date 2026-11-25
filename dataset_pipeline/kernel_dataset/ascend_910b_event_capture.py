"""No-profiler device-Event capture plans for Ascend 910B runners."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from dataset_pipeline.kernel_dataset.ascend_910b import build_shapes
from dataset_pipeline.kernel_dataset.config import (
    canonical_json_sha256,
    load_kernel_dataset_mapping,
    get_kernel_config_path,
    get_kernel_dataset_path,
    get_kernel_script_path,
    load_pipeline_config,
    load_workload_config,
)


ASCEND_VECTOR_DIRECT_MODE = "direct"
ASCEND_VECTOR_NPU_GRAPH_MODE = "npu_graph_differential"
ASCEND_VECTOR_GRAPH_REPETITION_SELECTION_RULE = (
    "memory_bounded_unique_preallocated_calls_v1")
ASCEND_910B_L2_BYTES = 192 * 1024 * 1024
ASCEND_VECTOR_GRAPH_MEMORY_BUDGET_BYTES = 12 * 1024 * 1024 * 1024
_ASCEND_VECTOR_GRAPH_HIGHER_COUNTS = (256, 128, 64, 32, 16)


def validate_ascend_vector_capture_contract(
        selected_cases: Mapping[str, Sequence[str]], cache_eviction_mib: int,
        sample_count: int, explicit_full_vector: bool,
        vector_input_ring_mib: int = 0,
        vector_power_warmup_gemm_iterations: int = 0,
        vector_operation_repetitions_per_sample: int = 1,
        vector_event_mode: str = ASCEND_VECTOR_DIRECT_MODE) -> None:
    """Validate mutually exclusive direct, rotating-ring, and graph modes."""

    if vector_event_mode not in {
            ASCEND_VECTOR_DIRECT_MODE, ASCEND_VECTOR_NPU_GRAPH_MODE}:
        raise ValueError(
            f"unsupported Ascend Vector Event mode: {vector_event_mode!r}")
    if vector_input_ring_mib < 0:
        raise ValueError("Ascend Vector input ring must be nonnegative")
    if vector_input_ring_mib and vector_input_ring_mib < 1536:
        raise ValueError(
            "Ascend Vector input ring must be at least 1536 MiB")
    if "vector" not in selected_cases:
        if (vector_input_ring_mib or vector_power_warmup_gemm_iterations
                or vector_operation_repetitions_per_sample != 1
                or vector_event_mode != ASCEND_VECTOR_DIRECT_MODE):
            raise ValueError("Ascend input ring requires a Vector selection")
        return
    if vector_event_mode == ASCEND_VECTOR_NPU_GRAPH_MODE:
        if (vector_input_ring_mib or vector_power_warmup_gemm_iterations
                or vector_operation_repetitions_per_sample != 1):
            raise ValueError(
                "Ascend Vector NPUGraph mode cannot use eager ring, power "
                "warmup, or operation batching")
        if cache_eviction_mib != 2048:
            raise ValueError(
                "Ascend Vector NPUGraph capture requires a 2 GiB sweep")
        if explicit_full_vector and sample_count != 50:
            raise ValueError(
                "full Ascend Vector NPUGraph capture requires 50 samples")
        return
    if vector_operation_repetitions_per_sample not in {1, 32, 512, 2048, 8192}:
        raise ValueError(
            "Ascend Vector Event operation repetitions must be 1, 32, 512, "
            "2048, or 8192")
    if vector_input_ring_mib:
        if cache_eviction_mib != 0:
            raise ValueError(
                "Ascend Vector input ring cannot mix with cache eviction")
        if sample_count != 50:
            raise ValueError(
                "Ascend Vector input-ring capture requires 50 samples")
        if vector_power_warmup_gemm_iterations not in {0, 1000}:
            raise ValueError(
                "Ascend Vector device power warmup must be 0 or 1000 GEMMs")
        if (vector_operation_repetitions_per_sample != 1
                and vector_power_warmup_gemm_iterations != 1000):
            raise ValueError(
                "batched Ascend Vector Event timing requires 1000 GEMM power warmup")
        return
    if vector_power_warmup_gemm_iterations:
        raise ValueError(
            "Ascend Vector device power warmup requires an input ring")
    if vector_operation_repetitions_per_sample != 1:
        raise ValueError(
            "batched Ascend Vector Event timing requires the dense input ring")
    if cache_eviction_mib == 0:
        return
    if cache_eviction_mib not in {2048, 3072}:
        raise ValueError(
            "Ascend Vector capture requires a 2 GiB sweep; 3 GiB is "
            "permitted only for cache-plateau smoke validation")
    if explicit_full_vector and sample_count != 50:
        raise ValueError("full Ascend Vector capture requires 50 samples")


def build_ascend_vector_buffer_contract(
        attributes: Mapping[str, Any]) -> Mapping[str, int]:
    """Return unique storage bytes materialized for one graph call."""

    dtype_bytes = attributes["dtype_bytes"]
    tok = attributes["tok"]
    op = attributes["op"]
    if op == "rmsnorm":
        tensor_bytes = tok * attributes["hidden"] * dtype_bytes
        input_bytes = tensor_bytes + attributes["hidden"] * dtype_bytes
        output_bytes = tensor_bytes + tok * 4
    elif op == "add_rmsnorm":
        tensor_bytes = tok * attributes["hidden"] * dtype_bytes
        input_bytes = 2 * tensor_bytes + attributes["hidden"] * dtype_bytes
        output_bytes = tensor_bytes + tok * 4
    elif op == "rope":
        qk_bytes = (
            tok * (attributes["num_heads"] + attributes["num_kv_heads"])
            * attributes["head_dim"] * dtype_bytes)
        position_bytes = tok * 4
        cache_bytes = tok * attributes["head_dim"] * dtype_bytes
        input_bytes = qk_bytes + position_bytes + cache_bytes
        output_bytes = 0
    elif op == "silu":
        output_bytes = tok * attributes["intermediate"] * dtype_bytes
        input_bytes = 2 * output_bytes
    elif op == "elementwise":
        output_bytes = tok * attributes["hidden"] * dtype_bytes
        input_bytes = 2 * output_bytes
    elif op == "kvcache":
        key_value_input_bytes = (
            2 * tok * attributes["num_kv_heads"] * attributes["head_dim"]
            * dtype_bytes)
        block_size = 128
        num_blocks = math.ceil(tok / block_size)
        key_value_cache_bytes = (
            2 * num_blocks * block_size * attributes["num_kv_heads"]
            * attributes["head_dim"] * dtype_bytes)
        input_bytes = key_value_input_bytes + key_value_cache_bytes + tok * 4
        output_bytes = 0
    else:
        raise ValueError(f"unsupported Ascend Vector op: {op!r}")
    return {
        "per_call_distinct_input_bytes": input_bytes,
        "per_call_distinct_output_storage_bytes": output_bytes,
        "per_call_distinct_working_set_bytes": input_bytes + output_bytes,
    }


def build_ascend_vector_graph_contract(
        attributes: Mapping[str, Any]) -> Mapping[str, Any]:
    """Choose a memory-bounded pair of unique-call NPUGraph sizes."""

    buffers = build_ascend_vector_buffer_contract(attributes)
    per_call_bytes = buffers["per_call_distinct_working_set_bytes"]
    if per_call_bytes <= 0:
        raise ValueError("Ascend Vector graph working set must be positive")
    fitting_counts = [
        count for count in _ASCEND_VECTOR_GRAPH_HIGHER_COUNTS
        if count * per_call_bytes <= ASCEND_VECTOR_GRAPH_MEMORY_BUDGET_BYTES
    ]
    if not fitting_counts:
        raise ValueError(
            "Ascend Vector case exceeds the per-case NPUGraph memory budget")
    higher_count = max(fitting_counts)
    lower_count = max(1, higher_count // 16)
    return {
        **buffers,
        "hardware_l2_bytes": ASCEND_910B_L2_BYTES,
        "memory_budget_bytes": ASCEND_VECTOR_GRAPH_MEMORY_BUDGET_BYTES,
        "lower_repetition_count": lower_count,
        "higher_repetition_count": higher_count,
        "operation_count_delta": higher_count - lower_count,
        "higher_replay_working_set_bytes": higher_count * per_call_bytes,
        "repetition_selection_rule": (
            ASCEND_VECTOR_GRAPH_REPETITION_SELECTION_RULE),
    }


def _event_plan_fields(
        family: str, attributes: Mapping[str, Any],
        vector_event_mode: str) -> tuple[str, int | None,
                                         Mapping[str, Any] | None]:
    if family == "vector":
        if vector_event_mode == ASCEND_VECTOR_DIRECT_MODE:
            return ASCEND_VECTOR_DIRECT_MODE, None, None
        if vector_event_mode == ASCEND_VECTOR_NPU_GRAPH_MODE:
            contract = build_ascend_vector_graph_contract(attributes)
            return (
                ASCEND_VECTOR_NPU_GRAPH_MODE,
                contract["higher_replay_working_set_bytes"], contract)
        raise ValueError(
            f"unsupported Ascend Vector Event mode: {vector_event_mode!r}")
    if family in {"gemm", "flash_attn", "fused_moe"}:
        return ASCEND_VECTOR_DIRECT_MODE, None, None
    raise ValueError(f"unsupported Ascend 910B family: {family!r}")






def _remove_required_option(command: Sequence[str], option: str) -> tuple[list[str], str]:
    indexes = [index for index, value in enumerate(command) if value == option]
    if len(indexes) != 1 or indexes[0] + 1 >= len(command):
        raise ValueError(
            f"runner command must contain one valued {option}: {command!r}")
    index = indexes[0]
    return (list(command[:index]) + list(command[index + 2:]),
            command[index + 1])


def _support_decision(family: str, command: Sequence[str]) -> tuple[bool, str | None]:
    if family not in {"gemm", "flash_attn", "fused_moe", "vector"}:
        raise ValueError(f"unsupported Ascend 910B family: {family!r}")
    if not command:
        raise ValueError("Ascend runner command must not be empty")
    return True, None


def _validate_attributes(
        family: str, case_id: str, shape_attributes: Mapping[str, Any],
        canonical_attributes: Mapping[str, Any]) -> None:
    if family != "fused_moe":
        if shape_attributes != canonical_attributes:
            raise ValueError(
                f"Ascend workload expansion disagrees with Attr: {case_id}")
        return
    expected = dict(canonical_attributes)
    token_counts = expected.pop("tok_per_expert")
    if shape_attributes != expected:
        raise ValueError(
            f"Ascend MoE workload expansion disagrees with Attr: {case_id}")
    if (not isinstance(token_counts, list) or not token_counts
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 0 for value in token_counts)):
        raise ValueError(f"Ascend MoE routing counts are invalid: {case_id}")


def build_ascend_910b_event_plans(
        *, config_path: Path,
        selected_case_ids: Mapping[str, Sequence[str]],
        vector_event_mode: str = ASCEND_VECTOR_DIRECT_MODE,
        ) -> list[Mapping[str, Any]]:
    """Build exact runner commands for the fixed Ascend sample."""

    config = load_pipeline_config(str(config_path))
    if config["hardware_id"] != "ascend_910b":
        raise ValueError("910B Event capture requires ascend_910b config")
    selected_families = set(selected_case_ids)
    allowed_families = {"gemm", "flash_attn", "fused_moe", "vector"}
    if not selected_families or not selected_families <= allowed_families:
        raise ValueError(
            "910B Event selection must contain known non-empty families")
    plans = []
    for family in ("gemm", "flash_attn", "fused_moe", "vector"):
        if family not in selected_case_ids:
            continue
        requested = list(selected_case_ids[family])
        if (not requested or len(requested) != len(set(requested))
                or any(not isinstance(case_id, str) or not case_id
                       for case_id in requested)):
            raise ValueError(
                f"910B selected {family} case IDs must be unique and non-empty")
        workload_config = load_workload_config(
            get_kernel_config_path(config, family), family)
        shapes = build_shapes(family, workload_config)
        shape_index = {shape["filename"]: shape for shape in shapes}
        if len(shape_index) != len(shapes):
            raise ValueError(f"910B {family} workload grid has duplicate IDs")
        dataset_path = Path(get_kernel_dataset_path(config, family))
        dataset = load_kernel_dataset_mapping(dataset_path)
        runner_path = Path(get_kernel_script_path(config, family)).resolve()
        if not runner_path.is_file():
            raise FileNotFoundError(f"910B runner is absent: {runner_path}")
        for case_id in requested:
            if case_id not in shape_index:
                raise KeyError(
                    f"selected case is absent from 910B workload grid: {case_id}")
            if case_id not in dataset:
                raise KeyError(
                    f"selected case is absent from 910B dataset: {case_id}")
            shape = shape_index[case_id]
            attributes = dataset[case_id]["op_info"]["Attr"]
            _validate_attributes(family, case_id, shape["attr"], attributes)
            command = [sys.executable, str(runner_path), *shape["script_args"]]
            workload_sidecar = None
            if family == "flash_attn":
                command, q_lens_json = _remove_required_option(
                    command, "--q-lens-json")
                command, kv_lens_json = _remove_required_option(
                    command, "--kv-lens-json")
                workload_sidecar = {
                    "q_lens": json.loads(q_lens_json),
                    "kv_lens": json.loads(kv_lens_json),
                }
            elif family == "fused_moe":
                command += [
                    "--expected-tok-per-expert-json",
                    json.dumps(attributes["tok_per_expert"], separators=(",", ":")),
                ]
            elif family in {"gemm", "vector"}:
                if family == "vector":
                    command += [
                        "--event-vector-mode",
                        ASCEND_VECTOR_DIRECT_MODE,
                    ]
            else:
                raise AssertionError(f"unhandled Ascend family: {family}")
            supported, unsupported_reason = _support_decision(family, command)
            (event_measurement_mode, estimated_rotation_bytes,
             graph_contract) = _event_plan_fields(
                 family, attributes, vector_event_mode)
            if family == "vector":
                command[command.index("--event-vector-mode") + 1] = (
                    event_measurement_mode)
            plans.append({
                "hardware_id": "ascend_910b",
                "family": family,
                "case_id": case_id,
                "runner_path": runner_path,
                "command": command,
                "canonical_attributes": attributes,
                "canonical_attributes_sha256": canonical_json_sha256(
                    attributes),
                "workload_sidecar": workload_sidecar,
                "event_measurement_mode": event_measurement_mode,
                "estimated_rotation_bytes": estimated_rotation_bytes,
                "graph_contract": graph_contract,
                "supported": supported,
                "unsupported_reason": unsupported_reason,
            })
    return plans


def capture_ascend_910b_supported_case(
        *, plan: Mapping[str, Any], output: Path, baseline_commit: str,
        source_snapshot_sha256: str, timer_sanity_sha256: str,
        warmup_iterations: int, sample_count: int,
        cache_eviction_mib: int = 0,
        vector_input_ring_mib: int = 0,
        vector_power_warmup_gemm_iterations: int = 0) -> Mapping[str, Any]:
    """Execute one supported plan and require a structured Event artifact."""

    if plan["supported"] is not True:
        raise ValueError(f"cannot execute unsupported plan: {plan['case_id']}")
    command = list(plan["command"])
    if plan["workload_sidecar"] is not None:
        workload_path = output.with_suffix(".workload.json")
        workload_path.parent.mkdir(parents=True, exist_ok=True)
        with workload_path.open("w", encoding="utf-8") as stream:
            json.dump(
                plan["workload_sidecar"], stream, indent=2,
                ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        command += ["--workload-json", str(workload_path)]
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
    if vector_input_ring_mib:
        if plan["family"] != "vector":
            raise ValueError("Ascend input ring is valid only for Vector")
        if plan["canonical_attributes"]["op"] not in {"rmsnorm", "silu"}:
            raise ValueError(
                "Ascend input ring supports only RMSNorm and SiLU Vector cases")
        if cache_eviction_mib != 0:
            raise ValueError("Ascend input ring cannot mix with cache eviction")
        command += ["--event-input-ring-mib", str(vector_input_ring_mib)]
        if vector_power_warmup_gemm_iterations:
            command += [
                "--event-power-warmup-gemm-iterations",
                str(vector_power_warmup_gemm_iterations),
            ]
    elif vector_power_warmup_gemm_iterations:
        raise ValueError("Ascend device power warmup requires an input ring")
    elif cache_eviction_mib:
        command += ["--event-cache-eviction-mib", str(cache_eviction_mib)]
    graph_contract = plan["graph_contract"]
    if graph_contract is not None:
        command += [
            "--event-graph-repetitions",
            str(graph_contract["lower_repetition_count"]),
            str(graph_contract["higher_repetition_count"]),
            "--event-graph-buffer-count",
            str(graph_contract["higher_repetition_count"]),
        ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"910B Event runner failed for {plan['case_id']}: "
            f"stderr={completed.stderr!r}")
    if not output.is_file():
        raise RuntimeError(
            f"910B Event runner returned success without output: "
            f"{plan['case_id']}")
    return {
        "case_id": plan["case_id"],
        "status": "measured",
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output": str(output),
    }
