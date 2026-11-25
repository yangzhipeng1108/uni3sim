"""Compare exact Dataset and Serving predictor inputs for three hardware."""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from collections.abc import Mapping

from evaluation.kernel import evaluation_runner
from evaluation.kernel.dataset import (
    get_hardware_config_path,
    load_kernel_dataset,
)
from evaluation.serving.metrics import (
    compact_step_alignment,
    sharegpt_compact_evaluation as compact,
)
from evaluation.serving.metrics.diagnostics import (
    metax_c500_kernel_coverage as coverage,
)
from evaluation.serving.simulation import execution_artifact
from src.kernel import kernel_base
from src.kernel.hardware_spec import HWSpec
from src.serving.deployment.deployment_config import (
    DeploymentConfig,
    WorkerPoolConfig,
)
from src.serving.deployment.deployment_plan import (
    build_deployment_plan,
)
from src.serving.execution import ServingExecutionContext
from src.serving.kernel_lowering import lower_workload
from src.serving.kernel_lowering import kernel_predict
from src.serving.workload.model_config import load_model_config
from src.serving.workload.runtime_inputs import (
    ExpertRoutingInput,
    RuntimeInputs,
)
from src.serving.workload.workload_graph import (
    WorkloadGraph,
    WorkloadKind,
    WorkloadNode,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

HARDWARE_IDS = ("nvidia_a800", "metax_c500", "ascend_910b")
DATASET_REPLAY_CASES = {
    "gemm": "GEMM_B1_M1_K4096_N4096",
    "attention": "FLASHATTN_B1_S512_HQ32_HKV8_D128_C0",
    "vector": "VECTOR_SILU_T1_I12288_B1",
    "moe": "FUSEDMOE_TOK1_E8_TOPK2_H4096_INT14336",
}
DATASET_FAMILIES = {
    "gemm": "gemm",
    "attention": "flash_attn",
    "vector": "vector",
    "moe": "fused_moe",
}


def json_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: json_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: json_value(getattr(value, field.name))
            for field in fields(value)
        }
    raise TypeError(f"cannot serialize {type(value).__name__}")


def kernel_spec_value(spec):
    return json_value(spec)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diff(left, right, path=""):
    differences = []
    if type(left) is not type(right):
        return [{"path": path, "dataset": left, "serving": right}]
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                differences.append({
                    "path": child,
                    "dataset": left[key] if key in left else None,
                    "serving": right[key] if key in right else None,
                })
            else:
                differences.extend(diff(left[key], right[key], child))
    elif isinstance(left, list):
        if len(left) != len(right):
            differences.append({
                "path": f"{path}.length",
                "dataset": len(left), "serving": len(right),
            })
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(diff(
                left_item, right_item, f"{path}[{index}]"))
    elif left != right:
        differences.append({"path": path, "dataset": left,
                            "serving": right})
    return differences


def select_exact_row(report):
    matches = [
        row for row in report["logical_ops"]
        if row["canonical_mapping"]["status"] == "exact"
        and len(row["physical_key_ids"]) == 1
    ]
    if not matches:
        raise ValueError("coverage report has no exact single-leaf match")
    return max(matches, key=lambda row: row["predicted_device_time_s"])


def capture_dataset_specs(hardware_id, family, case_id):
    dataset = load_kernel_dataset(hardware_id, [family])
    entry = dataset[case_id]
    captured = []
    original = evaluation_runner.predict_production

    def record(spec, hardware, method, **kwargs):
        prediction = kernel_base.predict(spec, hardware, method, **kwargs)
        if method == "midlayer":
            captured.append((spec, hardware, prediction))
        return prediction

    evaluation_runner.predict_production = record
    try:
        result = evaluation_runner._process_op(
            case_id, entry, get_hardware_config_path(hardware_id),
            False, True)
    finally:
        evaluation_runner.predict_production = original
    if not captured:
        raise ValueError(f"Dataset emitted no predictor calls for {case_id}")
    if not isinstance(result[3], tuple):
        raise ValueError(f"Dataset Forward is unsupported for {case_id}")
    forward_total_ns = result[3][0]
    captured_total_ns = sum(item[2].t_total_ns for item in captured)
    if not math.isclose(
            captured_total_ns, forward_total_ns,
            rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("captured Dataset prediction disagrees with evaluator")
    return tuple(captured)


def capture_dataset(hardware_id, family, case_id):
    captured = capture_dataset_specs(hardware_id, family, case_id)
    if len(captured) != 1:
        raise ValueError(
            f"expected one Dataset predictor call for {case_id}, "
            f"got {len(captured)}")
    return captured[0]


def matching_routing(hardware_id, attributes):
    target = tuple(attributes["tok_per_expert"])
    tokens = attributes["TOK"]
    experts = attributes["E"]
    topk = attributes["TOPK"]
    for seed in range(10_000):
        resolution = ExpertRoutingInput.seeded_assumption(
            "dataset-serving-input-equivalence", seed).resolve(
                step_id="equivalence",
                layer_index=0,
                moe_invocation_id="equivalence:layer-0:moe",
                schedule_step_sha256="0" * 64,
                tokens=tokens,
                num_experts=experts,
                topk=topk,
                model_config_sha256="1" * 64,
                pool_name="main",
                tensor_parallel_size=1,
                expert_parallel_size=1,
                expert_rank_group=(0,),
                expert_placement_strategy="linear",
            )
        if resolution.tokens_per_expert == target:
            return seed, resolution
    raise ValueError(
        f"no deterministic Serving routing input reproduced {target!r} "
        f"for {hardware_id}")


def dataset_workload_node(hardware_id, family, entry):
    attributes = entry["op_info"]["Attr"]
    if family == "gemm":
        dtype = attributes["dtype"]
        return WorkloadNode(
            "equivalence", WorkloadKind.LINEAR, "main",
            {
                "b": attributes["B"],
                "m": attributes["M"],
                "n": attributes["N"],
                "k": attributes["K"],
            },
            dtype=dtype,
            variant="gemm",
            rank=0,
            attributes={
                "a_layout": "row_major",
                "b_layout": "row_major",
                "c_layout": "row_major",
                "epilogue": "none",
                "expert_parallel_size": 1,
                "has_bias": False,
                "launch_policy": "injected_launch",
                "position": "dataset_equivalence",
                "tensor_parallel_size": 1,
                "trans_a": False,
                "trans_b": True,
            },
        ), None
    if family == "attention":
        q_lens = tuple(attributes["Q_LENS"])
        kv_lens = tuple(attributes["KV_LENS"])
        split_count = attributes["SP"]
        if split_count is None:
            split_count = 1
        return WorkloadNode(
            "equivalence", WorkloadKind.FLASH_ATTN, "main",
            {
                "batch": attributes["B"],
                "q_len": attributes["Q_LEN"],
                "kv_len": attributes["KV_LEN"],
                "head_dim": attributes["D"],
                "num_query_heads": attributes["HQ"],
                "num_kv_heads": attributes["HKV"],
            },
            dtype="bf16",
            variant=("decode" if attributes["Q_LEN"] == 1 else "prefill"),
            rank=0,
            attributes={
                "causal": attributes["causal"],
                "fa_eff_mfrac": 1.0,
                "fa_ksteps": None,
                "kv_dist": attributes["kv_dist"],
                "kv_layout": attributes["kv_layout"],
                "kv_lens": kv_lens,
                "launch_policy": "injected_launch",
                "num_kv_splits": split_count,
                "q_lens": q_lens,
                "split_policy": attributes["split_policy"],
            },
        ), None
    if family == "vector":
        return WorkloadNode(
            "equivalence", WorkloadKind.ACTIVATION, "main",
            {
                "tok": attributes["tok"],
                "intermediate": attributes["intermediate"],
                "elements": attributes["element_count"],
            },
            dtype="bf16",
            variant="silu",
            rank=0,
            attributes={"launch_policy": "injected_launch"},
        ), None
    if family == "moe":
        seed, routing = matching_routing(hardware_id, attributes)
        experts = attributes["E"]
        counts = routing.tokens_per_expert
        return WorkloadNode(
            "equivalence", WorkloadKind.FUSED_MOE, "main",
            {
                "tokens": attributes["TOK"],
                "experts": experts,
                "topk": attributes["TOPK"],
                "hidden": attributes["H"],
                "intermediate": attributes["INT"],
            },
            dtype="bf16",
            variant="full",
            rank=0,
            attributes={
                "expert_parallel_size": 1,
                "expert_placement_strategy": "linear",
                "expert_rank": 0,
                "launch_policy": "injected_launch",
                "local_tokens_per_expert": counts,
                "moe_invocation_id": "equivalence:layer-0:moe",
                "output_is_reduced": False,
                "owned_experts": tuple(range(experts)),
                "routing_provenance": dict(routing.routing_provenance),
                "tokens_per_expert": counts,
            },
        ), seed
    raise ValueError(f"unknown Dataset replay family: {family!r}")


def capture_dataset_via_serving(hardware_id, family, entry):
    node, routing_seed = dataset_workload_node(hardware_id, family, entry)
    hardware = HWSpec(REPOSITORY_ROOT / f"configs/hardware/{hardware_id}.yaml")
    deployment = build_deployment_plan(
        DeploymentConfig(
            "colocated",
            (WorkerPoolConfig("main", "colocated", hardware_id),),
        ),
        num_layers=1,
        num_experts=(node.shape["experts"] if "experts" in node.shape else 0),
    )
    plan = lower_workload(
        WorkloadGraph((node,)), deployment, {hardware_id: hardware})
    captured = tuple(
        (
            call.kernel_spec,
            hardware,
            kernel_base.predict(call.kernel_spec, hardware, "midlayer"),
        )
        for call in plan.calls
    )
    return captured, routing_seed


def sequence_comparison(dataset_calls, serving_calls):
    dataset_specs = [kernel_spec_value(item[0]) for item in dataset_calls]
    serving_specs = [kernel_spec_value(item[0]) for item in serving_calls]
    dataset_predictions = [item[2].t_total_ns for item in dataset_calls]
    serving_predictions = [item[2].t_total_ns for item in serving_calls]
    first_difference = None
    for index in range(max(len(dataset_specs), len(serving_specs))):
        if index >= len(dataset_specs) or index >= len(serving_specs):
            first_difference = {
                "path": f"calls[{index}]",
                "dataset": (
                    None if index >= len(dataset_specs)
                    else dataset_specs[index]),
                "serving": (
                    None if index >= len(serving_specs)
                    else serving_specs[index]),
            }
            break
        differences = diff(
            dataset_specs[index], serving_specs[index], f"calls[{index}]")
        if differences:
            first_difference = differences[0]
            break
    dataset_total = sum(dataset_predictions)
    serving_total = sum(serving_predictions)
    return {
        "dataset_call_count": len(dataset_calls),
        "serving_call_count": len(serving_calls),
        "kernel_spec_sequence_equal": dataset_specs == serving_specs,
        "first_kernel_spec_difference": first_difference,
        "dataset_prediction_ns_by_call": dataset_predictions,
        "serving_prediction_ns_by_call": serving_predictions,
        "dataset_prediction_total_ns": dataset_total,
        "serving_prediction_total_ns": serving_total,
        "prediction_delta_ns": serving_total - dataset_total,
        "prediction_relative_delta_pct": (
            (serving_total - dataset_total) / dataset_total * 100.0),
        "prediction_equal": math.isclose(
            dataset_total, serving_total, rel_tol=1e-12, abs_tol=1e-9),
        "dataset_kernel_specs": dataset_specs,
        "serving_kernel_specs": serving_specs,
    }


def dataset_replay_rows():
    rows = []
    for hardware_id in HARDWARE_IDS:
        for family, case_id in DATASET_REPLAY_CASES.items():
            dataset_family = DATASET_FAMILIES[family]
            dataset = load_kernel_dataset(hardware_id, [dataset_family])
            entry = dataset[case_id]
            dataset_calls = capture_dataset_specs(
                hardware_id, dataset_family, case_id)
            try:
                serving_calls, routing_seed = capture_dataset_via_serving(
                    hardware_id, family, entry)
            except (KeyError, NotImplementedError) as error:
                rows.append({
                    "hardware_id": hardware_id,
                    "family": family,
                    "dataset_case_id": case_id,
                    "status": "unsupported",
                    "reason": f"{type(error).__name__}: {error}",
                    "dataset_prediction_total_ns": sum(
                        item[2].t_total_ns for item in dataset_calls),
                })
                continue
            comparison = sequence_comparison(dataset_calls, serving_calls)
            rows.append({
                "hardware_id": hardware_id,
                "family": family,
                "dataset_case_id": case_id,
                "status": "compared",
                "routing_seed": routing_seed,
                **comparison,
            })
    return rows


def capture_serving(config, cell, case, target_physical_id):
    model = load_model_config(str(REPOSITORY_ROOT / cell.model.simulation_model_path))
    hardware = HWSpec(REPOSITORY_ROOT / cell.hardware.hardware_config_path)
    prediction_config = config.prediction.values
    deployment = compact_step_alignment.build_deployment(
        prediction_config, model, hardware.hardware_id)
    runtime_inputs = RuntimeInputs(
        expert_routing=compact._expected_expert_routing(REPOSITORY_ROOT, cell),
        attention_kv_layout=prediction_config["attention_kv_layout"],
        attention_kv_distribution=prediction_config[
            "attention_kv_distribution"
        ],
        graph_execution=cell.hardware.graph_execution,
    )
    execution = ServingExecutionContext(
        model, deployment, {hardware.hardware_id: hardware},
        prediction_config["launch_policy"],
        prediction_config["prediction_method"], runtime_inputs,
        execution_mode="accelerated",
    )
    schedules = tuple(coverage._captured_schedule_steps(case))
    one_token = [
        schedule for schedule in schedules
        if sum(request.input_tokens + request.output_tokens
               for request in schedule.requests) == 1
    ]
    if not one_token:
        raise ValueError(f"{cell.cell_id} has no one-token captured step")
    captured = []
    original = kernel_predict.predict

    def record(spec, current_hardware, method, **kwargs):
        result = kernel_base.predict(spec, current_hardware, method, **kwargs)
        manifest = execution_artifact._kernel_spec_manifest(spec)
        physical = coverage._physical_key(manifest)
        if coverage._identity(physical) == target_physical_id:
            captured.append((spec, current_hardware, result))
        return result

    kernel_predict.predict = record
    try:
        execution.execute_step(one_token[0])
    finally:
        kernel_predict.predict = original
    if len(captured) != 1:
        raise ValueError(
            f"expected one Serving predictor call for {target_physical_id}, "
            f"got {len(captured)}")
    return captured[0]


def main(load_compact_case) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--coverage", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    coverage_paths = {}
    for value in args.coverage:
        hardware_id, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError(f"invalid coverage input: {value}")
        coverage_paths[hardware_id] = Path(raw_path)
    if set(coverage_paths) != set(HARDWARE_IDS):
        raise ValueError("coverage inputs must contain all three hardware IDs")
    config = compact.load_phase1_config(args.config)
    cells = {
        cell.hardware.hardware_id: cell
        for cell in compact.select_phase1_cells(config, (), all_cases=True)
        if cell.model.dataset_model_id == "qwen3-8b"
        and cell.case_id == "tp1_qps64"
    }
    rows = []
    for hardware_id in HARDWARE_IDS:
        report = json.loads(coverage_paths[hardware_id].read_text())
        row = select_exact_row(report)
        family, reason, _query = coverage._candidate_query(row["op_key"])
        if family is None:
            raise ValueError(f"exact row is unmapped: {reason}")
        dataset_case_id = row["canonical_mapping"]["matches"][0]
        dataset_hardware_path = Path(
            get_hardware_config_path(hardware_id)).resolve()
        dataset_spec, dataset_hw, dataset_prediction = capture_dataset(
            hardware_id, family, dataset_case_id)
        cell = cells[hardware_id]
        serving_hardware_path = (
            REPOSITORY_ROOT / cell.hardware.hardware_config_path).resolve()
        case = load_compact_case(
            compact.compact_case_path(REPOSITORY_ROOT, config, cell))
        serving_spec, serving_hw, serving_prediction = capture_serving(
            config, cell, case, row["physical_key_ids"][0])
        dataset_value = kernel_spec_value(dataset_spec)
        serving_value = kernel_spec_value(serving_spec)
        differences = diff(dataset_value, serving_value)
        expected_serving_ns = (
            row["predicted_device_time_s"] * 1e9
            / row["logical_occurrences"])
        rows.append({
            "hardware_id": hardware_id,
            "dataset_case_id": dataset_case_id,
            "serving_workload": row["op_key"]["workload"],
            "shared_public_predictor": (
                evaluation_runner.predict_production is kernel_base.predict
                and kernel_predict.predict is kernel_base.predict),
            "hardware_id_equal": (
                dataset_hw.hardware_id == serving_hw.hardware_id),
            "dataset_hardware_config_path": str(dataset_hardware_path),
            "serving_hardware_config_path": str(serving_hardware_path),
            "hardware_config_path_equal": (
                dataset_hardware_path == serving_hardware_path),
            "dataset_hardware_config_sha256": file_sha256(
                dataset_hardware_path),
            "serving_hardware_config_sha256": file_sha256(
                serving_hardware_path),
            "hardware_config_bytes_equal": (
                file_sha256(dataset_hardware_path)
                == file_sha256(serving_hardware_path)),
            "kernel_spec_equal": not differences,
            "kernel_spec_differences": differences,
            "dataset_prediction_ns": dataset_prediction.t_total_ns,
            "serving_prediction_ns": serving_prediction.t_total_ns,
            "coverage_prediction_ns": expected_serving_ns,
            "dataset_vs_serving_prediction_delta_ns": (
                dataset_prediction.t_total_ns
                - serving_prediction.t_total_ns),
            "serving_capture_matches_coverage": (
                math.isclose(
                    serving_prediction.t_total_ns,
                    expected_serving_ns,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )),
            "dataset_kernel_spec": dataset_value,
            "serving_kernel_spec": serving_value,
        })
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema_version": 2,
        "actual_serving_exact_matches": rows,
        "dataset_cases_via_serving_lowering": dataset_replay_rows(),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
