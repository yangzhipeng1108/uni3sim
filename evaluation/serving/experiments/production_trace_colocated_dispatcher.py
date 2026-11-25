"""Colocated Production Trace point validation dispatch across
parallelism topologies."""

from __future__ import annotations

from typing import Any
from src.serving.request_gen.request_stream import RequestStream
from src.serving.workload.model_config import ModelConfig

from evaluation.serving.experiments.production_trace_contracts import (
    validate_production_trace_scheduler_runtime,
)

from evaluation.serving.experiments.production_trace_tp1_validation import (
    validate_colocated_tp1_production_trace_point,
)
from evaluation.serving.experiments.production_trace_tp2_validation import (
    validate_colocated_tp2_production_trace_point,
)
from evaluation.serving.experiments.production_trace_ep_pp_validation import (
    validate_colocated_ep2_production_trace_point,
    validate_colocated_pp2_production_trace_point,
    validate_colocated_tp2_pp2_production_trace_point,
)


def validate_colocated_production_trace_point(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
) -> None:
    if (expected_simulation["pd_disagg"] is not None
            or expected_simulation["af_disagg"] is not None):
        raise ValueError(
            "colocated Production Trace command retained disaggregated "
            "configuration")
    scheduler_runtime = result["scheduler_runtime"]
    if set(scheduler_runtime) != {"main"}:
        raise ValueError(
            "Production Trace sweep must contain exactly the main Scheduler "
            "runtime")
    runtime = scheduler_runtime["main"]
    expected_scheduler = expected_simulation["scheduler"]
    validate_production_trace_scheduler_runtime(
        runtime,
        expected_scheduler,
        expected_model,
        expected_simulation["mtp"],
        role="colocated",
        tensor_parallel_field="prefill_tp_size",
    )
    parallelism = (
        expected_scheduler["prefill_tp_size"],
        expected_scheduler["prefill_pp_size"],
    )
    if parallelism == (1, 1):
        validate_colocated_tp1_production_trace_point(
            expected_simulation,
            expected_model,
            expected_model_path,
            result,
            expected_requests,
            expected_slo_contract,
        )
        return
    if parallelism == (2, 1):
        if expected_scheduler["enable_expert_parallel"]:
            validate_colocated_ep2_production_trace_point(
                expected_simulation,
                expected_model,
                expected_model_path,
                result,
                expected_requests,
                expected_slo_contract,
            )
        else:
            validate_colocated_tp2_production_trace_point(
                expected_simulation,
                expected_model,
                expected_model_path,
                result,
                expected_requests,
                expected_slo_contract,
            )
    elif parallelism == (1, 2):
        validate_colocated_pp2_production_trace_point(
            expected_simulation,
            expected_model,
            expected_model_path,
            result,
            expected_requests,
        )
    elif parallelism == (2, 2):
        validate_colocated_tp2_pp2_production_trace_point(
            expected_simulation,
            expected_model,
            expected_model_path,
            result,
            expected_requests,
            expected_slo_contract,
        )
    else:
        raise AssertionError(
            "validated colocated Production Trace parallelism vanished")
