"""Directional traffic rows and the per-hardware mechanism contract
for kernel mechanism reports."""

from __future__ import annotations

import math
from typing import Any, Mapping, NamedTuple, Sequence

from evaluation.kernel.mechanism_capture import (
    DEMAND_ROLES,
    PROFILER_EVIDENCE_KINDS,
    TIME_ROLES,
    canonical_capture_comparison_evidence,
    is_semantically_comparable_component,
)

from evaluation.kernel.mechanism_report_primitives import (
    memory_path_bytes,
)


def directional_kernel_traffic_fields(
        mechanism: Mapping[str, Any]) -> Mapping[str, Any]:
    """Expose native directional counters without inventing equivalence."""

    analytical = mechanism["analytical_input"]
    observation = mechanism["observation"]
    native = observation["native"]
    values = {
        "predicted_l2_read_bytes": memory_path_bytes(
            analytical, access_kind="read", resource_role="l2_cache"),
        "predicted_l2_write_bytes": memory_path_bytes(
            analytical, access_kind="write", resource_role="l2_cache"),
        "predicted_dram_read_bytes": memory_path_bytes(
            analytical, access_kind="read", resource_role="dram"),
        "predicted_dram_write_bytes": memory_path_bytes(
            analytical, access_kind="write", resource_role="dram"),
        "predicted_mte3_write_bytes": memory_path_bytes(
            analytical, access_kind="write", resource_role="mte3"),
        "observed_xbar_read_bytes": observation["l2_interface_read_bytes"],
        "observed_xbar_write_bytes": observation["l2_interface_write_bytes"],
        "observed_nvidia_l2_source_read_bytes": None,
        "observed_nvidia_l2_source_write_bytes": None,
        "observed_nvidia_l2_read_lookup_hit_bytes": None,
        "observed_nvidia_l2_read_lookup_miss_bytes": None,
        "observed_nvidia_l2_write_lookup_hit_bytes": None,
        "observed_nvidia_l2_write_lookup_miss_bytes": None,
        "observed_nvidia_l2_read_lookup_partition_closes": None,
        "observed_nvidia_l2_write_lookup_partition_closes": None,
        "observed_nvidia_dram_read_in_window_bytes": None,
        "observed_nvidia_dram_writeback_in_window_bytes": None,
        "observed_metax_l2c_source_read_bytes": None,
        "observed_metax_downstream_global_read_bytes": None,
        "observed_metax_downstream_global_write_bytes": None,
        "observed_ascend_gm_to_l1_bytes": None,
        "observed_ascend_gm_to_ub_bytes": None,
        "observed_ascend_ub_to_gm_bytes": None,
        "observed_ascend_l0c_to_gm_bytes": None,
        "observed_ascend_dram_read_derived_bytes": None,
    }
    if native["backend"] == "nvidia_ncu":
        values.update({
            "observed_nvidia_l2_source_read_bytes": native[
                "l2_source_read_bytes"],
            "observed_nvidia_l2_source_write_bytes": native[
                "l2_source_write_bytes"],
            "observed_nvidia_l2_read_lookup_hit_bytes": native[
                "l2_read_lookup_hit_bytes"],
            "observed_nvidia_l2_read_lookup_miss_bytes": native[
                "l2_read_lookup_miss_bytes"],
            "observed_nvidia_l2_write_lookup_hit_bytes": native[
                "l2_write_lookup_hit_bytes"],
            "observed_nvidia_l2_write_lookup_miss_bytes": native[
                "l2_write_lookup_miss_bytes"],
            "observed_nvidia_l2_read_lookup_partition_closes": native[
                "l2_read_lookup_partition_closes"],
            "observed_nvidia_l2_write_lookup_partition_closes": native[
                "l2_write_lookup_partition_closes"],
            "observed_nvidia_dram_read_in_window_bytes": native[
                "dram_read_bytes"],
            "observed_nvidia_dram_writeback_in_window_bytes": native[
                "dram_write_bytes"],
        })
    elif native["backend"] == "metax_mcprofiler":
        values.update({
            "observed_metax_l2c_source_read_bytes": native[
                "l2c_memory_access_bytes"],
            "observed_metax_downstream_global_read_bytes": native[
                "global_read_bytes"],
            "observed_metax_downstream_global_write_bytes": native[
                "global_write_bytes"],
        })
    elif native["backend"] == "ascend_msprof":
        values.update({
            "observed_ascend_gm_to_l1_bytes": native[
                "raw_gm_to_l1_bytes"],
            "observed_ascend_gm_to_ub_bytes": native["gm_to_ub_bytes"],
            "observed_ascend_ub_to_gm_bytes": native["ub_to_gm_bytes"],
            "observed_ascend_l0c_to_gm_bytes": native["l0c_to_gm_bytes"],
            "observed_ascend_dram_read_derived_bytes": native[
                "dram_read_bytes"],
        })
    else:
        raise ValueError(
            f"unknown directional observation backend: {native['backend']!r}")
    return values


def directional_traffic_row(
        *, task: Mapping[str, Any], mechanism: Mapping[str, Any],
        component: str, resource_role: str, access_kind: str,
        observed: float, provider: str, metric: str,
        comparison_evidence: Mapping[str, str] | None = None,
        ) -> Mapping[str, Any] | None:
    if not math.isfinite(observed) or observed < 0:
        raise ValueError(
            f"directional traffic observation must be finite and non-negative: "
            f"{observed!r}")
    predicted = memory_path_bytes(
        mechanism["analytical_input"],
        access_kind=access_kind,
        resource_role=resource_role,
    )
    if predicted == 0.0 and observed == 0.0:
        return None
    if comparison_evidence is None:
        evidence = canonical_capture_comparison_evidence(
            mechanism["hardware_id"])
    else:
        required_evidence_fields = {"comparison_status", "comparison_reason"}
        if set(comparison_evidence) != required_evidence_fields:
            raise ValueError(
                "directional comparison evidence must contain exactly "
                f"{sorted(required_evidence_fields)}")
        evidence = comparison_evidence
    return {
        "hardware_id": mechanism["hardware_id"],
        "family": task["family"],
        "case_id": task["case_id"],
        "kernel_ordinal": mechanism["kernel_ordinal"],
        "variant": mechanism["variant"],
        "kernel_name": mechanism["kernel_name"],
        "actual_ns": mechanism["timing_target"]["duration_ns"],
        "timing_source": mechanism["timing_target"]["source"],
        "component": component,
        "unit": "bytes",
        "predicted": predicted,
        "observed": observed,
        **evidence,
        "provenance_kind": "profiler",
        "provider": provider,
        "metric": metric,
    }


def nvidia_directional_traffic_rows(
        *, task: Mapping[str, Any],
        mechanism: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Compare A800 read/write traffic without an aggregate path split."""

    native = mechanism["observation"]["native"]
    if native["backend"] != "nvidia_ncu":
        raise ValueError(
            "NVIDIA directional traffic requires an NCU observation")
    observation = mechanism["observation"]
    specifications = [
        (
            "traffic:xbar_read", "l2_cache", "read",
            observation["l2_interface_read_bytes"],
            "l1tex__m_xbar2l1tex_read_bytes.sum",
        ),
        (
            "traffic:xbar_write", "l2_cache", "write",
            observation["l2_interface_write_bytes"],
            "l1tex__m_l1tex2xbar_write_bytes.sum",
        ),
        (
            "traffic:dram_read_in_window", "dram", "read",
            native["dram_read_bytes"], "dram__bytes_read.sum",
        ),
    ]
    if native["l2_directional_read_write_comparable"]:
        # L2 atom/reduction requests are separate native operations.  Only
        # compare the ordinary read/write source counters when those classes
        # are absent from the invocation.
        specifications.extend((
        (
            "traffic:l2_source_read", "l2_cache", "read",
            native["l2_source_read_bytes"],
            "lts__t_sectors_srcunit_tex_op_read.sum * 32 bytes",
        ),
        (
            "traffic:l2_source_write", "l2_cache", "write",
            native["l2_source_write_bytes"],
            "lts__t_sectors_srcunit_tex_op_write.sum * 32 bytes",
        ),
        ))
    rows = []
    for component, role, access_kind, observed, metric in specifications:
        row = directional_traffic_row(
            task=task,
            mechanism=mechanism,
            component=component,
            resource_role=role,
            access_kind=access_kind,
            observed=float(observed),
            provider="nvidia_ncu",
            metric=metric,
        )
        if row is not None:
            rows.append(row)
    writeback_row = directional_traffic_row(
        task=task,
        mechanism=mechanism,
        component="traffic:dram_writeback_in_window",
        resource_role="dram",
        access_kind="write",
        observed=float(native["dram_write_bytes"]),
        provider="nvidia_ncu",
        metric="dram__bytes_write.sum",
        comparison_evidence={
            "comparison_status": "not_comparable",
            "comparison_reason": (
                "ncu_kernel_window_writeback_is_not_the_forward_output_store_"
                "obligation_without_independent_eviction_boundary_evidence"),
        },
    )
    if writeback_row is not None:
        rows.append(writeback_row)
    return rows


def ascend_directional_traffic_rows(
        *, task: Mapping[str, Any],
        mechanism: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Compare pure AIC/AIV transfers at their native directional boundary."""

    native = mechanism["observation"]["native"]
    if native["backend"] != "ascend_msprof":
        raise ValueError(
            "Ascend directional traffic requires an msprof observation")
    if native["op_type"] == "mix":
        # Historical mixed-domain aggregates did not retain the per-core join
        # required to attribute the AIC and AIV directions independently.
        return []
    if native["op_type"] not in ("cube", "vector"):
        raise ValueError(
            "Ascend directional traffic requires cube/vector Op Type, got "
            f"{native['op_type']!r}")
    if native["gm_to_local_bytes"] is None:
        return []
    if native["op_type"] == "cube":
        mte2_component = "traffic:aic_mte2_gm_to_l1"
        mte2_metric = "GM_to_L1_datas"
    elif native["op_type"] == "vector":
        mte2_component = "traffic:aiv_mte2_gm_to_ub"
        mte2_metric = "GM_to_UB_datas"
    else:
        raise AssertionError(
            f"unhandled pure Ascend Op Type: {native['op_type']!r}")
    specifications = [
        (
            mte2_component, "l2_cache", "read",
            native["gm_to_local_bytes"], mte2_metric,
        ),
        (
            "traffic:l2_source_write", "l2_cache", "write",
            native["write_main_memory_bytes"],
            "write_main_memory_datas",
        ),
        (
            "traffic:dram_read_derived", "dram", "read",
            native["dram_read_bytes"],
            "read_main_memory_datas * (1 - directional L2 read hit rate)",
        ),
    ]
    if native["ub_to_gm_bytes"] is not None:
        specifications.append((
            "traffic:aiv_mte3_ub_to_gm", "mte3", "write",
            native["ub_to_gm_bytes"], "UB_to_GM_datas",
        ))
    rows = []
    for component, role, access_kind, observed, metric in specifications:
        if observed is None:
            continue
        row = directional_traffic_row(
            task=task,
            mechanism=mechanism,
            component=component,
            resource_role=role,
            access_kind=access_kind,
            observed=float(observed),
            provider="ascend_msprof",
            metric=metric,
        )
        if row is not None:
            rows.append(row)
    return rows


def metax_directional_traffic_rows(
        *, task: Mapping[str, Any], mechanism: Mapping[str, Any],
        provenance: Mapping[str, Any],
        unavailable: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Compare C500 traffic only at matching directional interfaces."""

    native = mechanism["observation"]["native"]
    if native["backend"] != "metax_mcprofiler":
        raise ValueError(
            "MetaX directional traffic requires mcProfiler observation")
    analytical = mechanism["analytical_input"]
    specifications = (
        (
            "traffic:l2c_source_read",
            "l2_cache",
            "read",
            "l2c_memory_access_bytes",
            "demand_per_resource.xbar",
            "mcProfiler l2c_memacs read-side request bytes",
        ),
        (
            "traffic:downstream_global_read",
            "dram",
            "read",
            "global_read_bytes",
            "demand_per_resource.dram",
            "mcProfiler Global Memory Read bytes",
        ),
        (
            "traffic:downstream_global_write",
            "dram",
            "write",
            "global_write_bytes",
            "demand_per_resource.dram",
            "mcProfiler Global Memory Write bytes",
        ),
    )
    comparison_evidence = canonical_capture_comparison_evidence(
        mechanism["hardware_id"])
    rows = []
    for (component, resource_role, access_kind, observed_field,
         provenance_path, metric) in specifications:
        if provenance_path not in provenance or provenance_path in unavailable:
            continue
        source = provenance[provenance_path]
        if source["kind"] not in PROFILER_EVIDENCE_KINDS:
            continue
        if source["uses_target_duration"]:
            raise ValueError(
                f"component evidence {provenance_path!r} for "
                f"{task['case_id']!r} consumes target duration")
        observed_value = native[observed_field]
        if observed_value is None:
            raise ValueError(
                f"MetaX directional observation {observed_field!r} is absent "
                "despite profiler provenance")
        observed = float(observed_value)
        if not math.isfinite(observed) or observed < 0:
            raise ValueError(
                f"MetaX directional observation must be finite and "
                f"non-negative: {observed!r}")
        predicted = memory_path_bytes(
            analytical,
            access_kind=access_kind,
            resource_role=resource_role,
        )
        if predicted == 0 and observed == 0:
            continue
        rows.append({
            "hardware_id": mechanism["hardware_id"],
            "family": task["family"],
            "case_id": task["case_id"],
            "kernel_ordinal": mechanism["kernel_ordinal"],
            "variant": mechanism["variant"],
            "kernel_name": mechanism["kernel_name"],
            "actual_ns": mechanism["timing_target"]["duration_ns"],
            "timing_source": mechanism["timing_target"]["source"],
            "component": component,
            "unit": "bytes",
            "predicted": predicted,
            "observed": observed,
            **comparison_evidence,
            "provenance_kind": source["kind"],
            "provider": source["provider"],
            "metric": metric,
        })
    return rows


class MechanismHardwareContract(NamedTuple):
    """What one hardware contributes to the mechanism-validation comparison.

    ``directional_traffic_rows`` produces that hardware's directional traffic
    rows, or None when that hardware's rows depend on the oracle fields and
    are appended separately where the comparison is built (the MetaX case).
    ``native_profiler_op_type`` is True when the retained
    observation carries the device profiler's native op type, which the
    report displays; the other backends leave it unset.
    """

    directional_traffic_rows: object = None
    native_profiler_op_type: bool = False


MECHANISM_HARDWARE = {
    "nvidia_a800": MechanismHardwareContract(
        directional_traffic_rows=nvidia_directional_traffic_rows),
    "metax_c500": MechanismHardwareContract(),
    "ascend_910b": MechanismHardwareContract(
        directional_traffic_rows=ascend_directional_traffic_rows,
        native_profiler_op_type=True),
}


def comparison_rows(documents: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = []
    for document in documents:
        hardware_id = document["hardware_id"]
        comparison_evidence = canonical_capture_comparison_evidence(
            hardware_id)
        for task in document["tasks"]:
            for kernel in task["kernels"]:
                mechanism = kernel["mechanisms"]
                analytical = mechanism["analytical_input"]
                if hardware_id not in MECHANISM_HARDWARE:
                    raise ValueError(
                        "unknown mechanism-validation hardware: "
                        f"{hardware_id!r}")
                directional = MECHANISM_HARDWARE[
                    hardware_id].directional_traffic_rows
                if directional is not None:
                    rows.extend(directional(task=task, mechanism=mechanism))
                oracle = mechanism["oracle"]
                if not oracle["available"]:
                    continue
                provenance = oracle["field_provenance"]
                unavailable = oracle["unavailable_fields"]
                observed_input = oracle["observed_input"]
                if MECHANISM_HARDWARE[hardware_id].native_profiler_op_type:
                    profiler_op_type = mechanism[
                        "observation"]["native"]["op_type"]
                else:
                    profiler_op_type = None
                if hardware_id == "metax_c500":
                    rows.extend(metax_directional_traffic_rows(
                        task=task,
                        mechanism=mechanism,
                        provenance=provenance,
                        unavailable=unavailable,
                    ))
                for role in DEMAND_ROLES:
                    path = f"demand_per_resource.{role}"
                    component = f"demand:{role}"
                    if not is_semantically_comparable_component(
                            hardware_id, task["family"],
                            mechanism["variant"], component,
                            profiler_op_type):
                        continue
                    if path not in provenance or path in unavailable:
                        continue
                    source = provenance[path]
                    if source["kind"] not in PROFILER_EVIDENCE_KINDS:
                        continue
                    if source["uses_target_duration"]:
                        raise ValueError(
                            f"component evidence {path!r} for "
                            f"{task['case_id']!r} consumes target duration")
                    predicted = analytical["demand_per_resource"][role]
                    observed = observed_input["demand_per_resource"][role]
                    if predicted == 0 and observed == 0:
                        continue
                    if role in {
                            "l1_cache", "smem", "xbar", "l2_cache",
                            "dram", "mte3"}:
                        unit = "bytes"
                    elif role in {
                            "compute_tensor", "compute_vector",
                            "compute_reduction", "compute_fma", "compute_sfu",
                            "compute_alu", "warp_issue"}:
                        unit = "operations"
                    else:
                        raise ValueError(
                            f"unknown demand role unit: {role!r}")
                    rows.append({
                        "hardware_id": hardware_id,
                        "family": task["family"],
                        "case_id": task["case_id"],
                        "kernel_ordinal": mechanism["kernel_ordinal"],
                        "variant": mechanism["variant"],
                        "kernel_name": mechanism["kernel_name"],
                        "actual_ns": mechanism["timing_target"]["duration_ns"],
                        "timing_source": mechanism["timing_target"]["source"],
                        "component": component,
                        "unit": unit,
                        "predicted": predicted,
                        "observed": observed,
                        **comparison_evidence,
                        "provenance_kind": source["kind"],
                        "provider": source["provider"],
                        "metric": source["metric"],
                    })
                for role in TIME_ROLES:
                    path = f"resource_time_overrides_ns.{role}"
                    if path not in provenance or path in unavailable:
                        continue
                    source = provenance[path]
                    if source["kind"] not in PROFILER_EVIDENCE_KINDS:
                        continue
                    if source["uses_target_duration"]:
                        raise ValueError(
                            f"component evidence {path!r} for "
                            f"{task['case_id']!r} consumes target duration")
                    overrides = observed_input["resource_time_overrides_ns"]
                    if role not in overrides:
                        raise RuntimeError(
                            f"provenance declares absent time override {path}")
                    predicted_times = mechanism["forward"][
                        "resource_time_by_role_ns"]
                    if role not in predicted_times:
                        continue
                    if predicted_times[role] == 0 and overrides[role] == 0:
                        continue
                    rows.append({
                        "hardware_id": hardware_id,
                        "family": task["family"],
                        "case_id": task["case_id"],
                        "kernel_ordinal": mechanism["kernel_ordinal"],
                        "variant": mechanism["variant"],
                        "kernel_name": mechanism["kernel_name"],
                        "actual_ns": mechanism["timing_target"]["duration_ns"],
                        "timing_source": mechanism["timing_target"]["source"],
                        "component": f"service_time:{role}",
                        "unit": "ns",
                        "predicted": predicted_times[role],
                        "observed": overrides[role],
                        **comparison_evidence,
                        "provenance_kind": source["kind"],
                        "provider": source["provider"],
                        "metric": source["metric"],
                    })
    return rows


_MechanismHardwareContract = MechanismHardwareContract
