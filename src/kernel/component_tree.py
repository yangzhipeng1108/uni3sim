"""Recursive component tree for hardware configuration.

Each node is a Component with optional children. Leaf nodes with a `role`
become resources in the flat resource model used by formula.py.
Clock domains bind by component name and propagate to descendants.
"""

import hashlib
import json
import math
from pathlib import Path
import warnings

from src.kernel.model_contract import (
    COMPUTE_EXACT_REQUEST_SERVICES_BY_ROLE,
    MEMORY_EXACT_REQUEST_SERVICES_BY_ROLE,
    TYPED_COMPUTE_SERVICES_BY_ROLE,
)


_COMPONENT_REQUIRED_FIELDS = {"name", "count"}
_COMPONENT_OPTIONAL_FIELDS = {
    "role", "description", "clock_domain", "type", "throughput_per_clk", "capacity_per_second",
    "variants", "primary_variant", "traffic_filter", "size_bytes",
    "num_partitions", "scope", "children", "service_rates_per_unit_s",
    "service_setup_time_s", "active_unit_service_time_s",
    "exact_request_services",
}
_COMPONENT_FIELDS = _COMPONENT_REQUIRED_FIELDS | _COMPONENT_OPTIONAL_FIELDS

_EXACT_REQUEST_SERVICE_FIELDS = {
    "request_size_unit", "service_time_unit", "lookup_policy",
    "evidence_ref", "evidence_sha256",
    "production_or_evaluation_duration_consumed", "entries",
}
_EXACT_REQUEST_SERVICE_ENTRY_FIELDS = {"request_size", "service_time_ns"}
_EXACT_REQUEST_SERVICE_UNITS = {
    "aic_scalar_global_get_value_serialized": "int32_values",
    "cann_grouped_matmul_finite_group_scan_control": "expert_groups",
    "cann_grouped_matmul_zero_expert_control": "zero_expert_iterations",
    "cann_nbuffer33_l1_to_l0_pair_completion": "completed_source_pairs",
    "moe_compute_expert_tokens_key1001_init_sync_phase": (
        "shared_output_experts"),
    "add_rmsnorm_key31_residual_reload_completion_chain": (
        "source_split_d_residual_reload_chains"),
    "aiv_mte2_request_completion": "payload_bytes",
    "aiv_rope_mte2_to_vector_completion": "payload_bytes",
    "aiv_rope_mte2_to_vector_completion_32_workers": "payload_bytes",
    "aiv_rope_mte2_to_vector_completion_43_workers": "payload_bytes",
    "aiv_rope_mte2_to_vector_completion_47_workers": "payload_bytes",
    "aiv_rope_key24_mte2_steady_completion_47_workers_4096b": (
        "source_steady_request_count"),
    "aiv_rope_key22_first_q_slice_mte2_to_vector_completion": (
        "payload_bytes"),
    "aiv_rope_key22_first_q_slice_mte2_to_vector_completion_4_workers": (
        "payload_bytes"),
    "aiv_rope_key22_first_q_slice_mte2_to_vector_completion_8_workers": (
        "payload_bytes"),
    "aiv_rope_key22_first_q_slice_mte2_to_vector_completion_16_workers": (
        "payload_bytes"),
    "aiv_datacopypad_mte2_to_vector_completion": "payload_bytes",
    "aiv_datacopypad_mte2_to_vector_completion_two_workers": (
        "payload_bytes"),
    "aiv_add_rmsnorm_row_group_input_pair_completion_32_workers": (
        "payload_bytes"),
    "aiv_add_rmsnorm_single_n_input_pair_completion_one_worker": (
        "payload_bytes"),
    "aiv_rmsnorm_key0_gamma_input_sequence_completion": (
        "payload_bytes_per_source_tensor"),
    "aiv_rmsnorm_key0_h2048_mte2_to_vector_completion_4_workers": (
        "payload_bytes"),
    "aiv_rmsnorm_key0_h2048_mte2_to_vector_completion_8_workers": (
        "payload_bytes"),
    "aiv_rmsnorm_key0_first_mte2_completion_16_workers": (
        "payload_bytes"),
    "aiv_datacopypad_mte2_to_vector_completion_32_workers": (
        "payload_bytes"),
    "aiv_datacopypad_mte2_to_vector_completion_43_workers": (
        "payload_bytes"),
    "aiv_datacopypad_mte2_to_vector_completion_47_workers": (
        "payload_bytes"),
    "aiv_rmsnorm_key0_steady_mte2_completion_32_workers": "payload_bytes",
    "aiv_rmsnorm_key0_steady_mte2_completion_43_workers": "payload_bytes",
    "aiv_rmsnorm_key0_steady_mte2_completion_47_workers": "payload_bytes",
    "aiv_swiglu_tque_mte2_to_vector_completion_24_workers": "payload_bytes",
    "aiv_swiglu_tque_mte2_to_vector_completion_40_workers": "payload_bytes",
    "aiv_swiglu_tque_mte2_to_vector_completion_48_workers": "payload_bytes",
    "aiv_swiglu_tque_input_pair_completion_24_workers_q2": "payload_bytes",
    "aiv_swiglu_tque_input_pair_completion_40_workers_q2": "payload_bytes",
    "aiv_swiglu_tque_input_pair_completion_48_workers_q1": "payload_bytes",
    "aiv_swiglu_tque_input_pair_completion_48_workers_q2": "payload_bytes",
    "aiv_mte3_request_completion": "payload_bytes",
    "aiv_rope_vector_to_mte3_completion": "payload_bytes",
    "aiv_rope_vector_to_mte3_completion_43_workers": "payload_bytes",
    "aiv_rope_vector_to_mte3_completion_47_workers": "payload_bytes",
    "aiv_rope_key22_first_q_slice_vector_to_mte3_completion": (
        "payload_bytes"),
    "aiv_rope_key22_first_q_slice_vector_to_mte3_completion_4_workers": (
        "payload_bytes"),
    "aiv_rope_key22_first_q_slice_vector_to_mte3_completion_8_workers": (
        "payload_bytes"),
    "aiv_rope_key22_first_q_slice_vector_to_mte3_completion_16_workers": (
        "payload_bytes"),
    "aiv_datacopypad_vector_to_mte3_completion": "payload_bytes",
    "aiv_datacopypad_vector_to_mte3_completion_two_workers": (
        "payload_bytes"),
    "aiv_rmsnorm_key0_h2048_vector_to_mte3_completion_4_workers": (
        "payload_bytes"),
    "aiv_rmsnorm_key0_h2048_vector_to_mte3_completion_8_workers": (
        "payload_bytes"),
    "aiv_rmsnorm_key0_h2048_first_mte3_completion_32_workers": (
        "payload_bytes"),
    "aiv_rmsnorm_key0_h5120_first_mte3_completion_32_workers": (
        "payload_bytes"),
    "aiv_rmsnorm_key0_finite_source_envelope": "active_aiv_workers",
    "aiv_add_rmsnorm_key30_terminal_output_sequence_32_workers": (
        "payload_bytes"),
    "aiv_datacopypad_vector_to_mte3_completion_32_workers": (
        "payload_bytes"),
    "aiv_datacopypad_vector_to_mte3_completion_43_workers": (
        "payload_bytes"),
    "aiv_datacopypad_vector_to_mte3_completion_47_workers": (
        "payload_bytes"),
    "aiv_swiglu_tque_vector_to_mte3_completion_24_workers": "payload_bytes",
    "aiv_swiglu_tque_vector_to_mte3_completion_40_workers": "payload_bytes",
    "aiv_swiglu_tque_vector_to_mte3_completion_48_workers": "payload_bytes",
    "aiv_mul_q4q1_finite_worker_completion_48_workers": (
        "source_worker_programs"),
    "aiv_mul_q4q1_finite_worker_completion_48_workers_11008b_x2": (
        "source_worker_programs"),
    "aiv_mul_q4q1_finite_worker_completion_48_workers_13824b_x2": (
        "source_worker_programs"),
    "aiv_mul_q4q1_finite_worker_completion_48_workers_18432b_x3": (
        "source_worker_programs"),
    "aiv_mul_q4q1_finite_worker_completion_48_workers_17664b_x5": (
        "source_worker_programs"),
    "aiv_elementwise_single_tile_finite_worker_completion_47_workers": (
        "source_worker_programs"),
    "aiv_reshape_cache_equal_chain_completion": "payload_bytes",
    "aiv_reshape_cache_equal_chain_completion_48_workers": "payload_bytes",
    "aiv_reshape_cache_equal_warm_chain_completion_2_workers": (
        "payload_bytes"),
    "aiv_reshape_cache_equal_warm_chain_completion_4_workers": (
        "payload_bytes"),
    "aiv_reshape_cache_equal_warm_chain_completion_8_workers": (
        "payload_bytes"),
    "aiv_reshape_cache_equal_warm_chain_completion_16_workers": (
        "payload_bytes"),
    "aiv_reshape_cache_equal_repeated_chain_completion_48_workers_1024b": (
        "source_kv_token_chains_per_maximum_worker"),
    "aiv_reshape_cache_equal_repeated_chain_completion_48_workers_2048b": (
        "source_kv_token_chains_per_maximum_worker"),
    "aiv_reshape_cache_unequal_repeated_pair_completion_48_workers_2048b": (
        "source_kv_queue_pairs_per_maximum_worker"),
    "local_scalar_fill_s_v": "fp32_scalar_writes",
    "local_ub_copy_contiguous": "payload_bytes",
    "local_ub_copy_stride_64b": "payload_bytes",
    "local_ub_copy_stride_128b": "payload_bytes",
    "norm_sqrt_duplicate_div_serialized": "fp32_elements",
    "rmsnorm_local_scalar_dependency": (
        "lowered_scalar_operations_per_chain"),
    "rmsnorm_block128_wave_lifetime_completion": (
        "source_program_completion"),
    "scalar_global_get_value_serialized": "int32_values",
    "add_rmsnorm_local_scalar_handoff": "source_scalar_handoff_chains",
    "add_rmsnorm_split_d_local_scalar_handoff": (
        "source_split_d_scalar_handoff_chains"),
    "vector_to_mte2_fence": "source_vector_to_mte2_fences",
    "aiv_four_queue_control_chain": "queue_depth",
    "aiv_four_queue_three_tbuf_lifecycle": "queue_depth",
    "aiv_queue_control_cycle": "source_queue_cycles",
    "aiv_scalar_u64_divrem_dependency": (
        "source_u64_divrem_dependency_groups"),
    "aiv_swiglu_tile_address_control": (
        "source_swiglu_tile_address_control_groups"),
    "aiv_three_queue_control_chain": "queue_depth",
    "aiv_three_queue_two_tbuf_lifecycle": "queue_depth",
    "aiv_single_tbuf_lifecycle": "buffer_bytes",
    "aiv_gather_v3_two_tque_lifecycle": (
        "source_two_tque_lifecycles"),
    "aiv_elementwise_tbuf_lifecycle_110080b": (
        "source_tbuf_lifecycles"),
    "aiv_elementwise_mul_prefix_46_workers": "source_vector_operations",
    "aiv_elementwise_mul_suffix_46_workers": "source_vector_operations",
    "aiv_elementwise_mul_prefix_47_workers": "source_vector_operations",
    "aiv_elementwise_mul_suffix_47_workers": "source_vector_operations",
}
_EXACT_REQUEST_SERVICE_NAMES = set(_EXACT_REQUEST_SERVICE_UNITS)
_EXACT_REQUEST_SERVICE_TIME_UNITS = {
    service: "ns_per_request_per_aiv"
    for service in _EXACT_REQUEST_SERVICE_NAMES
}
_EXACT_REQUEST_SERVICE_TIME_UNITS[
    "aic_scalar_global_get_value_serialized"
] = "ns_per_request_per_aic"
_EXACT_REQUEST_SERVICE_TIME_UNITS[
    "cann_grouped_matmul_finite_group_scan_control"
] = "ns_per_request_per_aic"
_EXACT_REQUEST_SERVICE_TIME_UNITS[
    "cann_grouped_matmul_zero_expert_control"
] = "ns_per_request_per_aic"
_EXACT_REQUEST_SERVICE_TIME_UNITS[
    "cann_nbuffer33_l1_to_l0_pair_completion"
] = "ns_per_completed_load_pair_per_aic"
_EXACT_REQUEST_SERVICE_TIME_UNITS[
    "moe_compute_expert_tokens_key1001_init_sync_phase"
] = "ns_per_complete_48_aiv_source_phase"
_EXACT_REQUEST_SERVICE_TIME_UNITS[
    "add_rmsnorm_key31_residual_reload_completion_chain"
] = "ns_per_maximum_worker_chain"
_EXACT_REQUEST_SERVICE_TIME_UNITS[
    "aiv_reshape_cache_equal_repeated_chain_completion_48_workers_1024b"
] = "ns_per_maximum_worker_chain"
_EXACT_REQUEST_SERVICE_TIME_UNITS[
    "aiv_reshape_cache_equal_repeated_chain_completion_48_workers_2048b"
] = "ns_per_maximum_worker_chain"
_EXACT_REQUEST_SERVICE_TIME_UNITS[
    "aiv_reshape_cache_unequal_repeated_pair_completion_48_workers_2048b"
] = "ns_per_maximum_worker_pair_sequence"
_EXACT_REQUEST_SERVICE_TIME_UNITS[
    "aiv_mul_q4q1_finite_worker_completion_48_workers"
] = "ns_per_maximum_worker_program"
_EXACT_REQUEST_SERVICE_TIME_UNITS[
    "aiv_mul_q4q1_finite_worker_completion_48_workers_11008b_x2"
] = "ns_per_maximum_worker_program"
_EXACT_REQUEST_SERVICE_TIME_UNITS[
    "aiv_mul_q4q1_finite_worker_completion_48_workers_13824b_x2"
] = "ns_per_maximum_worker_program"
_EXACT_REQUEST_SERVICE_TIME_UNITS[
    "aiv_mul_q4q1_finite_worker_completion_48_workers_18432b_x3"
] = "ns_per_maximum_worker_program"
_EXACT_REQUEST_SERVICE_TIME_UNITS[
    "aiv_mul_q4q1_finite_worker_completion_48_workers_17664b_x5"
] = "ns_per_maximum_worker_program"
_EXACT_REQUEST_SERVICE_TIME_UNITS[
    "aiv_elementwise_single_tile_finite_worker_completion_47_workers"
] = "ns_per_maximum_worker_program"
_EXACT_REQUEST_SERVICE_TIME_UNITS[
    "aiv_rmsnorm_key0_finite_source_envelope"
] = "ns_per_launch_wide_source_envelope"
_EXACT_REQUEST_SERVICE_TIME_UNITS[
    "rmsnorm_block128_wave_lifetime_completion"
] = "ns_per_cta_completion"
_EXACT_REQUEST_SERVICE_ROLES = {
    **{
        service: role
        for role, services in COMPUTE_EXACT_REQUEST_SERVICES_BY_ROLE.items()
        for service in services
    },
    **{
        service: role
        for role, services in MEMORY_EXACT_REQUEST_SERVICES_BY_ROLE.items()
        for service in services
    },
}
_EXACT_REQUEST_EVIDENCE_REQUIRED_FIELDS = {
    "service", "request_size_unit", "service_time_unit", "lookup_policy",
    "production_or_evaluation_duration_consumed",
    "service_time_ns_by_request_size",
}
_SWIGLU_QUEUE_EDGE_EVIDENCE_KEYS = {
    "aiv_swiglu_tque_mte2_to_vector_completion_24_workers": (
        24, "tque_mte2_to_vector"),
    "aiv_swiglu_tque_mte2_to_vector_completion_40_workers": (
        40, "tque_mte2_to_vector"),
    "aiv_swiglu_tque_mte2_to_vector_completion_48_workers": (
        48, "tque_mte2_to_vector"),
    "aiv_swiglu_tque_vector_to_mte3_completion_24_workers": (
        24, "tque_vector_to_mte3"),
    "aiv_swiglu_tque_vector_to_mte3_completion_40_workers": (
        40, "tque_vector_to_mte3"),
    "aiv_swiglu_tque_vector_to_mte3_completion_48_workers": (
        48, "tque_vector_to_mte3"),
}
_SWIGLU_QUEUE_EDGE_EVIDENCE_DOMAINS = frozenset({
    (worker_count, direction)
    for worker_count in (1, 24, 40, 48)
    for direction in (
        "tque_mte2_to_vector", "tque_vector_to_mte3")
})
_ROPE_KEY22_FIRST_Q_SLICE_EVIDENCE_KEYS = {
    "aiv_rope_key22_first_q_slice_mte2_to_vector_completion": (
        1, "mte2_to_vector"),
    "aiv_rope_key22_first_q_slice_mte2_to_vector_completion_16_workers": (
        16, "mte2_to_vector"),
    "aiv_rope_key22_first_q_slice_vector_to_mte3_completion": (
        1, "mte3_to_vector"),
    "aiv_rope_key22_first_q_slice_vector_to_mte3_completion_16_workers": (
        16, "mte3_to_vector"),
}
_RMSNORM_KEY0_H2048_MTE_EDGE_EVIDENCE_KEYS = {
    "aiv_rmsnorm_key0_h2048_mte2_to_vector_completion_4_workers": (
        4, "mte2_to_vector"),
    "aiv_rmsnorm_key0_h2048_mte2_to_vector_completion_8_workers": (
        8, "mte2_to_vector"),
    "aiv_rmsnorm_key0_h2048_vector_to_mte3_completion_4_workers": (
        4, "mte3_to_vector"),
    "aiv_rmsnorm_key0_h2048_vector_to_mte3_completion_8_workers": (
        8, "mte3_to_vector"),
}


def _load_rmsnorm_key0_h2048_mte_edge_evidence(
        owner, service, evidence):
    """Select one exact worker/direction RMSNorm key-0 edge service."""

    required_fields = {
        "campaign", "service_time_unit", "lookup_policy",
        "source_scope", "source_tile_evidence", "published_cells",
        "production_or_evaluation_duration_consumed",
        "mechanism_publishable",
    }
    missing_fields = required_fields - set(evidence)
    if missing_fields:
        raise ValueError(
            f"{owner} exact request service {service!r} RMSNorm evidence "
            f"is missing fields: {sorted(missing_fields)}")
    if evidence["campaign"] != "rmsnorm_key0_h2048_mte_edges":
        raise ValueError(
            f"{owner} exact request service {service!r} evidence campaign "
            "must be rmsnorm_key0_h2048_mte_edges")
    if evidence["service_time_unit"] != "ns_per_request_per_aiv":
        raise ValueError(
            f"{owner} exact request service {service!r} evidence unit "
            "changed")
    if evidence["lookup_policy"] != (
            "exact_cann_rmsnorm_key0_h2048_worker_count_direction_and_"
            "payload_only"):
        raise ValueError(
            f"{owner} exact request service {service!r} evidence lookup "
            "policy changed")
    if evidence["production_or_evaluation_duration_consumed"] is not False:
        raise ValueError(
            f"{owner} exact request service {service!r} evidence cannot "
            "consume production or evaluation duration")
    if evidence["mechanism_publishable"] is not True:
        raise ValueError(
            f"{owner} exact request service {service!r} evidence is not "
            "publishable")
    tile = evidence["source_tile_evidence"]
    if (not isinstance(tile, dict)
            or tile["hidden_elements"] != 2048
            or tile["dtype_bytes"] != 2
            or tile["payload_bytes"] != 4096
            or tile["production_worker_counts"] != [4, 8]
            or "single-row H2048 workers 4 or 8" not in
            tile["scope_boundary"]):
        raise ValueError(
            f"{owner} exact request service {service!r} RMSNorm source "
            "scope changed")
    published_cells = evidence["published_cells"]
    expected_cells = {
        f"{worker_count}:{direction}:4096"
        for worker_count in (4, 8)
        for direction in ("mte2_to_vector", "mte3_to_vector")
    }
    if (not isinstance(published_cells, dict)
            or set(published_cells) != expected_cells):
        raise ValueError(
            f"{owner} exact request service {service!r} RMSNorm "
            "published-cell domain changed")
    worker_count, direction = (
        _RMSNORM_KEY0_H2048_MTE_EDGE_EVIDENCE_KEYS[service])
    cell = f"{worker_count}:{direction}:4096"
    result = published_cells[cell]
    if not isinstance(result, dict) or "service_time_ns" not in result:
        raise ValueError(
            f"{owner} exact request service {service!r} evidence cell "
            f"schema changed: {cell!r}")
    service_time_ns = result["service_time_ns"]
    if (isinstance(service_time_ns, bool)
            or not isinstance(service_time_ns, (int, float))
            or not math.isfinite(service_time_ns)
            or service_time_ns <= 0):
        raise ValueError(
            f"{owner} exact request service {service!r} evidence cell has "
            f"invalid service time: {cell!r}")
    return {4096: float(service_time_ns)}


def _load_rope_key22_first_q_slice_evidence(owner, service, evidence):
    """Select one exact first-Q-slice edge from the retained campaign."""

    required_fields = {
        "campaign", "lookup_policy", "published_cells",
        "source_tile_evidence", "production_or_evaluation_duration_consumed",
        "mechanism_publishable",
    }
    missing_fields = required_fields - set(evidence)
    if missing_fields:
        raise ValueError(
            f"{owner} exact request service {service!r} key-22 RoPE "
            f"evidence is missing fields: {sorted(missing_fields)}")
    if evidence["campaign"] != "rope_key22_first_q_slice_edges":
        raise ValueError(
            f"{owner} exact request service {service!r} evidence campaign "
            "must be rope_key22_first_q_slice_edges")
    if evidence["lookup_policy"] != (
            "exact_worker_count_direction_payload_and_first_q_slice_only"):
        raise ValueError(
            f"{owner} exact request service {service!r} evidence lookup "
            "policy changed")
    if evidence["production_or_evaluation_duration_consumed"] is not False:
        raise ValueError(
            f"{owner} exact request service {service!r} evidence cannot "
            "consume production or evaluation duration")
    if evidence["mechanism_publishable"] is not True:
        raise ValueError(
            f"{owner} exact request service {service!r} evidence is not "
            "publishable")
    source_tile_evidence = evidence["source_tile_evidence"]
    if (not isinstance(source_tile_evidence, dict)
            or source_tile_evidence["slice_elements"] != 4096
            or source_tile_evidence["dtype_bytes"] != 2
            or source_tile_evidence["payload_bytes"] != 8192
            or source_tile_evidence["production_worker_counts"]
            != [1, 2, 4, 8, 16]
            or "first Q slice per worker only" not in
            source_tile_evidence["scope_boundary"]):
        raise ValueError(
            f"{owner} exact request service {service!r} source scope "
            "changed")
    published_cells = evidence["published_cells"]
    if not isinstance(published_cells, dict) or not published_cells:
        raise TypeError(
            f"{owner} exact request service {service!r} published_cells "
            "must be a non-empty mapping")
    expected_cells = {
        f"{worker_count}:{direction}:8192"
        for worker_count in (1, 16)
        for direction in ("mte2_to_vector", "mte3_to_vector")
    }
    if set(published_cells) != expected_cells:
        raise ValueError(
            f"{owner} exact request service {service!r} key-22 RoPE "
            "published-cell domain changed")
    for published_cell, published_result in published_cells.items():
        if (not isinstance(published_result, dict)
                or "service_time_ns" not in published_result):
            raise ValueError(
                f"{owner} exact request service {service!r} evidence cell "
                f"schema changed: {published_cell!r}")
        published_service_time_ns = published_result["service_time_ns"]
        if (isinstance(published_service_time_ns, bool)
                or not isinstance(published_service_time_ns, (int, float))
                or not math.isfinite(published_service_time_ns)
                or published_service_time_ns <= 0):
            raise ValueError(
                f"{owner} exact request service {service!r} evidence cell "
                f"has invalid service time: {published_cell!r}")
    worker_count, direction = _ROPE_KEY22_FIRST_Q_SLICE_EVIDENCE_KEYS[service]
    cell = f"{worker_count}:{direction}:8192"
    result = published_cells[cell]
    service_time_ns = result["service_time_ns"]
    return {8192: float(service_time_ns)}


def _load_swiglu_queue_edge_evidence(owner, service, evidence):
    """Select one production-valid worker/direction SwiGLU service."""

    required_fields = {
        "campaign", "lookup_policy", "published_cells",
        "production_or_evaluation_duration_consumed",
        "mechanism_publishable",
    }
    missing_fields = required_fields - set(evidence)
    if missing_fields:
        raise ValueError(
            f"{owner} exact request service {service!r} SwiGLU evidence is "
            f"missing fields: {sorted(missing_fields)}")
    worker_count, direction = _SWIGLU_QUEUE_EDGE_EVIDENCE_KEYS[service]
    expected_campaigns = (
        {"swiglu_t8_queue_edges"} if worker_count == 40
        else ({"swiglu_t1024_queue_edges", "swiglu_t64_queue_edges"}
              if worker_count == 48
              else {"swiglu_t1024_queue_edges"}))
    if evidence["campaign"] not in expected_campaigns:
        raise ValueError(
            f"{owner} exact request service {service!r} evidence campaign "
            f"must be one of {sorted(expected_campaigns)}")
    if evidence["lookup_policy"] != (
            "exact_worker_count_direction_and_payload_only"):
        raise ValueError(
            f"{owner} exact request service {service!r} evidence lookup "
            "policy changed")
    if evidence["production_or_evaluation_duration_consumed"] is not False:
        raise ValueError(
            f"{owner} exact request service {service!r} evidence cannot "
            "consume production or evaluation duration")
    if evidence["mechanism_publishable"] is not True:
        raise ValueError(
            f"{owner} exact request service {service!r} evidence is not "
            "publishable")
    published_cells = evidence["published_cells"]
    if not isinstance(published_cells, dict) or not published_cells:
        raise TypeError(
            f"{owner} exact request service {service!r} published_cells "
            "must be a non-empty mapping")
    prefix = f"{worker_count}:{direction}:"
    selected = {}
    for cell, result in published_cells.items():
        if not isinstance(cell, str):
            raise TypeError(
                f"{owner} exact request service {service!r} evidence cell "
                "names must be strings")
        parts = cell.split(":")
        if (len(parts) != 3 or not parts[0].isdecimal()
                or (int(parts[0]), parts[1])
                not in _SWIGLU_QUEUE_EDGE_EVIDENCE_DOMAINS):
            raise ValueError(
                f"{owner} exact request service {service!r} has an unknown "
                f"published cell: {cell!r}")
        if not cell.startswith(prefix):
            continue
        payload = cell.removeprefix(prefix)
        if not payload.isdecimal() or str(int(payload)) != payload:
            raise ValueError(
                f"{owner} exact request service {service!r} evidence "
                f"payload is not canonical: {cell!r}")
        if (not isinstance(result, dict)
                or "service_time_ns" not in result):
            raise ValueError(
                f"{owner} exact request service {service!r} evidence cell "
                f"schema changed: {cell!r}")
        service_time_ns = result["service_time_ns"]
        if (isinstance(service_time_ns, bool)
                or not isinstance(service_time_ns, (int, float))
                or not math.isfinite(service_time_ns)
                or service_time_ns <= 0):
            raise ValueError(
                f"{owner} exact request service {service!r} evidence cell "
                f"has invalid service time: {cell!r}")
        selected[int(payload)] = float(service_time_ns)
    if not selected:
        raise ValueError(
            f"{owner} exact request service {service!r} has no published "
            "evidence cells")
    return selected
def _load_exact_request_evidence(
        owner, service, spec, repository_root: Path):
    evidence_ref = spec["evidence_ref"].strip()
    relative_path = Path(evidence_ref)
    if relative_path.is_absolute():
        raise ValueError(
            f"{owner} exact request service {service!r} evidence_ref must "
            "be repository-relative")
    repository_root = repository_root.resolve()
    evidence_path = (repository_root / relative_path).resolve()
    try:
        evidence_path.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(
            f"{owner} exact request service {service!r} evidence_ref escapes "
            "the repository") from exc
    if not evidence_path.is_file():
        raise ValueError(
            f"{owner} exact request service {service!r} evidence_ref does "
            f"not resolve to a retained file: {evidence_ref!r}")
    evidence_bytes = evidence_path.read_bytes()
    actual_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
    if actual_sha256 != spec["evidence_sha256"]:
        raise ValueError(
            f"{owner} exact request service {service!r} evidence SHA256 "
            f"mismatch: declared={spec['evidence_sha256']}, "
            f"actual={actual_sha256}")
    try:
        evidence = json.loads(evidence_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{owner} exact request service {service!r} evidence_ref must "
            "contain valid UTF-8 JSON") from exc
    if not isinstance(evidence, dict):
        raise TypeError(
            f"{owner} exact request service {service!r} evidence must be a "
            "mapping")
    if service in _SWIGLU_QUEUE_EDGE_EVIDENCE_KEYS:
        return evidence_ref, _load_swiglu_queue_edge_evidence(
            owner, service, evidence)
    if service in _ROPE_KEY22_FIRST_Q_SLICE_EVIDENCE_KEYS:
        return evidence_ref, _load_rope_key22_first_q_slice_evidence(
            owner, service, evidence)
    if service in _RMSNORM_KEY0_H2048_MTE_EDGE_EVIDENCE_KEYS:
        return evidence_ref, _load_rmsnorm_key0_h2048_mte_edge_evidence(
            owner, service, evidence)
    missing_evidence_fields = (
        _EXACT_REQUEST_EVIDENCE_REQUIRED_FIELDS - set(evidence))
    if missing_evidence_fields:
        raise ValueError(
            f"{owner} exact request service {service!r} evidence is missing "
            f"fields: {sorted(missing_evidence_fields)}")
    evidence_identity = {
        "service": evidence["service"],
        "request_size_unit": evidence["request_size_unit"],
        "service_time_unit": evidence["service_time_unit"],
        "lookup_policy": evidence["lookup_policy"],
        "production_or_evaluation_duration_consumed": evidence[
            "production_or_evaluation_duration_consumed"],
    }
    declared_identity = {
        "service": service,
        "request_size_unit": spec["request_size_unit"],
        "service_time_unit": spec["service_time_unit"],
        "lookup_policy": spec["lookup_policy"],
        "production_or_evaluation_duration_consumed": spec[
            "production_or_evaluation_duration_consumed"],
    }
    if evidence_identity != declared_identity:
        raise ValueError(
            f"{owner} exact request service {service!r} inline identity "
            "disagrees with its evidence")
    evidence_entries = evidence["service_time_ns_by_request_size"]
    if not isinstance(evidence_entries, dict) or not evidence_entries:
        raise TypeError(
            f"{owner} exact request service {service!r} evidence entries "
            "must be a non-empty mapping")
    normalized_evidence_entries = {}
    for raw_request_size, raw_service_time_ns in evidence_entries.items():
        if (not isinstance(raw_request_size, str)
                or not raw_request_size.isdecimal()
                or str(int(raw_request_size)) != raw_request_size):
            raise ValueError(
                f"{owner} exact request service {service!r} evidence request "
                "sizes must be canonical positive decimal strings")
        request_size = int(raw_request_size)
        if request_size <= 0:
            raise ValueError(
                f"{owner} exact request service {service!r} evidence request "
                "sizes must be positive")
        if (isinstance(raw_service_time_ns, bool)
                or not isinstance(raw_service_time_ns, (int, float))):
            raise TypeError(
                f"{owner} exact request service {service!r} evidence service "
                "times must be numeric")
        service_time_ns = float(raw_service_time_ns)
        if not math.isfinite(service_time_ns) or service_time_ns <= 0:
            raise ValueError(
                f"{owner} exact request service {service!r} evidence service "
                "times must be finite and positive")
        normalized_evidence_entries[request_size] = service_time_ns
    return evidence_ref, normalized_evidence_entries


def _normalize_exact_request_services(owner, value, repository_root: Path):
    if not isinstance(value, dict) or not value:
        raise TypeError(f"{owner} exact_request_services must be a mapping")
    normalized = {}
    for service, spec in value.items():
        if service not in _EXACT_REQUEST_SERVICE_NAMES:
            raise ValueError(
                f"{owner} exact request service must be one of "
                f"{sorted(_EXACT_REQUEST_SERVICE_NAMES)}, got {service!r}")
        if not isinstance(spec, dict) or set(spec) != (
                _EXACT_REQUEST_SERVICE_FIELDS):
            raise ValueError(
                f"{owner} exact request service {service!r} schema changed")
        expected_request_size_unit = _EXACT_REQUEST_SERVICE_UNITS[service]
        if spec["request_size_unit"] != expected_request_size_unit:
            raise ValueError(
                f"{owner} exact request service {service!r} requires "
                f"{expected_request_size_unit} request sizes")
        expected_service_time_unit = _EXACT_REQUEST_SERVICE_TIME_UNITS[service]
        if spec["service_time_unit"] != expected_service_time_unit:
            raise ValueError(
                f"{owner} exact request service {service!r} requires "
                f"{expected_service_time_unit} service times")
        if spec["lookup_policy"] != "exact_only":
            raise ValueError(
                f"{owner} exact request service {service!r} must use "
                "exact_only lookup")
        if (not isinstance(spec["evidence_ref"], str)
                or not spec["evidence_ref"].strip()):
            raise ValueError(
                f"{owner} exact request service {service!r} requires an "
                "evidence_ref")
        evidence_sha256 = spec["evidence_sha256"]
        if (not isinstance(evidence_sha256, str)
                or len(evidence_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in evidence_sha256)):
            raise ValueError(
                f"{owner} exact request service {service!r} evidence_sha256 "
                "must be lowercase SHA256")
        if spec["production_or_evaluation_duration_consumed"] is not False:
            raise ValueError(
                f"{owner} exact request service {service!r} cannot consume "
                "production or evaluation duration")
        evidence_ref, evidence_service_time_ns = (
            _load_exact_request_evidence(
                owner, service, spec, repository_root))
        entries = spec["entries"]
        if not isinstance(entries, list) or not entries:
            raise TypeError(
                f"{owner} exact request service {service!r} entries must be "
                "a non-empty list")
        service_time_ns_by_request_size = {}
        for entry in entries:
            if (not isinstance(entry, dict)
                    or set(entry) != _EXACT_REQUEST_SERVICE_ENTRY_FIELDS):
                raise ValueError(
                    f"{owner} exact request service {service!r} entry schema "
                    "changed")
            request_size = entry["request_size"]
            if (isinstance(request_size, bool)
                    or not isinstance(request_size, int)
                    or request_size <= 0):
                raise ValueError(
                    f"{owner} exact request service {service!r} request_size "
                    "must be a positive integer")
            service_time_ns = entry["service_time_ns"]
            if (isinstance(service_time_ns, bool)
                    or not isinstance(service_time_ns, (int, float))):
                raise TypeError(
                    f"{owner} exact request service {service!r} service time "
                    "must be numeric")
            normalized_service_time_ns = float(service_time_ns)
            if (not math.isfinite(normalized_service_time_ns)
                    or normalized_service_time_ns <= 0):
                raise ValueError(
                    f"{owner} exact request service {service!r} service time "
                    "must be finite and positive")
            if request_size in service_time_ns_by_request_size:
                raise ValueError(
                    f"{owner} exact request service {service!r} duplicates "
                    f"request size {request_size}")
            service_time_ns_by_request_size[request_size] = (
                normalized_service_time_ns)
        if service_time_ns_by_request_size != evidence_service_time_ns:
            raise ValueError(
                f"{owner} exact request service {service!r} inline entries "
                "disagree with its evidence")
        normalized[service] = {
            "request_size_unit": spec["request_size_unit"],
            "service_time_unit": spec["service_time_unit"],
            "lookup_policy": spec["lookup_policy"],
            "evidence_ref": evidence_ref,
            "evidence_sha256": evidence_sha256,
            "production_or_evaluation_duration_consumed": False,
            "service_time_s_by_request_size": (
                {
                    request_size: service_time_ns * 1e-9
                    for request_size, service_time_ns
                    in service_time_ns_by_request_size.items()
                }),
        }
    return normalized


def _normalize_component_config(cfg):
    missing = _COMPONENT_REQUIRED_FIELDS - set(cfg)
    if missing:
        raise ValueError(f"component missing required fields: {sorted(missing)}")
    unknown = set(cfg) - _COMPONENT_FIELDS
    if unknown:
        raise ValueError(
            f"component {cfg['name']!r} has unknown fields: {sorted(unknown)}")
    normalized = dict.fromkeys(_COMPONENT_OPTIONAL_FIELDS)
    normalized.update({
        name: cfg[name] for name in _COMPONENT_OPTIONAL_FIELDS & set(cfg)
    })
    normalized.update({name: cfg[name] for name in _COMPONENT_REQUIRED_FIELDS})
    return normalized


class Component:
    """A node in the hardware component tree.

    Leaf components (no children) with a `role` become resources.
    Branch components (with children) define hierarchy only.

    Key fields:
        name: component name (also used for clock domain binding)
        role: semantic role for formula.py (e.g. 'compute_tensor', 'dram')
        count: number of instances of this component per parent
        throughput_per_clk: bytes or ops per clock per instance (leaf only)
        variants: list of {name, throughput_per_clk} for mutually exclusive
                  variants (e.g. fp16/bf16/int8 for Tensor Core)
        primary_variant: variant that owns the component's canonical role
        children: list of child Components
        clock_domain: explicit clock domain override (if not set, inherited
                      from parent or matched by name to clock_domains)
        size_bytes: memory size (for memory resources)
        num_partitions: partition count (for memory resources)
        traffic_filter: 'lru' | 'page_conflict' | null
        scope: explicit 'shared' | 'per_unit' on every resource leaf
        type: explicit 'compute' | 'memory' on every leaf
    """

    def __init__(self, cfg, parent_clock_domain=None, clock_domains=None,
                 repository_root=None):
        """Parse a component from YAML config dict.

        Required fields: name, count
        Optional fields (YAML schema, omission is valid):
            role, description, clock_domain, type,
            throughput_per_clk, variants, primary_variant, traffic_filter,
            size_bytes, num_partitions, scope, children
        """
        cfg = _normalize_component_config(cfg)
        self.name = cfg["name"]
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("component name must be a non-empty string")
        self.role = cfg["role"]
        if isinstance(cfg["count"], bool) or not isinstance(cfg["count"], int):
            raise TypeError(f"component {self.name!r} count must be an integer")
        self.count = cfg["count"]
        if self.count <= 0:
            raise ValueError(f"component {self.name!r} count must be positive")
        self.description = cfg["description"]

        # Clock domain: explicit > name-match > parent > None
        explicit_cd = cfg["clock_domain"]
        if (explicit_cd is not None
                and (not isinstance(explicit_cd, str) or not explicit_cd.strip())):
            raise ValueError(
                f"component {self.name!r} clock_domain must be non-empty")
        if explicit_cd is not None:
            self.clock_domain = explicit_cd
        elif clock_domains is not None and self.name in clock_domains:
            # Name matches a clock domain → use it (overrides parent)
            self.clock_domain = self.name
            if parent_clock_domain is not None and self.name != parent_clock_domain:
                warnings.warn(
                    f"Component '{self.name}' clock_domain='{self.name}' "
                    f"overrides parent '{parent_clock_domain}'"
                )
        elif parent_clock_domain is not None:
            self.clock_domain = parent_clock_domain
        else:
            self.clock_domain = None

        # Warn if explicit overrides parent
        if explicit_cd is not None and parent_clock_domain is not None and explicit_cd != parent_clock_domain:
            warnings.warn(
                f"Component '{self.name}' clock_domain='{explicit_cd}' "
                f"overrides parent '{parent_clock_domain}'"
            )

        # Leaf execution type is semantic behavior and must never be inferred
        # from the presence of throughput fields. Branches alone omit it.
        declared_type = cfg["type"]
        if cfg["children"] is not None and declared_type is None:
            self.type = "branch"
        elif cfg["children"] is not None:
            raise ValueError(
                f"branch component {self.name!r} cannot declare type "
                f"{declared_type!r}")
        elif declared_type in ("compute", "memory"):
            self.type = declared_type
        elif declared_type is None:
            raise ValueError(
                f"leaf component {self.name!r} requires an explicit type")
        else:
            raise ValueError(
                f"component {self.name!r} has unknown type {declared_type!r}")

        # Leaf fields
        self.throughput_per_clk = cfg["throughput_per_clk"]
        self.capacity_per_second = cfg["capacity_per_second"]
        if self.capacity_per_second is not None:
            if (isinstance(self.capacity_per_second, bool)
                    or not isinstance(self.capacity_per_second, (int, float))
                    or not math.isfinite(self.capacity_per_second) or self.capacity_per_second <= 0):
                raise ValueError("absolute resource capacity must be finite and positive")
            if (self.throughput_per_clk is not None or cfg["variants"] is not None
                    or self.clock_domain is not None or cfg["children"] is not None):
                raise ValueError("absolute resource capacity cannot declare clocks, cycles, variants or children")
        if self.throughput_per_clk is not None:
            if (isinstance(self.throughput_per_clk, bool)
                    or not isinstance(self.throughput_per_clk, (int, float))):
                raise TypeError(
                    f"component {self.name!r} throughput_per_clk must be numeric")
            throughput = float(self.throughput_per_clk)
            if not math.isfinite(throughput) or throughput <= 0:
                raise ValueError(
                    f"component {self.name!r} throughput_per_clk must be "
                    "finite and positive")
            self.throughput_per_clk = throughput
        self.variants = cfg["variants"]
        self.primary_variant = cfg["primary_variant"]
        if self.variants is not None:
            if not isinstance(self.variants, list) or not self.variants:
                raise ValueError(
                    f"component {self.name!r} variants must be a non-empty list")
            variant_names = set()
            for variant in self.variants:
                if set(variant) != {"name", "throughput_per_clk"}:
                    raise ValueError(
                        f"component {self.name!r} variant requires exactly "
                        "name and throughput_per_clk")
                variant_name = variant["name"]
                if (not isinstance(variant_name, str)
                        or not variant_name.strip()):
                    raise ValueError(
                        f"component {self.name!r} variant name must be non-empty")
                if variant_name in variant_names:
                    raise ValueError(
                        f"component {self.name!r} has duplicate variant "
                        f"{variant_name!r}")
                variant_names.add(variant_name)
                if (isinstance(variant["throughput_per_clk"], bool)
                        or not isinstance(
                            variant["throughput_per_clk"], (int, float))):
                    raise TypeError(
                        f"component {self.name!r} variant {variant_name!r} "
                        "throughput must be numeric")
                throughput = float(variant["throughput_per_clk"])
                if not math.isfinite(throughput) or throughput <= 0:
                    raise ValueError(
                        f"component {self.name!r} variant {variant_name!r} "
                        "throughput must be finite and positive")
            if (not isinstance(self.primary_variant, str)
                    or not self.primary_variant.strip()):
                raise ValueError(
                    f"component {self.name!r} with variants requires a "
                    "non-empty primary_variant")
            if self.primary_variant not in variant_names:
                raise ValueError(
                    f"component {self.name!r} primary_variant "
                    f"{self.primary_variant!r} is not a declared variant")
        elif self.primary_variant is not None:
            raise ValueError(
                f"component {self.name!r} declares primary_variant without "
                "variants")

        self.service_rates_per_unit_s = cfg["service_rates_per_unit_s"]
        if self.service_rates_per_unit_s is not None:
            if self.type != "compute":
                raise ValueError(
                    f"non-compute component {self.name!r} cannot declare "
                    "typed service rates")
            if self.role not in TYPED_COMPUTE_SERVICES_BY_ROLE:
                raise ValueError(
                    f"component {self.name!r} typed service rates require "
                    "a registered typed compute role")
            if self.variants is not None:
                raise ValueError(
                    f"variant component {self.name!r} cannot share one "
                    "ambiguous typed service-rate mapping")
            if (not isinstance(self.service_rates_per_unit_s, dict)
                    or not self.service_rates_per_unit_s):
                raise TypeError(
                    f"component {self.name!r} service_rates_per_unit_s must "
                    "be a non-empty mapping")
            normalized_service_rates = {}
            allowed_services = TYPED_COMPUTE_SERVICES_BY_ROLE[self.role]
            for service, rate in self.service_rates_per_unit_s.items():
                if not isinstance(service, str) or not service.strip():
                    raise ValueError(
                        f"component {self.name!r} typed service names must "
                        "be non-empty strings")
                normalized_service = service.strip()
                if normalized_service not in allowed_services:
                    raise ValueError(
                        f"component {self.name!r} typed service must be one "
                        f"of {sorted(allowed_services)}, got "
                        f"{normalized_service!r}")
                if (isinstance(rate, bool)
                        or not isinstance(rate, (int, float))):
                    raise TypeError(
                        f"component {self.name!r} typed service rate "
                        f"{service!r} must be numeric")
                normalized_rate = float(rate)
                if not math.isfinite(normalized_rate) or normalized_rate <= 0:
                    raise ValueError(
                        f"component {self.name!r} typed service rate "
                        f"{service!r} must be finite and positive")
                normalized_service_rates[normalized_service] = normalized_rate
            if len(normalized_service_rates) != len(
                    self.service_rates_per_unit_s):
                raise ValueError(
                    f"component {self.name!r} typed service names must be "
                    "unique after normalization")
            self.service_rates_per_unit_s = normalized_service_rates
        self.service_setup_time_s = cfg["service_setup_time_s"]
        if self.service_setup_time_s is not None:
            if self.type != "compute":
                raise ValueError(
                    f"non-compute component {self.name!r} cannot declare "
                    "typed service setup times")
            if self.role not in TYPED_COMPUTE_SERVICES_BY_ROLE:
                raise ValueError(
                    f"component {self.name!r} typed service setup times "
                    "require a registered typed compute role")
            if self.service_rates_per_unit_s is None:
                raise ValueError(
                    f"component {self.name!r} typed service setup times "
                    "require typed service rates")
            if (not isinstance(self.service_setup_time_s, dict)
                    or not self.service_setup_time_s):
                raise TypeError(
                    f"component {self.name!r} service_setup_time_s must "
                    "be a non-empty mapping")
            normalized_setup_times = {}
            for service, setup_time in self.service_setup_time_s.items():
                if not isinstance(service, str) or not service.strip():
                    raise ValueError(
                        f"component {self.name!r} typed setup service names "
                        "must be non-empty strings")
                normalized_service = service.strip()
                if normalized_service not in self.service_rates_per_unit_s:
                    raise ValueError(
                        f"component {self.name!r} typed setup service "
                        f"{normalized_service!r} has no matching rate")
                if (isinstance(setup_time, bool)
                        or not isinstance(setup_time, (int, float))):
                    raise TypeError(
                        f"component {self.name!r} typed service setup "
                        f"{service!r} must be numeric seconds")
                normalized_setup_time = float(setup_time)
                if (not math.isfinite(normalized_setup_time)
                        or normalized_setup_time <= 0):
                    raise ValueError(
                        f"component {self.name!r} typed service setup "
                        f"{service!r} must be finite positive seconds")
                normalized_setup_times[normalized_service] = (
                    normalized_setup_time)
            if len(normalized_setup_times) != len(self.service_setup_time_s):
                raise ValueError(
                    f"component {self.name!r} typed setup service names "
                    "must be unique after normalization")
            self.service_setup_time_s = normalized_setup_times
        self.active_unit_service_time_s = cfg[
            "active_unit_service_time_s"]
        if self.active_unit_service_time_s is not None:
            if self.type != "compute":
                raise ValueError(
                    f"non-compute component {self.name!r} cannot declare "
                    "active-unit service times")
            if self.role not in TYPED_COMPUTE_SERVICES_BY_ROLE:
                raise ValueError(
                    f"component {self.name!r} active-unit service times "
                    "require a typed compute role")
            if self.variants is not None:
                raise ValueError(
                    f"variant component {self.name!r} cannot share one "
                    "ambiguous active-unit service-time mapping")
            if (not isinstance(self.active_unit_service_time_s, dict)
                    or not self.active_unit_service_time_s):
                raise TypeError(
                    f"component {self.name!r} active_unit_service_time_s "
                    "must be a non-empty mapping")
            normalized_active_unit_services = {}
            allowed_services = TYPED_COMPUTE_SERVICES_BY_ROLE[self.role]
            for service, model in self.active_unit_service_time_s.items():
                if not isinstance(service, str) or not service.strip():
                    raise ValueError(
                        f"component {self.name!r} active-unit service names "
                        "must be non-empty strings")
                normalized_service = service.strip()
                if normalized_service not in allowed_services:
                    raise ValueError(
                        f"component {self.name!r} active-unit service must "
                        f"be one of {sorted(allowed_services)}, got "
                        f"{normalized_service!r}")
                if (not isinstance(model, dict)
                        or set(model) != {
                            "model", "active_units", "service_time_s"}):
                    raise ValueError(
                        f"component {self.name!r} active-unit service "
                        f"{normalized_service!r} schema changed")
                if model["model"] != "piecewise_linear_active_units_v1":
                    raise ValueError(
                        f"component {self.name!r} active-unit service "
                        f"{normalized_service!r} has an unsupported model")
                points = model["active_units"]
                service_times = model["service_time_s"]
                if (not isinstance(points, list)
                        or not isinstance(service_times, list)
                        or len(points) < 2
                        or len(points) != len(service_times)):
                    raise ValueError(
                        f"component {self.name!r} active-unit service "
                        f"{normalized_service!r} requires equal-length "
                        "point and time lists")
                if any(isinstance(point, bool) or not isinstance(point, int)
                       for point in points):
                    raise TypeError(
                        f"component {self.name!r} active-unit points must "
                        "be integers")
                if points != sorted(set(points)) or points[0] != 1:
                    raise ValueError(
                        f"component {self.name!r} active-unit points must "
                        "be unique, increasing, and start at one")
                normalized_times = []
                for service_time in service_times:
                    if (isinstance(service_time, bool)
                            or not isinstance(service_time, (int, float))):
                        raise TypeError(
                            f"component {self.name!r} active-unit service "
                            "times must be numeric seconds")
                    normalized_time = float(service_time)
                    if (not math.isfinite(normalized_time)
                            or normalized_time <= 0):
                        raise ValueError(
                            f"component {self.name!r} active-unit service "
                            "times must be finite and positive")
                    normalized_times.append(normalized_time)
                normalized_active_unit_services[normalized_service] = {
                    "model": model["model"],
                    "active_units": tuple(points),
                    "service_time_s": tuple(normalized_times),
                }
            if len(normalized_active_unit_services) != len(
                    self.active_unit_service_time_s):
                raise ValueError(
                    f"component {self.name!r} active-unit service names "
                    "must be unique after normalization")
            if (self.service_rates_per_unit_s is not None
                    and set(normalized_active_unit_services).intersection(
                        self.service_rates_per_unit_s)):
                raise ValueError(
                    f"component {self.name!r} cannot give one typed service "
                    "both a fixed rate and an active-unit curve")
            self.active_unit_service_time_s = (
                normalized_active_unit_services)
        self.exact_request_services = cfg["exact_request_services"]
        if self.exact_request_services is not None:
            if self.variants is not None:
                raise ValueError(
                    f"variant component {self.name!r} cannot declare exact "
                    "request services")
            service_roles = {
                _EXACT_REQUEST_SERVICE_ROLES[service]
                for service in self.exact_request_services
                if service in _EXACT_REQUEST_SERVICE_ROLES
            }
            unknown_services = (
                set(self.exact_request_services)
                - set(_EXACT_REQUEST_SERVICE_ROLES))
            if unknown_services:
                raise ValueError(
                    f"component {self.name!r} exact request service must be "
                    f"one of {sorted(_EXACT_REQUEST_SERVICE_NAMES)}, got "
                    f"{sorted(unknown_services)}")
            if service_roles != {self.role}:
                raise ValueError(
                    f"component {self.name!r} exact request services require "
                    f"role {self.role!r}, got services for "
                    f"{sorted(service_roles)}")
            self.exact_request_services = _normalize_exact_request_services(
                f"component {self.name!r}", self.exact_request_services,
                repository_root)

        # Memory-specific fields
        self.traffic_filter = cfg["traffic_filter"]
        for field in ("size_bytes", "num_partitions"):
            if (cfg[field] is not None
                    and (isinstance(cfg[field], bool)
                         or not isinstance(cfg[field], int))):
                raise TypeError(
                    f"component {self.name!r} {field} must be an integer")
        self.size_bytes = cfg["size_bytes"]
        self.num_partitions = cfg["num_partitions"]
        if self.size_bytes is not None and self.size_bytes <= 0:
            raise ValueError(
                f"component {self.name!r} size_bytes must be positive")
        if self.num_partitions is not None and self.num_partitions <= 0:
            raise ValueError(
                f"component {self.name!r} num_partitions must be positive")
        if self.type == "memory":
            if self.traffic_filter not in (None, "lru", "page_conflict"):
                raise ValueError(
                    f"component {self.name!r} has invalid traffic_filter")
            if (self.traffic_filter in ("lru", "page_conflict")
                    and self.size_bytes is None):
                raise ValueError(
                    f"component {self.name!r} requires size_bytes for "
                    f"{self.traffic_filter}")
        elif any(value is not None for value in (
                self.traffic_filter, self.size_bytes, self.num_partitions)):
            raise ValueError(
                f"non-memory component {self.name!r} contains memory fields")

        # Scope is semantic hardware behavior and must never be inferred from
        # a component name or tree position.
        self.scope = cfg["scope"]
        if self.scope not in (None, "shared", "per_unit"):
            raise ValueError(
                f"component {self.name!r} has unknown scope {self.scope!r}")

        # Children
        self.children = []
        if cfg["children"] is not None:
            if not isinstance(cfg["children"], list) or not cfg["children"]:
                raise ValueError(
                    f"component {self.name!r} children must be a non-empty list")
            for child_cfg in cfg["children"]:
                child = Component(child_cfg, parent_clock_domain=self.clock_domain,
                                  clock_domains=clock_domains,
                                  repository_root=repository_root)
                self.children.append(child)
        if self.children:
            if self.type != "branch":
                raise ValueError(
                    f"component {self.name!r} with children must be a branch")
            if self.role is not None:
                raise ValueError(
                    f"branch component {self.name!r} cannot declare a role")
            if self.scope is not None:
                raise ValueError(
                    f"branch component {self.name!r} cannot declare scope")
        else:
            if not isinstance(self.role, str) or not self.role.strip():
                raise ValueError(
                    f"leaf component {self.name!r} requires a non-empty role")
            if self.clock_domain is None and self.capacity_per_second is None:
                raise ValueError(
                    f"resource leaf {self.name!r} requires a clock domain")
            if self.scope is None:
                raise ValueError(
                    f"resource leaf {self.name!r} requires an explicit scope")

    def is_leaf(self):
        return len(self.children) == 0

    def has_role(self):
        return self.role is not None

    def __repr__(self):
        if self.children:
            return f"Component({self.name}, count={self.count}, children={len(self.children)})"
        return f"Component({self.name}, role={self.role}, count={self.count})"


class ComponentTree:
    """Parse YAML component tree and produce flat resource list.

    The tree is parsed from the `components` section of the YAML.
    Clock domains bind by name and propagate to descendants.
    Leaf nodes with roles become flat resources for formula.py.
    """

    def __init__(self, cfg, repository_root):
        """Parse component tree from YAML config dict.

        Args:
            cfg: the full YAML config dict (must have 'components' and 'clock_domains')
        """
        self.clock_domains = cfg["clock_domains"]
        if not isinstance(repository_root, Path):
            raise TypeError("component tree repository_root must be a Path")
        self.repository_root = repository_root
        self.root_children = []

        if "components" not in cfg:
            raise ValueError("YAML must have 'components' section")

        for comp_cfg in cfg["components"]:
            comp = Component(
                comp_cfg, clock_domains=self.clock_domains,
                repository_root=self.repository_root)
            self.root_children.append(comp)

        # Build flat resource list
        self._flat_resources = []
        self._flat_resource_names = []
        for comp in self.root_children:
            self._flatten(comp, count_product=1)

    def _flatten(self, comp, count_product):
        """Recursively flatten tree into resource-like dicts.

        Each leaf with a role becomes a flat resource dict compatible
        with the existing Resource class.
        """
        current_count = count_product * comp.count

        if comp.is_leaf() and comp.has_role():
            # This is a resource leaf
            if comp.variants:
                # Create resources for each variant with unique role
                # The declared primary variant gets the component's own role;
                # other variants get role "{role}_{variant_name}"
                # to avoid role collisions.
                declared_variants = {
                    variant["name"] for variant in comp.variants
                }
                for var in comp.variants:
                    var_name = f"{comp.name}_{var['name']}"
                    var_tp = var["throughput_per_clk"]
                    if var["name"] == comp.primary_variant:
                        var_role = comp.role
                    elif var["name"] in declared_variants:
                        var_role = f"{comp.role}_{var['name']}"
                    else:
                        raise AssertionError(
                            "component validation allowed an unknown variant: "
                            f"{var['name']!r}")
                    res_dict = self._make_resource_dict(
                        comp, var_name, var_tp, current_count, override_role=var_role)
                    self._flat_resources.append(res_dict)
                    self._flat_resource_names.append(var_name)
            elif comp.throughput_per_clk is not None:
                res_dict = self._make_resource_dict(
                    comp, comp.name, comp.throughput_per_clk, current_count)
                self._flat_resources.append(res_dict)
                self._flat_resource_names.append(comp.name)
            elif comp.capacity_per_second is not None:
                res_dict = self._make_resource_dict(comp, comp.name, None, current_count)
                self._flat_resources.append(res_dict)
                self._flat_resource_names.append(comp.name)
            else:
                raise ValueError(
                    f"Leaf component '{comp.name}' with role '{comp.role}' "
                    f"must have throughput_per_clk or variants"
                )
        elif not comp.is_leaf():
            # Recurse into children
            for child in comp.children:
                self._flatten(child, current_count)
        else:
            raise RuntimeError(
                "component-tree validation allowed an unowned leaf: "
                f"{comp.name!r}")

    def _make_resource_dict(self, comp, name, throughput_per_clk, count_product, override_role=None):
        """Create a flat resource dict compatible with Resource class."""
        if comp.type not in ("compute", "memory"):
            raise ValueError(
                f"resource leaf {comp.name!r} has invalid type {comp.type!r}")
        d = {
            "type": comp.type,
            "name": name,
            "role": override_role if override_role is not None else comp.role,
            "description": comp.description,
            "clock_domain": comp.clock_domain,
            "scope": comp.scope,
            "capacity_per_unit_per_clk": throughput_per_clk,
            "capacity_per_second": (None if comp.capacity_per_second is None
                else comp.capacity_per_second * count_product),
            "num_units": count_product,
            "size_bytes": comp.size_bytes,
            "num_partitions": comp.num_partitions,
            "service_rates_per_unit_s": comp.service_rates_per_unit_s,
            "service_setup_time_s": comp.service_setup_time_s,
            "active_unit_service_time_s": (
                comp.active_unit_service_time_s),
            "exact_request_services": comp.exact_request_services,
        }
        # Memory-specific fields
        if comp.traffic_filter is not None:
            d["traffic_filter"] = comp.traffic_filter
        elif comp.type == "memory":
            d["traffic_filter"] = None
        return d

    def get_flat_resources(self):
        """Return list of flat resource dicts (compatible with Resource class)."""
        return self._flat_resources

    def get_flat_resource_names(self):
        """Return list of flat resource names."""
        return self._flat_resource_names
