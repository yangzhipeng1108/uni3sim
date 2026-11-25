"""Materialize and validate selected-step injection inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from evaluation.serving.traces.scheduling_trajectory_replay import (
    predict_scheduling_trajectory_steps,
    require_exact_step_fingerprints,
)
from src.serving.host_timing import (
    HostLaunchProfile,
    HostRuntimePrepareProfile,
    RuntimePrepareEvent,
)
from src.serving.kernel_lowering.kernel_plan import KernelPlan
from src.serving.workload.graph_padding import GraphExecutionConfig

from evaluation.json_shape import require_object_dict as _mapping

from evaluation.json_shape import require_list_or_tuple as _sequence

from evaluation.json_shape import require_finite as _finite


_PHASES = ("prefill", "mixed", "decode")






def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{label} must be an integer >= {minimum}")
    return value




def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _token_sequence(value: object, label: str) -> list[int]:
    sequence = _sequence(value, label)
    result = []
    for index, token_id in enumerate(sequence):
        result.append(_integer(token_id, f"{label}[{index}]"))
    return result


def materialize_selected_step_token_prefixes(
    selected_step_manifest: Mapping[str, object],
    prompt_token_ids_by_request: Mapping[int, Sequence[int]],
    generated_token_ids_by_request: Mapping[int, Sequence[int]],
    content_source: Mapping[str, object],
) -> dict[str, object]:
    """Build exact token prefixes needed to establish a selected-step state.

    The prompt token stream and generated prefixes are intentionally supplied
    by the hardware runner rather than inferred from token counts.  This keeps
    the structural manifest useful for ranking while making a hardware probe
    fail closed when actual token values have not been reconstructed.
    """

    if selected_step_manifest["kind"] != "selected_step_injection_input":
        raise ValueError("selected-step token materialization requires an input manifest")
    expected = _mapping(
        selected_step_manifest["expected_fingerprint"], "expected fingerprint"
    )
    workload = [
        _mapping(value, "request workload")
        for value in _sequence(expected["request_workload"], "request workload")
    ]
    expected_indexes = [_integer(row["request_idx"], "request index") for row in workload]
    if len(set(expected_indexes)) != len(expected_indexes):
        raise ValueError("selected-step request workload contains duplicate request indexes")
    if set(prompt_token_ids_by_request) != set(expected_indexes):
        raise ValueError("prompt token IDs do not cover exactly the selected requests")
    decode_indexes = {
        _integer(row["request_idx"], "request index")
        for row in workload
        if _integer(row["decode_tokens"], "decode tokens") > 0
    }
    if set(generated_token_ids_by_request) != decode_indexes:
        raise ValueError(
            "generated token IDs do not cover exactly the selected decode requests"
        )

    requests = []
    for row in workload:
        request_idx = _integer(row["request_idx"], "request index")
        prompt_token_ids = _token_sequence(
            prompt_token_ids_by_request[request_idx],
            f"request {request_idx} prompt token IDs",
        )
        prefill_tokens = _integer(row["prefill_tokens"], "prefill tokens")
        decode_tokens = _integer(row["decode_tokens"], "decode tokens")
        if bool(prefill_tokens) == bool(decode_tokens):
            raise ValueError(
                f"request {request_idx} must be exactly prefill or decode"
            )
        computed_tokens = _integer(
            row["computed_tokens_before_step"], "computed tokens before step"
        )
        if prefill_tokens:
            if computed_tokens != 0 or prefill_tokens != len(prompt_token_ids):
                raise ValueError(
                    "selected-step injection currently requires a fresh, complete "
                    f"prefill request: request={request_idx}"
                )
            generated_prefix_count = 0
            prefix_token_ids = []
            injection_prompt_token_ids = prompt_token_ids
            role = "fresh_prefill_prompt"
        else:
            generated_token_ids = _token_sequence(
                generated_token_ids_by_request[request_idx],
                f"request {request_idx} generated token IDs",
            )
            generated_prefix_count = computed_tokens - len(prompt_token_ids)
            if generated_prefix_count < 0:
                raise ValueError(
                    f"request {request_idx} computed-token count predates its prompt: "
                    f"computed={computed_tokens}, prompt={len(prompt_token_ids)}"
                )
            if len(generated_token_ids) < generated_prefix_count:
                raise ValueError(
                    f"request {request_idx} generated-token prefix is incomplete: "
                    f"required={generated_prefix_count}, "
                    f"available={len(generated_token_ids)}"
                )
            generated_prefix = generated_token_ids[:generated_prefix_count]
            prefix_token_ids = prompt_token_ids + generated_prefix
            injection_prompt_token_ids = prefix_token_ids
            role = "decode_state_prefix"
        if len(prefix_token_ids) != computed_tokens:
            raise AssertionError("selected-step token prefix length is inconsistent")
        requests.append({
            "request_idx": request_idx,
            "request_id": row["request_id"],
            "role": role,
            "prompt_token_count": len(prompt_token_ids),
            "generated_prefix_token_count": generated_prefix_count,
            "computed_tokens_before_step": computed_tokens,
            "prefix_token_ids": prefix_token_ids,
            "prefix_token_ids_sha256": _canonical_sha256(prefix_token_ids),
            "injection_prompt_token_ids": injection_prompt_token_ids,
            "injection_prompt_token_ids_sha256": _canonical_sha256(
                injection_prompt_token_ids
            ),
        })

    return {
        "schema_version": 2,
        "kind": "selected_step_token_prefixes",
        "cell_id": selected_step_manifest["cell_id"],
        "step_id": selected_step_manifest["step_id"],
        "expected_step_fingerprint_sha256": _canonical_sha256(expected),
        "request_count": len(requests),
        "total_computed_tokens_before_step": sum(
            request["computed_tokens_before_step"] for request in requests
        ),
        "content_source": dict(content_source),
        "requests": requests,
    }


def validate_selected_step_injection_trace(
    selected_step_manifest: Mapping[str, object],
    token_prefix_payload: Mapping[str, object],
    injection_trace_path: Path,
    injection_replay_payload: Mapping[str, object],
    *,
    independent_eager: bool = False,
) -> dict[str, object]:
    """Find the raw injected step, then validate its replayed full fingerprint."""

    if token_prefix_payload["kind"] != "selected_step_token_prefixes":
        raise ValueError("injection validation requires token-prefix payload")
    expected = _mapping(
        selected_step_manifest["expected_fingerprint"], "expected fingerprint"
    )
    if token_prefix_payload["cell_id"] != selected_step_manifest["cell_id"]:
        raise ValueError("INVALID EXPERIMENT — token-prefix cell identity")
    if token_prefix_payload["step_id"] != selected_step_manifest["step_id"]:
        raise ValueError("INVALID EXPERIMENT — token-prefix step identity")
    if token_prefix_payload["expected_step_fingerprint_sha256"] != _canonical_sha256(
        expected
    ):
        raise ValueError("INVALID EXPERIMENT — token-prefix fingerprint digest")
    prefix_requests = [
        _mapping(value, "token-prefix request")
        for value in _sequence(token_prefix_payload["requests"], "token-prefix requests")
    ]
    prefix_lengths = {
        _integer(row["request_idx"], "token-prefix request index"): len(
            _token_sequence(row["prefix_token_ids"], "prefix token IDs")
        )
        for row in prefix_requests
    }
    expected_workload = [
        dict(_mapping(value, "expected request workload"))
        for value in _sequence(expected["request_workload"], "expected request workload")
    ]
    expected_indexes = [row["request_idx"] for row in expected_workload]
    if set(prefix_lengths) != set(expected_indexes):
        raise ValueError("INVALID EXPERIMENT — token-prefix request coverage")
    for row in expected_workload:
        request_idx = row["request_idx"]
        if prefix_lengths[request_idx] != row["computed_tokens_before_step"]:
            raise ValueError(
                "INVALID EXPERIMENT — token-prefix computed-token length: "
                f"request={request_idx}"
            )

    raw_rows = _load_json_lines(injection_trace_path)
    computed_by_request: dict[int, int] = {}
    matches = []
    for index, row in enumerate(raw_rows):
        workload = _raw_workload(row, computed_by_request)
        if workload == expected_workload:
            graph = _raw_graph_identity(row)
            expected_graph = (
                _independent_eager_graph_identity(expected)
                if independent_eager
                else _mapping(
                    expected["graph_execution"], "expected graph execution"
                )
            )
            graph_matches = _raw_graph_matches_expected(
                expected_graph, graph
            )
            if graph_matches:
                matches.append((index, row))
        for request in workload:
            request_idx = request["request_idx"]
            computed_by_request[request_idx] = request[
                "kv_context_tokens_after_step"
            ]
    if len(matches) != 1:
        raise ValueError(
            "INVALID EXPERIMENT — exact injected structural step count: "
            f"observed={len(matches)}, expected=1"
        )
    match_index, matched = matches[0]
    injected_step_id = matched["step_id"]

    if (
        "schema_version" not in injection_replay_payload
        or injection_replay_payload["schema_version"] not in (5, 6)
    ):
        raise ValueError("INVALID EXPERIMENT — injection_replay.schema_version")
    if (
        "forward_duration_observation_consumed" not in injection_replay_payload
        or injection_replay_payload[
            "forward_duration_observation_consumed"
        ] is not False
    ):
        raise ValueError(
            "INVALID EXPERIMENT — "
            "injection_replay.forward_duration_observation_consumed"
        )
    if (
        "cell_id" not in injection_replay_payload
        or injection_replay_payload["cell_id"] != selected_step_manifest["cell_id"]
    ):
        raise ValueError("INVALID EXPERIMENT — injection_replay.cell_id")
    source_sha256 = (
        injection_replay_payload["source_sha256"]
        if "source_sha256" in injection_replay_payload
        else None
    )
    if not isinstance(source_sha256, dict):
        raise ValueError("INVALID EXPERIMENT — injection_replay.source_sha256")
    trace_sha256 = _sha256(injection_trace_path)
    if (
        "scheduler_trace" not in source_sha256
        or source_sha256["scheduler_trace"] != trace_sha256
    ):
        raise ValueError(
            "INVALID EXPERIMENT — "
            "injection_replay.source_sha256.scheduler_trace"
        )
    replay_comparisons = (
        injection_replay_payload["step_comparison"]
        if "step_comparison" in injection_replay_payload
        else None
    )
    if not isinstance(replay_comparisons, (list, tuple)):
        raise ValueError("INVALID EXPERIMENT — injection_replay.step_comparison")
    replay_matches = []
    for comparison in replay_comparisons:
        if not isinstance(comparison, dict):
            raise ValueError(
                "INVALID EXPERIMENT — injection_replay.step_comparison.row"
            )
        if "step_id" not in comparison:
            raise ValueError(
                "INVALID EXPERIMENT — injection_replay.step_comparison.step_id"
            )
        if comparison["step_id"] == injected_step_id:
            replay_matches.append(comparison)
    if len(replay_matches) != 1:
        raise ValueError(
            "INVALID EXPERIMENT — exact injected replay step count: "
            f"observed={len(replay_matches)}, expected=1"
        )
    replay_match = replay_matches[0]
    replay_step_mapping = (
        injection_replay_payload["step_id_mapping"]
        if "step_id_mapping" in injection_replay_payload
        else None
    )
    if not isinstance(replay_step_mapping, dict):
        raise ValueError("INVALID EXPERIMENT — injection_replay.step_id_mapping")
    expected_step_mapping = {
        "canonical_workload_step_id": selected_step_manifest["step_id"],
        "injected_trace_step_id": injected_step_id,
    }
    if replay_step_mapping != expected_step_mapping:
        raise ValueError(
            "INVALID EXPERIMENT — injection_replay.step_id_mapping: "
            f"observed={replay_step_mapping!r}, expected={expected_step_mapping!r}"
        )
    injected_fingerprint = (
        replay_match["step_fingerprint"]
        if "step_fingerprint" in replay_match
        else None
    )
    if not isinstance(injected_fingerprint, dict):
        raise ValueError(
            "INVALID EXPERIMENT — injection_replay.step_fingerprint"
        )
    mapped_canonical_fingerprint = dict(expected)
    mapped_canonical_fingerprint["step_id"] = injected_step_id
    if independent_eager:
        mapped_canonical_fingerprint["graph_execution"] = (
            _independent_eager_graph_identity(expected)
        )
        for field in (
            "normalized_launch_program_sha256",
            "normalized_launch_call_count",
        ):
            if field not in injected_fingerprint:
                raise ValueError(
                    f"INVALID EXPERIMENT — injection_replay.step_fingerprint.{field}"
                )
            mapped_canonical_fingerprint[field] = injected_fingerprint[field]
    require_exact_step_fingerprints(
        mapped_canonical_fingerprint,
        injected_fingerprint,
    )

    previous = raw_rows[match_index - 1] if match_index else None
    following = (
        _boundary_summary(raw_rows[match_index + 1], matched)
        if match_index + 1 < len(raw_rows)
        else None
    )
    content_source = _mapping(
        token_prefix_payload["content_source"], "content source"
    )
    historical_outputs_observed = content_source[
        "historical_output_token_ids_observed"
    ]
    if not isinstance(historical_outputs_observed, bool):
        raise TypeError("historical output-token observation flag must be boolean")
    return {
        "schema_version": 1,
        "kind": "selected_step_injection_validation",
        "cell_id": selected_step_manifest["cell_id"],
        "canonical_step_id": selected_step_manifest["step_id"],
        "injected_step_id": injected_step_id,
        "injected_step_index": matched["step_index"],
        "step_id_mapping": {
            "canonical_step_id": selected_step_manifest["step_id"],
            "injected_step_id": injected_step_id,
        },
        "canonical_step_fingerprint_sha256": _canonical_sha256(expected),
        "mapped_canonical_step_fingerprint_sha256": _canonical_sha256(
            mapped_canonical_fingerprint
        ),
        "injected_step_fingerprint_sha256": _canonical_sha256(
            injected_fingerprint
        ),
        "structural_fingerprint_match": True,
        "prompt_token_identity": "canonical_sha256_match",
        "historical_output_token_identity_confirmed": historical_outputs_observed,
        "content_identity": (
            "canonical_observed"
            if historical_outputs_observed
            else "deterministic_greedy_reconstruction_not_historically_observed"
        ),
        "comparison_status": (
            "independent_eager_reconstruction"
            if independent_eager
            else (
                "exact"
                if historical_outputs_observed
                else "independent_reconstruction"
            )
        ),
        "execution_boundary": (
            "independent_eager" if independent_eager else "canonical_graph_mode"
        ),
        "injected_boundaries": _boundary_summary(matched, previous),
        "following_step_boundaries": following,
        "injection_trace": str(injection_trace_path.resolve()),
        "injection_trace_sha256": trace_sha256,
    }


def _load_json_lines(path: Path) -> list[Mapping[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            value = json.loads(line)
            rows.append(_mapping(value, f"{path}:{line_number}"))
    if not rows:
        raise ValueError("canonical Scheduler trace is empty")
    return rows


def _ranked_row(row: Mapping[str, object]) -> dict[str, object]:
    observed_ms = 1000.0 * _finite(
        row["observed_serialized_service_s"], "observed serialized service"
    )
    predicted_ms = 1000.0 * _finite(
        row["predicted_device_s"], "predicted device time"
    )
    signed_error_ms = predicted_ms - observed_ms
    fingerprint = _mapping(row["step_fingerprint"], "step fingerprint")
    request_workload = _sequence(
        fingerprint["request_workload"], "fingerprint request workload"
    )
    token_totals = _mapping(
        fingerprint["token_totals"], "fingerprint token totals"
    )
    graph = _mapping(
        fingerprint["graph_execution"], "fingerprint graph execution"
    )
    return {
        "step_id": row["step_id"],
        "step_index": row["step_index"],
        "phase": row["phase"],
        "occurrence_count": 1,
        "request_count": len(request_workload),
        "prefill_tokens": token_totals["prefill_tokens"],
        "decode_tokens": token_totals["decode_tokens"],
        "scheduled_tokens": token_totals["scheduled_tokens"],
        "graph_runtime_mode": graph["runtime_mode"],
        "graph_capture_bucket_tokens": graph["capture_bucket_tokens"],
        "graph_padding_tokens": graph["padding_tokens"],
        "step_fingerprint_sha256": _canonical_sha256(fingerprint),
        "normalized_launch_program_sha256": fingerprint[
            "normalized_launch_program_sha256"
        ],
        "normalized_launch_call_count": fingerprint[
            "normalized_launch_call_count"
        ],
        "observed_serialized_service_ms": observed_ms,
        "predicted_device_ms": predicted_ms,
        "signed_error_ms": signed_error_ms,
        "c_step_ms": abs(signed_error_ms),
    }


def build_step_rankings(replay_payload: Mapping[str, object]) -> dict[str, object]:
    """Rank every replayed step and select from the TPOT-relevant phase."""

    if replay_payload["schema_version"] not in (5, 6):
        raise ValueError("selected-step ranking requires replay schema 5 or 6")
    if replay_payload["forward_duration_observation_consumed"] is not False:
        raise ValueError("selected-step ranking refuses observation-fed Forward")
    comparisons = [
        _mapping(value, "step comparison row")
        for value in _sequence(replay_payload["step_comparison"], "step comparison")
    ]
    if not comparisons:
        raise ValueError("step comparison is empty")
    for row in comparisons:
        fingerprint = _mapping(row["step_fingerprint"], "step fingerprint")
        require_exact_step_fingerprints(
            fingerprint,
            fingerprint,
        )
    rows = [_ranked_row(row) for row in comparisons]

    overall = sorted(rows, key=lambda row: (-row["c_step_ms"], row["step_index"]))
    phase_rankings = {
        phase: sorted(
            (row for row in rows if row["phase"] == phase),
            key=lambda row: (-row["c_step_ms"], row["step_index"]),
        )
        for phase in _PHASES
    }
    phase_totals = {
        phase: {
            "step_count": len(phase_rankings[phase]),
            "observed_serialized_service_ms": sum(
                row["observed_serialized_service_ms"]
                for row in phase_rankings[phase]
            ),
            "predicted_device_ms": sum(
                row["predicted_device_ms"] for row in phase_rankings[phase]
            ),
            "signed_error_ms": sum(
                row["signed_error_ms"] for row in phase_rankings[phase]
            ),
            "absolute_error_contribution_ms": sum(
                row["c_step_ms"] for row in phase_rankings[phase]
            ),
        }
        for phase in _PHASES
    }
    tpot_phase = max(
        ("mixed", "decode"),
        key=lambda phase: phase_totals[phase]["absolute_error_contribution_ms"],
    )
    if not phase_rankings[tpot_phase]:
        raise ValueError("TPOT-relevant phase has no steps")
    for rank, row in enumerate(overall, start=1):
        row["overall_rank"] = rank
    for phase in _PHASES:
        for rank, row in enumerate(phase_rankings[phase], start=1):
            row["phase_rank"] = rank
    selected = phase_rankings[tpot_phase][0]
    return {
        "schema_version": 1,
        "kind": "selected_step_c_step_ranking",
        "cell_id": replay_payload["cell_id"],
        "selection_contract": (
            "largest C_step within the TPOT-relevant phase having the largest "
            "summed absolute step error; prefill is excluded from TPOT selection"
        ),
        "selected_phase": tpot_phase,
        "selected_step_id": selected["step_id"],
        "phase_totals": phase_totals,
        "overall_ranking": overall,
        "phase_rankings": phase_rankings,
    }


def _raw_graph_identity(row: Mapping[str, object]) -> dict[str, object]:
    schema_version = (
        _integer(row["schema_version"], "trace schema", minimum=1)
        if "schema_version" in row
        else 4
    )
    if schema_version not in (3, 4, 5):
        raise ValueError("selected-step materialization requires trace schema 3/4/5")
    if "rank_observations" not in row:
        requests = _sequence(row["scheduled_requests"], "scheduled requests")
        uniform_decode = all(
            _integer(
                _mapping(value, "scheduled request")["scheduled_tokens"],
                "scheduled request tokens",
                minimum=1,
            ) == 1
            for value in requests
        )
        return {
            "capture_bucket_tokens": row["capture_bucket_tokens"],
            "capture_limit_tokens": row["capture_limit_tokens"],
            "configured_mode": row["configured_cudagraph_mode"],
            "fallback_reason": row["graph_fallback_reason"],
            "logical_tokens": row["num_unpadded_tokens"],
            "model_input_tokens": row["padded_total_tokens"],
            "padding_tokens": row["padding_tokens"],
            "rank_ids": [],
            "runtime_mode": row["runtime_cudagraph_mode"],
            "uniform_decode": uniform_decode,
        }
    observations = [
        _mapping(value, "rank observation")
        for value in _sequence(row["rank_observations"], "rank observations")
    ]
    if not observations:
        raise ValueError("rank observations must not be empty")
    first = observations[0]
    first_runtime_mode = first["runtime_cudagraph_mode"]
    if first_runtime_mode == "NONE":
        if first["selected_graph_key"] is not None:
            raise ValueError("NONE Graph observation selected a graph key")
        graph_key = _mapping(first["dispatch_input_key"], "dispatch input key")
    else:
        graph_key = _mapping(first["selected_graph_key"], "selected graph key")
    identity = {
        "capture_bucket_tokens": (
            None if first_runtime_mode == "NONE" else graph_key["num_tokens"]
        ),
        # Per-rank A800 dispatcher observations do not expose the configured
        # capture limit.  Replay validates it against the suite configuration.
        "capture_limit_tokens": None,
        "configured_mode": first["configured_cudagraph_mode"],
        "fallback_reason": first["graph_fallback_reason"],
        "logical_tokens": first["num_unpadded_tokens"],
        "model_input_tokens": first["padded_total_tokens"],
        "padding_tokens": first["padding_tokens"],
        "rank_ids": [observation["rank"] for observation in observations],
        "runtime_mode": first_runtime_mode,
        "uniform_decode": graph_key["uniform_decode"],
    }
    for observation in observations[1:]:
        runtime_mode = observation["runtime_cudagraph_mode"]
        if runtime_mode == "NONE":
            if observation["selected_graph_key"] is not None:
                raise ValueError("NONE Graph observation selected a graph key")
            key = _mapping(
                observation["dispatch_input_key"], "dispatch input key"
            )
        else:
            key = _mapping(
                observation["selected_graph_key"], "selected graph key"
            )
        candidate = {
            "capture_bucket_tokens": (
                None if runtime_mode == "NONE" else key["num_tokens"]
            ),
            "configured_mode": observation["configured_cudagraph_mode"],
            "fallback_reason": observation["graph_fallback_reason"],
            "logical_tokens": observation["num_unpadded_tokens"],
            "model_input_tokens": observation["padded_total_tokens"],
            "padding_tokens": observation["padding_tokens"],
            "runtime_mode": runtime_mode,
            "uniform_decode": key["uniform_decode"],
        }
        for field, value in candidate.items():
            if identity[field] != value:
                raise ValueError(
                    f"INVALID EXPERIMENT — raw.rank_observations.{field}: "
                    f"rank0={identity[field]!r}, rank={value!r}"
                )
    return identity


def _raw_graph_matches_expected(
    expected: Mapping[str, object], observed: Mapping[str, object]
) -> bool:
    for field, value in observed.items():
        # Flat C500/910B runtime observations are emitted once at the
        # Scheduler boundary and therefore carry no per-rank rows.  The
        # subsequent replay derives and validates the exact deployment ranks.
        if (
            (field == "rank_ids" and value == [])
            or (field == "capture_limit_tokens" and value is None)
        ):
            continue
        if expected[field] != value:
            return False
    return True


def _independent_eager_graph_identity(
    expected_fingerprint: Mapping[str, object],
) -> dict[str, object]:
    canonical = _mapping(
        expected_fingerprint["graph_execution"], "canonical graph execution"
    )
    return {
        "capture_bucket_tokens": None,
        "capture_limit_tokens": None,
        "configured_mode": "NONE",
        "fallback_reason": None,
        "logical_tokens": canonical["logical_tokens"],
        "model_input_tokens": canonical["logical_tokens"],
        "padding_tokens": 0,
        "rank_ids": list(_sequence(canonical["rank_ids"], "canonical graph ranks")),
        "runtime_mode": "NONE",
        "uniform_decode": canonical["uniform_decode"],
    }


def _raw_workload(
    row: Mapping[str, object], computed_by_request: Mapping[int, int]
) -> list[dict[str, object]]:
    workload = []
    for value in _sequence(row["scheduled_requests"], "scheduled requests"):
        request = _mapping(value, "scheduled request")
        request_idx = _integer(request["request_idx"], "request index")
        scheduled_tokens = _integer(
            request["scheduled_tokens"], "scheduled tokens", minimum=1
        )
        computed_tokens = (
            computed_by_request[request_idx]
            if request_idx in computed_by_request
            else 0
        )
        workload.append({
            "request_idx": request_idx,
            "request_id": f"request-{request_idx:012d}",
            "computed_tokens_before_step": computed_tokens,
            "kv_context_tokens_after_step": computed_tokens + scheduled_tokens,
            "prefill_tokens": request["prefill_tokens"],
            "decode_tokens": request["decode_tokens"],
            "scheduled_tokens": scheduled_tokens,
        })
    return workload


def _find_injected_structural_step(
    expected_fingerprint: Mapping[str, object],
    raw_rows: Sequence[Mapping[str, object]],
    *,
    independent_eager: bool = False,
) -> tuple[int, Mapping[str, object], list[dict[str, object]]]:
    expected_workload = [
        dict(_mapping(value, "expected request workload"))
        for value in _sequence(
            expected_fingerprint["request_workload"], "expected request workload"
        )
    ]
    expected_graph = (
        _independent_eager_graph_identity(expected_fingerprint)
        if independent_eager
        else _mapping(
            expected_fingerprint["graph_execution"], "expected graph execution"
        )
    )
    computed_by_request: dict[int, int] = {}
    matches = []
    for index, row in enumerate(raw_rows):
        workload = _raw_workload(row, computed_by_request)
        if workload == expected_workload:
            graph = _raw_graph_identity(row)
            if _raw_graph_matches_expected(expected_graph, graph):
                matches.append((index, row, workload))
        for request in workload:
            computed_by_request[request["request_idx"]] = request[
                "kv_context_tokens_after_step"
            ]
    if len(matches) != 1:
        raise ValueError(
            "INVALID EXPERIMENT — exact injected structural step count: "
            f"observed={len(matches)}, expected=1"
        )
    return matches[0]


def build_selected_step_injection_replay(
    repository_root: Path,
    config_path: Path,
    selected_step_manifest: Mapping[str, object],
    token_prefix_payload: Mapping[str, object],
    injection_trace_path: Path,
    *,
    host_launch_profile: HostLaunchProfile | None = None,
    host_runtime_prepare_profile: HostRuntimePrepareProfile | None = None,
    runtime_prepare_event_builder: Callable[
        [KernelPlan], Mapping[
            tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]
        ]
    ] | None = None,
    independent_eager: bool = False,
) -> dict[str, object]:
    """Derive the injected step's full fingerprint from its actual raw row."""

    if selected_step_manifest["kind"] != "selected_step_injection_input":
        raise ValueError("injection replay requires a selected-step input manifest")
    if token_prefix_payload["kind"] != "selected_step_token_prefixes":
        raise ValueError("injection replay requires token-prefix content")
    if token_prefix_payload["cell_id"] != selected_step_manifest["cell_id"]:
        raise ValueError("INVALID EXPERIMENT — token-prefix cell identity")
    expected = _mapping(
        selected_step_manifest["expected_fingerprint"], "expected fingerprint"
    )
    expected_digest = _canonical_sha256(expected)
    if token_prefix_payload["expected_step_fingerprint_sha256"] != expected_digest:
        raise ValueError("INVALID EXPERIMENT — token-prefix fingerprint digest")
    trace_path = injection_trace_path.resolve(strict=True)
    raw_rows = _load_json_lines(trace_path)
    _, matched, workload = _find_injected_structural_step(
        expected, raw_rows, independent_eager=independent_eager
    )

    prefix_requests = [
        _mapping(value, "token-prefix request")
        for value in _sequence(token_prefix_payload["requests"], "token-prefix requests")
    ]
    request_by_index = {
        _integer(value["request_idx"], "token-prefix request index"): value
        for value in prefix_requests
    }
    expected_indexes = {row["request_idx"] for row in workload}
    if set(request_by_index) != expected_indexes:
        raise ValueError("INVALID EXPERIMENT — token-prefix request coverage")
    max_request_idx = max(expected_indexes)
    requests = [
        {
            "request_id": f"request-{index:012d}",
            "input_tokens": 1,
            "output_tokens": 2,
        }
        for index in range(max_request_idx + 1)
    ]
    for request_idx, value in request_by_index.items():
        prompt_tokens = _integer(value["prompt_token_count"], "prompt token count", minimum=1)
        generated_tokens = _integer(
            value["generated_prefix_token_count"], "generated prefix token count"
        )
        requests[request_idx] = {
            "request_id": value["request_id"],
            "input_tokens": prompt_tokens,
            "output_tokens": max(2, generated_tokens + 2),
        }
    scheduled_prefill = sum(row["prefill_tokens"] for row in workload)
    scheduled_decode = sum(row["decode_tokens"] for row in workload)
    phase = (
        "mixed"
        if scheduled_prefill and scheduled_decode
        else "prefill" if scheduled_prefill else "decode"
    )
    canonical_step_id = selected_step_manifest["step_id"]
    if not isinstance(canonical_step_id, str) or not canonical_step_id:
        raise TypeError("selected-step canonical step ID must be non-empty text")
    injected_step_id = matched["step_id"]
    if not isinstance(injected_step_id, str) or not injected_step_id:
        raise TypeError("injected raw step ID must be non-empty text")
    normalized_row = {
        "step_index": matched["step_index"],
        # The Midlayer expert-routing baseline is deterministically keyed by
        # step_id.  Preserve the canonical workload identity while replaying;
        # the raw injected trace ID is only an execution-instance label.
        "step_id": canonical_step_id,
        "phase": phase,
        "requests": [
            {
                "request_idx": row["request_idx"],
                "request_id": row["request_id"],
                "computed_tokens": row["computed_tokens_before_step"],
                "prefill_tokens": row["prefill_tokens"],
                "decode_tokens": row["decode_tokens"],
                "scheduled_tokens": row["scheduled_tokens"],
            }
            for row in workload
        ],
        "finished_request_indexes": matched["finished_request_indexes"],
        "scheduled_request_count": len(workload),
        "scheduled_prefill_tokens": scheduled_prefill,
        "scheduled_decode_tokens": scheduled_decode,
        "scheduled_total_tokens": scheduled_prefill + scheduled_decode,
        "graph_execution_observation": _raw_graph_identity(matched),
    }
    content_source = _mapping(
        token_prefix_payload["content_source"], "content source"
    )
    model_config_sha256 = content_source["model_config_sha256"]
    if not isinstance(model_config_sha256, str) or len(model_config_sha256) != 64:
        raise ValueError("INVALID EXPERIMENT — content_source.model_config_sha256")
    prediction_arguments = {
        "include_step_kernel_contributions": True,
    }
    if independent_eager:
        prediction_arguments["graph_execution_override"] = (
            GraphExecutionConfig.disabled()
        )
    predictions, _, predictor_source = predict_scheduling_trajectory_steps(
        repository_root.resolve(),
        config_path.resolve(),
        selected_step_manifest["cell_id"],
        [normalized_row],
        requests,
        model_config_sha256,
        host_launch_profile=host_launch_profile,
        host_runtime_prepare_profile=host_runtime_prepare_profile,
        runtime_prepare_event_builder=runtime_prepare_event_builder,
        **prediction_arguments,
    )
    if len(predictions) != 1:
        raise AssertionError("selected-step injection replay did not return one step")
    prediction = predictions[0]
    if "predicted_kernel_contributions" not in prediction:
        raise ValueError(
            "selected-step replay requires per-step kernel contributions"
        )
    host_timing = {}
    if host_launch_profile is not None:
        for field in (
                "predicted_device_plus_launch_ns",
                "predicted_launch_cpu_ns",
                "predicted_launch_critical_path_ns",
                "predicted_host_device_overlap_ns",
                "predicted_execution_timeline"):
            if field not in prediction:
                raise RuntimeError(
                    f"selected-step replay lost Host timing field {field!r}")
            host_timing[field] = prediction[field]
    replayed_fingerprint = dict(
        _mapping(prediction["step_fingerprint"], "replayed step fingerprint")
    )
    if replayed_fingerprint["step_id"] != canonical_step_id:
        raise AssertionError("selected-step replay changed canonical workload identity")
    replayed_fingerprint["step_id"] = injected_step_id
    return {
        "schema_version": (
            7
            if host_runtime_prepare_profile is not None
            else 6 if host_launch_profile is not None else 5
        ),
        "analysis_kind": (
            "independent_eager_selected_step_injection_replay"
            if independent_eager
            else "selected_step_injection_replay"
        ),
        "cell_id": selected_step_manifest["cell_id"],
        "forward_duration_observation_consumed": False,
        "step_id_mapping": {
            "canonical_workload_step_id": canonical_step_id,
            "injected_trace_step_id": injected_step_id,
        },
        "source_sha256": {"scheduler_trace": _sha256(trace_path)},
        "predictor_source": predictor_source,
        "step_comparison": [{
            "step_index": matched["step_index"],
            "step_id": injected_step_id,
            "phase": phase,
            "step_fingerprint": replayed_fingerprint,
            "predicted_device_s": prediction["predicted_timeline_ns"] / 1e9,
            **host_timing,
            "predicted_kernel_contributions": prediction[
                "predicted_kernel_contributions"
            ],
        }],
    }


def _milliseconds(end: object, start: object, label: str) -> float:
    duration_ns = _integer(end, f"{label} end") - _integer(
        start, f"{label} start"
    )
    if duration_ns < 0:
        raise ValueError(f"{label} duration is negative")
    return duration_ns / 1_000_000.0


def _boundary_summary(
    row: Mapping[str, object], previous: Mapping[str, object] | None
) -> dict[str, object]:
    schedule_end = _integer(row["schedule_end_ns"], "schedule end")
    update_start = _integer(row["update_start_ns"], "update start")
    previous_update_start = (
        schedule_end
        if previous is None
        else _integer(previous["update_start_ns"], "previous update start")
    )
    serialized_start = max(schedule_end, previous_update_start)
    serialized_ms = (update_start - serialized_start) / 1_000_000.0
    rank_observations = (
        _sequence(row["rank_observations"], "rank observations")
        if "rank_observations" in row
        else ()
    )
    worker_envelopes = []
    for value in rank_observations:
        observation = _mapping(value, "rank observation")
        worker_envelopes.append({
            "rank": observation["rank"],
            "worker_execute_ms": _milliseconds(
                observation["worker_execution_end_ns"],
                observation["worker_execution_start_ns"],
                "worker execute",
            ),
        })
    return {
        "clock_domain": "time.perf_counter_ns",
        "full_step_ms": _milliseconds(
            row["step_end_ns"], row["step_start_ns"], "full step"
        ),
        "schedule_ms": _milliseconds(
            row["schedule_end_ns"], row["schedule_start_ns"], "schedule"
        ),
        "rpc_envelope_ms": _milliseconds(
            row["device_execution_end_ns"],
            row["device_execution_start_ns"],
            "executor RPC",
        ),
        "scheduler_update_ms": _milliseconds(
            row["scheduler_update_end_ns"],
            row["scheduler_update_start_ns"],
            "scheduler update",
        ),
        "serialized_service_ms": serialized_ms,
        "pipeline_hidden_before_serialized_service_ms": (
            serialized_start - _integer(row["step_start_ns"], "step start")
        ) / 1_000_000.0,
        "worker_execute_envelopes": worker_envelopes,
        "device_task_union_ms": None,
        "runtime_inter_task_gap_ms": None,
    }


def materialize_selected_step(
    replay_payload: Mapping[str, object],
    scheduler_trace_path: Path,
    step_id: str,
) -> dict[str, object]:
    """Build a fail-closed selected-step input manifest from canonical data."""

    ranking = build_step_rankings(replay_payload)
    comparisons = [
        _mapping(value, "step comparison row")
        for value in _sequence(replay_payload["step_comparison"], "step comparison")
    ]
    selected_matches = [row for row in comparisons if row["step_id"] == step_id]
    if len(selected_matches) != 1:
        raise ValueError(f"selected step must occur exactly once: {step_id}")
    selected_comparison = selected_matches[0]
    expected_fingerprint = _mapping(
        selected_comparison["step_fingerprint"], "selected fingerprint"
    )
    require_exact_step_fingerprints(expected_fingerprint, expected_fingerprint)

    raw_rows = _load_json_lines(scheduler_trace_path)
    computed_by_request: dict[int, int] = {}
    raw_selected = None
    raw_selected_index = None
    previous = None
    for index, row in enumerate(raw_rows):
        current_step_id = row["step_id"]
        if current_step_id == step_id:
            raw_selected = row
            raw_selected_index = index
            break
        for value in _sequence(row["scheduled_requests"], "scheduled requests"):
            request = _mapping(value, "scheduled request")
            request_idx = _integer(request["request_idx"], "request index")
            scheduled_tokens = _integer(
                request["scheduled_tokens"], "scheduled tokens", minimum=1
            )
            computed_by_request[request_idx] = (
                computed_by_request[request_idx] + scheduled_tokens
                if request_idx in computed_by_request
                else scheduled_tokens
            )
        previous = row
    if raw_selected is None or raw_selected_index is None:
        raise ValueError(f"selected step is absent from canonical trace: {step_id}")

    observed_fingerprint = dict(expected_fingerprint)
    observed_fingerprint["request_workload"] = _raw_workload(
        raw_selected, computed_by_request
    )
    observed_graph = _raw_graph_identity(raw_selected)
    replay_graph = _mapping(
        expected_fingerprint["graph_execution"], "fingerprint graph execution"
    )
    for field, value in observed_graph.items():
        if (
            (field == "rank_ids" and value == [])
            or (field == "capture_limit_tokens" and value is None)
        ):
            continue
        if replay_graph[field] != value:
            raise ValueError(
                f"INVALID EXPERIMENT — fingerprint.graph_execution.{field}: "
                f"observed={value!r}, replayed={replay_graph[field]!r}"
            )
    require_exact_step_fingerprints(expected_fingerprint, observed_fingerprint)

    comparison_index = comparisons.index(selected_comparison)
    adjacent = []
    for index in range(max(0, comparison_index - 1), min(
        len(comparisons), comparison_index + 2
    )):
        row = comparisons[index]
        observed_ms = 1000.0 * _finite(
            row["observed_serialized_service_s"], "observed service"
        )
        predicted_ms = 1000.0 * _finite(
            row["predicted_device_s"], "predicted device"
        )
        adjacent.append({
            "step_id": row["step_id"],
            "observed_serialized_service_ms": observed_ms,
            "predicted_device_ms": predicted_ms,
            "signed_error_ms": predicted_ms - observed_ms,
        })

    replay_source_hashes = _mapping(
        replay_payload["source_sha256"], "replay source hashes"
    )
    if "candidate_scheduler_trace" in replay_source_hashes:
        replay_source_scheduler_trace_sha256 = replay_source_hashes[
            "candidate_scheduler_trace"
        ]
    elif "scheduler_trace" in replay_source_hashes:
        replay_source_scheduler_trace_sha256 = replay_source_hashes[
            "scheduler_trace"
        ]
    else:
        raise ValueError("replay source hashes omit the Scheduler trace")

    return {
        "schema_version": 1,
        "kind": "selected_step_injection_input",
        "cell_id": replay_payload["cell_id"],
        "step_id": step_id,
        "source": {
            "scheduler_trace": str(scheduler_trace_path.resolve()),
            "scheduler_trace_sha256": _sha256(scheduler_trace_path),
            "replay_source_scheduler_trace_sha256": (
                replay_source_scheduler_trace_sha256
            ),
        },
        "selection": {
            "ranking_contract": ranking["selection_contract"],
            "selected_phase_by_contract": ranking["selected_phase"],
            "automatic_selected_step_id": ranking["selected_step_id"],
            "explicit_step_id": step_id,
        },
        "expected_fingerprint": dict(expected_fingerprint),
        "canonical_boundaries": _boundary_summary(raw_selected, previous),
        "adjacent_step_window": adjacent,
        "injection_contract": {
            "state_construction": "direct_request_state_injection",
            "full_request_stream_replay_allowed": False,
            "pre_injection_fingerprint_required": True,
            "post_injection_fingerprint_required": True,
            "graph_runtime_must_match": True,
            "eager_composition_role": "independent_replay_only",
            "canonical_token_values_available": False,
            "canonical_kv_payload_available": False,
            "structural_context_identity_available": True,
        },
    }


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
