"""Build a deterministic Scheduler Host CPU profile from Scheduler traces."""

from __future__ import annotations

from dataset_pipeline.serving_dataset.json_shape import file_sha256

import argparse
import hashlib
import json
from numbers import Real
from pathlib import Path
from statistics import median

import zstandard


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_BOUNDARY = "vllm_scheduler_trace_host_monotonic_plan_and_update"
_MIN_TRAINING_SAMPLES_PER_PHASE = 32


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _positive_whole_number(value: object, label: str) -> int:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or value != int(value) or value < 1):
        raise TypeError(f"{label} must be a positive whole number")
    return int(value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _phase(record: dict[str, object]) -> str:
    prefill = _integer(record["prefill_request_count"], "prefill request count")
    decode = _integer(record["decode_request_count"], "decode request count")
    if prefill > 0 and decode == 0:
        return "prefill"
    if decode > 0 and prefill == 0:
        return "decode"
    if prefill > 0 and decode > 0:
        return "mixed"
    raise ValueError("Scheduler trace step has no scheduled request phase")


def _read_trace(path: Path, expected_qps: int) -> list[tuple[int, int, int, str]]:
    records = []
    with path.open("rb") as source:
        reader = zstandard.ZstdDecompressor().stream_reader(source)
        try:
            payload = reader.read().decode("utf-8")
        finally:
            reader.close()
    for line_number, line in enumerate(payload.splitlines(), start=1):
        record = _mapping(json.loads(line), f"{path}:{line_number}")
        count = _integer(record["scheduled_request_count"], "scheduled request count")
        if count == 0:
            continue
        if count < 0:
            raise ValueError("scheduled request count cannot be negative")
        phase = _phase(record)
        plan = (_integer(record["schedule_end_ns"], "schedule end")
                - _integer(record["schedule_start_ns"], "schedule start"))
        update = (_integer(record["scheduler_update_end_ns"], "update end")
                  - _integer(record["scheduler_update_start_ns"], "update start"))
        if plan <= 0 or update <= 0:
            raise ValueError("Scheduler Host durations must be positive")
        records.append((count, plan, update, phase))
    if not records:
        raise ValueError(f"Scheduler trace has no non-empty steps: {path}")
    return records


def _trace_inputs(dataset_roots: list[Path], hardware_id: str) -> tuple[
        list[tuple[Path, int, str]], str, str]:
    paths = []
    for dataset_root in dataset_roots:
        paths.extend(
            sorted((dataset_root / hardware_id).glob(
                "*/*/scheduler_steps.jsonl.zst")))
    if not paths:
        raise ValueError(f"no Scheduler traces for hardware: {hardware_id}")
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError(f"Scheduler trace roots overlap for: {hardware_id}")
    inputs = []
    versions = set()
    identities = set()
    for path in paths:
        manifest_path = path.with_name("manifest.json")
        manifest = _mapping(json.loads(manifest_path.read_bytes()),
                            f"manifest {manifest_path}")
        identity = _mapping(manifest["identity"], "trace identity")
        runtime = _mapping(manifest["runtime"], "trace runtime")
        if _string(identity["hardware_id"], "trace hardware ID") != hardware_id:
            raise ValueError(f"trace hardware identity disagrees: {path}")
        version = _string(runtime["runtime_version"], "trace runtime version")
        source_commit = runtime["source_commit"]
        if source_commit is None:
            source_identity = f"vllm={version};source_commit=not_recorded"
        else:
            source_identity = _string(source_commit, "trace source commit")
        qps = _positive_whole_number(
            runtime["target_qps"], "trace target QPS")
        inputs.append((path, qps, source_identity))
        versions.add(version)
        identities.add(source_identity)
    if len(versions) != 1:
        raise ValueError(f"Scheduler traces mix vLLM versions: {sorted(versions)}")
    version = versions.pop()
    source_identity = ";".join(sorted(identities))
    return inputs, version, f"vllm={version};source_identities={source_identity}"


def _fit_affine(
    records: list[tuple[int, int, int, str]], component: int,
) -> tuple[float, float]:
    by_count: dict[int, list[int]] = {}
    for record in records:
        if record[0] not in by_count:
            by_count[record[0]] = []
        by_count[record[0]].append(record[component])
    count_medians = tuple(
        (count, float(median(samples)))
        for count, samples in sorted(by_count.items())
    )
    if len(count_medians) < 2:
        raise ValueError("Scheduler Host profile needs at least two request counts")
    slopes = [
        (right_value - left_value) / (right_count - left_count)
        for index, (left_count, left_value) in enumerate(count_medians)
        for right_count, right_value in count_medians[index + 1:]
    ]
    slope = float(median(slopes))
    if slope <= 0:
        raise ValueError("Scheduler Host per-request slope must be positive")
    intercept = float(median(
        value - slope * count for count, value in count_medians))
    if intercept <= 0:
        raise ValueError("Scheduler Host intercept must be positive")
    return intercept / 1_000_000_000, slope / 1_000_000_000


def build_shared_cpu_state_profile(path: Path, hardware_id: str) -> dict[str, object]:
    """Project independent synthetic state samples as an explicit CPU proxy.

    Prompt sizes and finish/continue controls are pooled unchanged; the existing
    two phase-group count model is reused, independent of Serving errors.
    """
    payload = _mapping(json.loads(zstandard.ZstdDecompressor().decompress(path.read_bytes())),
                       "independent CPU state measurement")
    if (payload["kind"] != "independent_vllm011_cpu_scheduler_state_grid"
            or payload["vllm_version"] != "0.11.0"
            or payload["production_or_evaluation_duration_consumed"] is not False
            or payload["measurement_boundary"] != "schedule_and_update_from_output_cpu_perf_counter_ns"):
        raise ValueError("shared CPU timing requires independent mechanism samples")
    records = []
    phases = ("prefill", "mixed", "decode")
    for row in payload["rows"]:
        if row["phase"] not in phases:
            raise ValueError("unsupported independent Scheduler phase")
        records.append((_positive_whole_number(row["request_count"], "request count"),
            _positive_whole_number(row["plan_ns"], "plan sample"),
            _positive_whole_number(row["update_ns"], "update sample"), row["phase"]))
    if any(sum(record[3] == phase for record in records) < _MIN_TRAINING_SAMPLES_PER_PHASE
           for phase in phases):
        raise ValueError("independent state measurement lacks phase coverage")
    partition = _phase_partition(records)
    fits = {name: _fit_phase_group(group) for name, group in partition.items()}
    def terms(group, operation):
        intercept, slope = fits[group][operation]
        return {"intercept": intercept, "per_request": slope}
    identity = hashlib.sha256(json.dumps(payload["source_sha256"], sort_keys=True).encode()).hexdigest()
    return {
        "schema_version": 1, "hardware_id": hardware_id,
        "shared_cpu_timing_target": "0.30.0",
        "scheduler": {"vllm_version": payload["vllm_version"],
            "source_identity": f"vllm={payload['vllm_version']};independent_cpu_mechanism=true;source_files_sha256={identity}"},
        "model": {"kind": "scheduled_request_count_affine",
            "state_dimensions": ["scheduled_request_count"],
            "statistic": "median_of_count_medians_theil_sen"},
        "max_scheduled_request_count": max(row[0] for row in records),
        "supported_phases": list(phases),
        "service_time_s": {"plan": terms("mixed_decode", "plan"),
            "update": terms("mixed_decode", "update"),
            "prefill": {operation: terms("prefill", operation) for operation in ("plan", "update")}},
        "measurement": {"boundary": _BOUNDARY, "time_unit": "seconds",
            "training_selection": {"excluded_target_qps": 16,
                "minimum_samples_per_phase": _MIN_TRAINING_SAMPLES_PER_PHASE,
                "eligible_step_rule": "nonzero_scheduled_request_count_with_positive_plan_and_update"},
            "artifacts": [{"path": str(path.relative_to(REPOSITORY_ROOT)), "sha256": _sha256(path)}]},
    }


def _phase_partition(
    records: list[tuple[int, int, int, str]],
) -> dict[str, list[tuple[int, int, int, str]]]:
    """Split records into the prefill line and the mixed/decode line.

    A pure-prefill step admits new requests, so its planning work is several
    times one decode step's; pooling both into one count-affine line would
    misprice whichever phase is outnumbered.  Mixed and decode keep sharing a
    line so their already-published parameters do not move.
    """

    partition: dict[str, list[tuple[int, int, int, str]]] = {
        "prefill": [], "mixed_decode": []}
    for record in records:
        partition[
            "prefill" if record[3] == "prefill" else "mixed_decode"].append(record)
    return partition


def _fit_phase_group(
    records: list[tuple[int, int, int, str]],
) -> dict[str, tuple[float, float]]:
    return {
        operation: _fit_affine(records, component)
        for component, operation in ((1, "plan"), (2, "update"))
    }


def _state_sample_counts(
    records: list[tuple[int, int, int, str]],
) -> dict[str, dict[str, int]]:
    """Expose coverage without making phase a Forward model input."""

    by_count: dict[str, int] = {}
    by_phase = {phase: 0 for phase in ("prefill", "mixed", "decode")}
    for count, _, _, phase in records:
        count_key = str(count)
        if count_key not in by_count:
            by_count[count_key] = 0
        by_count[count_key] += 1
        by_phase[phase] += 1
    return {
        "scheduled_request_count": dict(
            sorted(by_count.items(), key=lambda item: int(item[0]))),
        "phase": by_phase,
    }


def _split_records(
    held_out_qps: int,
    inputs: list[tuple[Path, int, str]],
    excluded_phases: tuple[str, ...] = (),
) -> tuple[list[tuple[int, int, int, str]], list[tuple[int, int, int, str]]]:
    training = []
    held_out = []
    for path, qps, _ in inputs:
        destination = held_out if qps == held_out_qps else training
        destination.extend(
            record for record in _read_trace(path, qps)
            if record[3] not in excluded_phases)
    if not training or not held_out:
        raise ValueError("held-out QPS must split Scheduler trace inputs")
    return training, held_out


def _supported_phases(records: list[tuple[int, int, int, str]]) -> tuple[str, ...]:
    phase_counts = _state_sample_counts(records)["phase"]
    return tuple(
        phase for phase in ("prefill", "mixed", "decode")
        if phase_counts[phase] >= _MIN_TRAINING_SAMPLES_PER_PHASE)


def _evaluate_group(
    fit: dict[str, tuple[float, float]],
    held_out: list[tuple[int, int, int, str]],
    maximum: int,
) -> dict[str, object]:
    result = {}
    for component, operation in ((1, "plan"), (2, "update")):
        intercept, slope = fit[operation]
        supported = [record for record in held_out if record[0] <= maximum]
        unsupported = [record for record in held_out if record[0] > maximum]
        if not supported:
            raise ValueError("held-out Scheduler Host evidence has no covered steps")
        actual_ns = [record[component] for record in supported]
        predicted_ns = [
            (intercept + slope * record[0]) * 1_000_000_000
            for record in supported
        ]
        residuals = sorted(
            abs(predicted - actual)
            for predicted, actual in zip(predicted_ns, actual_ns, strict=True)
        )
        actual_total = sum(actual_ns)
        predicted_total = sum(predicted_ns)
        result[operation] = {
            "intercept_s": intercept,
            "per_request_s": slope,
            "aggregate_relative_error_percent": (
                abs(predicted_total - actual_total) / actual_total * 100),
            "median_absolute_residual_ns": median(residuals),
            "p95_absolute_residual_ns": residuals[(len(residuals) - 1) * 95 // 100],
            "supported_sample_count": len(supported),
            "unsupported_sample_count": len(unsupported),
        }
    return result


def _evaluation(
    held_out_qps: int,
    inputs: list[tuple[Path, int, str]],
    excluded_phases: tuple[str, ...] = (),
) -> dict[str, object]:
    raw_training, held_out = _split_records(
        held_out_qps, inputs, excluded_phases)
    supported_phases = _supported_phases(raw_training)
    if not supported_phases:
        raise ValueError("Scheduler Host profile has no supported phase")
    training = [
        record for record in raw_training if record[3] in supported_phases]
    training_partition = _phase_partition(training)
    held_out_partition = _phase_partition(
        [record for record in held_out if record[3] in supported_phases])
    phase_groups = {}
    for group, group_training in training_partition.items():
        if not group_training:
            continue
        maximum = max(record[0] for record in group_training)
        phase_groups[group] = {
            "training_sample_count": len(group_training),
            "held_out_sample_count": len(held_out_partition[group]),
            "max_supported_scheduled_request_count": maximum,
            "operations": _evaluate_group(
                _fit_phase_group(group_training),
                held_out_partition[group],
                maximum),
        }
    return {
        "held_out_qps": held_out_qps,
        "training_sample_count": len(training),
        "held_out_sample_count": len(held_out),
        "supported_phases": list(supported_phases),
        "unsupported_phase_sample_count": sum(
            count for phase, count in _state_sample_counts(held_out)["phase"].items()
            if phase not in supported_phases),
        "max_supported_scheduled_request_count": max(
            record[0] for record in training),
        "state_sample_counts": {
            "raw_training": _state_sample_counts(raw_training),
            "training": _state_sample_counts(training),
            "held_out": _state_sample_counts(held_out),
        },
        "phase_groups": phase_groups,
        "operations": phase_groups["mixed_decode"]["operations"],
    }


def _fixed_value_baseline(
    held_out_qps: int,
    inputs: list[tuple[Path, int, str]],
    excluded_phases: tuple[str, ...] = (),
) -> dict[str, object]:
    """Evaluate the required hardware/version fixed-service first candidate."""

    raw_training, held_out = _split_records(
        held_out_qps, inputs, excluded_phases)
    supported_phases = _supported_phases(raw_training)
    if not supported_phases:
        raise ValueError("Scheduler Host profile has no supported phase")
    training = [
        record for record in raw_training if record[3] in supported_phases]
    held_out = [record for record in held_out if record[3] in supported_phases]
    result = {}
    for component, operation in ((1, "plan"), (2, "update")):
        value_ns = float(median(record[component] for record in training))
        actual_total = sum(record[component] for record in held_out)
        predicted_total = value_ns * len(held_out)
        result[operation] = {
            "service_time_s": value_ns / 1_000_000_000,
            "aggregate_relative_error_percent": (
                abs(predicted_total - actual_total) / actual_total * 100),
        }
    return {
        "held_out_qps": held_out_qps,
        "training_sample_count": len(training),
        "held_out_sample_count": len(held_out),
        "supported_phases": list(supported_phases),
        "operations": result,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True,
                        action="append", dest="dataset_roots")
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument("--held-out-qps", type=int, required=True)
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--independent-measurement", type=Path, action="append",
                        default=[])
    parser.add_argument(
        "--prefill-line-substitute", type=Path,
        help=("published Scheduler Host profile of the same hardware whose "
              "prefill line replaces this fit's own; only for traces that hold "
              "a single prefill request count (no native prefill sweep)"))
    parser.add_argument("--prefill-line-substitution-reason")
    return parser


def _prefill_line_substitute(args, version: str) -> dict[str, object] | None:
    """Load the explicitly named prefill line of another published profile."""

    if args.prefill_line_substitute is None:
        if args.prefill_line_substitution_reason is not None:
            raise ValueError("a prefill substitution reason needs a substitute")
        return None
    if args.prefill_line_substitution_reason is None:
        raise ValueError("--prefill-line-substitute requires a reason")
    path = args.prefill_line_substitute.resolve()
    payload = _mapping(json.loads(path.read_bytes()), "prefill substitute profile")
    if _string(payload["hardware_id"], "substitute hardware") != args.hardware_id:
        raise ValueError("prefill substitute profile hardware disagrees")
    if "prefill" not in payload["supported_phases"]:
        raise ValueError("prefill substitute profile has no prefill line")
    scheduler = _mapping(payload["scheduler"], "substitute scheduler")
    substitute_version = _string(scheduler["vllm_version"], "substitute version")
    if substitute_version == version:
        raise ValueError("a same-version prefill line must be fitted, not substituted")
    services = _mapping(payload["service_time_s"], "substitute services")
    return {
        "path": str(path.relative_to(REPOSITORY_ROOT)),
        "sha256": _sha256(path),
        "vllm_version": substitute_version,
        "reason": _string(args.prefill_line_substitution_reason, "substitution reason"),
        "max_scheduled_request_count": payload["max_scheduled_request_count"],
        "prefill": _mapping(services["prefill"], "substitute prefill line"),
    }


def _substitute_check(
    substitute: dict[str, object],
    inputs: list[tuple[Path, int, str]],
) -> dict[str, object]:
    """Compare the substituted line with this version's own prefill steps."""

    records = [
        record for path, qps, _ in inputs for record in _read_trace(path, qps)
        if record[3] == "prefill"]
    prefill = substitute["prefill"]
    fit = {
        operation: (float(prefill[operation]["intercept"]),
                    float(prefill[operation]["per_request"]))
        for operation in ("plan", "update")
    }
    return {
        "path": substitute["path"],
        "sha256": substitute["sha256"],
        "vllm_version": substitute["vllm_version"],
        "reason": substitute["reason"],
        "own_prefill_state_sample_counts": _state_sample_counts(records),
        "own_prefill_steps_under_substitute": _evaluate_group(
            fit, records, int(substitute["max_scheduled_request_count"])),
    }


def main() -> None:
    args = _parser().parse_args()
    if args.held_out_qps < 1:
        raise ValueError("--held-out-qps must be positive")
    dataset_roots = [root.resolve() for root in args.dataset_roots]
    independent_measurements = []
    for measurement in args.independent_measurement:
        measurement_path = measurement.resolve()
        payload = _mapping(json.loads(measurement_path.read_bytes()),
                           f"independent measurement {measurement_path}")
        if _string(payload["hardware_id"], "independent measurement hardware") != (
                args.hardware_id):
            raise ValueError("independent measurement hardware disagrees")
        scheduler = _mapping(payload["scheduler"], "independent measurement scheduler")
        if "vllm_version" not in scheduler:
            raise ValueError("independent measurement lacks vLLM version")
        services = _mapping(payload["service_ns"], "independent measurement services")
        schedule_service = _mapping(
            services["schedule"], "independent measurement schedule")
        update_service = _mapping(
            services["update_from_output"], "independent measurement update")
        independent_measurements.append({
            "path": str(measurement_path.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256(measurement_path),
            "vllm_version": _string(
                scheduler["vllm_version"], "independent measurement version"),
            "boundary": _string(
                _mapping(payload["measurement"], "independent measurement")
                ["boundary"], "independent measurement boundary"),
            "schedule_median_ns": _positive_whole_number(
                schedule_service["median_ns"], "independent schedule median"),
            "update_median_ns": _positive_whole_number(
                update_service["median_ns"], "independent update median"),
        })
    inputs, version, source_identity = _trace_inputs(dataset_roots, args.hardware_id)
    substitute = _prefill_line_substitute(args, version)
    excluded_phases = () if substitute is None else ("prefill",)
    raw_training_records = []
    artifacts = []
    source_record_counts = []
    for path, qps, _ in inputs:
        trace_records = _read_trace(path, qps)
        if qps != args.held_out_qps:
            raw_training_records.extend(
                record for record in trace_records
                if record[3] not in excluded_phases)
        source_record_counts.append({
            "path": str(path.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256(path),
            "manifest_path": str(
                path.with_name("manifest.json").relative_to(REPOSITORY_ROOT)),
            "manifest_sha256": _sha256(path.with_name("manifest.json")),
            "qps": qps,
            "sample_count": len(trace_records),
        })
        if qps != args.held_out_qps:
            artifacts.append({
                "path": str(path.relative_to(REPOSITORY_ROOT)),
                "sha256": _sha256(path),
            })
    supported_phases = _supported_phases(raw_training_records)
    if not supported_phases:
        raise ValueError("Scheduler Host profile has no supported phase")
    if substitute is not None:
        supported_phases = ("prefill", *supported_phases)
        source_identity = (
            f"{source_identity};prefill_line_substitute="
            f"vllm={substitute['vllm_version']}:{substitute['path']}")
        artifacts.append({"path": substitute["path"], "sha256": substitute["sha256"]})
    training_records = [
        record for record in raw_training_records
        if record[3] in supported_phases]
    partition = _phase_partition(training_records)
    mixed_decode_fit = _fit_phase_group(partition["mixed_decode"])
    plan_intercept_s, plan_per_request_s = mixed_decode_fit["plan"]
    update_intercept_s, update_per_request_s = mixed_decode_fit["update"]
    maximum = max(record[0] for record in training_records)
    service_time_s = {
        "plan": {
            "intercept": plan_intercept_s,
            "per_request": plan_per_request_s,
        },
        "update": {
            "intercept": update_intercept_s,
            "per_request": update_per_request_s,
        },
    }
    if substitute is not None:
        service_time_s["prefill"] = substitute["prefill"]
    elif "prefill" in supported_phases:
        prefill_fit = _fit_phase_group(partition["prefill"])
        service_time_s["prefill"] = {
            "plan": {
                "intercept": prefill_fit["plan"][0],
                "per_request": prefill_fit["plan"][1],
            },
            "update": {
                "intercept": prefill_fit["update"][0],
                "per_request": prefill_fit["update"][1],
            },
        }
    profile = {
        "schema_version": 1,
        "hardware_id": args.hardware_id,
        "scheduler": {
            "vllm_version": version,
            "source_identity": source_identity,
        },
        "model": {
            "kind": "scheduled_request_count_affine",
            "state_dimensions": ["scheduled_request_count"],
            "statistic": "median_of_count_medians_theil_sen",
        },
        "max_scheduled_request_count": maximum,
        "supported_phases": list(supported_phases),
        "service_time_s": service_time_s,
        "measurement": {
            "boundary": _BOUNDARY,
            "time_unit": "seconds",
            "training_selection": {
                "excluded_target_qps": args.held_out_qps,
                "minimum_samples_per_phase": _MIN_TRAINING_SAMPLES_PER_PHASE,
                "eligible_step_rule": (
                    "nonzero_scheduled_request_count_with_positive_plan_and_update"
                ),
            },
            "artifacts": artifacts,
        },
    }
    profile_bytes = (json.dumps(profile, indent=2, sort_keys=True) + "\n").encode(
        "utf-8")
    report = {
        "schema_version": 1,
        "hardware_id": args.hardware_id,
        "vllm_version": version,
        "source_identity": source_identity,
        "model": profile["model"],
        "source_record_counts": source_record_counts,
        "independent_measurements": independent_measurements,
        "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "fixed_value_baseline": _fixed_value_baseline(
            args.held_out_qps, inputs, excluded_phases),
        "held_out_evaluation": _evaluation(
            args.held_out_qps, inputs, excluded_phases),
        "prefill_line_substitute": (
            None if substitute is None
            else _substitute_check(substitute, inputs)),
    }
    args.profile_output.parent.mkdir(parents=True, exist_ok=True)
    args.profile_output.write_bytes(profile_bytes)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
