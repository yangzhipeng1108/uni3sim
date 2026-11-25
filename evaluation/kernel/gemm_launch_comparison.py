"""Compare observed and injected GEMM launch facts without conflating roles."""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import math
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

from evaluation.kernel.dataset import (
    get_hardware_config_path,
    load_kernel_dataset,
    parse_kernel_evaluation_timing,
)
from evaluation.kernel.observation_normalizer import (
    normalize_kernel_spec,
    observed_gemm_split_k,
    observed_gemm_split_k_partial_bytes,
    observed_gemm_split_k_partial_dram_bytes,
    validate_kernel_observation_sequence,
)
from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import KernelKind
from src.kernel.launch_config.gemm_launch_selection import (
    load_gemm_launch_selection_files,
)
from src.kernel.launch_config.select_launch_config import (
    select_gemm_reducer_launch,
    select_launch_config,
)


_SUPPORTED_HARDWARE_IDS = (
    "nvidia_a800",
    "nvidia_h200",
    "metax_c500",
    "ascend_910b",
)

# The observed-side launch contract each hardware's kernel dataset carries.
# The H200 kernel dataset stores nsys CUPTI launch records with grid/block
# and shared-memory facts; the injected side is the candidate v6 LUT row.
_H200_GEMM_INJECTED_ATTRIBUTES = {
    "has_bias": False,
    "launch_policy": "injected_launch",
    "a_layout": "row_major",
    "b_layout": "row_major",
    "c_layout": "row_major",
    "trans_a": False,
    "trans_b": True,
    "epilogue": "none",
}

_PHYSICAL_LAUNCH_FIELDS = (
    "grid",
    "block",
)


def _kernel_records(entry: Mapping, case_id: str) -> list[Mapping]:
    raw = entry["kernels"]
    if isinstance(raw, dict):
        records = list(raw.values())
    elif isinstance(raw, list):
        records = list(raw)
    else:
        raise TypeError(
            f"GEMM dataset entry {case_id!r} kernels must be a mapping or list")
    if not records:
        raise ValueError(f"GEMM dataset entry {case_id!r} has no kernels")
    if any(not isinstance(record, Mapping) for record in records):
        raise TypeError(
            f"GEMM dataset entry {case_id!r} contains a non-mapping kernel")
    return records


# The static-workload inputs each hardware's GEMM selector requires beyond the
# observed shape.  A hardware outside this table has no comparison contract.
_GEMM_INJECTED_ATTRIBUTES = {
    "nvidia_a800": {
        "has_bias": False,
        "launch_policy": "injected_launch",
    },
    "metax_c500": {
        "position": "generic",
        "launch_policy": "injected_launch",
    },
    "ascend_910b": {
        "launch_policy": "injected_launch",
    },
}


def _injected_attributes(hardware_id: str) -> Mapping[str, object]:
    """Return explicit static-workload inputs required by each GEMM selector."""

    if hardware_id == "nvidia_h200":
        return dict(_H200_GEMM_INJECTED_ATTRIBUTES)
    if hardware_id not in _GEMM_INJECTED_ATTRIBUTES:
        raise ValueError(
            f"unsupported GEMM launch-comparison hardware: {hardware_id!r}")
    return dict(_GEMM_INJECTED_ATTRIBUTES[hardware_id])


def _physical_fields(launch) -> Mapping[str, object]:
    return {
        "grid": launch.grid,
        "block": launch.block,
    }


def _execution_plan(launch) -> Mapping[str, object]:
    tiling = launch.tiling
    if tiling is None:
        tiling_value = None
    else:
        tiling_value = {
            "tile_shape": dict(tiling.tile_shape),
            "pipeline_depth": tiling.pipeline_depth,
            "split_k": tiling.split_k,
        }
    return {
        "tiling": tiling_value,
        "parameters": dict(launch.parameters),
        "execution_variant": launch.execution_variant,
    }


def _with_injected_physical_fields(observed_spec, injected_launch):
    """Replace only physical launch metadata and retain the observed plan."""

    launch = dataclasses.replace(
        observed_spec.launch,
        grid=injected_launch.grid,
        block=injected_launch.block,
    )
    return dataclasses.replace(observed_spec, launch=launch)


def _with_probe_physical_fields(observed_spec):
    """Change every physical field without changing the execution plan."""

    observed = observed_spec.launch
    grid = (observed.grid[0] + 1, observed.grid[1] + 1, observed.grid[2] + 1)
    block = (observed.block[0] + 1, observed.block[1], observed.block[2])
    launch = dataclasses.replace(
        observed,
        grid=grid,
        block=block,
    )
    return dataclasses.replace(observed_spec, launch=launch)


def _with_injected_execution_plan(observed_spec, injected_launch):
    """Replace only tile/split/plan facts and retain physical metadata."""

    launch = dataclasses.replace(
        observed_spec.launch,
        tiling=injected_launch.tiling,
        parameters=injected_launch.parameters,
        execution_variant=injected_launch.execution_variant,
    )
    return dataclasses.replace(observed_spec, launch=launch)


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    if not materialized:
        raise ValueError("mean requires at least one value")
    return math.fsum(materialized) / len(materialized)


def _mape(predictions: Iterable[float], actuals: Iterable[float]) -> float:
    pairs = tuple(zip(predictions, actuals, strict=True))
    if not pairs:
        raise ValueError("MAPE requires at least one prediction")
    return 100.0 * _mean(abs(predicted - actual) / actual
                         for predicted, actual in pairs)


def compare_gemm_launch_sources(hardware_id: str) -> Mapping[str, object]:
    """Compare physical launch metadata separately from execution-plan facts."""

    if hardware_id not in _SUPPORTED_HARDWARE_IDS:
        raise ValueError(
            f"hardware_id must be one of {_SUPPORTED_HARDWARE_IDS}, "
            f"got {hardware_id!r}")
    if hardware_id == "nvidia_h200":
        return _compare_h200_gemm_launch_sources()
    hardware = HWSpec(get_hardware_config_path(hardware_id))
    dataset = load_kernel_dataset(hardware_id, kernels=("gemm",))
    rows = []

    for case_id, entry in sorted(dataset.items()):
        op_info = entry["op_info"]
        if op_info["Type"] != "GEMM":
            raise ValueError(
                f"gemm.json entry {case_id!r} has type {op_info['Type']!r}")
        records = _kernel_records(entry, case_id)
        validate_kernel_observation_sequence(records, hardware_id)
        timing = parse_kernel_evaluation_timing(entry, records, hardware_id)
        split_k = observed_gemm_split_k(records)
        partial_bytes = observed_gemm_split_k_partial_bytes(records)
        partial_dram_bytes = observed_gemm_split_k_partial_dram_bytes(records)

        for kernel_index, record in enumerate(records):
            observed_spec = normalize_kernel_spec(
                op_info,
                record,
                hardware_id,
                hardware,
                kernel_ordinal=kernel_index,
                gemm_split_k=split_k,
                gemm_split_k_partial_bytes=partial_bytes,
                gemm_split_k_partial_dram_bytes=partial_dram_bytes,
            )
            if observed_spec.kind is not KernelKind.GEMM:
                continue
            observed_prediction = predict(
                observed_spec, hardware, method="midlayer").t_total_ns
            probe_spec = _with_probe_physical_fields(observed_spec)
            probe_prediction = predict(
                probe_spec, hardware, method="midlayer").t_total_ns
            actual_ns = timing.durations_ns[kernel_index]
            common_row = {
                "case_id": case_id,
                "kernel_index": kernel_index,
                "actual_ns": actual_ns,
                "observed_prediction_ns": observed_prediction,
                "physical_contract_probe_prediction_ns": probe_prediction,
                "physical_contract_probe_delta_percent": (
                    100.0 * (probe_prediction - observed_prediction)
                    / observed_prediction),
            }
            injected_launch = select_launch_config(
                hardware_id,
                KernelKind.GEMM,
                observed_spec.shape,
                observed_spec.dtype,
                observed_spec.variant,
                _injected_attributes(hardware_id),
                hardware=hardware,
            )

            physical_spec = _with_injected_physical_fields(
                observed_spec, injected_launch)
            physical_prediction = predict(
                physical_spec, hardware, method="midlayer").t_total_ns
            execution_plan_spec = _with_injected_execution_plan(
                observed_spec, injected_launch)
            execution_plan_prediction = predict(
                execution_plan_spec, hardware, method="midlayer").t_total_ns
            observed_physical = _physical_fields(observed_spec.launch)
            injected_physical = _physical_fields(injected_launch)
            observed_plan = _execution_plan(observed_spec.launch)
            injected_plan = _execution_plan(injected_launch)
            rows.append(common_row | {
                "injected_execution_plan_status": "available",
                "observed_physical_launch": observed_physical,
                "injected_physical_launch": injected_physical,
                "physical_fields_match": {
                    name: observed_physical[name] == injected_physical[name]
                    for name in observed_physical
                },
                "physical_only_prediction_ns": physical_prediction,
                "physical_only_prediction_delta_percent": (
                    100.0 * (physical_prediction - observed_prediction)
                    / observed_prediction),
                "observed_execution_plan": observed_plan,
                "injected_execution_plan": injected_plan,
                "execution_plan_matches": observed_plan == injected_plan,
                "injected_plan_prediction_ns": execution_plan_prediction,
                "injected_plan_prediction_delta_percent": (
                    100.0
                    * (execution_plan_prediction - observed_prediction)
                    / observed_prediction),
            })

    covered = [
        row for row in rows
        if row["injected_execution_plan_status"] == "available"]
    if not covered:
        raise ValueError(
            f"{hardware_id} has no GEMM cases covered by injected launch data")
    field_match_percent = {
        name: 100.0 * sum(
            row["physical_fields_match"][name] for row in covered)
        / len(covered)
        for name in _PHYSICAL_LAUNCH_FIELDS
    }
    contract_probe_deltas = [
        abs(row["physical_contract_probe_delta_percent"])
        for row in rows
    ]
    physical_deltas = [
        abs(row["physical_only_prediction_delta_percent"])
        for row in covered
    ]
    plan_deltas = [
        abs(row["injected_plan_prediction_delta_percent"])
        for row in covered
    ]
    observed_predictions = [
        row["observed_prediction_ns"] for row in covered]
    plan_predictions = [
        row["injected_plan_prediction_ns"] for row in covered]
    actuals = [row["actual_ns"] for row in covered]
    summary = {
        "hardware_id": hardware_id,
        "gemm_cases": len(dataset),
        "gemm_main_kernels": len(rows),
        "injected_execution_plan_covered": len(covered),
        "injected_execution_plan_unavailable": 0,
        "comparison_contract": {
            "physical_launch_fields": [
                "grid", "block",
            ],
            "physical_launch_role": (
                "diagnostic only; held separate from the GEMM performance "
                "model"),
            "execution_plan_fields": [
                "tiling", "parameters", "execution_variant"],
            "execution_plan_role": (
                "retained algorithmic inputs including tile and split-K"),
        },
        "physical_launch": {
            "all_kernel_contract_probe_count": len(rows),
            "all_kernel_contract_probe_max_abs_delta_percent": max(
                contract_probe_deltas),
            "all_kernel_contract_probe_predictions_identical": all(
                delta == 0.0 for delta in contract_probe_deltas),
            "observed_vs_injected_covered_count": len(covered),
            "field_match_percent": field_match_percent,
            "prediction_mean_abs_delta_percent": _mean(physical_deltas),
            "prediction_max_abs_delta_percent": max(physical_deltas),
            "all_covered_predictions_identical": all(
                delta == 0.0 for delta in physical_deltas),
        },
        "execution_plan": {
            "plan_match_percent": 100.0 * sum(
                row["execution_plan_matches"] for row in covered)
            / len(covered),
            "observed_plan_mape_percent_on_covered": _mape(
                observed_predictions, actuals),
            "injected_plan_mape_percent_on_covered": _mape(
                plan_predictions, actuals),
            "prediction_mean_abs_delta_percent": _mean(plan_deltas),
            "prediction_max_abs_delta_percent": max(plan_deltas),
        },
        "missing_injected_key_counts": {},
    }
    return {"summary": summary, "cases": rows}


def _parse_nsys_grid(value: str) -> tuple[int, int, int]:
    text = value.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    parts = [int(p.strip()) for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"nsys grid must be 3 integers, got {value!r}")
    return tuple(parts)


def _parse_nsys_block(value: str) -> tuple[int, int, int]:
    text = value.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    parts = [int(p.strip()) for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"nsys block must be 3 integers, got {value!r}")
    return tuple(parts)


def _normalize_kernel_name(name: str) -> str:
    """Normalize a CUDA kernel symbol for comparison.

    The observed dataset stores the demangled symbol
    (``cublasLt::splitKreduce_kernel<(int)32, ...>``) while the capture
    projection stores the mangled Itanium ABI name
    (``_ZN8cublasLt19splitKreduce_kernelILi32ELi16E...``).  Both name
    the same instantiation; comparing them requires demangling the
    mangled form.  Full template arguments (types, bools, ints) are
    preserved as the identity-bearing comparison key.
    """
    if name.startswith("_Z"):
        return _canonical_template_signature(_demangle(name))
    return _canonical_template_signature(name)


_CPPFILT = ("/usr/bin/c++filt",)


@lru_cache(maxsize=4096)
def _demangle(mangled: str) -> str:
    """Demangle an Itanium ABI symbol via the system c++filt binary."""
    result = subprocess.run(
        _CPPFILT, input=mangled + "\n",
        capture_output=True, text=True, check=True)
    text = result.stdout.strip()
    if text == mangled or not text:
        raise ValueError(f"Cannot demangle kernel symbol: {mangled}")
    return text


def _canonical_template_signature(demangled: str) -> str:
    """Compare the full template signature, not nsys's parameter placeholders."""
    opening = demangled.find("<")
    if opening == -1:
        return demangled
    family = re.search(r"[\w:]+$", demangled[:opening].strip())
    if family is None:
        raise ValueError(f"Invalid kernel template signature: {demangled!r}")
    depth = 0
    for index in range(opening, len(demangled)):
        char = demangled[index]
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth == 0:
                args = demangled[opening + 1:index]
                args = re.sub(r"\((?:int|bool)\)\s*(-?\d+)", r"\1", args)
                args = re.sub(r"\btrue\b", "1", args)
                args = re.sub(r"\bfalse\b", "0", args)
                return family.group(0) + "<" + re.sub(r"\s+", "", args) + ">"
    raise ValueError(f"Unclosed kernel template signature: {demangled!r}")


_H200_CAPTURE_SEGMENTS = (
    ("segment1_compact_facts", "segment1_raw_capture/rank0_launch_facts.jsonl.gz"),
    ("segment2_compact_facts", "segment2_bmm_capture/rank0_launch_facts_seg2.jsonl.gz"),
    ("segment3_compact_facts", "segment3_lowm_capture/rank0_launch_facts_seg3.jsonl.gz"),
    ("segment4_compact_facts", "segment4_linear32k_capture/rank0_launch_facts_seg4.jsonl.gz"),
)


def _load_capture_projections(repo_root: Path, manifest: Mapping, needed_keys: set):
    """Read required facts and validate their canonical manifest bindings."""
    capture_root = repo_root / "profiler_output/launch_data/nvidia_h200/gemm_lut_20261104"
    sources = {item["kind"]: item["sha256"] for item in manifest["evidence_sources"]}
    projections = {}
    raw_sha256 = {}
    for kind, relative_path in _H200_CAPTURE_SEGMENTS:
        path = capture_root / relative_path
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha = digest.hexdigest()
        if actual_sha != sources[kind]:
            raise ValueError(f"H200 capture SHA256 mismatch: {path}")
        raw_sha256[relative_path] = actual_sha
        with gzip.open(path, "rt") as handle:
            for line in handle:
                rec = json.loads(line)
                op_key = json.loads(rec["operation_key"])
                key = tuple(op_key[:4])
                if key not in needed_keys:
                    continue
                if len(op_key) != 5 or op_key[4] is not False:
                    raise ValueError(f"Unexpected ordinary capture identity: {op_key}")
                projection = rec["projection"]
                if key in projections and projections[key] != projection:
                    raise ValueError(f"Conflicting capture facts: {key}")
                projections[key] = projection
    missing = needed_keys - projections.keys()
    if missing:
        raise ValueError(f"Missing required capture facts: {sorted(missing)}")
    return projections, raw_sha256


def _observed_h200_launch_facts(kdata: Mapping) -> Mapping[str, object]:
    metadata = kdata["Metadata"]
    launch_stats = kdata["Sections"]["Launch Statistics"]["Header"]
    return {
        "kernel_name": metadata["Kernel Name"],
        "grid": _parse_nsys_grid(metadata["Grid Size"]),
        "block": _parse_nsys_block(metadata["Block Size"]),
        "dynamic_shared_memory_bytes": int(
            launch_stats["Dynamic Shared Memory Per Block [byte/block]"]),
        "static_shared_memory_bytes": int(
            launch_stats["Static Shared Memory Per Block [byte/block]"]),
        "shared_memory_config_bytes": int(
            launch_stats["Shared Memory Configuration Size [byte]"]),
        "registers_per_thread": int(
            launch_stats["Registers Per Thread [register/thread]"]),
    }


def _compare_h200_gemm_launch_sources() -> Mapping[str, object]:
    """Compare the observed H200 kernel dataset launch facts against the v6 candidate.

    The observed side is the nsys CUPTI launch record (kernel name, grid,
    block, shared memory, registers) stored in the H200 kernel dataset; the
    injected side is the strict-loaded v6 candidate LUT row selected by the
    public selector.  Only physical launch facts are compared; execution
    plan (tiling) is the candidate's grid-block-only observation and stays
    a separate diagnostic role.
    """

    hardware = HWSpec(get_hardware_config_path("nvidia_h200"))
    dataset = load_kernel_dataset("nvidia_h200", kernels=("gemm",))
    repo_root = Path(__file__).resolve().parents[2]
    v6_csv = repo_root / "launch_data/nvidia_h200/gemm_launch_selection.csv.zst"
    v6_manifest = v6_csv.with_name("gemm_launch_selection.manifest.json")
    # Strict candidate load: duplicate keys already fail closed inside the
    # loader; this comparison consumes the typed LUT as-is.
    candidate_lut = load_gemm_launch_selection_files(
        v6_csv, v6_manifest, hardware_id="nvidia_h200",
        software_revision="cuda_12.9")
    # Load capture projections for kernel-name and shared-memory join.
    manifest = json.loads(v6_manifest.read_text())
    needed_keys = set()
    for entry in dataset.values():
        attrs = entry["op_info"]["Attr"]
        needed_keys.add((int(attrs["B"]) * int(attrs["M"]), int(attrs["N"]),
                         int(attrs["K"]), str(attrs["dtype"]).lower()))
    capture_projections, raw_sha256 = _load_capture_projections(repo_root, manifest, needed_keys)
    # Inject the v6 candidate into the selector with scoped restoration.
    import src.kernel.launch_config.nvidia_h200.gemm_launch as h200_mod
    original_lut_fn = h200_mod._observed_launch_lut
    h200_mod._observed_launch_lut = lambda: candidate_lut

    rows = []
    misses = []
    try:
        for case_id, entry in sorted(dataset.items()):
            op_info = entry["op_info"]
            if op_info["Type"] != "GEMM":
                raise ValueError(
                    f"gemm.json entry {case_id!r} has type {op_info['Type']!r}")
            records = _kernel_records(entry, case_id)
            validate_kernel_observation_sequence(records, "nvidia_h200")
            B = int(op_info["Attr"]["B"])
            M = int(op_info["Attr"]["M"])
            # All nn.Linear shapes, including B>1,M=1 decode, use physical
            # b=1, m=B*M.  Do not relabel as BMM or fabricate strides.
            shape = {"b": 1, "m": B * M,
                     "n": int(op_info["Attr"]["N"]),
                     "k": int(op_info["Attr"]["K"])}
            dtype = str(op_info["Attr"]["dtype"]).lower()
            if dtype not in ("bf16", "fp16"):
                raise ValueError(f"Unsupported dataset GEMM dtype: {dtype}")
            observed_facts = [
                _observed_h200_launch_facts(record) for record in records]
            injected_attributes = dict(_H200_GEMM_INJECTED_ATTRIBUTES)
            injected_attributes["beta"] = 0
            injected_attributes["output_dtype"] = None
            try:
                injected_launch = select_launch_config(
                    "nvidia_h200", KernelKind.GEMM, shape, dtype, "gemm",
                    injected_attributes, hardware=hardware)
            except (KeyError, NotImplementedError, ValueError) as exc:
                misses.append({"case_id": case_id, "reason": str(exc)})
                continue
            observed_main = observed_facts[0]
            observed_reducers = observed_facts[1:]
            injected_grid = injected_launch.grid
            injected_block = injected_launch.block
            # Join with capture projections for kernel name and shared memory.
            capture_key = (shape["m"], shape["n"], shape["k"], dtype)
            launches = capture_projections[capture_key]["launches"]
            if not 1 <= len(launches) <= 2:
                raise ValueError(f"Unexpected capture launch count: {capture_key}")
            captured_main_symbol = launches[0]["symbol"]
            captured_main_shared = launches[0]["shared_memory_bytes"]
            captured_reducer_symbol = launches[1]["symbol"] if len(launches) == 2 else None
            captured_reducer_shared = launches[1]["shared_memory_bytes"] if len(launches) == 2 else None
            # Kernel name comparison: observed dataset vs capture projection.
            observed_main_name = observed_main["kernel_name"]
            kernel_name_match = None
            if captured_main_symbol is not None:
                kernel_name_match = (
                    _normalize_kernel_name(observed_main_name)
                    == _normalize_kernel_name(captured_main_symbol))
            # Dynamic shared memory comparison: observed dataset vs capture.
            dynamic_shared_match = None
            if captured_main_shared is not None:
                dynamic_shared_match = (
                    observed_main["dynamic_shared_memory_bytes"]
                    == captured_main_shared)
            fields_match = {
                "grid": observed_main["grid"] == injected_grid,
                "block": observed_main["block"] == injected_block,
                "kernel_name": kernel_name_match,
                "dynamic_shared_memory_bytes": dynamic_shared_match,
                "registers_per_thread": None,
            }
            reducer_launch = select_gemm_reducer_launch(
                "nvidia_h200", shape, dtype, "gemm", injected_attributes,
                hardware=hardware)
            reducer_counts_match = len(launches) == len(observed_facts) and (
                (reducer_launch is None and not observed_reducers)
                or (reducer_launch is not None and len(observed_reducers) == 1))
            reducer_fields_match = None
            if reducer_launch is not None and len(observed_reducers) == 1:
                observed_reducer = observed_reducers[0]
                reducer_kernel_match = None
                if captured_reducer_symbol is not None:
                    reducer_kernel_match = (
                        _normalize_kernel_name(observed_reducer["kernel_name"])
                        == _normalize_kernel_name(captured_reducer_symbol))
                reducer_shared_match = None
                if captured_reducer_shared is not None:
                    reducer_shared_match = (
                        observed_reducer["dynamic_shared_memory_bytes"]
                        == captured_reducer_shared)
                reducer_fields_match = {
                    "grid": observed_reducer["grid"] == reducer_launch.grid,
                    "block": observed_reducer["block"] == reducer_launch.block,
                    "kernel_name": reducer_kernel_match,
                    "dynamic_shared_memory_bytes": reducer_shared_match,
                }
            rows.append({
                "case_id": case_id,
                "shape": shape,
                "dtype": dtype,
                "observed_kernel_count": len(observed_facts),
                "injected_kernel_count": 1 + (1 if reducer_launch else 0),
                "main_kernel_name": observed_main_name,
                "captured_main_symbol": captured_main_symbol,
                "captured_main_shared_bytes": captured_main_shared,
                "observed_reducer_name": (
                    observed_reducers[0]["kernel_name"] if observed_reducers else None),
                "captured_reducer_symbol": captured_reducer_symbol,
                "captured_reducer_shared_bytes": captured_reducer_shared,
                "injected_algorithm": injected_launch.algorithm,
                "kernel_name_match": kernel_name_match,
                "reducer_counts_match": reducer_counts_match,
                "reducer_fields_match": reducer_fields_match,
                "fields_match": fields_match,
                "injected_execution_plan_status": "available",
                "explicit_missing_fields": [
                    name for name, value in fields_match.items()
                    if value is None
                ] + (["reducer_fields"] if reducer_fields_match is None else []),
                "incompatible_fields": [
                    name for name, value in fields_match.items()
                    if value is False
                ] + (
                    [f"reducer.{name}"
                     for name, value in (reducer_fields_match or {}).items()
                     if value is False]
                ),
            })
    finally:
        # Restore the original LUT loader regardless of success or failure.
        h200_mod._observed_launch_lut = original_lut_fn

    covered = [row for row in rows if row["injected_execution_plan_status"] == "available"]
    if not covered:
        raise ValueError("nvidia_h200 has no GEMM cases covered by injected launch data")
    field_match_percent = {
        name: 100.0 * sum(
            1 for row in covered if row["fields_match"][name] is True)
        / len(covered)
        for name in ("grid", "block", "kernel_name", "dynamic_shared_memory_bytes")
    }
    reducer_field_match_percent = {
        name: 100.0 * sum(
            1 for row in covered
            if row["reducer_fields_match"] is not None
            and row["reducer_fields_match"][name] is True)
        / max(1, sum(1 for row in covered if row["reducer_fields_match"] is not None))
        for name in ("grid", "block", "kernel_name", "dynamic_shared_memory_bytes")
    }
    explicit_missing_counts = {
        name: sum(
            1 for row in covered if row["fields_match"][name] is None)
        for name in ("kernel_name", "dynamic_shared_memory_bytes", "registers_per_thread")
    }
    reducer_explicit_missing_counts = {
        name: sum(
            1 for row in covered
            if row["reducer_fields_match"] is not None
            and row["reducer_fields_match"][name] is None)
        for name in ("kernel_name", "dynamic_shared_memory_bytes")
    }
    kernel_name_match_count = sum(
        1 for row in covered if row["fields_match"]["kernel_name"] is True)
    dynamic_shared_match_count = sum(
        1 for row in covered if row["fields_match"]["dynamic_shared_memory_bytes"] is True)
    reducer_kernel_name_match_count = sum(
        1 for row in covered
        if row["reducer_fields_match"] is not None
        and row["reducer_fields_match"]["kernel_name"] is True)
    reducer_dynamic_shared_match_count = sum(
        1 for row in covered
        if row["reducer_fields_match"] is not None
        and row["reducer_fields_match"]["dynamic_shared_memory_bytes"] is True)
    reducer_total = sum(1 for row in covered if row["reducer_fields_match"] is not None)
    # Raw / candidate SHA binding for evidence.  All four segment
    # compact files were already verified against their canonical
    # capture receipts inside _load_capture_projections; recompute the
    # SHA here so the summary carries the actual bound values.
    candidate_sha = hashlib.sha256(v6_csv.read_bytes()).hexdigest()
    summary = {
        "hardware_id": "nvidia_h200",
        "gemm_cases": len(dataset),
        "compared_cases": len(rows),
        "missed_cases": len(misses),
        "injected_execution_plan_covered": len(covered),
        "injected_execution_plan_unavailable": len(misses),
        "candidate_csv_sha256": candidate_sha,
        "capture_raw_sha256": raw_sha256,
        "kernel_name_match_count": kernel_name_match_count,
        "dynamic_shared_memory_match_count": dynamic_shared_match_count,
        "reducer_kernel_name_match_count": reducer_kernel_name_match_count,
        "reducer_dynamic_shared_match_count": reducer_dynamic_shared_match_count,
        "reducer_total": reducer_total,
        "reducer_counts_match_count": sum(
            1 for row in covered if row["reducer_counts_match"]),
        "field_match_percent": field_match_percent,
        "reducer_field_match_percent": reducer_field_match_percent,
        "explicit_missing_counts": explicit_missing_counts,
        "reducer_explicit_missing_counts": reducer_explicit_missing_counts,
        "incompatible_field_counts": {
            name: sum(1 for row in covered if row["fields_match"][name] is False)
            for name in ("grid", "block", "kernel_name", "dynamic_shared_memory_bytes")
        },
        "reducer_incompatible_field_counts": {
            name: sum(
                1 for row in covered
                if row["reducer_fields_match"] is not None
                and row["reducer_fields_match"][name] is False)
            for name in ("grid", "block", "kernel_name", "dynamic_shared_memory_bytes")
        },
        "missing_injected_key_counts": {},
        "cluster_note": (
            "Cluster geometry is absent from the nsys kernel dataset; "
            "explicitly incomparable, no invented values."
        ),
        "registers_note": (
            "Registers per thread are absent from the v6 candidate; "
            "explicitly incomparable, no invented values."
        ),
        "historical_subset_note": (
            "Prior 1897-hit/13-miss/0-mismatch counts were a historical "
            "subset; this comparison covers the full 1910-case kernel "
            "dataset against the strict-loaded v6 candidate."
        ),
    }
    return {"summary": summary, "cases": rows, "misses": misses}


def compare_all_gemm_launch_sources() -> Mapping[str, object]:
    return {
        hardware_id: compare_gemm_launch_sources(hardware_id)
        for hardware_id in _SUPPORTED_HARDWARE_IDS
    }
