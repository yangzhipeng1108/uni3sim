#!/usr/bin/env python3
"""Rank current GEMM Forward error by static launch-program mechanism."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.kernel.dataset import (  # noqa: E402
    get_hardware_config_path,
    load_kernel_dataset,
)
from evaluation.kernel.evaluation_runner import process_op_star  # noqa: E402
from evaluation.kernel.mechanism_validation_report import (  # noqa: E402
    _kernel_launch_class,
)


HARDWARE_IDS = ("nvidia_a800", "metax_c500", "ascend_910b")
OUTPUT_ROOT = ROOT / "experiment_logs"
SCHEMA_VERSION = "gemm_mechanism_cluster_ranking_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_output(path: Path) -> Path:
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if root not in resolved.parents:
        raise ValueError(
            f"output must be a file below {root}, got {resolved}")
    if resolved.suffix != ".json":
        raise ValueError("output must end in .json")
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {resolved}")
    return resolved


def _work():
    work = []
    identities = {}
    for hardware_id in HARDWARE_IDS:
        dataset = load_kernel_dataset(hardware_id, ("gemm",))
        config = get_hardware_config_path(hardware_id)
        identities[hardware_id] = {
            "case_count": len(dataset),
            "hardware_config": str(Path(config).relative_to(ROOT)),
            "hardware_config_sha256": _sha256(Path(config)),
        }
        for case_id, entry in sorted(dataset.items()):
            work.append((
                hardware_id,
                (
                    case_id,
                    entry,
                    config,
                    True,
                    True,
                    False,
                    0.0,
                    False,
                    None,
                    True,
                ),
            ))
    return work, identities


def _result_row(hardware_id, result):
    case_id, metadata, _, forward_pair, _, _, kernels = result
    if metadata["op_type"] != "GEMM":
        raise ValueError(f"non-GEMM result in GEMM diagnostic: {case_id!r}")
    if not isinstance(forward_pair, tuple) or len(forward_pair) != 2:
        raise ValueError(f"GEMM result has no Forward pair: {case_id!r}")
    predicted_ns, observed_ns = forward_pair
    if (not math.isfinite(predicted_ns) or not math.isfinite(observed_ns)
            or predicted_ns <= 0.0 or observed_ns <= 0.0):
        raise ValueError(f"GEMM result has invalid timing pair: {case_id!r}")
    if not kernels:
        raise ValueError(f"GEMM result has no kernel mechanisms: {case_id!r}")
    launch_classes = []
    for kernel in kernels:
        mechanism = kernel["mechanisms"]
        launch_class = dict(_kernel_launch_class(mechanism))
        launch_class["source_program"] = mechanism["observed_launch"][
            "source_program"]
        launch_classes.append(launch_class)
    signature = json.dumps(
        launch_classes, separators=(",", ":"), sort_keys=True)
    return {
        "hardware_id": hardware_id,
        "case_id": case_id,
        "predicted_ns": predicted_ns,
        "observed_ns": observed_ns,
        "signed_error_ns": predicted_ns - observed_ns,
        "absolute_error_ns": abs(predicted_ns - observed_ns),
        "absolute_error_pct": abs(predicted_ns - observed_ns) / observed_ns * 100.0,
        "gemm_m": metadata["gemm_m"],
        "gemm_n": metadata["gemm_n"],
        "gemm_k": metadata["gemm_k"],
        "execution_signature": signature,
        "launch_classes": launch_classes,
    }


def _clusters(rows):
    grouped = {}
    for row in rows:
        key = (row["hardware_id"], row["execution_signature"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)
    total_absolute_error_ns = math.fsum(
        row["absolute_error_ns"] for row in rows)
    if total_absolute_error_ns <= 0.0:
        raise ValueError("GEMM absolute error numerator must be positive")
    clusters = []
    for (hardware_id, signature), members in grouped.items():
        members.sort(key=lambda row: (-row["absolute_error_ns"], row["case_id"]))
        absolute_error_ns = math.fsum(
            row["absolute_error_ns"] for row in members)
        observed_ns = math.fsum(row["observed_ns"] for row in members)
        if observed_ns <= 0.0:
            raise ValueError("GEMM mechanism cluster observed time must be positive")
        clusters.append({
            "hardware_id": hardware_id,
            "execution_signature": signature,
            "launch_classes": members[0]["launch_classes"],
            "case_count": len(members),
            "observed_ns": observed_ns,
            "signed_error_ns": math.fsum(
                row["signed_error_ns"] for row in members),
            "absolute_error_ns": absolute_error_ns,
            "wmae_pct": absolute_error_ns / observed_ns * 100.0,
            "three_hardware_gemm_numerator_share_pct": (
                absolute_error_ns / total_absolute_error_ns * 100.0),
            "cases": members,
        })
    clusters.sort(key=lambda cluster: (
        -cluster["absolute_error_ns"],
        cluster["hardware_id"],
        cluster["execution_signature"],
    ))
    return clusters, total_absolute_error_ns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.jobs <= 0:
        raise ValueError("--jobs must be positive")
    output = _validate_output(arguments.output)
    work, identities = _work()
    packed = [item[1] for item in work]
    if arguments.jobs == 1:
        results = [process_op_star(item) for item in packed]
    else:
        with ProcessPoolExecutor(max_workers=arguments.jobs) as pool:
            results = list(pool.map(process_op_star, packed))
    rows = [
        _result_row(hardware_id, result)
        for (hardware_id, _), result in zip(work, results, strict=True)
    ]
    clusters, total_absolute_error_ns = _clusters(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "scope": "current_forward_gemm_static_mechanism_ranking",
        "forward_commit": _git_commit(),
        "hardware": identities,
        "case_count": len(rows),
        "three_hardware_gemm_absolute_error_ns": total_absolute_error_ns,
        "cluster_count": len(clusters),
        "clusters": clusters,
    }
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "case_count": len(rows),
        "cluster_count": len(clusters),
        "top_cluster": {
            name: clusters[0][name]
            for name in (
                "hardware_id", "case_count", "absolute_error_ns",
                "three_hardware_gemm_numerator_share_pct",
            )
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
