#!/usr/bin/env python3
"""Write the three H200 ShareGPT suite configs of one vLLM version.

Each suite copies the A800 suite of the same model group (models, graph
execution semantics, prediction contract) and replaces only what H200 changes:

* the hardware entry: ``nvidia_h200`` and its hardware config SHA256, with the
  CUDA Graph capture sizes of the captured vLLM version (read from the
  candidates' vLLM server logs, recorded in ``_CAPTURE_SIZES``);
* the H200 ProtoBridge collective profile SHA256s for TP2;
* the cells, pinned to the candidate root's manifest/requests SHA256.

Outputs ``configs/evaluation/experiments/nvidia_h200_<tag>_{sharegpt_tp1,
qwen3_30b_a3b,qwen3_32b}_suite.json``; rerun after rebuilding a candidate root.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.serving.metrics.sharegpt_compact_evaluation import (  # noqa: E402
    NVIDIA_H200_SUITE_SELECTIONS,
    load_phase1_config,
)
from src.kernel.predictors.protobridge.adapter import (  # noqa: E402
    development_profile_config_sha256,
)

_EXPERIMENTS = ROOT / "configs/evaluation/experiments"
_HARDWARE_CONFIG = "configs/hardware/nvidia_h200.yaml"
# vLLM ``cudagraph_capture_sizes`` in the H200 capture server logs
# (trace_off/logs/vllm_server.stdout.log, identical for all 15 cases).
_A800_STYLE_SIZES = (
    1, 2, 4, 8, *range(16, 513, 8))
_CAPTURE_SIZES = {
    "0.11.0": _A800_STYLE_SIZES,
    "0.30.0": (1, 2, 4, 8, *range(16, 257, 8), *range(272, 513, 16)),
}
_GROUPS = {
    # suite name: (A800 template suites, model id)
    "sharegpt_tp1": (("sharegpt_phase1.json",), None),
    "qwen3_30b_a3b": (("nvidia_a800_qwen3_30b_a3b_suite.json",), "qwen3-30b-a3b"),
    "qwen3_32b": (("qwen3_32b_qps1_suite.json", "qwen3_32b_qps4_qps16_suite.json"),
                  "qwen3-32b"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vllm-version", choices=tuple(_CAPTURE_SIZES), required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    args = parser.parse_args()
    candidate_root = args.candidate_root.resolve(strict=True)
    tag = {"0.11.0": "vllm011", "0.30.0": "vllm030"}[args.vllm_version]
    hardware_sha = _sha256(ROOT / _HARDWARE_CONFIG)
    for group, (templates, model_filter) in _GROUPS.items():
        payloads = [json.loads((_EXPERIMENTS / name).read_text()) for name in templates]
        base = copy.deepcopy(payloads[0])
        for other in payloads[1:]:
            for field in ("schema_version", "dataset_root", "models", "hardware",
                          "prediction_contract"):
                if other[field] != base[field]:
                    raise ValueError(f"{group} templates disagree at {field}")
        a800 = [entry for entry in base["hardware"] if entry["hardware_id"] == "nvidia_a800"]
        if len(a800) != 1:
            raise ValueError(f"{group} template has no single A800 entry")
        hardware = copy.deepcopy(a800[0])
        hardware.update({
            "hardware_id": "nvidia_h200",
            "hardware_config_path": _HARDWARE_CONFIG,
            "hardware_config_sha256": hardware_sha,
        })
        hardware["graph_execution"]["capture_sizes"] = list(
            _CAPTURE_SIZES[args.vllm_version])
        cells = []
        qps_values = []
        for payload in payloads:
            for cell in payload["cells"]:
                if cell["hardware_id"] != "nvidia_a800":
                    continue
                if model_filter is not None and cell["model_id"] != model_filter:
                    raise ValueError(f"{group} template holds another model")
                case_dir = candidate_root / "nvidia_h200" / cell["model_id"] / cell["case_id"]
                manifest = json.loads((case_dir / "manifest.json").read_text())
                if manifest["runtime"]["runtime_version"] != args.vllm_version:
                    raise ValueError(f"{case_dir} is not a vLLM {args.vllm_version} case")
                cells.append({
                    "hardware_id": "nvidia_h200",
                    "model_id": cell["model_id"],
                    "case_id": cell["case_id"],
                    "manifest_sha256": _sha256(case_dir / "manifest.json"),
                    "requests_sha256": _sha256(case_dir / "requests.csv"),
                })
            qps_values.extend(payload["qps_values"])
        prediction = copy.deepcopy(base["prediction_contract"])
        if prediction["tensor_parallel_size"] > 1:
            prediction["tensor_all_reduce_profile_sha256"] = (
                development_profile_config_sha256("nvidia_h200", "all_reduce"))
            prediction["tensor_all_gather_profile_sha256"] = (
                development_profile_config_sha256("nvidia_h200", "all_gather"))
        suite = {
            "schema_version": base["schema_version"],
            "suite_id": f"sharegpt_nvidia_h200_{tag}_{group}",
            "dataset_root": base["dataset_root"],
            "models": base["models"],
            "hardware": [hardware],
            "qps_values": sorted(qps_values),
            "cells": cells,
            "prediction_contract": prediction,
        }
        destination = _EXPERIMENTS / f"nvidia_h200_{tag}_{group}_suite.json"
        destination.write_text(json.dumps(suite, indent=2) + "\n")
        load_phase1_config(destination)
        print(destination.relative_to(ROOT), len(cells))
    expected = {
        f"configs/evaluation/experiments/nvidia_h200_{tag}_{group}_suite.json"
        for group in _GROUPS
    }
    if expected != set(NVIDIA_H200_SUITE_SELECTIONS[args.vllm_version]):
        raise AssertionError("suite file names disagree with the evaluator's selection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
