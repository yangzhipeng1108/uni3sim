#!/usr/bin/env python3
"""CLI for creating one frozen ShareGPT 1P1D PD capture plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.serving.experiments.pd_capture_plan import (  # noqa: E402
    materialize_pd_capture_plan,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize one real 1P1D PD capture plan")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--suite-manifest", required=True)
    parser.add_argument("--colocated-ground-truth", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prefill-devices", required=True, type=int, nargs="+")
    parser.add_argument("--decode-devices", required=True, type=int, nargs="+")
    parser.add_argument("--prefill-http-port", required=True, type=int)
    parser.add_argument("--decode-http-port", required=True, type=int)
    parser.add_argument("--router-port", required=True, type=int)
    parser.add_argument("--prefill-kv-ip", required=True)
    parser.add_argument("--decode-kv-ip", required=True)
    parser.add_argument("--prefill-kv-port", required=True, type=int)
    parser.add_argument("--decode-kv-port", required=True, type=int)
    parser.add_argument("--connector", required=True)
    parser.add_argument(
        "--runtime-source-commit",
        help="explicit full vLLM executor commit when the frozen manifest lacks it",
    )
    parser.add_argument("--observer-effect-validation", required=True, type=Path)
    parser.add_argument(
        "--trace-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--canary-observer-pair", action="store_true")
    args = parser.parse_args(argv)
    validation = json.loads(args.observer_effect_validation.read_text(encoding="utf-8"))
    if not isinstance(validation, dict):
        raise ValueError("observer-effect validation must be an object")
    plan = materialize_pd_capture_plan(
        model=args.model, dataset=args.dataset, case_manifest=args.case_manifest,
        suite_manifest=args.suite_manifest,
        colocated_ground_truth=args.colocated_ground_truth,
        output_root=args.output_root,
        prefill_devices=args.prefill_devices, decode_devices=args.decode_devices,
        prefill_http_port=args.prefill_http_port, decode_http_port=args.decode_http_port,
        router_port=args.router_port, prefill_kv_ip=args.prefill_kv_ip,
        decode_kv_ip=args.decode_kv_ip, prefill_kv_port=args.prefill_kv_port,
        decode_kv_port=args.decode_kv_port, connector=args.connector,
        runtime_source_commit=args.runtime_source_commit,
        observer_effect_validation=validation, trace_enabled=args.trace_enabled,
        canary_observer_pair=args.canary_observer_pair,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
