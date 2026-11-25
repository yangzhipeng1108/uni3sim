"""Fail-closed comparison of a synthesized trace against nsys ground truth."""

import argparse
import json
import sys
from pathlib import Path

from evaluation.serving.traces.tiered_nsys_comparison import (
    ExclusionRegistry,
    compare,
    format_report,
    nsys_ordered_launches_from_sqlite,
)


def main(argv: list[str] | None = None) -> int:
    """Tiered trace-vs-nsys comparison (Tier 0/1/2) with pass/fail exit status.

    Default mode runs the tiered comparator
    (`evaluation.serving.traces.tiered_nsys_comparison`):
      Tier 0 scheduler-sidecar correctness (when a sidecar is given),
      Tier 1 per-category kernel counts, Tier 2 ordered per-launch sequence after
      pre-registered exclusions. Exits nonzero on any tier failure so it can gate
      validation.
    """
    p = argparse.ArgumentParser(description="Tiered trace-vs-nsys comparison")
    p.add_argument("trace", help="synthesized/emulation kernel trace JSON")
    p.add_argument("nsys_sqlite", nargs="?",
                   help="nsys SQLite export (ordered launches for Tier 1+2)")
    p.add_argument("--sidecar", default=None,
                   help="scheduler sidecar JSONL for Tier 0 (default: <trace>_scheduler.jsonl if present)")
    p.add_argument("--registry", default=None,
                   help="exclusions registry YAML (default: configs/validation/serving/nsys_exclusions.yaml)")
    p.add_argument("--device-id", type=int, default=0,
                   help="nsys deviceId (rank) to compare (default 0)")
    p.add_argument("--comm-requirements", default=None,
                   help="public-communication gate spec: inline JSON, a path to a "
                        "JSON file, or a compact 'op:min[,op:min...]' list (e.g. "
                        "'nccl_send:1,nccl_recv:1'). When given, the synthesized "
                        "trace must contain those PUBLIC collective/KV-transfer "
                        "ops (nonzero bytes for send/recv) or the run FAILS — this "
                        "closes the AC-3 hole where nsys NCCL internals are "
                        "registry-excluded so a trace missing its comm op would "
                        "otherwise pass.")
    args = p.parse_args(argv)

    with open(args.trace) as f:
        synth_trace = json.load(f)

    if not args.nsys_sqlite:
        print("[compare] nsys SQLite is required for Tier 1/2.", file=sys.stderr)
        return 2
    # Structured ordered launches (name + grid + block) — Tier 2 carries grid/block
    # through for reporting; the count-multiset (names) feeds Tier 1.
    ordered_launches = nsys_ordered_launches_from_sqlite(args.nsys_sqlite,
                                                         device_id=args.device_id)
    sidecar_path = args.sidecar
    if sidecar_path is None:
        cand = args.trace + "_scheduler.jsonl"
        sidecar_path = cand if Path(cand).exists() else None
    sidecar_records = None
    if sidecar_path and Path(sidecar_path).exists():
        sidecar_records = [json.loads(l) for l in open(sidecar_path) if l.strip()]

    registry = ExclusionRegistry.load(args.registry)
    comm_requirements = _parse_comm_requirements(args.comm_requirements)
    result = compare(synth_trace, ordered_launches,
                     sidecar_records=sidecar_records, registry=registry,
                     comm_requirements=comm_requirements)
    print(format_report(result))
    return 0 if result["passed"] else 1


def _parse_comm_requirements(spec):
    """Parse --comm-requirements into the dict public_comm_gate expects.

    Accepts: None (no gate); a path to a JSON file; inline JSON; or a compact
    'op:min[,op:min...]' list.
    Returns None or a {op: {"min": int}} dict.
    """
    if not spec:
        return None
    p = Path(spec)
    if p.exists():
        spec = p.read_text()
    spec = spec.strip()
    if spec.startswith("{"):
        return json.loads(spec)
    out = {}
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            raise ValueError("communication requirement contains an empty item")
        if ":" not in tok:
            raise ValueError(
                f"communication requirement {tok!r} must use 'op:min'")
        op, _, mn = tok.partition(":")
        op = op.strip()
        if not op or not mn.strip():
            raise ValueError(
                f"communication requirement {tok!r} must use 'op:min'")
        if op in out:
            raise ValueError(f"duplicate communication requirement: {op!r}")
        out[op] = {
            "min": int(mn),
            "exact": None,
            "require_bytes": op in ("nccl_send", "nccl_recv"),
        }
    return out
