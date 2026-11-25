"""Derive copy-only cold readiness: fixed latency plus finite byte service."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics


def derive(path):
    raw = json.loads(path.read_text())
    if (raw["identity"]["exit_code"] or raw["build"]["exit_code"]
            or raw["measurement"]["exit_code"] or raw["counter"]["exit_code"]):
        raise ValueError("copy or counter capture failed")
    if (raw["identity"]["stdout"].split(",", 1)[0].strip() != "NVIDIA H200"
            or float(raw["final_clock"]["stdout"].split()[0]) != 1980):
        raise ValueError("copy capture hardware/clock domain differs")
    for filename, field in (("executed_source.cu", "source_sha256"),
                            ("executed_runner.py", "runner_sha256")):
        if hashlib.sha256((path.parent / filename).read_bytes()).hexdigest() != raw[field]:
            raise ValueError("copy capture source/runner identity differs")
    rows = list(csv.DictReader(io.StringIO(raw["measurement"]["stdout"])))
    profiles, validation = [], []
    for producers in (1, 8, 32, 132):
        medians = {size: statistics.median(int(row["cycles"]) / 1.98 for row in rows
            if int(row["grid"]) == producers and int(row["bytes"]) == size
            and int(row["state"]) == 0)
            for size in (128, 512, 2048, 8192, 16384, 32768, 65536)}
        # Freeze the two endpoint derivation before evaluating any operator:
        # T(b)=L+b/B. Intermediate payloads validate the approximation only.
        byte_rate_ns = (65536 - 128) / (medians[65536] - medians[128])
        latency_ns = medians[128] - 128 / byte_rate_ns
        if latency_ns <= 0 or byte_rate_ns <= 0:
            raise ValueError("nonpositive independent copy service")
        profiles.append({"producers": producers, "latency_ns": latency_ns,
            "aggregate_bandwidth_bytes_s": byte_rate_ns * producers * 1e9})
        validation.extend({"producers": producers, "bytes": size,
            "observed_ready_ns": value,
            "affine_ready_ns": latency_ns + size / byte_rate_ns}
            for size, value in medians.items())
    lines = raw["counter"]["stdout"].splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
    counters = [row for row in csv.DictReader(lines[start:]) if row["ID"].isdigit()]
    if len(counters) != 2:
        raise ValueError("expected paired cold/repeated counters")
    cold_bytes = float(counters[0]["dram__bytes_read.sum"].replace(",", ""))
    warm_bytes = float(counters[1]["dram__bytes_read.sum"].replace(",", ""))
    payload = 132 * 32768
    if cold_bytes < .8 * payload or warm_bytes > .1 * payload:
        raise ValueError("counter does not establish cold vs repeated input")
    evidence = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_sha256": raw["source_sha256"], "runner_sha256": raw["runner_sha256"],
        "cold_dram_bytes": cold_bytes, "repeated_dram_bytes": warm_bytes,
        "payload_bytes": payload, "validation": validation}
    return profiles, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp-raw", type=Path, required=True)
    parser.add_argument("--tma-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profiles, evidence = {}, {}
    for name, raw in (("cp_async", args.cp_raw), ("tma", args.tma_raw)):
        profiles[name], evidence[name] = derive(raw)
    ref = {"hardware_id": "nvidia_h200", "mechanism": "cold_async_copy_ready",
        "production_or_evaluation_duration_consumed": False,
        "producer_threads": 128, "resident_ctas_per_sm": 1,
        "minimum_payload_bytes": 128, "maximum_payload_bytes": 65536,
        "measured_sm_clock_hz": 1980000000, "profiles": profiles,
        "derivation": "Per producer-count endpoint secant at 128 B / 64 KiB; median over five trials and CTAs. L is ns, B is aggregate bytes/s. Other payloads validate only. No production durations or parameters selected by operator accuracy.",
        "combination": "First-wave initial K readiness only, competing by maximum with source K-ready. Retain completion floors from measured producer cohorts no larger than demand to preserve nondecreasing contention. Not a steady per-panel bandwidth or an additive whole-operator price.",
        "limitations": ["affine finite-burst approximation; intermediate payload residuals retained",
            "fixed 1.98 GHz clock conversion; no per-operator frequency inference",
            "one resident CTA and independent contiguous source regions",
            "production address order, bank mapping and consumer overlap remain partial"],
        "evidence": evidence}
    args.output.write_text(json.dumps(ref, indent=2) + "\n")


if __name__ == "__main__":
    main()
