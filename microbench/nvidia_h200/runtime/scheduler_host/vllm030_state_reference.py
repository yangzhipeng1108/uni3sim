"""Fixed count-median/Theil-Sen reduction of independent native CPU calls.

Equal weights across prompt lengths and finish/continue states. Decode
represents mixed service explicitly; no mixed observation is fabricated.
No Serving observations, prediction errors, or selection by accuracy.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[4]
CAMPAIGN = ROOT / "profiler_output/microbench/nvidia_a800/serving_accuracy_scheduler_v030_20261107"
REFERENCE = ROOT / "microbench/nvidia_h200/runtime/scheduler_host/references/independent_state_vllm030_cpu.json"
PROFILE = ROOT / "configs/host_timing/scheduler_host_profiles/nvidia_h200_vllm030_native_cpu.json"
COUNTS = (1, 8, 32)


def fit(points):
    slopes = [(right[1] - left[1]) / (right[0] - left[0])
              for index, left in enumerate(points) for right in points[index + 1:]]
    slope = statistics.median(slopes)
    intercept = statistics.median(value - slope * count for count, value in points)
    if any(not math.isfinite(value) or value <= 0 for value in (slope, intercept)):
        raise ValueError("independent calls do not admit the fixed positive affine service; no clipping or policy search")
    return {"intercept": intercept * 1e-9, "per_request": slope * 1e-9}


def main():
    manifest_path = CAMPAIGN / "sdk_grid_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["production_duration_consumed"] is not False or manifest["model_or_gpu_executed"] is not False:
        raise ValueError("requires independent CPU-only mechanism evidence")
    if manifest["grid"] != {"counts": [1, 8, 32], "prompt_lengths": [64, 256], "output_counts": [1, 2, 3], "samples": 240}:
        raise ValueError("predeclared state grid changed")
    if len(manifest["runs"]) != 18:
        raise ValueError("requires all predeclared cells")
    cells = defaultdict(lambda: {"plan": [], "update": []})
    identities = []
    hosts = []
    for run in manifest["runs"]:
        path = ROOT / run["raw_path"]
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != run["sha256"]:
            raise ValueError("independent raw content differs")
        data = json.loads(content)
        identities.append(data["scheduler"]["source_identity"])
        hosts.append(data["host"])
        control = data["measurement"]["control_workload"]
        if (data["scheduler"]["vllm_version"] != "0.30.0"
                or data["scheduler"]["measurement_source"]["scheduler_class"] != "AsyncScheduler"
                or not control["async_scheduling"] or not control["enable_prefix_caching"]):
            raise ValueError("native source/config does not match declared CPU mechanism")
        for index, step in enumerate(data["steps"]):
            phase = "prefill" if index % run["output_tokens"] == 0 else "decode"
            outcome = "finish" if (index + 1) % run["output_tokens"] == 0 else "continue"
            if step["phase"] != phase or step["scheduled_request_count"] != run["request_count"]:
                raise ValueError("actual native phase/count differs from independent state")
            key = phase, outcome, run["prompt_tokens"], run["request_count"]
            for operation, start, end in (("plan", "schedule_start_ns", "schedule_end_ns"),
                ("update", "scheduler_update_start_ns", "scheduler_update_end_ns")):
                value = step[end] - step[start]
                if value <= 0:
                    raise ValueError("requires positive native CPU samples")
                cells[key][operation].append(value)
    if any(identity != identities[0] for identity in identities) or any(host != hosts[0] for host in hosts):
        raise ValueError("independent source/host changed during fixed grid")
    summaries = []
    for key, values in sorted(cells.items()):
        if min(len(samples) for samples in values.values()) < 32:
            raise ValueError("too few samples in an independent state cell")
        summaries.append({"phase": key[0], "outcome": key[1], "prompt_tokens": key[2], "count": key[3],
            **{operation: statistics.median(samples) for operation, samples in values.items()}})
    services = {}
    for phase in ("prefill", "decode"):
        services[phase] = {}
        for operation in ("plan", "update"):
            points = [(count, statistics.median(row[operation] for row in summaries
                if row["phase"] == phase and row["count"] == count)) for count in COUNTS]
            services[phase][operation] = fit(points)
    reference = {"kind": "independent_native_vllm030_cpu_scheduler_state_proxy", "source_identity": identities[0],
        "host": hosts[0], "production_duration_consumed": False,
        "parameter_rule": "equal_length_outcome_cell_medians_then_count_medians_then_Theil_Sen",
        "manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": hashlib.sha256(manifest_bytes).hexdigest()},
        "producer": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "cell_medians_ns": summaries, "service_time_s": services,
        "limitations": ["8358 CPU proxy, not target8558 measurement", "native0.30 CPU platform uses legacy worker flag; no GPU/runner certification",
            "mixed service uses independently measured decode count-affine proxy; mixed not observed",
            "count>32 extrapolates existing affine form; fixed two prompt lengths and finish/continue states"]}
    REFERENCE.write_text(json.dumps(reference, indent=2) + "\n")
    base = json.loads((ROOT / "configs/host_timing/scheduler_host_profiles/nvidia_h200_vllm030.json").read_text())
    base["scheduler"] = {"vllm_version": "0.30.0", "source_identity": "vllm=0.30.0;source_commit=" + identities[0]["source_commit"] + ";host_cpu=8358;independent_cpu_mechanism=true;decode_as_mixed_proxy=true"}
    del base["shared_cpu_timing_target"]
    base["service_time_s"] = {**services["decode"], "prefill": services["prefill"]}
    base["measurement"]["artifacts"] = [{"path": str(REFERENCE.relative_to(ROOT)), "sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest()}]
    PROFILE.write_text(json.dumps(base, indent=2) + "\n")
    print(json.dumps({"profile": str(PROFILE), "service_time_s": base["service_time_s"]}, indent=2))


if __name__ == "__main__":
    main()
