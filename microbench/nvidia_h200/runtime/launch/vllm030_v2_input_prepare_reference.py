"""Derive an opt-in constant CPU proxy from independent V2 preparation.

Rule: median of all nine predeclared cell medians, with equal cell weights.
Only Host enqueue samples are consumed. GPU completion/event times, Serving
data and prediction errors are not inputs. Existing wrapper services retain
their values and provenance as legacy vLLM0.11 proxies; the new V2
measurement uses actual vLLM0.30. Mixed software versions are explicit.
"""
from pathlib import Path
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / "profiler_output/microbench/nvidia_h200/serving_accuracy_v2_input_prepare_20261107/raw.json"
REFERENCE = ROOT / "microbench/nvidia_h200/runtime/launch/references/vllm030_v2_input_prepare.json"
PROFILE = ROOT / "configs/host_timing/host_runtime_prepare_profiles/nvidia_h200_vllm030_v2_input.json"
BASE_PROFILE = ROOT / "configs/host_timing/host_runtime_prepare_profiles/nvidia_h200_vllm030_eager.json"
MECHANISM = "vllm030_v2_input_prepare"
CELLS = ((1, 1), (8, 1), (32, 1), (128, 1), (1, 32), (8, 32), (32, 32), (1, 256), (8, 256))


def main():
    raw_bytes = RAW.read_bytes()
    execution = json.loads(raw_bytes)
    if execution["status"] != "complete":
        raise ValueError("requires complete independent measurement")
    data = execution["raw"]
    if (data["cpu_preflight_only"] is not False or data["production_duration_consumed"] is not False
            or data["model_or_sampler_executed"] is not False
            or data["kind"] != "independent_vllm030_v2_input_preparation"):
        raise ValueError("not an independent native V2 preparation measurement")
    rows = data["rows"]
    if tuple((row["requests"], row["query_tokens_each"]) for row in rows) != CELLS:
        raise ValueError("predeclared mechanism matrix changed")
    medians = []
    for row in rows:
        values = row["host_enqueue_ns"]
        if len(values) != 64 or any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("requires all64 positive Host enqueue samples in each cell")
        median = statistics.median(values)
        if median != row["median_host_enqueue_ns"]:
            raise ValueError("stored independent median disagrees with raw samples")
        medians.append(median)
    seconds = statistics.median(medians) * 1e-9
    reference = {"kind": "independent_vllm030_v2_input_prepare_constant_proxy",
        "mechanism": MECHANISM, "duration_unit": "s_per_worker_step",
        "timer_domain": "host_enqueue_only_no_completion_sync",
        "rule": "equal_cell_median_of_all_nine_predeclared_cell_medians",
        "service_time_s": seconds, "source_method_sha256": data["method_sha256"],
        "raw": {"path": str(RAW.relative_to(ROOT)), "sha256": hashlib.sha256(raw_bytes).hexdigest()},
        "producer": {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                     "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "measurement_boundary": data["measurement_boundary"], "measurement_torch": data["torch"],
        "production_duration_consumed": False,
        "limitations": data["limitations"] + [
            "explicit cross-host/software CPU proxy, not target-host performance certification",
            "constant preparation overhead approximates measured request/token-count variation",
            "input kernel and H2D device work remains unbound; only measured Host enqueue is priced",
            "no state-update, attention-metadata builder, sampler, completion fence or per-layer charge"]}
    REFERENCE.write_text(json.dumps(reference, indent=2) + "\n")
    base = json.loads(BASE_PROFILE.read_text())
    if MECHANISM in base["service_time_s"]:
        raise ValueError("base profile already contains the V2 input mechanism")
    base["service_time_s"][MECHANISM] = seconds
    # Wrapper profiles retain the vLLM0.11/Torch2.8/cu128 proxy sources, while
    # V2 preparation is measured on2.13/cu130. Neither is relabeled.
    base["framework"] = {"name": "PyTorch (vLLM0.11 wrapper proxies / vLLM0.30 V2 input)",
                         "version": "2.8.0+cu128 wrappers / 2.13.0+cu130 V2 input proxy"}
    base["runtime"] = {"name": "CUDA (explicit mixed mechanism sources)", "version": "12.8 wrappers / 13.0 V2 input proxy"}
    base["measurement"]["artifacts"].append({"path": str(REFERENCE.relative_to(ROOT)),
        "sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest()})
    PROFILE.write_text(json.dumps(base, indent=2) + "\n")
    print(json.dumps({"profile": str(PROFILE), "service_time_s": seconds, "cell_count": len(rows)}))


if __name__ == "__main__":
    main()
