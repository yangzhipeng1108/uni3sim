"""Reduce only independent request-state controls to source-count Host services."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / "profiler_output/microbench/nvidia_h200/serving_accuracy_v2_state_admission_20261108"
OUTPUT = Path(__file__).with_name("references") / "vllm030_v2_state_admission.json"
PREFIX = "vllm030_v2_state_"


def key(row):
    return (row["mode"], row["cached_requests"], row["finished_requests"], row["new_requests"], row["new_prompt_tokens_each"])


def slopes(points):
    return [(yb - ya) / (xb - xa) for index, (xa, ya) in enumerate(points)
        for xb, yb in points[index + 1:] if xb != xa]


data = json.loads((RAW / "raw.json").read_text())
preflight = json.loads((RAW / "local_source_preflight.json").read_text())
if data["status"] != "complete" or len(data["runs"]) != 2:
    raise ValueError("two complete source-state process rotations required")
groups = {}
for index, run in enumerate(data["runs"]):
    raw = run["raw"]
    if (run["exit_code"] != 0 or raw["status"] != "complete" or raw["rotation"] != index
            or raw["source_sha256"] != preflight["source_sha256"]
            or raw["production_duration_consumed"] is not False
            or raw["model_weights_or_attention_or_input_or_metadata_or_sampling_math_executed"] is not False):
        raise ValueError("source or independent mechanism boundary changed")
    if len(raw["rows"]) != 25 or len({key(row) for row in raw["rows"]}) != 25:
        raise ValueError("missing or repeated predeclared cell")
    for row in raw["rows"]:
        if len(row["samples"]) != 32 or any(value["state_ns"] <= 0 for value in row["samples"]):
            raise ValueError("incomplete state timing samples")
        if row["median_ns"]["state_ns"] != statistics.median(value["state_ns"] for value in row["samples"]):
            raise ValueError("state median changed")
        identity = key(row)
        if identity not in groups:
            groups[identity] = []
        groups[identity].append(row["median_ns"]["state_ns"])
if any(len(values) != 2 for values in groups.values()):
    raise ValueError("process rotations have different cells")
cells = [{"mode": k[0], "cached_requests": k[1], "finished_requests": k[2], "new_requests": k[3],
    "new_prompt_tokens_each": k[4], "median_ns": statistics.median(values),
    "rotation_medians_ns": values} for k, values in groups.items()]
empty, = (row for row in cells if row["mode"] == "empty")
base = empty["median_ns"]
cached = [row for row in cells if row["mode"] == "cached"]
cache_rate = statistics.median(slopes([(0, base)] + [(row["cached_requests"], row["median_ns"]) for row in cached]))
finished = [row for row in cells if row["mode"] == "finish"]
finish_rate = statistics.median(slopes([(row["finished_requests"], row["median_ns"]) for row in finished]))
finish_fixed = statistics.median(row["median_ns"] - base - row["finished_requests"] * finish_rate for row in finished)
admitted = [row for row in cells if row["mode"] == "admit"]
token_slopes = []
for count in (1, 8, 32, 128):
    token_slopes.extend(slopes([(row["new_requests"] * row["new_prompt_tokens_each"], row["median_ns"])
        for row in admitted if row["new_requests"] == count]))
token_rate = statistics.median(token_slopes)
request_slopes = []
for prompt in (32, 256, 1024):
    request_slopes.extend(slopes([(row["new_requests"], row["median_ns"]
        - row["new_requests"] * prompt * token_rate) for row in admitted if row["new_prompt_tokens_each"] == prompt]))
request_rate = statistics.median(request_slopes)
admit_fixed = statistics.median(row["median_ns"] - base - row["new_requests"] * request_rate
    - row["new_requests"] * row["new_prompt_tokens_each"] * token_rate for row in admitted)
parameters = {"fixed": base, "cached_request": cache_rate, "finish_nonempty": finish_fixed,
    "finished_request": finish_rate, "admit_nonempty": admit_fixed, "admitted_request": request_rate,
    "admitted_token": token_rate}
if any(value <= 0 for value in parameters.values()):
    raise ValueError("source-count controls did not produce positive services; no clipping/fallback")
for row in cells:
    c, f, n, length = row["cached_requests"], row["finished_requests"], row["new_requests"], row["new_prompt_tokens_each"]
    prediction = base + c * cache_rate + (finish_fixed + f * finish_rate if f else 0)
    prediction += admit_fixed + n * request_rate + n * length * token_rate if n else 0
    row.update(prediction_ns=prediction, absolute_relative_error=abs(prediction - row["median_ns"]) / row["median_ns"])
mixed = [row for row in cells if row["mode"] == "mixed"]
payload = {"kind": "independent_vllm030_v2_worker_request_state", "time_unit": "seconds",
    "rule": "equal-process cell medians; sequential Theil-Sen token/count slopes; median source intercepts; mixed cells validation only",
    "services_s": {PREFIX + name: value * 1e-9 for name, value in parameters.items()},
    "units": {PREFIX + name: unit for name, unit in (("fixed", "per_worker_step"), ("cached_request", "per_cached_request"),
        ("finish_nonempty", "per_step_with_finished_requests"), ("finished_request", "per_finished_request"),
        ("admit_nonempty", "per_step_with_admitted_requests"), ("admitted_request", "per_admitted_request"),
        ("admitted_token", "per_newly_uploaded_token_id"))},
    "geometry": {"max_requests": 256, "max_model_len": 8192, "new_request_counts": [1, 8, 32, 128],
        "prompt_lengths": [32, 256, 1024], "block_size_tokens": 16, "state_vocab_size": 256},
    "validation": {"mixed_cell_count": len(mixed), "mixed_max_relative_error": max(row["absolute_relative_error"] for row in mixed),
        "all_cells_max_relative_error": max(row["absolute_relative_error"] for row in cells)},
    "source_sha256": preflight["source_sha256"], "production_duration_consumed": False,
    "raw": {"path": str((RAW / "raw.json").relative_to(ROOT)), "sha256": hashlib.sha256((RAW / "raw.json").read_bytes()).hexdigest()},
    "producer": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    "cells": cells, "limitations": ["synthetic ordinary no-hit greedy state; no constructor or historical CPU/provider certification",
        "no block append/copy-on-write/zeroing or speculative state; model/input/metadata/sampling mathematics excluded",
        "token/count affine terms approximate source Python copying/staging and shared nonempty flush; no per-shape time lookup",
        "only configured256 slots/8192 context and INT32 token/block16 state were measured; prompt lengths outside32..1024 are linear source-copy extrapolations"]}
OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({"services_s": payload["services_s"], "validation": payload["validation"]}))
