#!/usr/bin/env python3
"""Published C500 host request path probes, re-run with an idle gap per call.

``microbench/nvidia_a800/runtime/host_request_path/vllm_host_request_path_runner.py``
times every region back to back, so each call finds its code hot.  Serving
requests reach these regions after tens of milliseconds of other work; the
render-region sweep (``render_gap_sweep_runner.py``) showed a cold plateau
beyond a 3--10 ms knee.  This runner drives the *same* published measurement
functions (API entry, input processor, admission/enqueue region, EngineCore
IPC region) with one change only: an idle gap before each timed call.  The
functions are taken from the published runner's source at run time and the
gap is the only textual substitution, so the regions and statistics are the
published ones.  No serving or evaluation duration is read.

Usage (inside the C500 vLLM container, cwd = checkout):
  python cold_host_request_path_runner.py OUT.json
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RUNNER_DIR = REPOSITORY_ROOT / "microbench/nvidia_a800/runtime/host_request_path"
sys.path.insert(0, str(RUNNER_DIR))
import vllm_host_request_path_runner as runner  # noqa: E402

if "MIDLAYER_C500_TOKENIZER" not in os.environ:
    raise SystemExit("MIDLAYER_C500_TOKENIZER must name the tokenizer directory")
TOKENIZER = os.environ["MIDLAYER_C500_TOKENIZER"]
GAPS_MS = (0, 10, 30, 100, 250)
_GAP = {"s": 0.0}


def _gapped(function_name, old, new):
    source = inspect.getsource(getattr(runner, function_name))
    if source.count(old) != 1:
        raise RuntimeError(f"{function_name}: gap anchor not unique")
    namespace = runner.__dict__
    exec(compile(source.replace(old, new), runner.__file__, "exec"), namespace)


def _time_calls_gapped(operation, trials, warmup):
    for _ in range(warmup):
        time.sleep(_GAP["s"])
        operation()
    samples = []
    for _ in range(trials):
        time.sleep(_GAP["s"])
        started_ns = time.perf_counter_ns()
        operation()
        samples.append(time.perf_counter_ns() - started_ns)
    return samples


def main():
    from transformers import AutoTokenizer

    runner._GAP = _GAP
    runner.__dict__["_time_calls"] = _time_calls_gapped
    _gapped("measure_http_path",
            "        for index in range(warmup + trials):\n            sent_ns = time.perf_counter_ns()",
            "        for index in range(warmup + trials):\n            time.sleep(_GAP['s'])\n            sent_ns = time.perf_counter_ns()")
    _gapped("measure_engine_core_ipc_region",
            "            time.sleep(0.001)",
            "            time.sleep(max(0.001, _GAP['s']))")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    cells = []
    for gap in GAPS_MS:
        _GAP["s"] = gap / 1000
        cell = {
            "gap_ms": gap,
            "api_entry": runner.measure_http_path(60, 5)["api_entry"],
            "input_processor": runner.measure_input_processor(TOKENIZER, (16, 128, 1024), 40, 5),
            "admission_enqueue_region": runner.measure_admission_enqueue_region(tokenizer, 86, 60, 5),
            "engine_core_ipc_region": runner.measure_engine_core_ipc_region(tokenizer, 86, 60, 5),
        }
        cells.append(cell)
        print("GAP", gap, json.dumps({
            "api_entry_us": cell["api_entry"]["median_s"] * 1e6,
            "input_processor_intercept_us": cell["input_processor"]["intercept_s"] * 1e6,
            "admission_us": cell["admission_enqueue_region"]["median_s"] * 1e6,
            "ipc_us": cell["engine_core_ipc_region"]["median_s"] * 1e6}), flush=True)
    published = runner.__file__
    with open(sys.argv[1], "w") as handle:
        json.dump({
            "schema_version": 1, "hardware_id": "metax_c500", "tokenizer_path": TOKENIZER,
            "published_runner": {"path": os.path.relpath(published, REPOSITORY_ROOT),
                                 "sha256": hashlib.sha256(open(published, "rb").read()).hexdigest()},
            "gap_substitution": "time.sleep(gap) before each timed call; nothing else changed",
            "production_or_evaluation_duration_consumed": False,
            "cells": cells,
        }, handle, indent=1, default=str)


if __name__ == "__main__":
    main()
