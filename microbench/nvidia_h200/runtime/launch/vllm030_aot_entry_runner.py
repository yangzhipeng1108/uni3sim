"""Independent CPU-only control service of the actual ready AOT dispatcher.

The installed PiecewiseBackend methods run unchanged on explicitly synthetic
compiled entries. Their runnable is an identity, with no model, Torch op,
tensor, compilation, CUDA graph or GPU execution. This excludes the compiled
body and cannot price its allocations, guards, submissions or cold paths.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    import torch
    import vllm
    from vllm.compilation.piecewise_backend import PiecewiseBackend, RangeEntry
    from vllm.config.utils import Range

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU control probe must not initialize CUDA")
    module = Path(inspect.getsourcefile(PiecewiseBackend))
    expected = args.source_root / "vllm/compilation/piecewise_backend.py"
    if module.read_bytes() != expected.read_bytes():
        raise ValueError("installed AOT dispatcher differs from frozen upstream source")
    marker = object()

    def identity(*inputs):
        return marker

    rows = []
    for mode in ("static", "dynamic_range", "dynamic_size"):
        # Ready-state fields only: no constructor/compiler or fake GPU state.
        backend = object.__new__(PiecewiseBackend)
        backend.sym_shape_indices = [] if mode == "static" else [0]
        ranges = [Range(start=1, end=128), Range(start=129, end=512), Range(start=513, end=8192)]
        backend.compile_ranges = ranges if mode != "static" else [ranges[0]]
        backend.compile_sizes = [256] if mode == "dynamic_size" else []
        backend.range_entries = {value: RangeEntry(value, compiled=True, runnable=identity)
                                 for value in backend.compile_ranges}
        if mode == "dynamic_size":
            value = Range(start=256, end=256)
            backend.range_entries[value] = RangeEntry(value, compiled=True, runnable=identity)
        inputs = (256, marker) if mode != "static" else (marker,)
        for pieces in (1, 37):
            controls = (identity,) * pieces
            dispatchers = (backend,) * pieces

            def chain(functions):
                result = None
                for function in functions:
                    result = function(*inputs)
                return result

            for functions in (controls, dispatchers):
                assert chain(functions) is marker
            repeats, samples = 4096, []
            for trial in range(15):
                timed = {}
                order = (("control", controls), ("dispatch", dispatchers))
                if trial % 2:
                    order = order[::-1]
                for label, functions in order:
                    started = time.perf_counter_ns()
                    for iteration in range(repeats):
                        result = chain(functions)
                    timed[label] = (time.perf_counter_ns() - started) / repeats
                    assert result is marker
                samples.append({"trial": trial, "control_chain_ns": timed["control"],
                    "dispatch_chain_ns": timed["dispatch"],
                    "increment_per_entry_ns": (timed["dispatch"] - timed["control"]) / pieces})
            rows.append({"mode": mode, "pieces": pieces, "repeats": repeats, "samples": samples,
                "median_dispatch_chain_ns": statistics.median(row["dispatch_chain_ns"] for row in samples),
                "median_increment_per_entry_ns": statistics.median(row["increment_per_entry_ns"] for row in samples)})
    if torch.cuda.is_initialized():
        raise RuntimeError("AOT CPU control unexpectedly initialized CUDA")
    report = {"kind": "independent_ready_vllm030_aot_cpu_control", "production_duration_consumed": False,
        "gpu_executed": False, "model_or_tensor_created": False, "forward_parameter_adopted": False,
        "identity": {"vllm_reported_version": vllm.__version__, "source_release_path": str(args.source_root),
            "torch": torch.__version__, "python": platform.python_version(),
            "cpu": [line.strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")][:1]},
        "source": {"module_path": str(module), "module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
            "upstream_commit": subprocess.check_output(["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True).strip(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "state": "object.__new__ with source Range/RangeEntry and compiled=True; identity runnable; no constructor claim",
        "limitations": ["actual source methods; synthetic ready CPU state and int dynamic shape, not a complete AOT body",
            "excludes compile/guard checks in the compiled callable, graph dispatch, tensor allocations, GPU submissions and cold initialization",
            "8358 CPU is a declared cross-host approximation, not historical8558 or Modal CPU",
            "control and dispatcher share loops/identity; raw signed paired increments are not clipped or used as Forward prices"],
        "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "rows": [{key: row[key] for key in (
        "mode", "pieces", "median_dispatch_chain_ns", "median_increment_per_entry_ns")} for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
