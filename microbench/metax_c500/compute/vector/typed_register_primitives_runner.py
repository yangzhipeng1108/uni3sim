#!/usr/bin/env python3
"""CPU reference or explicit owned-device capture for C500 register mechanisms.

No production operator durations are used. This runner does not select an idle
card: --device-id is mandatory for GPU modes, after the executor's preflight.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import tempfile

SOURCE = Path(__file__).with_name("typed_register_primitives.cpp")
PRIMITIVES = ("f32_gt", "f32_eq", "f32_nan", "s64_lt", "u64_gt", "shuffle32")
FIELDS = (
    "hardware_id", "device_name", "device_id", "ap_count", "wave_size", "clock_khz",
    "sample", "primitive", "chains", "threads", "grid", "outer", "wave_index",
    "block_index", "wave_in_block", "begin_cycle", "end_cycle", "wave_bundles",
    "checksum", "validation_status",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(argv, output):
    result = subprocess.run([str(value) for value in argv], capture_output=True, text=True)
    output.write_text(result.stdout)
    output.with_suffix(output.suffix + ".stderr").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"exit {result.returncode}: {argv}; see {output}")
    return result.stdout


def cpu_reference(output, cxx):
    with tempfile.TemporaryDirectory(prefix="midlayer-c500-typed-cpu-") as temp:
        binary = Path(temp) / "typed_cpu"
        compile_argv = [cxx, "-O2", "-std=c++17", "-Wall", "-Wextra", "-pedantic",
                        "-DMIDLAYER_TYPED_CPU_ONLY", SOURCE, "-o", binary]
        command(compile_argv, output / "cpu_build.log")
        result = json.loads(command([binary, "--self-test"], output / "cpu_reference.json"))
        if result["passed"] is not True or result["GPU_executed"] is not False:
            raise ValueError("CPU reference failed")
        return {"command": [str(v) for v in compile_argv], "result": result}


def summarize(path, primitive, chains, threads, grid, outer, samples, device):
    with path.open() as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames) != FIELDS:
            raise ValueError("unexpected C500 typed CSV schema")
        rows = list(reader)
    waves = threads * grid // 64
    if len(rows) != waves * samples:
        raise ValueError("missing or extra wave records")
    seen = set()
    intervals = []
    per_wave_cycles = []
    for row in rows:
        if row["hardware_id"] != "metax_c500" or row["primitive"] != primitive or row["validation_status"] != "pass":
            raise ValueError("identity or GPU correctness validation failed")
        integers = {key: int(row[key]) for key in FIELDS
                    if key not in ("hardware_id", "device_name", "primitive", "validation_status")}
        if (integers["device_id"], integers["ap_count"], integers["wave_size"]) != (device, 104, 64):
            raise ValueError("device does not match explicit C500 capture scope")
        if (integers["chains"], integers["threads"], integers["grid"], integers["outer"]) != (chains, threads, grid, outer):
            raise ValueError("configuration mismatch")
        sample, wave = integers["sample"], integers["wave_index"]
        if not 0 <= sample < samples or not 0 <= wave < waves:
            raise ValueError("sample or wave outside capture scope")
        if (integers["block_index"], integers["wave_in_block"]) != (wave // (threads // 64), wave % (threads // 64)):
            raise ValueError("wave does not match its block")
        if (sample, wave) in seen:
            raise ValueError("duplicate sample/wave")
        seen.add((sample, wave))
        bundles = outer * 31 * chains
        elapsed = integers["end_cycle"] - integers["begin_cycle"]
        if bundles != integers["wave_bundles"] or elapsed <= 0:
            raise ValueError("invalid work or same-wave interval")
        intervals.append(elapsed)
        per_wave_cycles.append(elapsed / bundles)
    return {
        "unit": "wave64 b32 XOR-distance1 shuffle" if primitive == "shuffle32" else "wave64 typed compare plus select bundle",
        "primitive": primitive, "chains": chains, "threads": threads, "grid": grid,
        "outer": outer, "samples": samples, "wave_records": len(rows),
        "cycles_per_wave_bundle_median": statistics.median(per_wave_cycles),
        "cycles_per_wave_bundle_min": min(per_wave_cycles),
        "cycles_per_wave_bundle_max": max(per_wave_cycles),
        "wave_bundles_per_local_cycle_median": statistics.median(1 / v for v in per_wave_cycles),
        "lane_bundles_per_local_cycle_median": statistics.median(64 / v for v in per_wave_cycles),
        "same_wave_elapsed_cycles_min": min(intervals),
        "same_wave_elapsed_cycles_max": max(intervals),
        "physical_ap_id_available": False, "aggregate_device_throughput_claimed": False,
        "GPU_correctness": "pass", "Forward_binding_approved": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true", help="CPU only; does not load MACA or query device")
    mode.add_argument("--smoke", action="store_true", help="12 small correctness launches")
    mode.add_argument("--matrix", action="store_true", help="bounded independent mechanism matrix")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--device-id", type=int)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output directory already exists; do not overwrite raw capture")
    if args.self_test and (args.binary is not None or args.device_id is not None):
        raise ValueError("CPU reference must not specify binary or device")
    if not args.self_test and (args.binary is None or args.device_id is None or args.device_id < 0):
        raise ValueError("GPU modes require explicit --binary and nonnegative --device-id")
    if args.trial < 1 or args.trial > 3:
        raise ValueError("trial must be 1, 2 or 3")
    args.output.mkdir(parents=True)
    receipt = {
        "schema": "metax_c500_typed_register_capture_v1", "hardware_id": "metax_c500",
        "source_sha256": sha256(SOURCE), "runner_sha256": sha256(__file__),
        "GPU_executed": False, "Forward_binding_approved": False,
        "production_duration_consumed": False, "trial": args.trial,
        "units": {"compare_select": "one wave64 typed compare plus select bundle (not a single predicate)",
                  "shuffle32": "one wave64 b32 XOR-distance1 shuffle"},
        "physical_ap_id_available": False,
        "timer_domain": "begin/end from each same wave, no cross-wave/AP timestamp subtraction",
        "static_compiled_program_review_required": True,
    }
    if args.self_test:
        receipt["cpu_reference"] = cpu_reference(args.output, args.cxx)
    else:
        binary = args.binary.resolve(strict=True)
        receipt["binary_sha256"] = sha256(binary)
        receipt["device_id"] = args.device_id
        if args.smoke:
            cells = list(itertools.product(PRIMITIVES, (1, 4), ((64, 1),), (1,)))
            samples = 1
        else:
            cells = list(itertools.product(PRIMITIVES, (1, 4), ((64, 1), (64, 104), (256, 104)), (127, 1023)))
            samples = 3
        summaries = []
        for index, (primitive, chains, (threads, grid), outer) in enumerate(cells):
            raw = args.output / f"{index:03d}_{primitive}_c{chains}_t{threads}_g{grid}_o{outer}.csv"
            argv = [binary, "--device-id", args.device_id, "--sample", primitive,
                    chains, threads, grid, outer, samples]
            command(argv, raw)
            summary = summarize(raw, primitive, chains, threads, grid, outer, samples, args.device_id)
            summary["raw_csv"] = raw.name
            summary["command"] = [str(v) for v in argv]
            summaries.append(summary)
        receipt["GPU_executed"] = True
        receipt["cells"] = summaries
        receipt["mode"] = "smoke" if args.smoke else "matrix"
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"receipt": str(args.output / "receipt.json"), "GPU_executed": receipt["GPU_executed"]}))


if __name__ == "__main__":
    main()
