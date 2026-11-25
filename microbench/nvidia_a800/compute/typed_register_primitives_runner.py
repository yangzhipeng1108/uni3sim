#!/usr/bin/env python3
"""Compile or capture shared typed register streams on explicitly owned A800 7.

Prepare/cpu-only never query a GPU. Capture requires separate device lease and
uses CUDA_VISIBLE_DEVICES=7, with the shared binary's internal ordinal zero.
No model/operator durations, HWSpec updates, or automatic rate binding.
"""
import argparse
import collections
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "microbench/nvidia_h200/compute/typed_register_primitives.cu"
PRIMITIVES = ("f32_gt", "f32_eq", "f32_nan", "s64_lt", "u64_gt", "shuffle32",
              "prmt_rotate16", "prmt_bf16_expand", "shuffle_down32_1_2_4_8_16")
FIELDS = ("sample", "primitive", "chains", "threads", "grid", "outer", "warp_index",
          "smid", "smid_space", "begin_cycle", "end_cycle", "warp_bundles", "checksum", "validation_status")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(argv, output, env=None):
    result = subprocess.run([str(v) for v in argv], capture_output=True, text=True, env=env)
    output.write_text(result.stdout)
    output.with_suffix(output.suffix + ".stderr").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"exit {result.returncode}; see {output}")
    return result.stdout


def cpu_reference(output, cxx):
    with tempfile.TemporaryDirectory(prefix="midlayer-a800-typed-cpu-") as temp:
        binary = Path(temp) / "typed_cpu"
        argv = [cxx, "-O2", "-std=c++17", "-DMIDLAYER_TYPED_CPU_ONLY", "-DMIDLAYER_TYPED_A800",
                "-x", "c++", SOURCE, "-o", binary]
        command(argv, output / "cpu_build.log")
        result = json.loads(command([binary, "--self-test"], output / "cpu_reference.json"))
        if result["passed"] is not True or result["GPU_executed"] is not False:
            raise ValueError("CPU reference failed")
        return {"command": [str(v) for v in argv], "result": result}


def static_inventory(sass, resources):
    inventory = []
    for block in re.split(r"Function\s*:\s*", sass)[1:]:
        symbol = block.splitlines()[0].strip()
        match = re.search(r"probeILi([0-8])ELi(1|4)EE", symbol)
        if match is None:
            continue
        kind, chains = map(int, match.groups())
        lines = [line for line in block.splitlines() if re.search(r"/\*[0-9a-f]+\*/", line)]
        clocks = [i for i, line in enumerate(lines) if "SR_CLOCKLO" in line]
        if len(clocks) != 2:
            raise ValueError(f"{symbol}: expected two clock reads")
        timed = lines[clocks[0]+1:clocks[1]]
        opcodes = collections.Counter()
        for line in timed:
            words = line.split("*/", 1)[1].strip().split()
            opcode = words[1] if words[0].startswith("@") else words[0]
            opcodes[opcode.split(".", 1)[0]] += 1
        if any(opcodes[op] for op in ("LDG", "STG", "LDL", "STL", "LDS", "STS", "ATOM", "RED")):
            raise ValueError(f"{symbol}: timed memory/spill/helper call")
        # ptxas can encode an intra-kernel loop exit as CALL.REL.NOINC.
        # Admit only that forward control transfer within this same clock region
        # or exactly to its terminating clock read;
        # an actual helper call/return is not an independent register stream.
        timed_begin = int(re.search(r"/\*([0-9a-f]+)\*/", lines[clocks[0]]).group(1), 16)
        timed_end = int(re.search(r"/\*([0-9a-f]+)\*/", lines[clocks[1]]).group(1), 16)
        for line in timed:
            if re.search(r"\bCALL\b", line):
                branch = re.search(r"CALL\.REL\.NOINC\s+0x([0-9a-f]+)", line)
                if branch is None or not timed_begin < int(branch.group(1), 16) <= timed_end:
                    raise ValueError(f"{symbol}: helper call or out-of-region transfer")
        required = 31 * chains
        if kind <= 2 and opcodes["FSETP"] < required:
            raise ValueError(f"{symbol}: FP32 predicates lost")
        if kind in (3, 4) and opcodes["ISETP"] + opcodes["UISETP"] < 2 * required:
            raise ValueError(f"{symbol}: full-width integer comparison lost")
        if kind == 5 and opcodes["SHFL"] != required:
            raise ValueError(f"{symbol}: shuffle count changed")
        if kind in (6, 7):
            prmt = [line for line in timed if re.search(r"\bPRMT\b", line)]
            if len(prmt) != required:
                raise ValueError(f"{symbol}: PRMT was removed/replaced (SHF is not accepted)")
            selector = "0x1032" if kind == 6 else "0x5410"
            if any(selector not in line for line in prmt):
                raise ValueError(f"{symbol}: PRMT selector changed")
            if kind == 7 and any(re.search(r"PRMT\s+R\d+,\s*RZ,\s*0x5410,\s*R\d+", line) is None for line in prmt):
                raise ValueError(f"{symbol}: not exact RZ/0x5410/source PRMT")
        if kind == 8:
            down = [line for line in timed if re.search(r"\bSHFL\.DOWN\b", line)]
            if len(down) != 5 * required or opcodes["SHFL"] != 5 * required:
                raise ValueError(f"{symbol}: expected five dependent SHFL.DOWN instructions per bundle")
            for offset in (1, 2, 4, 8, 16):
                if sum(re.search(rf"SHFL\.DOWN\b.*0x{offset:x}\b", line) is not None for line in down) != required:
                    raise ValueError(f"{symbol}: SHFL.DOWN offset{offset} count changed")
            parsed = []
            for line in down:
                match_down = re.search(r"SHFL\.DOWN\s+PT,\s*(R\d+),\s*(R\d+),\s*0x([0-9a-f]+),\s*0x1f", line)
                if match_down is None:
                    raise ValueError(f"{symbol}: cannot verify SHFL.DOWN register dependency")
                parsed.append((match_down.group(1), match_down.group(2), int(match_down.group(3), 16)))
            prior_outputs = None
            for begin in range(0, len(parsed), 5 * chains):
                bundle = parsed[begin:begin + 5 * chains]
                for stage, offset in enumerate((1, 2, 4, 8, 16)):
                    instructions = bundle[stage * chains:(stage + 1) * chains]
                    if {v[2] for v in instructions} != {offset}:
                        raise ValueError(f"{symbol}: SHFL.DOWN stage order changed")
                    inputs = {v[1] for v in instructions}
                    if stage == 0:
                        if prior_outputs is not None and inputs != prior_outputs:
                            raise ValueError(f"{symbol}: SHFL.DOWN recurrence dependency broken")
                    elif inputs != outputs:
                        raise ValueError(f"{symbol}: SHFL.DOWN cross-lane stage dependency broken")
                    outputs = {v[0] for v in instructions}
                prior_outputs = outputs
        match = re.search(r"Function " + re.escape(symbol) + r":\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", resources)
        if match is None:
            raise ValueError(f"{symbol}: missing resource declaration")
        registers, stack, shared, local = map(int, match.groups())
        if stack or shared or local:
            raise ValueError(f"{symbol}: register mechanism spilled/has shared storage")
        inventory.append({"symbol": symbol, "primitive": PRIMITIVES[kind], "chains": chains,
                          "bundles_per_loop": required, "timed_sass_opcodes": dict(opcodes),
                          "timed_sass": "\n".join(timed), "registers_per_thread": registers,
                          "stack_bytes": stack, "shared_bytes": shared, "local_bytes": local,
                          "shuffle_down_dependency_verified": kind == 8})
    if {(v["primitive"], v["chains"]) for v in inventory} != set(itertools.product(PRIMITIVES, (1, 4))):
        raise ValueError("missing one of 18 compiled primitive/chain variants")
    return inventory


def prepare(args):
    binary = args.output / "typed_register_gpu"
    argv = [args.nvcc, "-O3", "-std=c++17", "-arch=sm_80", "-lineinfo", "-DMIDLAYER_TYPED_A800", SOURCE, "-o", binary]
    command(argv, args.output / "gpu_build.log")
    sass = command([args.cuobjdump, "--dump-sass", binary], args.output / "typed_register.sass")
    resources = command([args.cuobjdump, "--dump-resource-usage", binary], args.output / "resources.txt")
    receipt = base_receipt(args)
    receipt.update({"mode": "prepare-only", "compile_command": [str(v) for v in argv],
                    "nvcc_version": command([args.nvcc, "--version"], args.output / "nvcc_version.txt"),
                    "cuobjdump_version": command([args.cuobjdump, "--version"], args.output / "cuobjdump_version.txt"),
                    "binary_name": binary.name, "binary_sha256": sha256(binary),
                    "cpu_reference": cpu_reference(args.output, args.cxx),
                    "static_inventory": static_inventory(sass, resources)})
    return receipt


def base_receipt(args):
    return {"schema": "nvidia_a800_typed_register_capture_v1", "hardware_id": "nvidia_a800",
            "baseline_commit": args.baseline_commit, "source_sha256": sha256(SOURCE),
            "runner_sha256": sha256(__file__), "GPU_executed": False,
            "Forward_binding_approved": False, "production_duration_consumed": False,
            "cuda_visible_devices": "7", "internal_cuda_ordinal": 0,
            "expected_sm_count": 108, "warp_size": 32,
            "units": {"compare_select": "one full-warp compare plus select bundle, not a pure predicate",
                      "shuffle32": "one full-warp b32 XOR-distance1 shuffle",
                      "prmt_rotate16": "one full-warp PRMT selector0x1032 (period2)",
                      "prmt_bf16_expand": "one full-warp PRMT Rdst,RZ,0x5410,Rsrc; state zero after2 steps",
                      "shuffle_down32_1_2_4_8_16": "one dependent five-instruction full-warp b32 SHFL.DOWN bundle; offsets1/2/4/8/16; clamp31; 129 source-active lane edges"},
            "shuffle_down_source_active_lanes_per_bundle": 31 + 30 + 28 + 24 + 16,
            "shuffle_down_full_warp_instruction_issues_per_bundle": 5,
            "prmt_bf16_expand_data_collapse_disclosed": True,
            "clock_domain": "clock64 cycles grouped only by actual SMID within same sample/cell",
            "clock_lock_requested": False}


def summarize(path, primitive, chains, threads, grid, outer, samples):
    with path.open() as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames) != FIELDS:
            raise ValueError("unexpected register CSV schema")
        rows = list(reader)
    waves = threads * grid // 32
    if len(rows) != samples * waves:
        raise ValueError("missing or extra warp records")
    groups = {}
    seen = set()
    for row in rows:
        if row["primitive"] != primitive or row["validation_status"] != "pass":
            raise ValueError("GPU correctness or primitive identity failed")
        v = {key: int(row[key]) for key in FIELDS if key not in ("primitive", "validation_status")}
        if (v["chains"], v["threads"], v["grid"], v["outer"]) != (chains, threads, grid, outer):
            raise ValueError("capture geometry differs")
        if (v["warp_bundles"] != outer * 31 * chains or v["end_cycle"] <= v["begin_cycle"]
                or not 0 <= v["sample"] < samples or not 0 <= v["warp_index"] < waves
                or v["smid_space"] < 108 or not 0 <= v["smid"] < v["smid_space"]):
            raise ValueError("invalid clock/work/SM identity")
        identity = (v["sample"], v["warp_index"])
        if identity in seen:
            raise ValueError("duplicate warp record")
        seen.add(identity)
        key = (v["sample"], v["smid"])
        if key not in groups:
            groups[key] = []
        groups[key].append(v)
    summaries = []
    for (sample, smid), values in sorted(groups.items()):
        # All stamps in this subtraction have the same actual SMID.
        span = max(v["end_cycle"] for v in values) - min(v["begin_cycle"] for v in values)
        work = sum(v["warp_bundles"] for v in values)
        cycles = [(v["end_cycle"] - v["begin_cycle"]) / v["warp_bundles"] for v in values]
        summaries.append({"sample": sample, "smid": smid, "warp_count": len(values),
                          "same_sm_span_cycles": span, "warp_bundles": work,
                          "observed_warp_bundles_per_sm_cycle": work / span,
                          "observed_lane_bundles_per_sm_cycle": 32 * work / span,
                          "cycles_per_warp_bundle_min": min(cycles),
                          "cycles_per_warp_bundle_median": statistics.median(cycles),
                          "cycles_per_warp_bundle_max": max(cycles)})
    return summaries


def pci_identity(text):
    parts = text.strip().lower().split(":")
    if len(parts) != 3:
        raise ValueError("malformed PCI bus identity")
    return tuple(int(v, 16) for v in parts[:2]) + (parts[2],)


def capture(args):
    preparation = json.loads((args.prepared / "receipt.json").read_text())
    if (preparation["mode"] != "prepare-only" or preparation["source_sha256"] != sha256(SOURCE)
            or preparation["runner_sha256"] != sha256(__file__)
            or preparation["baseline_commit"] != args.baseline_commit):
        raise ValueError("preparation does not match frozen source/runner/baseline")
    binary = (args.prepared / preparation["binary_name"]).resolve(strict=True)
    if preparation["binary_sha256"] != sha256(binary):
        raise ValueError("prepared binary changed")
    # Recheck archived static evidence; metadata success alone is insufficient.
    static_inventory((args.prepared / "typed_register.sass").read_text(), (args.prepared / "resources.txt").read_text())
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "7"
    identity_cmd = ["nvidia-smi", "-i", "7", "--query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,clocks.current.sm,clocks.current.memory", "--format=csv,noheader"]
    identity = command(identity_cmd, args.output / "device_before.csv")
    if "A800" not in identity:
        raise ValueError("explicit physical device7 is not A800")
    identity_fields = next(csv.reader(identity.splitlines()))
    expected_pci = pci_identity(identity_fields[2])
    telemetry_file = (args.output / "clock_telemetry.csv").open("w")
    telemetry_error = (args.output / "clock_telemetry.stderr").open("w")
    telemetry_cmd = ["nvidia-smi", "-i", "7", "--query-gpu=timestamp,uuid,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,utilization.gpu", "--format=csv", "--loop-ms=200"]
    telemetry = subprocess.Popen(telemetry_cmd, stdout=telemetry_file, stderr=telemetry_error, text=True)
    cells = []
    captured_primitives = PRIMITIVES if args.primitive is None else (args.primitive,)
    try:
        geometries = ((32, 1),) if args.smoke else ((32, 1), (128, 108), (256, 432))
        outers = (1, 2) if args.smoke else (127, 511)
        samples = 1 if args.smoke else 3
        for index, (primitive, chains, (threads, grid), outer) in enumerate(itertools.product(captured_primitives, (1, 4), geometries, outers)):
            path = args.output / f"{index:03d}_{primitive}_c{chains}_t{threads}_g{grid}_o{outer}.csv"
            argv = [binary, "--sample", primitive, chains, threads, grid, outer, samples]
            command(argv, path, env=env)
            cuda_identity = list(csv.reader(path.with_suffix(path.suffix + ".stderr").read_text().splitlines()))
            if (len(cuda_identity) != 1 or len(cuda_identity[0]) != 6
                    or cuda_identity[0][0] != "a800_cuda_device_identity"
                    or pci_identity(cuda_identity[0][2]) != expected_pci
                    or tuple(map(int, cuda_identity[0][3:])) != (8, 0, 108)):
                raise ValueError("CUDA ordinal0 differs from owned physical GPU7")
            cells.append({"primitive": primitive, "chains": chains, "threads": threads, "grid": grid,
                          "outer": outer, "samples": samples, "raw_csv": path.name,
                          "command": [str(v) for v in argv],
                          "same_sm_groups": summarize(path, primitive, chains, threads, grid, outer, samples)})
        if telemetry.poll() is not None:
            raise RuntimeError("clock telemetry exited before capture finished")
    finally:
        if telemetry.poll() is None:
            telemetry.terminate()
        telemetry.wait(timeout=10)
        telemetry_file.close()
        telemetry_error.close()
    receipt = base_receipt(args)
    receipt.update({"mode": "smoke" if args.smoke else "matrix", "trial": args.trial,
                    "GPU_executed": True, "binary_sha256": sha256(binary),
                    "preparation": str(args.prepared), "device_identity_before": identity.strip(),
                    "device_identity_after": command(identity_cmd, args.output / "device_after.csv").strip(),
                    "telemetry_command": telemetry_cmd,
                    "captured_primitives": list(captured_primitives),
                    "primitive_subset_requested": args.primitive is not None,
                    "cells": cells})
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--cpu-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--matrix", action="store_true")
    parser.add_argument("--device", type=int, choices=(7,))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--trial", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--primitive", choices=PRIMITIVES,
                        help="capture only this primitive; default captures every primitive")
    parser.add_argument("--nvcc", type=Path, default=Path("/usr/local/cuda/bin/nvcc"))
    parser.add_argument("--cuobjdump", type=Path, default=Path("/usr/local/cuda/bin/cuobjdump"))
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError("output exists; never overwrite capture")
    if not args.cpu_only and args.device != 7:
        raise ValueError("explicit --device 7 is required")
    allowed = (ROOT / "profiler_output").resolve()
    if not args.cpu_only and not args.output.is_relative_to(allowed):
        raise ValueError("prepare/capture output must be under this worktree profiler_output")
    if not re.fullmatch(r"[0-9a-f]{40}", args.baseline_commit):
        raise ValueError("baseline must be exact commit")
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if head != args.baseline_commit:
        raise ValueError("baseline is not actual HEAD")
    if (args.smoke or args.matrix) and args.prepared is None:
        raise ValueError("capture requires --prepared directory")
    if args.primitive is not None and not (args.smoke or args.matrix):
        raise ValueError("--primitive applies only to smoke or matrix capture")
    args.output.mkdir(parents=True)
    if args.cpu_only:
        receipt = base_receipt(args)
        receipt.update({"mode": "cpu-only", "cpu_reference": cpu_reference(args.output, args.cxx)})
    elif args.prepare_only:
        receipt = prepare(args)
    else:
        receipt = capture(args)
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"receipt": str(args.output / "receipt.json"), "GPU_executed": receipt["GPU_executed"]}))


if __name__ == "__main__":
    main()
