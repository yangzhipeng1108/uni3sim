#!/usr/bin/env python3
"""Prepare/capture the SM80 finite numeric comparator + selected pair bundle.

Reuse the A800 clock/CSV helpers. Never benchmarks ArgMax, loads, reductions,
shuffle or a production workload; no automatic Forward binding.
"""
import argparse
import collections
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import typed_register_primitives_runner as shared

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).with_name("numeric_pair_bundle.cu")


def cpu_reference(output):
    with tempfile.TemporaryDirectory(prefix="midlayer-a800-numeric-cpu-") as temporary:
        binary = Path(temporary) / "cpu_reference"
        argv = ["g++", "-O2", "-std=c++17", "-DMIDLAYER_NUMERIC_CPU_ONLY", "-x", "c++", SOURCE, "-o", binary]
        shared.command(argv, output / "cpu_build.log")
        result = json.loads(shared.command([binary, "--self-test"], output / "cpu_reference.json"))
        if result["passed"] is not True or result["GPU_executed"] is not False or result["idempotent_after_one"] is not True:
            raise ValueError("CPU reference failed")
        return result


def static_inventory(sass, resources):
    records = []
    for block in re.split(r"Function\s*:\s*", sass)[1:]:
        symbol = block.splitlines()[0].strip()
        match = re.search(r"probe_pairILi([01])ELi(1|4)EE", symbol)
        if match is None:
            continue
        scope, chains = map(int, match.groups())
        lines = [line for line in block.splitlines() if re.search(r"/\*[0-9a-f]+\*/", line)]
        clocks = [i for i, line in enumerate(lines) if "SR_CLOCKLO" in line]
        if len(clocks) != 2:
            raise ValueError("expected exactly two clock reads")
        timed = lines[clocks[0]+1:clocks[1]]
        counts = collections.Counter()
        for line in timed:
            words = line.split("*/", 1)[1].strip().split()
            opcode = words[1] if words[0].startswith("@") else words[0]
            counts[opcode] += 1
        forbidden = ("LDG", "STG", "LDL", "STL", "LDS", "STS", "ATOM", "RED", "SHFL")
        if any(count for op, count in counts.items() if op.split(".", 1)[0] in forbidden):
            raise ValueError("timed memory, spill, reduction or shuffle")
        work = 31 * chains
        required = {"FSETP.NEU.AND": work, "FSETP.GT.AND": work,
                    "ISETP.GE.U32.AND": work, "ISETP.GE.AND.EX": work,
                    "ISETP.NE.U32.AND" if scope == 0 else "ISETP.EQ.U32.AND": work, "FSEL": work, "SEL": 4 * work}
        for opcode, count in required.items():
            if counts[opcode] != count:
                raise ValueError(f"{symbol}: required {opcode}={count}, observed={counts[opcode]}")
        packs = [line for line in timed if "LOP3.LUT" in line and ", 0x1, RZ, 0xc0," in line]
        if len(packs) != work:
            raise ValueError("numeric bool packing was simplified/changed")
        # ptxas inverted final EQ + p into NE + !p. Verify every resulting
        # FSEL and both word selects consume that complemented predicate.
        fsel = [line for line in timed if re.search(r"\bFSEL\b", line)]
        expected_predicate = r", !P[0-6] ;" if scope == 0 else r", P[0-6] ;"
        if any(re.search(expected_predicate, line) is None for line in fsel):
            raise ValueError("final predicate sense no longer matches")
        predicated_bool = [line for line in timed if re.search(r"@!P[0-6]\s+SEL", line)]
        word_results = [line for line in timed if re.search(r"\bSEL\b", line) and re.search(r", !P[0-6] ;", line)]
        # One additional !P SEL per bundle materializes greater-than.
        if len(predicated_bool) != work or len(word_results) != (3 * work if scope == 0 else work):
            raise ValueError("pair select or conditional equality bool mapping changed")
        if scope == 1:
            for opcode, count in {"FSETP.GTU.AND": 2 * work, "ISETP.LT.U32.AND": work,
                                  "ISETP.LT.OR.EX": work, "BSSY": work, "BSYNC": work}.items():
                if counts[opcode] != count:
                    raise ValueError(f"source classification/branch opcode changed: {opcode}")
            abs_nan = [line for line in timed if "FSETP.GTU.AND" in line]
            if any("|R" not in line or "+INF" not in line for line in abs_nan):
                raise ValueError("NaN classification no longer exact abs(value)>INF")
            if counts["BRA"] < 2 * work:
                raise ValueError("actual source branches removed")
        for line in timed:
            if re.search(r"\bCALL\b", line) and "CALL.REL.NOINC" not in line:
                raise ValueError("timed helper call")
        resource = re.search(r"Function " + re.escape(symbol) + r":\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", resources)
        if resource is None:
            raise ValueError("missing compiled resources")
        registers, stack, smem, local = map(int, resource.groups())
        if stack or smem or local:
            raise ValueError("register mechanism has stack/shared/local allocation")
        records.append({"symbol": symbol, "scope": "numeric" if scope == 0 else "classified", "chains": chains, "bundles_per_loop": work,
                        "timed_sass_opcodes": dict(counts), "registers_per_thread": registers,
                        "stack_bytes": stack, "shared_bytes": smem, "local_bytes": local,
                        "bool_pack_sites": len(packs), "timed_sass": "\n".join(timed),
                        "source_mapping": ("NE test + inverted result predicates is EQ test + positive result predicates; full32+64 pair result retained"
                                           if scope == 0 else "Exact absNaN classification, retained numeric/NaN branches and reconvergence, fullFP32/two-wordINT64 result"),
                        "fixed_timer_loop_and_register_epilogue_included": True})
    if {(row["scope"], row["chains"]) for row in records} != {(scope, chains) for scope in ("numeric", "classified") for chains in (1, 4)}:
        raise ValueError("missing compiled scope/chain variant")
    return records


def base_receipt(args):
    return {"schema": "a800_numeric_pair_bundle_v1", "hardware_id": "nvidia_a800",
            "baseline_commit": args.baseline_commit, "source_sha256": shared.sha256(SOURCE),
            "runner_sha256": shared.sha256(__file__), "helper_sha256": shared.sha256(shared.__file__),
            "GPU_executed": False, "Forward_binding_approved": False,
            "unit": "one wave32 selectedFP32/INT64 comparator bundle, distinguished by scope",
            "numeric_scope_computational_instructions_per_bundle": 11,
            "classified_capture_inputs": "synthetic finite; not a claim about production domain",
            "classified_scope": "includes absNaN classification, actual numeric/NaN branch and BSSY/BSYNC; finite GPUinput path only",
            "idempotent_after_one_disclosed": True,
            "numeric_scope_excluded": ["NaNclassification", "BRA/BSYNC source branch costs"],
            "both_scopes_excluded": ["load/store", "reduction", "shuffle", "complete argmax", "production durations"],
            "pure_predicate_service_claimed": False,
            "actual_source_merge_count_bound": False,
            "clock_lock_requested": False}


def prepare(args):
    binary = args.output / "numeric_pair_gpu"
    argv = ["/usr/local/cuda/bin/nvcc", "-O3", "-std=c++17", "-arch=sm_80", "-lineinfo", SOURCE, "-o", binary]
    shared.command(argv, args.output / "gpu_build.log")
    sass = shared.command(["/usr/local/cuda/bin/cuobjdump", "--dump-sass", binary], args.output / "program.sass")
    resources = shared.command(["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", binary], args.output / "resources.txt")
    receipt = base_receipt(args)
    receipt.update({"mode": "prepare-only", "binary_name": binary.name, "binary_sha256": shared.sha256(binary),
                    "compile_command": [str(v) for v in argv], "cpu_reference": cpu_reference(args.output),
                    "nvcc_version": shared.command(["/usr/local/cuda/bin/nvcc", "--version"], args.output / "nvcc_version.txt"),
                    "static_inventory": static_inventory(sass, resources)})
    return receipt


def capture(args):
    prepared = json.loads((args.prepared / "receipt.json").read_text())
    for key in ("baseline_commit", "source_sha256", "runner_sha256", "helper_sha256"):
        if prepared[key] != base_receipt(args)[key]:
            raise ValueError("frozen preparation changed")
    binary = (args.prepared / prepared["binary_name"]).resolve(strict=True)
    if shared.sha256(binary) != prepared["binary_sha256"]:
        raise ValueError("binary changed")
    static_inventory((args.prepared / "program.sass").read_text(), (args.prepared / "resources.txt").read_text())
    identity_command = ["nvidia-smi", "-i", "7", "--query-gpu=name,uuid,pci.bus_id,clocks.current.sm,clocks.current.memory", "--format=csv,noheader"]
    identity = shared.command(identity_command, args.output / "identity_before.csv")
    fields = next(csv.reader(identity.splitlines()))
    expected_pci = shared.pci_identity(fields[2])
    if "A800" not in fields[0]:
        raise ValueError("physicalGPU7 is not A800")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "7"
    geometries = ((32, 1),) if args.smoke else ((32, 1), (128, 108), (256, 432))
    outers = (1, 2) if args.smoke else (127, 511)
    samples = 1 if args.smoke else 3
    cells = []
    scopes = ((0, "numeric"), (1, "classified")) if args.scope == "both" else ((0, "numeric"),) if args.scope == "numeric" else ((1, "classified"),)
    for scope_id, scope in scopes:
      for chains in (1, 4):
        for threads, grid in geometries:
            for outer in outers:
                path = args.output / f"{scope}_c{chains}_t{threads}_g{grid}_o{outer}.csv"
                argv = [binary, "--sample", scope_id, chains, threads, grid, outer, samples]
                shared.command(argv, path, env)
                cuda_identity = list(csv.reader(path.with_suffix(path.suffix + ".stderr").read_text().splitlines()))
                if len(cuda_identity) != 1 or shared.pci_identity(cuda_identity[0][2]) != expected_pci:
                    raise ValueError("CUDA ordinal differs from explicit physicalGPU7")
                cells.append({"scope": scope, "chains": chains, "threads": threads, "grid": grid, "outer": outer,
                              "samples": samples, "raw_csv": path.name,
                              "command": [str(v) for v in argv],
                              "same_sm_groups": shared.summarize(path, "argmax_numeric_pair" if scope_id == 0 else "argmax_classified_pair_finite_inputs", chains, threads, grid, outer, samples)})
    receipt = base_receipt(args)
    receipt.update({"mode": "smoke" if args.smoke else "matrix", "GPU_executed": True,
                    "binary_sha256": shared.sha256(binary), "trial": args.trial,
                    "identity_before": identity.strip(),
                    "identity_after": shared.command(identity_command, args.output / "identity_after.csv").strip(),
                    "cells": cells})
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--cpu-only", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--matrix", action="store_true")
    parser.add_argument("--scope", choices=("numeric", "classified", "both"), default="classified")
    parser.add_argument("--device", type=int, choices=(7,))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--trial", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError("output already exists")
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if args.baseline_commit != head or re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise ValueError("requires exact currentbaseline")
    if not args.cpu_only and (args.device != 7 or not args.output.is_relative_to(ROOT / "profiler_output")):
        raise ValueError("requires explicitdevice7 and registered profiler_output")
    if (args.smoke or args.matrix) and args.prepared is None:
        raise ValueError("capture requires frozenpreparation")
    args.output.mkdir(parents=True)
    if args.cpu_only:
        receipt = base_receipt(args)
        receipt.update({"mode": "cpu-only", "cpu_reference": cpu_reference(args.output)})
    elif args.prepare_only:
        receipt = prepare(args)
    else:
        receipt = capture(args)
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"receipt": str(args.output / "receipt.json"), "GPU_executed": receipt["GPU_executed"]}))


if __name__ == "__main__":
    main()
