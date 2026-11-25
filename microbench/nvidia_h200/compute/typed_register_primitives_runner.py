#!/usr/bin/env python3
"""Prepare or explicitly capture independent H200 register-stream mechanisms.

--prepare-only compiles SM90/PTX, dumps SASS/resources and runs a separate
CPU-only reference binary. It never queries or executes a GPU. --execute is
for a later authorized owner, with raw output under registered profiler_output.
"""

import argparse
import collections
import csv
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clock_gated_capture as capture

SOURCE = Path(__file__).with_name("typed_register_primitives.cu")
PRIMITIVES = ("f32_gt", "f32_eq", "f32_nan", "s64_lt", "u64_gt", "shuffle32")
UNROLL = 31
FIELDS = ("sample", "primitive", "chains", "threads", "grid", "outer", "warp_index", "smid", "smid_space",
          "begin_cycle", "end_cycle", "warp_bundles", "checksum", "validation_status")


def command(argv, log):
    result = subprocess.run([str(value) for value in argv], capture_output=True, text=True)
    Path(log).write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")
    return result.stdout


def static_inventory(sass):
    records = []
    for block in re.split(r"Function\s*:\s*", sass)[1:]:
        symbol = block.splitlines()[0].strip()
        match = re.search(r"probeILi([0-5])ELi(1|4)EE", symbol)
        if match is None:
            continue
        kind, chains = map(int, match.groups())
        lines = [line for line in block.splitlines() if re.search(r"/\*[0-9a-f]+\*/", line)]
        clocks = [i for i, line in enumerate(lines) if "SR_CLOCKLO" in line]
        if len(clocks) != 2:
            raise ValueError(f"{symbol}: expected exactly two SM clock reads")
        timed = lines[clocks[0] + 1:clocks[1]]
        opcodes = collections.Counter()
        for line in timed:
            instruction = line.split("*/", 1)[1].strip().split()
            opcode = instruction[1] if instruction[0].startswith("@") else instruction[0]
            opcodes[opcode.split(".", 1)[0]] += 1
        if any(opcodes[op] for op in ("LDG", "STG", "LDL", "STL", "LDS", "STS", "ATOM", "RED")):
            raise ValueError(f"{symbol}: timed region contains memory instructions")
        required = UNROLL * chains
        # This is a compile preservation check, not an instruction-to-rate map.
        if kind <= 2:
            if opcodes["FSETP"] < required:
                raise ValueError(f"{symbol}: floating predicates optimized away or require a new static interpretation")
        elif kind <= 4:
            if opcodes["ISETP"] + opcodes["UISETP"] < 2 * required:
                raise ValueError(f"{symbol}: full two-word comparison not preserved")
        elif opcodes["SHFL"] != required:
            raise ValueError(f"{symbol}: shuffle count changed")
        records.append(dict(symbol=symbol, primitive=PRIMITIVES[kind], chains=chains,
                            source_bundles_per_loop=required, timed_sass_opcodes=dict(opcodes),
                            timed_sass="\n".join(timed), measured_rate=None))
    if {(row["primitive"], row["chains"]) for row in records} != set(itertools.product(PRIMITIVES, (1, 4))):
        raise ValueError("missing compiled primitive/chain variants")
    return records


def prepare(out, nvcc, cxx, cuobjdump):
    capture.NVCC = Path(nvcc)
    build = capture.build(SOURCE, out / "typed_register_gpu")
    cpu = out / "typed_register_cpu"
    command([cxx, "-O2", "-std=c++17", "-DMIDLAYER_TYPED_CPU_ONLY", "-x", "c++", SOURCE, "-o", cpu], out / "cpu_build.log")
    cpu_result = json.loads(command([cpu, "--self-test"], out / "cpu_reference.json"))
    if cpu_result["passed"] is not True or cpu_result["GPU_executed"] is not False:
        raise ValueError("CPU reference verification failed")
    command([nvcc, "--ptx", "-O3", "-std=c++17", "-arch=compute_90", "-lineinfo", SOURCE,
             "-o", out / "typed_register.ptx"], out / "ptx_build.log")
    sass = command([cuobjdump, "--dump-sass", out / "typed_register_gpu"], out / "typed_register.sass")
    resources = command([cuobjdump, "--dump-resource-usage", out / "typed_register_gpu"], out / "resource_usage.txt")
    inventory = static_inventory(sass)
    for row in inventory:
        match = re.search(r"Function " + re.escape(row["symbol"]) + r":\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", resources)
        if match is None:
            raise ValueError(f"missing resource declaration: {row['symbol']}")
        registers, stack, shared, local = map(int, match.groups())
        if stack or shared or local:
            raise ValueError(f"register mechanism has stack/shared/local storage: {row['symbol']}")
        row["resources"] = dict(registers_per_thread=registers, stack_bytes=stack, shared_bytes=shared, local_bytes=local)
    return dict(schema="h200_typed_register_preparation_v1", hardware_id="nvidia_h200", build=build,
                runner_sha256=capture.sha256(Path(__file__)), capture_helper_sha256=capture.sha256(Path(capture.__file__)),
                cpu_binary_sha256=capture.sha256(cpu),
                cxx_version=command([cxx, "--version"], out / "cxx_version.txt").strip(),
                cuobjdump_version=command([cuobjdump, "--version"], out / "cuobjdump_version.txt").strip(),
                cpu_build_command=[str(cxx), "-O2", "-std=c++17", "-DMIDLAYER_TYPED_CPU_ONLY", "-x", "c++", str(SOURCE), "-o", str(cpu)],
                cpu_reference=cpu_result, static_inventory=inventory,
                primitive_unit="one full-warp compare+select bundle, or one full-warp b32 XOR shuffle",
                unroll=UNROLL, no_timed_global_memory=True, production_duration_consumed=False,
                GPU_executed=False, measured_rate=None, Forward_binding_approved=False,
                source_domains={"f32_gt/f32_eq": "independent finite binary-state register streams",
                                "f32_nan": "independent finite/qNaN binary-state register streams",
                                "s64_lt/u64_gt": "synthetic full-width words; runtime nonzero two-word pivot",
                                "shuffle32": "all 32 lanes active; XOR-distance1; no reduction or selection"})


def summarize_csv(path, primitive, chains, threads, grid, outer, samples):
    with Path(path).open() as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames) != FIELDS:
            raise ValueError("unexpected primitive CSV schema")
        rows = list(reader)
    if len(rows) != samples * grid * threads // 32:
        raise ValueError("missing warp records")
    groups = {}
    seen = set()
    for row in rows:
        if row["primitive"] != primitive or row["validation_status"] != "pass":
            raise ValueError("invalid primitive or failed validation")
        values = {key: int(row[key]) for key in FIELDS if key not in ("primitive", "validation_status")}
        if (values["chains"], values["threads"], values["grid"], values["outer"]) != (chains, threads, grid, outer):
            raise ValueError("record configuration mismatch")
        if (values["warp_bundles"] != outer * UNROLL * chains or values["begin_cycle"] >= values["end_cycle"]
                or not 0 <= values["sample"] < samples or values["smid_space"] < 132
                or not 0 <= values["smid"] < values["smid_space"]
                or not 0 <= values["warp_index"] < grid * threads // 32):
            raise ValueError("invalid work/timer/warp identity")
        identity = (values["sample"], values["warp_index"])
        if identity in seen:
            raise ValueError("duplicate warp record")
        seen.add(identity)
        key = (values["sample"], values["smid"])
        if key not in groups:
            groups[key] = []
        groups[key].append(values)
    summaries = []
    for (sample, smid), values in sorted(groups.items()):
        # Never compare absolute clock64 stamps across SMs.
        span = max(v["end_cycle"] for v in values) - min(v["begin_cycle"] for v in values)
        work = sum(v["warp_bundles"] for v in values)
        summaries.append(dict(sample=sample, smid=smid, span_sm_cycles=span, warp_bundles=work,
                              observed_warp_bundles_per_sm_cycle=work / span,
                              cycles_per_warp_bundle=[(v["end_cycle"] - v["begin_cycle"]) / v["warp_bundles"] for v in values]))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute", action="store_true", help="requires separate hardware authorization and owned device")
    parser.add_argument("--smoke-only", action="store_true", help="execute only 12 small correctness cells before a separate full capture")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--cuobjdump", type=Path, required=True)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--device", type=int)
    parser.add_argument("--baseline-commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.baseline_commit):
        raise ValueError("baseline must be the exact source commit")
    root = SOURCE.resolve().parents[3]
    actual_commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    if args.baseline_commit != actual_commit:
        raise ValueError("baseline does not match the actual checkout")
    dirty = subprocess.check_output(["git", "-C", root, "status", "--porcelain", "--untracked-files=all"], text=True).splitlines()
    if args.prepare_only and args.device is not None:
        raise ValueError("offline preparation does not accept a device")
    if args.prepare_only and args.smoke_only:
        raise ValueError("smoke-only is an execution mode")
    if args.execute and (args.device is None or "profiler_output" not in args.output.resolve().parts):
        raise ValueError("execution requires an explicit owned device and registered profiler_output path")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    preparation = prepare(out, args.nvcc, args.cxx, args.cuobjdump)
    preparation["baseline_commit"] = args.baseline_commit
    preparation["checkout_dirty_entries"] = dirty
    (out / "preparation.json").write_text(json.dumps(preparation, indent=2) + "\n")
    if args.prepare_only:
        print(out / "preparation.json")
        return
    identity = capture.device_identity(args.device)
    warmup = capture.build_warmup(out / "warmup")
    cells = list(itertools.product(PRIMITIVES, (1, 4), (32, 128, 256), (1, 132, 528), (127, 511)))
    trials, samples = 3, 3
    if args.smoke_only:
        cells = list(itertools.product(PRIMITIVES, (1, 4), (32,), (1,), (127,)))
        trials, samples = 1, 1
    verdicts, observations = [], []
    for trial in range(trials):
        cells = cells[7:] + cells[:7]  # Fixed rotation; unrelated to target durations.
        commands = [([out / "typed_register_gpu", "--sample", *cell, samples], out / f"trial{trial}_cell{index}.csv")
                    for index, cell in enumerate(cells)]
        verdict = capture.run_gated(f"trial{trial}", commands, out, args.device, warmup, timeout=120)
        verdicts.append(verdict)
        for cell, path in zip(cells, verdict["stdout_paths"], strict=True):
            observations.append(dict(trial=trial, primitive=cell[0], chains=cell[1], threads=cell[2], grid=cell[3], outer=cell[4],
                                     raw_path=path, per_sm=summarize_csv(path, *cell, samples)))
    manifest = dict(schema="h200_typed_register_observation_v1", baseline_commit=args.baseline_commit,
                    device=identity, clock_gate=capture.clock_summary(verdicts), clock_verdicts=verdicts,
                    observations=observations, units="warp bundles per SM cycle; source bundle is not one SASS instruction",
                    preparation="preparation.json", smoke_only=args.smoke_only,
                    Forward_binding_approved=False, production_duration_consumed=False)
    (out / "observations.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out / "observations.json")


if __name__ == "__main__":
    main()
