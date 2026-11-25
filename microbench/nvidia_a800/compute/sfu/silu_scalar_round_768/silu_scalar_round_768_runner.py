#!/usr/bin/env python3
"""Freeze and capture the independent A800 768-thread scalar-SiLU probe."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[5]
SOURCE = Path(__file__).with_name("silu_scalar_round_768.cu")
RUNNER = Path(__file__)
REPORT = Path(__file__).with_name("silu_scalar_round_768_report.py")
GATES = Path(__file__).with_name("silu_scalar_round_768_gates.json")
OUTPUT_ROOT = ROOT / "profiler_output/microbench/nvidia_a800"
NVCC = Path("/usr/local/cuda-12.2/bin/nvcc")
CUOBJDUMP = Path(shutil.which("cuobjdump") or "cuobjdump")
NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
PRODUCTION_BINARY = (
    Path(os.environ["MIDLAYER_A800_VLLM_BINARY"])
    if "MIDLAYER_A800_VLLM_BINARY" in os.environ else None
)
PRODUCTION_SYMBOL = "_ZN4vllm18act_and_mul_kernelIN3c108BFloat16EXadL_ZNS_11silu_kernelIS2_EET_RKS4_EELb0EEEvPS4_PS5_i"
EXPECTED_DEVICE = "NVIDIA A800-SXM4-80GB"
EXPECTED_CC = "8.0"
EXPECTED_SM_COUNT = 108
FUNCTION = re.compile(r"^\s*Function\s*:\s*(\S+)\s*$", re.MULTILINE)
INSTRUCTION = re.compile(r"/\*([0-9a-fA-F]+)\*/\s+(?:@[!A-Z0-9]+\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;")
REGISTER = re.compile(r"(?<!U)(R(?:Z|[0-9]+))(?:\.reuse)?")
SYMBOL = re.compile(r"Function(?:\s*:\s*|\s+)(\S+?):?$")
ACTIVE = None


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command, *, cwd):
    result = subprocess.run([str(value) for value in command], cwd=cwd, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=600, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed: {result.stderr}")
    return result


def _sections(sass):
    matches = tuple(FUNCTION.finditer(sass))
    return {match.group(1): sass[match.start(): matches[index + 1].start() if index + 1 < len(matches) else len(sass)]
            for index, match in enumerate(matches)}


def _instructions(section):
    return tuple((int(match.group(1), 16), match.group(2), match.group(3).strip())
                 for match in INSTRUCTION.finditer(section))


def _registers(operands):
    return tuple(match.group(1) for match in REGISTER.finditer(operands))


def _math_spine(section):
    """The production fast-path roles, dependencies, and opcode ordering."""
    instructions = _instructions(section)
    ex2 = [(index, row) for index, row in enumerate(instructions) if row[1] == "MUFU.EX2"]
    if len(ex2) != 1:
        raise ValueError(f"expected exactly one MUFU.EX2, got {len(ex2)}")
    ex2_index, ex2_row = ex2[0]
    def unique(opcode, token, before):
        found = [(index, row) for index, row in enumerate(instructions[:before])
                 if row[1] == opcode and token in row[2]]
        if len(found) != 1:
            raise ValueError(f"expected one {opcode} {token}, got {len(found)}")
        return found[0]
    sat_index, sat = unique("FFMA.SAT", "0.5", ex2_index)
    rm_index, rm = unique("FFMA.RM", "12582913", ex2_index)
    reduce_index, reduce = unique("FADD", "-12583039", ex2_index)
    hi_index, hi = unique("FFMA", "1.4426950216293334961", ex2_index)
    lo_index, lo = unique("FFMA", "1.925963033500011079e-08", ex2_index)
    if not (sat_index < rm_index < reduce_index < hi_index < lo_index < ex2_index):
        raise ValueError("exponential reduction ordering changed")
    ex2_registers = _registers(ex2_row[2])
    sat_registers = _registers(sat[2])
    if (len(ex2_registers) != 2 or len(sat_registers) < 2
            or _registers(rm[2])[:2] != (sat_registers[0], sat_registers[0])
            or _registers(reduce[2])[:2] != (ex2_registers[1], sat_registers[0])
            or _registers(hi[2])[0] != ex2_registers[1]
            or _registers(lo[2])[0] != ex2_registers[1]):
        raise ValueError("exponential reduction register dependencies changed")
    denominator_index = next((index for index in range(ex2_index + 1, len(instructions))
                              if instructions[index][1] == "FFMA"
                              and ex2_registers[0] in _registers(instructions[index][2])[1:]), None)
    if denominator_index is None or denominator_index + 7 >= len(instructions):
        raise ValueError("SiLU denominator/refinement is missing")
    denominator = _registers(instructions[denominator_index][2])
    rcp = instructions[denominator_index + 1]
    tail = instructions[denominator_index + 2: denominator_index + 8]
    if (rcp[1] != "MUFU.RCP" or _registers(rcp[2])[1] != denominator[0]
            or tuple(row[1] for row in tail) != ("FCHK", "FFMA", "FFMA", "FFMA", "FFMA", "FFMA")):
        raise ValueError("SiLU reciprocal/refinement sequence changed")
    return {"role_opcodes": ["FFMA.SAT", "FFMA.RM", "FADD", "FFMA", "FFMA", "MUFU.EX2", "FFMA", "MUFU.RCP", "FCHK", "FFMA", "FFMA", "FFMA", "FFMA", "FFMA"],
            "dependency_identity": "bf16_input->exp_range_reduce->MUFU.EX2->denominator->MUFU.RCP->two_step_division_refinement->silu_result"}


def _probe_section(sass):
    matches = [section for name, section in _sections(sass).items() if "silu_scalar_round_768" in name]
    if len(matches) != 1:
        raise ValueError(f"expected one scalar probe specialization, got {len(matches)}")
    return matches[0]


def _probe_contract(sass):
    section = _probe_section(sass)
    if any(token in section for token in ("LDL", "STL", "ATOM")):
        raise ValueError("probe contains local, spill, or atomic instruction")
    math = _math_spine(section)
    if section.count("MUFU.EX2") != 1:
        raise ValueError("recurrent scalar round must contain one SiLU exponential spine")
    return {"math_spine": math, "local_spill_atomic_absent": True,
            "single_recurrent_silu_math_spine": True}


def _resources(text):
    current = None
    values = None
    for line in text.splitlines():
        match = SYMBOL.search(line)
        if match:
            current = match.group(1)
            continue
        if current is None or "silu_scalar_round_768" not in current:
            continue
        registers = re.search(r"REG:(\d+)", line)
        shared = re.search(r"SHARED:(\d+)", line)
        local = re.search(r"LOCAL:(\d+)", line)
        if registers and shared and local:
            values = {"registers_per_thread": int(registers.group(1)), "static_shared_bytes": int(shared.group(1)), "local_bytes": int(local.group(1))}
    expected = {"registers_per_thread": 24, "static_shared_bytes": 1024, "local_bytes": 0}
    if values != expected:
        raise ValueError(f"resource contract changed: {values}")
    return values


def _validate_gates():
    gates = json.loads(GATES.read_text())
    expected = {"schema": "nvidia_a800_silu_scalar_round_768_gates_v1", "hardware_id": "nvidia_a800", "threads_per_cta": 768, "rounds_per_cta": 64, "static_shared_bytes": 1024, "registers_per_thread": 24, "grid_ctas": [108, 216], "trials_per_run": 3, "independent_runs": 3, "maximum_relative_spread": 0.2, "capacity_range_ctas_per_sm": [1.0, 2.0], "production_or_evaluation_duration_consumed": False, "production_parameter_published": False}
    if gates != expected:
        raise ValueError("frozen gates changed")
    return gates


def _device(device, idle):
    result = _run([NVIDIA_SMI, f"--id={device}", "--query-gpu=index,name,uuid,memory.used,utilization.gpu,compute_cap", "--format=csv,noheader,nounits"], cwd=ROOT)
    fields = next(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    row = {"index": int(fields[0]), "name": fields[1], "uuid": fields[2], "memory_used_mib": int(fields[3]), "utilization_gpu_pct": int(fields[4]), "compute_capability": fields[5]}
    if row["index"] != device or row["name"] != EXPECTED_DEVICE or row["compute_capability"] != EXPECTED_CC:
        raise RuntimeError(f"unexpected GPU: {row}")
    processes = _run([NVIDIA_SMI, "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], cwd=ROOT).stdout
    if any(next(csv.reader([line]))[0].strip() == row["uuid"] for line in processes.splitlines() if line.strip()):
        raise RuntimeError("target GPU has a compute process")
    if idle and (row["memory_used_mib"] > 8 or row["utilization_gpu_pct"] != 0):
        raise RuntimeError(f"target GPU is not idle: {row}")
    return row


def _wait_idle(device):
    error = None
    for _ in range(20):
        try:
            return _device(device, True)
        except RuntimeError as current:
            error = current
            time.sleep(0.5)
    raise RuntimeError("device 7 did not return idle within ten seconds") from error


def _extract_production(build):
    with tempfile.TemporaryDirectory(prefix="scalar_silu_prod_") as temporary:
        temporary = Path(temporary)
        _run([CUOBJDUMP, "--extract-elf", "all", PRODUCTION_BINARY], cwd=temporary)
        matches = [path for path in temporary.glob("*.sm_80.cubin") if PRODUCTION_SYMBOL.encode() in path.read_bytes()]
        if len(matches) != 1:
            raise ValueError("expected one production SM80 SiLU cubin")
        cubin = build / "production_silu_sm80.cubin"
        shutil.copy2(matches[0], cubin)
    sass = _run([CUOBJDUMP, "--dump-sass", cubin], cwd=build).stdout
    sections = _sections(sass)
    if PRODUCTION_SYMBOL not in sections:
        raise ValueError("production SiLU SASS symbol is missing")
    section = sections[PRODUCTION_SYMBOL]
    path = build / "production_silu_bf16.sass"
    path.write_text(section)
    return cubin, path, _math_spine(section)


def _terminate(_signal, _frame):
    global ACTIVE
    if ACTIVE is not None and ACTIVE.poll() is None:
        os.killpg(ACTIVE.pid, signal.SIGTERM)
        ACTIVE.wait(timeout=10)
    raise SystemExit(128 + _signal)


def main():
    global ACTIVE
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    if PRODUCTION_BINARY is None:
        parser.error("MIDLAYER_A800_VLLM_BINARY is required")
    if args.device != 7 or re.fullmatch(r"[A-Za-z0-9._-]{1,64}", args.owner) is None:
        parser.error("only device 7 and a safe owner are accepted")
    if any(not path.is_file() for path in (NVCC, CUOBJDUMP, NVIDIA_SMI, PRODUCTION_BINARY, SOURCE, REPORT, GATES)):
        parser.error("required tool or static evidence is missing")
    _validate_gates()
    output = args.output.resolve()
    root = OUTPUT_ROOT.resolve()
    if output == root or root not in output.parents or output.exists():
        parser.error("output must be a new child of profiler_output/microbench/nvidia_a800")
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _terminate)
    preflight = _wait_idle(args.device)
    output.mkdir(parents=True)
    build = output / "build"
    raw = output / "raw"
    snapshot = output / "source_snapshot"
    build.mkdir(); raw.mkdir(); snapshot.mkdir()
    for path in (SOURCE, RUNNER, REPORT, GATES): shutil.copy2(path, snapshot / path.name)
    compile = _run([NVCC, "-O3", "-std=c++17", "-arch=sm_80", "-lineinfo", "--maxrregcount=24", "-Xptxas=-v", snapshot / SOURCE.name, "-o", build / "silu_scalar_round_768"], cwd=ROOT)
    (build / "compile.stdout").write_text(compile.stdout)
    (build / "resource_usage.txt").write_text(_run([CUOBJDUMP, "--dump-resource-usage", build / "silu_scalar_round_768"], cwd=ROOT).stdout)
    resources = _resources((build / "resource_usage.txt").read_text())
    sass = _run([CUOBJDUMP, "--dump-sass", build / "silu_scalar_round_768"], cwd=ROOT).stdout
    (build / "probe.sass").write_text(sass)
    probe_contract = _probe_contract(sass)
    cubin, production_sass, production_contract = _extract_production(build)
    if probe_contract["math_spine"] != production_contract:
        raise ValueError("probe and production SiLU math-spine SASS differ")
    lifecycle = []
    for rotation in range(3):
        started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        environment = os.environ.copy(); environment["CUDA_VISIBLE_DEVICES"] = "7"
        with (raw / f"run_{rotation}.csv").open("w") as stdout, (raw / f"run_{rotation}.stderr").open("w") as stderr:
            ACTIVE = subprocess.Popen([build / "silu_scalar_round_768", "0", "3", str(rotation)], cwd=ROOT, text=True, stdout=stdout, stderr=stderr, start_new_session=True, env=environment)
            code = ACTIVE.wait(timeout=600)
            pid = ACTIVE.pid
            ACTIVE = None
        if code != 0:
            raise RuntimeError(f"native rotation {rotation} failed with {code}")
        lifecycle.append({"rotation": rotation, "pid": pid, "returncode": code, "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")})
    postflight = _wait_idle(args.device)
    provenance = {"schema": "nvidia_a800_silu_scalar_round_768_capture_v1", "owner": args.owner, "physical_device": args.device, "runner_pid": os.getpid(), "baseline_commit": _run(["git", "rev-parse", "HEAD"], cwd=ROOT).stdout.strip(), "device_preflight": preflight, "device_postflight": postflight, "native_runs": lifecycle, "source_sha256": _sha256(snapshot / SOURCE.name), "runner_sha256": _sha256(snapshot / RUNNER.name), "report_sha256": _sha256(snapshot / REPORT.name), "gates_sha256": _sha256(snapshot / GATES.name), "binary_sha256": _sha256(build / "silu_scalar_round_768"), "resource_usage_sha256": _sha256(build / "resource_usage.txt"), "probe_sass_sha256": _sha256(build / "probe.sass"), "production_binary_sha256": _sha256(PRODUCTION_BINARY), "production_cubin_sha256": _sha256(cubin), "production_sass_sha256": _sha256(production_sass), "resources": resources, "probe_sass_contract": probe_contract, "production_sass_contract": production_contract, "production_or_evaluation_duration_consumed": False, "production_parameter_published": False}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    report = _run([sys.executable, REPORT, "--capture", output, "--output", output / "candidate.json"], cwd=ROOT)
    if report.stderr:
        raise RuntimeError(report.stderr)


if __name__ == "__main__":
    main()
