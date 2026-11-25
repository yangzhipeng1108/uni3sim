#!/usr/bin/env python3
"""Capture one position-balanced A800 non-recurrent SiLU SFU run."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time


HARDWARE_ID = "nvidia_a800"
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = Path(__file__).with_name("silu_nonrecurrent_envelope.cu")
RUNNER_PATH = Path(__file__)
REPORT_PATH = Path(__file__).with_name("silu_nonrecurrent_envelope_report.py")
GATES_PATH = Path(__file__).with_name("silu_nonrecurrent_envelope_gates.json")
OUTPUT_ROOT = REPOSITORY_ROOT / "profiler_output/microbench/nvidia_a800"
EXPECTED_DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
EXPECTED_COMPUTE_CAPABILITY = "8.0"
EXPECTED_SM_COUNT = 108
EXPECTED_THREADS = 1024
MAXIMUM_IDLE_DRIVER_MEMORY_MIB = 8
TRIALS = 8
REQUIRED_INDEPENDENT_RUNS = 3
WIDTHS = (6144, 9728, 12288, 14336, 25600, 28672, 53248)
GRIDS = (108, 216, 256, 1024)
VARIANTS = ("arithmetic_control", "source_silu")
CELLS = tuple(
    (variant, grid, width)
    for width in WIDTHS for grid in GRIDS for variant in VARIANTS)
SCHEDULE_NAME = "silu_nonrecurrent_cyclic_v1"
PRODUCTION_SYMBOL = (
    "_ZN4vllm18act_and_mul_kernelIN3c108BFloat16EXadL_ZNS_11silu_kernel"
    "IS2_EET_RKS4_EELb0EEEvPS4_PS5_i")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
FUNCTION_MARKER = re.compile(r"^\s*Function\s*:\s*(\S+)\s*$", re.MULTILINE)
SASS_INSTRUCTION = re.compile(
    r"/\*([0-9a-fA-F]+)\*/\s+(?:@[!A-Z0-9]+\s+)?"
    r"([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;")
REGISTER = re.compile(r"(?<!U)(R(?:Z|[0-9]+))(?:\.reuse)?")
EXPECTED_RESOURCES = {"VariantE0": 14, "VariantE1": 21}
EXPECTED_STATIC_GATES = {
    "probe_and_production_source_math_spine_identical": True,
    "source_silu_mufu_ex2_per_loop": 1,
    "source_silu_fast_path_mufu_rcp_per_loop": 1,
    "arithmetic_control_mufu_ex2_per_loop": 0,
    "arithmetic_control_fast_path_mufu_rcp_per_loop": 0,
    "source_silu_result_cannot_feed_next_loop_input": True,
    "local_memory_instructions": 0,
    "resident_cta_limit_per_sm": 2,
}
EXPECTED_CAPTURE_GATES = {
    "position_balanced_cyclic_schedule": True,
    "all_cells_present_in_every_run": True,
    "within_run_interquartile_relative_spread_max": 0.2,
    "cross_run_median_relative_spread_max": 0.2,
    "held_out_segmented_duration_ape_max": 0.1,
    "source_silu_slower_than_arithmetic_control_for_every_cell": True,
}
EXPECTED_EXIT_CONDITIONS = [
    "any static success gate fails before a native launch",
    "the selected GPU is not an idle 108-SM NVIDIA A800-SXM4-80GB",
    "any capture schema, position balance, unit, or artifact identity differs",
    "any within-run or cross-run stability gate fails",
    "any held-out segmented source-math duration error exceeds 10 percent",
    "a production or evaluation duration is requested or consumed",
]
TERMINATION_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_ACTIVE_PROCESS = None
_DEFERRED_TERMINATION_SIGNAL = None


def cell_index_for(trial, order_position, run_rotation):
    return (order_position + 7 * trial + 19 * run_rotation) % len(CELLS)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command, *, cwd):
    command = [str(item) for item in command]
    result = subprocess.run(
        command, cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {shlex.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result


def _terminate_process_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _handle_termination(signum, _frame):
    if _ACTIVE_PROCESS is not None:
        _terminate_process_group(_ACTIVE_PROCESS)
    raise SystemExit(128 + signum)


def _install_signal_handlers():
    for signum in TERMINATION_SIGNALS:
        signal.signal(signum, _handle_termination)


def _defer_termination(signum, _frame):
    global _DEFERRED_TERMINATION_SIGNAL
    if _DEFERRED_TERMINATION_SIGNAL is None:
        _DEFERRED_TERMINATION_SIGNAL = signum


def _start_native_process(command, environment):
    global _ACTIVE_PROCESS, _DEFERRED_TERMINATION_SIGNAL
    previous_handlers = tuple(
        (signum, signal.signal(signum, _defer_termination))
        for signum in TERMINATION_SIGNALS)
    try:
        process = subprocess.Popen(
            command, cwd=REPOSITORY_ROOT, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True)
        _ACTIVE_PROCESS = process
    finally:
        for signum, previous_handler in previous_handlers:
            signal.signal(signum, previous_handler)
    if _DEFERRED_TERMINATION_SIGNAL is not None:
        signum = _DEFERRED_TERMINATION_SIGNAL
        _DEFERRED_TERMINATION_SIGNAL = None
        _handle_termination(signum, None)
    return process


def _validated_output(path):
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"output must be a child of {root}")
    if resolved.exists():
        raise FileExistsError(f"output already exists: {resolved}")
    return resolved


def _device_row(nvidia_smi, physical_device, *, require_idle):
    result = _run([
        nvidia_smi, f"--id={physical_device}",
        "--query-gpu=index,name,uuid,memory.used,utilization.gpu,"
        "compute_cap,clocks.current.sm,clocks.current.memory,driver_version",
        "--format=csv,noheader,nounits",
    ], cwd=REPOSITORY_ROOT)
    rows = tuple(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 9:
        raise ValueError("nvidia-smi returned an unexpected device row")
    values = tuple(value.strip() for value in rows[0])
    row = {
        "index": int(values[0]), "name": values[1], "uuid": values[2],
        "memory_used_mib": int(values[3]),
        "utilization_gpu_pct": int(values[4]),
        "compute_capability": values[5], "sm_clock_mhz": int(values[6]),
        "memory_clock_mhz": int(values[7]), "driver_version": values[8],
    }
    if (row["index"] != physical_device
            or row["name"] != EXPECTED_DEVICE_NAME
            or row["compute_capability"] != EXPECTED_COMPUTE_CAPABILITY):
        raise RuntimeError(f"unexpected target GPU: {row}")
    applications = _run([
        nvidia_smi, "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits",
    ], cwd=REPOSITORY_ROOT).stdout
    for line in applications.splitlines():
        fields = next(csv.reader([line], skipinitialspace=True))
        if fields and fields[0].strip() == row["uuid"]:
            raise RuntimeError(f"target GPU has a compute process: {line}")
    if require_idle and (
            row["memory_used_mib"] > MAXIMUM_IDLE_DRIVER_MEMORY_MIB
            or row["utilization_gpu_pct"] != 0):
        raise RuntimeError(f"target GPU is not idle: {row}")
    return row


def _wait_for_idle_device(nvidia_smi, physical_device):
    last_error = None
    for _ in range(20):
        try:
            return _device_row(nvidia_smi, physical_device, require_idle=True)
        except RuntimeError as error:
            last_error = error
            time.sleep(0.5)
    raise RuntimeError("target GPU did not become idle within 10 seconds") from last_error


def _function_sections(sass):
    matches = tuple(FUNCTION_MARKER.finditer(sass))
    sections = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(sass)
        sections[match.group(1)] = sass[match.start():end]
    return sections


def _instructions(section):
    return tuple(
        (int(match.group(1), 16), match.group(2), match.group(3).strip())
        for match in SASS_INSTRUCTION.finditer(section))


def _registers(operands):
    return tuple(match.group(1) for match in REGISTER.finditer(operands))


def _unique_instruction(instructions, opcode, token, *, before=None, after=None):
    matches = []
    for index, instruction in enumerate(instructions):
        if before is not None and index >= before:
            continue
        if after is not None and index <= after:
            continue
        if instruction[1] == opcode and token in instruction[2]:
            matches.append((index, instruction))
    if len(matches) != 1:
        raise ValueError(
            f"expected one {opcode} containing {token!r}, found {len(matches)}")
    return matches[0]


def _source_math_spine(section):
    instructions = _instructions(section)
    ex2 = tuple(
        (index, instruction) for index, instruction in enumerate(instructions)
        if instruction[1] == "MUFU.EX2")
    if len(ex2) != 1:
        raise ValueError(f"source math must contain one MUFU.EX2, found {len(ex2)}")
    ex2_index, ex2_instruction = ex2[0]
    sat_index, sat = _unique_instruction(
        instructions, "FFMA.SAT", "0.5", before=ex2_index)
    rm_index, rm = _unique_instruction(
        instructions, "FFMA.RM", "12582913", before=ex2_index)
    reduce_index, reduce_add = _unique_instruction(
        instructions, "FADD", "-12583039", before=ex2_index)
    hi_index, hi = _unique_instruction(
        instructions, "FFMA", "1.4426950216293334961", before=ex2_index)
    lo_index, lo = _unique_instruction(
        instructions, "FFMA", "1.925963033500011079e-08", before=ex2_index)
    scale_index, scale = _unique_instruction(
        instructions, "IMAD.SHL.U32", "0x800000")
    if not (sat_index < rm_index < reduce_index < hi_index < lo_index < ex2_index):
        raise ValueError("source exponential range-reduction order changed")
    if not (reduce_index < scale_index < ex2_index or ex2_index < scale_index):
        raise ValueError("source exponential scale reconstruction changed")

    ex2_registers = _registers(ex2_instruction[2])
    sat_registers = _registers(sat[2])
    rm_registers = _registers(rm[2])
    reduce_registers = _registers(reduce_add[2])
    hi_registers = _registers(hi[2])
    lo_registers = _registers(lo[2])
    scale_registers = _registers(scale[2])
    if (len(ex2_registers) != 2 or len(sat_registers) < 2
            or rm_registers[:2] != (sat_registers[0], sat_registers[0])
            or reduce_registers[:2] != (ex2_registers[1], sat_registers[0])
            or hi_registers[0] != ex2_registers[1]
            or ex2_registers[1] not in hi_registers[1:]
            or hi_registers[0] != lo_registers[0]
            or ex2_registers[1] not in lo_registers[1:]
            or scale_registers[:2] != (sat_registers[0], sat_registers[0])):
        raise ValueError("source exponential register dependencies changed")

    after_ex2 = instructions[ex2_index + 1:]
    denominator_offset = next((
        index for index, instruction in enumerate(after_ex2)
        if instruction[1] == "FFMA"
        and ex2_registers[0] in _registers(instruction[2])[1:]), None)
    if denominator_offset is None:
        raise ValueError("source math denominator FFMA is missing")
    denominator_index = ex2_index + 1 + denominator_offset
    denominator = instructions[denominator_index]
    denominator_registers = _registers(denominator[2])
    rcp_index = denominator_index + 1
    rcp = instructions[rcp_index]
    if (rcp[1] != "MUFU.RCP"
            or _registers(rcp[2])[1] != denominator_registers[0]):
        raise ValueError("source fast-path denominator MUFU.RCP changed")
    reciprocal_registers = _registers(rcp[2])
    following = instructions[rcp_index + 1:rcp_index + 8]
    if tuple(instruction[1] for instruction in following[:6]) != (
            "FCHK", "FFMA", "FFMA", "FFMA", "FFMA", "FFMA"):
        raise ValueError("source reciprocal refinement opcode sequence changed")
    fchk_registers = _registers(following[0][2])
    refinement = tuple(_registers(instruction[2]) for instruction in following[1:6])
    denominator_register = denominator_registers[0]
    reciprocal_register = reciprocal_registers[0]
    correction_register = refinement[0][0]
    refined_reciprocal_register = refinement[1][0]
    quotient_register = refinement[2][0]
    remainder_register = refinement[3][0]
    result_register = refinement[4][0]
    if (denominator_register not in fchk_registers
            or refinement[0][1:3] != (denominator_register, reciprocal_register)
            or refinement[1][1:] != (
                reciprocal_register, correction_register, reciprocal_register)
            or refined_reciprocal_register not in refinement[2][1:]
            or refinement[3][1] != denominator_register
            or refinement[3][2] != quotient_register
            or refinement[4] != (
                result_register, refined_reciprocal_register,
                remainder_register, quotient_register)):
        raise ValueError("source reciprocal register dependencies changed")
    return {
        "role_opcodes": [
            "FFMA.SAT", "FFMA.RM", "FADD", "FFMA", "FFMA",
            "MUFU.EX2", "IMAD.SHL.U32", "FFMA", "MUFU.RCP", "FCHK",
            "FFMA", "FFMA", "FFMA", "FFMA", "FFMA"],
        "dependency_identity": (
            "bf16_input->exp_range_reduce->MUFU.EX2->denominator->"
            "MUFU.RCP->two_step_division_refinement->silu_result"),
        "fast_path_mufu_ex2_count": 1,
        "fast_path_mufu_rcp_count": 1,
    }


def _probe_sass_contract(sass):
    sections = _function_sections(sass)
    variants = {}
    for token, name in (("VariantE0", "arithmetic_control"),
                        ("VariantE1", "source_silu")):
        matches = tuple(
            section for symbol, section in sections.items()
            if "silu_nonrecurrent_envelope" in symbol and token in symbol)
        if len(matches) != 1:
            raise ValueError(f"expected one probe specialization for {token}")
        variants[name] = matches[0]
    if "MUFU.EX2" in variants["arithmetic_control"]:
        raise ValueError("arithmetic control unexpectedly contains MUFU.EX2")
    for section in variants.values():
        if "LDL" in section or "STL" in section:
            raise ValueError("probe contains local-memory instructions")
    source_contract = _source_math_spine(variants["source_silu"])
    source_instructions = _instructions(variants["source_silu"])
    ex2_index = next(
        index for index, instruction in enumerate(source_instructions)
        if instruction[1] == "MUFU.EX2")
    result_ffma = source_instructions[ex2_index + 8]
    result_register = _registers(result_ffma[2])[0]
    sat = next(
        instruction for instruction in source_instructions[:ex2_index]
        if instruction[1] == "FFMA.SAT" and "0.5" in instruction[2])
    input_register = _registers(sat[2])[1]
    accumulations = tuple(
        (index, instruction)
        for index, instruction in enumerate(source_instructions)
        if index > ex2_index + 7
        if instruction[1] == "FADD"
        and result_register in _registers(instruction[2])[1:])
    if len(accumulations) != 1:
        raise ValueError("source SiLU result must feed exactly one accumulator")
    accumulation_index, accumulation = accumulations[0]
    accumulator_registers = _registers(accumulation[2])
    if (len(accumulator_registers) != 3
            or accumulator_registers[0] != accumulator_registers[1]
            or accumulator_registers[0] == result_register
            or input_register in (result_register, accumulator_registers[0])):
        raise ValueError("source result/accumulator separation changed")
    back_edges = tuple(
        instruction for instruction in source_instructions[accumulation_index + 1:]
        if instruction[1] == "BRA"
        and instruction[2].startswith("0x")
        and int(instruction[2], 16) < instruction[0])
    if len(back_edges) != 1 or int(back_edges[0][2], 16) >= sat[0]:
        raise ValueError("source loop back-edge no longer rebuilds its input")
    return {
        "variants": list(VARIANTS),
        "source_math_spine": source_contract,
        "source_result_feeds_accumulator_only": True,
        "source_result_cannot_feed_next_loop_input": True,
        "arithmetic_control_mufu_ex2_count": 0,
        "local_memory_instruction_count": 0,
    }


def _probe_resources(text):
    resources = {}
    current = None
    for line in text.splitlines():
        match = re.search(r"Function properties for\s+'?(\S+?)'?$", line)
        if match is not None:
            current = match.group(1)
        allocation = re.search(
            r"([0-9]+) bytes stack frame, ([0-9]+) bytes spill stores, "
            r"([0-9]+) bytes spill loads", line)
        if (current is not None
                and any(token in current for token in EXPECTED_RESOURCES)
                and allocation is not None
                and tuple(int(value) for value in allocation.groups()) != (0, 0, 0)):
            raise ValueError("probe resource report contains stack or spills")
        registers = re.search(r"Used ([0-9]+) registers", line)
        if current is None or registers is None:
            continue
        tokens = tuple(token for token in EXPECTED_RESOURCES if token in current)
        if len(tokens) == 1:
            resources[tokens[0]] = int(registers.group(1))
    if resources != EXPECTED_RESOURCES:
        raise ValueError(f"probe resource contract changed: {resources}")
    return resources


def _extract_production_artifact(production_binary, cuobjdump, build_dir):
    with tempfile.TemporaryDirectory(prefix="silu_nonrecurrent_prod_") as temporary:
        temporary_path = Path(temporary)
        _run([cuobjdump, "--extract-elf", "all", production_binary], cwd=temporary_path)
        matches = tuple(
            path for path in temporary_path.glob("*.sm_80.cubin")
            if PRODUCTION_SYMBOL.encode("ascii") in path.read_bytes())
        if len(matches) != 1:
            raise ValueError("expected one SM80 cubin containing production SiLU")
        cubin = build_dir / "production_silu_sm80.cubin"
        shutil.copy2(matches[0], cubin)
    sass = _run([cuobjdump, "--dump-sass", cubin], cwd=build_dir).stdout
    sections = _function_sections(sass)
    if PRODUCTION_SYMBOL not in sections:
        raise ValueError("production SiLU function is absent from extracted cubin")
    function_sass = sections[PRODUCTION_SYMBOL]
    sass_path = build_dir / "production_silu_bf16.sass"
    sass_path.write_text(function_sass, encoding="utf-8")
    return cubin, sass_path, _source_math_spine(function_sass)


def _validate_gates():
    gates = json.loads(GATES_PATH.read_text(encoding="utf-8"))
    if (set(gates) != {
            "schema", "hardware_id", "probe_variants",
            "intermediate_elements_per_cta", "grid_ctas", "threads_per_cta",
            "trials_per_cell_per_run", "independent_runs",
            "calibration_rotations", "held_out_rotation",
            "segmented_service_model", "cycle_unit",
            "work_unit", "service_rate_unit", "static_success_gates",
            "capture_success_gates", "exit_conditions",
            "production_or_evaluation_duration_consumed",
            "production_parameter_publication_allowed"}
            or gates["schema"]
            != "nvidia_a800_silu_nonrecurrent_envelope_gates_v2"
            or gates["hardware_id"] != HARDWARE_ID
            or tuple(gates["probe_variants"]) != VARIANTS
            or tuple(gates["intermediate_elements_per_cta"]) != WIDTHS
            or tuple(gates["grid_ctas"]) != GRIDS
            or gates["threads_per_cta"] != EXPECTED_THREADS
            or gates["trials_per_cell_per_run"] != TRIALS
            or gates["independent_runs"] != REQUIRED_INDEPENDENT_RUNS
            or gates["calibration_rotations"] != [0, 1]
            or gates["held_out_rotation"] != 2
            or gates["segmented_service_model"] != {
                "resident_ctas_per_sm": 2,
                "calibration_grid_ctas": [108, 216, 1024],
                "validation_grid_ctas": [256],
                "calibration_endpoint_widths": [6144, 53248],
                "service_segments": ["first", "resident", "queued"],
            }
            or gates["cycle_unit"] != "sm_clock_cycles"
            or gates["work_unit"] != "scalar_silu_evaluations"
            or gates["service_rate_unit"]
            != "source_rounds_per_sm_cycle"
            or gates["static_success_gates"] != EXPECTED_STATIC_GATES
            or gates["capture_success_gates"] != EXPECTED_CAPTURE_GATES
            or gates["exit_conditions"] != EXPECTED_EXIT_CONDITIONS
            or gates["production_or_evaluation_duration_consumed"]
            or not gates["production_parameter_publication_allowed"]):
        raise ValueError("frozen pre-capture gates changed")
    return gates


def main():
    global _ACTIVE_PROCESS
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rotation", required=True, type=int)
    parser.add_argument("--owner", required=True)
    parser.add_argument(
        "--nvcc", type=Path, default=Path("/usr/local/cuda-12.2/bin/nvcc"))
    parser.add_argument(
        "--cuobjdump", type=Path,
        default=Path(shutil.which("cuobjdump") or "cuobjdump"))
    parser.add_argument(
        "--nvidia-smi", type=Path, default=Path("/usr/bin/nvidia-smi"))
    parser.add_argument(
        "--production-binary", type=Path,
        default=(Path(os.environ["MIDLAYER_A800_VLLM_BINARY"])
                 if "MIDLAYER_A800_VLLM_BINARY" in os.environ else None))
    args = parser.parse_args()
    if args.production_binary is None:
        parser.error("--production-binary or MIDLAYER_A800_VLLM_BINARY is required")
    if args.device < 0:
        parser.error("--device must be non-negative")
    if args.rotation not in range(REQUIRED_INDEPENDENT_RUNS):
        parser.error("--rotation must be 0, 1, or 2")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    if RUN_ID_PATTERN.fullmatch(args.owner) is None:
        parser.error("--owner must match [A-Za-z0-9._-]{1,96}")
    paths = (
        args.nvcc, args.cuobjdump, args.nvidia_smi,
        args.production_binary, SOURCE_PATH, REPORT_PATH, GATES_PATH)
    if any(not path.resolve().is_file() for path in paths):
        parser.error("all tool, source, report, gate, and production paths must exist")
    gates = _validate_gates()
    _install_signal_handlers()

    output = _validated_output(args.output)
    preflight = _wait_for_idle_device(args.nvidia_smi.resolve(), args.device)
    output.mkdir(parents=True)
    build_dir = output / "build"
    snapshot_dir = output / "source_snapshot"
    build_dir.mkdir()
    snapshot_dir.mkdir()
    for path in (SOURCE_PATH, RUNNER_PATH, REPORT_PATH, GATES_PATH):
        shutil.copy2(path, snapshot_dir / path.name)
    source_snapshot = snapshot_dir / SOURCE_PATH.name
    runner_snapshot = snapshot_dir / RUNNER_PATH.name
    report_snapshot = snapshot_dir / REPORT_PATH.name
    gates_snapshot = snapshot_dir / GATES_PATH.name

    baseline = _run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT).stdout.strip()
    status = _run(["git", "status", "--short"], cwd=REPOSITORY_ROOT).stdout
    binary = build_dir / "silu_nonrecurrent_envelope"
    compile_command = [
        args.nvcc.resolve(), "-O3", "-std=c++17", "-arch=sm_80", "-lineinfo",
        "-Xptxas=-v", source_snapshot.name, "-o",
        Path("../build/silu_nonrecurrent_envelope")]
    compile_result = _run(compile_command, cwd=snapshot_dir)
    (build_dir / "compile.stdout").write_text(
        compile_result.stdout, encoding="utf-8")
    (build_dir / "compile.stderr").write_text(
        compile_result.stderr, encoding="utf-8")
    resources_path = build_dir / "probe_resource_usage.txt"
    resources_path.write_text(compile_result.stderr, encoding="utf-8")
    resources = _probe_resources(compile_result.stderr)
    probe_sass = _run([args.cuobjdump.resolve(), "--dump-sass", binary], cwd=build_dir).stdout
    probe_sass_path = build_dir / "probe.sass"
    probe_sass_path.write_text(probe_sass, encoding="utf-8")
    probe_contract = _probe_sass_contract(probe_sass)
    production_cubin, production_sass_path, production_contract = (
        _extract_production_artifact(
            args.production_binary.resolve(), args.cuobjdump.resolve(), build_dir))
    if probe_contract["source_math_spine"] != production_contract:
        raise ValueError("probe and production source-math spine differ")

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.device)
    native_command = [str(binary), "0", args.run_id, str(args.rotation)]
    process = _start_native_process(native_command, environment)
    try:
        stdout, stderr = process.communicate(timeout=600)
    except subprocess.TimeoutExpired as error:
        _terminate_process_group(process)
        raise RuntimeError(f"native PID {process.pid} exceeded 600 seconds") from error
    except BaseException:
        _terminate_process_group(process)
        raise
    finally:
        _ACTIVE_PROCESS = None
    if process.returncode != 0:
        raise RuntimeError(
            f"native PID {process.pid} failed ({process.returncode}): {stderr}")
    raw_path = output / "raw.csv"
    raw_path.write_text(stdout, encoding="utf-8")
    (output / "native.stderr").write_text(stderr, encoding="utf-8")
    postflight = _wait_for_idle_device(args.nvidia_smi.resolve(), args.device)

    manifest = {
        "schema": "nvidia_a800_silu_nonrecurrent_envelope_capture_v2",
        "hardware_id": HARDWARE_ID, "owner": args.owner,
        "run_id": args.run_id, "run_rotation": args.rotation,
        "runner_pid": os.getpid(), "native_pid": process.pid,
        "baseline_commit": baseline,
        "tracked_worktree_status": status.splitlines(),
        "command": shlex.join([sys.executable, os.fspath(Path(__file__)), *sys.argv[1:]]),
        "native_command": shlex.join(native_command),
        "device_preflight": preflight, "device_postflight": postflight,
        "schedule": SCHEDULE_NAME, "trials": TRIALS,
        "cycle_unit": gates["cycle_unit"], "work_unit": gates["work_unit"],
        "service_rate_unit": gates["service_rate_unit"],
        "source_sha256": _sha256(source_snapshot),
        "runner_sha256": _sha256(runner_snapshot),
        "report_sha256": _sha256(report_snapshot),
        "gates_sha256": _sha256(gates_snapshot),
        "probe_binary_sha256": _sha256(binary),
        "probe_resource_sha256": _sha256(resources_path),
        "probe_sass_sha256": _sha256(probe_sass_path),
        "probe_resources": resources, "probe_sass_contract": probe_contract,
        "production_binary_path": str(args.production_binary.resolve()),
        "production_binary_sha256": _sha256(args.production_binary.resolve()),
        "production_cubin_sha256": _sha256(production_cubin),
        "production_sass_sha256": _sha256(production_sass_path),
        "production_sass_contract": production_contract,
        "raw_sha256": _sha256(raw_path),
        "production_or_evaluation_duration_consumed": False,
        "production_parameter_published": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
