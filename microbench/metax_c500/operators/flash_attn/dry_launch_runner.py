#!/usr/bin/env python3
"""Capture one MetaX FlashAttention launch without submitting its kernel."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
HARDWARE_ID = "metax_c500"
OUTPUT_OWNER = Path("profiler_output/microbench/metax_c500")
RUNNER_RELATIVE = Path(
    "microbench/metax_c500/operators/flash_attn/dry_launch_runner.py")
INTERPOSER_RELATIVE = Path(
    "microbench/metax_c500/operators/gemm/dry_launch_interposer.cpp")
SOURCE_PATHS = (RUNNER_RELATIVE, INTERPOSER_RELATIVE)
INTERPOSED_LAUNCH_APIS = frozenset({
    "mcLaunchKernel", "mcModuleLaunchKernel", "mcLaunchCooperativeKernel",
    "mcGraphLaunch",
})
REQUIRED_INTERPOSED_SYMBOLS = INTERPOSED_LAUNCH_APIS | frozenset({
    "midlayer_dry_launch_begin",
    "midlayer_dry_launch_end", "midlayer_dry_launch_error",
})


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command, *, environment=None, timeout=300, cwd=None):
    result = subprocess.run(
        command, check=False, capture_output=True, text=True,
        env=environment, timeout=timeout, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with status {result.returncode}: {command!r}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result


def _validate_source_identity(repository_root, baseline_commit):
    if COMMIT_PATTERN.fullmatch(baseline_commit) is None:
        raise ValueError("baseline commit must be 40 lowercase hex")
    for relative_path in SOURCE_PATHS:
        if not (repository_root / relative_path).is_file():
            raise FileNotFoundError(
                f"required dry-launch source is missing: {relative_path}")
    git_marker = repository_root / ".git"
    snapshot_marker = repository_root / ".source_commit"
    if git_marker.exists():
        head = _run(
            ["git", "rev-parse", "HEAD"],
            environment=None, cwd=repository_root).stdout.strip()
        if head != baseline_commit:
            raise ValueError("repository HEAD differs from baseline commit")
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            cwd=repository_root, check=True, capture_output=True, text=True)
        if status.stdout:
            raise ValueError("Git source checkout has tracked modifications")
        mode = "clean_exact_git_checkout"
        marker_path = None
    else:
        if not snapshot_marker.is_file():
            raise FileNotFoundError(
                "source snapshot requires repository-root/.source_commit")
        if snapshot_marker.read_text(encoding="utf-8") != baseline_commit + "\n":
            raise ValueError(".source_commit differs from baseline commit")
        mode = "audited_source_snapshot"
        marker_path = str(snapshot_marker.resolve())
    return {
        "mode": mode,
        "baseline_commit": baseline_commit,
        "marker_path": marker_path,
        "source_sha256": {
            str(relative_path): _sha256(repository_root / relative_path)
            for relative_path in SOURCE_PATHS
        },
    }


def _require_output_dir(output_dir, repository_root):
    owner = (repository_root / OUTPUT_OWNER).resolve()
    output_dir = output_dir.resolve()
    if output_dir.parent != owner:
        raise ValueError(f"output must be a direct child of {owner}")
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    owner.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    return output_dir


def _build_interposer(repository_root, maca_path, compiler, build_dir):
    header = maca_path / "include/mcr/mc_runtime.h"
    runtime = maca_path / "lib/libmcruntime.so"
    if not header.is_file() or not runtime.is_file():
        raise FileNotFoundError("MACA runtime installation is incomplete")
    resolved_compiler = shutil.which(compiler)
    if resolved_compiler is None:
        raise FileNotFoundError(f"C++ compiler is unavailable: {compiler}")
    interposer = build_dir / "libmidlayer_c500_dry_launch.so"
    _run([
        resolved_compiler, "-std=c++14", "-O2", "-Wall", "-Wextra",
        "-Werror", "-fPIC", "-shared", "-isystem",
        str(maca_path / "include/mcr"),
        "-isystem", str(maca_path / "include"),
        str(repository_root / INTERPOSER_RELATIVE), "-ldl", "-pthread",
        "-o", str(interposer),
    ])
    symbols = _run(
        ["nm", "-D", "--defined-only", str(interposer)]).stdout
    defined = {
        line.split()[-1] for line in symbols.splitlines() if line.split()}
    missing = REQUIRED_INTERPOSED_SYMBOLS - defined
    if missing:
        raise RuntimeError(
            f"dry-launch interposer symbols are missing: {sorted(missing)}")
    return interposer


def _parse_capture(path):
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid capture JSON at line {line_number}") from error
            if not isinstance(record, dict):
                raise TypeError("capture record must be a mapping")
            records.append(record)
    if len(records) != 2:
        raise ValueError("FlashAttention PoC requires one launch and summary")
    launch, summary = records
    if summary != {
            "record_type": "summary", "expected_launch_count": 1,
            "captured_launch_count": 1}:
        raise ValueError("dry-launch summary differs from one-launch contract")
    expected_fields = {
        "record_type", "ordinal", "api", "kernel_identity",
        "identity_source", "grid", "block",
        "dynamic_shared_memory_bytes", "static_shared_memory_bytes",
        "static_shared_memory_status",
    }
    if set(launch) != expected_fields:
        raise ValueError("dry-launch launch fields changed")
    if launch["record_type"] != "launch" or launch["ordinal"] != 1:
        raise ValueError("dry-launch launch ordinal is invalid")
    if launch["api"] not in {"mcLaunchKernel", "mcModuleLaunchKernel"}:
        raise ValueError("FlashAttention used an unsupported launch API")
    identity = launch["kernel_identity"]
    if not isinstance(identity, str) or "flash" not in identity.lower():
        raise ValueError(
            f"captured kernel is not attributable to FlashAttention: {identity!r}")
    if launch["identity_source"] not in {"dladdr", "mcModuleGetFunction"}:
        raise ValueError("captured FlashAttention kernel identity is unresolved")
    for field in ("grid", "block"):
        dimensions = launch[field]
        if (not isinstance(dimensions, list) or len(dimensions) != 3
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value <= 0 for value in dimensions)):
            raise ValueError(f"captured {field} must be positive dim3")
    shared_memory = launch["dynamic_shared_memory_bytes"]
    if (isinstance(shared_memory, bool) or not isinstance(shared_memory, int)
            or shared_memory < 0):
        raise ValueError("dynamic shared memory must be non-negative")
    if (launch["static_shared_memory_bytes"] is not None
            or launch["static_shared_memory_status"]
            != "not_exposed_by_launch_api"):
        raise ValueError("capture claims unavailable static shared memory")
    return launch


def _worker(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--q-len", type=int, required=True)
    parser.add_argument("--kv-len", type=int, required=True)
    parser.add_argument("--query-heads", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    args = parser.parse_args(argv)
    if min(
            args.q_len, args.kv_len, args.query_heads, args.kv_heads,
            args.head_dim) <= 0:
        raise ValueError("FlashAttention dimensions must be positive")
    if args.q_len > args.kv_len:
        raise ValueError("FlashAttention requires q_len <= kv_len")
    import torch
    import flash_attn
    from flash_attn import flash_attn_varlen_func

    torch.cuda.set_device(args.device)
    block_size = 128
    block_count = (args.kv_len + block_size - 1) // block_size
    query = torch.empty(
        args.q_len, args.query_heads, args.head_dim,
        dtype=torch.bfloat16, device="cuda")
    key = torch.empty(
        block_count, block_size, args.kv_heads, args.head_dim,
        dtype=torch.bfloat16, device="cuda")
    value = torch.empty_like(key)
    query_cumulative = torch.tensor(
        [0, args.q_len], dtype=torch.int32, device="cuda")
    kv_cumulative = torch.tensor(
        [0, args.kv_len], dtype=torch.int32, device="cuda")
    block_table = torch.tensor(
        [list(range(block_count))], dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()

    process = ctypes.CDLL(None)
    begin = process.midlayer_dry_launch_begin
    begin.argtypes = [ctypes.c_size_t]
    begin.restype = ctypes.c_int
    end = process.midlayer_dry_launch_end
    end.argtypes = []
    end.restype = ctypes.c_int
    error = process.midlayer_dry_launch_error
    error.argtypes = []
    error.restype = ctypes.c_char_p
    if begin(1) != 0:
        raise RuntimeError(error().decode("utf-8"))
    try:
        output = flash_attn_varlen_func(
            query, key, value, query_cumulative, kv_cumulative,
            max_seqlen_q=args.q_len, max_seqlen_k=args.kv_len,
            softmax_scale=1.0 / (args.head_dim ** 0.5), causal=True,
            block_table=block_table)
    finally:
        end_status = end()
        if end_status != 0:
            raise RuntimeError(error().decode("utf-8"))
    print(json.dumps({
        "execute_api_called": True,
        "output_shape": list(output.shape),
        "torch_version": torch.__version__,
        "flash_attn_version": flash_attn.__version__,
    }, sort_keys=True))


def _parse_worker_output(stdout):
    lines = [line for line in stdout.splitlines() if line]
    if len(lines) != 1:
        raise ValueError(f"unexpected worker stdout: {stdout!r}")
    payload = json.loads(lines[0])
    expected = {
        "execute_api_called", "output_shape", "torch_version",
        "flash_attn_version",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("worker metadata fields changed")
    if payload["execute_api_called"] is not True:
        raise ValueError("worker did not call FlashAttention")
    if payload["torch_version"] != "2.4.0+metax3.7.2.0":
        raise ValueError("unexpected MetaX torch version")
    if payload["flash_attn_version"] != "2.6.3":
        raise ValueError("unexpected MetaX FlashAttention version")
    return payload


def _write_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def _parse_bool(value):
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("boolean value must be true or false")


def build_report(args):
    if args.device < 0:
        raise ValueError("device must be non-negative")
    if args.batch != 1:
        raise ValueError("FlashAttention dry-launch PoC supports batch=1")
    if min(
            args.q_len, args.kv_len, args.query_heads, args.kv_heads,
            args.head_dim) <= 0:
        raise ValueError("FlashAttention dimensions must be positive")
    if args.q_len > args.kv_len:
        raise ValueError("FlashAttention requires q_len <= kv_len")
    if args.query_heads % args.kv_heads:
        raise ValueError("FlashAttention requires integral GQA head groups")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    if args.dtype != "bf16" or args.kv_layout != "paged_varlen":
        raise ValueError("PoC supports only BF16 paged_varlen")
    if args.causal is not True:
        raise ValueError("PoC supports only causal attention")
    repository_root = args.repository_root.resolve()
    source_identity = _validate_source_identity(
        repository_root, args.baseline_commit)
    output_dir = _require_output_dir(args.output_dir, repository_root)
    build_dir = output_dir / "build"
    raw_dir = output_dir / "raw_capture"
    logs_dir = output_dir / "logs"
    build_dir.mkdir()
    raw_dir.mkdir()
    logs_dir.mkdir()
    maca_path = args.maca_path.resolve()
    interposer = _build_interposer(
        repository_root, maca_path, args.compiler, build_dir)
    capture_path = raw_dir / "capture.jsonl"
    environment = os.environ.copy()
    if "LD_PRELOAD" in environment and environment["LD_PRELOAD"]:
        raise ValueError("LD_PRELOAD must be empty for dry-launch attribution")
    environment["LD_PRELOAD"] = str(interposer)
    environment["MIDLAYER_C500_DRY_LAUNCH_CAPTURE"] = str(capture_path)
    environment["MACA_VISIBLE_DEVICES"] = str(args.device)
    command = [
        args.python, str(repository_root / RUNNER_RELATIVE), "_worker",
        "--device", str(args.device), "--q-len", str(args.q_len),
        "--kv-len", str(args.kv_len), "--query-heads",
        str(args.query_heads), "--kv-heads", str(args.kv_heads),
        "--head-dim", str(args.head_dim),
    ]
    result = _run(
        command, environment=environment, timeout=args.timeout_seconds)
    (logs_dir / "worker_stdout.txt").write_text(
        result.stdout, encoding="utf-8")
    (logs_dir / "worker_stderr.txt").write_text(
        result.stderr, encoding="utf-8")
    worker = _parse_worker_output(result.stdout)
    if worker["output_shape"] != [
            args.q_len, args.query_heads, args.head_dim]:
        raise ValueError("FlashAttention output shape changed")
    launch = _parse_capture(capture_path)
    identity_after = _validate_source_identity(
        repository_root, args.baseline_commit)
    if identity_after != source_identity:
        raise RuntimeError("source identity changed during dry launch")
    return {
        "schema_version": 1,
        "hardware_id": HARDWARE_ID,
        "mechanism": "flash_attn_execute_dry_launch",
        "baseline_commit": args.baseline_commit,
        "source_provenance": source_identity,
        "software": {
            "maca": "3.7.2", "torch": worker["torch_version"],
            "flash_attn": worker["flash_attn_version"],
        },
        "workload": {
            "model": args.model, "variant": "prefill",
            "batch": args.batch, "q_len": args.q_len,
            "kv_len": args.kv_len, "num_query_heads": args.query_heads,
            "num_kv_heads": args.kv_heads, "head_dim": args.head_dim,
            "dtype": args.dtype, "kv_layout": args.kv_layout,
            "causal": args.causal,
        },
        "execute_api_called": True,
        "device_kernel_submitted": False,
        "duration_consumed": False,
        "output_shape": worker["output_shape"],
        "interposition": {
            "expected_launch_count": 1,
            "captured_launch_count": 1,
            "active_marker_launch_policy": "recorded_and_not_submitted",
            "blocked_launch_apis": sorted(INTERPOSED_LAUNCH_APIS),
        },
        "launch": launch,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maca-path", type=Path, default=Path("/opt/maca"))
    parser.add_argument("--python", required=True)
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--model", default="Qwen3-4B")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-len", type=int, default=729)
    parser.add_argument("--kv-len", type=int, default=729)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--kv-layout", default="paged_varlen")
    parser.add_argument("--causal", type=_parse_bool,
                        choices=(True, False), default=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = build_report(args)
    result_path = args.output_dir.resolve() / "result.json"
    _write_json(result_path, report)
    print(json.dumps({
        "device_kernel_submitted": report["device_kernel_submitted"],
        "kernel_identity": report["launch"]["kernel_identity"],
        "output": str(result_path),
    }, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        _worker(sys.argv[2:])
    else:
        main()
