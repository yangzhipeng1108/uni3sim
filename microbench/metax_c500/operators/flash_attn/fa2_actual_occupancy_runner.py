#!/usr/bin/env python3
"""Dry-launch resource probe for C500's actual FA2 SplitKV reducer.

The marker suppresses every selected kernel launch.  This runner records only
the driver-reported resources and occupancy of the function chosen by the
FlashAttention host selector for a declared synthetic source geometry.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


INTERPOSER = Path(__file__).with_name("fa2_actual_occupancy_interposer.cpp")
CASES = ("frozen56_split5", "graph_b1_split103", "graph_b1_auto")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command, *, environment=None, cwd=None):
    result = subprocess.run(
        command, check=False, capture_output=True, text=True,
        env=environment, cwd=cwd)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command!r}\\n"
            f"stdout:\\n{result.stdout}\\nstderr:\\n{result.stderr}")
    return result


def _build(args, build_dir):
    compiler = shutil.which(args.compiler)
    if compiler is None:
        raise FileNotFoundError(f"compiler is unavailable: {args.compiler}")
    header = args.maca_path / "include/mcr/mc_runtime.h"
    if not header.is_file():
        raise FileNotFoundError(f"missing MACA runtime header: {header}")
    output = build_dir / "libmidlayer_fa2_actual_occupancy.so"
    _run([
        compiler, "-std=c++14", "-O2", "-Wall", "-Wextra", "-Werror",
        "-fPIC", "-shared", "-isystem", str(args.maca_path / "include/mcr"),
        "-isystem", str(args.maca_path / "include"), str(INTERPOSER),
        "-ldl", "-pthread", "-o", str(output),
    ])
    return output


def _worker(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--case", required=True, choices=CASES)
    args = parser.parse_args(argv)
    import flash_attn
    from flash_attn import flash_attn_with_kvcache
    import torch

    kv_lens = (
        745, 23, 123, 71, 33, 69, 48, 88, 677, 236, 15, 386,
        408, 24, 750, 252, 27, 409, 331, 67, 418, 94, 186, 31,
        33, 193, 512, 254, 389, 839, 390, 9, 683, 29, 745, 388,
        61, 335, 14, 40, 164, 791, 32, 60, 26, 337, 60, 275, 7,
        50, 159, 307, 773, 316, 21, 76)
    if len(kv_lens) != 56:
        raise AssertionError("frozen request vector changed")
    if args.case in ("graph_b1_split103", "graph_b1_auto"):
        kv_lens = (729,)
        page_size = 16
        max_pages = 512
        requested_splits = 0 if args.case == "graph_b1_auto" else 103
    else:
        page_size = 128
        max_pages = max((length + page_size - 1) // page_size
                        for length in kv_lens)
        requested_splits = 5
    # The parent constrains MACA_VISIBLE_DEVICES to the requested physical
    # device.  The child therefore sees that device at logical ordinal zero;
    # passing the physical ordinal here would select a non-existent device
    # whenever the requested card is not zero.
    torch.cuda.set_device(0)
    q_heads, kv_heads, head_dim = 32, 8, 128
    batch = len(kv_lens)
    query = torch.empty(batch, 1, q_heads, head_dim, dtype=torch.bfloat16,
                        device="cuda")
    key = torch.empty(batch * max_pages, page_size, kv_heads, head_dim,
                      dtype=torch.bfloat16, device="cuda")
    value = torch.empty_like(key)
    cache_seqlens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    block_table = torch.arange(batch * max_pages, dtype=torch.int32,
                               device="cuda").reshape(batch, max_pages)
    process = ctypes.CDLL(None)
    begin = process.midlayer_fa_occupancy_begin
    begin.argtypes, begin.restype = [], ctypes.c_int
    end = process.midlayer_fa_occupancy_end
    end.argtypes, end.restype = [], ctypes.c_int
    error = process.midlayer_fa_occupancy_error
    error.argtypes, error.restype = [], ctypes.c_char_p
    if begin() != 0:
        raise RuntimeError(error().decode("utf-8"))
    try:
        output = flash_attn_with_kvcache(
            query, key, value, cache_seqlens=cache_seqlens,
            softmax_scale=1.0 / (head_dim ** 0.5), causal=True,
            block_table=block_table, num_splits=requested_splits)
    finally:
        if end() != 0:
            raise RuntimeError(error().decode("utf-8"))
    print(json.dumps({
        "torch": torch.__version__, "flash_attn": flash_attn.__version__,
        "output_shape": list(output.shape),
        "marked_flash_attn_kernel_submitted": False,
        "case": args.case,
    }, sort_keys=True))


def _parse_capture(path, expected_grid_x):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if not records or records[-1]["record_type"] != "summary":
        raise ValueError("occupancy capture lacks a summary")
    launches = records[:-1]
    combine = [record for record in launches if "splitkv_combine" in
               record["kernel_identity"].lower()]
    if len(combine) != 1:
        raise ValueError(f"expected one SplitKV combine launch, got {len(combine)}")
    target, = combine
    if (target["grid"] != [expected_grid_x, 1, 1]
            or target["block"] != [256, 1, 1]):
        raise ValueError(f"frozen reducer geometry changed: {target}")
    if target["resident_ctas_per_ap"] <= 0:
        raise ValueError("runtime did not provide a positive occupancy")
    return launches, target


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--case", choices=CASES, default="frozen56_split5")
    parser.add_argument("--maca-path", type=Path, default=Path("/opt/maca"))
    parser.add_argument("--compiler", default="c++")
    args = parser.parse_args(argv)
    if args.device < 0:
        raise ValueError("device must be non-negative")
    if args.output_dir.exists():
        raise FileExistsError(f"output directory exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    build_dir = args.output_dir / "build"
    build_dir.mkdir()
    interposer = _build(args, build_dir)
    capture = args.output_dir / "capture.jsonl"
    environment = os.environ.copy()
    if "LD_PRELOAD" in environment and environment["LD_PRELOAD"]:
        raise ValueError("LD_PRELOAD must be empty")
    environment["LD_PRELOAD"] = str(interposer)
    environment["MIDLAYER_C500_FA_OCCUPANCY_CAPTURE"] = str(capture)
    environment["MACA_VISIBLE_DEVICES"] = str(args.device)
    result = _run([
        sys.executable, str(Path(__file__).resolve()), "_worker",
        "--device", str(args.device), "--case", args.case],
        environment=environment)
    (args.output_dir / "worker_stdout.txt").write_text(result.stdout)
    (args.output_dir / "worker_stderr.txt").write_text(result.stderr)
    worker = json.loads(result.stdout)
    launches, combine = _parse_capture(
        capture, 4 if args.case in ("graph_b1_split103", "graph_b1_auto")
        else 224)
    report = {
        "schema_version": 1,
        "hardware_id": "metax_c500",
        "mechanism": "flash_attn_splitkv_combine_runtime_occupancy",
        "purpose": "static runtime resource and CTA-admission evidence",
        "production_or_evaluation_duration_consumed": False,
        "marked_flash_attn_kernel_submitted": False,
        "software": worker,
        "frozen_workload": (
            {"batch": 1, "q_len": 1, "kv_lens": [729],
             "block_table_width": 512, "page_size": 16,
             "head_dim": 128, "num_query_heads": 32,
             "num_kv_heads": 8,
             "requested_splits": 0 if args.case == "graph_b1_auto" else 103}
            if args.case in ("graph_b1_split103", "graph_b1_auto") else
            {"batch": 56, "q_len": 1, "kv_lens_count": 56,
             "head_dim": 128, "num_query_heads": 32,
             "num_kv_heads": 8, "requested_splits": 5}),
        "source": {
            "interposer": str(INTERPOSER),
            "interposer_sha256": _sha256(INTERPOSER),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": _sha256(Path(__file__)),
        },
        "captured_launch_count": len(launches),
        "splitkv_combine": combine,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(args.output_dir / "report.json"),
                      "resident_ctas_per_ap": combine["resident_ctas_per_ap"]},
                     sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        _worker(sys.argv[2:])
    else:
        main()
