import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import tempfile

import pandas as pd
from tqdm import tqdm


from dataset_pipeline.kernel_dataset.config import (
    REPO_ROOT as PROJECT_ROOT,
    resolve_project_path,
    load_kernel_workload_config as _load_kernel_config,
    to_float,
    load_pipeline_config,
    clean_dir,
    get_kernel_pipeline_settings,
    get_kernel_config_path,
    get_kernel_script_path,
    get_kernel_rep_dir,
    get_kernel_csv_dir,
    get_kernel_dataset_path,
    get_profiler_output_dir,
    parse_kernel_list,
    require_mapping,
    require_non_empty_list,
    require_non_empty_string,
    require_profiler_executable,
    require_positive_int,
    current_python_executable,
)
from dataset_pipeline.kernel_dataset.dataset_writer import (
    build_kernel_evaluation_timing,
    publish_complete_dataset,
    write_complete_candidate,
)
from dataset_pipeline.kernel_dataset.workload_shapes import (
    require_staging_baseline_keys,
    build_flash_attn_shapes as build_common_flash_attn_shapes,
    build_gemm_shapes,
    build_vector_shapes,
)
from src.kernel.dataset_storage import load_kernel_dataset_document


# Expected A800 (CUDA) target-kernel name substring per vector op — the serving
# kernel each op must resolve to (goal §5 alignment red line). These are the
# vllm 0.11.0 CUDA kernels, distinct from the 910B Ascend names in
# build_vector_shapes' target_kernel (RmsNorm/SwiGlu/...). Confirmed on-device
# via NCU capture. elementwise add/mul both compile to the same templated
# elementwise kernel, so a shared "elementwise" substring covers both.
_VECTOR_A800_KERNEL = {
    "rmsnorm": "rms_norm_kernel",
    "add_rmsnorm": "fused_add_rms_norm_kernel",
    "rope": "rotary_embedding_kernel",
    "silu": "act_and_mul_kernel",
    "elementwise": "elementwise_kernel",
    "kvcache": "reshape_and_cache_flash_kernel",
}

_NSYS_PROFILE_ARGS = (
    "profile",
    "--trace=cuda,nvtx",
    "--sample=none",
    "--cpuctxsw=none",
    "--export=sqlite",
    "--force-overwrite=true",
)


def _required_profiler_text(value, context):
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{context} must be a non-empty string")
    return value.strip()


def _load_vector_timing_observations(path, shape_index):
    """Load one serving-path kernel median for every Vector workload.

    NCU remains the counter/launch provider.  This independent nsys capture is
    the evaluation ground truth because it records serving-path device time,
    rather than NCU replay duration.  Production Forward does not consume it.
    """
    if not isinstance(path, str) or not path:
        raise TypeError("A800 Vector timing CSV path must be non-empty")
    if not isinstance(shape_index, dict) or not shape_index:
        raise TypeError("A800 Vector shape index must be a non-empty mapping")
    required_columns = {
        "NVTX Range", "Style", "Kern Inst", "Med (ns)", "Kernel Name",
    }
    observations = {}
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"A800 Vector timing CSV has no header: {path}")
        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                "A800 Vector timing CSV is missing columns: "
                f"{sorted(missing_columns)}")
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(
                    "A800 Vector timing CSV row has extra columns at "
                    f"{path}:{line_number}")
            nvtx_range = row["NVTX Range"].strip()
            if not nvtx_range.startswith(":VECTOR_"):
                continue
            case_key = nvtx_range[1:]
            if case_key in observations:
                raise ValueError(
                    f"duplicate A800 Vector timing observation: {case_key}")
            if row["Style"].strip() != "PushPop":
                raise ValueError(
                    f"A800 Vector timing range must be PushPop: {case_key}")
            kernel_instances = to_float(row["Kern Inst"])
            if (kernel_instances <= 0
                    or not kernel_instances.is_integer()):
                raise ValueError(
                    "A800 Vector timing kernel count must be a positive "
                    f"integer for {case_key}: {kernel_instances}")
            duration_ns = to_float(row["Med (ns)"])
            if duration_ns <= 0:
                raise ValueError(
                    f"A800 Vector timing must be positive for {case_key}")
            if case_key not in shape_index:
                raise ValueError(
                    f"A800 Vector timing has unknown case: {case_key}")
            operation = shape_index[case_key]["attr"]["op"]
            expected_kernel = _VECTOR_A800_KERNEL[operation]
            kernel_name = _required_profiler_text(
                row["Kernel Name"], "nsys Kernel Name")
            if expected_kernel not in kernel_name:
                raise ValueError(
                    "A800 Vector timing range contains the wrong kernel for "
                    f"{case_key}: expected {expected_kernel!r}, got "
                    f"{kernel_name!r}")
            observations[case_key] = duration_ns

    expected = set(shape_index)
    actual = set(observations)
    if actual != expected:
        raise ValueError(
            "A800 Vector timing cases do not match the workload contract: "
            f"missing={sorted(expected - actual)[:8]}, "
            f"unexpected={sorted(actual - expected)[:8]}")
    return observations


def _profiler_scalar_text(value, context):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError(
            f"{context} must be a string or numeric profiler scalar")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context} must be finite")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{context} must not be empty")
    return text


def _resolve_kernel_cfg(cfg, kernel):
    pipeline_settings = get_kernel_pipeline_settings(cfg)
    kernel_config_path = get_kernel_config_path(cfg, kernel)

    kernel_cfg = _load_kernel_config(kernel_config_path, kernel)
    kernel_cfg["nvtx_range_name"] = pipeline_settings["nvtx_range_name"]
    kernel_cfg["script_path"] = get_kernel_script_path(cfg, kernel)

    return kernel_cfg


def build_flash_attn_shapes(kernel_cfg):
    """Attach A800-specific public split requests to shared FA cases."""

    result = []
    for shape in build_common_flash_attn_shapes(kernel_cfg):
        attributes = dict(shape["attr"])
        if (attributes["kv_layout"] == "mla"
                and attributes["split_policy"] == "serving"):
            # The Ampere vLLM Triton MLA fallback uses a fixed four-way split.
            # This value is passed to decode_attention_fwd and sizes its scratch
            # tensor; it is an executed input, not a filename label.
            attributes["SP_REQUESTED"] = 4
        result.append({**shape, "attr": attributes})
    return result


def _build_gemm_jobs(cfg, kernel_cfg, ncu_bin, python_bin):
    jobs = []
    nvtx = kernel_cfg["nvtx_range_name"]
    script_path = kernel_cfg["script_path"]

    for shape in build_gemm_shapes(kernel_cfg):
        filename = shape["filename"]
        runner_cmd = [
            python_bin, script_path,
            *shape["script_args"],
            "--nvtx_range_name", nvtx,
        ]
        cmd = [
            ncu_bin,
            "--set", "full",
            "--export", filename,
            "--force-overwrite",
            "--target-processes", "all",
            "--nvtx",
            "--nvtx-include", nvtx + "/",
            "--apply-rules", "off",
            *runner_cmd,
        ]
        jobs.append({
            "filename": filename,
            "cmd": cmd,
            "runner_cmd": runner_cmd,
        })

    return jobs


def _fa_shape_index(kernel_cfg):
    """filename -> shape dict for the fa op grid. Shapes (kv_layout × kv_dist ×
    seq_len_q × causal × layers, machine-independent) are produced by the SAME
    expander as the 910B msprof flow (build_flash_attn_shapes) — this keeps
    910B/A800 fa entries shape-comparable (goal: consistent with 910B). The rich attr
    (kv_layout/kv_dist/Q_LEN/KV_LEN/causal/layer) is injected in stage3 from
    this index (the filename integer-pair parser can't encode those fields)."""
    return {s["filename"]: s for s in build_flash_attn_shapes(kernel_cfg)}


def _flash_attn_kernel_names(entry):
    return [
        _required_profiler_text(
            kernel["Metadata"]["Kernel Name"], "NCU Kernel Name")
        for kernel in entry["kernels"]
    ]


def _metadata_dimensions(kernel, field):
    value = _required_profiler_text(
        kernel["Metadata"][field], f"NCU Metadata.{field}")
    match = re.fullmatch(
        r"\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\)",
        value)
    if match is None:
        raise ValueError(f"invalid NCU Metadata.{field}: {value!r}")
    dimensions = tuple(int(group) for group in match.groups())
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(
            f"NCU {field} dimensions must be positive: {dimensions!r}")
    return dimensions


def _metadata_grid(kernel):
    return _metadata_dimensions(kernel, "Grid Size")


def _observed_flash_attn_splits(shape, entry):
    """Return the executed A800 split count from profiler launch geometry."""
    attributes = shape["attr"]
    if attributes["Q_LEN"] > 1:
        return 1
    if attributes["kv_layout"] == "mla":
        # The A800 Triton MLA runner allocates its scratch tensor with the
        # explicitly requested split count.  Its grid_y is a head-group axis,
        # not a split axis, so treating grid_y as SP would fabricate SP=4.
        return attributes["SP_REQUESTED"]

    batch_heads = attributes["B"] * attributes["HKV"]
    candidates = set()
    for kernel in entry["kernels"]:
        name = _required_profiler_text(
            kernel["Metadata"]["Kernel Name"], "NCU Kernel Name").lower()
        grid_x, grid_y, grid_z = _metadata_grid(kernel)
        if "flash_fwd_splitkv_kernel" in name:
            if (grid_y == attributes["B"]
                    and grid_z == attributes["HKV"]):
                candidates.add(1)
            elif grid_z == batch_heads:
                candidates.add(grid_y)
            elif grid_y == batch_heads:
                candidates.add(grid_z)
            elif grid_x == batch_heads and grid_z == 1:
                candidates.add(grid_y)
            else:
                raise ValueError(
                    "A800 split-KV grid does not contain B*HKV: "
                    f"grid={(grid_x, grid_y, grid_z)}, B*HKV={batch_heads}")
        elif "flash_fwd_kernel" in name and "splitkv" not in name:
            candidates.add(1)
    if len(candidates) != 1:
        raise ValueError(
            "A800 FA case must expose exactly one executed split count, got "
            f"{sorted(candidates)} for {shape['filename']}")
    executed = next(iter(candidates))
    if executed <= 0:
        raise ValueError("A800 executed split count must be positive")
    return executed


def _validate_forced_split_capture(case_key, shape, entry):
    """Prove that one controlled-split A800 capture used its configured SP.

    The 2026-08-03 runner wrote configured SP values into case IDs while
    passing SP=1 to FlashAttention. Case IDs therefore are not provenance.
    For SP>1, the real launch must contain both split-KV stages and the main
    stage grid must be B * HKV * SP. This rejects mislabeled fixed-SP1 data.
    """
    if shape["split_policy"] != "forced":
        raise ValueError(
            f"flash_attn case is not a forced-split case: {case_key}")

    expected_attr = shape["attr"]
    names = _flash_attn_kernel_names(entry)
    splits = expected_attr["SP_REQUESTED"]
    split_main = [
        kernel for kernel in entry["kernels"]
        if "flash_fwd_splitkv_kernel" in kernel["Metadata"]["Kernel Name"]
    ]
    split_combine = [
        kernel for kernel in entry["kernels"]
        if "flash_fwd_splitkv_combine_kernel"
        in kernel["Metadata"]["Kernel Name"]
    ]

    if splits == 1:
        if (len(names) != 1 or "flash_fwd_kernel" not in names[0]
                or "splitkv" in names[0]):
            raise ValueError(
                f"forced SP=1 case has an invalid launch for {case_key}: "
                f"kernels={names}")
        return

    if len(names) != 2 or len(split_main) != 1 or len(split_combine) != 1:
        raise ValueError(
            f"forced SP={splits} case was not captured with split-KV and "
            f"combine stages for {case_key}: kernels={names}")

    main_grid = to_float(
        split_main[0]["Sections"]["Launch Statistics"]["Header"]["Grid Size"])
    expected_main_grid = (
        expected_attr["B"] * expected_attr["HKV"] * splits)
    if main_grid != expected_main_grid:
        raise ValueError(
            f"forced SP={splits} case has the wrong split-KV grid for "
            f"{case_key}: expected={expected_main_grid}, observed={main_grid}")


def _build_flash_attn_jobs(kernel_cfg, ncu_bin, python_bin):
    """One NCU job per fa shape. Each runs flash_attn_runner.py --kv_layout ... with the
    A800 serving op (contiguous Dao / paged / paged_varlen / mla triton /
    chunked_prefill) wrapped in NVTX; --nvtx-include profile_region/ profiles
    exactly that kernel (setup/warmup run outside the range). The configured
    num_splits is passed to the runner. Contiguous decode executes it directly;
    FA2 varlen accepts only 1 and may select the actual split inside its C++
    implementation, which evaluation recovers from the observed launch grid."""
    jobs = []
    nvtx = kernel_cfg["nvtx_range_name"]
    script_path = kernel_cfg["script_path"]

    for shape in _fa_shape_index(kernel_cfg).values():
        filename = shape["filename"]
        runner_cmd = [
            python_bin, script_path,
            *shape["script_args"],
            "--num_splits", str(shape["attr"]["SP_REQUESTED"]),
            "--nvtx_range_name", nvtx,
        ]
        cmd = [
            ncu_bin,
            "--set", "full",
            "--export", filename,
            "--force-overwrite",
            "--target-processes", "all",
            "--nvtx",
            "--nvtx-include", nvtx + "/",
            "--apply-rules", "off",
            *runner_cmd,
        ]
        jobs.append({
            "filename": filename,
            "cmd": cmd,
            "runner_cmd": runner_cmd,
            "requires_workload_reprofile": (
                shape["attr"]["Q_LEN"] > 1
                and shape["attr"]["kv_layout"] == "paged_varlen"
            ),
        })

    return jobs


def _build_fused_moe_jobs(kernel_cfg, ncu_bin, python_bin):
    jobs = []
    nvtx = kernel_cfg["nvtx_range_name"]
    script_path = kernel_cfg["script_path"]
    seen_filenames = set()

    workload_configs = require_non_empty_list(
        kernel_cfg, "workload_configs", "fused_moe config")
    layers = require_non_empty_list(
        kernel_cfg, "layers", "fused_moe config")
    cases = []
    for workload_cfg in workload_configs:
        require_mapping(workload_cfg, "each fused_moe workload")
        num_tokens = require_positive_int(
            workload_cfg, "num_tokens", "fused_moe workload")
        for layer_cfg in layers:
            require_mapping(layer_cfg, "each fused_moe layer")
            cases.append({
                "num_tokens": num_tokens,
                "num_experts": require_positive_int(
                    layer_cfg, "num_experts", "fused_moe layer"),
                "top_k": require_positive_int(
                    layer_cfg, "top_k", "fused_moe layer"),
                "hidden_size": require_positive_int(
                    layer_cfg, "hidden_size", "fused_moe layer"),
                "intermediate_size": require_positive_int(
                    layer_cfg, "intermediate_size", "fused_moe layer"),
                "dtype": require_non_empty_string(
                    layer_cfg, "dtype", "fused_moe layer"),
            })

    for case in cases:
        num_tokens = case["num_tokens"]
        num_experts = case["num_experts"]
        top_k = case["top_k"]
        hidden_size = case["hidden_size"]
        intermediate_size = case["intermediate_size"]
        dtype = case["dtype"]

        filename = (
            f"FUSEDMOE_TOK{num_tokens}_E{num_experts}_TOPK{top_k}"
            f"_H{hidden_size}_INT{intermediate_size}"
        )
        if filename in seen_filenames:
            raise ValueError(f"duplicate fused_moe filename: {filename}")
        seen_filenames.add(filename)

        runner_cmd = [
            python_bin, script_path,
            "--num_tokens", str(num_tokens),
            "--num_experts", str(num_experts),
            "--top_k", str(top_k),
            "--hidden_size", str(hidden_size),
            "--intermediate_size", str(intermediate_size),
            "--dtype", dtype,
            "--nvtx_range_name", nvtx,
            "--tiling_output", f"__tiling_{filename}.json",
        ]
        cmd = [
            ncu_bin,
            "--set", "full",
            "--export", filename,
            "--force-overwrite",
            "--target-processes", "all",
            "--nvtx",
            "--nvtx-include", nvtx + "/",
            "--apply-rules", "off",
            *runner_cmd,
        ]

        jobs.append({
            "filename": filename,
            "cmd": cmd,
            "runner_cmd": runner_cmd,
        })

    return jobs


def _vector_shape_index(kernel_cfg):
    """filename -> shape dict for the vector op grid. Shapes (op × layers × toks
    cartesian, real-model dims) are machine-independent, so we reuse the SAME
    expander as the 910B msprof flow (build_vector_shapes) — this is what keeps
    910B/A800/C500 vector entries byte-for-byte shape-comparable (goal §2)."""
    return {s["filename"]: s for s in build_vector_shapes(kernel_cfg)}


def _build_vector_jobs(kernel_cfg, ncu_bin, python_bin):
    """One NCU job per vector shape. Each runs vector_runner.py --op ... with the
    target serving op wrapped in NVTX; --nvtx-include profile_region/ makes NCU
    profile exactly that kernel (setup randn/init run outside the range)."""
    jobs = []
    nvtx = kernel_cfg["nvtx_range_name"]
    script_path = kernel_cfg["script_path"]

    for shape in _vector_shape_index(kernel_cfg).values():
        filename = shape["filename"]
        runner_cmd = [
            python_bin, script_path,
            *shape["script_args"],
            "--nvtx_range_name", nvtx,
        ]
        cmd = [
            ncu_bin,
            "--set", "full",
            "--export", filename,
            "--force-overwrite",
            "--target-processes", "all",
            "--nvtx",
            "--nvtx-include", nvtx + "/",
            "--apply-rules", "off",
            *runner_cmd,
        ]
        jobs.append({
            "filename": filename,
            "cmd": cmd,
            "runner_cmd": runner_cmd,
            "vector_op": shape["attr"]["op"],
        })

    return jobs


def _build_jobs(cfg, kernel, kernel_cfg):
    ncu_bin = require_profiler_executable("ncu", "NCU_BIN")
    python_bin = current_python_executable()

    return _build_jobs_with_executables(
        cfg, kernel, kernel_cfg, ncu_bin, python_bin)


def _build_jobs_with_executables(
        cfg, kernel, kernel_cfg, ncu_bin, python_bin):

    if kernel == "gemm":
        return _build_gemm_jobs(cfg, kernel_cfg, ncu_bin, python_bin)
    elif kernel == "flash_attn":
        return _build_flash_attn_jobs(kernel_cfg, ncu_bin, python_bin)
    elif kernel == "fused_moe":
        return _build_fused_moe_jobs(kernel_cfg, ncu_bin, python_bin)
    elif kernel == "vector":
        return _build_vector_jobs(kernel_cfg, ncu_bin, python_bin)
    else:
        raise ValueError(f"unsupported NVIDIA A800 kernel: {kernel!r}")


def _replace_command_option(command, option, value):
    indices = [index for index, token in enumerate(command) if token == option]
    if len(indices) != 1:
        raise ValueError(
            f"runner command must contain exactly one {option}, got "
            f"{len(indices)}: {command!r}")
    value_index = indices[0] + 1
    if value_index >= len(command):
        raise ValueError(f"runner command has no value for {option}")
    result = list(command)
    result[value_index] = value
    return result


def _timing_marker(kernel, case_key):
    for value, context in ((kernel, "kernel"), (case_key, "case key")):
        if (not isinstance(value, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value)
                is None):
            raise ValueError(
                f"A800 timing {context} is not path/marker safe: {value!r}")
    return f"A800_KERNEL::{kernel}::{case_key}"


def _build_nsys_timing_jobs(cfg, kernel, kernel_cfg, nsys_bin, python_bin):
    ncu_jobs = _build_jobs_with_executables(
        cfg, kernel, kernel_cfg, "ncu", python_bin)
    jobs = []
    for ncu_job in ncu_jobs:
        job = dict(ncu_job)
        marker = _timing_marker(kernel, job["filename"])
        job["marker"] = marker
        job["runner_cmd"] = _replace_command_option(
            job["runner_cmd"], "--nvtx_range_name", marker)
        job["nsys_bin"] = nsys_bin
        jobs.append(job)
    return jobs


def _apply_nsys_pre_nvtx_cache_eviction(jobs, cache_eviction_mib):
    """Add an explicit pre-NVTX cold-cache sweep to every NSYS runner."""
    if (isinstance(cache_eviction_mib, bool)
            or not isinstance(cache_eviction_mib, int)):
        raise TypeError("A800 timing cache eviction MiB must be an integer")
    if cache_eviction_mib < 0:
        raise ValueError("A800 timing cache eviction MiB cannot be negative")
    if 0 < cache_eviction_mib < 320:
        raise ValueError(
            "A800 timing cache eviction must be zero or at least 320 MiB "
            "(8x the 40 MiB L2)")
    if cache_eviction_mib == 0:
        return jobs
    updated = []
    for job in jobs:
        runner_command = list(job["runner_cmd"])
        if "--pre-nvtx-cache-eviction-mib" in runner_command:
            raise ValueError(
                "A800 NSYS runner already declares a pre-NVTX cache sweep")
        rewritten = dict(job)
        rewritten["runner_cmd"] = [
            *runner_command,
            "--pre-nvtx-cache-eviction-mib", str(cache_eviction_mib),
        ]
        updated.append(rewritten)
    return updated


def _canonical_gemm_dtype_evidence(entry, case_key):
    kernels = entry["kernels"]
    if not isinstance(kernels, list) or not kernels:
        raise ValueError(
            f"canonical GEMM case has no NCU kernels: {case_key}")
    evidence = set()
    for kernel in kernels:
        name = _required_profiler_text(
            kernel["Metadata"]["Kernel Name"], "canonical NCU Kernel Name")
        lowered = name.lower()
        if any(token in lowered for token in (
                "bf16", "bfloat16", "__nv_bfloat16")):
            evidence.add("bf16")
        fp16_without_bf16 = re.search(r"(?<!b)f16", lowered) is not None
        if ("fp16" in lowered or fp16_without_bf16
                or "__half" in lowered or "c10::half" in lowered):
            evidence.add("fp16")
    if len(evidence) != 1:
        raise ValueError(
            "canonical GEMM NCU kernel identity has ambiguous or missing "
            f"dtype evidence for {case_key}: {sorted(evidence)}")
    return next(iter(evidence))


def _apply_canonical_gemm_timing_dtypes(kernel, jobs, baseline):
    if kernel != "gemm":
        return jobs
    if not isinstance(baseline, dict):
        raise TypeError("canonical GEMM baseline must be a mapping")

    rewritten = []
    for job in jobs:
        case_key = job["filename"]
        if case_key not in baseline:
            raise ValueError(
                f"canonical GEMM baseline is missing case {case_key}")
        entry = baseline[case_key]
        if not isinstance(entry, dict) or "op_info" not in entry:
            raise ValueError(
                f"canonical GEMM case is missing op_info: {case_key}")
        op_info = entry["op_info"]
        if not isinstance(op_info, dict) or "Attr" not in op_info:
            raise ValueError(
                f"canonical GEMM case is missing op_info.Attr: {case_key}")
        attributes = op_info["Attr"]
        if not isinstance(attributes, dict) or "dtype" not in attributes:
            raise ValueError(
                f"canonical GEMM case is missing op_info.Attr.dtype: "
                f"{case_key}")
        dtype = attributes["dtype"]
        if not isinstance(dtype, str) or dtype not in ("fp16", "bf16"):
            raise ValueError(
                "canonical GEMM dtype must be 'fp16' or 'bf16': "
                f"case={case_key}, dtype={dtype!r}")
        identity_dtype = _canonical_gemm_dtype_evidence(entry, case_key)
        if identity_dtype != dtype:
            raise ValueError(
                "canonical GEMM dtype is inconsistent with NCU kernel "
                f"identity: case={case_key}, Attr.dtype={dtype!r}, "
                f"identity_dtype={identity_dtype!r}")
        rewritten_job = dict(job)
        rewritten_job["runner_cmd"] = _replace_command_option(
            job["runner_cmd"], "--dtype", dtype)
        rewritten.append(rewritten_job)
    return rewritten


def _profile_run_output_dir(cfg, kernel, profile_run):
    if profile_run is None:
        return get_kernel_rep_dir(cfg, kernel)
    if (not isinstance(profile_run, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", profile_run)
            is None):
        raise ValueError(
            "profile_run must contain only letters, digits, '.', '_' or '-' ")
    return os.path.join(
        get_profiler_output_dir(cfg), "profile_runs", profile_run, kernel)


def _profile_run_csv_dir(cfg, kernel, profile_run):
    if profile_run is None:
        return get_kernel_csv_dir(cfg, kernel)
    run_dir = os.path.dirname(
        _profile_run_output_dir(cfg, kernel, profile_run))
    return os.path.join(run_dir, f"{kernel}_csv")


def _nsys_case_output_dir(cfg, profile_run, kernel, case_key):
    if profile_run is None:
        raise ValueError("A800 NSYS timing requires a named profile_run")
    run_kernel_dir = _profile_run_output_dir(cfg, kernel, profile_run)
    run_dir = os.path.dirname(run_kernel_dir)
    _timing_marker(kernel, case_key)
    return os.path.join(run_dir, "nsys", kernel, case_key)


def _nsys_kernel_output_dir(cfg, profile_run, kernel):
    case_dir = _nsys_case_output_dir(
        cfg, profile_run, kernel, "MANIFEST_PATH_CHECK")
    return os.path.dirname(case_dir)


def _nsys_manifest_path(cfg, profile_run, kernel):
    return os.path.join(
        _nsys_kernel_output_dir(cfg, profile_run, kernel),
        "_capture_manifest.json")


def _nsys_sample_paths(cfg, profile_run, kernel, case_key, sample_number):
    if (isinstance(sample_number, bool)
            or not isinstance(sample_number, int)
            or sample_number <= 0):
        raise ValueError("A800 NSYS sample number must be a positive integer")
    case_dir = _nsys_case_output_dir(
        cfg, profile_run, kernel, case_key)
    stem = os.path.join(case_dir, f"sample-{sample_number:03d}")
    return stem, stem + ".nsys-rep", stem + ".sqlite"


def _select_profile_jobs(kernel, jobs, target_vector_op,
                         target_flash_attn_reprofile):
    if target_vector_op is not None and target_flash_attn_reprofile:
        raise ValueError(
            "vector and flash_attn reprofile selectors are mutually exclusive")
    if target_vector_op is not None:
        if kernel != "vector":
            raise ValueError(
                "target_vector_op is only valid for the vector workload")
        jobs = [
            job for job in jobs
            if job["vector_op"] == target_vector_op
        ]
        if not jobs:
            raise ValueError(
                f"no vector profiling jobs match {target_vector_op!r}")
        if target_vector_op == "rope" and len(jobs) != 32:
            raise ValueError(
                "A800 RoPE reprofile set must contain exactly 32 cases, "
                f"got {len(jobs)}")
    if target_flash_attn_reprofile:
        if kernel != "flash_attn":
            raise ValueError(
                "target_flash_attn_reprofile is only valid for flash_attn")
        jobs = [
            job for job in jobs
            if job["requires_workload_reprofile"]
        ]
        if len(jobs) != 9:
            raise ValueError(
                "A800 paged-varlen prefill correction set must contain "
                f"exactly 9 cases, got {len(jobs)}")
    return jobs


def _select_task_window(jobs, max_tasks, start_idx, end_idx):
    for value, name in (
            (max_tasks, "max_tasks"), (start_idx, "start_idx"),
            (end_idx, "end_idx")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} cannot be negative")
    selected = list(jobs)
    if max_tasks > 0:
        selected = selected[:max_tasks]
    if not selected:
        raise ValueError("profiling task selection must not be empty")
    if end_idx > 0 and start_idx == 0:
        raise ValueError("end_idx requires a positive start_idx")
    if start_idx > 0:
        if end_idx > 0 and end_idx < start_idx:
            raise ValueError("end_idx cannot be smaller than start_idx")
        if start_idx > len(selected):
            raise ValueError(
                "start_idx is outside the selected profiling task range: "
                f"start_idx={start_idx}, tasks={len(selected)}")
        if end_idx > len(selected):
            raise ValueError(
                "end_idx is outside the selected profiling task range: "
                f"end_idx={end_idx}, tasks={len(selected)}")
        end = end_idx if end_idx > 0 else len(selected)
        selected = selected[start_idx - 1:end]
    if not selected:
        raise ValueError("profiling task selection must not be empty")
    return selected


def _require_sqlite_columns(connection, table, required):
    table_rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchall()
    if len(table_rows) != 1:
        raise ValueError(f"NSYS SQLite is missing required table {table!r}")
    columns = {
        row[1] for row in connection.execute(
            f'PRAGMA table_info("{table}")').fetchall()
    }
    missing = sorted(set(required) - columns)
    if missing:
        raise ValueError(
            f"NSYS SQLite table {table!r} is missing columns: {missing}")
    return columns


def _strict_sqlite_int(value, context, *, allow_zero=True):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{context} must be {qualifier}: {value!r}")
    return value


def _normalized_kernel_identity(value):
    text = _required_profiler_text(value, "kernel identity")
    # Preserve the full demangled template and signature. Only runs of
    # whitespace are canonicalized; template/overload evidence is never
    # discarded because same-base, same-geometry kernels can differ in work.
    return re.sub(r"\s+", " ", text).strip()


def load_nsys_timing_sqlite(sqlite_path, marker):
    """Load one marker's strictly correlated, ordered device launches."""
    if not isinstance(sqlite_path, (str, os.PathLike)):
        raise TypeError("NSYS SQLite path must be path-like")
    marker = _required_profiler_text(marker, "NSYS marker")
    if not os.path.isfile(sqlite_path):
        raise FileNotFoundError(f"NSYS SQLite does not exist: {sqlite_path}")

    connection = None
    try:
        connection = sqlite3.connect(str(sqlite_path))
        with connection:
            nvtx_columns = _require_sqlite_columns(
                connection, "NVTX_EVENTS",
                ("start", "end", "eventType", "globalTid", "endGlobalTid"))
            _require_sqlite_columns(
                connection, "StringIds", ("id", "value"))
            _require_sqlite_columns(
                connection, "CUPTI_ACTIVITY_KIND_RUNTIME",
                ("start", "end", "globalTid", "correlationId", "nameId"))
            _require_sqlite_columns(
                connection, "CUPTI_ACTIVITY_KIND_KERNEL",
                ("start", "end", "deviceId", "correlationId",
                 "globalPid", "demangledName", "gridX", "gridY", "gridZ",
                 "blockX", "blockY", "blockZ", "gridId"))

            if "text" in nvtx_columns and "textId" in nvtx_columns:
                marker_rows = connection.execute(
                    """
                    SELECT n.start, n.end, n.globalTid,
                           CASE WHEN n.text IS NOT NULL
                                THEN n.text ELSE s.value END
                    FROM NVTX_EVENTS n
                    LEFT JOIN StringIds s ON n.textId = s.id
                    WHERE (n.text = ? OR s.value = ?)
                      AND n.eventType IN (59, 70)
                      AND n.endGlobalTid IS NULL
                    ORDER BY n.start, n.rowid
                    """,
                    (marker, marker),
                ).fetchall()
            elif "text" in nvtx_columns:
                marker_rows = connection.execute(
                    """
                    SELECT start, end, globalTid, text
                    FROM NVTX_EVENTS
                    WHERE text = ?
                      AND eventType IN (59, 70)
                      AND endGlobalTid IS NULL
                    ORDER BY start, rowid
                    """,
                    (marker,),
                ).fetchall()
            elif "textId" in nvtx_columns:
                marker_rows = connection.execute(
                    """
                    SELECT n.start, n.end, n.globalTid, s.value
                    FROM NVTX_EVENTS n
                    JOIN StringIds s ON n.textId = s.id
                    WHERE s.value = ?
                      AND n.eventType IN (59, 70)
                      AND n.endGlobalTid IS NULL
                    ORDER BY n.start, n.rowid
                    """,
                    (marker,),
                ).fetchall()
            else:
                raise ValueError(
                    "NSYS NVTX_EVENTS must contain text or textId")

            if not marker_rows:
                raise ValueError(
                    f"NSYS SQLite contains no marker {marker!r}")
            if len(marker_rows) != 1:
                raise ValueError(
                    "NSYS SQLite contains duplicate/overlapping marker rows "
                    f"for {marker!r}: {len(marker_rows)}")
            marker_start, marker_end, marker_tid, marker_text = marker_rows[0]
            if marker_text != marker:
                raise ValueError(f"ambiguous NSYS marker identity: {marker_text!r}")
            marker_start = _strict_sqlite_int(
                marker_start, "NSYS marker start")
            marker_end = _strict_sqlite_int(marker_end, "NSYS marker end")
            marker_tid = _strict_sqlite_int(
                marker_tid, "NSYS marker globalTid")
            if marker_end <= marker_start:
                raise ValueError(
                    f"NSYS marker is not completed: start={marker_start}, "
                    f"end={marker_end}")

            rows = connection.execute(
                """
                SELECT r.rowid, r.start, r.end, r.correlationId,
                       runtime_name.value,
                       k.rowid, k.start, k.end, k.deviceId,
                       k.correlationId, kernel_name.value,
                       k.gridX, k.gridY, k.gridZ,
                       k.blockX, k.blockY, k.blockZ, k.gridId
                FROM CUPTI_ACTIVITY_KIND_RUNTIME r
                JOIN StringIds runtime_name ON r.nameId = runtime_name.id
                LEFT JOIN CUPTI_ACTIVITY_KIND_KERNEL k
                       ON r.correlationId = k.correlationId
                      AND k.globalPid =
                          (r.globalTid & 0xFFFFFFFFFF000000)
                LEFT JOIN StringIds kernel_name
                       ON k.demangledName = kernel_name.id
                WHERE r.globalTid = ? AND r.start >= ? AND r.end <= ?
                ORDER BY r.start, r.rowid, k.start, k.gridId, k.rowid
                """,
                (marker_tid, marker_start, marker_end),
            ).fetchall()
    except sqlite3.Error as error:
        raise ValueError(
            f"failed to load NSYS SQLite {sqlite_path}: {error}") from error
    finally:
        if connection is not None:
            connection.close()

    launch_rows = {}
    for row in rows:
        runtime_name = _required_profiler_text(
            row[4], "NSYS CUDA runtime API name")
        if "launch" not in runtime_name.lower():
            continue
        runtime_rowid = row[0]
        if runtime_rowid not in launch_rows:
            launch_rows[runtime_rowid] = []
        launch_rows[runtime_rowid].append(row)
    if not launch_rows:
        raise ValueError(f"NSYS marker {marker!r} contains no CUDA launches")

    launches = []
    seen_kernel_rows = set()
    for runtime_rowid, correlated_rows in launch_rows.items():
        correlation = _strict_sqlite_int(
            correlated_rows[0][3],
            f"NSYS runtime correlation for row {runtime_rowid}")
        kernel_rows = [row for row in correlated_rows if row[5] is not None]
        if not kernel_rows:
            raise ValueError(
                "NSYS CUDA launch is missing a correlated kernel row: "
                f"runtime_rowid={runtime_rowid}, correlation={correlation}")
        kernel_rowids = {row[5] for row in kernel_rows}
        if len(kernel_rowids) != 1 or len(kernel_rows) != 1:
            raise ValueError(
                "ambiguous NSYS runtime-to-kernel correlation: "
                f"runtime_rowid={runtime_rowid}, correlation={correlation}, "
                f"kernel_rows={sorted(kernel_rowids)}")
        row = kernel_rows[0]
        kernel_rowid = row[5]
        if kernel_rowid in seen_kernel_rows:
            raise ValueError(
                f"NSYS kernel row {kernel_rowid} has ambiguous runtime identity")
        seen_kernel_rows.add(kernel_rowid)
        if row[9] != correlation:
            raise ValueError(
                f"NSYS kernel correlation mismatch for row {kernel_rowid}")
        start = _strict_sqlite_int(row[6], "NSYS kernel start")
        end = _strict_sqlite_int(row[7], "NSYS kernel end")
        if end <= start:
            raise ValueError(
                f"NSYS kernel duration must be positive: start={start}, "
                f"end={end}")
        device = _strict_sqlite_int(row[8], "NSYS kernel deviceId")
        grid = tuple(
            _strict_sqlite_int(value, "NSYS kernel grid", allow_zero=False)
            for value in row[11:14])
        block = tuple(
            _strict_sqlite_int(value, "NSYS kernel block", allow_zero=False)
            for value in row[14:17])
        grid_id = _strict_sqlite_int(row[17], "NSYS kernel gridId")
        name = _required_profiler_text(row[10], "NSYS kernel name")
        launches.append({
            "name": name,
            "identity": _normalized_kernel_identity(name),
            "grid": grid,
            "block": block,
            "device": device,
            "duration_ns": end - start,
            "start": start,
            "grid_id": grid_id,
        })

    devices = {launch["device"] for launch in launches}
    if len(devices) != 1:
        raise ValueError(
            f"NSYS marker must contain exactly one device, got {sorted(devices)}")
    launches.sort(key=lambda launch: (launch["start"], launch["grid_id"]))
    order_keys = [
        (launch["start"], launch["grid_id"]) for launch in launches]
    if len(set(order_keys)) != len(order_keys):
        raise ValueError(
            f"ambiguous NSYS kernel ordering for marker {marker!r}")
    return launches


def _nsys_timing_command(job, output_stem, tiling_path):
    runner_cmd = list(job["runner_cmd"])
    if "--tiling_output" in runner_cmd:
        runner_cmd = _replace_command_option(
            runner_cmd, "--tiling_output", tiling_path)
    return [
        job["nsys_bin"], *_NSYS_PROFILE_ARGS,
        "--output", output_stem,
        *runner_cmd,
    ]


def _bounded_process_text(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value[-2000:]


def _checked_metadata_command(command, context, *, cwd=None):
    try:
        return subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"{context} failed: command={command!r}, "
            f"stdout={_bounded_process_text(error.stdout)!r}, "
            f"stderr={_bounded_process_text(error.stderr)!r}") from error
    except OSError as error:
        raise RuntimeError(
            f"{context} failed to start: command={command!r}, "
            f"error={_bounded_process_text(error)!r}") from error


def _capture_source_identity():
    commit = _checked_metadata_command(
        ["git", "rev-parse", "HEAD"], "source commit query",
        cwd=PROJECT_ROOT)
    status = _checked_metadata_command(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        "source dirty-state query", cwd=PROJECT_ROOT)
    diff = _checked_metadata_command(
        ["git", "diff", "--binary", "HEAD"],
        "source diff query", cwd=PROJECT_ROOT)
    untracked = _checked_metadata_command(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        "source untracked-file query", cwd=PROJECT_ROOT)
    digest = hashlib.sha256()
    digest.update(status.encode("utf-8"))
    digest.update(b"\0")
    digest.update(diff.encode("utf-8"))
    for relative_path in untracked.split("\0"):
        if not relative_path:
            continue
        normalized_path = os.path.normpath(relative_path)
        if (os.path.isabs(relative_path) or normalized_path == ".."
                or normalized_path.startswith(".." + os.sep)):
            raise ValueError(
                f"git returned an unsafe untracked path: {relative_path!r}")
        path = os.path.join(PROJECT_ROOT, normalized_path)
        digest.update(normalized_path.encode("utf-8"))
        digest.update(b"\0")
        if os.path.islink(path):
            digest.update(os.readlink(path).encode("utf-8"))
        elif os.path.isfile(path):
            with open(path, "rb") as stream:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
        else:
            raise ValueError(
                f"untracked source path is not a file or symlink: {path}")
        digest.update(b"\0")
    return {
        "commit": _required_profiler_text(commit, "source commit"),
        "dirty": bool(status),
        "dirty_state_sha256": digest.hexdigest(),
    }


def _validate_a800_device_identity(device):
    if not isinstance(device, dict) or set(device) != {
            "index", "name", "capability", "total_memory_bytes"}:
        raise ValueError("A800 NSYS device identity fields are invalid")
    _strict_sqlite_int(device["index"], "A800 device index")
    device_name = _required_profiler_text(
        device["name"], "A800 device name")
    if re.match(r"^NVIDIA A800(?:\b|-)", device_name,
                re.IGNORECASE) is None:
        raise ValueError(
            "A800 NSYS capture requires an NVIDIA A800 device, got "
            f"{device_name!r}")
    capability = device["capability"]
    if capability != [8, 0]:
        raise ValueError(
            "A800 device capability must be [8, 0], got "
            f"{capability!r}")
    _strict_sqlite_int(
        device["total_memory_bytes"], "A800 device memory",
        allow_zero=False)
    return device


def _capture_runtime_identity(nsys_bin, python_bin):
    nsys_version = _checked_metadata_command(
        [nsys_bin, "--version"], "NSYS version query")
    script = (
        "import json,sys,torch;"
        "device=torch.cuda.current_device();"
        "props=torch.cuda.get_device_properties(device);"
        "print(json.dumps({"
        "'python_version':sys.version,"
        "'torch_version':torch.__version__,"
        "'cuda_version':torch.version.cuda,"
        "'device':{"
        "'index':device,'name':props.name,"
        "'capability':[props.major,props.minor],"
        "'total_memory_bytes':props.total_memory}}))"
    )
    payload_text = _checked_metadata_command(
        [python_bin, "-c", script], "CUDA runtime identity query")
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"CUDA runtime identity is not JSON: {payload_text!r}") from error
    if not isinstance(payload, dict):
        raise TypeError("CUDA runtime identity must be a mapping")
    required = {
        "python_version", "torch_version", "cuda_version", "device"}
    if set(payload) != required:
        raise ValueError(
            "CUDA runtime identity fields are invalid: "
            f"expected={sorted(required)}, actual={sorted(payload)}")
    _validate_a800_device_identity(payload["device"])
    visible_devices = None
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
    return {
        "hardware_id": "nvidia_a800",
        "nsys_executable": nsys_bin,
        "nsys_version": _required_profiler_text(
            nsys_version, "NSYS version"),
        "python_executable": python_bin,
        "python_version": payload["python_version"],
        "torch_version": payload["torch_version"],
        "cuda_version": payload["cuda_version"],
        "device": payload["device"],
        "cuda_visible_devices": visible_devices,
    }


def _capture_manifest_payload(
        profile_run, kernel, sample_count, jobs, nsys_bin, python_bin,
        max_tasks, start_idx, end_idx, target_vector_op,
        target_flash_attn_reprofile):
    return {
        "schema_version": 1,
        "source": _capture_source_identity(),
        "runtime": _capture_runtime_identity(nsys_bin, python_bin),
        "profile_run": profile_run,
        "kernel": kernel,
        "sample_count": sample_count,
        "selection": {
            "max_tasks": max_tasks,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "target_vector_op": target_vector_op,
            "target_flash_attn_reprofile": target_flash_attn_reprofile,
        },
        "command_contract": {
            "nsys_profile_args": list(_NSYS_PROFILE_ARGS),
            "cases": [
                {
                    "case_key": job["filename"],
                    "marker": job["marker"],
                    "runner_command": list(job["runner_cmd"]),
                }
                for job in jobs
            ],
        },
    }


def _load_json_mapping(path, context):
    try:
        with open(path, encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to load {context} {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a mapping: {path}")
    return value


def _create_or_validate_capture_manifest(path, expected):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        existing = _load_json_mapping(path, "A800 NSYS capture manifest")
        if existing != expected:
            raise ValueError(
                "A800 NSYS capture manifest is incompatible with this "
                f"resume request: {path}")
        return path

    existing_entries = [
        name for name in os.listdir(os.path.dirname(path))
        if not name.startswith("._capture_manifest.")]
    if existing_entries:
        raise ValueError(
            "A800 NSYS raw artifacts exist without a provenance manifest; "
            f"refusing to adopt them: {sorted(existing_entries)}")

    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix="._capture_manifest.", suffix=".tmp",
        dir=os.path.dirname(path))
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(expected, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            existing = _load_json_mapping(
                path, "A800 NSYS capture manifest")
            if existing != expected:
                raise ValueError(
                    "A800 NSYS capture manifest was concurrently created "
                    f"with incompatible provenance: {path}")
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return path


def _validate_complete_capture_manifest(
        cfg, profile_run, kernel, sample_count, expected_jobs):
    expected_keys = [job["filename"] for job in expected_jobs]
    path = _nsys_manifest_path(cfg, profile_run, kernel)
    manifest = _load_json_mapping(path, "A800 NSYS capture manifest")
    required = {
        "schema_version", "source", "runtime", "profile_run", "kernel",
        "sample_count", "selection", "command_contract"}
    if set(manifest) != required:
        raise ValueError(
            "A800 NSYS capture manifest fields are invalid: "
            f"expected={sorted(required)}, actual={sorted(manifest)}")
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported A800 NSYS capture manifest version")
    if manifest["profile_run"] != profile_run or manifest["kernel"] != kernel:
        raise ValueError(
            "A800 NSYS capture manifest run/kernel identity mismatch")
    if manifest["sample_count"] != sample_count:
        raise ValueError(
            "A800 NSYS capture manifest sample count mismatch: "
            f"expected={sample_count}, actual={manifest['sample_count']}")

    source = manifest["source"]
    if not isinstance(source, dict) or set(source) != {
            "commit", "dirty", "dirty_state_sha256"}:
        raise ValueError("A800 NSYS manifest source identity is invalid")
    source_commit = _required_profiler_text(
        source["commit"], "manifest source commit")
    if re.fullmatch(r"[0-9a-f]{40,64}", source_commit) is None:
        raise ValueError("manifest source commit must be a Git object ID")
    if not isinstance(source["dirty"], bool):
        raise TypeError("manifest source dirty flag must be boolean")
    dirty_hash = _required_profiler_text(
        source["dirty_state_sha256"], "manifest dirty-state hash")
    if re.fullmatch(r"[0-9a-f]{64}", dirty_hash) is None:
        raise ValueError("manifest dirty-state hash must be SHA-256")

    runtime = manifest["runtime"]
    runtime_fields = {
        "hardware_id", "nsys_executable", "nsys_version", "python_executable",
        "python_version", "torch_version", "cuda_version", "device",
        "cuda_visible_devices"}
    if not isinstance(runtime, dict) or set(runtime) != runtime_fields:
        raise ValueError("A800 NSYS manifest runtime identity is invalid")
    if runtime["hardware_id"] != "nvidia_a800":
        raise ValueError(
            "A800 NSYS manifest hardware_id must be 'nvidia_a800'")
    for field in (
            "nsys_executable", "nsys_version", "python_executable",
            "python_version", "torch_version", "cuda_version"):
        _required_profiler_text(runtime[field], f"manifest runtime {field}")
    if (runtime["cuda_visible_devices"] is not None
            and not isinstance(runtime["cuda_visible_devices"], str)):
        raise TypeError(
            "manifest cuda_visible_devices must be a string or null")
    _validate_a800_device_identity(runtime["device"])

    selection = manifest["selection"]
    if not isinstance(selection, dict) or set(selection) != {
            "max_tasks", "start_idx", "end_idx", "target_vector_op",
            "target_flash_attn_reprofile"}:
        raise ValueError("A800 NSYS manifest selection is invalid")
    command_contract = manifest["command_contract"]
    if not isinstance(command_contract, dict) or set(command_contract) != {
            "nsys_profile_args", "cases"}:
        raise ValueError("A800 NSYS manifest command contract is invalid")
    if command_contract["nsys_profile_args"] != list(_NSYS_PROFILE_ARGS):
        raise ValueError("A800 NSYS manifest command flags are incompatible")
    cases = command_contract["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("A800 NSYS manifest cases must be a non-empty list")
    case_keys = []
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
                "case_key", "marker", "runner_command"}:
            raise ValueError("A800 NSYS manifest case record is invalid")
        case_key = _required_profiler_text(
            case["case_key"], "manifest case key")
        if case["marker"] != _timing_marker(kernel, case_key):
            raise ValueError(
                f"A800 NSYS manifest marker mismatch for {case_key}")
        runner_command = case["runner_command"]
        if (not isinstance(runner_command, list) or not runner_command
                or any(not isinstance(value, str) or not value
                       for value in runner_command)):
            raise ValueError(
                f"A800 NSYS manifest runner command is invalid for {case_key}")
        case_keys.append(case_key)
    if case_keys != list(expected_keys):
        raise ValueError(
            "A800 NSYS timing run is incomplete for candidate publication: "
            f"expected={list(expected_keys)}, captured={case_keys}")
    for case, expected_job in zip(cases, expected_jobs, strict=True):
        if case["runner_command"] != expected_job["runner_cmd"]:
            raise ValueError(
                "A800 NSYS manifest runner command does not match the "
                f"current case contract for {case['case_key']}")
    return manifest


def step_1_run_profiling(cfg, kernel, dry_run=False, max_tasks=0,
                         target_vector_op=None, profile_run=None,
                         target_flash_attn_reprofile=False,
                         start_idx=0, end_idx=0):
    kernel_cfg = _resolve_kernel_cfg(cfg, kernel)
    jobs = _build_jobs(cfg, kernel, kernel_cfg)
    jobs = _select_profile_jobs(
        kernel, jobs, target_vector_op, target_flash_attn_reprofile)
    jobs = _select_task_window(jobs, max_tasks, start_idx, end_idx)

    pipeline_settings = get_kernel_pipeline_settings(cfg)
    gpu_device_id = pipeline_settings["gpu_device_id"]
    if gpu_device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device_id)
        print(f"[Info] CUDA_VISIBLE_DEVICES set to: {gpu_device_id}")

    ncu_output_dir = _profile_run_output_dir(cfg, kernel, profile_run)
    if dry_run:
        print(f"[Info] Dry-run mode: keep existing profiling output directory: {ncu_output_dir}")
    elif profile_run is not None:
        # Named runs are resumable and never clean or overwrite existing
        # reports.  Each completed .ncu-rep is skipped below.
        os.makedirs(ncu_output_dir, exist_ok=True)
    else:
        clean_dir(ncu_output_dir)

    print(f">>> Stage 1 [{kernel}]: NCU Profiling, {len(jobs)} tasks")

    tiling_infos = {}  # filename → tiling info dict (for fused_moe)

    for index, job in enumerate(jobs, 1):
        file_path = os.path.join(ncu_output_dir, job["filename"])
        cmd = list(job["cmd"])

        export_idx = cmd.index("--export") + 1
        cmd[export_idx] = file_path

        # Resolve tiling_output path to ncu_output_dir
        tiling_file_path = None
        if "--tiling_output" in cmd:
            tiling_idx = cmd.index("--tiling_output") + 1
            tiling_file_path = os.path.join(ncu_output_dir, cmd[tiling_idx])
            cmd[tiling_idx] = tiling_file_path

        print(f"[{index}/{len(jobs)}] Profiling: {job['filename']}")

        if dry_run:
            print("  DRY-RUN CMD:", " ".join(cmd))
            continue
        if (profile_run is not None
                and os.path.isfile(file_path + ".ncu-rep")):
            print("  SKIP: named-run report already exists")
            continue

        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.stdout:
            print(result.stdout)
        # Read tiling info from the runner's sidecar file in the same run.
        if tiling_file_path is not None and os.path.exists(tiling_file_path):
            with open(tiling_file_path) as f:
                tiling_infos[job["filename"]] = json.load(f)
            os.remove(tiling_file_path)  # clean up per-case file

    # Save tiling infos if any were captured
    if tiling_infos:
        tiling_path = os.path.join(ncu_output_dir, "_tiling_configs.json")
        with open(tiling_path, "w") as f:
            json.dump(tiling_infos, f, indent=2)
        print(f"  Saved {len(tiling_infos)} tiling configs to {tiling_path}")

    print(f"\n=== Stage 1 [{kernel}] Completed ===")
    return ncu_output_dir


def step_run_nsys_timing(
        cfg, kernel, profile_run, sample_count, dry_run=False, max_tasks=0,
        target_vector_op=None, target_flash_attn_reprofile=False,
        start_idx=0, end_idx=0, timing_cache_eviction_mib=0):
    if (isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count <= 0):
        raise ValueError("A800 NSYS timing sample_count must be positive")
    if profile_run is None:
        raise ValueError("A800 NSYS timing requires --profile-run")

    kernel_cfg = _resolve_kernel_cfg(cfg, kernel)
    pipeline_settings = get_kernel_pipeline_settings(cfg)
    gpu_device_id = pipeline_settings["gpu_device_id"]
    if gpu_device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device_id)
        print(f"[Info] CUDA_VISIBLE_DEVICES set to: {gpu_device_id}")

    nsys_bin = require_profiler_executable("nsys", "NSYS_BIN")
    python_bin = current_python_executable()
    all_jobs = _build_nsys_timing_jobs(
        cfg, kernel, kernel_cfg, nsys_bin, python_bin)
    expected_keys = [job["filename"] for job in all_jobs]
    canonical_baseline = _load_staging_baseline(
        get_kernel_dataset_path(cfg, kernel), kernel, expected_keys)
    all_jobs = _apply_canonical_gemm_timing_dtypes(
        kernel, all_jobs, canonical_baseline)
    all_jobs = _apply_nsys_pre_nvtx_cache_eviction(
        all_jobs, timing_cache_eviction_mib)
    jobs = _select_profile_jobs(
        kernel, all_jobs, target_vector_op, target_flash_attn_reprofile)
    jobs = _select_task_window(jobs, max_tasks, start_idx, end_idx)
    if not dry_run:
        manifest = _capture_manifest_payload(
            profile_run, kernel, sample_count, jobs, nsys_bin, python_bin,
            max_tasks, start_idx, end_idx, target_vector_op,
            target_flash_attn_reprofile)
        manifest_path = _nsys_manifest_path(cfg, profile_run, kernel)
        _create_or_validate_capture_manifest(manifest_path, manifest)
        print(f"[Info] NSYS capture manifest: {manifest_path}")
    total = len(jobs) * sample_count
    print(
        f">>> NSYS Timing [{kernel}]: {len(jobs)} cases, "
        f"{sample_count} samples each")

    completed = 0
    for job in jobs:
        case_key = job["filename"]
        marker = job["marker"]
        for sample_number in range(1, sample_count + 1):
            completed += 1
            stem, report_path, sqlite_path = _nsys_sample_paths(
                cfg, profile_run, kernel, case_key, sample_number)
            tiling_path = stem + ".tiling.json"
            command = _nsys_timing_command(job, stem, tiling_path)
            print(
                f"[{completed}/{total}] Timing {case_key} "
                f"sample {sample_number:03d}")
            if dry_run:
                print("  DRY-RUN CMD:", " ".join(command))
                continue

            report_exists = os.path.isfile(report_path)
            sqlite_exists = os.path.isfile(sqlite_path)
            if report_exists != sqlite_exists:
                raise RuntimeError(
                    "partial A800 NSYS timing sample; refusing to overwrite: "
                    f"report={report_exists}, sqlite={sqlite_exists}, "
                    f"stem={stem}")
            if report_exists:
                load_nsys_timing_sqlite(sqlite_path, marker)
                print("  SKIP: valid NSYS timing sample already exists")
                continue

            os.makedirs(os.path.dirname(stem), exist_ok=True)
            try:
                result = subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    "A800 NSYS timing subprocess failed: "
                    f"case={case_key}, sample={sample_number:03d}, "
                    f"command={command!r}, "
                    f"stdout={_bounded_process_text(error.stdout)!r}, "
                    f"stderr={_bounded_process_text(error.stderr)!r}") \
                    from error
            except OSError as error:
                raise RuntimeError(
                    "A800 NSYS timing subprocess failed to start: "
                    f"case={case_key}, sample={sample_number:03d}, "
                    f"command={command!r}, "
                    f"error={_bounded_process_text(error)!r}") from error
            finally:
                if os.path.isfile(tiling_path):
                    os.remove(tiling_path)
            if not os.path.isfile(report_path) or not os.path.isfile(
                    sqlite_path):
                raise RuntimeError(
                    "nsys did not produce the required report and SQLite "
                    f"for {case_key} sample {sample_number:03d}; "
                    f"stdout={result.stdout[-1000:]!r}, "
                    f"stderr={result.stderr[-1000:]!r}")
            load_nsys_timing_sqlite(sqlite_path, marker)

        if dry_run:
            continue
        _load_case_nsys_timing(
            cfg, profile_run, kernel, case_key, sample_count,
            canonical_baseline[case_key])
        print(
            "  PASS: all NSYS samples match canonical NCU launch identity "
            "and ordinal geometry")

    print(f"\n=== NSYS Timing [{kernel}] Completed ===")


def _ncu_launch_sequence(entry, case_key):
    kernels = entry["kernels"]
    if not isinstance(kernels, list) or not kernels:
        raise ValueError(
            f"NCU case {case_key} must contain a non-empty kernel list")
    sequence = []
    for kernel in kernels:
        name = _required_profiler_text(
            kernel["Metadata"]["Kernel Name"], "NCU Kernel Name")
        sequence.append((
            _normalized_kernel_identity(name),
            _metadata_dimensions(kernel, "Grid Size"),
            _metadata_dimensions(kernel, "Block Size"),
        ))
    return tuple(sequence)


def _load_case_nsys_timing(
        cfg, profile_run, kernel, case_key, sample_count, entry):
    return _load_case_nsys_timing_windows(
        cfg, profile_run, kernel, case_key, sample_count, entry)[
            "selected_kernel_durations_ns"]


def _load_case_nsys_timing_windows(
        cfg, profile_run, kernel, case_key, sample_count, entry):
    """Load fixed NSYS windows and select the median whole-call trace."""

    if (isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count <= 0):
        raise ValueError("A800 NSYS timing sample_count must be positive")
    marker = _timing_marker(kernel, case_key)
    ncu_sequence = _ncu_launch_sequence(entry, case_key)
    sample_sequences = []
    sample_durations = []
    for sample_number in range(1, sample_count + 1):
        _, report_path, sqlite_path = _nsys_sample_paths(
            cfg, profile_run, kernel, case_key, sample_number)
        if not os.path.isfile(report_path) or not os.path.isfile(sqlite_path):
            raise FileNotFoundError(
                "missing selected A800 NSYS timing sample: "
                f"case={case_key}, sample={sample_number:03d}, "
                f"report={report_path}, sqlite={sqlite_path}")
        launches = load_nsys_timing_sqlite(sqlite_path, marker)
        sequence = tuple(
            (launch["identity"], launch["grid"], launch["block"])
            for launch in launches)
        sample_sequences.append(sequence)
        sample_durations.append(
            tuple(launch["duration_ns"] for launch in launches))

    first_sequence = sample_sequences[0]
    for sample_number, sequence in enumerate(sample_sequences[1:], start=2):
        if sequence != first_sequence:
            raise ValueError(
                "A800 NSYS timing samples have different launch sequences: "
                f"case={case_key}, sample={sample_number:03d}")
    if len(first_sequence) != len(ncu_sequence):
        raise ValueError(
            "A800 NSYS/NCU launch count mismatch: "
            f"case={case_key}, nsys={len(first_sequence)}, "
            f"ncu={len(ncu_sequence)}")
    for ordinal, (nsys_launch, ncu_launch) in enumerate(
            zip(first_sequence, ncu_sequence, strict=True)):
        if nsys_launch != ncu_launch:
            raise ValueError(
                "A800 NSYS/NCU launch identity or geometry mismatch: "
                f"case={case_key}, ordinal={ordinal}, "
                f"nsys={nsys_launch!r}, ncu={ncu_launch!r}")

    sample_total_durations_ns = [
        float(math.fsum(durations)) for durations in sample_durations]
    ordered_sample_indices = sorted(
        range(sample_count),
        key=lambda sample_index: (
            sample_total_durations_ns[sample_index], sample_index))
    selected_sample_index = ordered_sample_indices[sample_count // 2]
    pairwise_relative_differences_pct = []
    for first_index in range(sample_count):
        for second_index in range(first_index + 1, sample_count):
            first = sample_total_durations_ns[first_index]
            second = sample_total_durations_ns[second_index]
            pairwise_relative_differences_pct.append(
                abs(first - second) / max(first, second) * 100.0)
    maximum_relative_difference_pct = (
        max(pairwise_relative_differences_pct)
        if pairwise_relative_differences_pct else 0.0)
    return {
        "sample_kernel_durations_ns": [list(values)
                                        for values in sample_durations],
        "sample_total_durations_ns": sample_total_durations_ns,
        "selected_sample_number": selected_sample_index + 1,
        "selected_kernel_durations_ns": list(
            sample_durations[selected_sample_index]),
        "selected_total_duration_ns": sample_total_durations_ns[
            selected_sample_index],
        "maximum_relative_difference_pct": (
            maximum_relative_difference_pct),
    }


def _build_file_data_from_df(kernel_df):
    file_data = {
        "Metadata": {},
        "Sections": {},
    }

    meta_cols = [
        'Kernel Name', 'Grid Size', 'Block Size', 'Device', 'CC', 'Context',
        'Stream',
    ]
    required_cols = [
        *meta_cols, 'Section Name', 'Body Item Label', 'Metric Name',
        'Metric Unit', 'Metric Value',
    ]
    missing = sorted(set(required_cols) - set(kernel_df.columns))
    if missing:
        raise ValueError(f"NCU CSV is missing required columns: {missing}")
    for col in meta_cols:
        value = kernel_df.iloc[0][col]
        if pd.isna(value):
            raise ValueError(f"NCU CSV metadata column {col!r} is empty")
        file_data["Metadata"][col] = _profiler_scalar_text(
            value, f"NCU metadata {col}")

    for _, row in kernel_df.iterrows():
        section_name = row['Section Name']
        if pd.isna(section_name):
            continue
        section_name = _required_profiler_text(
            section_name, "NCU Section Name")

        ignore_keywords = ["NUMA Affinity", "System", "Process"]
        if any(k in section_name for k in ignore_keywords):
            continue

        if section_name not in file_data["Sections"]:
            file_data["Sections"][section_name] = {}

        raw_label = row['Body Item Label']
        if pd.isna(raw_label) or raw_label == "":
            label_key = "Header"
        else:
            label_key = _required_profiler_text(
                raw_label, "NCU Body Item Label")

        if label_key not in file_data["Sections"][section_name]:
            file_data["Sections"][section_name][label_key] = {}

        metric_name_raw = _required_profiler_text(
            row['Metric Name'], "NCU Metric Name")
        metric_unit = row['Metric Unit']

        if pd.notna(metric_unit) and metric_unit != "":
            unit = _required_profiler_text(metric_unit, "NCU Metric Unit")
            final_metric_name = f"{metric_name_raw} [{unit}]"
        else:
            final_metric_name = metric_name_raw

        val = _normalize_optional_ncu_metric(row['Metric Value'])
        file_data["Sections"][section_name][label_key][final_metric_name] = val

    return file_data


def _normalize_optional_ncu_metric(value):
    """Normalize NCU's explicit unavailable marker; reject non-finite numbers."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, bool):
        raise TypeError("NCU metric cannot be boolean")
    if not isinstance(value, (int, float, str)):
        raise TypeError(
            f"NCU metric has unsupported type: {type(value).__name__}")
    normalized = str(value).strip().lower()
    if normalized in ("", "n/a", "nan"):
        return None
    if normalized in ("inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"):
        raise ValueError(f"NCU metric must be finite, got {value!r}")
    return value


def parse_single_csv(cfg, csv_path, nvtx_range_name):
    filename = os.path.basename(csv_path)

    type_match = re.match(r"^([a-zA-Z0-9]+)_", filename)
    if not type_match:
        raise ValueError(f"invalid NCU CSV filename: {filename}")

    op_type = type_match.group(1)

    param_matches = re.findall(r"_([a-zA-Z]+)(\d+)", filename)
    if not param_matches:
        raise ValueError(f"NCU CSV filename has no shape parameters: {filename}")

    op_info = {"Type": op_type, "Attr": dict()}
    for p_name, p_val in param_matches:
        op_info["Attr"][p_name] = int(p_val)

    required_attrs = {
        "GEMM": ["B", "M", "K", "N"],
        "FLASHATTN": ["B", "S", "HQ", "HKV", "D"],
        "FUSEDMOE": ["TOK", "E", "TOPK", "H", "INT"],
    }
    if op_type in required_attrs:
        for req in required_attrs[op_type]:
            if req not in op_info["Attr"]:
                raise ValueError(
                    f"{op_type} CSV filename is missing dimension {req}: "
                    f"{filename}")

    df = pd.read_csv(csv_path, low_memory=False, engine='c')

    if df.empty:
        raise ValueError(f"NCU CSV is empty: {csv_path}")

    nvtx_col = "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"
    if nvtx_col in df.columns:
        df = df[df[nvtx_col].astype(str).str.contains(nvtx_range_name, na=False)]
        if df.empty:
            raise ValueError(
                f"NCU CSV {filename} has no rows for NVTX range "
                f"{nvtx_range_name!r}")
    else:
        raise ValueError(
            f"NCU CSV {filename} is missing NVTX column {nvtx_col!r}")

    unique_ids = [None]
    if 'ID' in df.columns:
        id_series = df['ID'].dropna()
        if not id_series.empty:
            unique_ids = list(pd.unique(id_series))

    kernel_records = []
    for raw_kernel_id in unique_ids:
        if raw_kernel_id is None:
            kernel_df = df
        else:
            kernel_df = df[df['ID'] == raw_kernel_id]

        if kernel_df.empty:
            continue

        kernel_records.append(_build_file_data_from_df(kernel_df))

    if not kernel_records:
        raise ValueError(f"NCU CSV contains no kernel records: {csv_path}")

    return {
        "op_info": {
            "Type": op_info["Type"],
            "Attr": dict(op_info["Attr"]),
        },
        "kernels": kernel_records,
    }


def step_2_convert_reps_to_csv(cfg, kernel, profile_run=None):
    kernel_cfg = _resolve_kernel_cfg(cfg, kernel)

    ncu_bin = require_profiler_executable("ncu", "NCU_BIN")

    rep_dir = _profile_run_output_dir(cfg, kernel, profile_run)
    csv_dir = _profile_run_csv_dir(cfg, kernel, profile_run)

    if profile_run is None:
        clean_dir(csv_dir)
    else:
        os.makedirs(csv_dir, exist_ok=True)

    rep_files = glob.glob(os.path.join(rep_dir, "*.ncu-rep"))
    if not rep_files:
        raise FileNotFoundError(f"no .ncu-rep files found in {rep_dir}")

    print(f"\n>>> Stage 2 [{kernel}]: Converting {len(rep_files)} reps to CSV")
    for rep_path in tqdm(rep_files, desc="Converting"):
        filename = os.path.basename(rep_path)
        output_path = os.path.join(csv_dir, filename.replace(".ncu-rep", ".csv"))
        if profile_run is not None and os.path.isfile(output_path):
            continue

        cmd = [
            ncu_bin, "-i", rep_path,
            "--print-details", "all", "--print-units", "base",
            "--csv", "--print-metric-name", "label",
        ]

        with open(output_path, "w", encoding='utf-8') as f_out:
            subprocess.run(cmd, stdout=f_out, stderr=subprocess.PIPE, check=True, text=True)

    return csv_dir


def _load_staging_baseline(path, kernel, expected_keys):
    baseline = load_kernel_dataset_document(path)
    if not isinstance(baseline, dict):
        raise TypeError(f"canonical dataset must be a mapping: {path}")

    if kernel == "flash_attn":
        migrated = {}
        for old_key, entry in baseline.items():
            if old_key in expected_keys:
                new_key = old_key
            else:
                new_key, substitutions = re.subn(
                    r"_SP[0-9]+_", "_", old_key)
                if substitutions != 1:
                    raise ValueError(
                        "A800 FLASHATTN baseline key is neither canonical nor "
                        f"the legacy SP-tag form: {old_key}")
            if new_key in migrated:
                raise ValueError(
                    f"A800 FLASHATTN baseline key collision: {new_key}")
            migrated[new_key] = entry
        baseline = migrated

    return require_staging_baseline_keys(baseline, expected_keys, path)


def step_aggregate_nsys_timing(
        cfg, kernel, profile_run, timing_samples, timing_cache_eviction_mib=0):
    if profile_run is None:
        raise ValueError(
            "A800 NSYS timing aggregate requires a named profile_run")
    if (isinstance(timing_samples, bool)
            or not isinstance(timing_samples, int)
            or timing_samples <= 0):
        raise ValueError("A800 NSYS timing_samples must be positive")

    kernel_cfg = _resolve_kernel_cfg(cfg, kernel)
    all_jobs = _build_nsys_timing_jobs(
        cfg, kernel, kernel_cfg, "nsys", current_python_executable())
    expected_keys = [job["filename"] for job in all_jobs]
    json_path = get_kernel_dataset_path(cfg, kernel)
    full_dataset = _load_staging_baseline(
        json_path, kernel, expected_keys)
    all_jobs = _apply_canonical_gemm_timing_dtypes(
        kernel, all_jobs, full_dataset)
    all_jobs = _apply_nsys_pre_nvtx_cache_eviction(
        all_jobs, timing_cache_eviction_mib)
    _validate_complete_capture_manifest(
        cfg, profile_run, kernel, timing_samples, all_jobs)
    for case_key in expected_keys:
        entry = full_dataset[case_key]
        entry["op_info"]["evaluation"] = build_kernel_evaluation_timing(
            "nvidia_a800",
            entry["kernels"],
            _load_case_nsys_timing(
                cfg, profile_run, kernel, case_key, timing_samples, entry),
        )
        if "evaluation" in entry:
            del entry["evaluation"]
        if "Overall Duration [nsecond]" in entry["op_info"]:
            del entry["op_info"]["Overall Duration [nsecond]"]
    missing_timing = [
        case_key for case_key in expected_keys
        if "evaluation" not in full_dataset[case_key]["op_info"]]
    if missing_timing:
        raise ValueError(
            "A800 NSYS timing candidate is incomplete: "
            f"missing={missing_timing}")

    run_dir = os.path.dirname(
        _profile_run_output_dir(cfg, kernel, profile_run))
    candidate_path = os.path.join(
        run_dir, "candidate", f"{kernel}.json")
    write_complete_candidate(
        full_dataset, expected_keys, candidate_path)
    print(
        f"\n[Success] Timing-only candidate saved to: {candidate_path} "
        f"(Total: {len(full_dataset)} entries)")
    return candidate_path


def _write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=os.path.dirname(path),
                prefix=".a800-nsys-candidate-", delete=False) as stream:
            temporary_path = stream.name
            json.dump(payload, stream, indent=2, sort_keys=True,
                      ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _validate_nsys_shard_manifest(
        cfg, profile_run, kernel, timing_samples, expected_job_index):
    path = _nsys_manifest_path(cfg, profile_run, kernel)
    manifest = _load_json_mapping(path, "A800 NSYS capture manifest")
    if "command_contract" not in manifest:
        raise ValueError(
            f"A800 NSYS shard manifest has no command_contract: {path}")
    command_contract = manifest["command_contract"]
    if not isinstance(command_contract, dict) or "cases" not in command_contract:
        raise ValueError(
            f"A800 NSYS shard manifest has no case records: {path}")
    cases = command_contract["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError(
            f"A800 NSYS shard manifest case records are invalid: {path}")
    case_keys = []
    for case in cases:
        if not isinstance(case, dict) or "case_key" not in case:
            raise ValueError(
                f"A800 NSYS shard manifest case record is invalid: {path}")
        case_key = _required_profiler_text(
            case["case_key"], "manifest case key")
        if case_key not in expected_job_index:
            raise ValueError(
                "A800 NSYS shard manifest contains an unexpected case: "
                f"profile_run={profile_run}, case={case_key}")
        case_keys.append(case_key)
    if len(case_keys) != len(set(case_keys)):
        raise ValueError(
            f"A800 NSYS shard manifest contains duplicate cases: {path}")
    selected_jobs = [expected_job_index[case_key] for case_key in case_keys]
    validated = _validate_complete_capture_manifest(
        cfg, profile_run, kernel, timing_samples, selected_jobs)
    return validated, selected_jobs


def step_aggregate_nsys_timing_shards(
        cfg, kernel, profile_runs, candidate_run, timing_samples,
        maximum_relative_difference_pct=5.0, retry_profile_runs=None):
    """Merge disjoint capture shards into one timing-only candidate."""

    if not isinstance(profile_runs, list) or not profile_runs:
        raise TypeError("A800 NSYS timing shard runs must be a non-empty list")
    if len(profile_runs) != len(set(profile_runs)):
        raise ValueError("A800 NSYS timing shard runs must be distinct")
    if retry_profile_runs is None:
        retry_profile_runs = []
    if not isinstance(retry_profile_runs, list):
        raise TypeError("A800 NSYS timing retry runs must be a list")
    if len(retry_profile_runs) != len(set(retry_profile_runs)):
        raise ValueError("A800 NSYS timing retry runs must be distinct")
    overlap = sorted(set(profile_runs) & set(retry_profile_runs))
    if overlap:
        raise ValueError(
            f"A800 NSYS timing shard/retry runs overlap: {overlap}")
    if timing_samples != 3:
        raise ValueError(
            "A800 NSYS golden candidate requires exactly three windows")
    if (isinstance(maximum_relative_difference_pct, bool)
            or not isinstance(maximum_relative_difference_pct, (int, float))
            or not math.isfinite(maximum_relative_difference_pct)
            or maximum_relative_difference_pct <= 0):
        raise ValueError(
            "A800 NSYS maximum relative difference must be positive")
    _profile_run_output_dir(cfg, kernel, candidate_run)

    kernel_cfg = _resolve_kernel_cfg(cfg, kernel)
    all_jobs = _build_nsys_timing_jobs(
        cfg, kernel, kernel_cfg, "nsys", current_python_executable())
    expected_keys = [job["filename"] for job in all_jobs]
    json_path = get_kernel_dataset_path(cfg, kernel)
    full_dataset = _load_staging_baseline(
        json_path, kernel, expected_keys)
    all_jobs = _apply_canonical_gemm_timing_dtypes(
        kernel, all_jobs, full_dataset)
    all_jobs = _apply_nsys_pre_nvtx_cache_eviction(all_jobs, 0)
    expected_job_index = {job["filename"]: job for job in all_jobs}
    if len(expected_job_index) != len(all_jobs):
        raise ValueError("A800 NSYS workload contains duplicate case keys")

    case_profile_runs = {}
    manifests = []
    for profile_run in profile_runs:
        manifest, selected_jobs = _validate_nsys_shard_manifest(
            cfg, profile_run, kernel, timing_samples, expected_job_index)
        manifests.append({
            "profile_run": profile_run,
            "source": manifest["source"],
            "runtime": manifest["runtime"],
            "selection": manifest["selection"],
            "case_count": len(selected_jobs),
        })
        for job in selected_jobs:
            case_key = job["filename"]
            if case_key in case_profile_runs:
                raise ValueError(
                    "A800 NSYS shard runs overlap: "
                    f"case={case_key}, first={case_profile_runs[case_key]}, "
                    f"second={profile_run}")
            case_profile_runs[case_key] = profile_run
    missing = sorted(set(expected_keys) - set(case_profile_runs))
    unexpected = sorted(set(case_profile_runs) - set(expected_keys))
    if missing or unexpected:
        raise ValueError(
            "A800 NSYS shard union is incomplete: "
            f"missing={missing[:16]}, unexpected={unexpected[:16]}")

    retry_manifests = []
    retry_case_profile_runs = {}
    for profile_run in retry_profile_runs:
        manifest, selected_jobs = _validate_nsys_shard_manifest(
            cfg, profile_run, kernel, timing_samples, expected_job_index)
        retry_manifests.append({
            "profile_run": profile_run,
            "source": manifest["source"],
            "runtime": manifest["runtime"],
            "selection": manifest["selection"],
            "case_count": len(selected_jobs),
        })
        for job in selected_jobs:
            case_key = job["filename"]
            if case_key in retry_case_profile_runs:
                raise ValueError(
                    "A800 NSYS retry runs overlap: "
                    f"case={case_key}, "
                    f"first={retry_case_profile_runs[case_key]}, "
                    f"second={profile_run}")
            retry_case_profile_runs[case_key] = profile_run

    case_evidence = {}
    failed_cases = {}
    for case_key in expected_keys:
        profile_run = (
            retry_case_profile_runs[case_key]
            if case_key in retry_case_profile_runs
            else case_profile_runs[case_key])
        entry = full_dataset[case_key]
        measurement = _load_case_nsys_timing_windows(
            cfg, profile_run, kernel, case_key, timing_samples, entry)
        relative_difference = measurement[
            "maximum_relative_difference_pct"]
        case_evidence[case_key] = {
            "profile_run": profile_run,
            **measurement,
        }
        if relative_difference > maximum_relative_difference_pct:
            failed_cases[case_key] = {
                "sample_total_durations_ns": measurement[
                    "sample_total_durations_ns"],
                "maximum_relative_difference_pct": relative_difference,
            }
            continue
        entry["op_info"]["evaluation"] = build_kernel_evaluation_timing(
            "nvidia_a800", entry["kernels"],
            measurement["selected_kernel_durations_ns"])
        if "evaluation" in entry:
            del entry["evaluation"]
        if "Overall Duration [nsecond]" in entry["op_info"]:
            del entry["op_info"]["Overall Duration [nsecond]"]

    candidate_root = os.path.join(
        get_profiler_output_dir(cfg), "profile_runs", candidate_run,
        "candidate")
    summary_path = os.path.join(
        candidate_root, f"{kernel}_summary_manifest.json")
    summary = {
        "schema_version": 1,
        "status": "pass" if not failed_cases else "rejected",
        "hardware_id": "nvidia_a800",
        "kernel": kernel,
        "timing_source": "nsys_cuda_device_kernel_start_end",
        "aggregation": "median_whole_trace_of_three_independent_windows",
        "maximum_relative_difference_limit_pct": (
            maximum_relative_difference_pct),
        "expected_case_count": len(expected_keys),
        "measured_case_count": len(case_evidence),
        "passing_case_count": len(case_evidence) - len(failed_cases),
        "failing_case_count": len(failed_cases),
        "profile_runs": manifests,
        "retry_profile_runs": retry_manifests,
        "failed_cases": failed_cases,
        "cases": case_evidence,
    }
    _write_json_atomic(summary_path, summary)
    if failed_cases:
        raise ValueError(
            "A800 NSYS timing shards exceed the reproducibility limit: "
            f"kernel={kernel}, failing_case_count={len(failed_cases)}, "
            f"summary={summary_path}")

    candidate_path = os.path.join(candidate_root, f"{kernel}.json")
    write_complete_candidate(
        full_dataset, expected_keys, candidate_path)
    print(
        f"\n[Success] Sharded timing-only candidate saved to: "
        f"{candidate_path} (Total: {len(full_dataset)} entries)")
    return candidate_path, summary_path


def step_3_aggregate_csv_to_json(
        cfg, kernel, profile_run=None, target_vector_op=None,
        target_flash_attn_reprofile=False):
    kernel_cfg = _resolve_kernel_cfg(cfg, kernel)

    csv_dir = _profile_run_csv_dir(cfg, kernel, profile_run)
    json_path = get_kernel_dataset_path(cfg, kernel)

    print(f"\n>>> Stage 3 [{kernel}]: Parsing CSVs into Dataset")
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"no NCU CSV files found in {csv_dir}")

    captured_dataset = {}
    for csv_file in tqdm(csv_files, desc="Parsing"):
        file_key = os.path.splitext(os.path.basename(csv_file))[0]
        parsed_data = parse_single_csv(cfg, csv_file, kernel_cfg["nvtx_range_name"])
        captured_dataset[file_key] = parsed_data

    all_jobs = _build_jobs(cfg, kernel, kernel_cfg)
    expected_keys = [job["filename"] for job in all_jobs]
    selected_jobs = _select_profile_jobs(
        kernel, all_jobs, target_vector_op,
        target_flash_attn_reprofile)
    expected_captured = {job["filename"] for job in selected_jobs}
    if set(captured_dataset) != expected_captured:
        raise ValueError(
            "staging capture does not exactly match its selected replacement "
            f"set: missing={sorted(expected_captured - set(captured_dataset))[:8]}, "
            f"unexpected={sorted(set(captured_dataset) - expected_captured)[:8]}")

    if profile_run is None:
        full_dataset = captured_dataset
    else:
        full_dataset = _load_staging_baseline(
            json_path, kernel, expected_keys)
        full_dataset.update(captured_dataset)

    if kernel == "gemm":
        shape_index = {
            shape["filename"]: shape for shape in build_gemm_shapes(kernel_cfg)
        }
        for case_key, entry in full_dataset.items():
            if case_key not in shape_index:
                raise ValueError(
                    f"gemm dataset contains an unknown case: {case_key}")
            entry["op_info"]["Attr"] = dict(
                shape_index[case_key]["attr"])

    # Inject MOE tiling configs into dataset if available
    if kernel == "fused_moe":
        rep_dir = _profile_run_output_dir(cfg, kernel, profile_run)
        tiling_path = os.path.join(rep_dir, "_tiling_configs.json")
        if not os.path.exists(tiling_path):
            raise FileNotFoundError(
                f"fused_moe aggregation requires tiling data: {tiling_path}")
        with open(tiling_path) as f:
            tiling_configs = json.load(f)
        if set(tiling_configs) != set(full_dataset):
            raise ValueError(
                "fused_moe tiling cases do not match captured cases: "
                f"missing={sorted(set(full_dataset) - set(tiling_configs))}, "
                f"unexpected={sorted(set(tiling_configs) - set(full_dataset))}")
        for case_key, tiling in tiling_configs.items():
            moe_tiling = {}
            for proj in ("w1", "w2"):
                tp = tiling[proj]
                moe_tiling[proj] = {
                    "N": tp["N"],
                    "K": tp["K"],
                    "BM": tp["BLOCK_SIZE_M"],
                    "BN": tp["BLOCK_SIZE_N"],
                    "BK": tp["BLOCK_SIZE_K"],
                    "GSM": tp["GROUP_SIZE_M"],
                    "EM": tp["EM"],
                    "grid": tp["grid"],
                }
            full_dataset[case_key]["op_info"]["Attr"]["moe_tiling"] = moe_tiling
            full_dataset[case_key]["op_info"]["Attr"]["tok_per_expert"] = tiling["tok_per_expert"]
        print(f"  Injected moe_tiling into {len(tiling_configs)} cases")

    # Vector: the filename regex can't encode the op string / element_count /
    # per-op dims, so parse_single_csv leaves Type="VECTOR" + integer-pair Attr.
    # Overwrite each entry with the real predictor Type (NORM/ROTARY/ACTIVATION/
    # ELEMENTWISE/KV_CACHE) and the full machine-independent Attr from the shared
    # expander, so A800 vector entries carry the SAME op_info schema as 910B.
    # Also verify (alignment red line, goal §5) that the captured kernel is the
    # target serving kernel — not a setup randn/copy/index kernel.
    if kernel == "vector":
        shape_index = _vector_shape_index(kernel_cfg)
        settings = get_kernel_pipeline_settings(cfg)
        timing_path = resolve_project_path(require_non_empty_string(
            settings, "vector_timing_csv",
            "nvidia_a800 kernel.pipeline.settings"))
        timing_observations = _load_vector_timing_observations(
            timing_path, shape_index)
        injected = 0
        bad_kernel = []
        for case_key, entry in full_dataset.items():
            if case_key not in shape_index:
                raise ValueError(
                    f"vector dataset contains an unknown case: {case_key}")
            shape = shape_index[case_key]
            oi = entry["op_info"]
            oi["Type"] = shape["type"]
            oi["Attr"] = dict(shape["attr"])
            injected += 1
            names = [
                _required_profiler_text(
                    k["Metadata"]["Kernel Name"], "NCU Kernel Name")
                for k in entry["kernels"]
            ]
            want = _VECTOR_A800_KERNEL[shape["attr"]["op"]]
            if not any(want in name for name in names):
                bad_kernel.append((case_key, want, names))
            kernels = entry["kernels"]
            if not isinstance(kernels, list) or len(kernels) != 1:
                raise ValueError(
                    "A800 Vector timing contract requires exactly one "
                    f"profiled serving kernel for {case_key}")
            entry["op_info"]["evaluation"] = build_kernel_evaluation_timing(
                "nvidia_a800", kernels, [timing_observations[case_key]])
            if "evaluation" in entry:
                del entry["evaluation"]
            if "Overall Duration [nsecond]" in entry["op_info"]:
                del entry["op_info"]["Overall Duration [nsecond]"]
        print(f"  Injected vector Type/Attr into {injected}/{len(shape_index)} cases")
        if bad_kernel:
            raise ValueError(
                "vector captures do not contain their required serving kernel: "
                f"{bad_kernel[:8]!r}")

    # Flash_attn: the filename integer-pair parser can't encode kv_layout/
    # kv_dist/seq_len_q/causal/layer, so overwrite each entry's Attr with the
    # full machine-independent Attr from build_flash_attn_shapes (same expander
    # as 910B → A800 fa entries carry the SAME op_info schema as 910B, goal:
    # consistent with 910B). Soft-verify the captured kernel is a real serving flash/mla
    # kernel (per layout), not a setup randn/copy kernel.
    if kernel == "flash_attn":
        shape_index = _fa_shape_index(kernel_cfg)
        injected = 0
        bad_kernel = []
        # Per-layout expected kernel-name substrings (the serving kernels).
        _FA_A800_KERNEL = {
            "contiguous": "flash_fwd",
            "paged": "flash_fwd",
            "paged_varlen": "flash_fwd",
            "chunked_prefill": "flash_fwd",
            "mla": "_fwd",            # triton MLA: _fwd_grouped_kernel_stage1 / _fwd_kernel_stage2
        }
        for case_key, entry in full_dataset.items():
            if case_key not in shape_index:
                raise ValueError(
                    f"flash_attn dataset contains an unknown case: {case_key}")
            shape = shape_index[case_key]
            if shape["split_policy"] == "forced":
                _validate_forced_split_capture(case_key, shape, entry)
            oi = entry["op_info"]
            oi["Type"] = shape["type"]
            attributes = dict(shape["attr"])
            attributes["SP"] = _observed_flash_attn_splits(shape, entry)
            oi["Attr"] = attributes
            injected += 1
            layout = shape["attr"]["kv_layout"]
            want = _FA_A800_KERNEL[layout]
            names = [
                _required_profiler_text(
                    k["Metadata"]["Kernel Name"], "NCU Kernel Name")
                for k in entry["kernels"]
            ]
            if not any(want in name for name in names):
                bad_kernel.append((case_key, layout, want, names))
        print(f"  Injected flash_attn Type/Attr into {injected}/{len(shape_index)} cases")
        if bad_kernel:
            raise ValueError(
                "flash_attn captures do not contain their required serving "
                f"kernel: {bad_kernel[:8]!r}")

    if profile_run is None:
        candidate_path = os.path.join(
            get_profiler_output_dir(cfg), "candidate", f"{kernel}.json")
        publish_complete_dataset(
            full_dataset, expected_keys, candidate_path, json_path)
        output_path = json_path
    else:
        run_dir = os.path.dirname(
            _profile_run_output_dir(cfg, kernel, profile_run))
        candidate_path = os.path.join(
            run_dir, "candidate", f"{kernel}.json")
        write_complete_candidate(
            full_dataset, expected_keys, candidate_path)
        output_path = candidate_path
    print(
        f"\n[Success] Dataset saved to: {output_path} "
        f"(Total: {len(full_dataset)} entries)")

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dataset/nvidia_a800.yaml")
    parser.add_argument(
        "--kernel",
        type=str,
        required=True,
        help=(
            "Comma-separated kernel list to process, "
            "e.g. '--kernel gemm,flash_attn'."
        ),
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["profile", "timing", "convert", "aggregate"],
        help=(
            "Run exactly one stage per invocation. "
            "profile: collect NCU reports; timing: collect per-case NSYS "
            "samples; convert: optional report-to-CSV step; "
            "aggregate: convert reports and publish canonical JSON. "
            "Per-kernel JSONs land in kernel_dataset/<hardware>/; eval merges them "
            "in memory (load_kernel_dataset) — no separate merge stage."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only print commands during profile or timing capture")
    parser.add_argument(
        "--max-tasks", type=int, default=0,
        help="Limit profile or timing case count for quick bring-up")
    parser.add_argument(
        "--start-idx", type=int, default=0,
        help="1-based first profiling case after selectors and max-tasks")
    parser.add_argument(
        "--end-idx", type=int, default=0,
        help="Inclusive last profiling case; requires a positive start-idx")
    parser.add_argument(
        "--timing-samples", type=int,
        help=("Positive per-case NSYS sample count; required by timing and "
              "selects timing samples during named-run aggregate"))
    parser.add_argument(
        "--timing-cache-eviction-mib", type=int, default=0,
        help=("BF16 cache sweep before the NSYS NVTX target call; zero "
              "disables it, otherwise requires at least 320 MiB"))
    parser.add_argument(
        "--target-vector-op",
        choices=("rmsnorm", "add_rmsnorm", "rope", "silu",
                 "elementwise", "kvcache"),
        help="Capture only this vector operation (profile/timing stages)")
    parser.add_argument(
        "--profile-run",
        help=("Write profile reports to a new named staging run under "
              "profiler_output instead of replacing the legacy report dir"))
    parser.add_argument(
        "--timing-shard-run", action="append",
        help=("One disjoint named NSYS timing shard consumed by aggregate; "
              "repeat for every shard"))
    parser.add_argument(
        "--timing-retry-run", action="append",
        help=("One named NSYS timing retry run overriding its captured cases; "
              "repeat for every retry run"))
    parser.add_argument(
        "--timing-candidate-run",
        help=("Named profile_runs directory receiving a merged timing-only "
              "candidate and validation manifest"))
    parser.add_argument(
        "--target-flash-attn-reprofile",
        action="store_true",
        help=("Profile only the 9 paged-varlen prefill cases whose corrected "
              "per-request workload requires Q_LENS=KV_LENS"))
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    kernels = parse_kernel_list(cfg, args.kernel)

    if args.timing_samples is not None and args.timing_samples <= 0:
        parser.error("--timing-samples must be a positive integer")
    try:
        _apply_nsys_pre_nvtx_cache_eviction(
            [], args.timing_cache_eviction_mib)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if args.end_idx and not args.start_idx:
        parser.error("--end-idx requires a positive --start-idx")

    if args.stage == "profile":
        if (args.timing_shard_run is not None
                or args.timing_retry_run is not None
                or args.timing_candidate_run is not None):
            parser.error(
                "timing shard options are valid only with --stage aggregate")
        if args.timing_samples is not None:
            parser.error("--timing-samples is not valid with --stage profile")
        if args.timing_cache_eviction_mib:
            parser.error(
                "--timing-cache-eviction-mib is only valid with --stage "
                "timing or timing aggregate")
        for kernel in kernels:
            step_1_run_profiling(
                cfg, kernel, dry_run=args.dry_run,
                max_tasks=args.max_tasks,
                target_vector_op=args.target_vector_op,
                profile_run=args.profile_run,
                target_flash_attn_reprofile=(
                    args.target_flash_attn_reprofile),
                start_idx=args.start_idx,
                end_idx=args.end_idx)
        return

    if args.stage == "timing":
        if (args.timing_shard_run is not None
                or args.timing_retry_run is not None
                or args.timing_candidate_run is not None):
            parser.error(
                "timing shard options are valid only with --stage aggregate")
        if args.profile_run is None:
            parser.error("--stage timing requires --profile-run")
        if args.timing_samples is None:
            parser.error("--stage timing requires --timing-samples")
        for kernel in kernels:
            timing_arguments = {
                "dry_run": args.dry_run,
                "max_tasks": args.max_tasks,
                "target_vector_op": args.target_vector_op,
                "target_flash_attn_reprofile": (
                    args.target_flash_attn_reprofile),
                "start_idx": args.start_idx,
                "end_idx": args.end_idx,
            }
            if args.timing_cache_eviction_mib:
                timing_arguments["timing_cache_eviction_mib"] = (
                    args.timing_cache_eviction_mib)
            step_run_nsys_timing(
                cfg, kernel, args.profile_run, args.timing_samples,
                **timing_arguments)
        return

    if args.dry_run:
        # convert/aggregate write real output
        # (CSV / kernel_dataset/<hw>/<kernel>.json.zst) and would clobber existing
        # data if allowed to run with --dry-run. Refuse instead of warning.
        parser.error(
            "--dry-run is only valid with --stage profile or timing")
    if args.max_tasks or args.start_idx or args.end_idx:
        parser.error(
            "profiling limits are only valid with --stage profile or timing")

    if args.stage == "convert":
        if (args.timing_shard_run is not None
                or args.timing_retry_run is not None
                or args.timing_candidate_run is not None):
            parser.error(
                "timing shard options are valid only with --stage aggregate")
        if args.timing_samples is not None:
            parser.error("--timing-samples is not valid with --stage convert")
        if args.timing_cache_eviction_mib:
            parser.error(
                "--timing-cache-eviction-mib is only valid with --stage "
                "timing or timing aggregate")
        if (args.target_vector_op is not None
                or args.target_flash_attn_reprofile):
            parser.error(
                "reprofile selectors are not valid with --stage convert")
        for kernel in kernels:
            step_2_convert_reps_to_csv(
                cfg, kernel, profile_run=args.profile_run)
        return

    if args.stage == "aggregate":
        if args.timing_shard_run is not None:
            if args.profile_run is not None:
                parser.error(
                    "--timing-shard-run cannot be mixed with --profile-run")
            if args.timing_samples is None:
                parser.error(
                    "--timing-shard-run requires --timing-samples")
            if args.timing_candidate_run is None:
                parser.error(
                    "--timing-shard-run requires --timing-candidate-run")
            if args.timing_cache_eviction_mib:
                parser.error(
                    "sharded NSYS timing aggregation requires ordinary warm "
                    "captures")
            if (args.target_vector_op is not None
                    or args.target_flash_attn_reprofile):
                parser.error(
                    "sharded timing aggregate requires every workload case")
            for kernel in kernels:
                step_aggregate_nsys_timing_shards(
                    cfg, kernel, args.timing_shard_run,
                    args.timing_candidate_run, args.timing_samples,
                    retry_profile_runs=args.timing_retry_run)
            return
        if args.timing_retry_run is not None:
            parser.error(
                "--timing-retry-run requires --timing-shard-run")
        if args.timing_candidate_run is not None:
            parser.error(
                "--timing-candidate-run requires --timing-shard-run")
        if args.timing_samples is not None and args.profile_run is None:
            parser.error(
                "--timing-samples aggregate selection requires --profile-run")
        if args.timing_samples is not None:
            if (args.target_vector_op is not None
                    or args.target_flash_attn_reprofile):
                parser.error(
                    "timing aggregate requires every workload case; capture "
                    "selectors are valid only with --stage timing")
            for kernel in kernels:
                if args.timing_cache_eviction_mib:
                    step_aggregate_nsys_timing(
                        cfg, kernel, args.profile_run, args.timing_samples,
                        args.timing_cache_eviction_mib)
                else:
                    step_aggregate_nsys_timing(
                        cfg, kernel, args.profile_run, args.timing_samples)
            return
        if args.timing_cache_eviction_mib:
            parser.error(
                "--timing-cache-eviction-mib requires --timing-samples "
                "during aggregate")
        if (args.profile_run is None
                and (args.target_vector_op is not None
                     or args.target_flash_attn_reprofile)):
            parser.error(
                "reprofile selectors require --profile-run during aggregate")
        for kernel in kernels:
            step_2_convert_reps_to_csv(
                cfg, kernel, profile_run=args.profile_run)
            step_3_aggregate_csv_to_json(
                cfg, kernel, profile_run=args.profile_run,
                target_vector_op=args.target_vector_op,
                target_flash_attn_reprofile=(
                    args.target_flash_attn_reprofile))
        return

    raise ValueError(f"Unexpected stage: {args.stage}")
