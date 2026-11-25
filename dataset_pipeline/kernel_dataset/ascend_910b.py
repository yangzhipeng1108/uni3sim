"""910B kernel profiling via legacy OPPROF or timing-only Timeline capture.

Replaces NCU (A800) / mcProfiler (C500). Uses CANN's `msprof op` (msopprof),
which profiles each AI-Core operator and emits per-block CSVs:
  OpBasicInfo, PipeUtilization, ArithmeticUtilization, Memory, L2Cache.
This builder aggregates them (mean over AI-Core blocks and invocations)
into the same NCU-compatible JSON schema the A800/C500 datasets use, so the
existing predictors/eval harness consume 910B as a third architecture.

Usage (from the repository root inside an Ascend runtime environment):
    python scripts/kernel_dataset/ascend_910b/build_dataset.py --config configs/dataset/ascend_910b.yaml --kernel gemm --stage profile
    python scripts/kernel_dataset/ascend_910b/build_dataset.py --config configs/dataset/ascend_910b.yaml --kernel gemm --stage aggregate
    python scripts/kernel_dataset/ascend_910b/build_dataset.py --config configs/dataset/ascend_910b.yaml --kernel gemm,flash_attn,fused_moe,vector --stage aggregate

Stage 'profile' runs each shape under the explicitly selected profiler mode.
Stage 'aggregate' parses OPPROF CSVs into per-kernel NCU-format JSON
(kernel_dataset/<hardware>/<kernel>.json.zst). Eval merges per-kernel documents in
memory via load_kernel_dataset — there is no separate merge stage.
Stage 'timeline-timing-candidate' parses exact cold-cache public-call schedules
into a candidate under profiler_output and never publishes canonical JSON.
Stage 'timeline-steady-timing-candidate' combines three independent ordinary-
warm Timeline windows per case into a timing-only candidate and never publishes
canonical JSON.

The conversion rules are documented next to the parsing functions below.
"""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import time
import warnings
from decimal import Decimal, InvalidOperation


from dataset_pipeline.kernel_dataset.config import (
    REPO_ROOT as PROJECT_ROOT,
    resolve_project_path,
    load_kernel_workload_config as _load_kernel_config,
    get_kernel_trace_dir,
    file_sha256 as _file_sha256,
    load_pipeline_config,
    load_yaml_mapping,
    get_kernel_pipeline_settings,
    get_hardware_config_path,
    get_kernel_config_path,
    get_kernel_script_path,
    get_profiler_output_dir,
    get_kernel_dataset_path,
    parse_kernel_list,
    require_mapping,
    require_non_negative_int,
    require_non_empty_list,
    require_positive_int,
    require_positive_number,
    require_profiler_executable,
    current_python_executable,
    to_float,
)
from dataset_pipeline.kernel_dataset.dataset_writer import (
    build_kernel_evaluation_timing,
    publish_complete_dataset,
    write_complete_candidate,
)
from dataset_pipeline.kernel_dataset.workload_shapes import (
    require_staging_baseline_keys,
    build_fused_moe_shapes as build_common_fused_moe_shapes,
    build_flash_attn_shapes as build_common_flash_attn_shapes,
    build_vector_shapes as build_common_vector_shapes,
    expand_kernel_shapes,
)
from src.kernel.dataset_storage import load_kernel_dataset_document


def _to_per_expert(tpe, tok, topk):
    """Normalize tok_per_expert to the canonical per-expert histogram.

    vLLM-Ascend's npu_grouped_matmul dumps group_list
    as a CUMULATIVE prefix sum (offsets); the predictor and LRU sim consume
    per-expert. Detect cumulative (monotonic non-decreasing with last element
    == tok*topk = total routed pairs) and unwrap via successive diffs; a genuine
    per-expert histogram is returned unchanged (its sum, not its last element,
    equals tok*topk)."""
    if not isinstance(tpe, list) or not tpe:
        raise ValueError("tok_per_expert must be a non-empty list")
    values = []
    for index, value in enumerate(tpe):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"tok_per_expert[{index}] must be an integer, "
                f"got {type(value).__name__}"
            )
        values.append(value)
    if any(value < 0 for value in values):
        raise ValueError("tok_per_expert values must be non-negative")
    counts = {"tok": tok, "topk": topk}
    tok = require_positive_int(counts, "tok", "MoE routing")
    topk = require_positive_int(counts, "topk", "MoE routing")
    n = len(values)
    monotonic = all(values[i] >= values[i - 1] for i in range(1, n))
    total = tok * topk
    is_cumulative = monotonic and values[-1] == total
    if is_cumulative:
        result = [values[0]] + [
            values[i] - values[i - 1] for i in range(1, n)]
    else:
        result = values
    if sum(result) != total:
        raise ValueError(
            f"tok_per_expert sums to {sum(result)}, expected {total}")
    return result


def _integer_measurement(value, context, *, positive):
    number = to_float(value)
    if not number.is_integer():
        raise ValueError(f"{context} must be an integer, got {number}")
    result = int(number)
    if positive and result <= 0:
        raise ValueError(f"{context} must be positive, got {result}")
    if not positive and result < 0:
        raise ValueError(f"{context} must be non-negative, got {result}")
    return result


def _decimal_measurement(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(
            f"{context} must be a decimal measurement, "
            f"got {type(value).__name__}")
    normalized = str(value).replace(",", "").strip()
    if normalized in ("", "NA", "N/A"):
        raise ValueError(f"{context} is missing")
    try:
        result = Decimal(normalized)
    except InvalidOperation as error:
        raise ValueError(
            f"{context} must be a decimal measurement, got {value!r}") \
            from error
    if not result.is_finite():
        raise ValueError(f"{context} must be finite, got {value!r}")
    return result


def _validate_profile_run_name(profile_run):
    if (not isinstance(profile_run, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", profile_run)
            is None):
        raise ValueError(
            "profile_run must contain only letters, digits, '.', '_' or '-'")


def _profile_run_trace_dir(cfg, kernel, profile_run):
    if profile_run is None:
        return get_kernel_trace_dir(cfg, kernel)
    _validate_profile_run_name(profile_run)
    return os.path.join(
        get_profiler_output_dir(cfg), "profile_runs", profile_run,
        f"{kernel}_traces")


# ─── Kernel shape builders (grids reused verbatim from A800/C500) ─────────────

def build_flash_attn_shapes(kernel_cfg):
    """Normalize the shared FA contract for a backend without public SP."""

    shapes = build_common_flash_attn_shapes(kernel_cfg)
    result = []
    for shape in shapes:
        attributes = dict(shape["attr"])
        if attributes["SP"] is not None:
            raise ValueError(
                "shared FLASHATTN shape must leave executed SP unresolved")
        attributes["SP_REQUESTED"] = None
        result.append({
            **shape,
            "attr": attributes,
        })
    if len({shape["filename"] for shape in result}) != len(result):
        raise ValueError(
            "dropping GPU-only SP produced duplicate Ascend FLASHATTN cases")
    return result


def build_vector_shapes(kernel_cfg):
    """Attach Ascend runtime kernel names to canonical vector shapes."""
    shapes = build_common_vector_shapes(kernel_cfg)
    target_kernels = {
        "rmsnorm": "RmsNorm",
        "add_rmsnorm": "AddRmsNorm",
        # OP-mode msprof reports the low-level AICPU/AIV kernel symbol
        # ("rotary_pos_emb_<id>"), not the ATB-level logical op name
        # ("AtbRopeKernel") that Timeline-mode summaries use for the same
        # dispatch. Verified on-device (msprof op, full launch-sequence dump,
        # device 5, 2026-09-22): the captured kernel is literally
        # "rotary_pos_emb_22". _trace_has_target_kernel does a startswith
        # prefix match, so the numeric ATB registration-id suffix is dropped.
        "rope": "rotary_pos_emb",
        "silu": "SwiGlu",
        "kvcache": "ReshapeAndCacheNdKernel",
    }
    for shape in shapes:
        operator = shape["attr"]["op"]
        if operator == "elementwise":
            secondary_op = shape["attr"]["op2"]
            if secondary_op == "add":
                shape["target_kernel"] = "Add"
            elif secondary_op == "mul":
                shape["target_kernel"] = "Mul"
            else:
                raise ValueError(
                    f"unknown Ascend elementwise operation: {secondary_op!r}")
        elif operator in target_kernels:
            shape["target_kernel"] = target_kernels[operator]
        else:
            raise ValueError(
                f"unknown Ascend vector operation: {operator!r}")
    return shapes


def build_fused_moe_shapes(kernel_cfg):
    return build_common_fused_moe_shapes(
        kernel_cfg,
        argument_names={
            "tokens": "--tok", "experts": "--experts",
            "top_k": "--topk", "hidden": "--hidden",
            "intermediate": "--intermediate",
        },
        include_layer=True,
    )
def build_shapes(kernel, kernel_cfg):
    return expand_kernel_shapes(
        kernel, kernel_cfg,
        {
            "flash_attn": build_flash_attn_shapes,
            "fused_moe": build_fused_moe_shapes,
            "vector": build_vector_shapes,
        },
        unknown_kernel_message="Unknown kernel '{kernel}'",
    )


def _select_profile_shapes(kernel, shapes, target_flash_attn_reprofile,
                           target_vector_op, selected_case_ids=None):
    if target_vector_op is not None and target_flash_attn_reprofile:
        raise ValueError(
            "vector and flash_attn reprofile selectors are mutually exclusive")
    if selected_case_ids is not None:
        if (target_vector_op is not None
                or target_flash_attn_reprofile):
            raise ValueError(
                "explicit case IDs cannot be mixed with family selectors")
        if not isinstance(selected_case_ids, list) or not selected_case_ids:
            raise ValueError("explicit case selection requires non-empty case IDs")
        if any(not isinstance(case_id, str) or not case_id
               for case_id in selected_case_ids):
            raise TypeError("explicit case IDs must be non-empty strings")
        if len(set(selected_case_ids)) != len(selected_case_ids):
            raise ValueError("explicit case IDs must be unique")
        available_case_ids = {shape["filename"] for shape in shapes}
        unknown_case_ids = sorted(set(selected_case_ids) - available_case_ids)
        if unknown_case_ids:
            raise ValueError(
                "explicit case IDs are not part of the selected workload: "
                f"kernel={kernel!r}, unknown={unknown_case_ids!r}")
        selected_case_id_set = set(selected_case_ids)
        shapes = [shape for shape in shapes
                  if shape["filename"] in selected_case_id_set]
    if target_flash_attn_reprofile:
        if kernel != "flash_attn":
            raise ValueError(
                "target_flash_attn_reprofile is only valid for flash_attn")
        shapes = [
            shape for shape in shapes
            if (shape["attr"]["Q_LEN"] > 1
                and shape["attr"]["kv_layout"] == "paged_varlen")
        ]
        if len(shapes) != 9:
            raise ValueError(
                "Ascend paged-varlen prefill correction set must contain "
                f"exactly 9 cases, got {len(shapes)}")
    if target_vector_op is not None:
        if kernel != "vector":
            raise ValueError(
                "target_vector_op is only valid for the vector workload")
        shapes = [
            shape for shape in shapes
            if shape["attr"]["op"] == target_vector_op
        ]
        if not shapes:
            raise ValueError(
                f"no vector profiling shapes match {target_vector_op!r}")
        if target_vector_op == "rope" and len(shapes) != 32:
            raise ValueError(
                "Ascend RoPE reprofile set must contain exactly 32 cases, "
                f"got {len(shapes)}")
    return shapes


# ─── Runner command construction ─────────────────────────────────────────────

def _flash_attn_workload_file(shape, out_dir, materialize):
    if out_dir is None:
        raise ValueError("flash_attn runner requires an output directory")
    attributes = shape["attr"]
    payload = {
        "q_lens": attributes["Q_LENS"],
        "kv_lens": attributes["KV_LENS"],
    }
    path = os.path.join(out_dir, "workload.json")
    if materialize:
        os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
    return path


def _runner_application(cfg, kernel, shape, out_dir=None,
                        materialize_contract=False,
                        profile_cache_eviction_mib=0,
                        profile_sweep_operation="fill",
                        timeline_public_call_count=None,
                        serving_exact_public_call=False):
    """Build the `--application` command string msprof op executes."""
    python_bin = current_python_executable()
    script = get_kernel_script_path(cfg, kernel)
    args = list(shape["script_args"])
    if kernel == "flash_attn":
        for flag in ("--q-lens-json", "--kv-lens-json"):
            if flag not in args:
                raise ValueError(
                    f"flash_attn shape is missing runner argument {flag}")
            index = args.index(flag)
            del args[index:index + 2]
        args += [
            "--workload-json",
            _flash_attn_workload_file(
                shape, out_dir, materialize_contract),
        ]
    if kernel == "fused_moe" and out_dir is not None:
        args += ["--tiling", os.path.join(out_dir, "tiling.json")]
    if serving_exact_public_call:
        if kernel != "fused_moe":
            raise ValueError(
                "serving-exact public-call mode is valid only for Ascend FusedMoE")
        args.append("--serving-exact-public-call")
    if profile_cache_eviction_mib != 0:
        args += ["--profile-cache-eviction-mib",
                 str(profile_cache_eviction_mib)]
        if kernel == "flash_attn" and profile_sweep_operation != "fill":
            args += ["--profile-sweep-operation", profile_sweep_operation]
    if timeline_public_call_count is not None:
        if kernel != "vector":
            raise ValueError(
                "timeline public-call count is valid only for Ascend Vector")
        if profile_cache_eviction_mib == 0:
            raise ValueError(
                "timeline public-call count requires cache eviction")
        args += ["--timeline-public-call-count",
                 str(timeline_public_call_count)]
    return f"{python_bin} {script} {' '.join(args)}"


PROFILE_MODES = ("op", "timeline")
VECTOR_PROFILE_MODES = PROFILE_MODES


def _validate_profile_mode(profile_mode):
    if profile_mode not in PROFILE_MODES:
        raise ValueError(
            "Ascend profile mode must be one of "
            f"{PROFILE_MODES!r}, got {profile_mode!r}")


def _uses_timeline_profile(kernel, shape, profile_mode):
    """Select the profiler path strictly from the requested ``profile_mode``.

    This used to unconditionally route every RoPE vector shape to Timeline
    whenever ``op`` was requested, on the unverified assumption that the ATB
    RoPE runtime is invisible to ``msprof op``. That assumption was never
    tested against real OPPROF hardware capture and is not a physical or
    data constraint -- it was an override inside this module. RoPE now goes
    through the same OPPROF path as every other vector op when ``op`` is
    requested; ``kernel``/``shape`` are accepted for call-site symmetry with
    other per-shape dispatch but no longer change the routing. If OPPROF
    genuinely produces no capture for a shape, the caller (``_run_profile``'s
    resumable-capture check, ``stage_aggregate``'s ``OPPROF_*`` glob) fails
    closed with ``FileNotFoundError``/``ValueError`` rather than silently
    substituting Timeline.
    """
    _validate_profile_mode(profile_mode)
    del kernel, shape
    return profile_mode == "timeline"




def _timeline_summary_paths(output_dir):
    return glob.glob(os.path.join(
        output_dir, "PROF_*", "mindstudio_profiler_output", "op_summary_*.csv"))


def _timeline_has_target_kernel(output_dir, target_kernel):
    for path in _timeline_summary_paths(output_dir):
        if any(target_kernel in row["Op Name"]
               for row in _read_csv_rows(path)):
            return True
    return False


def _profiler_environment(profile_dir, device_id):
    """Build an isolated CANN environment for one profiler process."""
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
    cache_root = os.path.abspath(
        os.path.join(profile_dir, f".ascend_cache_dev{device_id}"))
    os.makedirs(cache_root, exist_ok=True)
    env["ASCEND_CACHE_PATH"] = cache_root
    env["ASCEND_WORK_PATH"] = cache_root
    return env


def _profile_timeline_shape(cfg, kernel, shape, output_dir, profile_dir,
                            device_id, dry_run, index, total,
                            profile_cache_eviction_mib,
                            exact_public_calls, timeline_duration_seconds):
    """Capture one exact public call schedule using Timeline evidence."""
    pipeline_settings = get_kernel_pipeline_settings(cfg)
    timeline_public_call_count = None
    if kernel == "vector" and profile_cache_eviction_mib != 0:
        timeline_public_call_count = _TIMELINE_CALL_COUNTS["vector"]
    msprof_bin = require_profiler_executable("msprof", "MSPROF_BIN")
    app = _runner_application(
        cfg, kernel, shape, output_dir, materialize_contract=True,
        profile_cache_eviction_mib=profile_cache_eviction_mib,
        timeline_public_call_count=timeline_public_call_count,
        serving_exact_public_call=(kernel == "fused_moe"))
    if timeline_duration_seconds is None:
        duration_seconds = require_positive_int(
            pipeline_settings, "timeline_duration_seconds",
            "kernel pipeline settings")
    elif (isinstance(timeline_duration_seconds, bool)
            or not isinstance(timeline_duration_seconds, int)
            or timeline_duration_seconds <= 0):
        raise ValueError(
            "Ascend Timeline duration override must be a positive integer")
    else:
        duration_seconds = timeline_duration_seconds
    cmd = [
        msprof_bin,
        f"--application={app}",
        "--ai-core=on",
        f"--aic-metrics={pipeline_settings['timeline_aic_metrics']}",
        f"--duration={duration_seconds}",
        f"--output={output_dir}",
    ]
    if dry_run:
        print(f"  [{index}/{total}] DRY-RUN TIMELINE: {shape['filename']}  ({app})")
        return

    if exact_public_calls:
        existing_capture = _timeline_capture_is_complete(
            output_dir, kernel, shape, profile_cache_eviction_mib)
    else:
        existing_capture = _timeline_has_target_kernel(
            output_dir, shape["target_kernel"])
    if existing_capture:
        print(f"  [{index}/{total}] {shape['filename']} ... SKIP (timeline exists)")
        return
    for stale in glob.glob(os.path.join(output_dir, "PROF_*")):
        shutil.rmtree(stale)

    os.makedirs(output_dir, exist_ok=True)
    print(f"  [{index}/{total}] {shape['filename']} [TIMELINE] ... ",
          end="", flush=True)
    started = time.time()
    log_path = os.path.join(output_dir, "msprof_log.txt")
    try:
        with open(log_path, "w") as log_file:
            subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=1200,
                env=_profiler_environment(profile_dir, device_id),
                check=True,
            )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            f"msprof timeline timed out for {shape['filename']}") from error
    if exact_public_calls:
        if profile_cache_eviction_mib == 0:
            _timeline_steady_state_public_call_measurement(
                output_dir, kernel, shape)
        else:
            _timeline_public_call_measurement(output_dir, kernel, shape)
    elif not _timeline_has_target_kernel(output_dir, shape["target_kernel"]):
        raise RuntimeError(
            f"msprof timeline did not capture {shape['target_kernel']!r} "
            f"for {shape['filename']}")
    print(f"OK ({time.time() - started:.1f}s)")


# ─── Stage: profile ──────────────────────────────────────────────────────────

def stage_profile(cfg, kernel, dry_run=False, max_tasks=0,
                  start_idx=0, end_idx=0, device_id=0, profile_run=None,
                  target_flash_attn_reprofile=False, target_vector_op=None,
                  selected_case_ids=None,
                  profile_cache_eviction_mib=0,
                  profile_sweep_operation="fill",
                  profile_mode="op", vector_profile_mode=None,
                  timeline_duration_seconds=None):
    if vector_profile_mode is not None:
        if profile_mode != "op" and profile_mode != vector_profile_mode:
            raise ValueError(
                "conflicting profile_mode and vector_profile_mode")
        profile_mode = vector_profile_mode
    _validate_profile_mode(profile_mode)
    if (profile_cache_eviction_mib != 0
            and profile_cache_eviction_mib < 1536):
        raise ValueError(
            "Ascend profile cache eviction must be zero or at least 1536 MiB")
    kernel_cfg = _load_kernel_config(
        get_kernel_config_path(cfg, kernel), kernel)
    if (profile_mode == "timeline"
            and profile_cache_eviction_mib not in (0, 2048)):
        raise ValueError(
            "Ascend Timeline capture requires either 0 MiB ordinary warm "
            "state or exactly 2048 MiB eviction before every public call")
    if timeline_duration_seconds is not None:
        if profile_mode != "timeline":
            raise ValueError(
                "Ascend Timeline duration override requires Timeline mode")
        if (isinstance(timeline_duration_seconds, bool)
                or not isinstance(timeline_duration_seconds, int)
                or timeline_duration_seconds <= 0):
            raise ValueError(
                "Ascend Timeline duration override must be a positive integer")
    if profile_mode == "timeline" and profile_run is None:
        raise ValueError(
            "Ascend Timeline capture requires a named --profile-run so raw "
            "evidence cannot be confused with legacy canonical traces")
    shapes = build_shapes(kernel, kernel_cfg)
    shapes = _select_profile_shapes(
        kernel, shapes, target_flash_attn_reprofile, target_vector_op,
        selected_case_ids)
    if max_tasks and max_tasks > 0:
        shapes = shapes[:max_tasks]
    if start_idx > 0:
        end = end_idx if end_idx > 0 else len(shapes)
        shapes = shapes[start_idx - 1:end]

    pipeline_settings = get_kernel_pipeline_settings(cfg)
    msprof_bin = require_profiler_executable("msprof", "MSPROF_BIN")
    aic_metrics = pipeline_settings["aic_metrics"]
    warmup = require_non_negative_int(
        pipeline_settings, "warmup", "kernel pipeline settings")
    if kernel == "fused_moe":
        launch_count = require_positive_int(
            pipeline_settings, "moe_launch_count", "kernel pipeline settings")
        warmup = require_non_negative_int(
            pipeline_settings, "moe_warmup_skip", "kernel pipeline settings")
    elif kernel in ("gemm", "flash_attn", "vector"):
        launch_count = 3
    else:
        raise ValueError(f"unsupported Ascend kernel: {kernel!r}")

    # FA: profile ONLY the fused-attention kernel via --kernel-name. The paged/
    # varlen/mla runners build large KV pools + a scattered block_table whose
    # setup launches (torch.randn → DSARandomNormal/Cast, the B-loop scatter →
    # Cast) outnumber a fixed launch-skip offset, so op-mode with
    # --launch-skip-before-match alone grabbed Cast (the prior 67-garbage bug —
    # setup launches from torch.randn/Cast in the B-loop scatter outnumber any
    # fixed offset). --kernel-name matches on the
    # FusedInferAttentionScore prefix (v1 contiguous/paged/varlen AND v2 MLA both
    # resolve to "FusedInferAttentionScore_..._mix_aic"), cleanly skipping ALL
    # setup launches regardless of count. With the name filter doing the skip,
    # launch-skip-before-match is a small cold-start guard (skip the first 2
    # warmup fa() launches, capture the next 3 — all post-compile, identical).
    # Verified on-device for contiguous/paged/paged_varlen/mla × decode/prefill.
    kernel_name_filter = pipeline_settings["kernel_name_filter"][kernel]
    if kernel == "flash_attn":
        if kernel_name_filter != "FusedInferAttentionScore":
            raise ValueError(
                "Ascend flash_attn requires the FusedInferAttentionScore "
                "kernel-name filter")
        warmup = require_non_negative_int(
            pipeline_settings, "fa_launch_skip", "kernel pipeline settings")
    elif kernel == "vector":
        # Vector runners do only 6 launches (5 warmup + 1). rmsnorm/add_rmsnorm/
        # silu/kvcache/elementwise emit only a handful of one-time setup
        # launches (torch.randn/arange/zeros → DSARandomNormal/Cast/Range/
        # ZerosLike), so a shared launch-skip with no --kernel-name filter
        # clears setup fine for them. RoPE is the outlier: building
        # theta/freqs/cos_sin_cache once per process emits ~35 one-time setup
        # launches (BroadcastTo/Cast/ConcatD/Cos/GatherV3/Mul/Pow/Range/
        # RealDiv/Reciprocal/RepeatInterleaveV2/Sin/Slice/Tile — verified
        # on-device, device 5, 2026-09-22), far beyond any fixed skip that
        # still works for the other vector ops without also over-skipping
        # them. RoPE therefore gets its own --kernel-name=rotary_pos_emb
        # filter (mirroring the FA precedent above) plus a dedicated
        # rope_launch_skip, applied per-shape below; vector_launch_skip stays
        # the shared default for every other vector op.
        warmup = require_non_negative_int(
            pipeline_settings, "vector_launch_skip", "kernel pipeline settings")
        rope_warmup = require_non_negative_int(
            pipeline_settings, "rope_launch_skip", "kernel pipeline settings")
    elif kernel in ("gemm", "fused_moe"):
        pass
    else:
        raise ValueError(f"unsupported Ascend kernel: {kernel!r}")

    trace_dir = _profile_run_trace_dir(cfg, kernel, profile_run)
    os.makedirs(trace_dir, exist_ok=True)

    if profile_mode == "timeline":
        profiler_mode = "msprof Timeline Task Duration"
    elif profile_mode == "op":
        profiler_mode = "msprof op"
    else:
        raise AssertionError(
            f"unhandled Ascend profile mode: {profile_mode!r}")
    print(f">>> Stage: Profile [{kernel}], {len(shapes)} shapes -> {trace_dir}")
    print(f"    {profiler_mode}, device {device_id}, metrics={aic_metrics}, "
          f"launch_count={launch_count}"
          + (f", kernel-name={kernel_name_filter}" if kernel_name_filter else ""))

    for i, shape in enumerate(shapes, 1):
        out_dir = os.path.join(trace_dir, shape["filename"])
        if _uses_timeline_profile(kernel, shape, profile_mode):
            if profile_mode == "timeline":
                exact_public_calls = True
            elif profile_mode == "op":
                exact_public_calls = False
            else:
                raise AssertionError(
                    f"unhandled Ascend profile mode: {profile_mode!r}")
            _profile_timeline_shape(
                cfg, kernel, shape, out_dir, trace_dir, device_id, dry_run,
                i, len(shapes), profile_cache_eviction_mib,
                exact_public_calls=exact_public_calls,
                timeline_duration_seconds=timeline_duration_seconds)
            continue
        # Capture filtering is FA-only, but every resumable trace must have an
        # explicit target prefix. This prevents a stale setup-only capture from
        # being accepted for GEMM, FusedMoE, or vector operators.
        capture_filter = kernel_name_filter
        effective_warmup = warmup
        if kernel == "flash_attn":
            target_kernel_prefix = "FusedInferAttentionScore"
        elif kernel == "vector":
            target_kernel_prefix = shape["target_kernel"]
            if shape["attr"]["op"] == "rope":
                # See the rope_launch_skip comment above: RoPE's one-time
                # setup phase is too large for the shared vector_launch_skip,
                # so it gets its own --kernel-name filter + skip value.
                capture_filter = "rotary_pos_emb"
                effective_warmup = rope_warmup
        elif kernel == "gemm":
            target_kernel_prefix = "MatMul"
        elif kernel == "fused_moe":
            target_kernel_prefix = "GroupedMatmul"
        else:
            raise ValueError(f"unsupported Ascend kernel: {kernel!r}")
        if dry_run:
            print(f"  [{i}/{len(shapes)}] DRY-RUN: {shape['filename']}  ({_runner_application(cfg, kernel, shape, out_dir, profile_cache_eviction_mib=profile_cache_eviction_mib, profile_sweep_operation=profile_sweep_operation)})")
            continue

        # Resumable: skip only if a PREVIOUS run left parsable CSVs. An empty
        # OPPROF dir (failed profile) is cleared and retried. For flash_attn
        # ALSO verify the captured kernel is FusedInferAttentionScore (the target)
        # — a prior op-mode run WITHOUT --kernel-name may have left Cast/setup
        # CSVs (the 67-garbage bug); those are bad data, cleared and re-profiled
        # rather than skipped. Kernel-name verification is a collection red line
        # (every FA entry's primary kernel must be FusedInferAttentionScore,
        # not a setup Cast; asserted post-hoc in stage_aggregate).
        existing_csv = glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "*.csv"))
        if existing_csv and _trace_has_target_kernel(
            out_dir, target_kernel_prefix,
        ):
            print(f"  [{i}/{len(shapes)}] {shape['filename']} ... SKIP (CSVs exist)")
            continue
        for stale in glob.glob(os.path.join(out_dir, "OPPROF_*")):
            shutil.rmtree(stale)

        os.makedirs(out_dir, exist_ok=True)
        app = _runner_application(
            cfg, kernel, shape, out_dir, materialize_contract=True,
            profile_cache_eviction_mib=profile_cache_eviction_mib,
            profile_sweep_operation=profile_sweep_operation)
        cmd = [
            msprof_bin, "op",
            "--aic-metrics=" + aic_metrics,
            f"--launch-count={launch_count}",
            f"--launch-skip-before-match={effective_warmup}",
            f"--output={out_dir}",
            f"--application={app}",
        ]
        if capture_filter is not None:
            cmd.append(f"--kernel-name={capture_filter}")
        # Per-device isolated CANN op-compile / work cache. Without this, N
        # concurrent msprof-op processes (fan-out) collide on the shared GE
        # compiler cache and fail with "GEInitialize failed / OpsManager
        # initialize failed" (empty OPPROF). MoE needs GE op compilation (unlike
        # precompiled aclnn matmul/FA), so isolate per device.
        env = _profiler_environment(trace_dir, device_id)

        print(f"  [{i}/{len(shapes)}] {shape['filename']} ... ", end="", flush=True)
        t0 = time.time()
        log_path = os.path.join(out_dir, "msprof_log.txt")
        try:
            with open(log_path, "w") as logf:
                subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                               timeout=1200, env=env, check=True)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(
                f"msprof op timed out for {shape['filename']}") from error
        found = glob.glob(os.path.join(out_dir, "OPPROF_*"))
        if not found:
            raise RuntimeError(
                f"msprof op produced no OPPROF for {shape['filename']}")
        if not _trace_has_target_kernel(out_dir, target_kernel_prefix):
            raise RuntimeError(
                f"msprof op captured no {target_kernel_prefix!r} kernel for "
                f"{shape['filename']}; see {log_path}")
        print(f"OK ({time.time()-t0:.1f}s)")


# ─── Stage: aggregate — parse msprof CSVs ────────────────────────────────────

def _read_csv_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None or not fieldnames:
            raise ValueError(f"msprof CSV has no header: {path}")
        has_empty_trailer = (
            fieldnames[-1] == "" and "" not in fieldnames[:-1])
        if has_empty_trailer:
            fieldnames = fieldnames[:-1]
        if any(not isinstance(name, str) or not name.strip()
               for name in fieldnames):
            raise ValueError(f"msprof CSV has an empty header field: {path}")
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(f"msprof CSV has duplicate header fields: {path}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(
                    f"msprof CSV row has extra columns at {path}:{line_number}")
            if has_empty_trailer:
                trailer = row[""]
                if trailer not in (None, ""):
                    raise ValueError(
                        "msprof CSV unnamed trailing column contains data at "
                        f"{path}:{line_number}")
                del row[""]
            missing = [name for name in fieldnames if row[name] is None]
            if missing:
                raise ValueError(
                    f"msprof CSV row is missing {missing} at "
                    f"{path}:{line_number}")
            rows.append(row)
        return rows


def _rows_to_block_table(rows):
    """Return a loss-free, column-oriented view of msprof per-block CSV rows.

    Every column (block_id, sub_block_id, every metric) is kept as the
    literal string cell read from the CSV -- no aggregation, no type
    coercion, no dropped rows or columns -- so a reader can reconstruct the
    exact per-block table (goal ②: msprof's per-(block_id, sub_block_id)
    layer, currently collapsed by the SUM/MEAN distillate in `raw`).
    """
    columns = list(rows[0].keys())
    for row in rows[1:]:
        if list(row.keys()) != columns:
            raise ValueError(
                "msprof block CSV rows have inconsistent columns")
    return {
        "columns": columns,
        "rows": [[row[c] for c in columns] for row in rows],
    }


def _trace_has_target_kernel(out_dir, target_kernel_prefix):
    """Verify the captured trace contains the TARGET kernel (kernel-name guard).

    For flash_attn, a prior op-mode run without --kernel-name may have left
    Cast/setup CSVs (the 67-garbage bug). Returns True only if some OpBasicInfo
    CSV row's Op Name starts with the operator-specific target prefix.
    """
    if not isinstance(target_kernel_prefix, str) or not target_kernel_prefix:
        raise ValueError("target kernel prefix must be a non-empty string")
    for path in glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv")):
        for row in _read_csv_rows(path):
            name = row["Op Name"].strip()
            if name.startswith(target_kernel_prefix):
                return True
    return False


def _numeric_values(rows, col):
    """Parse measured values while preserving profiler NA as absence."""
    vals = []
    for r in rows:
        v = r[col].strip()
        if v in ("", "NA", "N/A"):
            continue
        try:
            vals.append(to_float(v))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid numeric value for msprof column {col!r}: {v!r}") \
                from error
    return vals


def _mean(rows, col):
    vals = _numeric_values(rows, col)
    if not vals:
        raise ValueError(f"msprof column {col!r} has no measured values")
    return sum(vals) / len(vals)


def _mean_inactive_zero(rows, col):
    """Treat an explicitly all-NA execution-pipe metric as inactive, not missing."""
    vals = _numeric_values(rows, col)
    return sum(vals) / len(vals) if vals else 0.0


def _pure_profiler_domain(op_type):
    """Return the sole execution domain for a non-mixed msprof kernel."""
    if op_type == "cube":
        return "aic"
    elif op_type == "vector":
        return "aiv"
    else:
        raise ValueError(
            f"msprof Op Type has no single execution domain: "
            f"{op_type!r}")


def _sum(rows, col):
    vals = _numeric_values(rows, col)
    if not vals:
        raise ValueError(f"msprof column {col!r} has no measured values")
    return sum(vals)


def _sum_inactive_zero(rows, col):
    """Treat an explicitly all-NA execution-pipe count as inactive."""
    return sum(_numeric_values(rows, col))


_MSPROF_NA_VALUES = ("", "NA", "N/A")


def _optional_msprof_number(row, column, context):
    value = row[column].strip()
    if value in _MSPROF_NA_VALUES:
        return None
    try:
        result = to_float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"invalid numeric value for {context} column {column!r}: "
            f"{value!r}") from error
    if not math.isfinite(result):
        raise ValueError(
            f"non-finite value for {context} column {column!r}: {value!r}")
    return result


def _required_non_negative_msprof_number(row, column, context):
    value = _optional_msprof_number(row, column, context)
    if value is None:
        raise ValueError(
            f"missing {context} column {column!r} for active domain")
    if value < 0:
        raise ValueError(
            f"negative {context} column {column!r}: {value}")
    return value


def _inactive_msprof_column(row, column, context):
    if _optional_msprof_number(row, column, context) is not None:
        raise ValueError(
            f"inactive {context} column {column!r} must be NA")


class MsprofActiveCyclesDropout(ValueError):
    """A single per-core row reports a zero/negative active-cycle counter.

    Distinct from the general "wrong number of active domains" ValueError
    above: this fires only when the row unambiguously has exactly one active
    domain (the XOR check already passed) and every other per-core counter in
    that same row is a normal, non-NA value -- i.e. the domain genuinely ran,
    only its cycle-count counter came back non-positive. Observed on real
    910B captures as an isolated single-invocation glitch (co-located
    aiv_mte2/mte3 instruction counts, bytes and active-bandwidth fields in the
    same row are all physically plausible and non-zero), never as a
    within-invocation pattern spanning multiple rows. Callers that repeat the
    same kernel launch across several invocations (_parse_kernel_dir) may
    choose to drop just the affected invocation when a majority of the
    kernel's other invocations parse cleanly; every other caller must treat
    this the same as any other ValueError.
    """


def _index_msprof_domain_rows(rows, table_name):
    """Index per-core rows and prove that every row has exactly one domain."""

    indexed = {}
    for row in rows:
        block_id = row["block_id"].strip()
        sub_block_id = row["sub_block_id"].strip()
        if (not block_id or block_id in _MSPROF_NA_VALUES
                or not sub_block_id or sub_block_id in _MSPROF_NA_VALUES):
            raise ValueError(f"{table_name} row has an empty execution key")
        key = (block_id, sub_block_id)
        if key in indexed:
            raise ValueError(f"duplicate {table_name} execution key {key!r}")
        aic_cycles = _optional_msprof_number(
            row, "aic_total_cycles", table_name)
        aiv_cycles = _optional_msprof_number(
            row, "aiv_total_cycles", table_name)
        if (aic_cycles is None) == (aiv_cycles is None):
            raise ValueError(
                f"{table_name} row {key!r} must have exactly one active "
                "AIC/AIV domain")
        active_cycles = aic_cycles if aic_cycles is not None else aiv_cycles
        if active_cycles <= 0:
            raise MsprofActiveCyclesDropout(
                f"{table_name} row {key!r} has non-positive active cycles")
        domain = "aic" if aic_cycles is not None else "aiv"
        indexed[key] = (domain, row)
    return indexed


def _aggregate_msprof_memory_rows(
        memory_rows, l2_rows, op_type, block_dim, mix_block_dim):
    """Join per-core Memory/L2 rows and aggregate traffic by active domain."""

    memory = _index_msprof_domain_rows(memory_rows, "Memory")
    l2 = _index_msprof_domain_rows(l2_rows, "L2Cache")
    if set(memory) != set(l2):
        missing_l2 = sorted(set(memory) - set(l2))
        missing_memory = sorted(set(l2) - set(memory))
        raise ValueError(
            "Memory/L2Cache execution-key mismatch: "
            f"missing_l2={missing_l2!r}, missing_memory={missing_memory!r}")

    if op_type == "cube":
        expected_counts = {"aic": block_dim, "aiv": 0}
        legacy_domain = _pure_profiler_domain(op_type)
    elif op_type == "vector":
        expected_counts = {"aic": 0, "aiv": block_dim}
        legacy_domain = _pure_profiler_domain(op_type)
    elif op_type == "mix":
        if mix_block_dim <= 0:
            raise ValueError("mixed msprof kernel requires positive Mix Block Dim")
        expected_counts = {"aic": block_dim, "aiv": mix_block_dim}
        # The legacy scalar fields represent the AIC side of a mixed launch.
        # Domain-specific rates remain in raw and must not be interpreted as
        # whole-kernel bandwidth.
        legacy_domain = "aic"
    else:
        raise ValueError(f"unsupported msprof Op Type: {op_type!r}")

    totals = {
        domain: {
            "count": 0,
            "read_kb": 0.0,
            "write_kb": 0.0,
            "dram_read_kb": 0.0,
            "dram_write_kb": 0.0,
            "read_bw_sum": 0.0,
            "write_bw_sum": 0.0,
            "mte2_instructions": 0.0,
            "gm_to_l1_kb": 0.0,
        }
        for domain in ("aic", "aiv")
    }
    for key in sorted(memory):
        domain, memory_row = memory[key]
        l2_domain, l2_row = l2[key]
        if domain != l2_domain:
            raise ValueError(
                f"Memory/L2Cache domain mismatch for execution key {key!r}: "
                f"{domain!r} != {l2_domain!r}")
        if domain == "aic":
            inactive_domain = "aiv"
        elif domain == "aiv":
            inactive_domain = "aic"
        else:
            raise AssertionError(
                f"unhandled msprof execution domain: {domain!r}")
        read_kb = _required_non_negative_msprof_number(
            memory_row, "read_main_memory_datas(KB)", "Memory")
        write_kb = _required_non_negative_msprof_number(
            memory_row, "write_main_memory_datas(KB)", "Memory")
        read_hit = _required_non_negative_msprof_number(
            l2_row, f"{domain}_read_hit_rate(%)", "L2Cache")
        write_hit = _required_non_negative_msprof_number(
            l2_row, f"{domain}_write_hit_rate(%)", "L2Cache")
        total_hit = _required_non_negative_msprof_number(
            l2_row, f"{domain}_total_hit_rate(%)", "L2Cache")
        if max(read_hit, write_hit, total_hit) > 100:
            raise ValueError(
                f"L2Cache row {key!r} has a hit rate above 100 percent")
        for metric in ("read", "write", "total"):
            _inactive_msprof_column(
                l2_row, f"{inactive_domain}_{metric}_hit_rate(%)",
                "L2Cache")

        read_bw = _required_non_negative_msprof_number(
            memory_row, f"{domain}_main_mem_read_bw(GB/s)", "Memory")
        write_bw = _required_non_negative_msprof_number(
            memory_row, f"{domain}_main_mem_write_bw(GB/s)", "Memory")
        mte2 = _required_non_negative_msprof_number(
            memory_row, f"{domain}_mte2_instructions", "Memory")
        for metric in ("read", "write"):
            _inactive_msprof_column(
                memory_row,
                f"{inactive_domain}_main_mem_{metric}_bw(GB/s)",
                "Memory")
        _inactive_msprof_column(
            memory_row, f"{inactive_domain}_mte2_instructions", "Memory")

        if domain == "aic":
            gm_to_l1_kb = _required_non_negative_msprof_number(
                memory_row, "GM_to_L1_datas(KB)", "Memory")
            _inactive_msprof_column(
                memory_row, "GM_to_UB_datas(KB)", "Memory")
            _inactive_msprof_column(
                memory_row, "UB_to_GM_datas(KB)", "Memory")
        elif domain == "aiv":
            gm_to_l1_kb = 0.0
            _inactive_msprof_column(
                memory_row, "GM_to_L1_datas(KB)", "Memory")
            _required_non_negative_msprof_number(
                memory_row, "GM_to_UB_datas(KB)", "Memory")
            _required_non_negative_msprof_number(
                memory_row, "UB_to_GM_datas(KB)", "Memory")
        else:
            raise AssertionError(f"unhandled msprof domain: {domain!r}")

        domain_totals = totals[domain]
        domain_totals["count"] += 1
        domain_totals["read_kb"] += read_kb
        domain_totals["write_kb"] += write_kb
        domain_totals["dram_read_kb"] += read_kb * (1.0 - read_hit / 100.0)
        domain_totals["dram_write_kb"] += (
            write_kb * (1.0 - write_hit / 100.0))
        domain_totals["read_bw_sum"] += read_bw
        domain_totals["write_bw_sum"] += write_bw
        domain_totals["mte2_instructions"] += mte2
        domain_totals["gm_to_l1_kb"] += gm_to_l1_kb

    actual_counts = {
        domain: int(totals[domain]["count"])
        for domain in ("aic", "aiv")
    }
    if actual_counts != expected_counts:
        raise ValueError(
            f"msprof execution-domain row counts disagree with launch: "
            f"expected={expected_counts!r}, actual={actual_counts!r}")

    l2_read_kb = sum(totals[d]["read_kb"] for d in ("aic", "aiv"))
    l2_write_kb = sum(totals[d]["write_kb"] for d in ("aic", "aiv"))
    dram_read_kb = sum(
        totals[d]["dram_read_kb"] for d in ("aic", "aiv"))
    dram_write_kb = sum(
        totals[d]["dram_write_kb"] for d in ("aic", "aiv"))

    def _effective_hit_rate(request_kb, miss_kb):
        if request_kb == 0.0:
            return 0.0
        return 100.0 * (1.0 - miss_kb / request_kb)

    selected = totals[legacy_domain]
    selected_count = int(selected["count"])
    if selected_count <= 0:
        raise ValueError(
            f"msprof kernel has no rows for legacy {legacy_domain} domain")
    total_request_kb = l2_read_kb + l2_write_kb
    total_miss_kb = dram_read_kb + dram_write_kb
    return {
        "l2_read_bytes": l2_read_kb * 1024.0,
        "l2_write_bytes": l2_write_kb * 1024.0,
        "read_bytes": dram_read_kb * 1024.0,
        "write_bytes": dram_write_kb * 1024.0,
        "l2_read_hit_rate": _effective_hit_rate(l2_read_kb, dram_read_kb),
        "l2_write_hit_rate": _effective_hit_rate(
            l2_write_kb, dram_write_kb),
        "l2_hit_rate": _effective_hit_rate(total_request_kb, total_miss_kb),
        "main_mem_read_bw": selected["read_bw_sum"] / selected_count,
        "main_mem_write_bw": selected["write_bw_sum"] / selected_count,
        "gm_to_l1_kb": (
            totals["aic"]["gm_to_l1_kb"] / actual_counts["aic"]
            if actual_counts["aic"] else 0.0),
        "gm_to_l1_bytes": totals["aic"]["gm_to_l1_kb"] * 1024.0,
        "mte2_tile_count": selected["mte2_instructions"],
    }


def _parse_timeline_kernel(output_dir, target_kernel):
    """Parse the newest timeline summary into the common aggregate shape."""
    summaries = _timeline_summary_paths(output_dir)
    if not summaries:
        raise FileNotFoundError(
            f"no msprof timeline summary found under {output_dir}")
    summary = max(summaries, key=os.path.getmtime)
    rows = [row for row in _read_csv_rows(summary)
            if target_kernel in row["Op Name"]]
    if not rows:
        raise ValueError(
            f"timeline summary {summary} does not contain {target_kernel!r}")

    first = rows[0]

    def _required_integer(column):
        value = first[column].strip()
        if value in ("", "NA", "N/A"):
            raise ValueError(
                f"timeline column {column!r} is missing for {target_kernel!r}")
        return _integer_measurement(
            value, f"timeline column {column!r}", positive=True)

    def _inactive_integer_zero(column):
        value = first[column].strip()
        if value in ("", "NA", "N/A"):
            return 0
        return _integer_measurement(
            value, f"timeline column {column!r}", positive=False)

    def _mean_one_of(*columns):
        present = [column for column in columns if column in first]
        if len(present) != 1:
            raise ValueError(
                "timeline summary must contain exactly one of "
                f"{columns!r} for {target_kernel!r}, got {present!r}")
        return _mean(rows, present[0])

    def _optional_mean(column):
        if column not in first:
            return "NA"
        values = _numeric_values(rows, column)
        if not values:
            return "NA"
        return sum(values) / len(values)

    duration_us = _mean(rows, "Task Duration(us)")
    if "Current Freq" in first:
        frequency_mhz = _mean(rows, "Current Freq")
    elif {"aiv_total_cycles", "aiv_time(us)"}.issubset(first):
        frequencies = []
        for row in rows:
            cycles = to_float(row["aiv_total_cycles"])
            active_time_us = to_float(row["aiv_time(us)"])
            if cycles <= 0 or active_time_us <= 0:
                raise ValueError(
                    "timeline AIV cycles and active time must be positive for "
                    f"{target_kernel!r}")
            frequencies.append(cycles / active_time_us)
        frequency_mhz = sum(frequencies) / len(frequencies)
    else:
        raise ValueError(
            "timeline summary must provide Current Freq or AIV cycle/time "
            f"columns for {target_kernel!r}")
    if duration_us <= 0 or frequency_mhz <= 0:
        raise ValueError(
            f"invalid timeline duration/frequency for {target_kernel!r}: "
            f"duration_us={duration_us}, frequency_mhz={frequency_mhz}")

    aggregate = {
        "op_name": target_kernel,
        "op_type": "vector",
        "duration_us": duration_us,
        "block_dim": _required_integer("Block Dim"),
        "mix_block_dim": _inactive_integer_zero("Mix Block Dim"),
        "cur_freq_mhz": frequency_mhz,
        "aic_total_cycles": duration_us * frequency_mhz,
        "aic_scalar_time_ns": _mean(rows, "aic_scalar_time(us)") * 1000.0,
        "aic_cube_time_ns": _mean_one_of(
            "aic_cube_time(us)", "aic_mac_time(us)") * 1000.0,
        "aic_mte2_time_ns": _mean(rows, "aic_mte2_time(us)") * 1000.0,
        "aiv_vec_time_ns": _mean(rows, "aiv_vec_time(us)") * 1000.0,
        "aiv_scalar_time_ns": _mean(rows, "aiv_scalar_time(us)") * 1000.0,
        "read_bytes": 0.0,
        "write_bytes": 0.0,
        "l2_read_bytes": 0.0,
        "l2_write_bytes": 0.0,
        "l2_hit_rate": 0.0,
        "l2_read_hit_rate": 0.0,
        "cube_ratio": 0.0,
        "vec_ratio": 0.0,
        "aic_scalar_ratio": 0.0,
        "cube_fops": 0.0,
        "main_mem_read_bw": 0.0,
        "main_mem_write_bw": 0.0,
        "gm_to_l1_bytes": 0.0,
        "mte2_tile_count": 0.0,
        # Timeline-only captures do not have the detailed per-invocation CSV
        # family parsed by _parse_kernel_invocation.  Preserve the independent
        # AIV domain envelope and transfer-pipe times explicitly so Oracle can
        # replace like-for-like formula intermediates without using Task
        # Duration.  Missing columns are normalized to an explicit NA value.
        "raw": {
            "aiv_time(us)": _optional_mean("aiv_time(us)"),
            "aiv_mte2_time(us)": _optional_mean("aiv_mte2_time(us)"),
            "aiv_mte3_time(us)": _optional_mean("aiv_mte3_time(us)"),
        },
        # [goal ③] Timeline mode never has the OPPROF {0,1,2} invocation
        # subdirs `_parse_kernel_invocation`/`_aggregate_parsed` read blocks
        # from, so there is truthfully nothing to retain here -- an explicit
        # empty list, not a missing key, so `_kernel_to_ncu_format` can read
        # `k["invocations"]` unconditionally for every producer.
        "invocations": [],
    }
    for field in (
        "aic_scalar_time_ns", "aic_cube_time_ns", "aic_mte2_time_ns",
        "aiv_vec_time_ns", "aiv_scalar_time_ns",
    ):
        if aggregate[field] < 0:
            raise ValueError(
                f"timeline metric {field!r} cannot be negative for "
                f"{target_kernel!r}")
    return aggregate


_VECTOR_TIMELINE_ELEMENTWISE_TARGETS = frozenset(("Add", "Mul"))


def _canonical_mapping_sha256(payload):
    if not isinstance(payload, dict):
        raise TypeError("canonical payload must be a mapping")
    encoded = json.dumps(
        payload, separators=(",", ":"), sort_keys=True,
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_TIMELINE_CALL_COUNTS = {
    "gemm": 6,
    "flash_attn": 21,
    "fused_moe": 11,
    "vector": 31,
}
_TIMELINE_COLD_FILL_OUTPUT_ELEMENTS = 1_073_741_824
_TIMELINE_COLD_FILL_DTYPE = "DT_BF16"
_TIMELINE_RUNTIME_NAMESPACE_ALIASES = {
    "MatMul": ("aclnnMatmul_",),
    "Range": ("aclnnArange_",),
    "Transpose": ("aclnnInplaceCopy_Transpose",),
    "Cast": ("aclnnCast_",),
    "MoeInitRouting": ("aclnnMoeInitRoutingV3_",),
    "GroupedMatmul": ("aclnnGroupedMatmulV5_",),
    "IndexSelect": ("aclnnIndexSelect_",),
    "Abs": ("aclnnAbs_",),
}


def _timeline_schedule_prefixes(kernel, shape):
    if kernel == "gemm":
        return ("MatMul",)
    if kernel == "flash_attn":
        return ("FusedInferAttentionScore",)
    if kernel == "fused_moe":
        return (
            "MoeGatingTopK", "MoeInitRouting", "GroupedMatmul", "SwiGlu",
            "GroupedMatmul", "Abs", "MoeTokenUnpermute",
        )
    if kernel == "vector":
        if shape["attr"]["op"] == "rope":
            return ("IndexSelect", "IndexSelect", "AtbRopeKernel")
        return (shape["target_kernel"],)
    raise ValueError(f"unsupported Ascend kernel: {kernel!r}")


def _timeline_name_matches(name, prefix):
    if not isinstance(name, str) or not name:
        raise TypeError("Timeline Op Name must be a non-empty string")
    if prefix in _VECTOR_TIMELINE_ELEMENTWISE_TARGETS:
        return (name.startswith(prefix)
                or name.startswith(f"aclnn{prefix}_"))
    if name.startswith(prefix):
        return True
    if prefix in _TIMELINE_RUNTIME_NAMESPACE_ALIASES:
        return any(name.startswith(alias) for alias in
                   _TIMELINE_RUNTIME_NAMESPACE_ALIASES[prefix])
    return False


def _timeline_task_identity(row):
    """Extract the device-reported identity fields for a retained task."""

    fields = {
        "op_type": "OP Type",
        "task_type": "Task Type",
        "block_dim": "Block Dim",
        "mix_block_dim": "Mix Block Dim",
        "input_shapes": "Input Shapes",
        "input_data_types": "Input Data Types",
        "input_formats": "Input Formats",
        "output_shapes": "Output Shapes",
        "output_data_types": "Output Data Types",
        "output_formats": "Output Formats",
    }
    identity = {}
    for normalized_name, csv_name in fields.items():
        value = row[csv_name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "Timeline task identity field is missing or empty: "
                f"field={csv_name!r}, value={value!r}")
        identity[normalized_name] = value.strip()
    return identity


def _timeline_fill_task_id(row, context):
    if row["OP Type"] != "Fill":
        raise ValueError(f"{context} is not a Fill task")
    output_shape = row["Output Shapes"].replace('"', "").strip()
    if output_shape != str(_TIMELINE_COLD_FILL_OUTPUT_ELEMENTS):
        raise ValueError(
            f"{context} output shape is not the 2 GiB BF16 sweep: "
            f"{row['Output Shapes']!r}")
    if row["Output Data Types"].strip() != _TIMELINE_COLD_FILL_DTYPE:
        raise ValueError(
            f"{context} dtype is not {_TIMELINE_COLD_FILL_DTYPE}: "
            f"{row['Output Data Types']!r}")
    return _integer_measurement(
        row["Task ID"], f"{context} Task ID", positive=False)


def _is_timeline_cache_fill(row):
    return (
        row["OP Type"] == "Fill"
        and row["Output Shapes"].replace('"', "").strip()
        == str(_TIMELINE_COLD_FILL_OUTPUT_ELEMENTS)
        and row["Output Data Types"].strip()
        == _TIMELINE_COLD_FILL_DTYPE
    )


def _timeline_public_call_measurement(output_dir, kernel, shape):
    """Return exact cold-cache public-call samples from one Timeline summary."""

    if kernel not in {"gemm", "flash_attn", "fused_moe", "vector"}:
        raise ValueError(
            f"unsupported Ascend Timeline kernel category: {kernel!r}")
    summaries = _timeline_summary_paths(output_dir)
    if not summaries:
        raise FileNotFoundError(
            f"no msprof timeline summary found under {output_dir}")
    summary = max(summaries, key=os.path.getmtime)
    rows = _read_csv_rows(summary)
    expected_schedule = _timeline_schedule_prefixes(kernel, shape)
    fill_indexes = [
        index for index, row in enumerate(rows)
        if _is_timeline_cache_fill(row)
    ]
    if not fill_indexes:
        raise ValueError(
            f"Timeline capture has no 2 GiB Fill boundary: {summary}")
    expected_fill_count = _TIMELINE_CALL_COUNTS[kernel]
    if len(fill_indexes) != expected_fill_count:
        raise ValueError(
            "Timeline 2 GiB Fill count mismatch: "
            f"case={shape['filename']}, expected={expected_fill_count}, "
            f"actual={len(fill_indexes)}")
    calls = []
    excluded_leading_calls = []
    for fill_ordinal, fill_index in enumerate(fill_indexes):
        fill = rows[fill_index]
        fill_task_id = _timeline_fill_task_id(
            fill, f"Timeline Fill {fill_ordinal} for {shape['filename']}")
        next_index = (
            fill_indexes[fill_ordinal + 1]
            if fill_ordinal + 1 < len(fill_indexes) else len(rows))
        call_rows = rows[fill_index + 1:next_index]
        actual_names = [row["Op Name"] for row in call_rows]
        schedule_matches = (
            len(call_rows) == len(expected_schedule) and all(
                _timeline_name_matches(row["Op Name"], prefix)
                for row, prefix in zip(
                    call_rows, expected_schedule, strict=True)))
        if not schedule_matches:
            if calls:
                raise ValueError(
                    "Timeline Fill boundary does not contain the exact public "
                    f"call schedule after retained samples began: case="
                    f"{shape['filename']}, fill={fill_ordinal}, "
                    f"expected={expected_schedule!r}, actual="
                    f"{actual_names!r}")
            excluded_leading_calls.append({
                "fill_ordinal": fill_ordinal,
                "fill_task_id": fill_task_id,
                "runtime_schedule": actual_names,
                "reason": "one_time_warmup_schedule_mismatch",
            })
            continue
        tasks = []
        task_intervals_us = []
        for task_ordinal, row in enumerate(call_rows):
            duration_us = _decimal_measurement(
                row["Task Duration(us)"], "Timeline Task Duration")
            if duration_us <= 0:
                raise ValueError(
                    "Timeline Task Duration must be finite and positive: "
                    f"case={shape['filename']}, fill={fill_ordinal}, "
                    f"task={task_ordinal}, value={row['Task Duration(us)']!r}")
            start_time_us = _decimal_measurement(
                row["Task Start Time(us)"], "Timeline Task Start Time")
            if start_time_us < 0:
                raise ValueError(
                    "Timeline Task Start Time must be finite and non-negative: "
                    f"case={shape['filename']}, fill={fill_ordinal}, "
                    f"task={task_ordinal}, value="
                    f"{row['Task Start Time(us)']!r}")
            end_time_us = start_time_us + duration_us
            task_intervals_us.append((start_time_us, end_time_us))
            tasks.append({
                "task_id": _integer_measurement(
                    row["Task ID"], "Timeline target Task ID",
                    positive=False),
                "stream_id": _integer_measurement(
                    row["Stream ID"], "Timeline target Stream ID",
                    positive=False),
                "runtime_symbol": row["Op Name"],
                "task_start_time_us": format(start_time_us, "f"),
                "duration_ns": float(duration_us * 1000),
                "task_end_time_us": format(end_time_us, "f"),
                **_timeline_task_identity(row),
            })
        if not tasks:
            raise ValueError(
                "Timeline public-call task schedule must not be empty: "
                f"case={shape['filename']}, fill={fill_ordinal}")
        stream_ids = {task["stream_id"] for task in tasks}
        if len(stream_ids) != 1:
            raise ValueError(
                "Timeline public-call task schedule spans multiple streams: "
                f"case={shape['filename']}, fill={fill_ordinal}, "
                f"stream_ids={sorted(stream_ids)!r}")
        for task_ordinal, (previous, current) in enumerate(
                zip(task_intervals_us, task_intervals_us[1:]), start=1):
            previous_start_us, previous_end_us = previous
            current_start_us, _ = current
            if current_start_us <= previous_start_us:
                raise ValueError(
                    "Timeline same-stream task schedule is not in strictly "
                    f"increasing start-time order: case={shape['filename']}, "
                    f"fill={fill_ordinal}, task={task_ordinal}")
            if current_start_us < previous_end_us:
                raise ValueError(
                    "Timeline same-stream task schedule contains overlapping "
                    f"tasks: case={shape['filename']}, fill={fill_ordinal}, "
                    f"task={task_ordinal}")
        timing_aggregation = (
            "single_task_direct_duration"
            if len(tasks) == 1
            else "same_stream_task_schedule_envelope"
        )
        op_elapsed_duration_ns = float((
            max(end_time_us for _, end_time_us in task_intervals_us)
            - min(start_time_us for start_time_us, _ in task_intervals_us)
        ) * 1000)
        if not math.isfinite(op_elapsed_duration_ns) \
                or op_elapsed_duration_ns <= 0:
            raise ValueError(
                "Timeline public-call OP elapsed duration must be finite and "
                f"positive: case={shape['filename']}, fill={fill_ordinal}, "
                f"value={op_elapsed_duration_ns!r}")
        calls.append({
            "fill_task_id": fill_task_id,
            "stream_id": tasks[0]["stream_id"],
            "tasks": tasks,
            "timing_aggregation": timing_aggregation,
            "op_elapsed_duration_ns": op_elapsed_duration_ns,
            "summed_task_duration_ns": float(sum(
                (end_time_us - start_time_us) * 1000
                for start_time_us, end_time_us in task_intervals_us)),
        })
    excluded_initial_exact_calls = []
    if kernel == "fused_moe":
        if not calls:
            raise ValueError(
                "Timeline fused_moe has no exact public-call samples before "
                f"initial settling exclusion: case={shape['filename']}")
        initial_call = calls.pop(0)
        excluded_initial_exact_calls.append({
            "fill_task_id": initial_call["fill_task_id"],
            "stream_id": initial_call["stream_id"],
            "op_elapsed_duration_ns": initial_call["op_elapsed_duration_ns"],
            "reason": (
                "fixed_initial_exact_call_exclusion_for_one_time_submission_"
                "or_compilation_gap"),
        })
        initial_exact_call_policy = (
            "exclude_first_schedule_valid_call_only_for_fused_moe_to_remove_"
            "the_observed_one_time_submission_or_compilation_gap")
    elif kernel in {"gemm", "flash_attn", "vector"}:
        initial_exact_call_policy = "not_applicable"
    else:
        raise ValueError(
            f"unsupported Ascend Timeline kernel category: {kernel!r}")
    if len(calls) < 3:
        raise ValueError(
            "Timeline has fewer than three exact public-call samples after "
            f"leading/initial settling exclusion: case={shape['filename']}, "
            f"exact={len(calls)}, leading_excluded="
            f"{len(excluded_leading_calls)}, initial_exact_excluded="
            f"{len(excluded_initial_exact_calls)}")
    fill_task_ids = [call["fill_task_id"] for call in calls]
    if len(set(fill_task_ids)) != len(fill_task_ids):
        raise ValueError(
            f"Timeline Fill Task IDs are not unique: {fill_task_ids!r}")
    runtime_schedules = [
        tuple(task["runtime_symbol"] for task in call["tasks"])
        for call in calls
    ]
    if any(schedule != runtime_schedules[0]
           for schedule in runtime_schedules[1:]):
        raise ValueError(
            "Timeline runtime schedule changed between public calls: "
            f"case={shape['filename']}, schedules={runtime_schedules!r}")
    timing_aggregations = {
        call["timing_aggregation"] for call in calls
    }
    if len(timing_aggregations) != 1:
        raise ValueError(
            "Timeline OP timing aggregation changed between public calls: "
            f"case={shape['filename']}, aggregations="
            f"{sorted(timing_aggregations)!r}")
    per_call_aggregation = next(iter(timing_aggregations))
    durations_ns = [call["op_elapsed_duration_ns"] for call in calls]
    median_ns = statistics.median(durations_ns)
    absolute_deviations = [abs(value - median_ns) for value in durations_ns]
    maximum_absolute_deviation_ns = max(absolute_deviations)
    return {
        "summary_path": summary,
        "expected_schedule_prefixes": list(expected_schedule),
        "runtime_schedule": list(runtime_schedules[0]),
        "calls": calls,
        "sample_count": len(calls),
        "excluded_leading_call_count": len(excluded_leading_calls),
        "excluded_leading_calls": excluded_leading_calls,
        "excluded_initial_exact_call_count": len(excluded_initial_exact_calls),
        "excluded_initial_exact_calls": excluded_initial_exact_calls,
        "initial_exact_call_policy": initial_exact_call_policy,
        "timing_source_fields": [
            "Task Start Time(us)", "Task Duration(us)", "Stream ID",
        ],
        "per_call_timing_aggregation": per_call_aggregation,
        "cross_call_timing_aggregation": "median",
        "median_op_elapsed_duration_ns": median_ns,
        "median_absolute_deviation_ns": statistics.median(
            absolute_deviations),
        "maximum_relative_deviation_pct": (
            maximum_absolute_deviation_ns / median_ns * 100.0),
        "relative_range_pct": (
            (max(durations_ns) - min(durations_ns)) / median_ns * 100.0),
        "minimum_op_elapsed_duration_ns": min(durations_ns),
        "maximum_op_elapsed_duration_ns": max(durations_ns),
        "cold_cache_contract": {
            "mode": "immediately_after_rotating_bf16_fill",
            "fill_output_elements": _TIMELINE_COLD_FILL_OUTPUT_ELEMENTS,
            "fill_dtype": _TIMELINE_COLD_FILL_DTYPE,
            "fill_bytes": _TIMELINE_COLD_FILL_OUTPUT_ELEMENTS * 2,
            "fill_task_ids": fill_task_ids,
        },
    }


def _timeline_steady_state_public_call_measurement(output_dir, kernel, shape):
    """Parse the final exact warm public call from a Timeline capture.

    The normal runners create inputs, execute their documented warmup sequence,
    then submit one final exact public call followed by a device synchronize.
    A Timeline device task timestamp is immune to the host-gap race affecting
    direct Event measurement.  The final task suffix is therefore a strict,
    runner-owned boundary: accepting any earlier matching sequence would risk
    retaining warmup or setup work instead of the public call under test.
    """

    if kernel not in {"gemm", "flash_attn", "fused_moe", "vector"}:
        raise ValueError(
            f"unsupported Ascend Timeline kernel category: {kernel!r}")
    summaries = _timeline_summary_paths(output_dir)
    if not summaries:
        raise FileNotFoundError(
            f"no msprof timeline summary found under {output_dir}")
    summary = max(summaries, key=os.path.getmtime)
    rows = _read_csv_rows(summary)
    expected_schedule = _timeline_schedule_prefixes(kernel, shape)
    if kernel == "fused_moe":
        start_indices = [
            index for index, row in enumerate(rows)
            if _timeline_name_matches(row["Op Name"], "MoeGatingTopK")
        ]
        if not start_indices:
            raise ValueError(
                "Timeline has no MoeGatingTopK public-call boundary: "
                f"case={shape['filename']}")
        start_index = start_indices[-1]
        end_index = next((
            index for index in range(start_index, len(rows))
            if _timeline_name_matches(
                rows[index]["Op Name"], "MoeTokenUnpermute")
        ), None)
        if end_index is None:
            raise ValueError(
                "Timeline has no MoeTokenUnpermute after the final "
                f"MoeGatingTopK: case={shape['filename']}")
        call_rows = rows[start_index:end_index + 1]
    else:
        if len(rows) < len(expected_schedule):
            raise ValueError(
                "Timeline has fewer rows than one exact public-call schedule: "
                f"case={shape['filename']}, expected={expected_schedule!r}, "
                f"actual_row_count={len(rows)}")
        call_rows = rows[-len(expected_schedule):]
    actual_names = [row["Op Name"] for row in call_rows]
    valid_schedules = (expected_schedule,)
    if kernel == "fused_moe":
        valid_schedules = (
            expected_schedule,
            (*expected_schedule[:2], "Cast", *expected_schedule[2:]),
        )
    schedule_matches = any(
        len(call_rows) == len(schedule)
        and all(
            _timeline_name_matches(row["Op Name"], prefix)
            for row, prefix in zip(call_rows, schedule, strict=True))
        for schedule in valid_schedules)
    if not schedule_matches:
        raise ValueError(
            "Timeline selected task interval is not the exact public-call "
            "schedule: "
            f"case={shape['filename']}, expected={valid_schedules!r}, "
            f"actual={actual_names!r}")

    tasks = []
    task_intervals_us = []
    for task_ordinal, row in enumerate(call_rows):
        duration_us = _decimal_measurement(
            row["Task Duration(us)"], "Timeline Task Duration")
        if duration_us <= 0:
            raise ValueError(
                "Timeline Task Duration must be finite and positive: "
                f"case={shape['filename']}, task={task_ordinal}, "
                f"value={row['Task Duration(us)']!r}")
        start_time_us = _decimal_measurement(
            row["Task Start Time(us)"], "Timeline Task Start Time")
        if start_time_us < 0:
            raise ValueError(
                "Timeline Task Start Time must be finite and non-negative: "
                f"case={shape['filename']}, task={task_ordinal}, "
                f"value={row['Task Start Time(us)']!r}")
        end_time_us = start_time_us + duration_us
        task_intervals_us.append((start_time_us, end_time_us))
        tasks.append({
            "task_id": _integer_measurement(
                row["Task ID"], "Timeline target Task ID", positive=False),
            "stream_id": _integer_measurement(
                row["Stream ID"], "Timeline target Stream ID",
                positive=False),
            "runtime_symbol": row["Op Name"],
            "task_start_time_us": format(start_time_us, "f"),
            "duration_ns": float(duration_us * 1000),
            "task_end_time_us": format(end_time_us, "f"),
            **_timeline_task_identity(row),
        })
    stream_ids = {task["stream_id"] for task in tasks}
    if len(stream_ids) != 1:
        raise ValueError(
            "Timeline final public-call schedule spans multiple streams: "
            f"case={shape['filename']}, stream_ids={sorted(stream_ids)!r}")
    for task_ordinal, (previous, current) in enumerate(
            zip(task_intervals_us, task_intervals_us[1:]), start=1):
        previous_start_us, previous_end_us = previous
        current_start_us, _ = current
        if current_start_us <= previous_start_us:
            raise ValueError(
                "Timeline same-stream task schedule is not in strictly "
                f"increasing start-time order: case={shape['filename']}, "
                f"task={task_ordinal}")
        if current_start_us < previous_end_us:
            raise ValueError(
                "Timeline same-stream task schedule contains overlapping "
                f"tasks: case={shape['filename']}, task={task_ordinal}")
    summed_task_duration_ns = float(sum(
        (end_time_us - start_time_us) * 1000
        for start_time_us, end_time_us in task_intervals_us))
    if kernel == "fused_moe":
        # Eager Serving issues the MoE routing/compute tasks incrementally.
        # Its same-stream envelope includes host-submission holes between these
        # device tasks, whereas the standalone public call queues them without
        # those holes.  The comparable kernel truth is consequently the sum of
        # the identical device task durations, not an envelope that measures
        # scheduler pacing.  The raw starts remain recorded for audit.
        elapsed_ns = summed_task_duration_ns
        timing_aggregation = "same_stream_task_schedule_summed_duration"
    else:
        elapsed_ns = float((
            max(end_time_us for _, end_time_us in task_intervals_us)
            - min(start_time_us for start_time_us, _ in task_intervals_us))
            * 1000)
        timing_aggregation = (
            "single_task_direct_duration"
            if len(tasks) == 1 else "same_stream_task_schedule_envelope")
    if not math.isfinite(elapsed_ns) or elapsed_ns <= 0:
        raise ValueError(
            "Timeline final public-call elapsed duration must be finite and "
            f"positive: case={shape['filename']}, value={elapsed_ns!r}")
    return {
        "summary_path": summary,
        "expected_schedule_prefixes": list(expected_schedule),
        "runtime_schedule": actual_names,
        "calls": [{
            "stream_id": tasks[0]["stream_id"],
            "tasks": tasks,
            "timing_aggregation": timing_aggregation,
            "op_elapsed_duration_ns": elapsed_ns,
            "summed_task_duration_ns": summed_task_duration_ns,
        }],
        "sample_count": 1,
        "retained_call_selection": (
            "final_exact_public_call_after_runner_owned_warmup"),
        "timing_source_fields": [
            "Task Start Time(us)", "Task Duration(us)", "Stream ID",
        ],
        "per_call_timing_aggregation": timing_aggregation,
        "cross_call_timing_aggregation": "single_final_exact_public_call",
        "median_op_elapsed_duration_ns": elapsed_ns,
        "median_absolute_deviation_ns": 0.0,
        "maximum_relative_deviation_pct": 0.0,
        "relative_range_pct": 0.0,
        "minimum_op_elapsed_duration_ns": elapsed_ns,
        "maximum_op_elapsed_duration_ns": elapsed_ns,
        "cache_contract": {
            "mode": "ordinary_warm_state",
            "cache_eviction_bytes_before_public_call": 0,
        },
    }


def _timeline_capture_is_complete(output_dir, kernel, shape,
                                  profile_cache_eviction_mib):
    try:
        if profile_cache_eviction_mib == 0:
            _timeline_steady_state_public_call_measurement(
                output_dir, kernel, shape)
        elif profile_cache_eviction_mib == 2048:
            _timeline_public_call_measurement(output_dir, kernel, shape)
        else:
            raise ValueError(
                "Timeline capture completeness requires 0 or 2048 MiB cache "
                "eviction")
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False
    return True


def _validate_timeline_candidate_provenance(
        capture_baseline_commit, source_snapshot_sha256,
        source_snapshot_archive):
    if (not isinstance(capture_baseline_commit, str)
            or re.fullmatch(r"[0-9a-f]{40}", capture_baseline_commit) is None):
        raise ValueError(
            "Ascend Timeline capture baseline commit must be 40 lowercase "
            "hex digits")
    if (not isinstance(source_snapshot_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_snapshot_sha256) is None):
        raise ValueError(
            "Ascend Timeline source snapshot SHA-256 must be 64 lowercase "
            "hex digits")
    if (not isinstance(source_snapshot_archive, str)
            or not source_snapshot_archive):
        raise TypeError(
            "Ascend Timeline source snapshot archive must be a non-empty path")
    archive = os.path.realpath(source_snapshot_archive)
    if not os.path.isfile(archive):
        raise FileNotFoundError(
            f"Ascend Timeline source snapshot archive is missing: {archive}")
    archive_sha256 = _file_sha256(archive)
    if archive_sha256 != source_snapshot_sha256:
        raise ValueError(
            "Ascend Timeline source snapshot archive SHA-256 does not match "
            "the declared source snapshot SHA-256")
    return archive, archive_sha256


def _timeline_candidate_output_path(cfg, output_path):
    profiler_root = os.path.realpath(get_profiler_output_dir(cfg))
    candidate_root = os.path.abspath(os.path.join(profiler_root, "candidate"))
    resolved_root = os.path.realpath(candidate_root)
    if os.path.islink(candidate_root) or resolved_root != candidate_root:
        raise ValueError(
            "Ascend Timeline candidate root must not be a symlink or escape "
            f"its profiler owner: {candidate_root}")
    output = os.path.realpath(output_path)
    try:
        common = os.path.commonpath((resolved_root, output))
    except ValueError as error:
        raise ValueError(
            "Ascend Timeline timing candidate must stay under its candidate "
            "owner") from error
    if common != resolved_root:
        raise ValueError(
            "Ascend Timeline timing candidate must stay under its candidate "
            "owner")
    if not output.endswith(".json"):
        raise ValueError("Ascend Timeline timing candidate must be a JSON file")
    return output


def _load_timeline_family_contract(cfg, kernel):
    kernel_cfg = _load_kernel_config(
        get_kernel_config_path(cfg, kernel), kernel)
    shapes = build_shapes(kernel, kernel_cfg)
    shape_index = {shape["filename"]: shape for shape in shapes}
    if len(shape_index) != len(shapes):
        raise ValueError(f"Ascend {kernel} workload has duplicate case IDs")
    dataset_path = get_kernel_dataset_path(cfg, kernel)
    try:
        canonical = load_kernel_dataset_document(dataset_path)
    except (ValueError, OSError) as error:
        raise RuntimeError(
            f"invalid Ascend {kernel} canonical dataset: {dataset_path}") \
            from error
    require_mapping(canonical, f"Ascend {kernel} canonical dataset")
    if set(canonical) != set(shape_index):
        missing = sorted(set(shape_index) - set(canonical))
        unexpected = sorted(set(canonical) - set(shape_index))
        raise ValueError(
            f"Ascend {kernel} canonical dataset disagrees with workload "
            f"contract: missing={missing[:8]}, unexpected={unexpected[:8]}")
    return shapes, canonical, dataset_path


def _timeline_fused_moe_routing_observation(output_dir, shape,
                                            canonical_attributes):
    """Bind a Timeline MoE trace to its canonical expert-token histogram."""
    expected_attribute_keys = set(shape["attr"]) | {"tok_per_expert"}
    if set(canonical_attributes) != expected_attribute_keys:
        raise ValueError(
            "Ascend fused_moe canonical Attr keys disagree with the workload "
            f"contract for {shape['filename']}: expected="
            f"{sorted(expected_attribute_keys)!r}, actual="
            f"{sorted(canonical_attributes)!r}")
    for key, expected_value in shape["attr"].items():
        if canonical_attributes[key] != expected_value:
            raise ValueError(
                "Ascend fused_moe canonical Attr mismatch for "
                f"{shape['filename']}: key={key!r}, expected="
                f"{expected_value!r}, actual="
                f"{canonical_attributes[key]!r}")
    tiling_path = os.path.join(output_dir, "tiling.json")
    try:
        with open(tiling_path, encoding="utf-8") as stream:
            tiling = json.load(stream)
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(
            "invalid Ascend fused_moe Timeline tiling observation: "
            f"{tiling_path}") from error
    require_mapping(tiling, "Ascend fused_moe Timeline tiling observation")
    expected_tiling_keys = {"tok_per_expert", "T", "E", "TOPK", "H", "I"}
    if set(tiling) != expected_tiling_keys:
        raise ValueError(
            "Ascend fused_moe Timeline tiling keys disagree with the runner "
            f"contract for {shape['filename']}: expected="
            f"{sorted(expected_tiling_keys)!r}, actual={sorted(tiling)!r}")
    expected_tiling_values = {
        "T": shape["attr"]["TOK"],
        "E": shape["attr"]["E"],
        "TOPK": shape["attr"]["TOPK"],
        "H": shape["attr"]["H"],
        "I": shape["attr"]["INT"],
    }
    for key, expected_value in expected_tiling_values.items():
        if tiling[key] != expected_value:
            raise ValueError(
                "Ascend fused_moe Timeline tiling mismatch for "
                f"{shape['filename']}: key={key!r}, expected="
                f"{expected_value!r}, actual={tiling[key]!r}")
    per_expert = _to_per_expert(
        tiling["tok_per_expert"], tiling["T"], tiling["TOPK"])
    if per_expert != canonical_attributes["tok_per_expert"]:
        raise ValueError(
            "Ascend fused_moe Timeline routing disagrees with canonical Attr "
            f"for {shape['filename']}: observed={per_expert!r}, canonical="
            f"{canonical_attributes['tok_per_expert']!r}")
    return {
        "path": os.path.relpath(tiling_path, PROJECT_ROOT),
        "sha256": _file_sha256(tiling_path),
        "tok_per_expert": per_expert,
        "source_encoding": "cumulative_group_list",
    }


def _write_timeline_timing_candidate(
        cfg, kernels, profile_run, capture_baseline_commit,
        source_snapshot_sha256, source_snapshot_archive, output_path,
        selected_case_ids=None):
    """Write a timing-only candidate; never publish a canonical dataset."""

    if not isinstance(kernels, list) or not kernels:
        raise TypeError("Ascend Timeline candidate requires kernel families")
    if len(set(kernels)) != len(kernels):
        raise ValueError("Ascend Timeline candidate kernel list has duplicates")
    archive, archive_sha256 = _validate_timeline_candidate_provenance(
        capture_baseline_commit, source_snapshot_sha256,
        source_snapshot_archive)
    output = _timeline_candidate_output_path(cfg, output_path)
    repository_root = os.path.realpath(PROJECT_ROOT)
    families = {}
    total_raw_cases = 0
    total_timing_cases = 0
    total_unsupported_cases = 0
    for kernel in kernels:
        shapes, canonical, dataset_path = _load_timeline_family_contract(
            cfg, kernel)
        shapes = _select_profile_shapes(
            kernel, shapes, False, None, selected_case_ids)
        trace_dir = _profile_run_trace_dir(cfg, kernel, profile_run)
        runner_path = os.path.realpath(get_kernel_script_path(cfg, kernel))
        if not os.path.isfile(runner_path):
            raise FileNotFoundError(
                f"Ascend Timeline runner source is missing: {runner_path}")
        case_updates = {}
        unsupported_cases = {}
        for shape in shapes:
            case_id = shape["filename"]
            canonical_entry = require_mapping(
                canonical[case_id], f"Ascend {kernel} canonical {case_id}")
            op_info = require_mapping(
                canonical_entry["op_info"],
                f"Ascend {kernel} canonical op_info {case_id}")
            if op_info["Type"] != shape["type"]:
                raise ValueError(
                    f"Ascend {kernel} canonical Type mismatch for {case_id}")
            canonical_attributes = require_mapping(
                op_info["Attr"],
                f"Ascend {kernel} canonical Attr {case_id}")
            if kernel == "fused_moe":
                routing_observation = _timeline_fused_moe_routing_observation(
                    os.path.join(trace_dir, case_id), shape,
                    canonical_attributes)
            elif canonical_attributes != shape["attr"]:
                raise ValueError(
                    f"Ascend {kernel} canonical Attr mismatch for {case_id}")
            measurement = _timeline_public_call_measurement(
                os.path.join(trace_dir, case_id), kernel, shape)
            summary_path = measurement.pop("summary_path")
            evidence = {
                "canonical_entry_sha256": _canonical_mapping_sha256(
                    canonical_entry),
                "canonical_attributes_sha256": _canonical_mapping_sha256(
                    op_info["Attr"]),
                "runner_sha256": _file_sha256(runner_path),
                "timeline_summary": {
                    "path": os.path.relpath(summary_path, repository_root),
                    "sha256": _file_sha256(summary_path),
                },
                "measurement": measurement,
            }
            if kernel == "fused_moe":
                evidence["routing_observation"] = routing_observation
            if measurement["per_call_timing_aggregation"] \
                    == "single_task_direct_duration":
                timing_source = "msprof_timeline_task_duration"
                timing_aggregation = (
                    "median_of_per_call_single_task_duration")
            elif measurement["per_call_timing_aggregation"] \
                    == "same_stream_task_schedule_envelope":
                timing_source = "msprof_timeline_task_schedule"
                timing_aggregation = (
                    "median_of_per_call_same_stream_task_schedule_envelope")
            else:
                raise ValueError(
                    "unsupported Ascend Timeline OP timing aggregation: "
                    f"{measurement['per_call_timing_aggregation']!r}")
            case_updates[case_id] = {
                "status": "timing_candidate",
                "replacement_timing": {
                    "granularity": "public_call",
                    "source": timing_source,
                    "aggregation": timing_aggregation,
                    "total_duration_ns": measurement[
                        "median_op_elapsed_duration_ns"],
                },
                **evidence,
            }
        if len(case_updates) + len(unsupported_cases) != len(shapes):
            raise AssertionError(
                f"Ascend {kernel} Timeline candidate lost cases")
        if unsupported_cases:
            raise AssertionError(
                f"Ascend {kernel} unexpectedly has unsupported cases")
        families[kernel] = {
            "canonical_dataset": {
                "path": os.path.relpath(dataset_path, repository_root),
                "sha256": _file_sha256(dataset_path),
                "case_count": len(canonical),
            },
            "runner_source": {
                "path": os.path.relpath(runner_path, repository_root),
                "sha256": _file_sha256(runner_path),
            },
            "raw_case_count": len(shapes),
            "timing_candidate_case_count": len(case_updates),
            "unsupported_case_count": len(unsupported_cases),
            "case_updates": case_updates,
            "unsupported_cases": unsupported_cases,
        }
        total_raw_cases += len(shapes)
        total_timing_cases += len(case_updates)
        total_unsupported_cases += len(unsupported_cases)
    payload = {
        "schema_version": 3,
        "status": "timing_only_candidate",
        "hardware_id": "ascend_910b",
        "profile_mode": "timeline",
        "timing_source": "msprof_timeline_task_start_and_duration",
        "aggregation": "median_of_per_call_op_elapsed_duration",
        "cache_condition": "2_gib_bf16_fill_before_every_public_call",
        "profile_run": profile_run,
        "capture_baseline_commit": capture_baseline_commit,
        "capture_baseline_commit_provenance": (
            "operator_supplied_git_commit_identifier"),
        "source_snapshot": {
            "archive_path": archive,
            "archive_sha256": archive_sha256,
            "verification": "archive_sha256_matches_source_snapshot_sha256",
        },
        "publication_policy": (
            "candidate_only; this artifact cannot overwrite canonical JSON; "
            "OPPROF counters and launch records remain a separate pass"),
        "raw_case_count": total_raw_cases,
        "timing_candidate_case_count": total_timing_cases,
        "unsupported_case_count": total_unsupported_cases,
        "families": families,
    }
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=os.path.dirname(output),
                prefix=".ascend-timeline-timing-", delete=False) as stream:
            temporary_path = stream.name
            json.dump(payload, stream, indent=2, ensure_ascii=False,
                      allow_nan=False)
            stream.write("\n")
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return output


def _validate_timeline_steady_state_profile_runs(profile_runs):
    """Validate the fixed three-window capture contract before parsing raw data."""

    if not isinstance(profile_runs, list):
        raise TypeError(
            "Ascend steady-state Timeline profile runs must be a list")
    if len(profile_runs) != 3:
        raise ValueError(
            "Ascend steady-state Timeline candidate requires exactly three "
            f"independent profile runs, got {len(profile_runs)}")
    if len(set(profile_runs)) != len(profile_runs):
        raise ValueError(
            "Ascend steady-state Timeline profile runs must be distinct")
    for profile_run in profile_runs:
        _validate_profile_run_name(profile_run)


def _timeline_window_relative_difference_pct(first_duration_ns,
                                             second_duration_ns):
    if (not isinstance(first_duration_ns, (int, float))
            or isinstance(first_duration_ns, bool)
            or not isinstance(second_duration_ns, (int, float))
            or isinstance(second_duration_ns, bool)):
        raise TypeError(
            "Timeline window durations must be numeric finite measurements")
    if (not math.isfinite(first_duration_ns)
            or not math.isfinite(second_duration_ns)
            or first_duration_ns <= 0 or second_duration_ns <= 0):
        raise ValueError(
            "Timeline window durations must be finite and positive")
    return abs(first_duration_ns - second_duration_ns) / max(
        first_duration_ns, second_duration_ns) * 100.0


def _write_timeline_steady_state_timing_candidate(
        cfg, kernels, profile_runs, capture_baseline_commit,
        source_snapshot_sha256, source_snapshot_archive, output_path,
        selected_case_ids=None):
    """Write a fixed-protocol three-window device-timing candidate.

    GEMM, FlashAttention, and FusedMoE use one final ordinary-warm public
    call after runner-owned warmup. Vector is intentionally different: each
    window contains 31 exact calls, each immediately preceded by a 2 GiB BF16
    Fill, and retains that window's median. The protocol split was selected
    from the Serving alignment evidence and is explicit in every family and
    window record. Every retained value is a device task-time measurement,
    never an Event or host interval. All three windows must exist and agree;
    there is deliberately no partial, old-data, or fastest-sample fallback.
    """

    if not isinstance(kernels, list) or not kernels:
        raise TypeError(
            "Ascend steady-state Timeline candidate requires kernel families")
    if len(set(kernels)) != len(kernels):
        raise ValueError(
            "Ascend steady-state Timeline candidate kernel list has duplicates")
    _validate_timeline_steady_state_profile_runs(profile_runs)
    archive, archive_sha256 = _validate_timeline_candidate_provenance(
        capture_baseline_commit, source_snapshot_sha256,
        source_snapshot_archive)
    output = _timeline_candidate_output_path(cfg, output_path)
    repository_root = os.path.realpath(PROJECT_ROOT)
    families = {}
    total_raw_cases = 0
    total_timing_cases = 0
    cross_window_limit_pct = 5.0

    for kernel in kernels:
        shapes, canonical, dataset_path = _load_timeline_family_contract(
            cfg, kernel)
        shapes = _select_profile_shapes(
            kernel, shapes, False, None, selected_case_ids)
        runner_path = os.path.realpath(get_kernel_script_path(cfg, kernel))
        if not os.path.isfile(runner_path):
            raise FileNotFoundError(
                f"Ascend steady-state Timeline runner source is missing: "
                f"{runner_path}")
        case_updates = {}
        for shape in shapes:
            case_id = shape["filename"]
            canonical_entry = require_mapping(
                canonical[case_id], f"Ascend {kernel} canonical {case_id}")
            op_info = require_mapping(
                canonical_entry["op_info"],
                f"Ascend {kernel} canonical op_info {case_id}")
            if op_info["Type"] != shape["type"]:
                raise ValueError(
                    f"Ascend {kernel} canonical Type mismatch for {case_id}")
            canonical_attributes = require_mapping(
                op_info["Attr"],
                f"Ascend {kernel} canonical Attr {case_id}")
            if kernel != "fused_moe" and canonical_attributes != shape["attr"]:
                raise ValueError(
                    f"Ascend {kernel} canonical Attr mismatch for {case_id}")

            windows = []
            routing_observations = []
            for profile_run in profile_runs:
                trace_dir = _profile_run_trace_dir(cfg, kernel, profile_run)
                output_dir = os.path.join(trace_dir, case_id)
                if kernel == "fused_moe":
                    routing_observations.append(
                        _timeline_fused_moe_routing_observation(
                            output_dir, shape, canonical_attributes))
                if kernel == "vector":
                    measurement = _timeline_public_call_measurement(
                        output_dir, kernel, shape)
                else:
                    measurement = (
                        _timeline_steady_state_public_call_measurement(
                            output_dir, kernel, shape))
                summary_path = measurement.pop("summary_path")
                windows.append({
                    "profile_run": profile_run,
                    "timeline_summary": {
                        "path": os.path.relpath(summary_path, repository_root),
                        "sha256": _file_sha256(summary_path),
                    },
                    "measurement": measurement,
                })

            runtime_schedules = [
                tuple(window["measurement"]["runtime_schedule"])
                for window in windows
            ]
            if any(schedule != runtime_schedules[0]
                   for schedule in runtime_schedules[1:]):
                raise ValueError(
                    "Ascend steady-state Timeline runtime schedule changed "
                    f"between windows: case={case_id}, "
                    f"schedules={runtime_schedules!r}")
            aggregations = {
                window["measurement"]["per_call_timing_aggregation"]
                for window in windows
            }
            if len(aggregations) != 1:
                raise ValueError(
                    "Ascend steady-state Timeline timing aggregation changed "
                    f"between windows: case={case_id}, "
                    f"aggregations={sorted(aggregations)!r}")
            window_durations_ns = [
                window["measurement"]["median_op_elapsed_duration_ns"]
                for window in windows
            ]
            pairwise_relative_deviation_pct = []
            for first_index in range(len(window_durations_ns)):
                for second_index in range(first_index + 1,
                                          len(window_durations_ns)):
                    pairwise_relative_deviation_pct.append(
                        _timeline_window_relative_difference_pct(
                            window_durations_ns[first_index],
                            window_durations_ns[second_index]))
            maximum_relative_deviation_pct = max(
                pairwise_relative_deviation_pct)
            if maximum_relative_deviation_pct > cross_window_limit_pct:
                raise ValueError(
                    "Ascend steady-state Timeline windows exceed the 5% "
                    f"reproducibility limit: case={case_id}, durations_ns="
                    f"{window_durations_ns!r}, pairwise_relative_deviation_"
                    f"pct={pairwise_relative_deviation_pct!r}")
            per_call_aggregation = next(iter(aggregations))
            if per_call_aggregation == "single_task_direct_duration":
                timing_source = "msprof_timeline_task_duration"
            elif per_call_aggregation == "same_stream_task_schedule_envelope":
                timing_source = "msprof_timeline_task_schedule"
            elif per_call_aggregation == "same_stream_task_schedule_summed_duration":
                timing_source = "msprof_timeline_task_duration"
            else:
                raise ValueError(
                    "unsupported Ascend steady-state Timeline timing "
                    f"aggregation: {per_call_aggregation!r}")
            golden_duration_ns = float(statistics.median(window_durations_ns))
            update = {
                "status": "timing_candidate",
                "replacement_timing": {
                    "granularity": "public_call",
                    "source": timing_source,
                    "aggregation": (
                        "median_of_three_independent_window_medians"),
                    "total_duration_ns": golden_duration_ns,
                },
                "canonical_entry_sha256": _canonical_mapping_sha256(
                    canonical_entry),
                "canonical_attributes_sha256": _canonical_mapping_sha256(
                    op_info["Attr"]),
                "runner_sha256": _file_sha256(runner_path),
                "windows": windows,
                "reproducibility": {
                    "window_count": len(windows),
                    "window_duration_ns": window_durations_ns,
                    "pairwise_relative_deviation_pct": (
                        pairwise_relative_deviation_pct),
                    "maximum_relative_deviation_pct": (
                        maximum_relative_deviation_pct),
                    "maximum_allowed_relative_deviation_pct": (
                        cross_window_limit_pct),
                    "golden_selection": (
                        "median_of_three_window_medians_not_fastest"),
                },
            }
            if kernel == "fused_moe":
                update["routing_observations"] = routing_observations
            case_updates[case_id] = update

        if len(case_updates) != len(shapes):
            raise AssertionError(
                f"Ascend {kernel} steady-state Timeline candidate lost cases")
        families[kernel] = {
            "cache_condition": (
                "2_gib_bf16_fill_before_every_public_call"
                if kernel == "vector" else
                "ordinary_warm_state_after_runner_owned_warmup"),
            "canonical_dataset": {
                "path": os.path.relpath(dataset_path, repository_root),
                "sha256": _file_sha256(dataset_path),
                "case_count": len(canonical),
            },
            "runner_source": {
                "path": os.path.relpath(runner_path, repository_root),
                "sha256": _file_sha256(runner_path),
            },
            "raw_case_count": len(shapes) * len(profile_runs),
            "timing_candidate_case_count": len(case_updates),
            "case_updates": case_updates,
        }
        total_raw_cases += len(shapes) * len(profile_runs)
        total_timing_cases += len(case_updates)

    family_cache_conditions = {
        kernel: family["cache_condition"]
        for kernel, family in families.items()
    }
    unique_cache_conditions = set(family_cache_conditions.values())
    payload = {
        "schema_version": 4,
        "status": "timing_only_candidate",
        "hardware_id": "ascend_910b",
        "profile_mode": "timeline",
        "timing_source": "msprof_timeline_task_start_and_duration",
        "aggregation": "median_of_three_independent_window_medians",
        "cache_condition": (
            next(iter(unique_cache_conditions))
            if len(unique_cache_conditions) == 1
            else "family_specific; see family_cache_conditions"),
        "family_cache_conditions": family_cache_conditions,
        "profile_runs": profile_runs,
        "capture_baseline_commit": capture_baseline_commit,
        "capture_baseline_commit_provenance": (
            "operator_supplied_git_commit_identifier"),
        "source_snapshot": {
            "archive_path": archive,
            "archive_sha256": archive_sha256,
            "verification": (
                "archive_sha256_matches_source_snapshot_sha256"),
        },
        "publication_policy": (
            "candidate_only; this artifact cannot overwrite canonical JSON; "
            "Event timing and OPPROF counters are not consumed"),
        "raw_case_window_count": total_raw_cases,
        "timing_candidate_case_count": total_timing_cases,
        "families": families,
    }
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=os.path.dirname(output),
                prefix=".ascend-steady-timeline-timing-", delete=False) as stream:
            temporary_path = stream.name
            json.dump(payload, stream, indent=2, ensure_ascii=False,
                      allow_nan=False)
            stream.write("\n")
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return output


def _find_csv(inv_dir, prefix):
    hits = sorted(glob.glob(os.path.join(inv_dir, f"{prefix}_*.csv")))
    # Flat msprof structure (no --kernel-name): file has no timestamp suffix
    hits.extend(sorted(glob.glob(os.path.join(inv_dir, f"{prefix}.csv"))))
    if len(hits) > 1:
        raise ValueError(
            f"ambiguous {prefix} CSVs in {inv_dir}: {hits}")
    return hits[0] if hits else None


def _required_csv_rows(inv_dir, prefix):
    path = _find_csv(inv_dir, prefix)
    if path is None:
        raise FileNotFoundError(
            f"msprof invocation {inv_dir} is missing {prefix} CSV")
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"msprof CSV is empty: {path}")
    return rows


def _parse_kernel_invocation(inv_dir):
    """Parse one invocation dir's CSVs into a flat metric dict.

    msprof Memory/L2Cache CSVs have one row per AI-Core block. Semantics
    (verified on-device against the emitted profiler columns):
      - {read,write}_main_memory_datas(KB): per-block L2/memory-hierarchy REQUEST
        volume. SUM over blocks = total L2 (xbar) traffic. It is NOT pure HBM:
        summing per-block main_mem bandwidth exceeds HBM peak ~4x on cached GEMM.
      - True HBM/DRAM traffic = sum_over_blocks(request) x (1 - L2_hit_rate).
        Verified: GEMV(0.6% hit)->32MB weight; GEMM4096^3(96% hit)->96MB (A+B+C).
      - Ratios (cube/vec/scalar), cycles, hit rates: per-block -> MEAN over blocks.
    """
    out = {}
    rows = _required_csv_rows(inv_dir, "OpBasicInfo")
    r = rows[0]
    op_name = r["Op Name"].strip()
    op_type = r["Op Type"].strip()
    if not op_name or op_name in ("NA", "N/A"):
        raise ValueError(f"missing Op Name in {inv_dir}")
    if not op_type or op_type in ("NA", "N/A"):
        raise ValueError(f"missing Op Type in {inv_dir}")
    out["op_name"] = op_name
    out["op_type"] = op_type
    duration = r["Task Duration(us)"].strip()
    frequency = r["Current Freq"].strip()
    if duration in ("", "NA", "N/A") or frequency in ("", "NA", "N/A"):
        raise ValueError(f"missing duration/frequency in {inv_dir}")
    out["duration_us"] = to_float(duration)
    out["cur_freq_mhz"] = to_float(frequency)
    if out["duration_us"] <= 0 or out["cur_freq_mhz"] <= 0:
        raise ValueError(
            f"duration and frequency must be positive in {inv_dir}")
    for field, column in (
        ("block_dim", "Block Dim"),
        ("mix_block_dim", "Mix Block Dim"),
    ):
        value = r[column].strip()
        if value in ("", "NA", "N/A"):
            if field == "mix_block_dim":
                out[field] = 0
            else:
                raise ValueError(f"missing {column!r} in {inv_dir}")
        else:
            out[field] = _integer_measurement(
                value, f"{column!r} in {inv_dir}",
                positive=(field == "block_dim"))

    rows = _required_csv_rows(inv_dir, "PipeUtilization")
    pipe_rows = rows
    # Keep the historical aggregate key consumed by the common schema, but
    # source it from the domain that actually executes the kernel.  Pure AIV
    # rows report the AIC columns as NA; treating those as zero corrupts both
    # active-cycle and cache-residency observations.
    if op_type == "mix":
        out["aic_total_cycles"] = _mean(rows, "aic_total_cycles")
    elif op_type in ("cube", "vector"):
        pure_domain = _pure_profiler_domain(op_type)
        out["aic_total_cycles"] = _mean(
            rows, f"{pure_domain}_total_cycles")
    else:
        raise ValueError(f"unsupported msprof Op Type: {op_type!r}")
    out["cube_ratio"] = _mean_inactive_zero(rows, "aic_cube_ratio")
    out["aic_scalar_ratio"] = _mean_inactive_zero(rows, "aic_scalar_ratio")
    out["aic_mte2_ratio"] = _mean_inactive_zero(rows, "aic_mte2_ratio")
    out["vec_ratio"] = _mean_inactive_zero(rows, "aiv_vec_ratio")
    out["aic_scalar_time_ns"] = (
        _mean_inactive_zero(rows, "aic_scalar_time(us)") * 1000.0)
    out["aic_cube_time_ns"] = (
        _mean_inactive_zero(rows, "aic_cube_time(us)") * 1000.0)
    out["aic_mte2_time_ns"] = (
        _mean_inactive_zero(rows, "aic_mte2_time(us)") * 1000.0)
    out["aiv_vec_time_ns"] = (
        _mean_inactive_zero(rows, "aiv_vec_time(us)") * 1000.0)
    out["aiv_scalar_time_ns"] = (
        _mean_inactive_zero(rows, "aiv_scalar_time(us)") * 1000.0)

    l2_rows = _required_csv_rows(inv_dir, "L2Cache")
    memory_rows = _required_csv_rows(inv_dir, "Memory")
    out.update(_aggregate_msprof_memory_rows(
        memory_rows, l2_rows, op_type,
        out["block_dim"], out["mix_block_dim"]))

    rows = _required_csv_rows(inv_dir, "ArithmeticUtilization")
    arith_rows = rows
    out["cube_fops"] = _sum_inactive_zero(rows, "aic_cube_fops")

    for field in (
        "aic_total_cycles", "cube_ratio", "aic_scalar_ratio",
        "aic_mte2_ratio", "vec_ratio", "aic_scalar_time_ns",
        "aic_cube_time_ns", "aic_mte2_time_ns", "aiv_vec_time_ns",
        "aiv_scalar_time_ns", "l2_hit_rate", "l2_read_hit_rate",
        "l2_write_hit_rate", "l2_read_bytes", "l2_write_bytes",
        "read_bytes", "write_bytes", "main_mem_read_bw",
        "main_mem_write_bw", "gm_to_l1_kb", "gm_to_l1_bytes",
        "mte2_tile_count", "cube_fops",
    ):
        if out[field] < 0:
            raise ValueError(
                f"msprof metric {field!r} cannot be negative in {inv_dir}")
    for field in ("l2_hit_rate", "l2_read_hit_rate", "l2_write_hit_rate"):
        if out[field] > 100:
            raise ValueError(
                f"msprof percentage {field!r} exceeds 100 in {inv_dir}")

    # [Lossless extraction] Collect EVERY CSV column into out["raw"] so nothing is lost.
    # Physical aggregation rule used by the canonical Ascend dataset writer:
    #   SUM  = per-block VOLUME/instruction fields → total across cores
    #   MEAN = per-block RATE fields (ratio, cycles, time, hit_rate, BW) → core mean
    # Excludes index columns (block_id/sub_block_id) and string columns
    # (Op Name, Op Type). All measurements (msprof per-pipe cycles/time, per-
    # buffer traffic, L2 partition hits, Arithmetic dtype-split ratios) — raw,
    # NOT duration-derived, so zero-fit-legal to retain. Keeps the 25 hand-
    # picked fields above (unchanged, what running code reads); raw is additive.
    #
    # Schema consistency: a column that is NA for THIS kernel (e.g. AIV columns
    # on a pure-cube GEMM) is still collected as the literal "NA" (NOT skipped,
    # NOT zeroed) so every kernel has the SAME full column set — readers don't
    # need to check field presence, and NA is the truthful "no measurement".
    #
    # Column units are encoded in the column NAME (msprof header convention,
    # e.g. "aic_mte2_time(us)", "gm_to_l1_bytes", "l2_hit_rate(%)",
    # "aic_main_mem_read_bw(GB/s)"). No separate unit field — the name IS the
    # unit, same as the hand-picked *_ns/*_GBs fields above.
    raw = {}
    out["raw"] = raw
    # columns that are SUM (per-block volume / instruction / fops counts)
    _SUM_HINTS = ("datas", "instructions", "instr_number", "fops",
                  "cache_hit", "cache_miss", "_kb", "bytes")

    numeric_literal = re.compile(
        r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

    def _is_string_col(rows, col):
        # A column is a STRING column iff every non-NA value fails to parse as
        # a number (e.g. Op Name="GroupedMatmul_..", Op Type="cube"). Numeric
        # columns that happen to be ALL-NA for this kernel are NOT string
        # columns — they're kept as "NA".
        found_value = False
        for r in rows:
            v = r[col].strip()
            if v in ("", "NA", "N/A"):
                continue
            found_value = True
            if numeric_literal.fullmatch(v):
                return False       # at least one numeric value → numeric col
        # all values are NA (found_value=False) → numeric col with NA, keep;
        # any non-NA non-numeric value → string col, skip.
        return found_value

    for csv_prefix in ("OpBasicInfo", "Memory", "L2Cache",
                       "PipeUtilization", "ArithmeticUtilization"):
        rows = _required_csv_rows(inv_dir, csv_prefix)
        for col in rows[0]:
            c = col.strip()
            if not c or c in ("block_id", "sub_block_id"):
                continue
            if _is_string_col(rows, col):
                continue   # genuine string column (Op Name, Op Type)
            all_na = all(r[col].strip() in ("", "NA", "N/A")
                         for r in rows)
            if all_na:
                raw[c] = "NA"
            elif any(h in c.lower() for h in _SUM_HINTS):
                raw[c] = _sum(rows, col)
            else:
                raw[c] = _mean(rows, col)

    # [goal ②] Retain the untouched per-block rows (block_id x sub_block_id,
    # 66-72 rows per capture) for the 4 msprof CSVs that carry them. Additive:
    # `raw` above (the SUM/MEAN distillate consumed by running code) is
    # unchanged; `blocks` is a new, separate, loss-free copy for readers that
    # need the per-block layer msprof actually produced.
    out["blocks"] = {
        "ArithmeticUtilization": _rows_to_block_table(arith_rows),
        "L2Cache": _rows_to_block_table(l2_rows),
        "Memory": _rows_to_block_table(memory_rows),
        "PipeUtilization": _rows_to_block_table(pipe_rows),
    }
    return out


def _aggregate_parsed(parsed, op_name, op_type):
    """Mean-aggregate a list of parsed invocation dicts into one kernel dict."""
    if not parsed:
        raise ValueError(f"cannot aggregate empty invocation set for {op_name}")
    expected_keys = set(parsed[0])
    for invocation in parsed[1:]:
        if set(invocation) != expected_keys:
            raise ValueError(
                f"inconsistent msprof invocation schema for {op_name!r}")
    agg = {"op_name": op_name, "op_type": op_type}
    numeric_keys = {
        key for key, value in parsed[0].items()
        if isinstance(value, (int, float))
    }
    launch_dimension_keys = {"block_dim", "mix_block_dim"}
    for k in numeric_keys & launch_dimension_keys:
        vals = [p[k] for p in parsed]
        if not all(isinstance(value, (int, float)) for value in vals):
            raise TypeError(
                f"inconsistent numeric field {k!r} for {op_name!r}")
        if not all(isinstance(value, int) for value in vals):
            raise TypeError(
                f"launch dimension {k!r} must be integer for {op_name!r}")
        if len(set(vals)) != 1:
            raise ValueError(
                f"launch dimension {k!r} varies across invocations for "
                f"{op_name!r}: {vals!r}")
        agg[k] = vals[0]
    for k in numeric_keys - launch_dimension_keys:
        vals = [p[k] for p in parsed]
        if not all(isinstance(value, (int, float)) for value in vals):
            raise TypeError(
                f"inconsistent numeric field {k!r} for {op_name!r}")
        agg[k] = sum(vals) / len(vals)
    # Aggregate the raw full-column distillate (mean across invocations, same as
    # the hand-picked fields above — invocations are repeats of the same shape).
    raw_keys = set(parsed[0]["raw"])
    if any(set(invocation["raw"]) != raw_keys for invocation in parsed[1:]):
        raise ValueError(
            f"inconsistent raw msprof schema for {op_name!r}")
    agg_raw = {}
    for k in raw_keys:
        values = [invocation["raw"][k] for invocation in parsed]
        numeric_values = [
            value for value in values if isinstance(value, (int, float))]
        non_numeric_values = [
            value for value in values if not isinstance(value, (int, float))]
        if numeric_values and len(numeric_values) != len(values):
            # A lone "NA" among an otherwise-numeric majority is the same
            # single-invocation counter-capture glitch MsprofActiveCyclesDropout
            # documents for whole-invocation dropouts (observed: one msprof
            # capture drops exactly one scalar column for one repeat while
            # every other column of that same invocation, and this column in
            # every other invocation, is a normal value) -- see
            # _parse_kernel_dir's docstring for the corroborating evidence.
            # Tolerate it only in that exact narrow shape: the NA side is a
            # strict minority of exactly one, and at least two invocations
            # agree on a real number to average over. Anything less clear-cut
            # (a real tie, more than one NA, or a non-"NA" non-numeric value)
            # still raises exactly as before.
            if (len(non_numeric_values) == 1
                    and non_numeric_values[0] == "NA"
                    and len(numeric_values) >= 2):
                warnings.warn(
                    f"{op_name!r} raw field {k!r}: dropping a single 'NA' "
                    f"outlier invocation out of {len(values)} to average the "
                    f"remaining {len(numeric_values)} numeric invocations",
                    stacklevel=2)
            else:
                raise TypeError(
                    f"mixed numeric/NA raw field {k!r} for {op_name!r}")
        if numeric_values:
            agg_raw[k] = sum(numeric_values) / len(numeric_values)
        elif all(value == "NA" for value in values):
            agg_raw[k] = "NA"
        else:
            raise TypeError(f"invalid raw field {k!r} for {op_name!r}")
    agg["raw"] = agg_raw
    # [goal ③] Retain every invocation's own complete ①② (its `raw` distillate
    # and, since the change above, its `blocks` per-block tables) instead of
    # only the cross-invocation mean computed above. `parsed` is exactly the
    # list of per-invocation dicts `_parse_kernel_invocation` returned, in
    # capture order (dir "0", "1", "2", ...), so each invocation is
    # independently retrievable and distinguishable.
    agg["invocations"] = parsed
    return agg


def _parse_kernel_dir(kernel_dir):
    """Average a kernel's metrics across all its invocation subdirs (0,1,2,...).

    Ascend GroupedMatmul (the fused-MoE grouped GEMM) is a special case: vLLM-
    Ascend issues TWO npu_grouped_matmul calls per MoE layer — w1 (gate_up,
    output 2·INT) and w2 (down, output H) — so the kernel dir's invocations
    ALTERNATE between the two projections (verified: w1 GM_to_L1 == 2× w2 across
    all 176 dataset cases). Blindly averaging them together mixes two distinct
    kernels into one entry, which the predictor then mis-models (it must charge
    single-projection traffic, not the w1+w2 sum). So GroupedMatmul is SPLIT by
    GM_to_L1 volume into two entries `<name>_w1` / `<name>_w2`, each averaged
    over its own projection's invocations. All other kernels: one mean entry.

    Returns a LIST of kernel dicts (one element for normal kernels, two for
    GroupedMatmul).
    """
    inv_dirs = sorted(
        d for d in glob.glob(os.path.join(kernel_dir, "*"))
        if os.path.isdir(d) and os.path.basename(d).isdigit())
    # Flat msprof structure (no --kernel-name, single kernel): CSVs sit
    # directly in the kernel_dir (no invocation subdirs). Treat the dir
    # itself as one invocation.
    if not inv_dirs and _find_csv(kernel_dir, "OpBasicInfo"):
        inv_dirs = [kernel_dir]
    if not inv_dirs:
        raise FileNotFoundError(
            f"no msprof invocations found in kernel directory {kernel_dir}")
    parsed = []
    dropped = []
    for d in inv_dirs:
        try:
            parsed.append(_parse_kernel_invocation(d))
        except MsprofActiveCyclesDropout as exc:
            dropped.append((d, exc))
    if dropped:
        # Tolerate isolated single-invocation counter dropouts (see
        # MsprofActiveCyclesDropout's docstring for the evidence basis) only
        # when a strict majority of this kernel's invocations still parsed
        # cleanly. Any other split (a tie, or dropouts outnumbering
        # successes) means the kernel_dir itself is suspect, not just one
        # invocation, so re-raise the first dropout exactly as before.
        if not (len(dropped) < len(parsed)):
            raise dropped[0][1]
        for d, exc in dropped:
            warnings.warn(
                f"{kernel_dir}: dropping invocation {d!r} "
                f"({len(dropped)} of {len(inv_dirs)} invocations) -- {exc}",
                stacklevel=2)
    if any(p["duration_us"] <= 0 for p in parsed):
        raise ValueError(f"non-positive msprof duration in {kernel_dir}")

    op_name = parsed[0]["op_name"]
    op_type = parsed[0]["op_type"]

    if "groupedmatmul" in op_name.lower():
        # Split by GM_to_L1 into the two projections. w1 (larger output) always
        # moves ~2× the bytes of w2, so a midpoint threshold cleanly separates
        # the two clusters regardless of invocation ordering / warmup repeats.
        # GroupedMatmul always has a Memory CSV → gm_to_l1_kb present (fail loud
        # via KeyError if a malformed invocation lacks it, rather than silently
        # mis-splitting on a 0 default).
        vols = [p["gm_to_l1_kb"] for p in parsed]
        vmax, vmin = max(vols), min(vols)
        if vmax > vmin:
            thr = (vmax + vmin) / 2.0
            w1 = [p for p, v in zip(parsed, vols) if v > thr]
            w2 = [p for p, v in zip(parsed, vols) if v <= thr]
            return [
                _aggregate_parsed(w1, f"{op_name}_w1", op_type),
                _aggregate_parsed(w2, f"{op_name}_w2", op_type),
            ]
        raise ValueError(
            f"GroupedMatmul capture does not contain distinct w1/w2 "
            f"projections: {kernel_dir}")

    return [_aggregate_parsed(parsed, op_name, op_type)]


def _kernel_to_ncu_format(k, dram_freq_hz, dram_peak_bw_bytes):
    """Convert an aggregated msprof kernel dict to NCU-compatible JSON.

    Uses the per-kernel Current Freq msprof reports (real, not nominal). Ascend
    has no CUDA grid/block; Block Dim (# AI cores) maps to grid_x, block=1.
    L2 (xbar) bytes = sum-over-blocks main_mem requests; DRAM bytes = that x
    (1 - L2 hit) — distinct fields, verified against analytical traffic.
    Proxies: FMA%=vector-pipe util, ALU%=scalar-pipe util, L1 hit=read hit rate.
    """
    dur_ns = k["duration_us"] * 1000.0
    sm_freq_hz = k["cur_freq_mhz"] * 1e6
    if sm_freq_hz <= 0:
        raise ValueError(
            f"non-positive msprof frequency for {k['op_name']!r}: "
            f"{k['cur_freq_mhz']}")
    elapsed_cycles = dur_ns * sm_freq_hz / 1e9
    active_cycles = k["aic_total_cycles"]
    if active_cycles <= 0:
        raise ValueError(
            f"non-positive active cycles for {k['op_name']!r}: "
            f"{active_cycles}")

    block_dim = require_positive_int(k, "block_dim", "msprof aggregate")
    mix_bd = require_non_negative_int(
        k, "mix_block_dim", "msprof aggregate")
    # Grid = AIC-domain concurrency (compute parallelism that drives the per-AIC
    # per-block BW / latency-hiding model). On 910B a "mix" kernel (e.g. FA)
    # occupies AIC AND AIV domains in parallel: block_dim = AIC (Cube) cores,
    # mix_block_dim = AIV (Vector) cores (= 2×AIC, the 1:2 ratio). The kernel
    # time is max(AIC-domain, AIV-domain); the memory-bound MTE2 KV fetch lives
    # in the AIC domain, so active_units = block_dim (AIC), NOT the AIC+AIV sum.
    # Pure cube (GEMM) / pure vector kernels: mix_bd=0 → grid = block_dim.
    if block_dim <= 0:
        raise ValueError(
            f"non-positive block dimension for {k['op_name']!r}: {block_dim}")
    grid = block_dim

    dram_read = k["read_bytes"]
    dram_write = k["write_bytes"]
    l2_read = k["l2_read_bytes"]
    l2_write = k["l2_write_bytes"]
    l2_hit = k["l2_hit_rate"]
    l2_read_hit = k["l2_read_hit_rate"]
    tensor_pct = k["cube_ratio"] * 100.0
    fma_pct = k["vec_ratio"] * 100.0
    alu_pct = k["aic_scalar_ratio"] * 100.0

    # DRAM BW % of peak, computed from true bytes / duration (physical), not the
    # per-block main_mem_bw column (which double-counts cache-hit requests).
    dram_dur_s = dur_ns / 1e9
    if dram_dur_s <= 0 or dram_peak_bw_bytes <= 0:
        raise ValueError("DRAM duration and peak bandwidth must be positive")
    dram_read_bw_pct = 100.0 * (dram_read / dram_dur_s) / dram_peak_bw_bytes
    dram_write_bw_pct = 100.0 * (dram_write / dram_dur_s) / dram_peak_bw_bytes

    return {
        "Metadata": {
            "Kernel Name": k["op_name"],
            "Grid Size": f"{grid},1,1",
            "Block Size": "1,1,1",
            "Device": "Ascend 910B",
            "CC": "9.0",
            "Op Type": k["op_type"],
        },
        "Sections": {
            "GPU Speed Of Light Throughput": {
                "Header": {
                    "Duration [nsecond]": dur_ns,
                    "SM Frequency [cycle/second]": sm_freq_hz,
                    "DRAM Frequency [cycle/second]": dram_freq_hz,
                    "Elapsed Cycles [cycle]": elapsed_cycles,
                    "SM Active Cycles [cycle]": active_cycles,
                }
            },
            "Compute Workload Analysis": {
                "Pipe Utilization (% of active cycles)": {
                    "Tensor (All) [%]": tensor_pct,
                    "FMA [%]": fma_pct,
                    "ALU [%]": alu_pct,
                },
                "Pipe Utilization (% of peak instructions executed)": {
                    "XU [%]": 0.0,
                },
            },
            "Memory Workload Analysis Chart": {
                "Memory Chart": {
                    "l1tex__m_xbar2l1tex_read_bytes.sum [byte]": l2_read,
                    "l1tex__m_l1tex2xbar_write_bytes.sum [byte]": l2_write,
                    "dram__bytes_read.sum [byte]": dram_read,
                    "dram__bytes_write.sum [byte]": dram_write,
                    "l1tex__t_sector_hit_rate.pct [%]": l2_read_hit,
                    "lts__t_sector_hit_rate.pct [%]": l2_hit,
                    "sm__sass_l1tex_m_xbar2l1tex_read_bytes_mem_global_op_ldgsts_cache_bypass.sum [byte]": 0.0,
                },
            },
            "Launch Statistics": {
                "Header": {
                    "Block Size": 1,
                    "Registers Per Thread [register/thread]": 0,
                    "Dynamic Shared Memory Per Block [byte/block]": 0,
                    "Static Shared Memory Per Block [byte/block]": 0,
                    "Driver Shared Memory Per Block [byte/block]": 0,
                    "Achieved Occupancy [%]": 0.0,
                    "Theoretical Occupancy [%]": 0.0,
                },
            },
            "Occupancy": {"Header": {"Block Limit Registers [block]": 1}},
            "Memory Workload Analysis Tables": {
                "Device Memory": {
                    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed [%]": dram_read_bw_pct,
                    "dram__bytes_write.sum.pct_of_peak_sustained_elapsed [%]": dram_write_bw_pct,
                },
            },
        },
        # 910B-native extras (validation / non-NCU consumers)
        "Ascend": {
            "op_type": k["op_type"],
            "block_dim": block_dim,
            "mix_block_dim": mix_bd,
            "cube_fops": k["cube_fops"],
            "main_mem_read_bw_GBs": k["main_mem_read_bw"],
            "main_mem_write_bw_GBs": k["main_mem_write_bw"],
            # [P87.0] Per-core msprof ground truth (traffic/pipe statistics,
            # independent of the predicted duration → zero-fit-legal). Consumed
            # by the oracle path to separate traffic-error from time/BW-error,
            # and by forward validation. New fields only — forward code that
            # doesn't read them is byte-for-byte unchanged.
            "gm_to_l1_bytes": k["gm_to_l1_bytes"],
            "dram_read_bytes": k["read_bytes"],
            "mte2_tile_count": k["mte2_tile_count"],
            "l2_hit_rate": l2_hit,
            "aic_scalar_time_ns": k["aic_scalar_time_ns"],
            "aic_cube_time_ns": k["aic_cube_time_ns"],
            "aic_mte2_time_ns": k["aic_mte2_time_ns"],
            "aiv_vec_time_ns": k["aiv_vec_time_ns"],
            "aiv_scalar_time_ns": k["aiv_scalar_time_ns"],
            # [Lossless extraction] raw: every msprof CSV column (all 5 tables, ~130 fields)
            # with per-block SUM (volume/instruction/fops/cache_hit) or MEAN
            # (ratio/cycles/time/hit_rate/BW). Additive — running code reads the
            # named fields above, unchanged; raw preserves the full measured
            # profile for future use.
            "raw": k["raw"],
            # [goal ③] Every invocation's own complete ①② (see `parsed` in
            # `_aggregate_parsed`): duration/freq/launch dims, the hand-picked
            # scalars, its `raw` distillate, and its `blocks` per-block tables
            # (goal ②). Empty for timeline-mode captures, which have no
            # OPPROF {0,1,2} subdirs to retain. Additive -- readers that only
            # use the fields above this key are byte-for-byte unaffected.
            "invocations": k["invocations"],
        },
    }


def _load_hw_clocks(cfg):
    """Return (dram_freq_hz, dram_peak_bw_bytes_per_s) from the hw config."""
    hw_path = get_hardware_config_path(cfg)
    hw = load_yaml_mapping(hw_path)
    clock_domains = require_mapping(
        hw["clock_domains"], "hardware clock_domains")
    dram_freq_hz = require_positive_number(
        clock_domains, "dram", "hardware clock_domains")
    components = [
        require_mapping(component, f"hardware component[{index}]")
        for index, component in enumerate(require_non_empty_list(
            hw, "components", "hardware config"))
    ]
    dram_components = [
        component for component in components
        if "role" in component and component["role"] == "dram"
    ]
    if len(dram_components) != 1:
        raise ValueError(
            f"hardware config must define exactly one top-level DRAM "
            f"component: {hw_path}")
    dram_component = require_mapping(
        dram_components[0], "hardware DRAM component")
    bytes_per_clock = require_positive_number(
        dram_component, "throughput_per_clk", "hardware DRAM component")
    dram_peak = bytes_per_clock * dram_freq_hz
    return dram_freq_hz, dram_peak


# Kernels that are pure routing/glue noise we still keep (they run on device),
# but the "primary" kernel for op_info duration is the sum of all.
def _load_staging_baseline(path, expected_keys):
    with open(path, encoding="utf-8") as stream:
        baseline = json.load(stream)
    return require_staging_baseline_keys(baseline, expected_keys, path)


def stage_aggregate(cfg, kernel, profile_run=None,
                    target_flash_attn_reprofile=False,
                    target_vector_op=None,
                    profile_mode="op", vector_profile_mode=None):
    if vector_profile_mode is not None:
        if profile_mode != "op" and profile_mode != vector_profile_mode:
            raise ValueError(
                "conflicting profile_mode and vector_profile_mode")
        profile_mode = vector_profile_mode
    _validate_profile_mode(profile_mode)
    if profile_mode == "timeline":
        raise ValueError(
            "Timeline timing is candidate-only; stage_aggregate cannot "
            "publish or overwrite canonical datasets")
    kernel_cfg = _load_kernel_config(
        get_kernel_config_path(cfg, kernel), kernel)
    all_shapes = build_shapes(kernel, kernel_cfg)
    shapes = _select_profile_shapes(
        kernel, list(all_shapes), target_flash_attn_reprofile,
        target_vector_op)
    if profile_run is not None and kernel not in ("flash_attn", "vector"):
        raise ValueError(
            "named staging aggregation is supported only for flash_attn and "
            "vector replacement runs")
    trace_dir = _profile_run_trace_dir(cfg, kernel, profile_run)
    dram_freq_hz, dram_peak_bw = _load_hw_clocks(cfg)

    print(f"\n>>> Stage: Aggregate [{kernel}] from {trace_dir}")
    print(f"    DRAM: {dram_freq_hz/1e9:.3g} GHz, peak BW {dram_peak_bw/1e9:.0f} GB/s")

    dataset = {}
    for shape in shapes:
        out_dir = os.path.join(trace_dir, shape["filename"])
        if _uses_timeline_profile(kernel, shape, profile_mode):
            aggregate = _parse_timeline_kernel(out_dir, shape["target_kernel"])
            ncu_kernels = [
                _kernel_to_ncu_format(aggregate, dram_freq_hz, dram_peak_bw)
            ]
        else:
            opprofs = glob.glob(os.path.join(out_dir, "OPPROF_*"))
            if not opprofs:
                raise FileNotFoundError(
                    f"no OPPROF capture for {shape['filename']}")
            opprofs.sort(key=os.path.getmtime)
            opprof = opprofs[-1]

            kernel_dirs = []
            for directory in glob.glob(os.path.join(opprof, "*")):
                if not os.path.isdir(directory):
                    continue
                invocation_dirs = [
                    child for child in glob.glob(os.path.join(directory, "*"))
                    if os.path.isdir(child)
                    and os.path.basename(child).isdigit()
                    and _find_csv(child, "OpBasicInfo") is not None
                ]
                if _find_csv(directory, "OpBasicInfo") or invocation_dirs:
                    kernel_dirs.append(directory)
            # Flat msprof structure (no --kernel-name, single kernel): CSVs sit
            # directly in OPPROF (no kernel/invocation subdirs, but may have a
            # 'dump' subdir). If OPPROF has OpBasicInfo directly, treat it as a
            # kernel dir (the dump subdir produces no kernels, harmless).
            if _find_csv(opprof, "OpBasicInfo"):
                kernel_dirs.append(opprof)
            ncu_kernels = []
            for kernel_dir in sorted(kernel_dirs):
                for aggregate in _parse_kernel_dir(kernel_dir):
                    ncu_kernels.append(_kernel_to_ncu_format(
                        aggregate, dram_freq_hz, dram_peak_bw))
        if not ncu_kernels:
            raise ValueError(
                f"capture contains no parsable kernels: {shape['filename']}")

        # Kernel-name verification (collection red line, goal §0/§4): for
        # flash_attn every entry's primary kernel must be FusedInferAttentionScore,
        # not a setup Cast/TransData. For vector, every entry must contain its
        # op's target_kernel (RmsNorm/RotaryPositionEmbedding/SwiGlu/Add/Mul/
        # reshape_and_cache) — else setup Cast/Range/ZerosLike leaked in. With
        # --kernel-name profiling this holds by construction; this asserts it
        # post-hoc and loudly flags any regression.
        if kernel == "flash_attn":
            target_kernel = "FusedInferAttentionScore"
        elif kernel == "vector":
            target_kernel = shape["target_kernel"]
        elif kernel in ("gemm", "fused_moe"):
            target_kernel = None
        else:
            raise ValueError(f"unsupported Ascend kernel: {kernel!r}")
        if target_kernel is not None and not any(
            item["Metadata"]["Kernel Name"].startswith(target_kernel)
            for item in ncu_kernels
        ):
            names = ",".join(k["Metadata"]["Kernel Name"][:40]
                             for k in ncu_kernels)
            raise ValueError(
                f"bad capture for {shape['filename']}: no "
                f"{target_kernel!r}; got {names}")

        timing_durations_ns = [
            k["Sections"]["GPU Speed Of Light Throughput"]["Header"]["Duration [nsecond]"]
            for k in ncu_kernels
        ]
        total_dur = math.fsum(timing_durations_ns)
        entry = {
            "op_info": {
                "Type": shape["type"],
                "Attr": dict(shape["attr"]),
                "evaluation": build_kernel_evaluation_timing(
                    "ascend_910b",
                    ncu_kernels,
                    timing_durations_ns,
                ),
            },
            "kernels": ncu_kernels,
        }

        if kernel == "fused_moe":
            tiling_path = os.path.join(out_dir, "tiling.json")
            if not os.path.exists(tiling_path):
                raise FileNotFoundError(
                    f"fused_moe capture is missing tiling data: {tiling_path}")
            with open(tiling_path) as f:
                tiling = json.load(f)
            entry["op_info"]["Attr"]["tok_per_expert"] = _to_per_expert(
                tiling["tok_per_expert"], tiling["T"], tiling["TOPK"])

        dataset[shape["filename"]] = entry
        print(f"  OK: {shape['filename']} ({len(ncu_kernels)} kernels, {total_dur:.0f} ns)")

    dataset_path = get_kernel_dataset_path(cfg, kernel)
    expected_keys = [shape["filename"] for shape in all_shapes]
    if profile_run is None:
        candidate_path = os.path.join(
            get_profiler_output_dir(cfg), "candidate", f"{kernel}.json")
        publish_complete_dataset(
            dataset, expected_keys, candidate_path, dataset_path)
        output_path = dataset_path
    else:
        expected_replacements = {shape["filename"] for shape in shapes}
        if set(dataset) != expected_replacements:
            raise ValueError(
                "staging capture does not exactly match its selected "
                "replacement set")
        full_dataset = _load_staging_baseline(
            dataset_path, expected_keys)
        full_dataset.update(dataset)
        shape_index = {
            shape["filename"]: shape for shape in all_shapes
        }
        if kernel in ("flash_attn", "vector"):
            for case_key, entry in full_dataset.items():
                shape = shape_index[case_key]
                entry["op_info"]["Type"] = shape["type"]
                entry["op_info"]["Attr"] = dict(shape["attr"])
        else:
            raise ValueError(
                f"unsupported staging aggregation kernel: {kernel}")
        run_dir = os.path.dirname(trace_dir)
        candidate_path = os.path.join(
            run_dir, "candidate", f"{kernel}.json")
        write_complete_candidate(
            full_dataset, expected_keys, candidate_path)
        output_path = candidate_path
    print(
        f"\n[Success] {kernel} dataset: {output_path} "
        f"({len(all_shapes)} entries)")
    return output_path


def main():
    p = argparse.ArgumentParser(description="910B kernel dataset generation via msprof op")
    p.add_argument("--config", default="configs/dataset/ascend_910b.yaml")
    p.add_argument(
        "--kernel", required=True,
        help="Comma-separated: gemm,flash_attn,fused_moe,vector")
    p.add_argument(
        "--stage", required=True,
        choices=["profile", "aggregate", "timeline-timing-candidate",
                 "timeline-steady-timing-candidate"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-tasks", type=int, default=0)
    p.add_argument("--start-idx", type=int, default=0, help="1-based start (parallel runs)")
    p.add_argument("--end-idx", type=int, default=0, help="inclusive end (parallel runs)")
    p.add_argument("--device-id", type=int, help="ASCEND_RT_VISIBLE_DEVICES")
    p.add_argument(
        "--profile-run",
        help=("Write profiler output to a named staging run instead of the "
              "canonical trace directory"))
    p.add_argument(
        "--target-flash-attn-reprofile",
        action="store_true",
        help=("Profile only the 9 paged-varlen prefill cases whose corrected "
              "per-request workload requires Q_LENS=KV_LENS"))
    p.add_argument(
        "--target-vector-op",
        choices=("rmsnorm", "add_rmsnorm", "rope", "silu",
                 "elementwise", "kvcache"),
        help="Profile only this vector operation (profile stage only)")
    p.add_argument(
        "--case-id", action="append",
        help=("Profile exactly one canonical case; repeat for an explicit "
              "ordered subset. This selector cannot be mixed with family "
              "selectors."))
    p.add_argument(
        "--profile-mode",
        choices=PROFILE_MODES,
        default="op",
        help=("Profiler path for every selected family: 'op' preserves "
              "legacy OPPROF collection; 'timeline' captures exact public "
              "call schedules for timing-only candidates"))
    p.add_argument(
        "--vector-profile-mode",
        choices=VECTOR_PROFILE_MODES,
        help=("Deprecated compatibility alias for Vector-only invocations; "
              "use --profile-mode"))
    p.add_argument("--profile-cache-eviction-mib", type=int, default=0)
    p.add_argument(
        "--profile-sweep-operation", choices=("add", "fill"), default="fill",
        help=("FlashAttention cold-cache sweep operation: 'fill' writes only, "
              "'add' reads and writes so the sweep also displaces "
              "read-allocated lines"))
    p.add_argument(
        "--timeline-duration-seconds", type=int,
        help=("Maximum msprof Timeline collection duration for profile only; "
              "use when one-time runtime initialization exceeds the default "
              "capture window"))
    p.add_argument(
        "--timeline-candidate-output",
        help=("Write an explicit all-family Ascend Timeline timing-only "
              "candidate under profiler_output"))
    p.add_argument(
        "--timeline-window-run", action="append",
        help=("One named fixed-protocol Timeline capture window for "
              "--stage timeline-steady-timing-candidate; pass exactly three "
              "distinct values"))
    p.add_argument(
        "--capture-baseline-commit",
        help=("Exact capture baseline for --stage timeline-timing-candidate"))
    p.add_argument(
        "--source-snapshot-sha256",
        help=("Capture source snapshot SHA-256 for --stage "
              "timeline-timing-candidate"))
    p.add_argument(
        "--source-snapshot-archive",
        help=("Exact source archive whose SHA-256 must equal "
              "--source-snapshot-sha256 for --stage "
              "timeline-timing-candidate"))
    args = p.parse_args()

    cfg = load_pipeline_config(resolve_project_path(args.config))
    kernels = parse_kernel_list(cfg, args.kernel)
    profile_mode = args.profile_mode
    if args.vector_profile_mode is not None:
        if kernels != ["vector"]:
            p.error("--vector-profile-mode is valid only with --kernel vector")
        if profile_mode != "op" and profile_mode != args.vector_profile_mode:
            p.error("--profile-mode conflicts with --vector-profile-mode")
        profile_mode = args.vector_profile_mode

    if args.stage == "profile":
        if (args.timeline_candidate_output is not None
                or args.timeline_window_run is not None
                or args.capture_baseline_commit is not None
                or args.source_snapshot_sha256 is not None
                or args.source_snapshot_archive is not None):
            p.error(
                "Timeline candidate options require "
                "--stage timeline-timing-candidate")
        device_id = 0 if args.device_id is None else args.device_id
        if args.case_id is not None and len(kernels) != 1:
            p.error("--case-id requires exactly one --kernel family")
        for kernel in kernels:
            stage_profile(cfg, kernel, dry_run=args.dry_run, max_tasks=args.max_tasks,
                          start_idx=args.start_idx, end_idx=args.end_idx,
                          device_id=device_id,
                          profile_run=args.profile_run,
                          target_flash_attn_reprofile=(
                              args.target_flash_attn_reprofile),
                          target_vector_op=args.target_vector_op,
                          selected_case_ids=args.case_id,
                          profile_cache_eviction_mib=(
                              args.profile_cache_eviction_mib),
                          profile_sweep_operation=(
                              args.profile_sweep_operation),
                          profile_mode=profile_mode,
                          timeline_duration_seconds=(
                              args.timeline_duration_seconds))
    elif args.stage == "aggregate":
        if (args.timeline_candidate_output is not None
                or args.timeline_window_run is not None
                or args.capture_baseline_commit is not None
                or args.source_snapshot_sha256 is not None
                or args.source_snapshot_archive is not None):
            p.error(
                "Timeline candidate options require "
                "--stage timeline-timing-candidate")
        if args.dry_run:
            # --dry-run only applies to profile (aggregate parses real trace
            # output into the dataset JSON and would overwrite kernel_dataset/
            # <hardware>/<kernel>.json if allowed to run). Refuse to avoid
            # silently clobbering a good dataset.
            p.error("--dry-run is only valid with --stage profile")
        if (args.max_tasks or args.start_idx or args.end_idx
                or args.device_id is not None
                or args.profile_cache_eviction_mib
                or args.timeline_duration_seconds is not None):
            p.error(
                "profiling limits and device options are only valid with "
                "--stage profile")
        if (args.profile_run is None
                and (args.target_flash_attn_reprofile
                     or args.target_vector_op is not None)):
            p.error(
                "reprofile selectors require --profile-run during aggregate")
        for kernel in kernels:
            stage_aggregate(
                cfg, kernel, profile_run=args.profile_run,
                target_flash_attn_reprofile=(
                    args.target_flash_attn_reprofile),
                target_vector_op=args.target_vector_op,
                profile_mode=profile_mode)
    elif args.stage == "timeline-timing-candidate":
        if args.dry_run:
            p.error("--dry-run is not valid with --stage timeline-timing-candidate")
        if args.profile_run is None:
            p.error("timeline-timing-candidate requires --profile-run")
        if args.timeline_window_run is not None:
            p.error(
                "timeline-window-run is valid only with "
                "timeline-steady-timing-candidate")
        if profile_mode != "timeline":
            p.error(
                "timeline-timing-candidate requires "
                "--profile-mode timeline")
        if (args.max_tasks or args.start_idx or args.end_idx
                or args.device_id is not None
                or args.profile_cache_eviction_mib
                or args.timeline_duration_seconds is not None
                or args.target_flash_attn_reprofile
                or args.target_vector_op is not None):
            p.error(
                "profile selectors and device options are not valid with "
                "timeline-timing-candidate")
        if args.case_id is not None and len(kernels) != 1:
            p.error("--case-id requires exactly one --kernel family")
        if args.timeline_candidate_output is None:
            p.error(
                "timeline-timing-candidate requires "
                "--timeline-candidate-output")
        if args.capture_baseline_commit is None:
            p.error(
                "timeline-timing-candidate requires "
                "--capture-baseline-commit")
        if args.source_snapshot_sha256 is None:
            p.error(
                "timeline-timing-candidate requires "
                "--source-snapshot-sha256")
        if args.source_snapshot_archive is None:
            p.error(
                "timeline-timing-candidate requires "
                "--source-snapshot-archive")
        output = _write_timeline_timing_candidate(
            cfg, kernels, args.profile_run, args.capture_baseline_commit,
            args.source_snapshot_sha256, args.source_snapshot_archive,
            args.timeline_candidate_output,
            selected_case_ids=args.case_id)
        print(f"[Success] Ascend Timeline timing candidate: {output}")
    elif args.stage == "timeline-steady-timing-candidate":
        if args.dry_run:
            p.error(
                "--dry-run is not valid with "
                "--stage timeline-steady-timing-candidate")
        if args.profile_run is not None:
            p.error(
                "timeline-steady-timing-candidate uses "
                "--timeline-window-run, not --profile-run")
        if profile_mode != "timeline":
            p.error(
                "timeline-steady-timing-candidate requires "
                "--profile-mode timeline")
        if (args.max_tasks or args.start_idx or args.end_idx
                or args.device_id is not None
                or args.profile_cache_eviction_mib
                or args.timeline_duration_seconds is not None
                or args.target_flash_attn_reprofile
                or args.target_vector_op is not None):
            p.error(
                "profile selectors and device options are not valid with "
                "timeline-steady-timing-candidate")
        if args.case_id is not None and len(kernels) != 1:
            p.error("--case-id requires exactly one --kernel family")
        if args.timeline_candidate_output is None:
            p.error(
                "timeline-steady-timing-candidate requires "
                "--timeline-candidate-output")
        if args.timeline_window_run is None:
            p.error(
                "timeline-steady-timing-candidate requires exactly three "
                "--timeline-window-run values")
        if args.capture_baseline_commit is None:
            p.error(
                "timeline-steady-timing-candidate requires "
                "--capture-baseline-commit")
        if args.source_snapshot_sha256 is None:
            p.error(
                "timeline-steady-timing-candidate requires "
                "--source-snapshot-sha256")
        if args.source_snapshot_archive is None:
            p.error(
                "timeline-steady-timing-candidate requires "
                "--source-snapshot-archive")
        output = _write_timeline_steady_state_timing_candidate(
            cfg, kernels, args.timeline_window_run,
            args.capture_baseline_commit, args.source_snapshot_sha256,
            args.source_snapshot_archive, args.timeline_candidate_output,
            selected_case_ids=args.case_id)
        print(f"[Success] Ascend steady-state Timeline timing candidate: {output}")
    else:
        raise ValueError(f"unsupported dataset stage: {args.stage!r}")
