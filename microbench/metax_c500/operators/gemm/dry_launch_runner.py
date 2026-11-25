#!/usr/bin/env python3
"""Build and run the MetaX C500 mcBLAS host-selector dry-launch probe."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile


SCRIPT_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(SCRIPT_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPOSITORY_ROOT))

from dataset_pipeline.launch_data.gemm_workload_candidates import (  # noqa: E402
    GEMM_WORKLOAD_KEY_FIELDS,
    build_gemm_workload_candidate,
)


SCHEMA_VERSION = 2
HARDWARE_ID = "metax_c500"
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SOURCE_DIR = Path(__file__).resolve().parent
INTERPOSER_SOURCE = SOURCE_DIR / "dry_launch_interposer.cpp"
PROBE_SOURCE = SOURCE_DIR / "dry_launch_probe.cpp"
SOURCE_PATHS = (
    Path("microbench/metax_c500/operators/gemm/dry_launch_runner.py"),
    Path("microbench/metax_c500/operators/gemm/dry_launch_probe.cpp"),
    Path("microbench/metax_c500/operators/gemm/dry_launch_interposer.cpp"),
    Path("dataset_pipeline/launch_data/gemm_workload_candidates.py"),
    Path("src/kernel/launch_config/gemm_workload.py"),
    Path("src/kernel/launch_config/selection_primitives.py"),
)
OUTPUT_OWNER = Path("profiler_output/microbench/metax_c500")
REQUIRED_INTERPOSED_SYMBOLS = frozenset({
    "mcLaunchKernel",
    "mcModuleLaunchKernel",
})
BLOCKED_UNSUPPORTED_SYMBOLS = frozenset({
    "mcGraphLaunch",
    "mcLaunchCooperativeKernel",
})
SUPPORTED_DTYPE = "bfloat16"
SUPPORTED_LAYOUT = "row_major_linear_tn"
SUPPORTED_ALGORITHM = "MCBLAS_GEMM_DEFAULT"
DEFAULT_MODEL = "Qwen3-4B"
DEFAULT_VARIANT = "qkv_proj"
BULK_SCHEMA_VERSION = 1
SUPPORTED_POSITIONS = frozenset({
    "qkv_proj",
    "gate_up_proj",
    "o_proj",
    "down_proj",
    "lm_head",
    "router",
})
LAUNCH_COUNT_MISMATCH = re.compile(
    r"captured ([0-9]+) launch\(es\), expected ([0-9]+)"
)


def _run(command, *, env=None):
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with status {result.returncode}: {command!r}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _validate_maca_path(maca_path):
    maca_path = Path(maca_path).resolve()
    required = (
        maca_path / "include/mcr/mc_runtime.h",
        maca_path / "include/mcblas/mcblas.h",
        maca_path / "lib/libmcruntime.so",
        maca_path / "lib/libmcblas.so",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "MACA installation is incomplete: " + ", ".join(missing)
        )
    return maca_path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_repository_root(repository_root):
    repository_root = Path(repository_root).resolve()
    if not (repository_root / "pyproject.toml").is_file():
        raise FileNotFoundError(
            f"repository root has no pyproject.toml: {repository_root}"
        )
    expected_source_dir = (
        repository_root / "microbench/metax_c500/operators/gemm"
    ).resolve()
    if SOURCE_DIR != expected_source_dir:
        raise ValueError(
            "dry-launch runner source is outside --repository-root"
        )
    for relative in SOURCE_PATHS:
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"dry-launch source is absent: {path}")
    return repository_root


def _validate_source_identity(repository_root, baseline_commit):
    if COMMIT_PATTERN.fullmatch(baseline_commit) is None:
        raise ValueError(
            "--baseline-commit must be a full lowercase commit hash"
        )
    top_level = _run([
        "git", "-C", str(repository_root), "rev-parse", "--show-toplevel"
    ]) if (repository_root / ".git").exists() else None
    if top_level is not None \
            and Path(top_level.stdout.strip()).resolve() == repository_root:
        head = _run([
            "git", "-C", str(repository_root), "rev-parse", "HEAD"
        ]).stdout.strip()
        if head != baseline_commit:
            raise ValueError(
                "source commit mismatch: requested "
                f"{baseline_commit}, checkout {head}"
            )
        status = _run([
            "git", "-C", str(repository_root), "status", "--porcelain",
            "--untracked-files=no",
        ]).stdout
        if status:
            raise ValueError(
                "Git source checkout has tracked modifications: "
                + status.strip()
            )
        return {
            "mode": "clean_exact_git_checkout",
            "baseline_commit": baseline_commit,
            "marker_path": None,
        }
    marker = repository_root / ".source_commit"
    if not marker.is_file():
        raise FileNotFoundError(
            "non-Git source snapshot requires repository-root/.source_commit"
        )
    marker_text = marker.read_text(encoding="utf-8")
    if marker_text != baseline_commit + "\n":
        raise ValueError(
            ".source_commit must contain exactly --baseline-commit and newline"
        )
    return {
        "mode": "audited_source_snapshot",
        "baseline_commit": baseline_commit,
        "marker_path": str(marker.resolve()),
    }


def _validate_output_path(path, repository_root, label, suffix=None):
    resolved = Path(path).resolve()
    owner = (repository_root / OUTPUT_OWNER).resolve()
    try:
        relative = resolved.relative_to(owner)
    except ValueError as error:
        raise ValueError(
            f"{label} must be below {owner}"
        ) from error
    if not relative.parts:
        raise ValueError(f"{label} must name a path below {owner}")
    if suffix is not None and resolved.suffix != suffix:
        raise ValueError(f"{label} must have suffix {suffix}")
    return resolved


def _source_identities(repository_root):
    return {
        relative.as_posix(): _sha256(repository_root / relative)
        for relative in SOURCE_PATHS
    }


def _validate_candidate_path(path, repository_root):
    path = Path(path).resolve()
    owner = (repository_root / "profiler_output/launch_data").resolve()
    try:
        relative = path.relative_to(owner)
    except ValueError as error:
        raise ValueError(
            f"--candidate-manifest must be below {owner}"
        ) from error
    if not relative.parts or path.suffix != ".json":
        raise ValueError(
            "--candidate-manifest must name a JSON file below "
            f"{owner}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"candidate manifest is absent: {path}")
    return path


def _selector_operation(exact_key):
    values = json.loads(exact_key)
    workload_key = dict(zip(GEMM_WORKLOAD_KEY_FIELDS, values, strict=True))
    dtype = workload_key["dtype"]
    variant = workload_key["variant"]
    batch = workload_key["b"]
    m = workload_key["m"]
    n = workload_key["n"]
    k = workload_key["k"]
    if dtype != "bf16":
        raise ValueError(
            f"C500 dry launch requires BF16 exact key: {exact_key}"
        )
    if batch != 1:
        raise ValueError("C500 bulk dry launch supports exact-key B=1 only")
    if tuple(workload_key[name] for name in (
            "a_layout", "b_layout", "c_layout")) != (
            "row_major", "row_major", "row_major"):
        raise ValueError("C500 dry launch requires row-major exact keys")
    if workload_key["trans_a"] is not False \
            or workload_key["trans_b"] is not True:
        raise ValueError("C500 dry launch requires NT logical operands")
    if workload_key["epilogue"] != "none" \
            or workload_key["has_bias"] is not False:
        raise ValueError("C500 dry launch requires an unbiased exact key")
    position = workload_key["position"]
    if position not in SUPPORTED_POSITIONS:
        raise ValueError(
            f"C500 dry launch does not support position {position!r}"
        )
    expected_variant = (
        "gemv" if position == "lm_head" and m == 1 else "gemm"
    )
    if variant != expected_variant:
        raise ValueError(
            "exact-key variant conflicts with position/M semantics: "
            f"expected {expected_variant!r}, got {variant!r}"
        )
    for field in ("tensor_parallel_size", "expert_parallel_size"):
        value = workload_key[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"C500 dry launch requires positive {field}")
    operation_key = json.dumps(
        [batch, m, n, k, SUPPORTED_DTYPE, SUPPORTED_LAYOUT,
         SUPPORTED_ALGORITHM],
        separators=(",", ":"),
    )
    return operation_key, {
        "batch": batch,
        "m": m,
        "n": n,
        "k": k,
        "dtype": SUPPORTED_DTYPE,
        "layout": SUPPORTED_LAYOUT,
        "algorithm": SUPPORTED_ALGORITHM,
        "algorithm_value": -1,
    }


def _load_candidate_manifest(path, repository_root, models_root):
    path = _validate_candidate_path(path, repository_root)
    manifest_sha256 = _sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("candidate manifest must be an object")
    source_config = payload["source_config"]
    if not isinstance(source_config, str) or not source_config:
        raise TypeError("candidate source_config must be a non-empty string")
    source_config_relative = Path(source_config)
    if source_config_relative.is_absolute() \
            or source_config_relative.as_posix() != source_config \
            or "." in source_config_relative.parts \
            or ".." in source_config_relative.parts:
        raise ValueError(
            "candidate source_config must be a canonical relative path"
        )
    source_config_path = (repository_root / source_config_relative).resolve()
    try:
        source_config_path.relative_to(repository_root)
    except ValueError as error:
        raise ValueError(
            "candidate source_config must stay below repository root"
        ) from error
    if not source_config_path.is_file():
        raise FileNotFoundError(
            f"candidate source_config is absent: {source_config_path}"
        )
    authoritative = build_gemm_workload_candidate(
        repository_root, source_config_path, models_root=models_root
    )
    if payload != authoritative:
        raise ValueError(
            "candidate manifest differs from authoritative rebuild"
        )
    if payload["schema_version"] != 2:
        raise ValueError("candidate manifest schema_version must be 2")
    hardware_domain = payload["hardware_domains"][HARDWARE_ID]
    if hardware_domain["status"] != "exact":
        raise ValueError("C500 candidate workload domain is not exact")
    exact_keys = payload["workload_keys"]
    if (not isinstance(exact_keys, list) or not exact_keys
            or any(not isinstance(key, str) for key in exact_keys)
            or exact_keys != sorted(set(exact_keys))):
        raise ValueError("candidate workload_keys must be non-empty and exact")
    if hardware_domain["semantic_key_count"] != len(exact_keys):
        raise ValueError("C500 candidate semantic key count is inconsistent")
    operations = {}
    exact_key_map = {}
    for exact_key in exact_keys:
        operation_key, workload = _selector_operation(exact_key)
        operation_id = "sha256:" + hashlib.sha256(
            operation_key.encode("utf-8")
        ).hexdigest()
        if operation_id in operations:
            if operations[operation_id]["operation_key"] != operation_key:
                raise ValueError("selector operation SHA-256 collision")
        else:
            operations[operation_id] = {
                "operation_key": operation_key,
                "workload": workload,
            }
        exact_key_map[exact_key] = {
            "operation_id": operation_id,
            "result_id": operation_id,
        }
    return {
        "path": path,
        "sha256": manifest_sha256,
        "source_config_path": source_config_path,
        "payload": payload,
        "exact_keys": exact_keys,
        "operations": operations,
        "exact_key_map": exact_key_map,
    }


def build_artifacts(maca_path, compiler, build_dir):
    maca_path = _validate_maca_path(maca_path)
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    interposer = build_dir / "libmidlayer_c500_gemm_dry_launch.so"
    probe = build_dir / "midlayer_c500_gemm_dry_launch_probe"
    include_flags = (
        "-isystem", str(maca_path / "include/mcr"),
        "-isystem", str(maca_path / "include/mcblas"),
        "-isystem", str(maca_path / "include"),
    )
    _run([
        compiler,
        "-std=c++14",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fPIC",
        "-shared",
        *include_flags,
        str(INTERPOSER_SOURCE),
        "-ldl",
        "-pthread",
        "-o",
        str(interposer),
    ])
    _run([
        compiler,
        "-std=c++14",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        *include_flags,
        str(PROBE_SOURCE),
        f"-L{maca_path / 'lib'}",
        f"-Wl,-rpath,{maca_path / 'lib'}",
        "-lmcblas",
        "-lmcruntime",
        "-ldl",
        "-o",
        str(probe),
    ])
    symbols = _run(["nm", "-D", "--defined-only", str(interposer)]).stdout
    defined = {
        line.split()[-1]
        for line in symbols.splitlines()
        if line.split()
    }
    required_symbols = (
        REQUIRED_INTERPOSED_SYMBOLS | BLOCKED_UNSUPPORTED_SYMBOLS
    )
    missing_symbols = required_symbols - defined
    if missing_symbols:
        raise RuntimeError(
            "interposer is missing required launch API symbols: "
            + ", ".join(sorted(missing_symbols))
        )
    return interposer, probe


def _parse_capture(path, expected_launches):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid dry-launch JSON at line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise TypeError("dry-launch capture records must be mappings")
            records.append(record)
    if len(records) != expected_launches + 1:
        raise ValueError("dry-launch capture record count is inconsistent")
    launches = records[:-1]
    summary = records[-1]
    if summary["record_type"] != "summary":
        raise ValueError("dry-launch capture has no terminal summary")
    if summary["expected_launch_count"] != expected_launches:
        raise ValueError("dry-launch expected count changed across the process")
    if summary["captured_launch_count"] != expected_launches:
        raise ValueError("dry-launch captured count differs from the contract")
    for ordinal, launch in enumerate(launches, start=1):
        if launch["record_type"] != "launch" or launch["ordinal"] != ordinal:
            raise ValueError("dry-launch launch ordinals are inconsistent")
        if launch["api"] not in REQUIRED_INTERPOSED_SYMBOLS:
            raise ValueError(
                f"unexpected dry-launch API in capture: {launch['api']!r}"
            )
        if launch["identity_source"] not in {
                "mcModuleGetFunction", "mcFuncGetName", "dladdr"}:
            raise ValueError("dry-launch kernel identity is unresolved")
        if not isinstance(launch["kernel_identity"], str) \
                or not launch["kernel_identity"]:
            raise TypeError("dry-launch kernel identity must be non-empty")
        for field in ("grid", "block"):
            dimensions = launch[field]
            if not isinstance(dimensions, list) or len(dimensions) != 3:
                raise TypeError(f"dry-launch {field} must contain 3 dimensions")
            if any(not isinstance(value, int) or value <= 0
                   for value in dimensions):
                raise ValueError(
                    f"dry-launch {field} dimensions must be positive ints"
                )
        if not isinstance(launch["dynamic_shared_memory_bytes"], int) \
                or launch["dynamic_shared_memory_bytes"] < 0:
            raise ValueError(
                "dry-launch dynamic shared memory must be a non-negative int"
            )
        if launch["static_shared_memory_bytes"] is not None:
            raise ValueError(
                "launch API must not claim unobserved static shared memory"
            )
        if launch["static_shared_memory_status"] != \
                "not_exposed_by_launch_api":
            raise ValueError("static shared-memory provenance is invalid")
    return launches


def _parse_probe_metadata(stdout):
    match = re.fullmatch(r"mcblas_version=([0-9]+)\n?", stdout)
    if match is None:
        raise ValueError(f"unexpected dry-launch probe stdout: {stdout!r}")
    return int(match.group(1))


def _probe_environment(
        interposer, maca_path, device, capture_path,
        m, n, k, expected_launches):
    env = os.environ.copy()
    env["MIDLAYER_C500_DRY_LAUNCH_CAPTURE"] = str(capture_path)
    env["MIDLAYER_C500_GEMM_M"] = str(m)
    env["MIDLAYER_C500_GEMM_N"] = str(n)
    env["MIDLAYER_C500_GEMM_K"] = str(k)
    env["MIDLAYER_C500_EXPECTED_LAUNCHES"] = str(expected_launches)
    env["MACA_VISIBLE_DEVICES"] = str(device)
    existing_preload = None
    if "LD_PRELOAD" in env:
        existing_preload = env["LD_PRELOAD"]
    if existing_preload is None or existing_preload == "":
        env["LD_PRELOAD"] = str(interposer)
    else:
        raise ValueError(
            "LD_PRELOAD must be empty for an attributable dry launch"
        )
    existing_library_path = None
    if "LD_LIBRARY_PATH" in env:
        existing_library_path = env["LD_LIBRARY_PATH"]
    if existing_library_path is None or existing_library_path == "":
        env["LD_LIBRARY_PATH"] = str(maca_path / "lib")
    else:
        env["LD_LIBRARY_PATH"] = (
            f"{maca_path / 'lib'}:{existing_library_path}"
        )
    return env


def _raise_probe_failure(command, result):
    raise RuntimeError(
        f"command failed with status {result.returncode}: {command!r}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _execute_operation(
        interposer, probe, maca_path, build_dir, device,
        m, n, k, expected_launches, allow_positive_count_retry):
    if expected_launches <= 0:
        raise ValueError("expected launch count must be positive")
    with tempfile.TemporaryDirectory(
            prefix="c500-gemm-dry-launch-", dir=build_dir) as temporary:
        capture_path = Path(temporary) / "capture.jsonl"

        def execute_once(current_expected):
            env = _probe_environment(
                interposer, maca_path, device, capture_path,
                m, n, k, current_expected,
            )
            command = [str(probe)]
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            if result.returncode == 0:
                return {
                    "launches": _parse_capture(
                        capture_path, current_expected
                    ),
                    "mcblas_version": _parse_probe_metadata(result.stdout),
                }
            return result

        first = execute_once(expected_launches)
        if isinstance(first, dict):
            return {
                **first,
                "selector_attempt_count": 1,
                "initial_expected_launch_count": expected_launches,
                "validated_launch_count": expected_launches,
                "stability_capture_count": 1,
            }
        mismatches = LAUNCH_COUNT_MISMATCH.findall(first.stderr)
        if not (allow_positive_count_retry and len(mismatches) == 1):
            _raise_probe_failure([str(probe)], first)
        observed = int(mismatches[0][0])
        reported_expected = int(mismatches[0][1])
        if reported_expected != expected_launches:
            raise ValueError(
                "dry-launch mismatch diagnostic changed expected count"
            )
        if observed <= 0 or observed == expected_launches:
            _raise_probe_failure([str(probe)], first)
        confirmations = []
        for _ in range(2):
            confirmation = execute_once(observed)
            if not isinstance(confirmation, dict):
                _raise_probe_failure([str(probe)], confirmation)
            confirmations.append(confirmation)
        if confirmations[0] != confirmations[1]:
            raise ValueError(
                "dry-launch sequence changed across count-stability captures"
            )
        return {
            **confirmations[0],
            "selector_attempt_count": 3,
            "initial_expected_launch_count": expected_launches,
            "validated_launch_count": observed,
            "stability_capture_count": 2,
        }


def _execute_operations_batch(
        interposer, probe, maca_path, build_dir, device, operations,
        expected_launches):
    if expected_launches <= 0:
        raise ValueError("expected launch count must be positive")
    with tempfile.TemporaryDirectory(
            prefix="c500-gemm-dry-launch-batch-", dir=build_dir) as temporary:
        temporary_path = Path(temporary)
        requests = []
        capture_paths = {}
        for index, (operation_id, record) in enumerate(operations.items()):
            workload = record["workload"]
            capture_path = temporary_path / f"capture_{index:08d}.jsonl"
            capture_paths[operation_id] = capture_path
            requests.append(
                f"{operation_id} {workload['m']} {workload['n']} "
                f"{workload['k']} {expected_launches} {capture_path}\n")
        first = next(iter(operations.values()))["workload"]
        environment = _probe_environment(
            interposer, maca_path, device,
            next(iter(capture_paths.values())), first["m"], first["n"],
            first["k"], expected_launches)
        result = subprocess.run(
            [str(probe), "--batch"], input="".join(requests), check=False,
            capture_output=True, text=True, env=environment)
        if result.returncode != 0:
            _raise_probe_failure([str(probe), "--batch"], result)
        lines = result.stdout.splitlines()
        if len(lines) != len(operations):
            raise ValueError(
                "persistent C500 probe returned a different result count")
        outputs = {}
        for line, expected_operation_id in zip(lines, operations):
            fields = line.split("\t")
            if len(fields) != 5 or fields[0] != expected_operation_id:
                raise ValueError(
                    "persistent C500 probe result identity/order changed")
            version, attempts, validated, stability = (
                int(value) for value in fields[1:])
            capture_path = capture_paths[expected_operation_id]
            if stability == 1:
                launches = _parse_capture(capture_path, validated)
            elif stability == 2:
                first_launches = _parse_capture(
                    Path(str(capture_path) + ".stability1"), validated)
                second_launches = _parse_capture(
                    Path(str(capture_path) + ".stability2"), validated)
                if first_launches != second_launches:
                    raise ValueError(
                        "dry-launch sequence changed across count-stability "
                        "captures")
                launches = first_launches
            else:
                raise ValueError("persistent C500 probe stability count changed")
            outputs[expected_operation_id] = {
                "launches": launches,
                "mcblas_version": version,
                "selector_attempt_count": attempts,
                "initial_expected_launch_count": expected_launches,
                "validated_launch_count": validated,
                "stability_capture_count": stability,
            }
        return outputs


def build_report(args):
    if args.start_index != 0 or args.end_index is not None:
        raise ValueError(
            "--start-index/--end-index require --candidate-manifest"
        )
    if args.batch != 1:
        raise ValueError("the GEMM dry-launch PoC supports batch=1 only")
    if args.m <= 0 or args.n <= 0 or args.k <= 0:
        raise ValueError("GEMM M, N, and K must be positive")
    if args.device < 0:
        raise ValueError("device must be non-negative")
    if args.dtype != SUPPORTED_DTYPE:
        raise ValueError(f"unsupported dtype: {args.dtype!r}")
    if args.layout != SUPPORTED_LAYOUT:
        raise ValueError(f"unsupported layout: {args.layout!r}")
    if args.algorithm != SUPPORTED_ALGORITHM:
        raise ValueError(f"unsupported algorithm: {args.algorithm!r}")
    if args.expected_launches <= 0:
        raise ValueError("expected launch count must be positive")
    repository_root = _validate_repository_root(args.repository_root)
    source_identity = _validate_source_identity(
        repository_root, args.baseline_commit
    )
    source_identities = _source_identities(repository_root)
    maca_path = _validate_maca_path(args.maca_path)
    build_dir = _validate_output_path(
        args.build_dir, repository_root, "--build-dir"
    )
    output_path = _validate_output_path(
        args.output, repository_root, "--output", suffix=".json"
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_path}")
    interposer, probe = build_artifacts(
        maca_path, args.compiler, build_dir
    )
    operation = _execute_operation(
        interposer, probe, maca_path, build_dir, args.device,
        args.m, args.n, args.k, args.expected_launches,
        allow_positive_count_retry=False,
    )
    launches = operation["launches"]
    mcblas_version = operation["mcblas_version"]
    source_identity_after = _validate_source_identity(
        repository_root, args.baseline_commit
    )
    if source_identity_after != source_identity:
        raise RuntimeError("source identity changed during dry launch")
    if _source_identities(repository_root) != source_identities:
        raise RuntimeError("dry-launch source SHA-256 changed during execution")
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": "mcblas_default_host_selector_dry_launch",
        "gemm_executed": False,
        "duration_consumed": False,
        "production_or_evaluation_duration_consumed": False,
        "workload": {
            "model": args.model,
            "operator": "linear",
            "variant": args.variant,
            "batch": args.batch,
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "dtype": args.dtype,
            "layout": args.layout,
            "algorithm": args.algorithm,
            "algorithm_value": -1,
            "mcblas_column_major_call": {
                "transa": "MCBLAS_OP_T",
                "transb": "MCBLAS_OP_N",
                "m": args.n,
                "n": args.m,
                "k": args.k,
                "lda": args.k,
                "ldb": args.k,
                "ldc": args.n,
                "compute_type": "MCBLAS_COMPUTE_32F",
            },
        },
        "environment": {
            "source_identity": source_identity,
            "source_sha256": source_identities,
            "maca_path": str(maca_path),
            "mcblas_version": mcblas_version,
            "device": args.device,
            "compiler": args.compiler,
            "host_platform": platform.platform(),
        },
        "interposition": {
            "required_launch_apis": sorted(REQUIRED_INTERPOSED_SYMBOLS),
            "unsupported_execution_apis_blocked": sorted(
                BLOCKED_UNSUPPORTED_SYMBOLS
            ),
            "expected_launch_count": args.expected_launches,
            "captured_launch_count": len(launches),
            "initialization_launch_policy": "forwarded_outside_active_marker",
            "active_marker_launch_policy": "recorded_and_not_submitted",
        },
        "launches": launches,
    }


def build_bulk_report(args):
    ignored_overrides = []
    for flag, actual, expected in (
        ("--model", args.model, DEFAULT_MODEL),
        ("--variant", args.variant, DEFAULT_VARIANT),
        ("--batch", args.batch, 1),
        ("--m", args.m, 729),
        ("--n", args.n, 6144),
        ("--k", args.k, 2560),
        ("--dtype", args.dtype, SUPPORTED_DTYPE),
        ("--layout", args.layout, SUPPORTED_LAYOUT),
        ("--algorithm", args.algorithm, SUPPORTED_ALGORITHM),
    ):
        if actual != expected:
            ignored_overrides.append(flag)
    if ignored_overrides:
        raise ValueError(
            "single-shape options cannot be overridden with "
            "--candidate-manifest: " + ", ".join(ignored_overrides)
        )
    if args.device < 0:
        raise ValueError("device must be non-negative")
    if args.expected_launches <= 0:
        raise ValueError("expected launch count must be positive")
    repository_root = _validate_repository_root(args.repository_root)
    source_identity = _validate_source_identity(
        repository_root, args.baseline_commit
    )
    source_identities = _source_identities(repository_root)
    candidate = _load_candidate_manifest(
        args.candidate_manifest, repository_root, args.models_root
    )
    operations = candidate["operations"]
    full_operation_ids = sorted(operations)
    full_operation_count = len(full_operation_ids)
    end_index = (
        full_operation_count
        if args.end_index is None
        else args.end_index
    )
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if end_index <= args.start_index:
        raise ValueError(
            "--end-index must be greater than --start-index"
        )
    if end_index > full_operation_count:
        raise ValueError(
            "--end-index exceeds full selector operation count "
            f"{full_operation_count}"
        )
    selected_operation_ids = full_operation_ids[args.start_index:end_index]
    selected_operation_id_set = set(selected_operation_ids)
    selected_operations = {
        operation_id: operations[operation_id]
        for operation_id in selected_operation_ids
    }
    selected_exact_key_map = {
        exact_key: mapping
        for exact_key, mapping in candidate["exact_key_map"].items()
        if mapping["operation_id"] in selected_operation_id_set
    }
    maca_path = _validate_maca_path(args.maca_path)
    build_dir = _validate_output_path(
        args.build_dir, repository_root, "--build-dir"
    )
    output_path = _validate_output_path(
        args.output, repository_root, "--output", suffix=".json"
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_path}")
    interposer, probe = build_artifacts(
        maca_path, args.compiler, build_dir
    )
    mcblas_versions = set()
    total_attempts = 0
    total_launches = 0
    selector_results = {}
    batch_results = _execute_operations_batch(
        interposer, probe, maca_path, build_dir, args.device,
        selected_operations, args.expected_launches)
    for operation_id in selected_operation_ids:
        record = selected_operations[operation_id]
        result = batch_results[operation_id]
        mcblas_versions.add(result["mcblas_version"])
        total_attempts += result["selector_attempt_count"]
        total_launches += len(result["launches"])
        selector_results[operation_id] = {
            "gemm_executed": False,
            "duration_consumed": False,
            "selector_attempt_count": result["selector_attempt_count"],
            "initial_expected_launch_count": result[
                "initial_expected_launch_count"],
            "validated_launch_count": result["validated_launch_count"],
            "stability_capture_count": result["stability_capture_count"],
            "launches": result["launches"],
        }
    if len(mcblas_versions) != 1:
        raise ValueError(
            "mcBLAS version changed across selector operations: "
            f"{sorted(mcblas_versions)}"
        )
    source_identity_after = _validate_source_identity(
        repository_root, args.baseline_commit
    )
    if source_identity_after != source_identity:
        raise RuntimeError("source identity changed during bulk dry launch")
    if _source_identities(repository_root) != source_identities:
        raise RuntimeError("dry-launch source SHA-256 changed during execution")
    if _sha256(candidate["path"]) != candidate["sha256"]:
        raise RuntimeError("candidate manifest changed during dry launch")
    if build_gemm_workload_candidate(
            repository_root, candidate["source_config_path"],
            models_root=args.models_root
    ) != candidate["payload"]:
        raise RuntimeError(
            "authoritative candidate inputs changed during dry launch"
        )
    mcblas_version = next(iter(mcblas_versions))
    return {
        "schema_version": BULK_SCHEMA_VERSION,
        "artifact_type": "metax_c500_mcblas_gemm_dry_launch_candidate",
        "hardware_id": HARDWARE_ID,
        "mechanism": "mcblas_default_host_selector_dry_launch",
        "gemm_executed": False,
        "duration_consumed": False,
        "production_or_evaluation_duration_consumed": False,
        "input_manifest": {
            "path": str(candidate["path"].relative_to(repository_root)),
            "sha256": candidate["sha256"],
            "schema_version": candidate["payload"]["schema_version"],
            "artifact_type": candidate["payload"]["artifact_type"],
            "source_config": candidate["payload"]["source_config"],
            "source_config_sha256": candidate["payload"][
                "source_config_sha256"],
            "workload_key_fields": list(GEMM_WORKLOAD_KEY_FIELDS),
            "hardware_id": HARDWARE_ID,
            "exact_key_count": len(candidate["exact_keys"]),
            "selected_exact_key_count": len(selected_exact_key_map),
        },
        "environment": {
            "source_identity": source_identity,
            "source_sha256": source_identities,
            "maca_path": str(maca_path),
            "mcblas_version": mcblas_version,
            "device": args.device,
            "compiler": args.compiler,
            "host_platform": platform.platform(),
        },
        "interposition": {
            "required_launch_apis": sorted(REQUIRED_INTERPOSED_SYMBOLS),
            "unsupported_execution_apis_blocked": sorted(
                BLOCKED_UNSUPPORTED_SYMBOLS
            ),
            "initial_expected_launch_count": args.expected_launches,
            "positive_count_mismatch_policy": (
                "discover_positive_count_then_require_two_exact_captures"
            ),
            "zero_launch_policy": "fail_loud_without_retry",
            "initialization_launch_policy": "forwarded_outside_active_marker",
            "active_marker_launch_policy": "recorded_and_not_submitted",
            "full_selector_operation_count": full_operation_count,
            "selected_selector_operation_count": len(
                selected_operation_ids
            ),
            "selector_operation_order": "lexicographic_operation_id",
            "selector_operation_range": {
                "start_index": args.start_index,
                "end_index": end_index,
            },
            "selector_attempt_count": total_attempts,
            "captured_launch_count": total_launches,
        },
        "selector_operations": selected_operations,
        "selector_results": selector_results,
        "exact_key_map": selected_exact_key_map,
    }


def _write_report(path, report):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--maca-path", type=Path, required=True)
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--m", type=int, default=729)
    parser.add_argument("--n", type=int, default=6144)
    parser.add_argument("--k", type=int, default=2560)
    parser.add_argument("--dtype", default=SUPPORTED_DTYPE)
    parser.add_argument("--layout", default=SUPPORTED_LAYOUT)
    parser.add_argument("--algorithm", default=SUPPORTED_ALGORITHM)
    parser.add_argument("--expected-launches", type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.candidate_manifest is None:
        if args.models_root is not None:
            raise ValueError("--models-root requires --candidate-manifest")
        report = build_report(args)
    else:
        if args.models_root is None:
            raise ValueError("--candidate-manifest requires --models-root")
        report = build_bulk_report(args)
    _write_report(args.output, report)
    summary = {
        "captured_launch_count": report["interposition"][
            "captured_launch_count"],
        "gemm_executed": report["gemm_executed"],
        "output": str(Path(args.output).resolve()),
    }
    if args.candidate_manifest is not None:
        summary["exact_key_count"] = report["input_manifest"][
            "exact_key_count"]
        summary["selected_exact_key_count"] = report["input_manifest"][
            "selected_exact_key_count"]
        summary["full_selector_operation_count"] = report[
            "interposition"]["full_selector_operation_count"]
        summary["selected_selector_operation_count"] = report[
            "interposition"]["selected_selector_operation_count"]
        summary["selector_operation_range"] = report["interposition"][
            "selector_operation_range"]
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
