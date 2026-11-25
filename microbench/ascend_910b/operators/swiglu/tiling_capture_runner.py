#!/usr/bin/env python3
"""Capture exact-domain 910B SwiGLU tiling payload shards without timing."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

import tiling_capture_report


BASELINE_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", "ascend_910b")
INTERPOSER_SOURCE = (
    "microbench/ascend_910b/operators/swiglu/tiling_capture_interposer.cpp")
RUNNER_SOURCE = (
    "microbench/ascend_910b/operators/swiglu/tiling_capture_runner.py")
REPORT_SOURCE = (
    "microbench/ascend_910b/operators/swiglu/tiling_capture_report.py")
APPLICATION_SOURCE = (
    "microbench/ascend_910b/operators/swiglu/"
    "tiling_capture_application_runner.py")
SOURCE_FILES = (
    INTERPOSER_SOURCE,
    RUNNER_SOURCE,
    REPORT_SOURCE,
    APPLICATION_SOURCE,
    "dataset_pipeline/launch_data/gemm_workload_candidates.py",
    "src/kernel/launch_config/gemm_workload.py",
    "src/kernel/launch_config/selection_primitives.py",
)
CANN_RELATIVE_FILES = (
    "opp/version.info",
    "aarch64-linux/include/acl/acl_rt.h",
    "aarch64-linux/include/aclnn/acl_meta.h",
    "aarch64-linux/include/aclnnop/aclnn_swi_glu.h",
    "aarch64-linux/lib64/libacl_rt.so",
    "aarch64-linux/lib64/libnnopbase.so",
    "aarch64-linux/lib64/libopapi_nn.so",
    "aarch64-linux/lib64/libregister.so",
    "opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/aarch64/"
    "libophost_nn.so",
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/swi_glu/"
    "swi_glu.cpp",
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/swi_glu/"
    "swi_glu_bf16.hpp",
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/swi_glu/"
    "swi_glu_single.hpp",
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/swi_glu/"
    "glu_tiling_kernel.hpp",
    "opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/ops_nn/swi_glu/"
    "SwiGlu_3_high_performance.o",
    "opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910b/ops_nn/"
    "swi_glu.json",
)
LAUNCH_IMPORT_RELATIVE_FILES = (
    "aarch64-linux/lib64/libnnopbase.so",
    "aarch64-linux/lib64/libopapi_nn.so",
    "aarch64-linux/lib64/libregister.so",
    "opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/aarch64/"
    "libophost_nn.so",
)
EXPECTED_IMPORTED_LAUNCH_APIS = ("aclrtLaunchKernelWithHostArgs",)
INTERCEPTED_LAUNCH_APIS = (
    "aclrtLaunchKernel",
    "aclrtLaunchKernelV2",
    "aclrtLaunchKernelWithConfig",
    "aclrtLaunchKernelWithHostArgs",
)
LAUNCH_IMPORT_PATTERN = re.compile(
    r"\bUND\b[^\n]*\b(aclrtLaunchKernel[A-Za-z0-9_]*)")
SUPPORTED_MANIFEST_DTYPE = "bf16"
SUPPORTED_MANIFEST_LAYOUT = "ND"
SUPPORTED_MANIFEST_DIM = -1
EXPECTED_DENSE_MODEL_IDS = {
    "Llama-3.1-8B", "Qwen3-4B", "Qwen3-8B", "Qwen3-32B",
}
MODEL_DERIVED_INTERMEDIATES = (9728, 12288, 12800, 14336)
ADDITIONAL_SERVING_INTERMEDIATES = (6144, 25600, 28672, 53248)
EXACT_SWIGLU_INTERMEDIATES = tuple(sorted(
    MODEL_DERIVED_INTERMEDIATES + ADDITIONAL_SERVING_INTERMEDIATES))
EXPECTED_SWIGLU_WORKLOAD_COUNT = len(EXACT_SWIGLU_INTERMEDIATES) * 8192


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def _require_integer(value, context, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{context} must be positive")
    return value


def load_case_contract(dataset_path, expected_sha256, case_key):
    if SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError("canonical dataset SHA256 must be lowercase hex")
    actual_sha256 = _sha256(dataset_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "canonical dataset SHA256 mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict):
        raise TypeError("canonical vector dataset must be a mapping")
    entry = dataset[case_key]
    if not isinstance(entry, dict):
        raise TypeError("canonical SwiGlu case must be a mapping")
    op_info = entry["op_info"]
    if op_info["Type"] != "ACTIVATION":
        raise ValueError("canonical SwiGlu case must have ACTIVATION type")
    attributes = op_info["Attr"]
    expected_attribute_keys = {
        "op", "op2", "tok", "hidden", "num_heads", "head_dim",
        "intermediate", "num_kv_heads", "batch", "element_count",
        "dtype_bytes", "layer",
    }
    if set(attributes) != expected_attribute_keys:
        raise ValueError(
            f"canonical SwiGlu attributes changed: {sorted(attributes)}")
    if attributes["op"] != "silu" or attributes["op2"] != "add":
        raise ValueError("canonical case is not the npu_swiglu workload")
    kernels = entry["kernels"]
    if not isinstance(kernels, list) or len(kernels) != 1:
        raise ValueError("canonical SwiGlu case must contain one kernel")
    kernel = kernels[0]
    kernel_name = kernel["Metadata"]["Kernel Name"]
    if kernel_name != "SwiGlu_3_high_performance_27":
        raise ValueError(f"unexpected canonical SwiGlu kernel: {kernel_name!r}")
    block_dim = _require_integer(
        kernel["Ascend"]["block_dim"], "canonical block_dim", positive=True)
    integer_fields = (
        "tok", "hidden", "num_heads", "head_dim", "intermediate",
        "num_kv_heads", "batch", "element_count", "dtype_bytes",
    )
    normalized = {
        name: _require_integer(
            attributes[name], f"canonical {name}", positive=True)
        for name in integer_fields
    }
    if normalized["dtype_bytes"] != 2:
        raise ValueError("SwiGlu tiling capture requires canonical BF16")
    if (normalized["element_count"]
            != normalized["tok"] * normalized["intermediate"]):
        raise ValueError("canonical SwiGlu element_count is inconsistent")
    if not isinstance(attributes["layer"], str) or not attributes["layer"]:
        raise TypeError("canonical layer must be a non-empty string")
    return {
        "case_key": case_key,
        "canonical_dataset_path": str(dataset_path.resolve()),
        "canonical_dataset_sha256": actual_sha256,
        "kernel_name": kernel_name,
        "block_dim": block_dim,
        "dtype": "bf16",
        "layer": attributes["layer"],
        **normalized,
    }


def _workload_key(tok, intermediate):
    return f"VECTOR_SILU_T{tok}_I{intermediate}_B1"


def _repository_file(repository_root, relative_path, expected_sha256, context):
    if (not isinstance(relative_path, str) or not relative_path
            or Path(relative_path).is_absolute()):
        raise ValueError(f"{context} must be a repository-relative path")
    path = (repository_root / relative_path).resolve()
    if repository_root not in path.parents:
        raise ValueError(f"{context} escapes the repository: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(f"{context} is missing: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{context} SHA256 mismatch: expected={expected_sha256}, "
            f"actual={actual_sha256}")
    return path


def _build_authoritative_candidate(repository_root, config_path, models_root):
    repository_root_text = str(repository_root)
    if repository_root_text not in sys.path:
        sys.path.insert(0, repository_root_text)
    from dataset_pipeline.launch_data.gemm_workload_candidates import (  # noqa: PLC0415
        build_gemm_workload_candidate,
    )
    return build_gemm_workload_candidate(
        repository_root, config_path, models_root=models_root)


def _exact_swiglu_keys(candidate):
    if not isinstance(candidate, dict):
        raise TypeError("authoritative GEMM workload candidate must be a mapping")
    if (candidate["schema_version"] != 2
            or candidate["artifact_type"] != "gemm_workload_candidate"):
        raise ValueError("SwiGlu requires GEMM workload candidate schema 2")
    if candidate["duration_consumed"] is not False:
        raise ValueError("SwiGlu candidate must not consume duration")
    hardware = candidate["hardware_domains"]["ascend_910b"]
    if hardware["status"] != "exact":
        raise ValueError("SwiGlu requires an exact Ascend GEMM domain")
    model_domains = candidate["model_domains"]
    dense_models = {
        model["model_id"] for model in model_domains
        if any(position["position"] == "gate_up_proj"
               for position in model["positions"])
    }
    if dense_models != EXPECTED_DENSE_MODEL_IDS:
        raise ValueError(
            "SwiGlu gate_up model set differs from the four dense models")
    encoded_gemm_keys = candidate["workload_keys"]
    if (not isinstance(encoded_gemm_keys, list)
            or encoded_gemm_keys != sorted(set(encoded_gemm_keys))):
        raise ValueError("authoritative GEMM workload keys must be exact")
    swiglu_keys = set()
    for encoded_key in encoded_gemm_keys:
        workload = json.loads(encoded_key)
        if not isinstance(workload, list) or len(workload) != 16:
            raise ValueError("authoritative GEMM workload key shape changed")
        if workload[13] != "gate_up_proj":
            continue
        if (workload[0] != SUPPORTED_MANIFEST_DTYPE
                or workload[1] != "gemm" or workload[2] != 1
                or workload[6:9] != ["row_major"] * 3
                or workload[9] is not False or workload[10] is not True
                or workload[11] != "none" or workload[12] is not False):
            raise ValueError("gate_up_proj workload semantics are unsupported")
        b = _require_integer(workload[2], "gate_up_proj b", positive=True)
        m = _require_integer(workload[3], "gate_up_proj m", positive=True)
        gate_up_width = _require_integer(
            workload[4], "gate_up_proj n", positive=True)
        if gate_up_width % 2 != 0:
            raise ValueError("gate_up_proj width must be even")
        swiglu_keys.add(json.dumps(
            [b * m, gate_up_width // 2], separators=(",", ":")))
    model_derived_keys = set(swiglu_keys)
    model_derived_intermediates = {
        json.loads(encoded_key)[1] for encoded_key in model_derived_keys}
    if model_derived_intermediates != set(MODEL_DERIVED_INTERMEDIATES):
        raise ValueError(
            "authoritative candidate SwiGLU intermediate domain changed: "
            f"expected={list(MODEL_DERIVED_INTERMEDIATES)}, "
            f"actual={sorted(model_derived_intermediates)}")
    for tok in range(1, 8193):
        for intermediate in ADDITIONAL_SERVING_INTERMEDIATES:
            swiglu_keys.add(json.dumps(
                [tok, intermediate], separators=(",", ":")))
    if len(swiglu_keys) != EXPECTED_SWIGLU_WORKLOAD_COUNT:
        raise ValueError(
            "expanded SwiGLU domain must contain exactly 65,536 workloads, "
            f"got {len(swiglu_keys)}")
    by_intermediate = {}
    for encoded_key in swiglu_keys:
        tok, intermediate = json.loads(encoded_key)
        if intermediate not in by_intermediate:
            by_intermediate[intermediate] = set()
        by_intermediate[intermediate].add(tok)
    if (tuple(sorted(by_intermediate)) != EXACT_SWIGLU_INTERMEDIATES
            or any(values != set(range(1, 8193))
                   for values in by_intermediate.values())):
        raise ValueError(
            "each exact SwiGLU intermediate must cover tok=1..8192 exactly")
    return sorted(swiglu_keys, key=lambda key: tuple(json.loads(key)))


def load_workload_manifest(
        manifest_path, expected_sha256, repository_root, models_root):
    repository_root = repository_root.resolve()
    if not repository_root.is_dir():
        raise FileNotFoundError(f"repository root is missing: {repository_root}")
    if SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError("workload manifest SHA256 must be lowercase hex")
    actual_sha256 = _sha256(manifest_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "workload manifest SHA256 mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise TypeError("SwiGlu GEMM candidate must be a mapping")
    source_config_path = _repository_file(
        repository_root, manifest["source_config"],
        manifest["source_config_sha256"], "SwiGlu source config")
    authoritative_candidate = _build_authoritative_candidate(
        repository_root, source_config_path, models_root)
    if manifest != authoritative_candidate:
        raise ValueError(
            "SwiGlu GEMM candidate differs from authoritative rebuild")
    workload_keys = _exact_swiglu_keys(manifest)
    contracts = []
    for index, encoded_key in enumerate(workload_keys):
        try:
            workload = json.loads(encoded_key)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"SwiGlu workload key {index} is not JSON") from error
        if not isinstance(workload, list) or len(workload) != 2:
            raise ValueError(
                f"SwiGlu workload key {index} must be [tok,intermediate]")
        if json.dumps(workload, separators=(",", ":")) != encoded_key:
            raise ValueError(
                f"SwiGlu workload key {index} is not canonical JSON")
        tok = _require_integer(
            workload[0], f"SwiGlu workload {index} tok", positive=True)
        intermediate = _require_integer(
            workload[1],
            f"SwiGlu workload {index} intermediate", positive=True)
        case_key = _workload_key(tok, intermediate)
        contracts.append({
            "case_key": case_key,
            "capture_input": "workload_manifest",
            "block_dim": None,
            "kernel_name": None,
            "dtype": SUPPORTED_MANIFEST_DTYPE,
            "layout": SUPPORTED_MANIFEST_LAYOUT,
            "dim": SUPPORTED_MANIFEST_DIM,
            "tok": tok,
            "intermediate": intermediate,
            "element_count": tok * intermediate,
            "dtype_bytes": 2,
            "hidden": 1,
            "num_heads": 1,
            "head_dim": 1,
            "num_kv_heads": 1,
            "batch": 1,
            "layer": None,
        })
    contracts.sort(key=lambda contract: (
        contract["tok"], contract["intermediate"]))
    frontier_workload_keys = [contract["case_key"] for contract in contracts]
    identity = {
        "path": str(manifest_path.resolve()),
        "sha256": actual_sha256,
        "schema_version": manifest["schema_version"],
        "artifact_type": manifest["artifact_type"],
        "duration_consumed": manifest["duration_consumed"],
        "source_config": manifest["source_config"],
        "source_config_sha256": manifest["source_config_sha256"],
        "workload_keys": frontier_workload_keys,
        "encoded_swiglu_keys": workload_keys,
        "coverage_claim": "complete_exact_swiglu_launch_domain",
    }
    return contracts, identity


def verify_source_archive(archive_path, repository_root, baseline_commit):
    if BASELINE_PATTERN.fullmatch(baseline_commit) is None:
        raise ValueError("baseline commit must be 40 lowercase hex")
    try:
        archive = tarfile.open(archive_path, "r:*")
    except tarfile.TarError as error:
        raise ValueError(f"unreadable source archive: {archive_path}") from error
    identities = []
    with archive:
        if "comment" not in archive.pax_headers:
            raise ValueError("source snapshot is not a Git archive")
        archive_commit = archive.pax_headers["comment"]
        if archive_commit != baseline_commit:
            raise ValueError(
                f"archive commit {archive_commit} differs from baseline")
        members = [member for member in archive.getmembers() if member.isfile()]
        for relative in SOURCE_FILES:
            matches = [
                member for member in members
                if member.name == relative or member.name.endswith(f"/{relative}")
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"source archive must contain one {relative}: {len(matches)}")
            stream = archive.extractfile(matches[0])
            if stream is None:
                raise ValueError(f"cannot read archived source {relative}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            current = repository_root / relative
            if not current.is_file() or _sha256(current) != digest.hexdigest():
                raise ValueError(f"deployed source differs from archive: {relative}")
            identities.append({
                "path": str(current.resolve()),
                "relative_path": relative,
                "sha256": digest.hexdigest(),
            })
    return {
        "archive_path": str(archive_path.resolve()),
        "archive_sha256": _sha256(archive_path),
        "git_archive_commit": archive_commit,
        "verified_sources": identities,
    }


def collect_cann_provenance(cann_root):
    identities = []
    for relative in CANN_RELATIVE_FILES:
        path = cann_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required CANN evidence is missing: {path}")
        identities.append({
            "path": str(path.resolve()),
            "relative_path": relative,
            "sha256": _sha256(path),
        })
    version_path = cann_root / "opp/version.info"
    if "Version=8.5.1" not in version_path.read_text(
            encoding="utf-8").splitlines():
        raise ValueError("SwiGlu capture requires exact CANN Version=8.5.1")
    return {
        "cann_root": str(cann_root.resolve()),
        "files": identities,
    }


def audit_launch_imports(readelf, cann_root):
    audited_files = []
    imported_apis = set()
    for relative in LAUNCH_IMPORT_RELATIVE_FILES:
        path = (cann_root / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"launch-import audit target is missing: {path}")
        completed = subprocess.run(
            [str(readelf), "--wide", "--dyn-syms", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"readelf failed for {path}: {completed.stderr.strip()}")
        file_imports = sorted(set(
            LAUNCH_IMPORT_PATTERN.findall(completed.stdout)))
        imported_apis.update(file_imports)
        audited_files.append({
            "path": str(path),
            "relative_path": relative,
            "sha256": _sha256(path),
            "imported_launch_apis": file_imports,
        })
    if imported_apis != set(EXPECTED_IMPORTED_LAUNCH_APIS):
        raise ValueError(
            "SwiGlu vendor libraries must import exactly the audited launch "
            f"API set: expected={list(EXPECTED_IMPORTED_LAUNCH_APIS)}, "
            f"actual={sorted(imported_apis)}")
    return {
        "tool": {"path": str(readelf), "sha256": _sha256(readelf)},
        "audited_files": audited_files,
        "imported_launch_apis": sorted(imported_apis),
        "intercepted_launch_apis": list(INTERCEPTED_LAUNCH_APIS),
        "all_imported_launch_apis_intercepted": True,
    }


def _resolve_executable(command, context):
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(f"required {context} is unavailable: {command}")
    path = Path(resolved).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"resolved {context} is not a file: {path}")
    return path


def _run_checked(command, output_path, *, environment=None, cwd=None,
                 timeout=300):
    with output_path.open("wb") as stream:
        subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT,
            env=environment, cwd=cwd, timeout=timeout, check=True)


def _validate_npu_smi_idle(path, device_id, label):
    text = path.read_text(encoding="utf-8")
    if "npu-smi" not in text or "AICore(%)" not in text:
        raise ValueError(f"{label} lacks npu-smi AICore evidence")
    current_device = None
    target_rows = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        columns = [column.strip() for column in line.strip("|").split("|")]
        if len(columns) < 3:
            continue
        identity = columns[0].split()
        if (len(identity) >= 2 and identity[0].isdigit()
                and identity[1].startswith("910")):
            current_device = int(identity[0])
            if current_device == device_id and columns[1] != "OK":
                raise ValueError(f"{label} reports unhealthy target device")
            continue
        if (current_device == device_id
                and re.fullmatch(
                    r"[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:"
                    r"[0-9A-Fa-f]{2}\.[0-9A-Fa-f]", columns[1]) is not None):
            values = columns[2].split()
            if not values or re.fullmatch(r"[0-9]+", values[0]) is None:
                raise ValueError(f"{label} has invalid AICore utilization")
            target_rows.append(int(values[0]))
    if target_rows != [0]:
        raise ValueError(
            f"{label} does not prove device {device_id} is AICore-idle")


def _capture_npu_smi(executable, output_path, environment, device_id, label):
    _run_checked(
        [str(executable), "info"], output_path, environment=environment,
        timeout=60)
    _validate_npu_smi_idle(output_path, device_id, label)
    return {"path": str(output_path.resolve()), "sha256": _sha256(output_path)}


def _runtime_environment(cann_root):
    environment = os.environ.copy()
    library_paths = (
        cann_root / "aarch64-linux/lib64",
        cann_root / "aarch64-linux/lib64/plugin/opskernel",
        cann_root / "aarch64-linux/lib64/plugin/nnengine",
        cann_root / "opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64",
        cann_root / "tools/aml/lib64",
        cann_root / "tools/aml/lib64/plugin",
        Path("/usr/local/Ascend/driver/lib64"),
        Path("/usr/local/Ascend/driver/lib64/common"),
        Path("/usr/local/Ascend/driver/lib64/driver"),
    )
    missing = [str(path) for path in library_paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"required runtime library paths missing: {missing}")
    if "LD_LIBRARY_PATH" in environment:
        inherited = environment["LD_LIBRARY_PATH"]
    else:
        inherited = ""
    components = [str(path.resolve()) for path in library_paths]
    if inherited:
        components.append(inherited)
    environment["LD_LIBRARY_PATH"] = ":".join(components)
    return environment


def _require_output_scope(output_dir, repository_root):
    owner_root = repository_root.joinpath(*OUTPUT_OWNER_PARTS).resolve()
    if output_dir.resolve().parent != owner_root:
        raise ValueError(f"output must be a direct child of {owner_root}")
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    owner_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()


def _write_artifact_manifest(output_dir):
    manifest_path = output_dir / "artifact.sha256"
    lines = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path == manifest_path:
            continue
        lines.append(f"{_sha256(path)}  ./{path.relative_to(output_dir)}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": str(manifest_path.resolve()), "sha256": _sha256(manifest_path)}


def _identity_digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _verify_postflight_identities(
        args, repository_root, source_provenance, cann_provenance,
        workload_manifest, full_contracts):
    post_source = verify_source_archive(
        args.deployed_source_archive.resolve(), repository_root,
        args.baseline_commit)
    post_cann = collect_cann_provenance(args.cann_root.resolve())
    if source_provenance != post_source:
        raise ValueError("SwiGLU source identity changed during capture")
    if cann_provenance != post_cann:
        raise ValueError("SwiGLU CANN identity changed during capture")
    if workload_manifest is None:
        post_workload_manifest = None
        post_contracts = full_contracts
    else:
        post_contracts, post_workload_manifest = load_workload_manifest(
            args.workload_manifest.resolve(), args.workload_manifest_sha256,
            repository_root, args.models_root)
        if (post_workload_manifest != workload_manifest
                or post_contracts != full_contracts):
            raise ValueError("SwiGLU workload candidate changed during capture")
    return {
        "source_provenance_sha256": _identity_digest(post_source),
        "cann_provenance_sha256": _identity_digest(post_cann),
        "workload_candidate_sha256": (
            None if post_workload_manifest is None
            else _identity_digest(post_workload_manifest)),
    }


def _select_capture_contracts(args, repository_root):
    legacy_values = (
        args.canonical_dataset,
        args.canonical_dataset_sha256,
        args.case_key,
    )
    bulk_values = (
        args.workload_manifest,
        args.workload_manifest_sha256,
    )
    if all(value is not None for value in legacy_values):
        if any(value is not None for value in bulk_values):
            raise ValueError(
                "canonical case input and workload manifest are mutually exclusive")
        contract = load_case_contract(
            args.canonical_dataset.resolve(),
            args.canonical_dataset_sha256, args.case_key)
        return "canonical_case", [contract], None
    if any(value is not None for value in legacy_values):
        raise ValueError(
            "canonical case input requires dataset, dataset SHA256 and case key")
    if all(value is not None for value in bulk_values):
        if args.models_root is None:
            raise ValueError("workload manifest input requires --models-root")
        contracts, identity = load_workload_manifest(
            args.workload_manifest.resolve(), args.workload_manifest_sha256,
            repository_root, args.models_root)
        return "workload_manifest", contracts, identity
    if any(value is not None for value in bulk_values):
        raise ValueError(
            "workload manifest input requires path and SHA256")
    raise ValueError("either canonical case input or workload manifest is required")


def _select_workload_range(capture_mode, contracts, start_index, end_index):
    if capture_mode == "canonical_case":
        if start_index is not None or end_index is not None:
            raise ValueError(
                "canonical case input does not support workload shard indices")
        return contracts, 0, len(contracts), len(contracts)
    if capture_mode != "workload_manifest":
        raise AssertionError(f"unknown SwiGlu capture mode: {capture_mode!r}")
    full_workload_count = len(contracts)
    selected_start = 0 if start_index is None else start_index
    selected_end = full_workload_count if end_index is None else end_index
    if not 0 <= selected_start < selected_end <= full_workload_count:
        raise ValueError(
            "workload shard must satisfy "
            "0 <= start_index < end_index <= full_workload_count: "
            f"start_index={selected_start}, end_index={selected_end}, "
            f"full_workload_count={full_workload_count}")
    return (
        contracts[selected_start:selected_end], selected_start, selected_end,
        full_workload_count)


def _application_command(
        python, repository_root, workload_batch, device_id):
    return [
        str(python), str(repository_root / APPLICATION_SOURCE),
        "--workload-batch", str(workload_batch),
        "--device-id", str(device_id),
    ]


def _prepare_capture_contract(contract, case_dir):
    raw_dir = case_dir / "raw"
    raw_dir.mkdir()
    contract_path = case_dir / "case_contract.json"
    _write_json(contract_path, contract)
    return {
        "capture_directory": str(raw_dir.resolve()),
        "case_key": contract["case_key"],
        "dtype": contract["dtype"],
        "intermediate": contract["intermediate"],
        "tok": contract["tok"],
    }


def _finalize_capture_contract(
        contract, case_dir, reflection, application_command,
        application_log, cann_root):
    raw_dir = case_dir / "raw"
    contract_path = case_dir / "case_contract.json"
    records = tiling_capture_report.parse_capture_records(raw_dir)
    report = tiling_capture_report.build_report(
        records, reflection, contract, cann_root)
    records_path = case_dir / "records.json"
    report_path = case_dir / "report.json"
    _write_json(records_path, records)
    _write_json(report_path, report)
    return {
        "workload_key": contract["case_key"],
        "semantic_key": {
            "tok": contract["tok"],
            "intermediate": contract["intermediate"],
            "dtype": contract["dtype"],
            "layout": SUPPORTED_MANIFEST_LAYOUT,
            "dim": SUPPORTED_MANIFEST_DIM,
        },
        "contract": {
            "path": str(contract_path.resolve()),
            "sha256": _sha256(contract_path),
        },
        "records": {
            "path": str(records_path.resolve()),
            "sha256": _sha256(records_path),
        },
        "report": {
            "path": str(report_path.resolve()),
            "sha256": _sha256(report_path),
        },
        "application_log": {
            "path": str(application_log.resolve()),
            "sha256": _sha256(application_log),
        },
        "application_command": application_command,
        "launch_result": {
            "target_kernel_submitted": report["target_kernel_submitted"],
            "kernel_identity": report["kernel_identity"],
            "kernel_symbol": report["kernel_symbol"],
            "block_dim": report["block_dim"],
            "serialized_tiling_sha256": (
                report["tiling_payload_selection"]["payload_sha256"]),
            "decoded_tiling": report["decoded_tiling"],
            "source_derived_tile_schedule": (
                report["source_derived_tile_schedule"]),
        },
        "limitations": report["limitations"],
    }, report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--deployed-source-archive", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--canonical-dataset", type=Path)
    parser.add_argument("--canonical-dataset-sha256")
    parser.add_argument("--case-key")
    parser.add_argument("--workload-manifest", type=Path)
    parser.add_argument("--workload-manifest-sha256")
    parser.add_argument("--models-root", type=Path)
    parser.add_argument(
        "--start-index", type=int,
        help="inclusive index in numeric (tok, intermediate) workload order")
    parser.add_argument(
        "--end-index", type=int,
        help="exclusive index in numeric (tok, intermediate) workload order")
    parser.add_argument("--reflection-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument(
        "--cann-root", type=Path,
        default=Path("/usr/local/Ascend/cann-8.5.1"))
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--readelf", default="readelf")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--npu-smi", default="npu-smi")
    parser.add_argument("--application-batch-size", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args(argv)
    if args.device_id < 0:
        raise ValueError("device id must be non-negative")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    if (args.application_batch_size is not None
            and args.application_batch_size <= 0):
        raise ValueError("application batch size must be positive")
    repository_root = args.repository_root.resolve()
    if not repository_root.is_dir():
        raise FileNotFoundError(f"repository root is missing: {repository_root}")
    capture_mode, full_contracts, workload_manifest = (
        _select_capture_contracts(args, repository_root))
    if args.application_batch_size is None:
        if capture_mode == "canonical_case":
            application_batch_size = 1
        elif capture_mode == "workload_manifest":
            raise ValueError(
                "workload manifest capture requires an application batch size "
                "chosen from the hardware pilot")
        else:
            raise AssertionError(
                f"unknown SwiGlu capture mode: {capture_mode!r}")
    else:
        application_batch_size = args.application_batch_size
    contracts, start_index, end_index, full_workload_count = (
        _select_workload_range(
            capture_mode, full_contracts, args.start_index, args.end_index))
    _require_output_scope(args.output_dir, repository_root)
    output_dir = args.output_dir.resolve()
    logs_dir = output_dir / "logs"
    build_dir = output_dir / "build"
    logs_dir.mkdir()
    build_dir.mkdir()

    source_provenance = verify_source_archive(
        args.deployed_source_archive.resolve(), repository_root,
        args.baseline_commit)
    cann_root = args.cann_root.resolve()
    cann_provenance = collect_cann_provenance(cann_root)
    reflection = tiling_capture_report.parse_reflection(
        args.reflection_tsv.resolve())

    compiler = _resolve_executable(args.compiler, "C++ compiler")
    readelf = _resolve_executable(args.readelf, "ELF symbol inspector")
    python = _resolve_executable(args.python, "Python interpreter")
    npu_smi = _resolve_executable(args.npu_smi, "npu-smi")
    launch_import_audit = audit_launch_imports(readelf, cann_root)
    interposer = build_dir / "libmidlayer_swiglu_tiling_capture.so"
    compile_command = [
        str(compiler), "-std=c++17", "-O2", "-fPIC", "-shared",
        "-Wall", "-Wextra", "-Werror", "-o", str(interposer),
        str(repository_root / INTERPOSER_SOURCE),
        "-I", str(cann_root / "aarch64-linux/include"),
        "-I", str(cann_root / "aarch64-linux/pkg_inc"),
        "-I", str(cann_root / "aarch64-linux/pkg_inc/profiling"),
        "-I", str(cann_root / "aarch64-linux/pkg_inc/toolchain"),
        "-I", str(cann_root / "aarch64-linux/pkg_inc/runtime"),
        "-ldl",
    ]
    _run_checked(
        compile_command, logs_dir / "interposer_compile.log", timeout=300)
    if not interposer.is_file():
        raise RuntimeError("compiler did not produce the capture interposer")

    environment = _runtime_environment(cann_root)
    pre_telemetry = _capture_npu_smi(
        npu_smi, logs_dir / "npu_smi_before.txt", environment,
        args.device_id, "pre-capture telemetry")
    if "LD_PRELOAD" in environment and environment["LD_PRELOAD"]:
        raise ValueError("SwiGlu capture requires an empty LD_PRELOAD")
    environment["LD_PRELOAD"] = str(interposer)
    if capture_mode == "canonical_case":
        case_directories = [output_dir]
    elif capture_mode == "workload_manifest":
        cases_dir = output_dir / "cases"
        cases_dir.mkdir()
        case_directories = [
            cases_dir / contract["case_key"] for contract in contracts]
        for case_dir in case_directories:
            case_dir.mkdir()
    else:
        raise AssertionError(f"unknown SwiGlu capture mode: {capture_mode!r}")
    application_workloads = [
        _prepare_capture_contract(contract, case_dir)
        for contract, case_dir in zip(contracts, case_directories)
    ]
    application_context = {}
    application_process_count = 0
    for batch_start in range(
            0, len(application_workloads), application_batch_size):
        batch_end = min(
            batch_start + application_batch_size,
            len(application_workloads))
        batch_index = application_process_count
        batch_path = build_dir / f"application_batch_{batch_index:06d}.json"
        _write_json(batch_path, {
            "schema_version": 1,
            "workloads": application_workloads[batch_start:batch_end],
        })
        application_log = logs_dir / f"application_batch_{batch_index:06d}.log"
        command = _application_command(
            python, repository_root, batch_path, args.device_id)
        sentinel = build_dir / f"preload_sentinel_{batch_index:06d}"
        sentinel.mkdir()
        batch_environment = environment.copy()
        batch_environment["MIDLAYER_ASCEND_SWIGLU_CAPTURE_DIR"] = str(sentinel)
        _run_checked(
            command, application_log, environment=batch_environment,
            cwd=repository_root, timeout=args.timeout_seconds)
        if any(sentinel.iterdir()):
            raise RuntimeError(
                "SwiGLu application emitted capture data before selecting a "
                f"workload directory: {sentinel}")
        for workload in application_workloads[batch_start:batch_end]:
            workload_key = workload["case_key"]
            application_context[workload_key] = (command, application_log)
        application_process_count += 1
    results = {}
    reports = {}
    for contract, case_dir in zip(contracts, case_directories):
        workload_key = contract["case_key"]
        command, application_log = application_context[workload_key]
        result, report = _finalize_capture_contract(
            contract, case_dir, reflection, command, application_log,
            cann_root)
        if workload_key in results:
            raise ValueError(f"duplicate capture result key: {workload_key}")
        results[workload_key] = result
        reports[workload_key] = report
    post_telemetry = _capture_npu_smi(
        npu_smi, logs_dir / "npu_smi_after.txt", _runtime_environment(cann_root),
        args.device_id, "post-capture telemetry")
    postflight_identity = _verify_postflight_identities(
        args, repository_root, source_provenance, cann_provenance,
        workload_manifest, full_contracts)

    if capture_mode == "canonical_case":
        legacy_key = contracts[0]["case_key"]
        legacy_result = results[legacy_key]
        legacy_report = reports[legacy_key]
        manifest = {
            "schema_version": 2,
            "capture_state": "complete_non_duration_observation",
            "duration_consumed": False,
            "target_kernel_submitted": False,
            "baseline_commit": args.baseline_commit,
            "case_contract": contracts[0],
            "source_provenance": source_provenance,
            "cann_provenance": cann_provenance,
            "postflight_identity": postflight_identity,
            "launch_import_audit": launch_import_audit,
            "reflection": reflection,
            "compiler": {"path": str(compiler), "sha256": _sha256(compiler)},
            "python": {"path": str(python), "sha256": _sha256(python)},
            "compile_command": compile_command,
            "application_batch_size": application_batch_size,
            "application_process_count": application_process_count,
            "application_command": legacy_result["application_command"],
            "environment": {
                "ASCEND_DEVICE_ID": str(args.device_id),
                "ASCEND_RT_VISIBLE_DEVICES": str(args.device_id),
                "MIDLAYER_ASCEND_SWIGLU_CAPTURE_DIR": str(
                    (output_dir / "raw").resolve()),
            },
            "interposer": {
                "path": str(interposer), "sha256": _sha256(interposer)},
            "pre_telemetry": pre_telemetry,
            "post_telemetry": post_telemetry,
            "records": legacy_result["records"],
            "report": legacy_result["report"],
            "limitations": legacy_report["limitations"],
        }
        printed_result = {"report": legacy_report}
    elif capture_mode == "workload_manifest":
        manifest = {
            "schema_version": 2,
            "artifact_type": "ascend_910b_swiglu_bulk_tiling_observation",
            "capture_state": "complete_non_duration_observation",
            "duration_consumed": False,
            "target_kernel_submitted": False,
            "baseline_commit": args.baseline_commit,
            "workload_frontier": workload_manifest,
            "source_provenance": source_provenance,
            "cann_provenance": cann_provenance,
            "postflight_identity": postflight_identity,
            "launch_import_audit": launch_import_audit,
            "reflection": reflection,
            "compiler": {"path": str(compiler), "sha256": _sha256(compiler)},
            "python": {"path": str(python), "sha256": _sha256(python)},
            "compile_command": compile_command,
            "compile_count": 1,
            "application_batch_size": application_batch_size,
            "application_process_count": application_process_count,
            "full_workload_count": full_workload_count,
            "selected_range": {
                "start_index": start_index,
                "end_index": end_index,
            },
            "selected_workload_count": len(contracts),
            "environment": {
                "ASCEND_DEVICE_ID": str(args.device_id),
                "ASCEND_RT_VISIBLE_DEVICES": str(args.device_id),
            },
            "interposer": {
                "path": str(interposer), "sha256": _sha256(interposer)},
            "pre_telemetry": pre_telemetry,
            "post_telemetry": post_telemetry,
            "result_count": len(results),
            "results": [results[contract["case_key"]] for contract in contracts],
        }
        printed_result = {"results": manifest["results"]}
    else:
        raise AssertionError(f"unknown SwiGlu capture mode: {capture_mode!r}")
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    artifact = _write_artifact_manifest(output_dir)
    printed_result["artifact_manifest"] = artifact
    print(json.dumps(printed_result, sort_keys=True))


if __name__ == "__main__":
    main()
