#!/usr/bin/env python3
"""Query serialized aclnnMatmul tiling without executing the matmul."""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time

import zstandard


BASELINE_PATTERN = re.compile(r"[0-9a-f]{40}")
OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", "ascend_910b")
HELPER_SOURCE = (
    "microbench/ascend_910b/operators/gemm/tiling_query_helper.cpp")
GUARD_SOURCE = (
    "microbench/ascend_910b/operators/gemm/launch_guard_interposer.cpp")
RUNNER_SOURCE = (
    "microbench/ascend_910b/operators/gemm/tiling_query_runner.py")
BUILDER_SOURCE = (
    "dataset_pipeline/launch_data/gemm_workload_candidates.py")
WORKLOAD_KEY_SOURCE = "src/kernel/launch_config/gemm_workload.py"
ASCEND_GEMM_LAUNCH_SELECTION = (
    "launch_data/ascend_910b/gemm_launch_selection.csv.zst")
SOURCE_FILES = (
    HELPER_SOURCE, GUARD_SOURCE, RUNNER_SOURCE, BUILDER_SOURCE,
    WORKLOAD_KEY_SOURCE)
RUNNER_VERSION = 7
EXPECTED_LAUNCH_API = "aclrtLaunchKernelWithHostArgs"
INTERPOSED_LAUNCH_APIS = frozenset({
    "aclrtLaunchKernel",
    "aclrtLaunchKernelV2",
    "aclrtLaunchKernelWithConfig",
    EXPECTED_LAUNCH_API,
})
GEMM_WORKLOAD_KEY_FIELDS = (
    "dtype", "variant", "b", "m", "n", "k", "a_layout", "b_layout",
    "c_layout", "trans_a", "trans_b", "epilogue", "has_bias",
    "position", "tensor_parallel_size", "expert_parallel_size",
)
HARDWARE_IDS = ("nvidia_a800", "metax_c500", "ascend_910b")
PHYSICAL_OPERATION_FIELDS = ("m", "n", "k", "dtype", "has_bias")
SINGLE_SHAPE_DEFAULTS = {"b": 1, "m": 729, "n": 6144, "k": 2560}
EXACT_DOMAIN_STATUS = "static_complete_legal_launch_selector_domain"
EXACT_SEMANTIC_KEY_COUNT = 140544
EXACT_PHYSICAL_OPERATION_COUNT = 140544
EXACT_DOMAIN_SHA256 = (
    "8a935c5226fc5b719fac1ab0e8dd78df130b9deeaa0725c9c936e5eeb9d14bee")
CANN_ROOT = Path("/usr/local/Ascend/cann-8.5.1")
CANN_ABI_ARTIFACTS = (
    ("opp/version.info",
     "49437edb9a235b28a87d07d2a2ad189022b6d246dc7c4cbcd013d2b7ecaf838b"),
    ("python/site-packages/tbe/common/utils/op_tiling.py",
     "60ce20a9d26ecb7a7bb761362d7670466a748376099ddda0d5790e0d49956d31"),
    ("python/site-packages/tbe/dsl/unify_schedule/gemm_tiling_util.py",
     "05e033838e714b2648f75f09ae193b73206c825712a55fa5d70c9b7ff1029b39"),
    ("opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/ops_legacy/"
     "mat_mul/MatMulV2_ND_ND_FP16_FP16_false_true_all.json",
     "95885de8b6e57ae6b828581f001dc90fa84b9a1341d1cd4a9fdccf81d92197f0"),
    ("opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/ops_legacy/"
     "mat_mul/MatMulV2_ND_ND_FP16_FP16_false_true_all.o",
     "87bed199882fee1f7b890bce46fd78059c0d465344757e1227bc2c25071b5d0b"),
    ("opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/ops_nn/"
     "mat_mul_v3/MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_"
     "high_performance.json",
     "0d252989ae68724145b139d747dc49d9416d8a929cf422d0790531a7904008a7"),
    ("opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/ops_nn/"
     "mat_mul_v3/MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_"
     "high_performance.o",
     "53f6b25ef79582e2e49f80ec9d68df1ef58f05856d4b9ba04d0fd2ce39d883b9"),
    ("opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/mat_mul_v3/"
     "arch35/mat_mul_tiling_data.h",
     "f4ea1cc98a05dc5f4b313b3beb41c131fa440a314affd7fa713aa9f2ae88780a"),
    ("aarch64-linux/asc/include/adv_api/kernel_tiling.h",
     "0f50fe7d81d4e01f80b62c19d816ad28151ac1502f708b70454efc8599ddaf44"),
    ("opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/mat_mul_v3/"
     "arch35/mat_mul_tiling_key.h",
     "9ccbd8a9548310b5b4551e25128f0641b531c38ec3292f81dea4329454b0cf46"),
    ("opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/dynamic/mat_mul_v3.py",
     "89a51f41fbe3d02720172e0d5a2225b21f6409f36fd80001591244dd7040d32c"),
    ("opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/3rd/"
     "mat_mul_v3/op_host/op_tiling/arch35/"
     "matmul_v3_asw_full_load_tiling.cpp",
     "15a2e1ebf77c79e730dc2c4ef18bb49ed1681eca3b62d987b0dc413646bb88ad"),
    ("opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/3rd/"
     "mat_mul_v3/op_host/op_tiling/arch35/"
     "matmul_v3_base_tiling_advanced.h",
     "74b677d7476fbfe499bd981a1e5343363af2653b6b56d438680165e8cced5eba"),
)
MATMUL_V2_KERNEL_NAME = "MatMulV2_ND_ND_FP16_FP16_false_true_all"
MATMUL_V2_METADATA_RELATIVE_PATH = CANN_ABI_ARTIFACTS[3][0]
MATMUL_V2_KERNEL_TILING_KEY_COUNT = 441
MATMUL_V2_KERNEL_TILING_KEYS_SHA256 = (
    "85a1cd131480205c4cbb489800047b385269a9b80f5ba5b558072eee9503b2d7")
_MATMUL_V2_BASE_TILING_KEYS = (
    98304, 98305, 98312, 98313, 98324, 98325, 98336, 98337, 98344,
    98345, 98356, 98357, 98370, 98371, 98378, 98379, 98384, 98385,
    98386, 98387, 98388, 98389, 98432, 98433, 98440, 98441, 98452,
    98453, 98464, 98465, 98472, 98473, 98484, 98485, 98498, 98499,
    98506, 98507, 98512, 98513, 98514, 98515, 98516, 98517, 98816,
    98817, 98824, 98825, 98836, 98837, 98848, 98849, 98856, 98857,
    98868, 98869, 98882, 98883, 98890, 98891, 98896, 98897, 98898,
    98899, 98900, 98901, 98944, 98945, 98952, 98953, 98964, 98965,
    98976, 98977, 98984, 98985, 98996, 98997, 99010, 99011, 99018,
    99019, 99024, 99025, 99026, 99027, 99028, 99029,
)
_MATMUL_V2_TILING_KEY_OFFSETS = (0, 1048576, 2097152, 3145728, 16777216)
MATMUL_V2_KERNEL_TILING_KEYS = frozenset(
    {base + offset
     for base in _MATMUL_V2_BASE_TILING_KEYS
     for offset in _MATMUL_V2_TILING_KEY_OFFSETS}
    | {360489})
MATMUL_V3_KERNEL_NAME = (
    "MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance")
MATMUL_V3_METADATA_RELATIVE_PATH = CANN_ABI_ARTIFACTS[5][0]
MATMUL_V3_KERNEL_TILING_KEYS = frozenset({
    0, 65536, 1114112, 65537, 2, 65538, 131074, 4098, 69634, 73730,
    32, 65568, 96, 65632, 65616, 65600, 48, 65584, 8240, 73776,
})
MATMUL_V3_KERNEL_TILING_KEY_COUNT = 20
MATMUL_V3_KERNEL_TILING_KEYS_SHA256 = (
    "1fc2464b7a7493a6fff4eb865c8e837f6b7ef4a2676be6b0c6ed4ba285b3ae35")
MATMUL_KERNEL_FAMILY_CONTRACTS = {
    "matmul_v2": {
        "kernel_name": MATMUL_V2_KERNEL_NAME,
        "symbol_prefix": MATMUL_V2_KERNEL_NAME + "_",
        "host_args_size_bytes": 200,
        "placeholder": {"address_offset": 24, "data_offset": 40},
        "tiling_payload_size_bytes": 160,
        "schem_mode": 0,
    },
    "matmul_v3": {
        "kernel_name": MATMUL_V3_KERNEL_NAME,
        "symbol_prefix": MATMUL_V3_KERNEL_NAME + "_",
        "host_args_size_bytes": 360,
        "placeholder": {"address_offset": 56, "data_offset": 72},
        "tiling_payload_size_bytes": 288,
        "schem_mode": 1,
    },
}
CANDIDATE_FIELDS = {
    "artifact_type", "coverage_scope", "cross_hardware",
    "dataset_model_inventory", "duration_consumed", "hardware_domains",
    "limitations", "model_domains", "physical_operation_count",
    "physical_operation_fields", "scheduler_contract", "schema_version",
    "semantic_key_count", "source_config", "source_config_sha256",
    "workload_key_fields", "workload_keys",
}
CAPTURE_LIMITATIONS = [
    "the complete host-argument, placeholder, and tiling payload buffers are "
    "retained; the capture runner identifies the pinned CANN 8.5.1 MatMulV2 "
    "or MatMulV3 family but does not semantically decode MatMulV3 tiling",
    "the result is specific to CANN 8.5.1, whose family metadata, source "
    "artifacts, and libnnopbase import table are audited before dry launches",
    "the MatMulV2 final eight payload bytes and the complete MatMulV3 payload "
    "are retained verbatim without unsupported semantic interpretation",
    "the helper does not call aclSetAclOpExecutorRepeatable; under CANN 8.5.1 "
    "acl_meta.h and opdev/op_executor.h semantics, the successful "
    "non-repeatable aclnnMatmul CommonOpExecutorRun path consumes the executor, "
    "so the helper clears its borrowed pointer and does not destroy it again",
    "the requested input rank is recorded explicitly: rank-two capture creates "
    "LHS [M,K] and output [M,N], while rank-three capture creates [B,M,K] "
    "and [B,M,N]; the [K,N] RHS is a transposed view of [N,K] storage",
    "mixBlockDim is not an argument of aclrtLaunchKernelWithHostArgs; it is "
    "therefore retained as opaque rather than inferred from blockDim",
]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(payload):
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
        "utf-8")


def _output_path(path, output_dir):
    resolved = path.resolve()
    output_root = output_dir.resolve()
    if resolved.is_relative_to(output_root):
        return str(resolved.relative_to(output_root))
    return str(resolved)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def _resolve_executable(command, context):
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(f"required {context} is unavailable: {command}")
    path = Path(resolved).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"resolved {context} is not a file: {path}")
    return path


def _require_positive(value, context):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if value <= 0:
        raise ValueError(f"{context} must be positive")
    return value


def _require_output_scope(output_dir, repository_root):
    owner_root = repository_root.joinpath(*OUTPUT_OWNER_PARTS).resolve()
    if output_dir.resolve().parent != owner_root:
        raise ValueError(f"output must be a direct child of {owner_root}")
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    owner_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()


def _run_checked(command, output_path, *, environment=None, timeout=300):
    with output_path.open("wb") as stream:
        subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT,
            env=environment, timeout=timeout, check=True)


def _source_file_identities(repository_root):
    identities = []
    for relative_path in SOURCE_FILES:
        path = repository_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"required query source is missing: {path}")
        identities.append({
            "relative_path": relative_path,
            "sha256": _sha256(path),
        })
    return identities


def collect_source_identity(repository_root, baseline_commit):
    if BASELINE_PATTERN.fullmatch(baseline_commit) is None:
        raise ValueError("baseline commit must be 40 lowercase hex")
    source_files = _source_file_identities(repository_root)
    git_marker = repository_root / ".git"
    snapshot_marker = repository_root / ".source_commit"
    if not git_marker.exists():
        if not snapshot_marker.is_file():
            raise FileNotFoundError(
                "source root has neither .git nor .source_commit identity")
        marker_text = snapshot_marker.read_text(encoding="utf-8")
        if marker_text not in (baseline_commit, baseline_commit + "\n"):
            raise ValueError(".source_commit differs from baseline commit")
        return {
            "mode": "audited_snapshot",
            "baseline_commit": baseline_commit,
            "marker_path": str(snapshot_marker.resolve()),
            "marker_sha256": _sha256(snapshot_marker),
            "source_files": source_files,
        }

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root,
        check=True, capture_output=True, text=True)
    commit = completed.stdout.strip()
    if BASELINE_PATTERN.fullmatch(commit) is None:
        raise ValueError(f"repository HEAD is not a commit id: {commit!r}")
    if commit != baseline_commit:
        raise ValueError("repository HEAD differs from baseline commit")
    for source_path in SOURCE_FILES:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", source_path],
            cwd=repository_root, check=True, capture_output=True, text=True)
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=repository_root, check=True, capture_output=True, text=True)
    if status.stdout:
        raise ValueError("Git source checkout is not clean")
    return {
        "mode": "git_checkout",
        "baseline_commit": commit,
        "marker_path": None,
        "marker_sha256": None,
        "source_files": source_files,
    }


def _cann_version(cann_root):
    version_path = cann_root / "opp/version.info"
    if not version_path.is_file():
        raise FileNotFoundError(f"CANN version file is missing: {version_path}")
    lines = version_path.read_text(encoding="utf-8").splitlines()
    matches = [line.removeprefix("Version=") for line in lines
               if line.startswith("Version=")]
    if len(matches) != 1 or not matches[0]:
        raise ValueError("CANN version.info must contain one Version entry")
    if matches[0] != "8.5.1":
        raise ValueError("GEMM tiling query requires exact CANN Version=8.5.1")
    return matches[0]


def _collect_kernel_family_metadata(
        metadata_path, family, expected_keys, expected_keys_sha256):
    contract = MATMUL_KERNEL_FAMILY_CONTRACTS[family]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (not isinstance(metadata, dict)
            or metadata["kernelName"] != contract["kernel_name"]
            or metadata["opParaSize"]
            != contract["tiling_payload_size_bytes"]):
        raise ValueError(f"{family} kernel metadata identity changed")
    kernel_list = metadata["kernelList"]
    if not isinstance(kernel_list, list):
        raise TypeError(f"{family} kernelList must be an array")
    prefix = contract["symbol_prefix"]
    kernel_symbols = set()
    tiling_keys = []
    for index, value in enumerate(kernel_list):
        if not isinstance(value, dict) or "kernelName" not in value:
            raise ValueError(f"{family} kernelList[{index}] schema changed")
        symbol = value["kernelName"]
        if (not isinstance(symbol, str) or not symbol.startswith(prefix)
                or re.fullmatch(r"[0-9]+", symbol.removeprefix(prefix)) is None):
            raise ValueError(f"{family} kernelList[{index}] name changed")
        tiling_key = int(symbol.removeprefix(prefix))
        if family == "matmul_v2":
            if ("deterministic" not in value
                    or value["deterministic"] != "ignore"):
                raise ValueError(
                    f"{family} kernelList[{index}] schema changed")
        elif family == "matmul_v3":
            if (set(value) != {
                    "crossCoreSync", "kernelName", "kernelType",
                    "taskRation", "tilingKey"}
                    or value["tilingKey"] != tiling_key
                    or value["kernelType"] != "MIX_AIC"
                    or value["crossCoreSync"] not in (0, 1)
                    or value["taskRation"] not in ("1:0", "1:2")):
                raise ValueError(
                    f"{family} kernelList[{index}] schema changed")
        else:
            raise AssertionError(f"unhandled MatMul family {family!r}")
        if symbol in kernel_symbols:
            raise ValueError(f"{family} kernelList repeats a kernel symbol")
        kernel_symbols.add(symbol)
        tiling_keys.append(tiling_key)
    tiling_keys.sort()
    tiling_key_payload = json.dumps(
        tiling_keys, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    if (len(tiling_keys) != len(expected_keys)
            or frozenset(tiling_keys) != expected_keys
            or hashlib.sha256(tiling_key_payload).hexdigest()
            != expected_keys_sha256):
        raise ValueError(f"{family} kernel tiling-key set changed")
    return ({
        "kernel_name": contract["kernel_name"],
        "kernel_symbol_prefix": prefix,
        "tiling_payload_size_bytes": contract["tiling_payload_size_bytes"],
        "kernel_tiling_key_count": len(tiling_keys),
        "kernel_tiling_keys_sha256": expected_keys_sha256,
        "kernel_tiling_keys_canonicalization": (
            "numeric_sort_compact_json_no_trailing_newline"),
    }, frozenset(kernel_symbols))


def collect_cann_abi_identity(cann_root):
    """Hash and validate the exact CANN sources used by GEMM capture."""

    cann_root = cann_root.resolve()
    if cann_root != CANN_ROOT:
        raise ValueError(f"GEMM tiling query requires CANN root {CANN_ROOT}")
    version = _cann_version(cann_root)
    artifacts = []
    paths = {}
    artifact_identities = {}
    for relative_path, expected_sha256 in CANN_ABI_ARTIFACTS:
        path = cann_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"required CANN ABI artifact is missing: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"CANN ABI artifact SHA256 changed: {path}")
        identity = {"path": str(path), "sha256": actual_sha256}
        artifacts.append(identity)
        paths[relative_path] = path
        artifact_identities[relative_path] = identity
    matmul_v2, matmul_v2_symbols = _collect_kernel_family_metadata(
        paths[MATMUL_V2_METADATA_RELATIVE_PATH], "matmul_v2",
        MATMUL_V2_KERNEL_TILING_KEYS,
        MATMUL_V2_KERNEL_TILING_KEYS_SHA256)
    matmul_v2.update({
        "decoded_prefix": {
            "encoding": "little_endian_signed_int32",
            "field_count": 38,
            "size_bytes": 152,
        },
        "opaque_tail_size_bytes": 8,
        "source_artifacts": [
            artifact_identities[relative_path]
            for relative_path, _ in CANN_ABI_ARTIFACTS[:5]
        ],
    })
    matmul_v3, matmul_v3_symbols = _collect_kernel_family_metadata(
        paths[MATMUL_V3_METADATA_RELATIVE_PATH], "matmul_v3",
        MATMUL_V3_KERNEL_TILING_KEYS,
        MATMUL_V3_KERNEL_TILING_KEYS_SHA256)
    matmul_v3.update({
        "decoded_prefix": {
            "encoding": "little_endian_signed_int32",
            "size_bytes": 268,
            "tcube_tiling_size_bytes": 200,
            "tcube_projected_field_count": 15,
            "legacy_extension_offset_bytes": 200,
            "legacy_extension_field_count": 17,
        },
        "opaque_tail_size_bytes": 20,
        "source_artifacts": [artifact_identities[CANN_ABI_ARTIFACTS[0][0]]] + [
            artifact_identities[relative_path]
            for relative_path, _ in CANN_ABI_ARTIFACTS[5:]
        ],
    })
    return ({
        "abi": "cann_8.5.1_matmul_tiling_families",
        "cann_version": version,
        "kernel_families": {
            "matmul_v2": matmul_v2,
            "matmul_v3": matmul_v3,
        },
        "source_artifacts": artifacts,
    }, matmul_v2_symbols | matmul_v3_symbols)


def _runtime_environment(cann_root):
    environment = os.environ.copy()
    library_paths = (
        cann_root / "aarch64-linux/lib64",
        Path("/usr/local/Ascend/driver/lib64"),
        Path("/usr/local/Ascend/driver/lib64/common"),
        Path("/usr/local/Ascend/driver/lib64/driver"),
    )
    missing = [str(path) for path in library_paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"required runtime library paths are missing: {missing}")
    components = [str(path.resolve()) for path in library_paths]
    if "LD_LIBRARY_PATH" in environment and environment["LD_LIBRARY_PATH"]:
        components.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(components)
    return environment


def validate_npu_smi_idle(path, device_id):
    text = path.read_text(encoding="utf-8")
    if "npu-smi" not in text or "AICore(%)" not in text:
        raise ValueError("npu-smi output lacks AICore evidence")
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
                raise ValueError("npu-smi reports an unhealthy target device")
            continue
        if (current_device == device_id
                and re.fullmatch(
                    r"[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:"
                    r"[0-9A-Fa-f]{2}\.[0-9A-Fa-f]", columns[1]) is not None):
            values = columns[2].split()
            if not values or re.fullmatch(r"[0-9]+", values[0]) is None:
                raise ValueError("npu-smi has invalid AICore utilization")
            target_rows.append(int(values[0]))
    if target_rows != [0]:
        raise ValueError(
            f"npu-smi does not prove device {device_id} is AICore-idle")


def parse_helper_metadata(path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.count("=") != 1:
            raise ValueError(f"invalid helper metadata line: {line!r}")
        key, value = line.split("=", 1)
        if not key or key in result:
            raise ValueError(f"duplicate or empty helper key: {key!r}")
        result[key] = value
    expected = {
        "schema_version", "input_rank", "batch_size", "workspace_size_bytes", "execute_api_called",
        "device_kernel_submitted", "duration_consumed",
        "executor_lifecycle", "process_peak_rss_kib",
    }
    if set(result) != expected:
        raise ValueError(f"helper metadata keys changed: {sorted(result)}")
    if result["schema_version"] != "6":
        raise ValueError("unsupported helper metadata schema")
    input_rank = int(result["input_rank"])
    if input_rank not in (2, 3):
        raise ValueError("helper reported an unsupported input rank")
    if result["execute_api_called"] != "true":
        raise ValueError("helper did not call the MatMul execute API")
    if (result["device_kernel_submitted"] != "false"
            or result["duration_consumed"] != "false"):
        raise ValueError("dry-launch helper reported an unsafe state")
    if result["executor_lifecycle"] != "consumed_by_aclnnMatmul":
        raise ValueError("helper executor lifecycle contract changed")
    workspace_size = int(result["workspace_size_bytes"])
    if workspace_size < 0:
        raise ValueError("helper reported an invalid workspace size")
    batch_size = int(result["batch_size"])
    if batch_size <= 0:
        raise ValueError("helper reported an invalid batch size")
    process_peak_rss_kib = int(result["process_peak_rss_kib"])
    if process_peak_rss_kib <= 0:
        raise ValueError("helper reported an invalid process peak RSS")
    return {
        "input_rank": input_rank,
        "batch_size": batch_size,
        "workspace_size_bytes": workspace_size,
        "execute_api_called": True,
        "device_kernel_submitted": False,
        "duration_consumed": False,
        "executor_lifecycle": result["executor_lifecycle"],
        "process_peak_rss_kib": process_peak_rss_kib,
    }


def load_executor_tiling(path, expected_size, getter_api, output_dir=None):
    if not path.is_file():
        raise FileNotFoundError(f"executor tiling payload is missing: {path}")
    payload = path.read_bytes()
    if len(payload) != expected_size:
        raise ValueError("executor tiling payload size differs from metadata")
    return {
        "available": True,
        "getter_api": getter_api,
        "serialized_hex": payload.hex(),
        "size_bytes": len(payload),
        "sha256": _sha256(path),
        "path": (
            str(path.resolve()) if output_dir is None
            else _output_path(path, output_dir)),
        "decoded_fields": None,
    }


def _kernel_family_from_symbol(symbol):
    if re.fullmatch(
            re.escape(MATMUL_KERNEL_FAMILY_CONTRACTS["matmul_v2"][
                "symbol_prefix"]) + r"[0-9]+", symbol) is not None:
        return "matmul_v2"
    if re.fullmatch(
            re.escape(MATMUL_KERNEL_FAMILY_CONTRACTS["matmul_v3"][
                "symbol_prefix"]) + r"[0-9]+", symbol) is not None:
        return "matmul_v3"
    raise ValueError(
        f"captured kernel stub is outside the pinned MatMul families: "
        f"{symbol!r}")


def extract_embedded_tiling(launch):
    placeholders = launch["placeholders"]["entries"]
    if len(placeholders) != 1:
        raise ValueError(
            "aclnnMatmul launch must expose exactly one embedded-data "
            "placeholder")
    placeholder = placeholders[0]
    host_args = bytes.fromhex(launch["host_args"]["serialized_hex"])
    offset = placeholder["data_offset"]
    if offset < placeholder["address_offset"] + 8 or offset >= len(host_args):
        raise ValueError("aclnnMatmul embedded tiling offset is invalid")
    payload = host_args[offset:]
    return {
        "available": True,
        "kernel_family": launch["kernel_family"],
        "source": "aclrtLaunchKernelWithHostArgs_placeholder_tail",
        "pointer_slot_offset_bytes": placeholder["address_offset"],
        "offset_bytes": offset,
        "size_bytes": len(payload),
        "serialized_hex": payload.hex(),
        "sha256": _sha256_bytes(payload),
        "payload_abi": {
            "status": "size_validated_only",
            "cann_version": "8.5.1",
            "kernel_family": launch["kernel_family"],
            "size_bytes": len(payload),
        },
        "decoded_fields": {
            "status": "opaque",
            "reason": (
                "no fixed CANN 8.5.1 source or ABI evidence has yet proved "
                "the semantic layout for this complete payload"),
        },
    }


def _parse_key_value_file(path):
    values = {}
    for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.count("=") != 1:
            raise ValueError(
                f"invalid metadata line at {path}:{line_number}")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise ValueError(f"duplicate or empty metadata key: {key!r}")
        values[key] = value
    return values


def _parse_nonnegative_decimal(value, context, *, positive=False):
    if re.fullmatch(r"[0-9]+", value) is None:
        raise ValueError(f"{context} is not an unsigned decimal integer")
    parsed = int(value)
    if positive and parsed == 0:
        raise ValueError(f"{context} must be positive")
    return parsed


def _capture_child(capture_dir, name, context):
    if not name or Path(name).name != name:
        raise ValueError(f"{context} must be a plain file name")
    path = capture_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"{context} is missing: {path}")
    return path


def parse_dry_launch_capture(capture_dir, output_dir=None):
    metadata_path = capture_dir / "launch.meta"
    if not metadata_path.is_file():
        raise FileNotFoundError("dry-launch interposer emitted no launch.meta")
    values = _parse_key_value_file(metadata_path)
    expected = {
        "schema_version", "record_type", "api", "execute_api",
        "kernel_stub", "block_dim", "stream", "configuration_present",
        "configuration_attr_count", "configuration_attr_entry_size",
        "configuration_attr_file", "host_args_size", "host_args_file",
        "placeholder_count", "placeholder_entry_size", "placeholder_file",
        "device_kernel_submitted", "duration_consumed",
    }
    if set(values) != expected:
        raise ValueError(f"dry-launch metadata keys changed: {sorted(values)}")
    if values["schema_version"] != "2" or values["record_type"] != "launch":
        raise ValueError("unsupported dry-launch capture schema")
    if (values["api"] != EXPECTED_LAUNCH_API
            or values["execute_api"] != "aclnnMatmul"):
        raise ValueError("capture is not the aclnnMatmul host-args launch")
    if not values["kernel_stub"]:
        raise ValueError("captured kernel stub is empty")
    kernel_family = _kernel_family_from_symbol(values["kernel_stub"])
    family_contract = MATMUL_KERNEL_FAMILY_CONTRACTS[kernel_family]
    if (values["device_kernel_submitted"] != "false"
            or values["duration_consumed"] != "false"):
        raise ValueError("capture claims device execution or duration use")
    block_dim = _parse_nonnegative_decimal(
        values["block_dim"], "block dim", positive=True)
    stream = _parse_nonnegative_decimal(values["stream"], "stream")
    host_args_size = _parse_nonnegative_decimal(
        values["host_args_size"], "host args size", positive=True)
    host_args_path = _capture_child(
        capture_dir, values["host_args_file"], "host args")
    host_args = host_args_path.read_bytes()
    if len(host_args) != host_args_size:
        raise ValueError("host args size differs from metadata")
    if host_args_size != family_contract["host_args_size_bytes"]:
        raise ValueError(
            f"{kernel_family} host args size changed: {host_args_size}")
    placeholder_count = _parse_nonnegative_decimal(
        values["placeholder_count"], "placeholder count")
    placeholder_entry_size = _parse_nonnegative_decimal(
        values["placeholder_entry_size"], "placeholder entry size",
        positive=True)
    if placeholder_entry_size != struct.calcsize("<II"):
        raise ValueError("unexpected aclrtPlaceHolderInfo size")
    placeholder_path = _capture_child(
        capture_dir, values["placeholder_file"], "placeholder map")
    placeholder_bytes = placeholder_path.read_bytes()
    if len(placeholder_bytes) != placeholder_count * placeholder_entry_size:
        raise ValueError("placeholder map size differs from metadata")
    placeholders = [
        {"address_offset": address_offset, "data_offset": data_offset}
        for address_offset, data_offset in struct.iter_unpack(
            "<II", placeholder_bytes)
    ]
    if placeholders != [family_contract["placeholder"]]:
        raise ValueError(f"{kernel_family} placeholder contract changed")
    if (host_args_size - placeholders[0]["data_offset"]
            != family_contract["tiling_payload_size_bytes"]):
        raise ValueError(f"{kernel_family} tiling payload size changed")
    for placeholder in placeholders:
        if placeholder["address_offset"] + 8 > host_args_size:
            raise ValueError("placeholder address slot exceeds host args")
        if placeholder["data_offset"] >= host_args_size:
            raise ValueError("placeholder data offset exceeds host args")
    configuration_present = values["configuration_present"]
    if configuration_present not in {"true", "false"}:
        raise ValueError("configuration_present is not boolean")
    configuration_attr_count = _parse_nonnegative_decimal(
        values["configuration_attr_count"], "configuration attr count")
    configuration_attr_entry_size = _parse_nonnegative_decimal(
        values["configuration_attr_entry_size"],
        "configuration attr entry size", positive=True)
    configuration_path = _capture_child(
        capture_dir, values["configuration_attr_file"],
        "launch configuration attributes")
    configuration_bytes = configuration_path.read_bytes()
    if len(configuration_bytes) != (
            configuration_attr_count * configuration_attr_entry_size):
        raise ValueError("launch configuration size differs from metadata")
    if (configuration_present != "true"
            or configuration_attr_count != 2
            or configuration_attr_entry_size != 20
            or len(configuration_bytes) != 40):
        raise ValueError(
            f"{kernel_family} launch configuration contract changed")
    decoded_attributes = []
    expected_attributes = ((1, "SCHEM_MODE"), (2, "LOCAL_MEMORY_SIZE"))
    expected_attribute_values = (family_contract["schem_mode"], 0)
    for index, (attribute_id, name) in enumerate(expected_attributes):
        entry = configuration_bytes[index * 20:(index + 1) * 20]
        captured_id, captured_value = struct.unpack_from("<II", entry)
        expected_value = expected_attribute_values[index]
        if (captured_id != attribute_id
                or captured_value != expected_value
                or entry[8:] != bytes(12)):
            raise ValueError(
                f"{kernel_family} launch configuration {name} changed")
        decoded_attributes.append({"name": name, "value": captured_value})
    def recorded_path(path):
        if output_dir is None:
            return str(path.resolve())
        return _output_path(path, output_dir)

    tiling_key_text = values["kernel_stub"].rsplit("_", 1)[-1]
    tiling_key = _parse_nonnegative_decimal(tiling_key_text, "tiling key")
    expected_tiling_keys = (
        MATMUL_V2_KERNEL_TILING_KEYS if kernel_family == "matmul_v2"
        else MATMUL_V3_KERNEL_TILING_KEYS)
    if tiling_key not in expected_tiling_keys:
        raise ValueError("captured tiling key is absent from pinned metadata")
    return {
        "api": values["api"],
        "execute_api": values["execute_api"],
        "kernel_stub": values["kernel_stub"],
        "kernel_family": kernel_family,
        "tiling_key": tiling_key,
        "block_dim": block_dim,
        "mix_block_dim": {
            "status": "opaque",
            "value": None,
            "reason": (
                "aclrtLaunchKernelWithHostArgs exposes blockDim but no "
                "mixBlockDim argument"),
        },
        "stream": stream,
        "host_args": {
            "size_bytes": len(host_args),
            "serialized_hex": host_args.hex(),
            "sha256": _sha256(host_args_path),
            "path": recorded_path(host_args_path),
        },
        "placeholders": {
            "count": placeholder_count,
            "entry_size_bytes": placeholder_entry_size,
            "entries": placeholders,
            "serialized_hex": placeholder_bytes.hex(),
            "sha256": _sha256(placeholder_path),
            "path": recorded_path(placeholder_path),
        },
        "configuration": {
            "present": True,
            "attribute_count": configuration_attr_count,
            "attribute_entry_size_bytes": configuration_attr_entry_size,
            "decoded_attributes": decoded_attributes,
            "serialized_hex": configuration_bytes.hex(),
            "sha256": _sha256(configuration_path),
            "path": recorded_path(configuration_path),
        },
        "device_kernel_submitted": False,
        "duration_consumed": False,
        "metadata_path": recorded_path(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
    }


def audit_nnopbase_launch_imports(readelf, nnopbase):
    completed = subprocess.run(
        [str(readelf), "-Ws", str(nnopbase)], check=True,
        capture_output=True, text=True)
    imports = sorted(set(re.findall(
        r"\bUND\s+(aclrtLaunchKernel[A-Za-z0-9_]*)", completed.stdout)))
    if imports != [EXPECTED_LAUNCH_API]:
        raise RuntimeError(
            f"libnnopbase launch imports changed: {imports}")
    return imports


def validate_interposer_symbols(nm, interposer):
    completed = subprocess.run(
        [str(nm), "-D", "--defined-only", str(interposer)], check=True,
        capture_output=True, text=True)
    defined = {
        line.split()[-1] for line in completed.stdout.splitlines()
        if line.split()
    }
    required = INTERPOSED_LAUNCH_APIS | {
        "midlayer_ascend_gemm_dry_launch_begin",
        "midlayer_ascend_gemm_dry_launch_end",
        "midlayer_ascend_gemm_dry_launch_error",
    }
    missing = required - defined
    if missing:
        raise RuntimeError(
            f"dry-launch interposer symbols are missing: {sorted(missing)}")


def _require_mapping(value, context):
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return value


def _require_keys(mapping, required, context):
    missing = set(required) - set(mapping)
    if missing:
        raise ValueError(f"{context} is missing fields: {sorted(missing)}")


def _require_nonempty_string(value, context):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_sha256(value, context):
    if (not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None):
        raise ValueError(f"{context} must be a lowercase SHA256")
    return value


def _build_authoritative_candidate(
        repository_root, source_config, models_root):
    relative_config = Path(source_config)
    if relative_config.is_absolute() or ".." in relative_config.parts:
        raise ValueError(
            "workload candidate source_config must be repository-relative")
    config_path = (repository_root / relative_config).resolve()
    if (not config_path.is_relative_to(repository_root)
            or not config_path.is_file()):
        raise FileNotFoundError(
            f"workload candidate source_config is missing: {config_path}")
    module_path = repository_root / BUILDER_SOURCE
    if not module_path.is_file():
        raise FileNotFoundError(
            f"authoritative workload candidate builder is missing: "
            f"{module_path}")
    module_name = "_midlayer_gemm_workload_candidates"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"cannot load authoritative workload candidate builder: "
            f"{module_path}")
    module = importlib.util.module_from_spec(spec)
    repository_text = str(repository_root)
    inserted_path = repository_text not in sys.path
    if inserted_path:
        sys.path.insert(0, repository_text)
    try:
        spec.loader.exec_module(module)
        builder = module.build_gemm_workload_candidate
        return builder(
            repository_root, config_path, models_root=models_root)
    finally:
        if inserted_path:
            sys.path.remove(repository_text)


def _require_key_positive_integer(value, context):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if value <= 0:
        raise ValueError(f"{context} must be positive")
    return value


def parse_supported_workload_key(key):
    if not isinstance(key, str) or not key:
        raise TypeError("GEMM workload key must be a non-empty string")
    values = json.loads(key)
    if not isinstance(values, list) or len(values) != len(
            GEMM_WORKLOAD_KEY_FIELDS):
        raise ValueError("GEMM workload key must contain exactly 16 fields")
    if json.dumps(values, separators=(",", ":")) != key:
        raise ValueError("GEMM workload key is not canonical")
    (dtype, variant, b, m, n, k, a_layout, b_layout, c_layout, trans_a,
     trans_b, epilogue, has_bias, position, tensor_parallel_size,
     expert_parallel_size) = values
    if dtype != "bf16":
        raise ValueError("Ascend dry-launch supports only BF16 GEMM keys")
    if variant not in ("gemm", "gemv"):
        raise ValueError(f"unsupported Ascend GEMM variant {variant!r}")
    if _require_key_positive_integer(b, "GEMM batch") != 1:
        raise ValueError("Ascend dry-launch supports only GEMM batch B=1")
    shape = {
        "m": _require_key_positive_integer(m, "GEMM M"),
        "n": _require_key_positive_integer(n, "GEMM N"),
        "k": _require_key_positive_integer(k, "GEMM K"),
    }
    if (a_layout, b_layout, c_layout) != (
            "row_major", "row_major", "row_major"):
        raise ValueError("Ascend dry-launch requires row-major A/B/C layouts")
    if not isinstance(trans_a, bool) or not isinstance(trans_b, bool):
        raise TypeError("GEMM transpose fields must be booleans")
    if trans_a or not trans_b:
        raise ValueError(
            "Ascend dry-launch requires trans_a=false and trans_b=true")
    if epilogue != "none" or has_bias is not False:
        raise ValueError("Ascend dry-launch does not support bias or epilogue")
    if not isinstance(position, str) or not position or position == "generic":
        raise ValueError("GEMM position must identify the logical operation")
    _require_key_positive_integer(
        tensor_parallel_size, "tensor_parallel_size")
    _require_key_positive_integer(
        expert_parallel_size, "expert_parallel_size")
    return {
        "workload_key": key,
        "dtype": dtype,
        "variant": variant,
        "b": b,
        **shape,
        "a_layout": a_layout,
        "b_layout": b_layout,
        "c_layout": c_layout,
        "trans_a": trans_a,
        "trans_b": trans_b,
        "epilogue": epilogue,
        "has_bias": has_bias,
        "position": position,
        "tensor_parallel_size": tensor_parallel_size,
        "expert_parallel_size": expert_parallel_size,
    }


def _canonical_physical_domain_sha256(keys):
    serialized = json.dumps(
        [list(key) for key in sorted(keys)], separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _read_selection_text(path):
    if path.suffix != ".zst":
        raise ValueError("canonical Ascend GEMM launch selection must be zstd")
    try:
        content = zstandard.ZstdDecompressor().decompress(path.read_bytes())
    except zstandard.ZstdError as error:
        raise ValueError("canonical Ascend GEMM launch selection is invalid zstd") from error
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("canonical Ascend GEMM launch selection must be UTF-8") from error


def _validate_selection_domain(selection_path, operations):
    selection_path = selection_path.resolve()
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"canonical Ascend GEMM launch selection is missing: "
            f"{selection_path}")
    with io.StringIO(_read_selection_text(selection_path), newline="") as stream:
        reader = csv.DictReader(stream)
        expected_fields = {
            "dtype", "variant", "b", "m", "n", "k", "a_layout",
            "b_layout", "c_layout", "trans_a", "trans_b", "epilogue",
            "has_bias", "tile_m", "tile_n", "tile_k", "pipeline_depth",
            "split_k", "reduction", "aic_workers", "worker_grid_m",
            "worker_grid_n", "worker_grid_k", "single_core_m",
            "single_core_n", "single_core_k", "base_m", "base_n",
            "base_k", "depth_a1", "depth_b1", "a_l1_k_tiles",
            "b_l1_k_tiles", "step_m", "step_n", "m_l2_regions",
            "n_l2_regions", "m_l2_block", "n_l2_block",
            "traversal_order", "a_uses_l2", "b_uses_l2",
        }
        if reader.fieldnames is None or set(reader.fieldnames) != expected_fields:
            raise ValueError("Ascend GEMM launch selection schema changed")
        selection_keys = set()
        for row in reader:
            if (row["dtype"] != "bf16" or row["b"] != "1"
                    or row["has_bias"] != "false"):
                raise ValueError("launch selection has unsupported physical row")
            key = (
                int(row["m"]), int(row["n"]), int(row["k"]),
                row["dtype"], False)
            if key in selection_keys:
                raise ValueError("launch selection repeats a physical GEMM key")
            selection_keys.add(key)
    operation_keys = {
        (operation["m"], operation["n"], operation["k"],
         operation["dtype"], False)
        for operation in operations
    }
    if len(operation_keys) != len(operations):
        raise ValueError("workload candidate repeats a physical GEMM key")
    if selection_keys != operation_keys:
        missing = len(selection_keys - operation_keys)
        extra = len(operation_keys - selection_keys)
        raise ValueError(
            "workload candidate physical domain differs from canonical "
            f"launch selection: missing={missing}, extra={extra}")
    return {
        "path": str(selection_path),
        "sha256": _sha256(selection_path),
        "physical_operation_count": len(selection_keys),
        "physical_domain_sha256": _canonical_physical_domain_sha256(
            selection_keys),
    }


def load_bulk_launch_selection(path, cube_math_type, repository_root):
    """Use the canonical physical launch-selection CSV without recomposition."""
    selection_path = path.resolve()
    if not selection_path.is_file():
        raise FileNotFoundError(f"launch selection is missing: {selection_path}")
    operations = []
    with io.StringIO(_read_selection_text(selection_path), newline="") as stream:
        reader = csv.DictReader(stream)
        expected_fields = {
            "dtype", "variant", "b", "m", "n", "k", "a_layout",
            "b_layout", "c_layout", "trans_a", "trans_b", "epilogue",
            "has_bias", "tile_m", "tile_n", "tile_k", "pipeline_depth",
            "split_k", "reduction", "aic_workers", "worker_grid_m",
            "worker_grid_n", "worker_grid_k", "single_core_m",
            "single_core_n", "single_core_k", "base_m", "base_n",
            "base_k", "depth_a1", "depth_b1", "a_l1_k_tiles",
            "b_l1_k_tiles", "step_m", "step_n", "m_l2_regions",
            "n_l2_regions", "m_l2_block", "n_l2_block",
            "traversal_order", "a_uses_l2", "b_uses_l2",
        }
        if reader.fieldnames is None or set(reader.fieldnames) != expected_fields:
            raise ValueError("Ascend GEMM launch selection schema changed")
        for index, row in enumerate(reader):
            if (row["dtype"] != "bf16" or row["b"] != "1"
                    or row["has_bias"] != "false"):
                raise ValueError("launch selection has unsupported physical row")
            operations.append({
                "operation_id": f"operation_{index:06d}",
                "b": 1,
                "m": _require_positive(int(row["m"]), "M"),
                "n": _require_positive(int(row["n"]), "N"),
                "k": _require_positive(int(row["k"]), "K"),
                "dtype": "bf16",
                "cube_math_type": cube_math_type,
                "workload_keys": [],
            })
    selection_domain = _validate_selection_domain(selection_path, operations)
    if len(operations) != EXACT_PHYSICAL_OPERATION_COUNT:
        raise ValueError("launch selection operation count changed")
    return {
        "path": str(selection_path),
        "sha256": _sha256(selection_path),
        "operation_count": len(operations),
        "launch_selection_domain": selection_domain,
        "operations": operations,
    }


def load_kernel_dataset_probe(path, cube_math_type, repository_root):
    """Build rank-two dry-launch requests from canonical GEMM observations.

    This is deliberately a launch-contract probe, not a timing evaluation:
    it consumes only the logical GEMM shape and dtype.  Batch GEMMs are
    flattened exactly as Dataset normalization projects them into the shared
    rank-two production contract.
    """

    repository_root = repository_root.resolve()
    dataset_path = path.resolve()
    if not dataset_path.is_relative_to(repository_root):
        raise ValueError("kernel dataset probe must stay below repository root")
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"kernel dataset probe is missing: {dataset_path}")
    relative_path = dataset_path.relative_to(repository_root)
    if relative_path.as_posix() != "kernel_dataset/ascend_910b/gemm.json.zst":
        raise ValueError(
            "kernel dataset probe requires the canonical Ascend GEMM dataset")
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", str(relative_path)],
        cwd=repository_root, check=True, capture_output=True, text=True)
    dataset_bytes = zstandard.ZstdDecompressor().decompress(
        dataset_path.read_bytes())
    dataset = _require_mapping(
        json.loads(dataset_bytes), "Ascend GEMM kernel dataset")
    if not dataset:
        raise ValueError("Ascend GEMM kernel dataset must not be empty")
    operations = []
    for case_id in sorted(dataset):
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Ascend GEMM kernel dataset case id is invalid")
        record = _require_mapping(dataset[case_id], f"dataset {case_id}")
        op_info = _require_mapping(record["op_info"], f"dataset {case_id}.op_info")
        if op_info["Type"] != "GEMM":
            raise ValueError(f"dataset {case_id} is not a GEMM observation")
        attributes = _require_mapping(
            op_info["Attr"], f"dataset {case_id}.op_info.Attr")
        if attributes["dtype"] != "bf16":
            raise ValueError(f"dataset {case_id} must use BF16")
        logical_shape = {}
        for field in ("B", "M", "N", "K"):
            value = attributes[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"dataset {case_id} logical {field} must be positive")
            logical_shape[field.lower()] = value
        operations.append({
            "operation_id": case_id,
            "b": 1,
            "m": logical_shape["b"] * logical_shape["m"],
            "n": logical_shape["n"],
            "k": logical_shape["k"],
            "dtype": "bf16",
            "cube_math_type": cube_math_type,
            "logical_shape": logical_shape,
            "workload_keys": [],
        })
    return {
        "path": str(dataset_path),
        "sha256": _sha256_bytes(dataset_bytes),
        "case_count": len(operations),
        "operation_count": len(operations),
        "operations": operations,
    }


def load_bulk_workload_candidate(
        path, cube_math_type, repository_root, models_root):
    candidate_bytes = path.read_bytes()
    candidate = _require_mapping(
        json.loads(candidate_bytes), "workload candidate")
    if set(candidate) != CANDIDATE_FIELDS:
        raise ValueError("workload candidate root fields changed")
    if candidate["schema_version"] != 2:
        raise ValueError("unsupported workload candidate schema")
    if candidate["artifact_type"] != "gemm_workload_candidate":
        raise ValueError("input is not a GEMM workload candidate")
    if candidate["duration_consumed"] is not False:
        raise ValueError("workload candidate must not consume duration")
    if candidate["workload_key_fields"] != list(GEMM_WORKLOAD_KEY_FIELDS):
        raise ValueError("workload candidate GEMM key fields changed")
    if candidate["physical_operation_fields"] != list(
            PHYSICAL_OPERATION_FIELDS):
        raise ValueError("workload candidate physical fields changed")
    coverage = _require_mapping(
        candidate["coverage_scope"], "workload candidate coverage_scope")
    if coverage["kind"] != EXACT_DOMAIN_STATUS:
        raise ValueError("Ascend workload candidate is not the exact domain")
    hardware_domains = _require_mapping(
        candidate["hardware_domains"], "hardware domains")
    if set(hardware_domains) != set(HARDWARE_IDS):
        raise ValueError("workload candidate hardware set changed")
    expected_domain = {
        "status": "exact",
        "domain_sha256": EXACT_DOMAIN_SHA256,
        "semantic_key_count": EXACT_SEMANTIC_KEY_COUNT,
        "physical_operation_count": EXACT_PHYSICAL_OPERATION_COUNT,
    }
    for hardware_id in HARDWARE_IDS:
        if hardware_domains[hardware_id] != expected_domain:
            raise ValueError(f"{hardware_id} workload domain is not exact")
    expected_cross_hardware = {
        "status": "exact_equal",
        "hardware_ids": list(HARDWARE_IDS),
        "domain_sha256": EXACT_DOMAIN_SHA256,
        "semantic_key_count": EXACT_SEMANTIC_KEY_COUNT,
        "physical_operation_count": EXACT_PHYSICAL_OPERATION_COUNT,
    }
    if candidate["cross_hardware"] != expected_cross_hardware:
        raise ValueError("cross-hardware workload domain changed")
    keys = candidate["workload_keys"]
    if not isinstance(keys, list) or not keys:
        raise ValueError("ascend_910b candidate_keys must be a non-empty list")
    if any(not isinstance(key, str) for key in keys):
        raise TypeError("ascend_910b candidate keys must be strings")
    if keys != sorted(set(keys)):
        raise ValueError("ascend_910b candidate keys must be sorted and unique")
    decoded = [parse_supported_workload_key(key) for key in keys]
    grouped = {}
    for entry in decoded:
        operation_key = (
            entry["b"], entry["m"], entry["n"], entry["k"],
            entry["dtype"], cube_math_type)
        if operation_key not in grouped:
            grouped[operation_key] = []
        grouped[operation_key].append(entry["workload_key"])
    operations = []
    operation_ids = {}
    for index, operation_key in enumerate(sorted(grouped)):
        operation_id = f"operation_{index:06d}"
        operation_ids[operation_key] = operation_id
        operations.append({
            "operation_id": operation_id,
            "b": operation_key[0],
            "m": operation_key[1],
            "n": operation_key[2],
            "k": operation_key[3],
            "dtype": operation_key[4],
            "cube_math_type": operation_key[5],
            "workload_keys": grouped[operation_key],
        })
    mappings = []
    for entry in decoded:
        operation_key = (
            entry["b"], entry["m"], entry["n"], entry["k"],
            entry["dtype"], cube_math_type)
        mappings.append({
            "workload_key": entry["workload_key"],
            "operation_id": operation_ids[operation_key],
        })
    if (candidate["semantic_key_count"] != EXACT_SEMANTIC_KEY_COUNT
            or candidate["physical_operation_count"]
            != EXACT_PHYSICAL_OPERATION_COUNT
            or len(keys) != EXACT_SEMANTIC_KEY_COUNT
            or len(operations) != EXACT_PHYSICAL_OPERATION_COUNT):
        raise ValueError("Ascend exact-domain counts are inconsistent")
    domain_payload = json.dumps(
        keys, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    if hashlib.sha256(domain_payload).hexdigest() != EXACT_DOMAIN_SHA256:
        raise ValueError("Ascend exact-domain workload-key set changed")
    authoritative_candidate = _build_authoritative_candidate(
        repository_root, candidate["source_config"], models_root)
    if candidate_bytes != _canonical_json_bytes(authoritative_candidate):
        raise ValueError(
            "workload candidate bytes differ from the canonical "
            "authoritative rebuilt exact launch domain")
    selection_domain = _validate_selection_domain(
        repository_root / ASCEND_GEMM_LAUNCH_SELECTION, operations)
    if selection_domain["physical_operation_count"] != EXACT_PHYSICAL_OPERATION_COUNT:
        raise ValueError("canonical launch-selection operation count changed")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_bytes(candidate_bytes),
        "input_manifest": candidate,
        "workload_key_count": len(keys),
        "operation_count": len(operations),
        "candidate_scope": EXACT_DOMAIN_STATUS,
        "final_exact_workload_universe": True,
        "operations": operations,
        "workload_mappings": mappings,
        "launch_selection_domain": selection_domain,
    }


def validate_bulk_shard_results(shard_results):
    if not isinstance(shard_results, list) or not shard_results:
        raise ValueError("bulk shard results must be a non-empty list")
    normalized = []
    candidate_sha256 = None
    input_manifest = None
    full_operation_count = None
    full_workload_key_count = None
    input_rank = None
    seen_operation_ids = set()
    seen_result_paths = set()
    seen_result_sha256s = set()
    seen_workload_keys = set()
    for shard_index, value in enumerate(shard_results):
        context = f"bulk shard {shard_index}"
        shard = _require_mapping(value, context)
        _require_keys(
            shard,
            {"schema_version", "artifact_type", "workload_candidate",
             "full_input_workload_key_count", "full_input_operation_count",
             "selected_operation_range", "selected_operation_count",
             "operation_count", "workload_key_count", "operations",
             "workload_mappings", "device_kernel_submitted",
             "duration_consumed", "input_rank", "launch_selection_domain"},
            context)
        if (shard["schema_version"] != 4
                or shard["artifact_type"]
                != "ascend_910b_gemm_dry_launch_bulk"):
            raise ValueError(f"{context} has an unsupported artifact schema")
        shard_input_rank = shard["input_rank"]
        if shard_input_rank not in (2, 3):
            raise ValueError(f"{context} has an unsupported input rank")
        if (shard["device_kernel_submitted"] is not False
                or shard["duration_consumed"] is not False):
            raise ValueError(f"{context} claims device execution or duration")
        candidate = _require_mapping(
            shard["workload_candidate"], f"{context} candidate")
        _require_keys(
            candidate, {"sha256", "candidate_scope", "input_manifest"},
            f"{context} candidate")
        shard_candidate_sha256 = _require_sha256(
            candidate["sha256"], f"{context} candidate SHA256")
        if candidate["candidate_scope"] != EXACT_DOMAIN_STATUS:
            raise ValueError(f"{context} candidate scope is unsupported")
        selection_domain = _require_mapping(
            shard["launch_selection_domain"],
            f"{context} launch selection domain")
        _require_keys(
            selection_domain,
            {"path", "sha256", "physical_operation_count",
             "physical_domain_sha256"},
            f"{context} launch selection domain")
        shard_input_manifest = _require_mapping(
            candidate["input_manifest"], f"{context} input manifest")
        shard_full_operation_count = shard["full_input_operation_count"]
        shard_full_workload_key_count = shard["full_input_workload_key_count"]
        for count, label in (
                (shard_full_operation_count, "full operation count"),
                (shard_full_workload_key_count, "full workload key count")):
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"{context} {label} must be positive")
        if candidate_sha256 is None:
            candidate_sha256 = shard_candidate_sha256
            input_manifest = shard_input_manifest
            full_operation_count = shard_full_operation_count
            full_workload_key_count = shard_full_workload_key_count
            input_rank = shard_input_rank
        elif (shard_candidate_sha256 != candidate_sha256
              or shard_input_manifest != input_manifest
              or shard_full_operation_count != full_operation_count
              or shard_full_workload_key_count != full_workload_key_count
              or shard_input_rank != input_rank):
            raise ValueError(
                "bulk shards disagree on candidate identity or full counts")
        selected_range = _require_mapping(
            shard["selected_operation_range"],
            f"{context} selected operation range")
        _require_keys(
            selected_range, {"start_index", "end_index"},
            f"{context} selected operation range")
        start_index = selected_range["start_index"]
        end_index = selected_range["end_index"]
        if (isinstance(start_index, bool) or not isinstance(start_index, int)
                or isinstance(end_index, bool)
                or not isinstance(end_index, int)
                or start_index < 0 or end_index <= start_index):
            raise ValueError(f"{context} has an invalid non-empty range")
        operations = shard["operations"]
        mappings = shard["workload_mappings"]
        if not isinstance(operations, list) or not isinstance(mappings, list):
            raise TypeError(f"{context} operations and mappings must be lists")
        selected_count = end_index - start_index
        if (shard["selected_operation_count"] != selected_count
                or shard["operation_count"] != selected_count
                or len(operations) != selected_count
                or shard["workload_key_count"] != len(mappings)):
            raise ValueError(f"{context} selected counts are inconsistent")
        operation_results = {}
        operation_workload_keys = set()
        for offset, operation_value in enumerate(operations):
            operation = _require_mapping(
                operation_value, f"{context} operation {offset}")
            _require_keys(
                operation, {"operation_id", "workload_keys", "result"},
                f"{context} operation {offset}")
            expected_operation_id = f"operation_{start_index + offset:06d}"
            operation_id = operation["operation_id"]
            if operation_id != expected_operation_id:
                raise ValueError(
                    f"{context} operation sequence is not globally indexed")
            if operation_id in seen_operation_ids:
                raise ValueError(f"duplicate bulk operation {operation_id}")
            seen_operation_ids.add(operation_id)
            result = _require_mapping(
                operation["result"], f"{context} {operation_id} result")
            _require_keys(
                result, {"path", "sha256"},
                f"{context} {operation_id} result")
            result_path = _require_nonempty_string(
                result["path"], f"{context} {operation_id} result path")
            result_sha256 = _require_sha256(
                result["sha256"], f"{context} {operation_id} result SHA256")
            if (result_path in seen_result_paths
                    or result_sha256 in seen_result_sha256s):
                raise ValueError("bulk operations reuse a result identity")
            seen_result_paths.add(result_path)
            seen_result_sha256s.add(result_sha256)
            operation_results[operation_id] = result
            workload_keys = operation["workload_keys"]
            if (not isinstance(workload_keys, list) or not workload_keys
                    or any(not isinstance(key, str) or not key
                           for key in workload_keys)
                    or len(workload_keys) != len(set(workload_keys))):
                raise ValueError(
                    f"{context} {operation_id} workload keys are invalid")
            for workload_key in workload_keys:
                if workload_key in operation_workload_keys:
                    raise ValueError(
                        f"{context} repeats workload key {workload_key!r}")
                operation_workload_keys.add(workload_key)
        mapped_workload_keys = set()
        for mapping_index, mapping_value in enumerate(mappings):
            mapping = _require_mapping(
                mapping_value, f"{context} mapping {mapping_index}")
            _require_keys(
                mapping, {"workload_key", "operation_id", "result"},
                f"{context} mapping {mapping_index}")
            workload_key = mapping["workload_key"]
            operation_id = mapping["operation_id"]
            if workload_key in mapped_workload_keys:
                raise ValueError(f"{context} repeats a workload mapping")
            if operation_id not in operation_results:
                raise ValueError(f"{context} maps to an unselected operation")
            if mapping["result"] != operation_results[operation_id]:
                raise ValueError(f"{context} mapping result identity disagrees")
            mapped_workload_keys.add(workload_key)
        if mapped_workload_keys != operation_workload_keys:
            raise ValueError(f"{context} workload mappings are incomplete")
        overlap = seen_workload_keys & mapped_workload_keys
        if overlap:
            raise ValueError("bulk shards repeat workload mappings")
        seen_workload_keys.update(mapped_workload_keys)
        normalized.append((start_index, end_index))
    cursor = 0
    for start_index, end_index in sorted(normalized):
        if start_index != cursor:
            raise ValueError("bulk shard ranges contain a gap or overlap")
        cursor = end_index
    if cursor != full_operation_count:
        raise ValueError("bulk shard ranges do not cover the full operation set")
    hardware = _require_mapping(
        input_manifest["hardware_domains"]["ascend_910b"],
        "aggregate input Ascend hardware domain")
    if hardware["status"] != "exact":
        raise ValueError("aggregate input Ascend domain is not exact")
    candidate_keys = input_manifest["workload_keys"]
    if (not isinstance(candidate_keys, list)
            or len(candidate_keys) != full_workload_key_count
            or set(candidate_keys) != seen_workload_keys):
        raise ValueError(
            "bulk shard workload mappings do not cover the input manifest")
    if len(seen_operation_ids) != full_operation_count:
        raise ValueError("bulk shards do not contain every unique operation")
    return {
        "candidate_sha256": candidate_sha256,
        "input_rank": input_rank,
        "full_operation_count": full_operation_count,
        "full_workload_key_count": full_workload_key_count,
        "shard_count": len(shard_results),
    }


def validate_post_capture_inputs(
        initial_source_identity, final_source_identity,
        initial_cann_abi, final_cann_abi,
        initial_bulk_plan=None, final_bulk_plan=None):
    if final_source_identity != initial_source_identity:
        raise RuntimeError("query source identity changed during capture")
    if final_cann_abi != initial_cann_abi:
        raise RuntimeError("CANN ABI identity changed during capture")
    if (initial_bulk_plan is None) != (final_bulk_plan is None):
        raise RuntimeError("bulk candidate revalidation state is inconsistent")
    if (initial_bulk_plan is not None
            and final_bulk_plan != initial_bulk_plan):
        raise RuntimeError(
            "workload candidate or rebuilt source evidence changed during "
            "capture")


def validate_captured_kernel_symbols(operation_results, kernel_symbols):
    if not isinstance(kernel_symbols, frozenset) or not kernel_symbols:
        raise TypeError(
            "MatMul kernel symbol union must be a non-empty frozenset")
    for index, result in enumerate(operation_results):
        symbol = result["dry_launch"]["launch"]["kernel_stub"]
        if symbol not in kernel_symbols:
            raise ValueError(
                f"operation {index} kernel is absent from pinned metadata: "
                f"{symbol!r}")


def _operation_layout(operation):
    batch = operation["b"]
    m = operation["m"]
    n = operation["n"]
    k = operation["k"]
    input_rank = operation["input_rank"]
    if input_rank not in (2, 3):
        raise ValueError("operation input rank must be 2 or 3")
    if input_rank == 2:
        if batch != 1:
            raise ValueError("rank-two operation requires B=1")
        return {
            "input_rank": 2,
            "lhs": {
                "kind": "row_major",
                "view_shape": [m, k],
                "strides": [k, 1],
                "storage_shape": [m, k],
            },
            "rhs": {
                "kind": "transposed_view",
                "view_shape": [k, n],
                "strides": [1, k],
                "storage_shape": [n, k],
            },
            "output": {
                "kind": "row_major",
                "view_shape": [m, n],
                "strides": [n, 1],
                "storage_shape": [m, n],
            },
        }
    return {
        "input_rank": 3,
        "lhs": {
            "kind": "row_major",
            "view_shape": [batch, m, k],
            "strides": [m * k, k, 1],
            "storage_shape": [batch, m, k],
        },
        "rhs": {
            "kind": "transposed_view",
            "view_shape": [k, n],
            "strides": [1, k],
            "storage_shape": [n, k],
        },
        "output": {
            "kind": "row_major",
            "view_shape": [batch, m, n],
            "strides": [m * n, n, 1],
            "storage_shape": [batch, m, n],
        },
    }


def _operation_result_from_artifacts(
        operation, operation_dir, launch_imports, output_dir):
    metadata_path = operation_dir / "helper_metadata.txt"
    capture_dir = operation_dir / "raw_capture"
    helper_result = parse_helper_metadata(metadata_path)
    if helper_result["batch_size"] != operation["b"]:
        raise ValueError("helper batch size differs from requested operation")
    if helper_result["input_rank"] != operation["input_rank"]:
        raise ValueError("helper input rank differs from requested operation")
    launch = parse_dry_launch_capture(capture_dir, output_dir)
    executor_tiling = extract_embedded_tiling(launch)
    return {
        "schema_version": 4,
        "operator": "aclnnMatmul",
        "operation_id": operation["operation_id"],
        "shape": {
            "b": operation["b"],
            "m": operation["m"],
            "n": operation["n"],
            "k": operation["k"],
        },
        "input_rank": operation["input_rank"],
        "dtype": operation["dtype"],
        "layout": _operation_layout(operation),
        "cube_math_type": operation["cube_math_type"],
        "workload_keys": operation["workload_keys"],
        **helper_result,
        "executor_tiling": executor_tiling,
        "dry_launch": {
            "expected_launch_count": 1,
            "captured_launch_count": 1,
            "interposed_launch_apis": sorted(INTERPOSED_LAUNCH_APIS),
            "nnopbase_imported_launch_apis": launch_imports,
            "launch": launch,
        },
    }


def _run_dry_launch_operation(
        operation, operation_dir, query_log, helper, guard,
        base_environment, device_id, timeout_seconds, launch_imports,
        output_dir):
    metadata_path = operation_dir / "helper_metadata.txt"
    capture_dir = operation_dir / "raw_capture"
    capture_dir.mkdir()
    query_environment = base_environment.copy()
    query_environment["LD_PRELOAD"] = str(guard)
    query_environment["MIDLAYER_ASCEND_GEMM_DRY_LAUNCH_DIR"] = str(
        capture_dir)
    query_environment["ASCEND_DEVICE_ID"] = str(device_id)
    query_environment["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
    query_command = [
        str(helper), str(operation["input_rank"]), str(operation["b"]), str(operation["m"]),
        str(operation["n"]), str(operation["k"]),
        str(operation["cube_math_type"]), "0", str(metadata_path),
    ]
    _run_checked(
        query_command, query_log, environment=query_environment,
        timeout=timeout_seconds)
    return _operation_result_from_artifacts(
        operation, operation_dir, launch_imports, output_dir)


def _inline_capture_paths(operation_result):
    launch = operation_result["dry_launch"]["launch"]
    for artifact in ("host_args", "placeholders", "configuration"):
        launch[artifact]["path"] = "embedded_in_result_json"
    launch["metadata_path"] = "embedded_in_result_json"
    operation_result["capture_storage"] = "inline_result_json"
    return operation_result


def _run_dry_launch_batch(
        operations, query_log, helper, guard,
        base_environment, device_id, timeout_seconds, launch_imports,
        output_dir, scratch_dir):
    scratch_dir = scratch_dir.resolve()
    if not scratch_dir.is_dir():
        raise FileNotFoundError(f"capture scratch directory is missing: {scratch_dir}")
    environment = base_environment.copy()
    environment["LD_PRELOAD"] = str(guard)
    environment["ASCEND_DEVICE_ID"] = str(device_id)
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
    command = [str(helper), "--batch", "0"]
    deadline = time.monotonic() + timeout_seconds
    stderr = ""
    results = []
    with tempfile.TemporaryDirectory(
            prefix="midlayer_ascend_gemm_dry_launch_", dir=scratch_dir) as temp:
        temporary_root = Path(temp)
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, env=environment)
        try:
            for operation in operations:
                if time.monotonic() >= deadline:
                    raise TimeoutError("persistent Ascend GEMM helper timed out")
                operation_dir = temporary_root / operation["operation_id"]
                operation_dir.mkdir()
                capture_dir = operation_dir / "raw_capture"
                capture_dir.mkdir()
                metadata_path = operation_dir / "helper_metadata.txt"
                request = (
                    f"{operation['operation_id']} {operation['input_rank']} "
                    f"{operation['b']} {operation['m']} {operation['n']} "
                    f"{operation['k']} {operation['cube_math_type']} "
                    f"{capture_dir} {metadata_path}\n")
                if process.stdin is None or process.stdout is None:
                    raise RuntimeError("persistent Ascend GEMM helper pipes are absent")
                process.stdin.write(request)
                process.stdin.flush()
                identity = process.stdout.readline().rstrip("\n")
                if identity != operation["operation_id"]:
                    raise RuntimeError(
                        "persistent Ascend GEMM helper result identity/order "
                        f"changed: expected={operation['operation_id']!r}, "
                        f"actual={identity!r}")
                result = _operation_result_from_artifacts(
                    operation, operation_dir, launch_imports, output_dir)
                results.append(_inline_capture_paths(result))
                shutil.rmtree(operation_dir)
            process.stdin.close()
            remaining = max(0.1, deadline - time.monotonic())
            if process.wait(timeout=remaining) != 0:
                raise RuntimeError("persistent Ascend GEMM helper failed")
            if process.stdout.read():
                raise RuntimeError("persistent Ascend GEMM helper emitted extra output")
            if process.stderr is None:
                raise RuntimeError("persistent Ascend GEMM helper stderr is absent")
            stderr = process.stderr.read()
        except BaseException:
            process.kill()
            process.wait()
            if process.stderr is not None:
                stderr = process.stderr.read()
            raise
        finally:
            query_log.write_text(stderr, encoding="utf-8")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--workload-candidate", type=Path)
    parser.add_argument("--launch-selection", type=Path)
    parser.add_argument("--kernel-dataset", type=Path)
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--b", type=int)
    parser.add_argument("--input-rank", type=int, choices=(2, 3), required=True)
    parser.add_argument("--cube-math-type", type=int, default=0)
    parser.add_argument(
        "--cann-root", type=Path,
        default=CANN_ROOT)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--npu-smi", default="npu-smi")
    parser.add_argument("--nm", default="nm")
    parser.add_argument("--readelf", default="readelf")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--scratch-dir", type=Path, default=Path("/tmp"))
    args = parser.parse_args(argv)
    if args.device_id < 0:
        raise ValueError("device id must be non-negative")
    if args.cube_math_type != 0:
        raise ValueError("this PoC supports only cubeMathType=0")
    _require_positive(args.timeout_seconds, "timeout")
    if not args.scratch_dir.is_dir():
        raise FileNotFoundError(f"--scratch-dir is absent: {args.scratch_dir}")
    repository_root = args.repository_root.resolve()
    source_identity = collect_source_identity(
        repository_root, args.baseline_commit)
    bulk_inputs = sum(
        value is not None
        for value in (
            args.workload_candidate, args.launch_selection,
            args.kernel_dataset))
    if bulk_inputs > 1:
        raise ValueError(
            "--workload-candidate, --launch-selection, and --kernel-dataset "
            "are mutually exclusive")
    if bulk_inputs == 0:
        if args.models_root is not None:
            raise ValueError("--models-root requires --workload-candidate")
        if args.start_index is not None or args.end_index is not None:
            raise ValueError(
                "--start-index/--end-index require a bulk input")
        shape = {}
        for name in ("b", "m", "n", "k"):
            value = vars(args)[name]
            if value is None:
                value = SINGLE_SHAPE_DEFAULTS[name]
            shape[name] = _require_positive(value, name.upper())
        mode = "single_shape"
        operations = [{
            "operation_id": "single_shape",
            "input_rank": args.input_rank,
            "b": shape["b"],
            "m": shape["m"],
            "n": shape["n"],
            "k": shape["k"],
            "dtype": "bf16",
            "cube_math_type": args.cube_math_type,
            "workload_keys": [],
        }]
        bulk_plan = None
    elif args.workload_candidate is not None:
        if args.launch_selection is not None:
            raise ValueError(
                "--workload-candidate and --launch-selection are exclusive")
        if args.models_root is None:
            raise ValueError("--workload-candidate requires --models-root")
        if any(vars(args)[name] is not None for name in ("b", "m", "n", "k")):
            raise ValueError(
                "bulk mode cannot combine --workload-candidate with "
                "--b/--m/--n/--k")
        candidate_path = args.workload_candidate.resolve()
        if not candidate_path.is_file():
            raise FileNotFoundError(
                f"workload candidate is missing: {candidate_path}")
        mode = "bulk_candidate"
        bulk_plan = load_bulk_workload_candidate(
            candidate_path, args.cube_math_type, repository_root,
            args.models_root)
        full_operation_count = bulk_plan["operation_count"]
        start_index = 0 if args.start_index is None else args.start_index
        end_index = (
            full_operation_count if args.end_index is None
            else args.end_index)
        if start_index < 0 or end_index < 0:
            raise ValueError("bulk operation shard indices must be non-negative")
        if (start_index > full_operation_count
                or end_index > full_operation_count):
            raise ValueError(
                "bulk operation shard range exceeds the full operation count")
        if start_index >= end_index:
            raise ValueError("bulk operation shard must be non-empty")
        operations = bulk_plan["operations"][start_index:end_index]
        operations = [
            {**operation, "input_rank": args.input_rank}
            for operation in operations]
    elif args.launch_selection is not None:
        if args.models_root is not None:
            raise ValueError("--models-root is not used with --launch-selection")
        if any(vars(args)[name] is not None for name in ("b", "m", "n", "k")):
            raise ValueError(
                "launch-selection bulk mode cannot combine --b/--m/--n/--k")
        selection_path = args.launch_selection.resolve()
        mode = "bulk_launch_selection"
        bulk_plan = load_bulk_launch_selection(
            selection_path, args.cube_math_type, repository_root)
        full_operation_count = bulk_plan["operation_count"]
        start_index = 0 if args.start_index is None else args.start_index
        end_index = (
            full_operation_count if args.end_index is None
            else args.end_index)
        if start_index < 0 or end_index < 0:
            raise ValueError("bulk operation shard indices must be non-negative")
        if (start_index > full_operation_count
                or end_index > full_operation_count):
            raise ValueError(
                "bulk operation shard range exceeds the full operation count")
        if start_index >= end_index:
            raise ValueError("bulk operation shard must be non-empty")
        operations = [
            {**operation, "input_rank": args.input_rank}
            for operation in bulk_plan["operations"][start_index:end_index]]
    elif args.kernel_dataset is not None:
        if args.models_root is not None:
            raise ValueError("--models-root is not used with --kernel-dataset")
        if args.start_index is not None or args.end_index is not None:
            raise ValueError(
                "--start-index/--end-index are not supported for "
                "--kernel-dataset")
        if any(vars(args)[name] is not None for name in ("b", "m", "n", "k")):
            raise ValueError(
                "--kernel-dataset cannot combine --b/--m/--n/--k")
        if args.input_rank != 2:
            raise ValueError(
                "--kernel-dataset requires rank-two normalized launches")
        mode = "kernel_dataset_probe"
        bulk_plan = load_kernel_dataset_probe(
            args.kernel_dataset, args.cube_math_type, repository_root)
        operations = [
            {**operation, "input_rank": args.input_rank}
            for operation in bulk_plan["operations"]]
        start_index = 0
        end_index = bulk_plan["operation_count"]
    else:
        raise AssertionError("validated GEMM capture input disappeared")
    _require_output_scope(args.output_dir, repository_root)
    output_dir = args.output_dir.resolve()
    build_dir = output_dir / "build"
    logs_dir = output_dir / "logs"
    build_dir.mkdir()
    logs_dir.mkdir()
    cann_root = args.cann_root.resolve()
    initial_cann_abi, kernel_symbols = collect_cann_abi_identity(cann_root)
    compiler = _resolve_executable(args.compiler, "C++ compiler")
    npu_smi = _resolve_executable(args.npu_smi, "npu-smi")
    nm = _resolve_executable(args.nm, "nm")
    readelf = _resolve_executable(args.readelf, "readelf")
    helper = build_dir / "tiling_query_helper"
    guard = build_dir / "libgemm_launch_guard.so"
    include_dir = cann_root / "aarch64-linux/include"
    library_dir = cann_root / "aarch64-linux/lib64"
    helper_compile = [
        str(compiler), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
        "-o", str(helper), str(repository_root / HELPER_SOURCE),
        "-I", str(include_dir), "-L", str(library_dir),
        "-Wl,-rpath," + str(library_dir), "-lopapi", "-lnnopbase",
        "-lascendcl", "-ldl",
    ]
    guard_compile = [
        str(compiler), "-std=c++17", "-O2", "-fPIC", "-shared", "-Wall",
        "-Wextra", "-Werror", "-o", str(guard),
        str(repository_root / GUARD_SOURCE), "-I", str(include_dir),
        "-L", str(library_dir), "-Wl,-rpath," + str(library_dir),
        "-lascendcl", "-ldl", "-pthread",
    ]
    _run_checked(helper_compile, logs_dir / "helper_compile.log")
    _run_checked(guard_compile, logs_dir / "guard_compile.log")
    if not helper.is_file() or not guard.is_file():
        raise RuntimeError("compiler did not produce helper and guard")
    validate_interposer_symbols(nm, guard)
    launch_imports = audit_nnopbase_launch_imports(
        readelf, library_dir / "libnnopbase.so")

    environment = _runtime_environment(cann_root)
    npu_smi_path = logs_dir / "npu_smi_before.txt"
    _run_checked(
        [str(npu_smi), "info"], npu_smi_path, environment=environment,
        timeout=60)
    validate_npu_smi_idle(npu_smi_path, args.device_id)
    if "LD_PRELOAD" in environment and environment["LD_PRELOAD"]:
        raise ValueError("GEMM tiling query requires an empty LD_PRELOAD")
    npu_smi_after_path = logs_dir / "npu_smi_after.txt"
    operation_results = []
    result_paths = {}
    if mode == "bulk_candidate":
        operations_dir = output_dir / "operations"
        operations_dir.mkdir()
    elif mode in (
            "single_shape", "bulk_launch_selection",
            "kernel_dataset_probe"):
        operations_dir = output_dir
    else:
        raise AssertionError(f"unknown GEMM query mode {mode!r}")
    try:
        if mode in (
                "bulk_candidate", "bulk_launch_selection",
                "kernel_dataset_probe"):
            operation_results = _run_dry_launch_batch(
                operations, logs_dir / "batch_query.log",
                helper, guard,
                environment, args.device_id, args.timeout_seconds,
                launch_imports, output_dir, args.scratch_dir)
            if mode == "bulk_candidate":
                for operation, operation_result in zip(
                        operations, operation_results):
                    operation_dir = operations_dir / operation["operation_id"]
                    operation_result_path = operation_dir / "result.json"
                    _write_json(operation_result_path, operation_result)
                    result_paths[operation["operation_id"]] = {
                        "path": _output_path(operation_result_path, output_dir),
                        "sha256": _sha256(operation_result_path),
                    }
        elif mode == "single_shape":
            operation_results = [_run_dry_launch_operation(
                operations[0], output_dir, logs_dir / "query.log", helper,
                guard, environment, args.device_id, args.timeout_seconds,
                launch_imports, output_dir)]
        else:
            raise AssertionError(f"unknown GEMM query mode {mode!r}")
        validate_captured_kernel_symbols(operation_results, kernel_symbols)
    finally:
        _run_checked(
            [str(npu_smi), "info"], npu_smi_after_path,
            environment=environment, timeout=60)
    validate_npu_smi_idle(npu_smi_after_path, args.device_id)
    final_cann_abi, final_kernel_symbols = collect_cann_abi_identity(cann_root)
    if final_kernel_symbols != kernel_symbols:
        raise RuntimeError("MatMul kernel symbol union changed during capture")
    final_source_identity = collect_source_identity(
        repository_root, args.baseline_commit)
    if mode == "bulk_candidate":
        final_bulk_plan = load_bulk_workload_candidate(
            candidate_path, args.cube_math_type, repository_root,
            args.models_root)
    elif mode == "bulk_launch_selection":
        final_bulk_plan = load_bulk_launch_selection(
            selection_path, args.cube_math_type, repository_root)
    elif mode == "kernel_dataset_probe":
        final_bulk_plan = load_kernel_dataset_probe(
            args.kernel_dataset, args.cube_math_type, repository_root)
    elif mode == "single_shape":
        final_bulk_plan = None
    else:
        raise AssertionError(f"unknown GEMM query mode {mode!r}")
    validate_post_capture_inputs(
        source_identity, final_source_identity,
        initial_cann_abi, final_cann_abi, bulk_plan, final_bulk_plan)
    common = {
        "runner_version": RUNNER_VERSION,
        "baseline_commit": args.baseline_commit,
        "source_provenance": source_identity,
        "cann_version": initial_cann_abi["cann_version"],
        "cann_abi_provenance": initial_cann_abi,
        "hardware_id": "ascend_910b",
        "device_id": args.device_id,
        "input_rank": args.input_rank,
        "build_and_abi_audit": {
            "helper_compile_count": 1,
            "interposer_compile_count": 1,
            "helper_path": _output_path(helper, output_dir),
            "helper_sha256": _sha256(helper),
            "interposer_path": _output_path(guard, output_dir),
            "interposer_sha256": _sha256(guard),
            "interposed_launch_apis": sorted(INTERPOSED_LAUNCH_APIS),
            "nnopbase_imported_launch_apis": launch_imports,
        },
        "limitations": list(CAPTURE_LIMITATIONS),
    }
    if mode == "single_shape":
        result = {**operation_results[0], **common, "mode": mode}
    elif mode == "bulk_candidate":
        selected_operation_ids = {
            operation["operation_id"] for operation in operations}
        operation_records = []
        for operation in operations:
            operation_records.append({
                **operation,
                "result": result_paths[operation["operation_id"]],
            })
        mappings = []
        for mapping in bulk_plan["workload_mappings"]:
            if mapping["operation_id"] not in selected_operation_ids:
                continue
            mappings.append({
                **mapping,
                "result": result_paths[mapping["operation_id"]],
            })
        result = {
            "schema_version": 4,
            "artifact_type": "ascend_910b_gemm_dry_launch_bulk",
            **common,
            "mode": mode,
            "operator": "aclnnMatmul",
            "cube_math_type": args.cube_math_type,
            "input_rank": args.input_rank,
            "workload_candidate": {
                "path": bulk_plan["path"],
                "sha256": bulk_plan["sha256"],
                "candidate_scope": bulk_plan["candidate_scope"],
                "input_manifest": bulk_plan["input_manifest"],
                "coverage_claim": "complete_exact_legal_domain",
                "final_exact_workload_universe": (
                    bulk_plan["final_exact_workload_universe"]),
            },
            "launch_selection_domain": bulk_plan["launch_selection_domain"],
            "workload_key_fields": list(GEMM_WORKLOAD_KEY_FIELDS),
            "full_input_workload_key_count": bulk_plan["workload_key_count"],
            "workload_key_count": len(mappings),
            "full_input_operation_count": bulk_plan["operation_count"],
            "selected_operation_range": {
                "start_index": start_index,
                "end_index": end_index,
            },
            "selected_operation_count": len(operations),
            "operation_count": len(operations),
            "workload_mappings": mappings,
            "operations": operation_records,
            "execute_api_called": True,
            "device_kernel_submitted": False,
            "duration_consumed": False,
        }
    elif mode == "bulk_launch_selection":
        operation_records = []
        for operation, operation_result in zip(operations, operation_results):
            operation_records.append({
                **operation,
                "capture": operation_result,
                "capture_sha256": _sha256_bytes(
                    _canonical_json_bytes(operation_result)),
            })
        result = {
            "schema_version": 1,
            "artifact_type": "ascend_910b_gemm_rank2_dry_launch_selection_bulk",
            **common,
            "mode": mode,
            "operator": "aclnnMatmul",
            "cube_math_type": args.cube_math_type,
            "input_rank": args.input_rank,
            "launch_selection_domain": bulk_plan["launch_selection_domain"],
            "full_input_operation_count": bulk_plan["operation_count"],
            "selected_operation_range": {
                "start_index": start_index,
                "end_index": end_index,
            },
            "selected_operation_count": len(operations),
            "operation_count": len(operations),
            "operations": operation_records,
            "execute_api_called": True,
            "device_kernel_submitted": False,
            "duration_consumed": False,
        }
    elif mode == "kernel_dataset_probe":
        operation_records = []
        for operation, operation_result in zip(operations, operation_results):
            operation_records.append({
                **operation,
                "capture": operation_result,
                "capture_sha256": _sha256_bytes(
                    _canonical_json_bytes(operation_result)),
            })
        result = {
            "schema_version": 1,
            "artifact_type": "ascend_910b_gemm_kernel_dataset_dry_launch_probe",
            **common,
            "mode": mode,
            "operator": "aclnnMatmul",
            "cube_math_type": args.cube_math_type,
            "input_rank": args.input_rank,
            "kernel_dataset": {
                "path": bulk_plan["path"],
                "sha256": bulk_plan["sha256"],
                "case_count": bulk_plan["case_count"],
            },
            "full_input_operation_count": bulk_plan["operation_count"],
            "operation_count": len(operations),
            "operations": operation_records,
            "execute_api_called": True,
            "device_kernel_submitted": False,
            "duration_consumed": False,
        }
    else:
        raise AssertionError(f"unknown GEMM query mode {mode!r}")
    result_path = output_dir / "result.json"
    _write_json(result_path, result)
    print(json.dumps({
        "result_path": str(result_path),
        "result_sha256": _sha256(result_path),
        "operation_count": len(operations),
        "device_kernel_submitted": False,
        "duration_consumed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
