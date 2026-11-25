"""Configuration and path helpers owned by the kernel dataset pipeline."""

import hashlib
import json
import math
import os
import shutil
import sys

import yaml

from src.kernel.hardware_ids import canonical_hardware_id
from src.kernel.dataset_storage import (
    KERNEL_DATASET_SUFFIX,
    load_kernel_dataset_document,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


_KERNEL_PIPELINE_SETTING_FIELDS = {
    "ascend_910b": frozenset({
        "aic_metrics", "fa_launch_skip", "gpu_device_id",
        "kernel_name_filter", "moe_launch_count",
        "moe_warmup_skip", "rope_launch_skip", "timeline_aic_metrics",
        "timeline_duration_seconds",
        "vector_launch_skip", "warmup",
    }),
    "metax_c500": frozenset({
        "gpu_device_id",
    }),
    "nvidia_a800": frozenset({
        "gpu_device_id",
        "ncu_csv_output_dir", "ncu_rep_output_dir", "nvtx_range_name",
        "vector_timing_csv",
    }),
    "samsung_hbm_pim": frozenset({
        "sim_timeout_s", "unindp_arch",
        "unindp_quicksearch", "unindp_topk",
        "unindp_workspace_subdir",
    }),
}

_WORKLOAD_SCHEMAS = {
    "gemm": (
        frozenset({"kernel_name", "layers", "workload_configs"}),
        frozenset({"batch", "seq_len"}),
        frozenset({"dtype", "k", "n", "name"}),
    ),
    "fused_moe": (
        frozenset({"kernel_name", "layers", "workload_configs"}),
        frozenset({"num_tokens"}),
        frozenset({
            "dtype", "hidden_size", "intermediate_size", "name",
            "num_experts", "top_k",
        }),
    ),
    "flash_attn": (
        frozenset({"kernel_name", "layers", "workload_configs"}),
        frozenset({
            "batch", "causal", "kv_dist", "kv_layout", "num_splits",
            "seq_len_cache", "seq_len_q", "split_policy",
        }),
        frozenset({
            "dtype", "head_dim", "name", "num_heads_kv", "num_heads_q",
            "override_only",
        }),
    ),
    "vector": (
        frozenset({"kernel_name", "layers", "op_groups"}),
        frozenset({"batch", "layers", "op", "op2", "toks"}),
        frozenset({
            "dtype", "head_dim", "hidden", "intermediate", "name",
            "num_heads", "num_kv_heads", "override_only",
        }),
    ),
}

_WORKLOAD_COLLECTION_FIELDS = {
    "gemm": "workload_configs",
    "fused_moe": "workload_configs",
    "flash_attn": "workload_configs",
    "vector": "op_groups",
}

_WORKLOAD_OPTIONAL_FIELDS = {
    "gemm": frozenset(),
    "fused_moe": frozenset(),
    "flash_attn": frozenset({"layer"}),
    "vector": frozenset(),
}

_PIPELINE_OPERATOR_NAMES = frozenset({
    "eager_attn", "flash_attn", "fused_moe", "unfused_moe", "gemm",
    "vector",
})


def require_mapping(value, context):
    if not isinstance(value, dict):
        raise TypeError(
            f"{context} must be a mapping, got {type(value).__name__}"
        )
    return value


def require_exact_fields(mapping, required_fields, context, optional_fields=()):
    require_mapping(mapping, context)
    required = set(required_fields)
    optional = set(optional_fields)
    actual = set(mapping)
    missing = required - actual
    unexpected = actual - required - optional
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if unexpected:
            details.append(f"unexpected={sorted(unexpected)}")
        raise ValueError(f"{context} fields are invalid: {', '.join(details)}")
    return mapping


def require_positive_int(mapping, key, context):
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{context}.{key} must be an integer, got {type(value).__name__}"
        )
    if value <= 0:
        raise ValueError(f"{context}.{key} must be positive, got {value}")
    return value


def require_non_negative_int(mapping, key, context):
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{context}.{key} must be an integer, got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(
            f"{context}.{key} must be non-negative, got {value}"
        )
    return value


def require_positive_number(mapping, key, context):
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{context}.{key} must be numeric, got {type(value).__name__}"
        )
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{context}.{key} must be finite and positive, got {value!r}"
        )
    return value


def require_non_empty_string(mapping, key, context):
    value = mapping[key]
    if not isinstance(value, str):
        raise TypeError(
            f"{context}.{key} must be a string, got {type(value).__name__}"
        )
    if not value:
        raise ValueError(f"{context}.{key} must not be empty")
    return value


def require_non_empty_list(mapping, key, context):
    value = mapping[key]
    if not isinstance(value, list):
        raise TypeError(
            f"{context}.{key} must be a list, got {type(value).__name__}"
        )
    if not value:
        raise ValueError(f"{context}.{key} must not be empty")
    return value


def to_float(value):
    if value is None:
        raise ValueError("to_float: value is None — check dataset for missing fields")
    if isinstance(value, bool):
        raise TypeError("to_float: boolean is not a numeric measurement")
    if isinstance(value, str):
        result = float(value.replace(",", "").strip())
    elif isinstance(value, (int, float)):
        result = float(value)
    else:
        raise TypeError(
            "to_float: value must be a string or numeric measurement, "
            f"got {type(value).__name__}"
        )
    if not math.isfinite(result):
        raise ValueError(f"to_float: value must be finite, got {value!r}")
    return result


def load_yaml_mapping(path):
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"configuration root must be a mapping: {path}")
    return config


def load_dataset_config(path):
    """Load one hardware's dataset contract under a closed schema."""
    config = load_yaml_mapping(path)
    require_exact_fields(
        config,
        {"schema_version", "hardware_id", "kernel", "serving"},
        "dataset config",
    )
    if config["schema_version"] != 1:
        raise ValueError(
            "dataset config schema_version must be 1, got "
            f"{config['schema_version']!r}")
    hardware_id = get_hardware_id(config)
    path_hardware_id = os.path.splitext(os.path.basename(path))[0]
    if hardware_id != path_hardware_id:
        raise ValueError(
            f"dataset config hardware_id {hardware_id!r} does not match "
            f"filename {path_hardware_id!r}")

    kernel = require_exact_fields(
        config["kernel"], {"pipeline"}, f"{hardware_id} kernel")
    pipeline = kernel["pipeline"]
    if pipeline is not None:
        if hardware_id not in _KERNEL_PIPELINE_SETTING_FIELDS:
            raise ValueError(
                f"no active kernel dataset pipeline for {hardware_id!r}")
        require_exact_fields(
            pipeline, {"workloads", "settings"},
            f"{hardware_id} kernel.pipeline")
        workloads = require_non_empty_list(
            pipeline, "workloads", f"{hardware_id} kernel.pipeline")
        if any(not isinstance(name, str) or not name for name in workloads):
            raise TypeError(
                f"{hardware_id} kernel.pipeline.workloads must contain "
                "non-empty strings")
        if len(workloads) != len(set(workloads)):
            raise ValueError(
                f"{hardware_id} kernel.pipeline.workloads contains duplicates")
        unknown = sorted(set(workloads) - _PIPELINE_OPERATOR_NAMES)
        if unknown:
            raise ValueError(
                f"{hardware_id} kernel.pipeline has unknown workloads: {unknown}")
        require_exact_fields(
            pipeline["settings"],
            _KERNEL_PIPELINE_SETTING_FIELDS[hardware_id],
            f"{hardware_id} kernel.pipeline.settings",
        )

    serving = require_exact_fields(
        config["serving"], {"suites"}, f"{hardware_id} serving")
    suites = require_mapping(
        serving["suites"], f"{hardware_id} serving.suites")
    for suite_name, suite in suites.items():
        if not isinstance(suite_name, str) or not suite_name:
            raise TypeError("serving suite names must be non-empty strings")
        require_exact_fields(
            suite, {"historical_alignment_root"},
            f"{hardware_id} serving.suites.{suite_name}")
        require_non_empty_string(
            suite, "historical_alignment_root",
            f"{hardware_id} serving.suites.{suite_name}")
    return config


def load_pipeline_config(path):
    """Load a dataset config and require an active kernel collection pipeline."""
    config = load_dataset_config(path)
    if config["kernel"]["pipeline"] is None:
        raise ValueError(
            f"{config['hardware_id']} has no active kernel dataset pipeline")
    return config


def load_workload_config(path, expected_kernel=None):
    """Load a canonical workload config and reject missing or extra fields."""
    config = load_yaml_mapping(path)
    kernel_name = require_non_empty_string(
        config, "kernel_name", "workload config")
    if kernel_name not in _WORKLOAD_SCHEMAS:
        raise ValueError(f"unsupported workload kernel_name: {kernel_name!r}")
    if expected_kernel is not None and kernel_name != expected_kernel:
        raise ValueError(
            f"workload kernel_name mismatch: expected {expected_kernel!r}, "
            f"got {kernel_name!r} in {path}")

    top_fields, workload_fields, layer_fields = _WORKLOAD_SCHEMAS[kernel_name]
    require_exact_fields(config, top_fields, f"{kernel_name} workload config")
    workload_key = _WORKLOAD_COLLECTION_FIELDS[kernel_name]
    workloads = require_non_empty_list(
        config, workload_key, f"{kernel_name} workload config")
    for index, workload in enumerate(workloads):
        optional = _WORKLOAD_OPTIONAL_FIELDS[kernel_name]
        require_exact_fields(
            workload, workload_fields,
            f"{kernel_name} {workload_key}[{index}]", optional)
    layers = require_non_empty_list(
        config, "layers", f"{kernel_name} workload config")
    for index, layer in enumerate(layers):
        require_exact_fields(
            layer, layer_fields, f"{kernel_name} layers[{index}]")
    return config


def get_hardware_config_path(config):
    hardware_id = get_hardware_id(config)
    path = os.path.join("configs", "hardware", f"{hardware_id}.yaml")
    resolved = _resolve_repo_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"hardware config does not exist: {resolved}")
    return resolved


def get_hardware_id(config):
    hardware_id = require_non_empty_string(
        config, "hardware_id", "dataset config")
    canonical_id = canonical_hardware_id(hardware_id)
    if canonical_id != hardware_id:
        raise ValueError(
            "dataset config must use a canonical hardware ID: "
            f"{hardware_id!r} resolves to {canonical_id!r}")
    return hardware_id


def get_kernel_pipeline_settings(config):
    pipeline = config["kernel"]["pipeline"]
    if pipeline is None:
        raise ValueError(
            f"{get_hardware_id(config)} has no active kernel dataset pipeline")
    return require_mapping(
        pipeline["settings"],
        f"{get_hardware_id(config)} kernel.pipeline.settings")


def _resolve_repo_path(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def resolve_project_path(path):
    """Resolve a repository-relative path; leave an absolute path unchanged."""
    return _resolve_repo_path(path)


def load_kernel_workload_config(path, expected_kernel):
    """Load one kernel workload YAML given a repository-relative path."""
    return load_workload_config(resolve_project_path(path), expected_kernel)


def get_kernel_trace_dir(config, kernel):
    """Return the profiler trace directory of one kernel family."""
    return os.path.join(get_profiler_output_dir(config), f"{kernel}_traces")


def file_sha256(path):
    """Stream one file's SHA256 in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload):
    """Digest one payload by its canonical JSON encoding.

    The three event-capture builders all bind their candidate artifacts to a
    content hash, and they must agree on the encoding, so it is stated once.
    """
    encoded = json.dumps(
        payload, separators=(",", ":"), sort_keys=True,
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_kernel_dataset_mapping(path):
    """Load one canonical dataset document, requiring a non-empty object."""
    payload = load_kernel_dataset_document(path)
    if not isinstance(payload, dict) or not payload:
        raise TypeError(f"kernel dataset must be a non-empty mapping: {path}")
    return payload


def get_profiler_output_dir(config):
    """Return the local, non-versioned profiler workspace for this hardware."""
    hardware_id = get_hardware_id(config)
    return os.path.realpath(os.path.join(
        REPO_ROOT, "profiler_output", "kernel_dataset", hardware_id))


def get_kernel_workloads(config):
    pipeline = config["kernel"]["pipeline"]
    if pipeline is None:
        raise ValueError(
            f"{get_hardware_id(config)} has no active kernel dataset pipeline")
    workloads = pipeline["workloads"]
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("kernel.pipeline.workloads must be a non-empty list")
    return tuple(workloads)


def _require_enabled_kernel(config, kernel):
    if kernel not in get_kernel_workloads(config):
        raise ValueError(
            f"kernel {kernel!r} is not enabled for {get_hardware_id(config)}; "
            f"expected one of {sorted(get_kernel_workloads(config))}")




def get_kernel_config_path(config, kernel):
    _require_enabled_kernel(config, kernel)
    return get_workload_config_path(kernel)


def get_workload_config_path(workload):
    if workload not in _WORKLOAD_SCHEMAS:
        raise ValueError(f"unknown workload config: {workload!r}")
    path = _resolve_repo_path(os.path.join(
        "configs", "dataset", "kernel", f"{workload}.yaml"))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"kernel workload config does not exist: {path}")
    return path


def get_kernel_script_path(config, kernel):
    _require_enabled_kernel(config, kernel)
    path = _resolve_repo_path(os.path.join(
        "scripts", "kernel_dataset", get_hardware_id(config),
        f"{kernel}_runner.py"))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"kernel runner does not exist: {path}")
    return path


def get_kernel_rep_dir(config, kernel):
    settings = get_kernel_pipeline_settings(config)
    return os.path.join(
        get_profiler_output_dir(config),
        settings["ncu_rep_output_dir"],
        kernel,
    )


def get_kernel_csv_dir(config, kernel):
    settings = get_kernel_pipeline_settings(config)
    return os.path.join(
        get_profiler_output_dir(config),
        settings["ncu_csv_output_dir"],
        kernel,
    )


def get_kernel_dataset_path(config, kernel):
    hardware_id = get_hardware_id(config)
    return os.path.join(
        REPO_ROOT, "kernel_dataset", hardware_id,
        f"{kernel}{KERNEL_DATASET_SUFFIX}")


def require_profiler_executable(command, environment_variable):
    """Resolve a profiler from an explicit environment override or PATH."""
    if environment_variable in os.environ:
        configured = os.environ[environment_variable]
        if not configured:
            raise ValueError(f"{environment_variable} must not be empty")
        resolved = shutil.which(configured)
        if resolved is None:
            raise FileNotFoundError(
                f"{environment_variable} executable not found: {configured}")
        return resolved
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(
            f"required profiler executable {command!r} is not on PATH; "
            f"set {environment_variable} to an explicit executable")
    return resolved


def require_environment_directory(environment_variable):
    if environment_variable not in os.environ:
        raise EnvironmentError(
            f"required environment variable is not set: {environment_variable}")
    configured = os.environ[environment_variable]
    if not configured:
        raise ValueError(f"{environment_variable} must not be empty")
    resolved = os.path.realpath(configured)
    if not os.path.isdir(resolved):
        raise NotADirectoryError(
            f"{environment_variable} is not a directory: {resolved}")
    return resolved


def current_python_executable():
    if not sys.executable:
        raise RuntimeError("current Python interpreter path is unavailable")
    if not os.path.isfile(sys.executable):
        raise FileNotFoundError(
            f"current Python interpreter does not exist: {sys.executable}")
    return sys.executable


def parse_kernel_list(config, raw_kernels):
    """Parse the shared comma-separated kernel CLI contract without omissions."""

    supported = get_kernel_workloads(config)
    if not isinstance(raw_kernels, str):
        raise TypeError(
            f"--kernel must be a string, got {type(raw_kernels).__name__}"
        )
    kernels = []
    for raw_token in raw_kernels.split(","):
        kernel = raw_token.strip()
        if not kernel:
            raise ValueError("--kernel contains an empty kernel name")
        if kernel not in supported:
            raise ValueError(
                f"unsupported kernel {kernel!r}; expected one of "
                f"{sorted(supported)}")
        if kernel in kernels:
            raise ValueError(f"--kernel contains duplicate kernel {kernel!r}")
        kernels.append(kernel)
    if not kernels:
        raise ValueError("--kernel must provide at least one kernel")
    return kernels


def clean_dir(directory):
    if os.path.exists(directory):
        print(f"[Info] Cleaning output directory: {directory}")
        shutil.rmtree(directory)
    os.makedirs(directory)
    print(f"[Info] Created directory: {directory}")
