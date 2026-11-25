"""Read canonical single-kernel datasets by hardware identity."""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path

import yaml

from src.kernel.hardware_ids import canonical_hardware_id
from src.kernel.kernel_spec import L2ExecutionContext
from src.kernel.dataset_storage import (
    KERNEL_DATASET_SUFFIX,
    KERNEL_EVALUATION_TIMING_SOURCE,
    load_kernel_dataset_document,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


_OP_EVALUATION_TIMING_SOURCE = {
    ("nvidia_a800", "NORM"): "device_event_op_median",
    ("nvidia_a800", "ROTARY"): "device_event_op_median",
    ("nvidia_a800", "ACTIVATION"): "device_event_op_median",
    ("nvidia_a800", "ELEMENTWISE"): "device_event_op_median",
    ("nvidia_a800", "KV_CACHE"): "device_event_op_median",
    ("metax_c500", "GEMM"): "device_event_op_median",
    ("metax_c500", "FLASHATTN"): "device_event_op_median",
    ("metax_c500", "FUSEDMOE"): "device_event_op_median",
    ("metax_c500", "NORM"): "device_event_op_median",
    ("metax_c500", "ROTARY"): "device_event_op_median",
    ("metax_c500", "ACTIVATION"): "device_event_op_median",
    ("metax_c500", "ELEMENTWISE"): "device_event_op_median",
    ("metax_c500", "KV_CACHE"): "device_event_op_median",
    ("ascend_910b", "GEMM"): "msprof_timeline_op_median",
    ("ascend_910b", "FLASHATTN"): "msprof_timeline_op_median",
    ("ascend_910b", "FUSEDMOE"): "msprof_timeline_op_median",
    ("ascend_910b", "NORM"): "msprof_timeline_op_median",
    ("ascend_910b", "ROTARY"): "msprof_timeline_op_median",
    ("ascend_910b", "ACTIVATION"): "msprof_timeline_op_median",
    ("ascend_910b", "ELEMENTWISE"): "msprof_timeline_op_median",
    ("ascend_910b", "KV_CACHE"): "msprof_timeline_op_median",
}

_OP_TIMING_COMMON_PROVENANCE_FIELDS = {
    "baseline_commit",
    "measurement_mode",
    "profiler",
    "runner_sha256",
    "scope",
    "source_artifact_sha256",
    "source_artifact_tree_sha256",
    "source_snapshot_sha256",
    "statistic",
}

_OP_EVENT_TIMING_PROVENANCE_FIELDS = _OP_TIMING_COMMON_PROVENANCE_FIELDS | {
    "timer_api",
    "timer_implementation",
    "timer_sanity_sha256",
}

_ASCEND_910B_GEMM_TIMELINE_RUNNER = (
    "scripts/kernel_dataset/ascend_910b/gemm_runner.py")
_ASCEND_910B_GEMM_TIMELINE_SAME_OPERAND_REPLAYS = 5


@dataclass(frozen=True)
class KernelEvaluationTiming:
    """Validated scoring targets, separate from raw profiler evidence."""

    granularity: str
    source: str
    total_duration_ns: float
    durations_ns: tuple[float, ...]


def kernel_dataset_l2_execution_context(
        op_info, hardware_id) -> L2ExecutionContext:
    """Return explicit L2 history guaranteed by a canonical capture runner.

    This is a workload-execution fact, not a duration-derived predictor
    parameter.  The 910B GEMM Timeline runner creates ``x`` and ``w`` once,
    executes the exact ``w(x)`` program five times, then measures a sixth
    call.  The provenance digest must bind the canonical row to that exact
    source before forwarding this physical context.  Other captures provide
    no equivalent context and therefore remain empty rather than guessing.
    """

    canonical_id = require_canonical_hardware_id(hardware_id)
    if canonical_id != "ascend_910b" or op_info["Type"] != "GEMM":
        return L2ExecutionContext()
    timing = op_info["evaluation"]["timing"]
    if timing["granularity"] != "op":
        raise ValueError(
            "910B GEMM execution context requires op-level Timeline timing")
    if timing["source"] != "msprof_timeline_op_median":
        raise ValueError(
            "910B GEMM execution context requires Timeline op timing")
    provenance = timing["provenance"]
    runner_path = Path(REPO_ROOT, _ASCEND_910B_GEMM_TIMELINE_RUNNER)
    if not runner_path.is_file():
        raise FileNotFoundError(
            "910B GEMM Timeline runner source is missing: "
            f"{runner_path}")
    runner_sha256 = hashlib.sha256(runner_path.read_bytes()).hexdigest()
    if provenance["runner_sha256"] != runner_sha256:
        raise ValueError(
            "910B GEMM Timeline runner provenance does not match the "
            "verified warmup source")
    runner_source = runner_path.read_text(encoding="utf-8")
    if runner_source.count("for _ in range(5):") != 1:
        raise ValueError(
            "910B GEMM Timeline runner no longer has exactly five explicit "
            "same-operand warmup launches")
    return L2ExecutionContext(
        same_program_operand_replays=(
            _ASCEND_910B_GEMM_TIMELINE_SAME_OPERAND_REPLAYS))


def _require_lower_hex(value, length, context):
    if (not isinstance(value, str) or len(value) != length
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(
            f"{context} must be a lowercase {length}-digit hexadecimal string")
    return value


# The device-Event OP-timing record each hardware's runner is expected to
# write.  The hardware lookup and, for MetaX, the per-op-family split are the
# same lookup the hand-written chain performed; the declared rows only stop
# the six-field record from being restated per branch.
_OP_EVENT_TIMING_EXPECTED = {
    "nvidia_a800": {
        "measurement_mode": "paired_differential_events_around_cuda_graph_replay",
        "profiler": "none",
        "scope": "logical_public_op",
        "statistic": "median_ns",
        "timer_api": "torch.cuda.Event",
        "timer_implementation": "CUDA_Runtime_Event",
    },
    "metax_c500": {
        "NORM": "paired_differential_events_around_cuda_graph_replay",
        "ROTARY": "paired_differential_events_around_cuda_graph_replay",
        "ACTIVATION": "paired_differential_events_around_cuda_graph_replay",
        "ELEMENTWISE": "paired_differential_events_around_cuda_graph_replay",
        "KV_CACHE": "paired_differential_events_around_cuda_graph_replay",
        "GEMM": "external_events_around_cuda_graph_replay",
        "FLASHATTN": "external_events_around_cuda_graph_replay",
        "FUSEDMOE": "external_events_around_cuda_graph_replay",
    },
    "ascend_910b": {
        "measurement_mode": "paired_differential_events_around_npu_graph_replay",
        "profiler": "none",
        "scope": "logical_public_op",
        "statistic": "median_ns",
        "timer_api": "torch.npu.Event",
        "timer_implementation": "ACL_Runtime_Event_via_torch_npu",
    },
}

def _expected_device_event_op_timing(hardware_id: str, op_type: str) -> dict:
    """Return the declared device-Event OP-timing record for one hardware/op.

    A row keyed by ``measurement_mode`` is the whole record for a hardware
    whose runner writes one shape.  MetaX's row is instead keyed by op type
    and its entries are the measurement mode; both MetaX families share the
    remaining fields.
    """

    if hardware_id not in _OP_EVENT_TIMING_EXPECTED:
        raise ValueError(
            "device Event OP timing is unsupported for "
            f"hardware={hardware_id!r}")
    row = _OP_EVENT_TIMING_EXPECTED[hardware_id]
    if "measurement_mode" in row:
        return dict(row)
    if op_type not in row:
        raise ValueError(
            "device Event OP timing is unsupported for "
            f"hardware={hardware_id!r}")
    return {
        "measurement_mode": row[op_type],
        "profiler": "none",
        "scope": "end_to_end_case",
        "statistic": "median_ns",
        "timer_api": "torch.cuda.Event",
        "timer_implementation": "wcudaEvent_to_maca_mcEvent",
    }


def _validate_op_timing_provenance(
        provenance, hardware_id, op_type, source):
    if not isinstance(provenance, dict):
        raise TypeError("entry evaluation op timing provenance must be a mapping")
    if source == "device_event_op_median":
        expected_fields = _OP_EVENT_TIMING_PROVENANCE_FIELDS
    elif source == "msprof_timeline_op_median":
        expected_fields = _OP_TIMING_COMMON_PROVENANCE_FIELDS
    else:
        raise ValueError(
            f"unsupported op timing provenance source: {source!r}")
    if set(provenance) != expected_fields:
        raise ValueError(
            "entry evaluation op timing provenance fields are invalid: "
            f"expected {sorted(expected_fields)}, "
            f"got={sorted(provenance)}")
    _require_lower_hex(
        provenance["baseline_commit"], 40,
        "entry evaluation op timing baseline_commit")
    for name in (
            "runner_sha256", "source_artifact_sha256",
            "source_artifact_tree_sha256", "source_snapshot_sha256"):
        _require_lower_hex(
            provenance[name], 64,
            f"entry evaluation op timing {name}")
    if source == "device_event_op_median":
        _require_lower_hex(
            provenance["timer_sanity_sha256"], 64,
            "entry evaluation op timing timer_sanity_sha256")
        expected = _expected_device_event_op_timing(hardware_id, op_type)
    elif source == "msprof_timeline_op_median":
        if hardware_id != "ascend_910b":
            raise ValueError(
                "msprof Timeline OP timing is supported only for ascend_910b")
        expected = {
            "measurement_mode": "msprof_timeline_task_start_and_duration",
            "profiler": "msprof",
            "scope": "logical_public_op",
            "statistic": "median_of_per_call_op_elapsed_duration",
        }
    else:
        raise AssertionError(
            f"validated OP timing source was not dispatched: {source!r}")
    for name, expected_value in expected.items():
        if provenance[name] != expected_value:
            raise ValueError(
                f"entry evaluation op timing {name} must be "
                f"{expected_value!r}, got {provenance[name]!r}")


def parse_kernel_evaluation_timing(entry, kernels, hardware_id):
    """Validate the explicit scoring target for one dataset case."""

    canonical_id = require_canonical_hardware_id(hardware_id)
    if canonical_id not in KERNEL_EVALUATION_TIMING_SOURCE:
        raise ValueError(
            f"hardware has no kernel evaluation timing contract: "
            f"{canonical_id!r}")
    if not isinstance(entry, dict):
        raise TypeError("dataset entry must be a mapping")
    if not isinstance(kernels, list) or not kernels:
        raise TypeError("kernel evaluation timing requires a non-empty kernel list")

    if "evaluation" in entry:
        raise ValueError(
            "dataset entry must not contain legacy top-level evaluation")
    op_info = entry["op_info"]
    if not isinstance(op_info, dict):
        raise TypeError("entry op_info must be a mapping")
    if "Overall Duration [nsecond]" in op_info:
        raise ValueError(
            "dataset entry must not contain legacy op_info Overall Duration")
    if "evaluation" not in op_info:
        raise ValueError(
            "dataset entry is missing required op_info.evaluation.timing")
    evaluation = op_info["evaluation"]
    if not isinstance(evaluation, dict):
        raise TypeError("entry evaluation must be a mapping")
    if set(evaluation) != {"timing"}:
        raise ValueError(
            "entry evaluation fields are invalid: expected ['timing'], "
            f"got={sorted(evaluation)}")
    timing = evaluation["timing"]
    if not isinstance(timing, dict):
        raise TypeError("entry evaluation.timing must be a mapping")
    if "granularity" not in timing:
        raise ValueError("entry evaluation.timing is missing granularity")
    granularity = timing["granularity"]
    if granularity == "kernel":
        expected_fields = {
            "granularity", "source", "total_duration_ns", "kernels"}
    elif granularity == "op":
        expected_fields = {
            "granularity", "source", "total_duration_ns", "provenance"}
    else:
        raise ValueError(
            "entry evaluation.timing.granularity must be 'kernel' or 'op'")
    if set(timing) != expected_fields:
        raise ValueError(
            "entry evaluation.timing fields are invalid: expected "
            f"{sorted(expected_fields)}, got={sorted(timing)}")
    if granularity == "kernel":
        expected_source = KERNEL_EVALUATION_TIMING_SOURCE[canonical_id]
    elif granularity == "op":
        op_type = op_info["Type"]
        source_key = (canonical_id, op_type)
        if source_key not in _OP_EVALUATION_TIMING_SOURCE:
            raise ValueError(
                "op-granularity evaluation timing is unsupported for "
                f"hardware={canonical_id!r}, op_type={op_type!r}")
        expected_source = _OP_EVALUATION_TIMING_SOURCE[source_key]
    else:
        raise AssertionError(
            f"validated timing granularity was not dispatched: {granularity!r}")
    if timing["source"] != expected_source:
        raise ValueError(
            f"entry evaluation.timing.source for {canonical_id} must be "
            f"{expected_source!r} for {granularity} granularity, "
            f"got {timing['source']!r}")

    total_duration_ns = timing["total_duration_ns"]
    if (isinstance(total_duration_ns, bool)
            or not isinstance(total_duration_ns, (int, float))):
        raise TypeError(
            "entry evaluation.timing.total_duration_ns must be numeric")
    normalized_total = float(total_duration_ns)
    if not math.isfinite(normalized_total) or normalized_total <= 0:
        raise ValueError(
            "entry evaluation.timing.total_duration_ns must be finite and "
            "positive")

    if granularity == "op":
        _validate_op_timing_provenance(
            timing["provenance"], canonical_id, op_info["Type"],
            expected_source)
        return KernelEvaluationTiming(
            granularity=granularity,
            source=expected_source,
            total_duration_ns=normalized_total,
            durations_ns=(normalized_total,) if len(kernels) == 1 else (),
        )

    targets = timing["kernels"]
    if not isinstance(targets, list) or not targets:
        raise TypeError(
            "entry evaluation.timing.kernels must be a non-empty list")
    if len(targets) != len(kernels):
        raise ValueError(
            "entry evaluation timing count does not match profiler launches: "
            f"timing={len(targets)}, profiler={len(kernels)}")

    durations = []
    target_fields = {"kernel_idx", "kernel_name", "duration_ns"}
    for expected_idx, (target, kernel) in enumerate(
            zip(targets, kernels, strict=True)):
        if not isinstance(target, dict):
            raise TypeError(
                f"entry evaluation timing kernel {expected_idx} must be a mapping")
        if set(target) != target_fields:
            raise ValueError(
                f"entry evaluation timing kernel {expected_idx} fields are "
                f"invalid: expected {sorted(target_fields)}, "
                f"got={sorted(target)}")
        kernel_idx = target["kernel_idx"]
        if isinstance(kernel_idx, bool) or not isinstance(kernel_idx, int):
            raise TypeError(
                f"entry evaluation timing kernel_idx {expected_idx} must be "
                "an integer")
        if kernel_idx != expected_idx:
            raise ValueError(
                "entry evaluation timing kernel indices must be contiguous and "
                f"ordered: expected {expected_idx}, got {kernel_idx}")
        if not isinstance(kernel, dict):
            raise TypeError(f"profiler kernel {expected_idx} must be a mapping")
        metadata = kernel["Metadata"]
        if not isinstance(metadata, dict):
            raise TypeError(
                f"profiler kernel {expected_idx} Metadata must be a mapping")
        profiler_name = metadata["Kernel Name"]
        target_name = target["kernel_name"]
        if not isinstance(target_name, str) or not target_name:
            raise TypeError(
                f"entry evaluation timing kernel name {expected_idx} must be "
                "non-empty")
        if target_name != profiler_name:
            raise ValueError(
                f"entry evaluation timing kernel name mismatch at index "
                f"{expected_idx}: target={target_name!r}, "
                f"profiler={profiler_name!r}")
        duration_ns = target["duration_ns"]
        if (isinstance(duration_ns, bool)
                or not isinstance(duration_ns, (int, float))):
            raise TypeError(
                f"entry evaluation timing duration {expected_idx} must be numeric")
        normalized_duration = float(duration_ns)
        if not math.isfinite(normalized_duration) or normalized_duration <= 0:
            raise ValueError(
                f"entry evaluation timing duration {expected_idx} must be "
                "finite and positive")
        durations.append(normalized_duration)

    summed_duration = math.fsum(durations)
    if normalized_total != summed_duration:
        raise ValueError(
            "entry evaluation timing total does not equal the sum of kernel "
            f"durations: total={normalized_total}, kernels={summed_duration}")

    return KernelEvaluationTiming(
        granularity=granularity,
        source=expected_source,
        total_duration_ns=normalized_total,
        durations_ns=tuple(durations),
    )


def to_float(value):
    """Convert one required finite profiler measurement to float."""
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
            f"got {type(value).__name__}")
    if not math.isfinite(result):
        raise ValueError(f"to_float: value must be finite, got {value!r}")
    return result


def require_canonical_hardware_id(hardware_id):
    canonical_id = canonical_hardware_id(hardware_id)
    if canonical_id != hardware_id:
        raise ValueError(
            f"hardware_id must be canonical: {hardware_id!r} resolves to "
            f"{canonical_id!r}")
    return canonical_id


def get_hardware_config_path(hardware_id):
    canonical_id = require_canonical_hardware_id(hardware_id)
    path = os.path.join(
        REPO_ROOT, "configs", "hardware", f"{canonical_id}.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"hardware config does not exist: {path}")
    return path


def get_kernel_dataset_path(hardware_id, kernel):
    canonical_id = require_canonical_hardware_id(hardware_id)
    if not isinstance(kernel, str) or not kernel:
        raise TypeError("kernel dataset name must be a non-empty string")
    return os.path.join(
        REPO_ROOT, "kernel_dataset", canonical_id,
        f"{kernel}{KERNEL_DATASET_SUFFIX}")


def get_kernel_dataset_dir(hardware_id):
    canonical_id = require_canonical_hardware_id(hardware_id)
    return os.path.join(REPO_ROOT, "kernel_dataset", canonical_id)


def load_kernel_dataset(hardware_id, kernels=None):
    """Load and merge a hardware's canonical per-kernel JSON payloads."""
    dataset_dir = get_kernel_dataset_dir(hardware_id)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"kernel dataset directory does not exist: {dataset_dir}")
    if kernels is None:
        kernels = [
            name[:-len(KERNEL_DATASET_SUFFIX)]
            for name in sorted(os.listdir(dataset_dir))
            if name.endswith(KERNEL_DATASET_SUFFIX)
        ]
    elif not isinstance(kernels, (list, tuple)):
        raise TypeError("kernels must be a list or tuple when provided")
    if not kernels:
        raise ValueError(f"no kernel dataset JSON files found in {dataset_dir}")
    if any(not isinstance(kernel, str) or not kernel for kernel in kernels):
        raise TypeError("kernel dataset names must be non-empty strings")
    if len(kernels) != len(set(kernels)):
        raise ValueError("kernel dataset names contain duplicates")

    merged = {}
    for kernel in kernels:
        path = get_kernel_dataset_path(hardware_id, kernel)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"kernel dataset file does not exist: {path}")
        payload = load_kernel_dataset_document(path)
        if not isinstance(payload, dict):
            raise TypeError(
                f"kernel dataset file must contain a mapping: {path}")
        duplicates = set(merged).intersection(payload)
        if duplicates:
            raise ValueError(
                f"duplicate kernel dataset case IDs in {path}: "
                f"{sorted(duplicates)}")
        merged.update(payload)
    return merged


# The kernel op types the canonical dataset groups into one evaluation
# family.  Reporters and the dataset contract share these groupings, so they
# are declared once instead of being restated as a literal set per module.
VECTOR_OP_TYPES = frozenset({
    "NORM", "ROTARY", "ACTIVATION", "KV_CACHE", "ELEMENTWISE",
})
MOE_OP_TYPES = frozenset({"FUSEDMOE", "UNFUSEDMOE"})
ATTENTION_OP_TYPES = frozenset({"FLASHATTN", "EAGERATTN"})


KERNEL_CANDIDATE_ROOTS_CONFIG = os.path.join(
    REPO_ROOT, "configs", "evaluation", "kernel_candidate_roots.yaml")
# Candidate manifests each hardware's dataset assembly writes; only these
# artifact kinds may stand in for a canonical kernel dataset in evaluation.
_CANDIDATE_ARTIFACT_KINDS = {
    "nvidia_h200": "nvidia_h200_kcampaign_kernel_dataset_candidate",
}


def _kernel_candidate_roots():
    with open(KERNEL_CANDIDATE_ROOTS_CONFIG) as stream:
        config = yaml.safe_load(stream)
    if (not isinstance(config, dict)
            or set(config) != {"schema_version", "candidate_roots"}
            or config["schema_version"] != 1
            or not isinstance(config["candidate_roots"], dict)):
        raise ValueError(
            f"{KERNEL_CANDIDATE_ROOTS_CONFIG} must hold schema_version 1 and "
            "a candidate_roots mapping")
    roots = {}
    canonical_root = os.path.join(REPO_ROOT, "kernel_dataset")
    for hardware_id, relative in config["candidate_roots"].items():
        require_canonical_hardware_id(hardware_id)
        if hardware_id not in _CANDIDATE_ARTIFACT_KINDS:
            raise ValueError(
                f"{hardware_id} has no kernel candidate artifact contract")
        if (not isinstance(relative, str) or not relative
                or os.path.isabs(relative)):
            raise ValueError(
                f"{hardware_id} candidate root must be repository-relative")
        path = os.path.normpath(os.path.join(REPO_ROOT, relative))
        if os.path.commonpath((path, REPO_ROOT)) != REPO_ROOT:
            raise ValueError(f"{hardware_id} candidate root leaves the repo")
        if os.path.commonpath((path, canonical_root)) == canonical_root:
            raise ValueError(
                f"{hardware_id} candidate root must not be canonical "
                "kernel_dataset/")
        roots[hardware_id] = path
    return roots


def kernel_evaluation_dataset_location(hardware_id):
    """Return ``(kind, directory)`` kernel evaluation reads for a hardware.

    ``kind`` is ``"candidate"`` for a hardware configured in
    ``configs/evaluation/kernel_candidate_roots.yaml`` and ``"canonical"``
    (``kernel_dataset/<hardware_id>/``) otherwise.
    """

    canonical_id = require_canonical_hardware_id(hardware_id)
    roots = _kernel_candidate_roots()
    if canonical_id in roots:
        return "candidate", roots[canonical_id]
    return "canonical", get_kernel_dataset_dir(canonical_id)


def kernel_evaluation_dataset_available(hardware_id):
    kind, directory = kernel_evaluation_dataset_location(hardware_id)
    if kind == "canonical":
        return (os.path.isdir(directory) and any(
            name.endswith(KERNEL_DATASET_SUFFIX)
            for name in os.listdir(directory)))
    return os.path.isdir(directory) and bool(
        _candidate_manifest_paths(directory))


def _candidate_manifest_paths(root):
    paths = []
    for plan in sorted(os.listdir(root)):
        plan_dir = os.path.join(root, plan)
        if not os.path.isdir(plan_dir):
            continue
        for name in sorted(os.listdir(plan_dir)):
            if name.endswith(".manifest.json"):
                paths.append(os.path.join(plan_dir, name))
    return paths


def load_kernel_candidate_root(hardware_id, root):
    """Merge every evaluation-supported candidate under ``root``.

    Case keys are ``<campaign>/<case_id>`` so a shape measured by two plans
    stays two independent observations.  Families whose manifest declares
    ``evaluation_supported: false`` are not evaluation inputs and are
    skipped by that declaration; a candidate root mixing software stacks is
    rejected.
    """

    canonical_id = require_canonical_hardware_id(hardware_id)
    if canonical_id not in _CANDIDATE_ARTIFACT_KINDS:
        raise ValueError(f"{canonical_id} has no kernel candidate contract")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"kernel candidate root does not exist: {root}")
    manifests = _candidate_manifest_paths(root)
    if not manifests:
        raise ValueError(f"no candidate manifests under {root}")
    merged = {}
    stacks = set()
    for manifest_path in manifests:
        with open(manifest_path) as stream:
            manifest = json.load(stream)
        family = os.path.basename(manifest_path)[:-len(".manifest.json")]
        campaign = os.path.basename(os.path.dirname(manifest_path))
        if (manifest["artifact_kind"] != _CANDIDATE_ARTIFACT_KINDS[canonical_id]
                or manifest["hardware_id"] != canonical_id
                or manifest["family"] != family
                or manifest["campaign"] != campaign):
            raise ValueError(
                f"{manifest_path} is not a {canonical_id} candidate manifest "
                "for its own plan/family")
        if manifest["evaluation_supported"] is not True:
            continue
        payload_path = os.path.join(
            os.path.dirname(manifest_path), f"{family}{KERNEL_DATASET_SUFFIX}")
        if manifest["candidate_path"] is None:
            if os.path.exists(payload_path):
                raise ValueError(
                    f"{payload_path} exists but its manifest declares no "
                    "measured case")
            continue
        if os.path.basename(manifest["candidate_path"]) != os.path.basename(
                payload_path):
            raise ValueError(f"{manifest_path} names another candidate file")
        payload = load_kernel_dataset_document(payload_path)
        if len(payload) != manifest["measured_case_count"]:
            raise ValueError(
                f"{payload_path} case count disagrees with its manifest")
        stacks.add(manifest["vllm_stack"])
        for case_id, entry in payload.items():
            key = f"{campaign}/{case_id}"
            if key in merged:
                raise ValueError(f"duplicate candidate case key {key}")
            merged[key] = entry
    if len(stacks) > 1:
        raise ValueError(
            f"kernel candidate root {root} mixes software stacks "
            f"{sorted(stacks)}")
    if not merged:
        raise ValueError(f"no evaluation-supported candidate under {root}")
    return merged


def load_kernel_evaluation_dataset(hardware_id):
    """Load what kernel evaluation scores for one hardware."""

    kind, directory = kernel_evaluation_dataset_location(hardware_id)
    if kind == "candidate":
        return load_kernel_candidate_root(hardware_id, directory)
    return load_kernel_dataset(hardware_id)
