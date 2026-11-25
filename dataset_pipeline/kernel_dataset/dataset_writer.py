"""Completeness-gated publication of canonical kernel datasets."""

import hashlib
import json
import math
import os
import tempfile

from src.kernel.dataset_storage import (
    KERNEL_DATASET_SUFFIX,
    KERNEL_EVALUATION_TIMING_SOURCE,
    compress_kernel_dataset_content,
    read_kernel_dataset_content,
)


def build_kernel_evaluation_timing(hardware_id, kernels, durations_ns):
    """Build the only scoring-time contract for one profiled case.

    Profiler-native duration fields remain in ``kernels`` as raw evidence.
    This independent, self-identifying structure is what evaluation consumes.
    Repeating the kernel name is intentional: it turns an ordering mistake into
    a loud name mismatch instead of silently assigning a duration to the wrong
    launch.
    """

    if hardware_id not in KERNEL_EVALUATION_TIMING_SOURCE:
        raise ValueError(
            f"unsupported kernel evaluation timing hardware: {hardware_id!r}")
    if not isinstance(kernels, list) or not kernels:
        raise TypeError("kernel evaluation timing requires a non-empty kernel list")
    if not isinstance(durations_ns, (list, tuple)) or not durations_ns:
        raise TypeError(
            "kernel evaluation timing requires a non-empty duration sequence")
    if len(kernels) != len(durations_ns):
        raise ValueError(
            "kernel evaluation timing count does not match kernel records: "
            f"timing={len(durations_ns)}, kernels={len(kernels)}")

    timing_kernels = []
    for kernel_idx, (kernel, duration_ns) in enumerate(
            zip(kernels, durations_ns, strict=True)):
        if not isinstance(kernel, dict):
            raise TypeError(
                f"kernel record {kernel_idx} must be a mapping")
        metadata = kernel["Metadata"]
        if not isinstance(metadata, dict):
            raise TypeError(
                f"kernel record {kernel_idx} Metadata must be a mapping")
        kernel_name = metadata["Kernel Name"]
        if not isinstance(kernel_name, str) or not kernel_name:
            raise TypeError(
                f"kernel record {kernel_idx} Kernel Name must be non-empty")
        if (isinstance(duration_ns, bool)
                or not isinstance(duration_ns, (int, float))):
            raise TypeError(
                f"kernel duration {kernel_idx} must be numeric")
        if not math.isfinite(duration_ns) or duration_ns <= 0:
            raise ValueError(
                f"kernel duration {kernel_idx} must be finite and positive")
        timing_kernels.append({
            "kernel_idx": kernel_idx,
            "kernel_name": kernel_name,
            "duration_ns": duration_ns,
        })

    return {
        "timing": {
            "granularity": "kernel",
            "source": KERNEL_EVALUATION_TIMING_SOURCE[hardware_id],
            "total_duration_ns": math.fsum(
                float(target["duration_ns"]) for target in timing_kernels),
            "kernels": timing_kernels,
        },
    }


class DatasetCompletenessError(RuntimeError):
    """Raised when a candidate does not cover its full workload contract."""

    def __init__(self, missing, unexpected, candidate_path):
        self.missing = tuple(missing)
        self.unexpected = tuple(unexpected)
        self.candidate_path = candidate_path
        super().__init__(
            f"candidate dataset is incomplete: {len(self.missing)} missing, "
            f"{len(self.unexpected)} extra candidate entries; "
            f"saved to {candidate_path}"
        )


def _serialized_bytes(payload):
    return json.dumps(
        payload, indent=2, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _write_bytes(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as stream:
        stream.write(payload)


def publish_complete_dataset(dataset, expected_keys, candidate_path,
                             canonical_path):
    """Save a candidate and publish only after every expected key is present."""
    expected_keys = tuple(expected_keys)
    expected = set(expected_keys)
    if len(expected) != len(expected_keys):
        raise ValueError("workload contract contains duplicate dataset keys")
    candidate_keys = set(dataset)
    candidate_payload = _serialized_bytes(dataset)
    _write_bytes(candidate_path, candidate_payload)

    missing = sorted(expected - candidate_keys)
    unexpected = sorted(candidate_keys - expected)
    if missing or unexpected:
        raise DatasetCompletenessError(missing, unexpected, candidate_path)

    canonical = {key: dataset[key] for key in expected_keys}
    canonical_payload = _serialized_bytes(canonical)
    if not os.fspath(canonical_path).endswith(KERNEL_DATASET_SUFFIX):
        raise ValueError(
            "canonical kernel dataset path must end with "
            f"{KERNEL_DATASET_SUFFIX!r}: {canonical_path}"
        )
    old_hash = None
    if os.path.exists(canonical_path):
        old_hash = hashlib.sha256(
            read_kernel_dataset_content(canonical_path)
        ).hexdigest()
    new_hash = hashlib.sha256(canonical_payload).hexdigest()
    compressed_payload = compress_kernel_dataset_content(canonical_payload)

    canonical_dir = os.path.dirname(canonical_path)
    os.makedirs(canonical_dir, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=canonical_dir, prefix=".dataset-", delete=False
        ) as stream:
            temporary_path = stream.name
            stream.write(compressed_payload)
        os.replace(temporary_path, canonical_path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)

    old_hash_label = "(new)" if old_hash is None else old_hash
    print(f"  canonical SHA-256: {old_hash_label} -> {new_hash}")
    return canonical_path


def write_complete_candidate(dataset, expected_keys, candidate_path):
    """Write a complete, deterministically ordered candidate without publishing."""
    expected_keys = tuple(expected_keys)
    expected = set(expected_keys)
    if len(expected) != len(expected_keys):
        raise ValueError("workload contract contains duplicate dataset keys")

    candidate_keys = set(dataset)
    missing = sorted(expected - candidate_keys)
    unexpected = sorted(candidate_keys - expected)
    if missing or unexpected:
        _write_bytes(candidate_path, _serialized_bytes(dataset))
        raise DatasetCompletenessError(missing, unexpected, candidate_path)

    ordered = {key: dataset[key] for key in expected_keys}
    _write_bytes(candidate_path, _serialized_bytes(ordered))
    return candidate_path
