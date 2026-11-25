"""Storage contract for canonical single-kernel datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import zstandard


KERNEL_DATASET_SUFFIX = ".json.zst"
KERNEL_DATASET_COMPRESSION_LEVEL = 3

# The profiler the canonical dataset's scoring timing comes from, per hardware.
# The dataset writer stamps it into `evaluation.timing.source` and the
# evaluator validates against it, so it is declared once, with the storage
# contract both read.
KERNEL_EVALUATION_TIMING_SOURCE = {
    "nvidia_a800": "nsys_nvtx_kernel_median",
    "nvidia_h200": "nsys_nvtx_kernel_median",
    "metax_c500": "mcTracer",
    "ascend_910b": "msprof_task_duration",
    "samsung_hbm_pim": "unindp_simulator_latency",
    "groq_lpu": "timeloop_lpu_scaled_hybrid_latency",
    "ucb_gemmini": "timeloop_mapper_simulated_latency",
}


def compress_kernel_dataset_content(content: bytes) -> bytes:
    """Return one deterministic zstd frame for canonical JSON bytes."""

    if not isinstance(content, bytes):
        raise TypeError("kernel dataset content must be bytes")
    return zstandard.ZstdCompressor(
        level=KERNEL_DATASET_COMPRESSION_LEVEL
    ).compress(content)


def read_kernel_dataset_content(path: str | Path) -> bytes:
    """Read and decompress one canonical kernel dataset artifact."""

    source = Path(path)
    if not source.name.endswith(KERNEL_DATASET_SUFFIX):
        raise ValueError(
            "canonical kernel dataset path must end with "
            f"{KERNEL_DATASET_SUFFIX!r}: {source}"
        )
    compressed = source.read_bytes()
    try:
        return zstandard.ZstdDecompressor().decompress(compressed)
    except zstandard.ZstdError as error:
        raise ValueError(
            f"canonical kernel dataset is not a valid zstd frame: {source}"
        ) from error


def load_kernel_dataset_document(path: str | Path) -> Any:
    """Decode one canonical compressed JSON document without fallback."""

    source = Path(path)
    content = read_kernel_dataset_content(source)
    try:
        return json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"canonical kernel dataset content is not valid JSON: {source}"
        ) from error


def kernel_dataset_content_sha256(path: str | Path) -> str:
    """Hash canonical JSON content independently of its storage encoding."""

    return hashlib.sha256(read_kernel_dataset_content(path)).hexdigest()
