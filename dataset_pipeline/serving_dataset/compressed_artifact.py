"""Deterministic zstd storage for canonical Serving payloads."""

from __future__ import annotations

import hashlib
from pathlib import Path

import zstandard


COMPRESSION_ALGORITHM = "zstd"
COMPRESSION_LEVEL = 1


def compress_canonical_bytes(content: bytes) -> bytes:
    """Compress canonical bytes with the repository's fixed zstd contract."""

    if not isinstance(content, bytes):
        raise TypeError("canonical compressed content must be bytes")
    return zstandard.ZstdCompressor(level=COMPRESSION_LEVEL).compress(content)


def write_compressed_canonical_bytes(path: str | Path, content: bytes) -> bytes:
    """Write one deterministic zstd frame and return its physical bytes."""

    compressed = compress_canonical_bytes(content)
    Path(path).write_bytes(compressed)
    return compressed


def read_compressed_canonical_bytes(
    path: str | Path,
    *,
    compressed_sha256: object,
    content_sha256: object,
    label: str,
) -> bytes:
    """Read one zstd artifact and verify both physical and logical identity."""

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"required {label} file is absent: {source}")
    compressed = source.read_bytes()
    actual_compressed_sha256 = hashlib.sha256(compressed).hexdigest()
    if actual_compressed_sha256 != compressed_sha256:
        raise ValueError(f"{label} compressed SHA256 is inconsistent")
    try:
        content = zstandard.ZstdDecompressor().decompress(compressed)
    except zstandard.ZstdError as error:
        raise ValueError(f"{label} is not a valid zstd frame: {error}") from error
    actual_content_sha256 = hashlib.sha256(content).hexdigest()
    if actual_content_sha256 != content_sha256:
        raise ValueError(f"{label} content SHA256 is inconsistent")
    return content
