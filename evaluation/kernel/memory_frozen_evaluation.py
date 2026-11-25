"""Read retained diagnostics and compare two explicit memory exports.

Historical source-isolated campaign execution remains reproducible from its
original revision; it is not imported into the current predictor process.
"""
import gzip
import io
import json
from pathlib import Path

from src.kernel.memory_access_codec import load_jsonl
from evaluation.reporting.memory_tiles import program_diagnostics


def read_snapshot_json(path: Path):
    """Read an unambiguous plain, gzip or zstd diagnostic snapshot."""
    path = Path(path)
    candidates = [candidate for candidate in (path, path.with_suffix(path.suffix + ".gz"),
        path.with_suffix(path.suffix + ".zst")) if candidate.is_file()]
    if len(candidates) != 1:
        raise ValueError("snapshot requires exactly one plain/compressed representation")
    selected, = candidates
    if selected == path:
        return json.loads(selected.read_text())
    if selected.suffix == ".gz":
        with gzip.open(selected, "rt") as stream:
            return json.load(stream)
    import zstandard
    with selected.open("rb") as stream:
        with zstandard.ZstdDecompressor(max_window_size=2**27).stream_reader(stream) as decoded:
            return json.load(io.TextIOWrapper(decoded, encoding="utf-8"))


def compare_exports(before: Path, after: Path):
    """Compare current typed programs; do not relabel different calls as pairs."""
    with Path(before).open() as stream:
        left = load_jsonl(stream)
    with Path(after).open() as stream:
        right = load_jsonl(stream)
    if (left.program_id, left.binding_id, left.hardware_id) != (right.program_id, right.binding_id, right.hardware_id):
        raise ValueError("comparison requires the same program, binding and hardware identity")
    left_stats, right_stats = program_diagnostics(left), program_diagnostics(right)
    fields = ("requested_bytes", "candidate_requested_bytes", "unique_footprint_bytes")
    return {"program_id": left.program_id, "program_equal": left == right,
        "before": left_stats, "after": right_stats,
        "deltas": {name: None if left_stats[name] is None or right_stats[name] is None
                   else right_stats[name] - left_stats[name] for name in fields},
        "timing_or_accuracy_claim": False}
