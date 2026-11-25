"""Bounded diagnostics for the current source-owned memory programs.

No shape-specific historical acceptance or predictor formula lives here.
"""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from io import StringIO
from pathlib import Path

from src.kernel.kernel_base import build_memory_access_program
from src.kernel.memory_access import Granularity, expand_tile_program, statistics
from src.kernel.memory_access_builder import default_memory_bindings
from src.kernel.memory_access_codec import dump_jsonl, load_jsonl


def encode(value):
    """Encode typed evidence without losing unknown values."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("evidence keys must be strings")
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"unsupported evidence value: {type(value)}")


def program_diagnostics(program):
    """Validate codec identity and report source envelopes, never DRAM traffic."""
    wire = StringIO()
    dump_jsonl(program, wire)
    if load_jsonl(StringIO(wire.getvalue())) != program:
        raise ValueError("memory program codec roundtrip differs")
    rows = statistics(program)
    conditional = any(row.conditional_read_requested_bytes or row.conditional_write_requested_bytes
                      for row in rows)
    known_unique = all(row.unique_footprint_bytes is not None for row in rows)
    return {
        "program_id": program.program_id,
        "binding_id": program.binding_id,
        "hardware_id": program.hardware_id,
        "coverage": encode(program.coverage),
        "roundtrip": "pass",
        "requested_bytes": None if conditional else sum(
            row.read_requested_bytes + row.write_requested_bytes for row in rows),
        "candidate_requested_bytes": sum(
            row.read_requested_bytes + row.write_requested_bytes for row in rows),
        "unique_footprint_bytes": sum(row.unique_footprint_bytes for row in rows) if known_unique else None,
        "allocation_statistics": encode(rows),
        "tile_program_count": len(program.tile_programs),
        "explicit_access_count": len(program.accesses),
        "physical_traffic_bytes": None,
        "kernel_duration_seconds": None,
        "acceptance": "program_only_not_provider_or_performance_acceptance",
    }


def export_kernel_memory(kernel, hardware, output: Path, *, bindings=None,
                         granularity=Granularity.KERNEL, program_id="memory-export", tile=None):
    """Export any currently admitted KernelSpec through the shared public owner.

    Omitted bindings declare standalone identities. Serving must pass the real
    call bindings; unsupported source domains propagate their original refusal.
    """
    if bindings is None:
        bindings = default_memory_bindings(kernel, hardware, call_id=program_id,
            binding_id=program_id, rank=0, stream="export", device_id="standalone-export")
    program = build_memory_access_program(kernel, hardware, bindings, granularity,
                                         program_id=program_id, tile=tile)
    program_diagnostics(program)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        dump_jsonl(program, stream)
    return program


def expand_export(source: Path, output: Path, *, max_accesses=10000, max_dependencies=100000):
    """Expand only explicitly bounded, fully resolved tile programs."""
    if type(max_accesses) is not int or type(max_dependencies) is not int or min(max_accesses, max_dependencies) <= 0:
        raise ValueError("tile expansion budgets must be positive integers")
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("tile expansion cannot overwrite its input")
    with source.open() as stream:
        program = load_jsonl(stream)
    expanded = expand_tile_program(program, max_accesses=max_accesses, max_dependencies=max_dependencies)
    before, after = program_diagnostics(program), program_diagnostics(expanded)
    for name in ("candidate_requested_bytes", "unique_footprint_bytes"):
        if before[name] != after[name]:
            raise ValueError(f"tile expansion changed {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        dump_jsonl(expanded, stream)
    return after
