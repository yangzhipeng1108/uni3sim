"""Strict JSONL codec derived from the memory access dataclasses."""
from __future__ import annotations

from dataclasses import fields, is_dataclass, MISSING
import json
import re
from types import UnionType
from typing import get_args, get_origin, get_type_hints

from src.kernel.memory_access import (
    AccessStatistics, MemoryAccessProgram, Typed, check_type, statistics, tile_template_instances,
)


def encode(value):
    from enum import Enum
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {"type": type(value).__name__, **{
            f.name: encode(getattr(value, f.name)) for f in fields(value)}}
    if type(value) is int:
        check_type(value, int)
        return str(value)
    if type(value) is tuple:
        return [encode(item) for item in value]
    if value is None or type(value) in (str, bool):
        return value
    raise TypeError(f"unsupported codec value {type(value)}")


def decode(value, annotation):
    from enum import Enum
    origin = get_origin(annotation)
    if origin is UnionType:
        for candidate in get_args(annotation):
            try:
                return decode(value, candidate)
            except (ValueError, TypeError):
                pass
        raise ValueError(f"invalid union payload for {annotation}")
    if origin is tuple:
        if type(value) is not list:
            raise TypeError("wire sequence must be array")
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(decode(item, args[0]) for item in value)
        if len(value) != len(args):
            raise ValueError("wire tuple arity mismatch")
        return tuple(decode(item, kind) for item, kind in zip(value, args))
    if isinstance(annotation, type) and is_dataclass(annotation):
        if type(value) is not dict:
            raise TypeError("wire object required")
        schema = {field.name:field for field in fields(annotation)}
        names=set(schema)
        if (not set(value) <= names|{"type"} or "type" not in value or value["type"] != annotation.__name__
                or any(schema[name].default is MISSING for name in names-set(value))):
            raise ValueError("unknown/missing contract fields or type")
        # Only schema-declared optional/default fields may be normalized.
        # Older tile records predate optional indirect targets; required fields
        # never receive inferred values.
        hints = get_type_hints(annotation)
        return annotation(**{name:decode(value[name],hints[name]) if name in value else schema[name].default for name in names})
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if type(value) is not str:
            raise TypeError("enum must be string")
        return annotation(value)
    if annotation is int:
        if type(value) is not str or re.fullmatch(r"0|[1-9][0-9]*", value) is None:
            raise ValueError("uint64 wire value must be canonical decimal string")
        value = int(value)
    check_type(value, annotation)
    return value


# Collection types and header fields come from the production contract itself.
COLLECTIONS = ("allocations", "views", "scopes", "events", "accesses", "tile_programs")
HEADER_FIELDS = tuple(f.name for f in fields(MemoryAccessProgram) if f.name not in COLLECTIONS)


def _trailer(program, record_count):
    from src.kernel.memory_access import LogicalSelectionView,Unmapped,program_symbolic_extents
    unbound_encoding=(any(isinstance(view,LogicalSelectionView) and view.element_bytes is None for view in program.views)
        or any(isinstance(access.region,Unmapped) and access.region.element_bytes is None for access in program.accesses))
    symbolic=bool(program_symbolic_extents(program))
    return {
        "record_count": encode(record_count),
        "access_instances": None if symbolic else encode(sum(a.repeat for a in program.accesses)+sum(
            tile_template_instances(family,access) for family in program.tile_programs for access in family.accesses)),
        "statistics": None if unbound_encoding or symbolic else encode(statistics(program)),
    }


def dump_jsonl(program: MemoryAccessProgram, stream):
    if not isinstance(program, MemoryAccessProgram):
        raise TypeError("MemoryAccessProgram required")
    # Revalidate the boundary even if someone bypassed frozen initialization.
    program.__post_init__()
    header = {name: encode(getattr(program, name)) for name in HEADER_FIELDS}
    stream.write(json.dumps({"record": "header", "value": header}, sort_keys=True) + "\n")
    count = 1
    for collection in COLLECTIONS:
        for value in getattr(program, collection):
            stream.write(json.dumps({"record": collection, "value": encode(value)}, sort_keys=True) + "\n")
            count += 1
    stream.write(json.dumps({"record": "trailer", "value": _trailer(program, count + 1)}, sort_keys=True) + "\n")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def load_jsonl(stream):
    hints = get_type_hints(MemoryAccessProgram)
    values = {name: [] for name in COLLECTIONS}
    header = None
    trailer = None
    count = 0
    last_collection = -1
    for line in stream:
        count += 1
        if not line.endswith("\n"):
            raise ValueError("unterminated/truncated JSONL record")
        record = json.loads(line, object_pairs_hook=_pairs)
        if type(record) is not dict or set(record) != {"record", "value"}:
            raise ValueError("invalid JSONL record")
        kind = record["record"]
        payload = record["value"]
        if trailer is not None:
            raise ValueError("records after trailer")
        if count == 1:
            if kind != "header" or type(payload) is not dict or set(payload) != set(HEADER_FIELDS):
                raise ValueError("missing or invalid header")
            header = {name: decode(payload[name], hints[name]) for name in HEADER_FIELDS}
        elif kind == "trailer":
            trailer = payload
        elif kind in COLLECTIONS:
            position = COLLECTIONS.index(kind)
            if position < last_collection:
                raise ValueError("invalid record order")
            last_collection = position
            element_type = get_args(hints[kind])[0]
            values[kind].append(decode(payload, element_type))
        else:
            raise ValueError("unknown JSONL record kind")
    if header is None or trailer is None:
        raise ValueError("incomplete JSONL program")
    program = MemoryAccessProgram(**header, **{name: tuple(items) for name, items in values.items()})
    if type(trailer) is not dict or set(trailer)!=set(_trailer(program,count)):
        raise ValueError('trailer fields mismatch')
    trailer={**trailer,'statistics':encode(decode(trailer['statistics'],tuple[AccessStatistics,...] | None))}
    if trailer != _trailer(program, count):
        raise ValueError("trailer counts/statistics mismatch")
    return program
