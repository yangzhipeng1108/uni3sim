"""Stream typed device events from native MetaX mcTracer Chrome traces."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterator, Literal, Mapping, TextIO, TypeAlias

from dataset_pipeline.conversion import require_object as _mapping


@dataclass(frozen=True)
class McTracerKernelEvent:
    """One device kernel event, preserving native launch metadata."""

    event_type: Literal["kernel"]
    name: str
    ts_ns: int
    dur_ns: int
    device_id: int
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    static_shared_memory_bytes: int
    dynamic_shared_memory_bytes: int
    registers_per_thread: int


@dataclass(frozen=True)
class McTracerMemcpyEvent:
    """One device copy event classified from native numeric kind fields."""

    event_type: Literal["memcpy"]
    name: str
    ts_ns: int
    dur_ns: int
    device_id: int
    bytes: int
    copy_kind: int
    src_kind: int
    dst_kind: int


@dataclass(frozen=True)
class McTracerMemsetEvent:
    """One device memset event classified from native numeric kind fields."""

    event_type: Literal["memset"]
    name: str
    ts_ns: int
    dur_ns: int
    device_id: int
    bytes: int
    mem_kind: int
    value: int


McTracerDeviceEvent: TypeAlias = (
    McTracerKernelEvent | McTracerMemcpyEvent | McTracerMemsetEvent
)


def iter_mctracer_trace_events(
    trace_path: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
    max_event_bytes: int = 16 * 1024 * 1024,
) -> Iterator[tuple[int, Mapping[str, object]]]:
    """Yield every raw Chrome event while retaining correlation metadata."""

    path = Path(trace_path)
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for index, raw_event in enumerate(
                _trace_event_values(stream, chunk_size, max_event_bytes)
            ):
                yield index, _mapping(
                    raw_event, f"mcTracer traceEvents[{index}]"
                )
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"cannot read mcTracer trace: {path}") from error


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


_JSON_DECODER = json.JSONDecoder(object_pairs_hook=_unique_object)
_JSON_WHITESPACE = re.compile(r"[ \t\r\n]*")


def _exceeds_encoded_size(
    text: str, start: int, end: int, maximum_bytes: int
) -> bool:
    character_count = end - start
    if character_count > maximum_bytes:
        return True
    if character_count <= maximum_bytes // 4:
        return False
    return len(text[start:end].encode("utf-8")) > maximum_bytes


class _ChunkedJsonReader:
    """Decode complete JSON values with bounded, chunked input buffering."""

    def __init__(
        self, stream: TextIO, chunk_size: int, max_value_bytes: int
    ) -> None:
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError("mcTracer parser chunk_size must be an integer")
        if isinstance(max_value_bytes, bool) or not isinstance(max_value_bytes, int):
            raise TypeError("mcTracer parser max_value_bytes must be an integer")
        if chunk_size <= 0:
            raise ValueError("mcTracer parser chunk_size must be positive")
        if max_value_bytes <= 0:
            raise ValueError("mcTracer parser max_value_bytes must be positive")
        if chunk_size > max_value_bytes:
            raise ValueError(
                "mcTracer parser chunk_size cannot exceed max_value_bytes"
            )
        self._stream = stream
        self._chunk_size = chunk_size
        self._max_value_bytes = max_value_bytes
        self._buffer = ""
        self._cursor = 0
        self._absolute_base = 0
        self._eof = False

    @property
    def position(self) -> int:
        return self._absolute_base + self._cursor

    def _read_more(self, preserve_from: int | None = None) -> bool:
        if self._eof:
            return False
        if preserve_from is None:
            preserve_from = self._cursor
        if preserve_from > 0:
            self._buffer = self._buffer[preserve_from:]
            self._absolute_base += preserve_from
            self._cursor -= preserve_from
        chunk = self._stream.read(self._chunk_size)
        if chunk == "":
            self._eof = True
            return False
        self._buffer += chunk
        return True

    def _ensure_available(self) -> bool:
        if self._cursor < len(self._buffer):
            return True
        return self._read_more()

    def peek(self) -> str | None:
        if not self._ensure_available():
            return None
        return self._buffer[self._cursor]

    def take(self) -> str:
        character = self.peek()
        if character is None:
            raise ValueError(
                f"unexpected end of mcTracer JSON at character {self.position}"
            )
        self._cursor += 1
        return character

    def skip_whitespace(self) -> None:
        while True:
            if not self._ensure_available():
                return
            match = _JSON_WHITESPACE.match(self._buffer, self._cursor)
            if match is None:
                raise AssertionError("JSON whitespace expression did not match")
            self._cursor = match.end()
            if self._cursor < len(self._buffer) or self._eof:
                return

    def expect(self, expected: str) -> None:
        actual = self.take()
        if actual != expected:
            raise ValueError(
                f"invalid mcTracer JSON at character {self.position - 1}: "
                f"expected {expected!r}, got {actual!r}"
            )

    def decode_value(self, context: str) -> object:
        self.skip_whitespace()
        if self.peek() is None:
            raise ValueError(
                f"unexpected end of mcTracer JSON while reading {context}"
            )
        start = self._cursor
        start_position = self.position
        while True:
            try:
                value, end = _JSON_DECODER.raw_decode(self._buffer, start)
            except json.JSONDecodeError as error:
                if _exceeds_encoded_size(
                    self._buffer,
                    start,
                    len(self._buffer),
                    self._max_value_bytes,
                ):
                    raise ValueError(
                        f"{context} exceeds maximum encoded size "
                        f"{self._max_value_bytes} bytes"
                    ) from error
                if not self._read_more(preserve_from=start):
                    raise ValueError(
                        f"invalid or truncated {context} JSON at character "
                        f"{start_position}"
                    ) from error
                start_position = self._absolute_base
                start = 0
            else:
                if _exceeds_encoded_size(
                    self._buffer, start, end, self._max_value_bytes
                ):
                    raise ValueError(
                        f"{context} exceeds maximum encoded size "
                        f"{self._max_value_bytes} bytes"
                    )
                if end == len(self._buffer) and not self._eof:
                    if self._read_more(preserve_from=start):
                        start_position = self._absolute_base
                        start = 0
                    else:
                        self._cursor = end
                        return value
                else:
                    self._cursor = end
                    return value


def _array_values(reader: _ChunkedJsonReader) -> Iterator[object]:
    reader.expect("[")
    reader.skip_whitespace()
    if reader.peek() == "]":
        reader.take()
        return
    index = 0
    while True:
        yield reader.decode_value(f"mcTracer traceEvents[{index}]")
        index += 1
        reader.skip_whitespace()
        delimiter = reader.take()
        if delimiter == "]":
            return
        if delimiter != ",":
            raise ValueError(
                "invalid mcTracer traceEvents array delimiter at character "
                f"{reader.position - 1}: {delimiter!r}"
            )
        reader.skip_whitespace()


def _trace_event_values(
    stream: TextIO, chunk_size: int, max_event_bytes: int
) -> Iterator[object]:
    reader = _ChunkedJsonReader(stream, chunk_size, max_event_bytes)
    reader.skip_whitespace()
    root = reader.peek()
    if root == "[":
        yield from _array_values(reader)
    elif root == "{":
        reader.take()
        found = False
        seen_keys = set()
        reader.skip_whitespace()
        empty_root = reader.peek() == "}"
        if empty_root:
            reader.take()
        if not empty_root:
            while True:
                if reader.peek() != '"':
                    raise ValueError(
                        "mcTracer root object key must be a string at character "
                        f"{reader.position}"
                    )
                key = reader.decode_value("mcTracer root key")
                if not isinstance(key, str):
                    raise TypeError("decoded mcTracer root key must be a string")
                if key in seen_keys:
                    raise ValueError(f"duplicate JSON object key {key!r}")
                seen_keys.add(key)
                reader.skip_whitespace()
                reader.expect(":")
                reader.skip_whitespace()
                is_trace_events = key == "traceEvents"
                if is_trace_events:
                    if found:
                        raise ValueError("mcTracer root contains duplicate traceEvents")
                    if reader.peek() != "[":
                        raise TypeError("mcTracer traceEvents must be an array")
                    found = True
                    yield from _array_values(reader)
                if not is_trace_events:
                    reader.decode_value(f"mcTracer root field {key!r}")
                reader.skip_whitespace()
                delimiter = reader.take()
                if delimiter == "}":
                    break
                if delimiter != ",":
                    raise ValueError(
                        "invalid mcTracer root object delimiter at character "
                        f"{reader.position - 1}: {delimiter!r}"
                    )
                reader.skip_whitespace()
        if not found:
            raise KeyError("mcTracer root object is missing traceEvents")
    else:
        raise TypeError("mcTracer root must be an array or object")
    reader.skip_whitespace()
    if reader.peek() is not None:
        raise ValueError(
            f"trailing data after mcTracer JSON at character {reader.position}"
        )




def _string(mapping: Mapping[str, object], key: str, context: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value:
        raise TypeError(f"{context}.{key} must be a non-empty string")
    return value


def _integer(
    mapping: Mapping[str, object], key: str, context: str, *, minimum: int
) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context}.{key} must be an integer")
    if value < minimum:
        raise ValueError(f"{context}.{key} must be at least {minimum}")
    return value


def _integer_choice(
    mapping: Mapping[str, object],
    key: str,
    context: str,
    choices: tuple[int, ...],
) -> int:
    value = _integer(mapping, key, context, minimum=0)
    if value not in choices:
        raise ValueError(
            f"{context}.{key} must be one of {choices}, got {value}"
        )
    return value


def _dim3(value: object, context: str) -> tuple[int, int, int]:
    dimensions = _mapping(value, context)
    return tuple(
        _integer(dimensions, axis, context, minimum=1)
        for axis in ("x", "y", "z")
    )


def _device_event(raw_event: object, index: int) -> McTracerDeviceEvent | None:
    event = _mapping(raw_event, f"mcTracer traceEvents[{index}]")
    if "ph" not in event or event["ph"] != "X":
        return None
    if "cat" not in event or event["cat"] != "0":
        return None
    context = f"mcTracer device event[{index}]"
    name = _string(event, "name", context)
    ts_ns = _integer(event, "ts", context, minimum=0)
    dur_ns = _integer(event, "dur", context, minimum=1)
    args = _mapping(event["args"], f"{context}.args")
    device_id = _integer(args, "device_id", f"{context}.args", minimum=0)

    kernel_keys = ("grid", "block", "mem")
    memory_keys = (
        "kind",
        "bytes",
        "copy_kind",
        "src_kind",
        "dst_kind",
        "mem_kind",
        "value",
    )
    has_kernel_discriminant = any(key in args for key in kernel_keys)
    has_memory_discriminant = any(key in args for key in memory_keys)
    if has_kernel_discriminant and has_memory_discriminant:
        raise ValueError(
            f"{context}.args mixes kernel and memory-event discriminants"
        )
    if has_kernel_discriminant:
        missing = [key for key in kernel_keys if key not in args]
        if missing:
            raise KeyError(
                f"{context}.args has incomplete kernel metadata; missing {missing}"
            )
        memory = _mapping(args["mem"], f"{context}.args.mem")
        return McTracerKernelEvent(
            event_type="kernel",
            name=name,
            ts_ns=ts_ns,
            dur_ns=dur_ns,
            device_id=device_id,
            grid=_dim3(args["grid"], f"{context}.args.grid"),
            block=_dim3(args["block"], f"{context}.args.block"),
            static_shared_memory_bytes=_integer(
                memory, "static_shared", f"{context}.args.mem", minimum=0
            ),
            dynamic_shared_memory_bytes=_integer(
                memory, "dynamic_shared", f"{context}.args.mem", minimum=0
            ),
            registers_per_thread=_integer(
                memory, "registers_per_thread", f"{context}.args.mem", minimum=0
            ),
        )

    if not has_memory_discriminant:
        raise ValueError(f"{context} is neither a kernel nor a typed memory event")
    byte_count = _integer(args, "bytes", f"{context}.args", minimum=0)
    kind = _integer(args, "kind", f"{context}.args", minimum=1)
    if kind == 1:
        memset_keys = ("mem_kind", "value")
        if any(key in args for key in memset_keys):
            raise ValueError(
                f"{context}.args mixes memcpy and memset discriminants"
            )
        return McTracerMemcpyEvent(
            event_type="memcpy",
            name=name,
            ts_ns=ts_ns,
            dur_ns=dur_ns,
            device_id=device_id,
            bytes=byte_count,
            # MACA 3.8 emits ``copy_kind=3`` for device-to-device HTOA copies.
            # Keep it as a typed memcpy event rather than rejecting an otherwise
            # complete native trace; the source/destination kinds remain explicit.
            copy_kind=_integer_choice(
                args, "copy_kind", f"{context}.args", (1, 2, 3, 8)
            ),
            src_kind=_integer_choice(
                args, "src_kind", f"{context}.args", (2, 3)
            ),
            dst_kind=_integer_choice(
                args, "dst_kind", f"{context}.args", (1, 2, 3)
            ),
        )
    if kind == 2:
        memcpy_keys = ("copy_kind", "src_kind", "dst_kind")
        if any(key in args for key in memcpy_keys):
            raise ValueError(
                f"{context}.args mixes memcpy and memset discriminants"
            )
        return McTracerMemsetEvent(
            event_type="memset",
            name=name,
            ts_ns=ts_ns,
            dur_ns=dur_ns,
            device_id=device_id,
            bytes=byte_count,
            mem_kind=_integer_choice(
                args, "mem_kind", f"{context}.args", (3,)
            ),
            value=_integer(args, "value", f"{context}.args", minimum=0),
        )
    raise ValueError(f"{context}.args.kind has unsupported value {kind}")


def iter_mctracer_device_events(
    trace_path: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
    max_event_bytes: int = 16 * 1024 * 1024,
) -> Iterator[McTracerDeviceEvent]:
    """Yield native typed device events without loading the trace into memory."""

    for index, raw_event in iter_mctracer_trace_events(
        trace_path,
        chunk_size=chunk_size,
        max_event_bytes=max_event_bytes,
    ):
        event = _device_event(raw_event, index)
        if event is not None:
            yield event
