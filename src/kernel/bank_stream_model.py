"""Serialized bank-column streaming for compute attached to a DRAM array.

A compute unit that reads its operands straight out of a DRAM bank cannot be
supplied faster than that bank's row/column protocol delivers them, and nothing
else is in flight to hide the protocol behind.  The unit therefore advances one
*column* of work at a time, and the DDR costs of opening rows and turning the
bank bus around sit directly in its critical path instead of being absorbed by
a cache hierarchy.

This module prices that serialized stream.  An operation is described as a
sequence of :class:`BankStream` stages; each stage says how many elements flow
through the units, how many bank column accesses one column of work costs, and
how often the unit's operand buffer has to be refilled from the bank.  The
mechanism itself is hardware-independent: the four protocol costs it charges
(``column access``, ``bus turnaround``, ``row cycle``, ``row close``) are
ordinary DDR command timings, and a hardware whose compute reads from a
register file or a cache instead of from an open DRAM row simply does not
publish them, so no stage can be built for it and the mechanism never applies.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Sequence


@dataclasses.dataclass(frozen=True)
class BankStream:
    """One serialized pass of work through the bank-attached compute units.

    ``accesses_per_column`` is the operator's own dataflow: how many bank
    column accesses (operand reads plus result writes, summed over the passes
    the operator makes over its data) one column of work costs.
    ``refills_per_chunk`` counts the operand-buffer refills that happen inside
    one buffer's worth of columns, and ``refills_per_row_change`` the refills
    that happen at a bank row boundary, where a reduction that spans the row
    must re-stream the row it just consumed.
    """

    work_elements: float
    accesses_per_column: int
    refills_per_chunk: int
    refills_per_row_change: int
    row_cycles_per_row_change: int
    row_change_follows_write: bool
    accumulator_roundtrip: bool
    delivered_elements: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.work_elements) or self.work_elements <= 0:
            raise ValueError(
                "BankStream.work_elements must be finite and positive, got "
                f"{self.work_elements!r}")
        if self.accesses_per_column <= 0:
            raise ValueError(
                "BankStream.accesses_per_column must be positive, got "
                f"{self.accesses_per_column!r}")
        for name in ("refills_per_chunk", "refills_per_row_change",
                     "row_cycles_per_row_change"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"BankStream.{name} must be an integer")
            if value < 0:
                raise ValueError(f"BankStream.{name} cannot be negative")
        if self.row_cycles_per_row_change < 1:
            raise ValueError(
                "BankStream must pay at least one row cycle per row change")
        if (not math.isfinite(self.delivered_elements)
                or self.delivered_elements < 0):
            raise ValueError(
                "BankStream.delivered_elements must be finite and "
                f"non-negative, got {self.delivered_elements!r}")


@dataclasses.dataclass(frozen=True)
class BankStreamGeometry:
    """Resolved bank/compute geometry and protocol costs of one hardware."""

    units: int
    lanes_per_column: int
    columns_per_row: int
    columns_per_buffer: int
    column_access_ns: float
    unit_latency_ns: float
    bus_turnaround_ns: float
    read_row_cycle_ns: float
    write_read_row_cycle_ns: float
    row_close_ns: float
    accumulator_roundtrip_ns: float

    @property
    def elements_per_column_step(self) -> int:
        return self.lanes_per_column * self.units


def resolve_geometry(hardware) -> BankStreamGeometry:
    """Resolve the bank-stream geometry published by ``hardware``."""

    memory = hardware.memory_subsystem
    row_bytes = float(memory["pim_row_bytes"])
    columns_per_row = int(memory["pim_columns_per_row"])
    if columns_per_row <= 0:
        raise ValueError("pim_columns_per_row must be positive")
    column_bytes = row_bytes / columns_per_row
    element_bytes = float(memory["dtype_bytes"])
    if element_bytes <= 0:
        raise ValueError("memory_subsystem.dtype_bytes must be positive")
    lanes = column_bytes / element_bytes
    if lanes <= 0 or lanes != int(lanes):
        raise ValueError(
            "a bank column must hold a whole number of elements, got "
            f"{lanes!r}")
    buffer_columns = float(memory["pim_pu_inbuf_bytes"]) / column_bytes
    if buffer_columns < 1 or buffer_columns != int(buffer_columns):
        raise ValueError(
            "the operand buffer must hold a whole number of columns, got "
            f"{buffer_columns!r}")
    units = int(hardware.num_compute_units)
    if units <= 0:
        raise ValueError("bank-stream hardware must have compute units")
    return BankStreamGeometry(
        units=units,
        lanes_per_column=int(lanes),
        columns_per_row=columns_per_row,
        columns_per_buffer=int(buffer_columns),
        column_access_ns=float(memory["pim_column_access_ns"]),
        unit_latency_ns=float(memory["pim_pu_lat_ns"]),
        bus_turnaround_ns=float(memory["pim_bus_turnaround_ns"]),
        read_row_cycle_ns=float(memory["pim_row_activate_ns"]),
        write_read_row_cycle_ns=float(
            memory["pim_write_read_row_cycle_ns"]),
        row_close_ns=float(memory["pim_row_close_ns"]),
        accumulator_roundtrip_ns=float(
            memory["pim_accumulator_roundtrip_ns"]),
    )


def _columns(elements: float, geometry: BankStreamGeometry) -> int:
    """Work columns one unit advances through for ``elements`` of work."""

    return max(1, int(math.ceil(elements / geometry.elements_per_column_step)))


def stream_breakdown(stage: BankStream,
                     geometry: BankStreamGeometry) -> dict:
    """Return one stage's time split across the phases that produce it.

    The keys name physical phases of the bank protocol, so a residual can be
    attributed to a phase instead of to the operation as a whole.
    """

    column = geometry.column_access_ns
    refill_burst = (geometry.columns_per_buffer - 1) * column
    row_cycle = (
        geometry.write_read_row_cycle_ns if stage.row_change_follows_write
        else geometry.read_row_cycle_ns)

    # One column of work: its own bank accesses, or the unit's SIMD latency
    # when that is the slower of the two.
    advance = max(stage.accesses_per_column * column, geometry.unit_latency_ns)
    # Crossing an operand-buffer boundary additionally streams a fresh buffer
    # from the bank, which first has to turn the bus back from write to read.
    at_buffer_edge = advance + stage.refills_per_chunk * (
        refill_burst + geometry.bus_turnaround_ns)
    # Crossing a bank row boundary closes the open row and opens the next one
    # instead of merely turning the bus around, and flushes the accumulator.
    at_row_edge = (
        advance
        + stage.refills_per_row_change * refill_burst
        + stage.row_cycles_per_row_change * row_cycle
        + (geometry.accumulator_roundtrip_ns
           if stage.accumulator_roundtrip else 0.0))
    # The stream's first column opens a row like any row boundary, except that
    # it has no previously open row to close first.
    entry = at_row_edge - geometry.row_close_ns

    columns = _columns(stage.work_elements, geometry)
    row_edges = int(math.ceil(columns / geometry.columns_per_row)) - 1
    buffer_edges = (
        int(math.ceil(columns / geometry.columns_per_buffer)) - 1 - row_edges)
    plain = columns - 1 - row_edges - buffer_edges

    breakdown = {
        "column_access": (columns - 1) * advance,
        "operand_buffer_refill": buffer_edges * (at_buffer_edge - advance),
        "bank_row_cycle": row_edges * (at_row_edge - advance),
        "pipeline_entry": entry,
        "result_delivery": 0.0,
    }
    if plain < 0:
        raise AssertionError("bank-stream column accounting went negative")

    if stage.delivered_elements > 0:
        # A result held in the compute unit's accumulator still has to be read
        # back out of the bank by whoever consumes it.
        delivered = _columns(stage.delivered_elements, geometry)
        delivered_rows = int(
            math.ceil(delivered / geometry.columns_per_row))
        breakdown["result_delivery"] = delivered * column + delivered_rows * (
            geometry.read_row_cycle_ns + geometry.accumulator_roundtrip_ns)
    return breakdown


def stream_ns(stage: BankStream, geometry: BankStreamGeometry) -> float:
    """Return the serialized time of one bank-stream stage, in nanoseconds."""

    return math.fsum(stream_breakdown(stage, geometry).values())


def bank_stream_ns(stages: Sequence[BankStream], hardware) -> float:
    """Return the serialized time of an operation's bank-stream stages."""

    if not stages:
        raise ValueError("a bank-stream operation needs at least one stage")
    geometry = resolve_geometry(hardware)
    return math.fsum(stream_ns(stage, geometry) for stage in stages)


# ─── Operator dataflow profiles ───────────────────────────────────────────
# Each profile states how one column of work moves through the bank, in that
# operator's own terms.  The counts are the operator's dataflow, not a
# hardware property: they hold for any bank-attached compute unit.

def reduction_stream(work_elements: float,
                     delivered_elements: float) -> BankStream:
    """A matrix product: stream one operand, accumulate in the unit.

    The streamed operand costs one column read per column of work; the other
    operand is pushed into the unit's buffer by the host, so no bank bus
    turnaround is needed.  The accumulator is parked in the output buffer once
    per row, and the finished result is read back out of the bank.
    """

    return BankStream(
        work_elements=work_elements,
        accesses_per_column=1,
        refills_per_chunk=0,
        refills_per_row_change=0,
        row_cycles_per_row_change=1,
        row_change_follows_write=False,
        accumulator_roundtrip=True,
        delivered_elements=delivered_elements,
    )


def pointwise_stream(work_elements: float) -> BankStream:
    """An element-wise operator: read one operand column, write one result.

    The second operand arrives through the unit's operand buffer, refilled
    once per buffer's worth of columns.  The result is written back into the
    bank in place, so nothing is read back out.
    """

    return BankStream(
        work_elements=work_elements,
        accesses_per_column=2,
        refills_per_chunk=1,
        refills_per_row_change=1,
        row_cycles_per_row_change=1,
        row_change_follows_write=True,
        accumulator_roundtrip=False,
    )


def normalization_stream(work_elements: float) -> BankStream:
    """A mean/variance normalization: reduce over the row, then rescale it.

    Four operand reads and one result write per column of work.  The rescale
    cannot start before the reduction it depends on finishes, so at every row
    boundary — the granularity the reduction closes at — the row just consumed
    is streamed through the operand buffer a second time.
    """

    return BankStream(
        work_elements=work_elements,
        accesses_per_column=5,
        refills_per_chunk=1,
        refills_per_row_change=2,
        row_cycles_per_row_change=1,
        row_change_follows_write=True,
        accumulator_roundtrip=False,
    )


def softmax_stream(work_elements: float) -> BankStream:
    """A softmax: exponentiate, accumulate the denominator, then normalize.

    Two operand reads and one result write per column of work, carried by two
    independent passes over the data, so both the operand buffer and the bank
    row are cycled twice as often as a single-pass operator cycles them.
    """

    return BankStream(
        work_elements=work_elements,
        accesses_per_column=3,
        refills_per_chunk=2,
        refills_per_row_change=2,
        row_cycles_per_row_change=2,
        row_change_follows_write=True,
        accumulator_roundtrip=False,
    )


def matmul_stream(rows: float, reduce_length: float,
                  columns: float) -> BankStream:
    """A ``rows x reduce_length x columns`` matrix product as a bank stream."""

    if rows <= 0 or reduce_length <= 0 or columns <= 0:
        raise ValueError(
            "a matrix-product bank stream needs positive dimensions, got "
            f"{(rows, reduce_length, columns)!r}")
    return reduction_stream(
        work_elements=float(rows) * float(reduce_length) * float(columns),
        delivered_elements=float(rows) * float(columns))
