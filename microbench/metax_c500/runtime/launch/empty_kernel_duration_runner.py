#!/usr/bin/env python3
"""Orchestrate the retained MetaX C500 empty-task capture."""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import statistics
import struct
import subprocess
import sys
import time


SCHEMA_VERSION = 6
TELEMETRY_SCHEMA_VERSION = 1
HARDWARE_ID = "metax_c500"
INDEPENDENT_RUNS = 3
TRIALS_PER_RUN = 31
SCHEDULE_SEED = 500031
MAXIMUM_WORKGROUP_BUDGET = 1048576
CHILD_TIMEOUT_SECONDS = 7200
TELEMETRY_TIMEOUT_SECONDS = 30
PROCESS_STOP_GRACE_SECONDS = 10
TARGET_XCORE_CLOCK_MHZ = 1125
TARGET_MC_CLOCK_MHZ = 1800
# The empty kernel issues no HBM instruction.  Freeze a conservative external
# traffic ceiling at 0.01% of the documented 1.8-TB/s C500 HBM envelope before
# the production-coordinate capture; this is a mechanism-isolation gate, not a
# value selected from launch duration or evaluation error.
NOMINAL_HBM_THROUGHPUT_MB_S = 1800000.0
MAX_LIVE_HBM_POLLUTION_FRACTION = 0.0001
MAX_LIVE_HBM_THROUGHPUT_MB_S = (
    NOMINAL_HBM_THROUGHPUT_MB_S * MAX_LIVE_HBM_POLLUTION_FRACTION
)
MAX_TELEMETRY_SAMPLE_GAP_NS = 500000000
LIVE_TELEMETRY_MAX_SAMPLES = 100000
POSTFLIGHT_TELEMETRY_MAX_SAMPLES = 256
POSTFLIGHT_TELEMETRY_TIMEOUT_SECONDS = 60
# A pointwise 95% interval at each of hundreds of production coordinates
# creates about 5% false incompatibilities even for a stationary device.
# Freeze one family-wise error budget, independent of duration and evaluation
# data, and divide it over the exact run/geometry interval family.
PRODUCTION_FAMILYWISE_ERROR_RATE = 0.05
BASELINE_PATTERN = re.compile(r"[0-9a-f]{40}")
IMAGE_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
BUS_ID_PATTERN = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
CONTAINER_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
AUDITED_CONTAINER_ID = (
    "d5653011acbb181a5ee925f3d80acc335d2397fe6e23fcbb390fdf3e4373e480"
)
AUDITED_IMAGE_ID = (
    "sha256:c7c1ae1509b7d5b53ba4665193a6bacef4247d7e14894d0d84b63efe1abdc2c9"
)
AUDITED_HOST_UID = 1017
AUDITED_DEVICE_GID = 44
CAPTURE_CONTAINER_USER = f"{AUDITED_HOST_UID}:{AUDITED_DEVICE_GID}"
REPO_ROOT = Path(__file__).resolve().parents[4]
PROFILER_OUTPUT_ROOT = (
    REPO_ROOT / "profiler_output/microbench/metax_c500"
)
TELEMETRY_RUNNER_PATH = Path(__file__).with_name(
    "mx_smi_telemetry_runner.py"
).resolve()
PRODUCTION_GEOMETRY_MANIFEST_PATH = Path(__file__).with_name(
    "production_launch_geometry_manifest.json"
).resolve()
MXOBJDUMP_CONTAINER_PATH = "/opt/maca/mxgpu_llvm/bin/mxobjdump"
MXCC_CONTAINER_PATH = "/opt/maca/mxgpu_llvm/bin/mxcc"
MXCC_CANONICAL_FLAGS = (
    "-x", "maca", "-offload-arch=native",
    "--maca-path=/opt/maca", "-O3",
)
CONTAINER_SHA256SUM_PATH = "/usr/bin/sha256sum"
CONTAINER_PYTHON_PATH = "/usr/bin/python3"
MX_SMI_PATH = Path("/opt/mxdriver/bin/mx-smi")
DOCKER_PATH = Path("/usr/bin/docker")
HANDLED_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
SAMPLE_FIELDS = (
    "run_index", "sample_id", "trial", "order_position", "pair_id",
    "geometry_id", "grid_x", "grid_y", "grid_z", "block_x",
    "block_y", "block_z", "launch_count", "host_start_ns",
    "host_end_ns", "host_elapsed_ns", "batch_device_ns",
    "per_launch_device_ns", "device_elapsed_ms_hex", "event_status",
    "grid_workgroups", "candidate_sample",
)
CAPTURE_RUN_RECORD_FIELDS = frozenset({
    "status", "run_index", "child_container_pid", "child_pid_identity",
    "child_host_pid", "launcher_pid", "launcher_pgid", "pid_mapping",
    "command", "started_monotonic_raw_ns", "ended_monotonic_raw_ns",
    "returncode", "container_pid_evidence_path",
    "container_pid_evidence_sha256", "plan_path", "plan_sha256",
    "stdout_path", "stdout_sha256", "stderr_path", "stderr_sha256",
    "pid_identity_source", "container_pid_tree_cleanup",
    "container_pid_tree_cleanup_attempts", "signal",
})


class RunnerSignalError(RuntimeError):
    def __init__(self, event):
        self.event = dict(event)
        super().__init__(
            "runner received "
            f"{event['signal_name']} ({event['signal_number']})"
        )


class RunnerSignalState:
    def __init__(self):
        self.state = "created"
        self.events = []
        self.error = None
        self.previous_handlers = []

    def install(self):
        if self.state != "created":
            raise RuntimeError("signal state can only be installed once")
        self.state = "armed"
        try:
            for handled_signal in HANDLED_SIGNALS:
                previous = signal.getsignal(handled_signal)
                signal.signal(handled_signal, self._handle)
                self.previous_handlers.append((handled_signal, previous))
        except (OSError, ValueError):
            for handled_signal, previous in reversed(self.previous_handlers):
                signal.signal(handled_signal, previous)
            self.state = "completed"
            raise

    def _handle(self, signal_number, frame):
        del frame
        event = {
            "ordinal": len(self.events) + 1,
            "signal_number": signal_number,
            "signal_name": signal.Signals(signal_number).name,
            "received_monotonic_raw_ns": time.clock_gettime_ns(
                time.CLOCK_MONOTONIC_RAW,
            ),
        }
        self.events.append(event)
        if self.error is None:
            self.error = RunnerSignalError(event)
        if self.state == "armed":
            self.state = "interrupted"
            raise self.error
        elif self.state in {"interrupted", "finalizing"}:
            return
        else:
            raise RuntimeError(
                f"signal arrived in invalid runner state: {self.state}"
            )

    def enter_finalization(self):
        if self.state in {"armed", "interrupted"}:
            self.state = "finalizing"
        elif self.state == "finalizing":
            return
        else:
            raise RuntimeError(
                f"cannot finalize runner signal state: {self.state}"
            )

    def rearm_after_success(self):
        if self.state != "finalizing":
            raise RuntimeError(
                f"cannot rearm runner signal state: {self.state}"
            )
        if self.events:
            raise self.error
        self.state = "armed"

    def signal_record(self):
        if not self.events:
            return None
        return {
            "state": self.state,
            "events": [dict(event) for event in self.events],
        }

    def restore(self):
        if self.state == "completed":
            return
        elif self.state in {
                "armed", "interrupted", "finalizing"}:
            for handled_signal, previous in reversed(self.previous_handlers):
                signal.signal(handled_signal, previous)
            self.state = "completed"
        else:
            raise RuntimeError(
                f"cannot restore runner signal state: {self.state}"
            )


@dataclass(frozen=True)
class Geometry:
    geometry_id: str
    grid_x: int
    grid_y: int
    grid_z: int
    block_x: int
    block_y: int
    block_z: int


@dataclass(frozen=True)
class GeometryPair:
    pair_id: str
    axis_geometry: Geometry
    reference_geometry: Geometry
    launch_counts: tuple


PAIRS = (
    GeometryPair(
        pair_id="xyz_32x64x128_vs_x_262144_block512",
        axis_geometry=Geometry(
            "grid_32x64x128_block_512x1x1",
            32, 64, 128, 512, 1, 1,
        ),
        reference_geometry=Geometry(
            "grid_262144x1x1_block_512x1x1",
            262144, 1, 1, 512, 1, 1,
        ),
        launch_counts=(1, 16, 256),
    ),
    GeometryPair(
        pair_id="yz_1x512x2_vs_x_1024_block1024",
        axis_geometry=Geometry(
            "grid_1x512x2_block_1024x1x1",
            1, 512, 2, 1024, 1, 1,
        ),
        reference_geometry=Geometry(
            "grid_1024x1x1_block_1024x1x1",
            1024, 1, 1, 1024, 1, 1,
        ),
        launch_counts=(1, 16, 256),
    ),
)
PRODUCTION_FAMILY_ID = "production_launch_geometry"
PRODUCTION_LAUNCH_COUNT = 1
SMOKE_PAIR_ID = "yz_1x512x2_vs_x_1024_block1024"
SMOKE_LAUNCH_COUNT = 1024
SMOKE_TRIALS = 8
SMOKE_SAMPLE_COUNT = 2 * SMOKE_TRIALS
GEOMETRY_PROBE_MAGIC = b"C500GEO1"
GEOMETRY_PROBE_PROTOCOL_VERSION = 1
GEOMETRY_PROBE_GEOMETRIES = tuple(
    geometry
    for pair in PAIRS
    for geometry in (pair.axis_geometry, pair.reference_geometry)
)
GEOMETRY_PROBE_CASE_HEADER = struct.Struct("<10I")
GEOMETRY_PROBE_RECORD = struct.Struct("<10I")


def _grid_workgroups(geometry):
    return geometry.grid_x * geometry.grid_y * geometry.grid_z


def _launch_count_allowed(pair, launch_count):
    axis_workgroups = _grid_workgroups(pair.axis_geometry)
    reference_workgroups = _grid_workgroups(pair.reference_geometry)
    if axis_workgroups != reference_workgroups:
        raise ValueError("paired geometries must have equal workgroup counts")
    return axis_workgroups * launch_count <= MAXIMUM_WORKGROUP_BUDGET


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _payload_sha256(payload):
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _write_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _nonnegative_integer(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _geometry_payload(geometry):
    return {
        "geometry_id": geometry.geometry_id,
        "grid_x": geometry.grid_x,
        "grid_y": geometry.grid_y,
        "grid_z": geometry.grid_z,
        "block_x": geometry.block_x,
        "block_y": geometry.block_y,
        "block_z": geometry.block_z,
        "grid_workgroups_diagnostic": (
            geometry.grid_x * geometry.grid_y * geometry.grid_z
        ),
    }


def parse_geometry_probe(raw):
    view = memoryview(raw)
    offset = 0

    def consume(size, label):
        nonlocal offset
        end = offset + size
        if end > len(view):
            raise ValueError(f"geometry probe raw is truncated in {label}")
        value = view[offset:end]
        offset = end
        return value

    if bytes(consume(len(GEOMETRY_PROBE_MAGIC), "magic")) != (
            GEOMETRY_PROBE_MAGIC):
        raise ValueError("geometry probe raw magic is invalid")
    geometry_count = struct.unpack("<I", consume(4, "geometry count"))[0]
    if geometry_count != len(GEOMETRY_PROBE_GEOMETRIES):
        raise ValueError("geometry probe geometry count is invalid")

    geometry_results = []
    total_record_count = 0
    for expected_index, geometry in enumerate(GEOMETRY_PROBE_GEOMETRIES):
        header = GEOMETRY_PROBE_CASE_HEADER.unpack(
            consume(GEOMETRY_PROBE_CASE_HEADER.size, "case header")
        )
        (probe_index, grid_x, grid_y, grid_z, block_x, block_y, block_z,
         expected_count, actual_count, record_count) = header
        expected_geometry = (
            geometry.grid_x, geometry.grid_y, geometry.grid_z,
            geometry.block_x, geometry.block_y, geometry.block_z,
        )
        if probe_index != expected_index:
            raise ValueError("geometry probe case order is invalid")
        if (grid_x, grid_y, grid_z, block_x, block_y, block_z) != (
                expected_geometry):
            raise ValueError("geometry probe case geometry is invalid")
        canonical_count = grid_x * grid_y * grid_z
        if expected_count != canonical_count:
            raise ValueError("geometry probe expected record count is invalid")
        if actual_count != canonical_count or record_count != canonical_count:
            raise ValueError(
                "geometry probe has a missing or extra workgroup record"
            )

        coordinates = set()
        slots = set()
        for _ in range(record_count):
            record = GEOMETRY_PROBE_RECORD.unpack(
                consume(GEOMETRY_PROBE_RECORD.size, "record")
            )
            (slot, coordinate_x, coordinate_y, coordinate_z,
             record_grid_x, record_grid_y, record_grid_z,
             record_block_x, record_block_y, record_block_z) = record
            if slot >= canonical_count or slot in slots:
                raise ValueError("geometry probe allocation slot is duplicated")
            slots.add(slot)
            if (record_grid_x, record_grid_y, record_grid_z) != (
                    grid_x, grid_y, grid_z):
                raise ValueError("geometry probe gridDim mismatch")
            if (record_block_x, record_block_y, record_block_z) != (
                    block_x, block_y, block_z):
                raise ValueError("geometry probe blockDim mismatch")
            coordinate = (coordinate_x, coordinate_y, coordinate_z)
            if (coordinate_x >= grid_x or coordinate_y >= grid_y
                    or coordinate_z >= grid_z):
                raise ValueError("geometry probe coordinate is out of range")
            if coordinate in coordinates:
                raise ValueError(
                    "geometry probe has a duplicate workgroup coordinate"
                )
            coordinates.add(coordinate)
        if len(slots) != canonical_count:
            raise ValueError("geometry probe allocation slots are incomplete")
        for coordinate_z in range(grid_z):
            for coordinate_y in range(grid_y):
                for coordinate_x in range(grid_x):
                    if (coordinate_x, coordinate_y, coordinate_z) not in (
                            coordinates):
                        raise ValueError(
                            "geometry probe coordinate set is incomplete"
                        )
        total_record_count += record_count
        geometry_results.append({
            "geometry_id": geometry.geometry_id,
            "grid": [grid_x, grid_y, grid_z],
            "block": [block_x, block_y, block_z],
            "record_count": record_count,
            "coordinate_set": "exact",
            "grid_dim": "exact",
            "block_dim": "exact",
        })
    if offset != len(view):
        raise ValueError("geometry probe raw has trailing bytes")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": GEOMETRY_PROBE_PROTOCOL_VERSION,
        "status": "verified",
        "probe_mode": "device_visible_launch_geometry",
        "timing": "none",
        "hw_spec_eligibility": "not_hw_spec_eligible",
        "geometry_count": geometry_count,
        "record_count": total_record_count,
        "geometries": geometry_results,
    }


def build_matrix():
    pairs = []
    for pair in PAIRS:
        launch_decisions = []
        for launch_count in pair.launch_counts:
            if _launch_count_allowed(pair, launch_count):
                if launch_count == 1:
                    disposition = "candidate"
                else:
                    disposition = "diagnostic"
                skip_reason = None
            else:
                disposition = "skipped"
                skip_reason = "workgroup_budget_exceeded"
            launch_decisions.append({
                "launch_count": launch_count,
                "disposition": disposition,
                "skip_reason": skip_reason,
                "requested_workgroups": (
                    _grid_workgroups(pair.axis_geometry) * launch_count
                ),
                "maximum_workgroup_budget": MAXIMUM_WORKGROUP_BUDGET,
            })
        pairs.append({
            "pair_id": pair.pair_id,
            "axis_geometry": _geometry_payload(pair.axis_geometry),
            "reference_geometry": _geometry_payload(
                pair.reference_geometry,
            ),
            "launch_counts": list(pair.launch_counts),
            "launch_decisions": launch_decisions,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "empty_task_duration",
        "hardware_id": HARDWARE_ID,
        "independent_runs": INDEPENDENT_RUNS,
        "trials_per_run": TRIALS_PER_RUN,
        "schedule_seed": SCHEDULE_SEED,
        "maximum_workgroup_budget": MAXIMUM_WORKGROUP_BUDGET,
        "schedule_contract": (
            "fixed-seed pair-local AB/BA assignment; absolute count "
            "difference is one because each run has 31 trials"
        ),
        "timing_contract": {
            "device": "ordinary mcEvent",
            "host": "CLOCK_MONOTONIC_RAW",
            "measured_kernel": "independent empty kernel (endk only)",
            "profilers": [],
        },
        "production_geometry_sweep": {
            "family_id": PRODUCTION_FAMILY_ID,
            "geometry_policy": "exact_grid_and_block_only",
            "launch_count": PRODUCTION_LAUNCH_COUNT,
            "geometry_manifest": load_production_geometry_manifest()[0],
            "geometries": [
                _geometry_payload(geometry)
                for geometry in load_production_geometry_manifest()[1]
            ],
        },
        "pairs": pairs,
    }


def _pair_orders(run_index, pair_index, launch_count_index):
    orders = ["AB"] * 16 + ["BA"] * 15
    seed = (
        SCHEDULE_SEED
        + run_index * 1009
        + pair_index * 101
        + launch_count_index * 17
    )
    random.Random(seed).shuffle(orders)
    return orders


def _production_geometry_order(run_index, trial):
    _, production_geometries = load_production_geometry_manifest()
    geometries = list(production_geometries)
    seed = SCHEDULE_SEED + run_index * 1000003 + trial * 10007
    random.Random(seed).shuffle(geometries)
    return geometries


def build_plan(run_index):
    if run_index < 1 or run_index > INDEPENDENT_RUNS:
        raise ValueError("run_index is outside the canonical run set")
    unit_orders = {}
    for pair_index, pair in enumerate(PAIRS):
        for launch_count_index, launch_count in enumerate(
                pair.launch_counts):
            if not _launch_count_allowed(pair, launch_count):
                continue
            key = (pair.pair_id, launch_count)
            unit_orders[key] = _pair_orders(
                run_index, pair_index, launch_count_index,
            )

    plan = []
    order_position = 0
    for trial in range(1, TRIALS_PER_RUN + 1):
        for pair in PAIRS:
            for launch_count in pair.launch_counts:
                if not _launch_count_allowed(pair, launch_count):
                    continue
                key = (pair.pair_id, launch_count)
                order = unit_orders[key][trial - 1]
                if order == "AB":
                    geometries = (
                        pair.axis_geometry, pair.reference_geometry,
                    )
                elif order == "BA":
                    geometries = (
                        pair.reference_geometry, pair.axis_geometry,
                    )
                else:
                    raise AssertionError("unreachable scheduled pair order")
                for geometry in geometries:
                    sample_id = (
                        f"r{run_index:02d}.t{trial:02d}."
                        f"p{len(plan):04d}"
                    )
                    plan.append({
                        "sample_id": sample_id,
                        "trial": trial,
                        "order_position": order_position,
                        "pair_id": pair.pair_id,
                        "geometry_id": geometry.geometry_id,
                        "grid_x": geometry.grid_x,
                        "grid_y": geometry.grid_y,
                        "grid_z": geometry.grid_z,
                        "block_x": geometry.block_x,
                        "block_y": geometry.block_y,
                        "block_z": geometry.block_z,
                        "launch_count": launch_count,
                    })
                    order_position += 1
        for geometry in _production_geometry_order(run_index, trial):
            sample_id = (
                f"r{run_index:02d}.t{trial:02d}."
                f"p{len(plan):04d}"
            )
            plan.append({
                "sample_id": sample_id,
                "trial": trial,
                "order_position": order_position,
                "pair_id": PRODUCTION_FAMILY_ID,
                "geometry_id": geometry.geometry_id,
                "grid_x": geometry.grid_x,
                "grid_y": geometry.grid_y,
                "grid_z": geometry.grid_z,
                "block_x": geometry.block_x,
                "block_y": geometry.block_y,
                "block_z": geometry.block_z,
                "launch_count": PRODUCTION_LAUNCH_COUNT,
            })
            order_position += 1
    validate_plan(plan)
    return plan


def build_smoke_plan(pair_id, launch_count):
    if pair_id != SMOKE_PAIR_ID or launch_count != SMOKE_LAUNCH_COUNT:
        raise ValueError("smoke plan is not the canonical frozen smoke")
    matches = [pair for pair in PAIRS if pair.pair_id == pair_id]
    if len(matches) != 1:
        raise ValueError("smoke pair_id is not canonical")
    pair = matches[0]
    if not _launch_count_allowed(pair, launch_count):
        raise ValueError("smoke launch_count exceeds workgroup budget")
    plan = []
    for trial in range(1, SMOKE_TRIALS + 1):
        if trial % 2 == 1:
            geometries = (pair.axis_geometry, pair.reference_geometry)
        else:
            geometries = (pair.reference_geometry, pair.axis_geometry)
        for geometry in geometries:
            plan.append({
                "sample_id": f"smoke.t{trial}.p{len(plan)}",
                "trial": trial,
                "order_position": len(plan),
                "pair_id": pair.pair_id,
                "geometry_id": geometry.geometry_id,
                "grid_x": geometry.grid_x, "grid_y": geometry.grid_y,
                "grid_z": geometry.grid_z, "block_x": geometry.block_x,
                "block_y": geometry.block_y, "block_z": geometry.block_z,
                "launch_count": launch_count,
            })
    if len(plan) != SMOKE_SAMPLE_COUNT:
        raise AssertionError("timed smoke plan size is invalid")
    return plan


def build_smoke_matrix(pair_id, launch_count):
    plan = build_smoke_plan(pair_id, launch_count)
    return {
        "schema_version": SCHEMA_VERSION,
        "capture_mode": "timed_smoke",
        "status": "not_candidate",
        "hw_spec_eligibility": "not_hw_spec_eligible",
        "independent_runs": 1,
        "trials": SMOKE_TRIALS,
        "pair_id": pair_id,
        "launch_count": launch_count,
        "sample_count": len(plan),
        "order_contract": "alternating_AB_BA",
    }


def validate_plan(plan):
    _, production_geometries = load_production_geometry_manifest()
    expected_rows = (
        TRIALS_PER_RUN
        * sum(
            2 * sum(
                1 for launch_count in pair.launch_counts
                if _launch_count_allowed(pair, launch_count)
            )
            for pair in PAIRS
        )
        + TRIALS_PER_RUN * len(production_geometries)
    )
    if len(plan) != expected_rows:
        raise ValueError("plan row count does not match the declared matrix")
    sample_ids = set()
    pair_orders = {}
    for row in plan:
        sample_id = row["sample_id"]
        if sample_id in sample_ids:
            raise ValueError("plan contains a duplicate sample_id")
        sample_ids.add(sample_id)
        key = (row["trial"], row["pair_id"], row["launch_count"])
        if key not in pair_orders:
            pair_orders[key] = []
        pair_orders[key].append(row["geometry_id"])
    for pair in PAIRS:
        for launch_count in pair.launch_counts:
            if not _launch_count_allowed(pair, launch_count):
                continue
            ab_count = 0
            ba_count = 0
            for trial in range(1, TRIALS_PER_RUN + 1):
                key = (trial, pair.pair_id, launch_count)
                order = pair_orders[key]
                expected_ids = {
                    pair.axis_geometry.geometry_id,
                    pair.reference_geometry.geometry_id,
                }
                if len(order) != 2 or set(order) != expected_ids:
                    raise ValueError("one pair/trial does not contain A and B")
                if order[0] == pair.axis_geometry.geometry_id:
                    ab_count += 1
                elif order[0] == pair.reference_geometry.geometry_id:
                    ba_count += 1
                else:
                    raise AssertionError("unreachable pair order")
            if abs(ab_count - ba_count) != 1:
                raise ValueError("pair-local AB/BA schedule is not balanced")
    expected_exact = {
        geometry.geometry_id: geometry
        for geometry in production_geometries
    }
    if len(expected_exact) != len(production_geometries):
        raise AssertionError("production geometry IDs are not unique")
    for trial in range(1, TRIALS_PER_RUN + 1):
        rows = [
            row for row in plan
            if row["trial"] == trial
            and row["pair_id"] == PRODUCTION_FAMILY_ID
        ]
        if len(rows) != len(production_geometries):
            raise ValueError(
                "one trial does not contain the production geometry sweep")
        if {row["geometry_id"] for row in rows} != set(expected_exact):
            raise ValueError(
                "one trial changed the production geometry set")
        for row in rows:
            geometry = expected_exact[row["geometry_id"]]
            actual = (
                row["grid_x"], row["grid_y"], row["grid_z"],
                row["block_x"], row["block_y"], row["block_z"],
                row["launch_count"],
            )
            expected = (
                geometry.grid_x, geometry.grid_y, geometry.grid_z,
                geometry.block_x, geometry.block_y, geometry.block_z,
                PRODUCTION_LAUNCH_COUNT,
            )
            if actual != expected:
                raise ValueError(
                    "one trial changed a production launch geometry")
    if [row["order_position"] for row in plan] != list(range(len(plan))):
        raise ValueError("plan order positions must be contiguous")


def _plan_text(plan):
    lines = []
    for row in plan:
        fields = (
            row["sample_id"], row["trial"], row["order_position"],
            row["pair_id"], row["geometry_id"],
            row["grid_x"], row["grid_y"], row["grid_z"],
            row["block_x"], row["block_y"], row["block_z"],
            row["launch_count"],
        )
        lines.append(" ".join(str(field) for field in fields))
    return "\n".join(lines) + "\n"


def _parse_int(text, label):
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{label} is not an integer") from exc


def _parse_float(text, label):
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{label} is not a float") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} is not finite")
    return value


def _parse_hex_float(text, label):
    try:
        value = float.fromhex(text)
    except ValueError as exc:
        raise ValueError(f"{label} is not a hexadecimal float") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} is invalid")
    return value


def parse_capture(stdout, plan, run_index):
    rows = list(csv.reader(io.StringIO(stdout)))
    if len(rows) != len(plan) + 2:
        raise ValueError("capture output row count does not match its plan")
    metadata_row = rows[0]
    if len(metadata_row) != 10 or metadata_row[0] != "metadata":
        raise ValueError("capture metadata schema is invalid")
    if metadata_row[1] != str(SCHEMA_VERSION):
        raise ValueError("capture metadata schema version is unsupported")
    if metadata_row[8] != "CLOCK_MONOTONIC_RAW":
        raise ValueError("capture changed the declared host clock")
    if metadata_row[9] != "mcEvent":
        raise ValueError("capture changed the declared device timing API")
    metadata = {
        "device_name": metadata_row[2],
        "device_id": _parse_int(metadata_row[3], "device_id"),
        "pci_bus_id": metadata_row[4],
        "ap_count": _parse_int(metadata_row[5], "ap_count"),
        "wave_size": _parse_int(metadata_row[6], "wave_size"),
        "runtime_reported_clock_khz_diagnostic": _parse_int(
            metadata_row[7], "reported_clock_khz",
        ),
        "host_clock": metadata_row[8],
        "device_timer": metadata_row[9],
    }
    if metadata["ap_count"] != 104 or metadata["wave_size"] != 64:
        raise ValueError("capture metadata is not a C500 device")
    if metadata["runtime_reported_clock_khz_diagnostic"] <= 0:
        raise ValueError("capture reported a non-positive device clock")

    expected_header = [
        "sample_header", "sample_id", "trial", "order_position",
        "pair_id", "geometry_id", "grid_x", "grid_y", "grid_z",
        "block_x", "block_y", "block_z", "launch_count",
        "host_start_ns", "host_end_ns", "batch_device_ns",
        "per_launch_device_ns", "device_elapsed_ms_hex", "event_status",
    ]
    if rows[1] != expected_header:
        raise ValueError("capture sample header is invalid")

    samples = []
    planned_integer_names = (
        "trial", "order_position", "grid_x", "grid_y", "grid_z",
        "block_x", "block_y", "block_z", "launch_count",
    )
    for output_row, planned in zip(rows[2:], plan):
        if len(output_row) != 19 or output_row[0] != "sample":
            raise ValueError("capture sample schema is invalid")
        parsed = {
            "sample_id": output_row[1],
            "trial": _parse_int(output_row[2], "trial"),
            "order_position": _parse_int(
                output_row[3], "order_position",
            ),
            "pair_id": output_row[4],
            "geometry_id": output_row[5],
            "grid_x": _parse_int(output_row[6], "grid_x"),
            "grid_y": _parse_int(output_row[7], "grid_y"),
            "grid_z": _parse_int(output_row[8], "grid_z"),
            "block_x": _parse_int(output_row[9], "block_x"),
            "block_y": _parse_int(output_row[10], "block_y"),
            "block_z": _parse_int(output_row[11], "block_z"),
            "launch_count": _parse_int(output_row[12], "launch_count"),
            "host_start_ns": _parse_int(
                output_row[13], "host_start_ns",
            ),
            "host_end_ns": _parse_int(output_row[14], "host_end_ns"),
            "batch_device_ns": _parse_float(
                output_row[15], "batch_device_ns",
            ),
            "per_launch_device_ns": _parse_float(
                output_row[16], "per_launch_device_ns",
            ),
            "device_elapsed_ms_hex": output_row[17],
            "event_status": output_row[18],
        }
        for name in (
                "sample_id", "pair_id", "geometry_id",
                *planned_integer_names):
            if parsed[name] != planned[name]:
                raise ValueError(
                    f"capture changed planned field {name} for "
                    f"{planned['sample_id']}"
                )
        elapsed_ms = _parse_hex_float(
            parsed["device_elapsed_ms_hex"], "device_elapsed_ms_hex",
        )
        if not math.isclose(
                parsed["batch_device_ns"], elapsed_ms * 1000000.0,
                rel_tol=1.0e-9, abs_tol=1.0e-6):
            raise ValueError("device elapsed hex does not match batch ns")
        if parsed["event_status"] == "resolved":
            if parsed["batch_device_ns"] <= 0.0:
                raise ValueError("resolved device elapsed must be positive")
        elif parsed["event_status"] == "quantized_zero":
            if parsed["batch_device_ns"] != 0.0:
                raise ValueError("quantized-zero event has nonzero duration")
        else:
            raise ValueError("unknown event_status")
        expected_per_launch = (
            parsed["batch_device_ns"] / parsed["launch_count"]
        )
        if not math.isclose(
                parsed["per_launch_device_ns"], expected_per_launch,
                rel_tol=1.0e-9, abs_tol=1.0e-6):
            raise ValueError("per-launch duration does not match batch/count")
        host_elapsed_ns = (
            parsed["host_end_ns"] - parsed["host_start_ns"]
        )
        if host_elapsed_ns <= 0:
            raise ValueError("host elapsed must be positive")
        if float(host_elapsed_ns) < parsed["batch_device_ns"]:
            raise ValueError("host elapsed is smaller than device elapsed")
        parsed["run_index"] = run_index
        parsed["host_elapsed_ns"] = host_elapsed_ns
        parsed["grid_workgroups"] = (
            parsed["grid_x"] * parsed["grid_y"] * parsed["grid_z"]
        )
        parsed["candidate_sample"] = (
            parsed["pair_id"] == PRODUCTION_FAMILY_ID
            and parsed["launch_count"] == PRODUCTION_LAUNCH_COUNT
        )
        samples.append(parsed)
    return metadata, samples


def _median_order_statistic_ci(values, alpha):
    if not values:
        raise ValueError("cannot summarize an empty sample set")
    ordered = sorted(values)
    sample_count = len(ordered)
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha) or alpha <= 0.0 or alpha >= 1.0):
        raise ValueError("median confidence alpha must be in (0, 1)")
    tail_limit = float(alpha) / 2.0
    cumulative = 0.0
    lower_rank = 1
    for successes in range(sample_count + 1):
        probability = (
            math.comb(sample_count, successes) * (0.5 ** sample_count)
        )
        cumulative += probability
        if cumulative <= tail_limit:
            lower_rank = successes + 1
        else:
            break
    if lower_rank > sample_count:
        raise AssertionError("median confidence rank is invalid")
    upper_rank = sample_count - lower_rank + 1
    if upper_rank < lower_rank:
        raise ValueError("sample set is too small for the declared CI")
    return {
        "sample_count": sample_count,
        "median_ns": statistics.median(ordered),
        "alpha": float(alpha),
        "lower_rank": lower_rank,
        "upper_rank": upper_rank,
        "median_ci_lower_ns": ordered[lower_rank - 1],
        "median_ci_upper_ns": ordered[upper_rank - 1],
        "minimum_ns": ordered[0],
        "maximum_ns": ordered[-1],
        "ci_method": "exact_distribution_free_order_statistic",
    }


def _median_ci95(values):
    interval = _median_order_statistic_ci(values, 0.05)
    return {
        "sample_count": interval["sample_count"],
        "median_ns": interval["median_ns"],
        "median_ci95_lower_ns": interval["median_ci_lower_ns"],
        "median_ci95_upper_ns": interval["median_ci_upper_ns"],
        "minimum_ns": interval["minimum_ns"],
        "maximum_ns": interval["maximum_ns"],
        "ci_method": interval["ci_method"],
    }


def build_summary(samples):
    _, production_geometries = load_production_geometry_manifest()
    production_interval_count = (
        len(production_geometries) * INDEPENDENT_RUNS
    )
    production_interval_alpha = (
        PRODUCTION_FAMILYWISE_ERROR_RATE / production_interval_count
    )
    geometry_groups = {}
    for sample in samples:
        key = (
            sample["run_index"], sample["pair_id"], sample["geometry_id"],
            sample["grid_x"], sample["grid_y"], sample["grid_z"],
            sample["block_x"], sample["block_y"], sample["block_z"],
            sample["launch_count"],
        )
        if key not in geometry_groups:
            geometry_groups[key] = []
        geometry_groups[key].append(sample)
    geometry_identities = {}
    production_familywise_intervals = {}
    for key, group_samples in geometry_groups.items():
        identity = key[1:]
        if identity not in geometry_identities:
            geometry_identities[identity] = []
        values = [sample["per_launch_device_ns"] for sample in group_samples]
        run_statistics = _median_ci95(values)
        run_statistics["run_index"] = key[0]
        run_statistics["quantized_zero_count"] = sum(
            1 for sample in group_samples
            if sample["event_status"] == "quantized_zero"
        )
        geometry_identities[identity].append(run_statistics)
        if (key[1] == PRODUCTION_FAMILY_ID
                and key[9] == PRODUCTION_LAUNCH_COUNT):
            production_familywise_intervals[(key[2], key[0])] = (
                _median_order_statistic_ci(
                    values, production_interval_alpha,
                )
            )
    geometry_statistics = []
    compatibility_failures = []
    for identity in sorted(geometry_identities):
        runs = sorted(
            geometry_identities[identity], key=lambda row: row["run_index"],
        )
        if len(runs) != INDEPENDENT_RUNS:
            raise ValueError("geometry is missing an independent run")
        overlap_lower = max(row["median_ci95_lower_ns"] for row in runs)
        overlap_upper = min(row["median_ci95_upper_ns"] for row in runs)
        if overlap_lower <= overlap_upper:
            compatibility_status = "pass"
        else:
            compatibility_status = "fail"
            compatibility_failures.append(
                f"geometry:{identity[0]}:launch_count={identity[8]}"
            )
        geometry_statistics.append({
            "pair_id": identity[0],
            "geometry_id": identity[1],
            "grid_x": identity[2],
            "grid_y": identity[3],
            "grid_z": identity[4],
            "block_x": identity[5],
            "block_y": identity[6],
            "block_z": identity[7],
            "launch_count": identity[8],
            "runs": runs,
            "cross_run_ci_compatibility": {
                "status": compatibility_status,
                "method": "descriptive_common_intersection_of_run_median_ci95",
                "intersection_lower_ns": overlap_lower,
                "intersection_upper_ns": overlap_upper,
            },
        })

    sample_index = {}
    for sample in samples:
        key = (
            sample["run_index"], sample["trial"], sample["pair_id"],
            sample["launch_count"], sample["geometry_id"],
        )
        if key in sample_index:
            raise ValueError("duplicate paired sample identity")
        sample_index[key] = sample
    paired_statistics = []
    for pair in PAIRS:
        for launch_count in pair.launch_counts:
            if not _launch_count_allowed(pair, launch_count):
                continue
            runs = []
            for run_index in range(1, INDEPENDENT_RUNS + 1):
                differences = []
                for trial in range(1, TRIALS_PER_RUN + 1):
                    prefix = (
                        run_index, trial, pair.pair_id, launch_count,
                    )
                    axis_sample = sample_index[
                        prefix + (pair.axis_geometry.geometry_id,)
                    ]
                    reference_sample = sample_index[
                        prefix + (pair.reference_geometry.geometry_id,)
                    ]
                    differences.append(
                        axis_sample["per_launch_device_ns"]
                        - reference_sample["per_launch_device_ns"]
                    )
                run_statistics = _median_ci95(differences)
                run_statistics["run_index"] = run_index
                runs.append(run_statistics)
            overlap_lower = max(
                row["median_ci95_lower_ns"] for row in runs
            )
            overlap_upper = min(
                row["median_ci95_upper_ns"] for row in runs
            )
            if overlap_lower <= overlap_upper:
                compatibility_status = "pass"
            else:
                compatibility_status = "fail"
                compatibility_failures.append(
                    f"pair:{pair.pair_id}:launch_count={launch_count}"
                )
            paired_statistics.append({
                "pair_id": pair.pair_id,
                "axis_geometry_id": pair.axis_geometry.geometry_id,
                "reference_geometry_id": (
                    pair.reference_geometry.geometry_id
                ),
                "launch_count": launch_count,
                "difference_definition": "axis_ns_minus_reference_ns",
                "runs": runs,
                "cross_run_ci_compatibility": {
                    "status": compatibility_status,
                    "method": "descriptive_common_intersection_of_run_median_ci95",
                    "intersection_lower_ns": overlap_lower,
                    "intersection_upper_ns": overlap_upper,
                },
            })
    production_familywise_geometries = []
    production_familywise_failures = []
    familywise_rank_pairs = set()
    for geometry in production_geometries:
        intervals = []
        for run_index in range(1, INDEPENDENT_RUNS + 1):
            interval = production_familywise_intervals[
                (geometry.geometry_id, run_index)
            ]
            familywise_rank_pairs.add((
                interval["lower_rank"], interval["upper_rank"],
            ))
            intervals.append({
                "run_index": run_index,
                "sample_count": interval["sample_count"],
                "median_ns": interval["median_ns"],
                "median_ci_lower_ns": interval["median_ci_lower_ns"],
                "median_ci_upper_ns": interval["median_ci_upper_ns"],
                "minimum_ns": interval["minimum_ns"],
                "maximum_ns": interval["maximum_ns"],
            })
        overlap_lower = max(
            row["median_ci_lower_ns"] for row in intervals
        )
        overlap_upper = min(
            row["median_ci_upper_ns"] for row in intervals
        )
        if overlap_lower <= overlap_upper:
            status = "pass"
        else:
            status = "fail"
            production_familywise_failures.append(geometry.geometry_id)
        production_familywise_geometries.append({
            "geometry_id": geometry.geometry_id,
            "status": status,
            "intersection_lower_ns": overlap_lower,
            "intersection_upper_ns": overlap_upper,
            "runs": intervals,
        })
    if len(familywise_rank_pairs) != 1:
        raise AssertionError(
            "production simultaneous intervals use inconsistent ranks"
        )
    familywise_lower_rank, familywise_upper_rank = next(
        iter(familywise_rank_pairs)
    )
    if compatibility_failures:
        overall_compatibility = "fail"
    else:
        overall_compatibility = "pass"
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "summary_contract": (
            "predeclared raw medians and exact distribution-free median "
            "confidence intervals; no regression, smoothing, interpolation, "
            "isotonic transform, or evaluation-error selection"
        ),
        "geometry_statistics": geometry_statistics,
        "paired_difference_statistics": paired_statistics,
        "cross_run_ci_compatibility_report": {
            "status": overall_compatibility,
            "method": "descriptive_only_not_an_effect_size_or_eligibility_gate",
            "failures": compatibility_failures,
        },
        "production_familywise_cross_run_compatibility_report": {
            "status": (
                "pass" if not production_familywise_failures else "fail"
            ),
            "method": (
                "bonferroni_simultaneous_exact_distribution_free_"
                "median_intervals"
            ),
            "familywise_error_rate": PRODUCTION_FAMILYWISE_ERROR_RATE,
            "interval_count": production_interval_count,
            "per_interval_alpha": production_interval_alpha,
            "lower_rank": familywise_lower_rank,
            "upper_rank": familywise_upper_rank,
            "failures": production_familywise_failures,
            "geometries": production_familywise_geometries,
        },
    }


def build_candidate(summary, telemetry_gate):
    geometry_manifest, production_geometries = (
        load_production_geometry_manifest()
    )
    entries = []
    quantized_zero_count = 0
    familywise_report = summary[
        "production_familywise_cross_run_compatibility_report"
    ]
    compatibility_failures = list(familywise_report["failures"])
    for row in summary["geometry_statistics"]:
        if (row["pair_id"] == PRODUCTION_FAMILY_ID
                and row["launch_count"] == PRODUCTION_LAUNCH_COUNT):
            quantized_zero_count += sum(
                run["quantized_zero_count"] for run in row["runs"]
            )
            entries.append({
                "pair_id": row["pair_id"],
                "geometry_id": row["geometry_id"],
                "grid_x": row["grid_x"],
                "grid_y": row["grid_y"],
                "grid_z": row["grid_z"],
                "block_x": row["block_x"],
                "block_y": row["block_y"],
                "block_z": row["block_z"],
                "launch_count": 1,
                "duration_ns": statistics.median(
                    run["median_ns"] for run in row["runs"]),
                "runs": row["runs"],
                "cross_run_ci_compatibility": (
                    row["cross_run_ci_compatibility"]
                ),
            })
    entries.sort(key=lambda row: (
        row["grid_x"], row["grid_y"], row["grid_z"],
        row["block_x"], row["block_y"], row["block_z"],
    ))
    if len(entries) != len(production_geometries):
        raise ValueError(
            "candidate does not contain the production geometry sweep")
    hbm_gate = telemetry_gate["hbm_pollution_assessment"]
    if (not isinstance(hbm_gate, dict)
            or set(hbm_gate) != {
                "status", "candidate_eligible", "threshold_mbytes_s",
                "observed_max_mbytes_s", "threshold_fraction_of_nominal_hbm",
                "nominal_hbm_mbytes_s", "physical_basis",
            }):
        raise ValueError("telemetry HBM pollution gate schema is invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "status": "negative_evidence_only",
        "geometry_policy": "exact_grid_and_block_only",
        "family_id": PRODUCTION_FAMILY_ID,
        "geometry_manifest": geometry_manifest,
        "duration_unit": "ns_per_launch",
        "duration_aggregation": "median_of_three_run_medians",
        "entry_count": len(entries),
        "hw_spec_eligibility": "not_hw_spec_eligible",
        "production_cross_run_compatibility": {
            "status": "pass" if not compatibility_failures else "fail",
            "method": familywise_report["method"],
            "familywise_error_rate": familywise_report[
                "familywise_error_rate"
            ],
            "interval_count": familywise_report["interval_count"],
            "per_interval_alpha": familywise_report[
                "per_interval_alpha"
            ],
            "lower_rank": familywise_report["lower_rank"],
            "upper_rank": familywise_report["upper_rank"],
            "failures": compatibility_failures,
        },
        "axis_difference_physical_gate": {
            "status": "not_applicable",
            "candidate_eligible": True,
            "reason": (
                "each full launch geometry has its own exact measurement; "
                "no axis equivalence or flattened lookup is asserted"
            ),
        },
        "hbm_pollution_gate": hbm_gate,
        "single_launch_quantized_zero_count": quantized_zero_count,
        "publication": "retained_negative_evidence_not_hw_spec_input",
        "entries": entries,
    }


def _write_samples(output_dir, samples):
    csv_path = output_dir / "samples.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        for sample in samples:
            writer.writerow({name: sample[name] for name in SAMPLE_FIELDS})
    jsonl_path = output_dir / "samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
    return {
        "samples_csv_sha256": _sha256(csv_path),
        "samples_jsonl_sha256": _sha256(jsonl_path),
    }


def _load_json_object(path, label):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


_PRODUCTION_GEOMETRY_CACHE = None


def _source_matches_payload_or_lfs_oid(path, payload_sha256):
    if _sha256(path) == payload_sha256:
        return True
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return False
    if len(lines) != 3:
        return False
    if lines[0] != "version https://git-lfs.github.com/spec/v1":
        return False
    if lines[1] != f"oid sha256:{payload_sha256}":
        return False
    size_match = re.fullmatch(r"size ([1-9][0-9]*)", lines[2])
    return size_match is not None


def load_production_geometry_manifest():
    """Load the versioned, duration-free production coordinate set."""
    global _PRODUCTION_GEOMETRY_CACHE
    if _PRODUCTION_GEOMETRY_CACHE is not None:
        return _PRODUCTION_GEOMETRY_CACHE
    payload = _load_json_object(
        PRODUCTION_GEOMETRY_MANIFEST_PATH,
        "production launch geometry manifest",
    )
    required = {
        "schema_version", "hardware_id", "source_contract",
        "duration_fields_consumed", "source_files", "total_case_count",
        "total_kernel_launch_count", "geometry_count", "geometry_policy",
        "geometries",
    }
    if set(payload) != required:
        raise ValueError("production launch geometry manifest schema is invalid")
    if (payload["schema_version"] != 1
            or payload["hardware_id"] != HARDWARE_ID
            or payload["source_contract"]
            != "canonical kernel_dataset Metadata Grid Size and Block Size only"
            or payload["duration_fields_consumed"] is not False
            or payload["geometry_policy"] != "exact_grid_and_block_only"):
        raise ValueError("production launch geometry manifest contract is invalid")
    source_files = payload["source_files"]
    if not isinstance(source_files, list) or len(source_files) != 4:
        raise ValueError("production geometry source file set is invalid")
    expected_families = {"flash_attn", "fused_moe", "gemm", "vector"}
    observed_families = set()
    total_cases = 0
    total_launches = 0
    source_fields = {
        "family", "path", "sha256", "case_count", "kernel_launch_count",
    }
    for source in source_files:
        if not isinstance(source, dict) or set(source) != source_fields:
            raise ValueError("production geometry source entry is invalid")
        family = source["family"]
        expected_path = f"kernel_dataset/{HARDWARE_ID}/{family}.json.zst"
        if (family not in expected_families
                or family in observed_families
                or source["path"] != expected_path):
            raise ValueError("production geometry source identity is invalid")
        observed_families.add(family)
        source_path = (REPO_ROOT / source["path"]).resolve()
        if (not source_path.is_file()
                or not _source_matches_payload_or_lfs_oid(
                    source_path, source["sha256"])):
            raise ValueError("production geometry source hash is invalid")
        for count_name in ("case_count", "kernel_launch_count"):
            count = source[count_name]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(
                    f"production geometry {count_name} must be positive"
                )
        total_cases += source["case_count"]
        total_launches += source["kernel_launch_count"]
    if observed_families != expected_families:
        raise ValueError("production geometry source families are incomplete")
    if (payload["total_case_count"] != total_cases
            or payload["total_kernel_launch_count"] != total_launches):
        raise ValueError("production geometry source totals are inconsistent")

    rows = payload["geometries"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("production geometry set must be a non-empty list")
    geometries = []
    identities = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"grid", "block"}:
            raise ValueError("production geometry entry schema is invalid")
        dimensions = []
        for name in ("grid", "block"):
            value = row[name]
            if (not isinstance(value, list) or len(value) != 3
                    or any(isinstance(axis, bool) or not isinstance(axis, int)
                           or axis <= 0 for axis in value)):
                raise ValueError(
                    f"production geometry {name} must be a positive dim3"
                )
            dimensions.append(tuple(value))
        grid, block = dimensions
        if math.prod(block) > 1024:
            raise ValueError("production geometry block exceeds 1024 threads")
        identity = (grid, block)
        if identity in identities:
            raise ValueError("production geometry set contains a duplicate")
        identities.add(identity)
        geometry_id = (
            f"grid_{grid[0]}x{grid[1]}x{grid[2]}_"
            f"block_{block[0]}x{block[1]}x{block[2]}"
        )
        geometries.append(Geometry(geometry_id, *grid, *block))
    if payload["geometry_count"] != len(geometries):
        raise ValueError("production geometry count is inconsistent")
    identity = {
        "path": str(PRODUCTION_GEOMETRY_MANIFEST_PATH),
        "sha256": _sha256(PRODUCTION_GEOMETRY_MANIFEST_PATH),
        "schema_version": payload["schema_version"],
        "geometry_count": payload["geometry_count"],
        "total_case_count": payload["total_case_count"],
        "total_kernel_launch_count": payload["total_kernel_launch_count"],
        "source_files": source_files,
    }
    _PRODUCTION_GEOMETRY_CACHE = identity, tuple(geometries)
    return _PRODUCTION_GEOMETRY_CACHE


def validate_source_and_build_evidence(
        baseline_commit, source_snapshot, snapshot_manifest_path,
        source_path, runner_path, compile_command_path,
        compile_stdout_path, compile_stderr_path, binary_path):
    snapshot_sha256 = _sha256(source_snapshot)
    binary_sha256 = _sha256(binary_path)
    snapshot_manifest = _load_json_object(
        snapshot_manifest_path, "source snapshot manifest",
    )
    required_snapshot_keys = {
        "schema_version", "baseline_commit", "source_snapshot_sha256",
        "benchmark_source_sha256", "runner_source_sha256",
        "telemetry_runner_source_sha256",
        "production_geometry_manifest_sha256",
    }
    if set(snapshot_manifest) != required_snapshot_keys:
        raise ValueError("source snapshot manifest schema is invalid")
    if snapshot_manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("source snapshot manifest version is invalid")
    if snapshot_manifest["baseline_commit"] != baseline_commit:
        raise ValueError("source snapshot baseline does not match capture")
    if snapshot_manifest["source_snapshot_sha256"] != snapshot_sha256:
        raise ValueError("source snapshot SHA-256 does not match manifest")
    if snapshot_manifest["benchmark_source_sha256"] != _sha256(source_path):
        raise ValueError("benchmark source does not match snapshot manifest")
    if snapshot_manifest["runner_source_sha256"] != _sha256(runner_path):
        raise ValueError("runner source does not match snapshot manifest")
    if (snapshot_manifest["telemetry_runner_source_sha256"]
            != _sha256(TELEMETRY_RUNNER_PATH)):
        raise ValueError("telemetry runner does not match snapshot manifest")
    if (snapshot_manifest["production_geometry_manifest_sha256"]
            != _sha256(PRODUCTION_GEOMETRY_MANIFEST_PATH)):
        raise ValueError(
            "production geometry manifest does not match source snapshot"
        )

    compile_command = _load_json_object(
        compile_command_path, "compile command",
    )
    required_compile_keys = {
        "schema_version", "baseline_commit", "source_snapshot_sha256",
        "binary_sha256", "returncode", "argv",
        "stdout_sha256", "stderr_sha256",
        "source_path", "source_sha256", "compiler_container_path",
        "compiler_container_sha256", "compiler_version",
        "compiler_version_argv", "compiler_version_returncode",
        "compiler_version_stdout_path", "compiler_version_stdout_sha256",
        "compiler_version_stderr_path", "compiler_version_stderr_sha256",
    }
    if set(compile_command) != required_compile_keys:
        raise ValueError("compile command schema is invalid")
    if compile_command["schema_version"] != SCHEMA_VERSION:
        raise ValueError("compile command version is invalid")
    if compile_command["baseline_commit"] != baseline_commit:
        raise ValueError("compile command baseline does not match capture")
    if compile_command["source_snapshot_sha256"] != snapshot_sha256:
        raise ValueError("compile command is not bound to source snapshot")
    if compile_command["binary_sha256"] != binary_sha256:
        raise ValueError("compile command is not bound to binary")
    if compile_command["returncode"] != 0:
        raise ValueError("compile command did not complete successfully")
    if compile_command["stdout_sha256"] != _sha256(compile_stdout_path):
        raise ValueError("compile stdout is not bound to compile command")
    if compile_command["stderr_sha256"] != _sha256(compile_stderr_path):
        raise ValueError("compile stderr is not bound to compile command")
    if (compile_command["source_path"] != str(source_path)
            or compile_command["source_sha256"] != _sha256(source_path)):
        raise ValueError("compile command is not bound to benchmark source")
    if (compile_command["compiler_container_path"]
            != MXCC_CONTAINER_PATH
            or SHA256_PATTERN.fullmatch(
                compile_command["compiler_container_sha256"]
            ) is None):
        raise ValueError("compile command is not bound to audited mxcc")
    version_argv = compile_command["compiler_version_argv"]
    expected_version_argv = [
        str(DOCKER_PATH), "exec", AUDITED_CONTAINER_ID,
        MXCC_CONTAINER_PATH, "--version",
    ]
    if (version_argv != expected_version_argv
            or compile_command["compiler_version_returncode"] != 0):
        raise ValueError("mxcc version command is invalid")
    version_stdout_path = Path(
        compile_command["compiler_version_stdout_path"]
    ).resolve()
    version_stderr_path = Path(
        compile_command["compiler_version_stderr_path"]
    ).resolve()
    if (not version_stdout_path.is_file()
            or not version_stderr_path.is_file()
            or _sha256(version_stdout_path)
            != compile_command["compiler_version_stdout_sha256"]
            or _sha256(version_stderr_path)
            != compile_command["compiler_version_stderr_sha256"]):
        raise ValueError("mxcc version raw evidence is invalid")
    compiler_version = compile_command["compiler_version"]
    if (not isinstance(compiler_version, str) or not compiler_version
            or compiler_version != version_stdout_path.read_text(
                encoding="utf-8",
            ).strip()):
        raise ValueError("mxcc normalized version is not present in raw output")
    argv = compile_command["argv"]
    if (not isinstance(argv, list) or not argv
            or any(not isinstance(value, str) or not value for value in argv)):
        raise ValueError("compile command argv must be non-empty strings")
    expected_argv = [
        str(DOCKER_PATH), "exec", AUDITED_CONTAINER_ID,
        MXCC_CONTAINER_PATH, *MXCC_CANONICAL_FLAGS,
        str(source_path), "-o", str(binary_path),
    ]
    if argv != expected_argv:
        raise ValueError(
            "compile argv does not match the canonical container build"
        )
    return {
        "source_files": {
            "benchmark_source_path": str(source_path),
            "benchmark_source_sha256": _sha256(source_path),
            "runner_source_path": str(runner_path),
            "runner_source_sha256": _sha256(runner_path),
            "telemetry_runner_source_path": str(TELEMETRY_RUNNER_PATH),
            "telemetry_runner_source_sha256": _sha256(
                TELEMETRY_RUNNER_PATH,
            ),
            "production_geometry_manifest_path": str(
                PRODUCTION_GEOMETRY_MANIFEST_PATH
            ),
            "production_geometry_manifest_sha256": _sha256(
                PRODUCTION_GEOMETRY_MANIFEST_PATH
            ),
        },
        "source_snapshot": {
            "path": str(source_snapshot),
            "sha256": snapshot_sha256,
            "manifest_path": str(snapshot_manifest_path),
            "manifest_sha256": _sha256(snapshot_manifest_path),
        },
        "compile": {
            "command_path": str(compile_command_path),
            "command_sha256": _sha256(compile_command_path),
            "argv": argv,
            "returncode": 0,
            "stdout_path": str(compile_stdout_path),
            "stdout_sha256": _sha256(compile_stdout_path),
            "stderr_path": str(compile_stderr_path),
            "stderr_sha256": _sha256(compile_stderr_path),
            "source_path": str(source_path),
            "source_sha256": _sha256(source_path),
            "compiler_container_path": MXCC_CONTAINER_PATH,
            "compiler_container_sha256": compile_command[
                "compiler_container_sha256"
            ],
            "compiler_version": compiler_version,
            "compiler_version_argv": version_argv,
            "compiler_version_stdout_path": str(version_stdout_path),
            "compiler_version_stdout_sha256": _sha256(version_stdout_path),
            "compiler_version_stderr_path": str(version_stderr_path),
            "compiler_version_stderr_sha256": _sha256(version_stderr_path),
        },
        "binary": {"path": str(binary_path), "sha256": binary_sha256},
    }


def validate_empty_kernel_disassembly(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    header_pattern = re.compile(
        r"^\s*[0-9a-fA-F]+\s+<[^>]*empty_kernel[^>]*>:\s*$"
    )
    any_header_pattern = re.compile(r"^\s*[0-9a-fA-F]+\s+<[^>]+>:\s*$")
    endk_pattern = re.compile(
        r"^\s*[0-9a-fA-F]+:\s+"
        r"(?:(?:0x)?[0-9a-fA-F]{2,16}\s+)*endk\s*$"
    )
    matching_headers = [index for index, line in enumerate(lines)
                        if header_pattern.fullmatch(line) is not None]
    if len(matching_headers) != 1:
        raise ValueError("disassembly must contain one empty_kernel symbol")
    instructions = []
    for line in lines[matching_headers[0] + 1:]:
        if any_header_pattern.fullmatch(line) is not None:
            break
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "...":
            continue
        if endk_pattern.fullmatch(stripped) is None:
            raise ValueError("empty_kernel contains an instruction other than endk")
        instructions.append("endk")
    if instructions != ["endk"]:
        raise ValueError("empty_kernel must contain exactly one endk")


def _run_retained_command(argv, stdout_path, stderr_path):
    try:
        result = subprocess.run(
            argv, check=False, capture_output=True,
            timeout=TELEMETRY_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_bytes(b"" if exc.stdout is None else exc.stdout)
        stderr_path.write_bytes(b"" if exc.stderr is None else exc.stderr)
        raise
    stdout_path.write_bytes(result.stdout)
    stderr_path.write_bytes(result.stderr)
    return result


def probe_container_tool(container_id, tool_path, label, output_dir):
    sha_stdout_path = output_dir / f"{label}.sha256.stdout.txt"
    sha_stderr_path = output_dir / f"{label}.sha256.stderr.txt"
    sha_argv = [
        str(DOCKER_PATH), "exec", container_id,
        CONTAINER_SHA256SUM_PATH, tool_path,
    ]
    sha_result = _run_retained_command(
        sha_argv, sha_stdout_path, sha_stderr_path,
    )
    if sha_result.returncode != 0:
        raise RuntimeError(f"container {label} SHA-256 probe failed")
    try:
        sha_fields = sha_result.stdout.decode("utf-8").strip().split()
    except UnicodeError as exc:
        raise ValueError(f"container {label} SHA-256 output is invalid") from exc
    if (len(sha_fields) != 2
            or SHA256_PATTERN.fullmatch(sha_fields[0]) is None
            or sha_fields[1] != tool_path):
        raise ValueError(f"container {label} SHA-256 output is invalid")

    version_stdout_path = output_dir / f"{label}.version.stdout.txt"
    version_stderr_path = output_dir / f"{label}.version.stderr.txt"
    version_argv = [
        str(DOCKER_PATH), "exec", container_id, tool_path, "--version",
    ]
    version_result = _run_retained_command(
        version_argv, version_stdout_path, version_stderr_path,
    )
    if version_result.returncode != 0:
        raise RuntimeError(f"container {label} version probe failed")
    try:
        version = version_result.stdout.decode("utf-8").strip()
    except UnicodeError as exc:
        raise ValueError(f"container {label} version output is invalid") from exc
    if not version:
        raise ValueError(f"container {label} version output is empty")
    return {
        "container_id": container_id,
        "container_path": tool_path,
        "sha256": sha_fields[0],
        "version": version,
        "sha256_argv": sha_argv,
        "sha256_returncode": sha_result.returncode,
        "sha256_stdout_path": str(sha_stdout_path),
        "sha256_stdout_sha256": _sha256(sha_stdout_path),
        "sha256_stderr_path": str(sha_stderr_path),
        "sha256_stderr_sha256": _sha256(sha_stderr_path),
        "version_argv": version_argv,
        "version_returncode": version_result.returncode,
        "version_stdout_path": str(version_stdout_path),
        "version_stdout_sha256": _sha256(version_stdout_path),
        "version_stderr_path": str(version_stderr_path),
        "version_stderr_sha256": _sha256(version_stderr_path),
    }


def container_tool_identity(probe):
    return {
        "container_id": probe["container_id"],
        "container_path": probe["container_path"],
        "sha256": probe["sha256"],
        "version": probe["version"],
    }


def container_probe_artifact_paths(probe):
    return tuple(Path(probe[key]).resolve() for key in (
        "sha256_stdout_path", "sha256_stderr_path",
        "version_stdout_path", "version_stderr_path",
    ))


def validate_compiler_startup_probe(compile_evidence, probe):
    if (probe["container_id"] != AUDITED_CONTAINER_ID
            or probe["container_path"] != MXCC_CONTAINER_PATH
            or probe["sha256"]
            != compile_evidence["compiler_container_sha256"]
            or probe["version"] != compile_evidence["compiler_version"]):
        raise ValueError("live container mxcc differs from compile evidence")


def run_and_validate_mxobjdump(
        binary_path, output_dir, container_id):
    probe = probe_container_tool(
        container_id, MXOBJDUMP_CONTAINER_PATH, "mxobjdump", output_dir,
    )
    stdout_path = output_dir / "mxobjdump.stdout.txt"
    stderr_path = output_dir / "mxobjdump.stderr.txt"
    command = [
        str(DOCKER_PATH), "exec", container_id,
        MXOBJDUMP_CONTAINER_PATH, "--print-code", str(binary_path),
    ]
    result = _run_retained_command(command, stdout_path, stderr_path)
    if result.returncode != 0:
        raise RuntimeError("container mxobjdump disassembly failed")
    validate_empty_kernel_disassembly(stdout_path)
    return {
        "tool_probe": probe,
        "argv": command,
        "returncode": result.returncode,
        "stdout_path": str(stdout_path),
        "stdout_sha256": _sha256(stdout_path),
        "stderr_path": str(stderr_path),
        "stderr_sha256": _sha256(stderr_path),
        "empty_kernel_instructions": ["endk"],
    }


def validate_git_baseline(baseline_commit, runner_path):
    root_result = subprocess.run(
        ["git", "-C", str(runner_path.parent), "rev-parse", "--show-toplevel"],
        check=False, capture_output=True, text=True,
    )
    if root_result.returncode != 0:
        if "not a git repository" not in root_result.stderr:
            raise RuntimeError("cannot determine deployed Git metadata")
        return {"present": False, "head": None}
    repository_root = Path(root_result.stdout.strip()).resolve()
    head_result = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    head = head_result.stdout.strip()
    if head != baseline_commit:
        raise ValueError("deployed Git HEAD does not equal baseline commit")
    return {"present": True, "root": str(repository_root), "head": head}


def validate_deployment_manifest(
        manifest_path, inspect_output_path, baseline_commit,
        source_snapshot_sha256, device_id, device_bus_id):
    manifest = _load_json_object(manifest_path, "deployment manifest")
    required = {
        "schema_version", "baseline_commit", "source_snapshot_sha256",
        "container_name", "container_id", "image_id", "pid_mode",
        "docker_inspect_argv", "docker_inspect_stdout_sha256",
        "device_mapping",
    }
    if set(manifest) != required:
        raise ValueError("deployment manifest schema is invalid")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("deployment manifest version is invalid")
    if manifest["baseline_commit"] != baseline_commit:
        raise ValueError("deployment manifest baseline is invalid")
    if manifest["source_snapshot_sha256"] != source_snapshot_sha256:
        raise ValueError("deployment manifest source snapshot is invalid")
    container_id = manifest["container_id"]
    if (not isinstance(container_id, str)
            or CONTAINER_ID_PATTERN.fullmatch(container_id) is None
            or container_id != AUDITED_CONTAINER_ID):
        raise ValueError("deployment container ID is not audited C500 container")
    image_id = manifest["image_id"]
    if (not isinstance(image_id, str)
            or IMAGE_DIGEST_PATTERN.fullmatch(image_id) is None
            or image_id != AUDITED_IMAGE_ID):
        raise ValueError("deployment image ID is not immutable")
    if manifest["pid_mode"] != "private":
        raise ValueError("deployment manifest must normalize private PidMode")
    expected_argv = ["docker", "inspect", manifest["container_name"]]
    if manifest["docker_inspect_argv"] != expected_argv:
        raise ValueError("deployment docker inspect command is invalid")
    if _sha256(inspect_output_path) != manifest["docker_inspect_stdout_sha256"]:
        raise ValueError("docker inspect raw output SHA-256 is invalid")
    try:
        inspect_payload = json.loads(
            inspect_output_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot parse raw docker inspect output") from exc
    if not isinstance(inspect_payload, list) or len(inspect_payload) != 1:
        raise ValueError("docker inspect raw output must contain one container")
    container = inspect_payload[0]
    if not isinstance(container, dict):
        raise ValueError("docker inspect container entry is invalid")
    if (container["Id"] != container_id or container["Image"] != image_id
            or container["Name"] != "/" + manifest["container_name"]):
        raise ValueError("docker inspect identity differs from deployment manifest")
    raw_pid_mode = container["HostConfig"]["PidMode"]
    if raw_pid_mode != "":
        raise ValueError("docker inspect does not show private PidMode")
    state = container["State"]
    if (not isinstance(state, dict) or state["Running"] is not True
            or not isinstance(state["Pid"], int) or state["Pid"] <= 0):
        raise ValueError("docker inspect does not show a running container")
    mapping = manifest["device_mapping"]
    if (not isinstance(mapping, dict)
            or set(mapping) != {"container_device_id", "pci_bus_id"}
            or mapping["container_device_id"] != device_id
            or mapping["pci_bus_id"] != device_bus_id):
        raise ValueError("deployment device mapping is invalid")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "docker_inspect_stdout_path": str(inspect_output_path),
        "docker_inspect_stdout_sha256": _sha256(inspect_output_path),
        "container_id": container_id,
        "image_id": image_id,
        "pid_mode": "private",
        "container_init_host_pid": state["Pid"],
        "device_mapping": mapping,
    }


def _validate_path_hash(path_text, expected_sha256, label):
    path = Path(path_text).resolve()
    if (not path.is_file() or SHA256_PATTERN.fullmatch(expected_sha256) is None
            or _sha256(path) != expected_sha256):
        raise ValueError(f"timed smoke {label} raw artifact is invalid")
    return path


def _validate_container_probe(probe, expected_identity, label):
    required = {
        "container_id", "container_path", "sha256", "version",
        "sha256_argv", "sha256_returncode",
        "sha256_stdout_path", "sha256_stdout_sha256",
        "sha256_stderr_path", "sha256_stderr_sha256",
        "version_argv", "version_returncode",
        "version_stdout_path", "version_stdout_sha256",
        "version_stderr_path", "version_stderr_sha256",
    }
    if not isinstance(probe, dict) or set(probe) != required:
        raise ValueError(f"timed smoke {label} probe schema is invalid")
    if container_tool_identity(probe) != expected_identity:
        raise ValueError(f"timed smoke {label} identity is invalid")
    if (probe["sha256_argv"] != [
            str(DOCKER_PATH), "exec", AUDITED_CONTAINER_ID,
            CONTAINER_SHA256SUM_PATH, probe["container_path"]]
            or probe["version_argv"] != [
                str(DOCKER_PATH), "exec", AUDITED_CONTAINER_ID,
                probe["container_path"], "--version"]
            or probe["sha256_returncode"] != 0
            or probe["version_returncode"] != 0):
        raise ValueError(f"timed smoke {label} probe command is invalid")
    retained_paths = {}
    for path_key, sha_key in (
            ("sha256_stdout_path", "sha256_stdout_sha256"),
            ("sha256_stderr_path", "sha256_stderr_sha256"),
            ("version_stdout_path", "version_stdout_sha256"),
            ("version_stderr_path", "version_stderr_sha256")):
        retained_paths[path_key] = _validate_path_hash(
            probe[path_key], probe[sha_key], label,
        )
    sha_fields = retained_paths["sha256_stdout_path"].read_text(
        encoding="utf-8",
    ).strip().split()
    if (sha_fields != [probe["sha256"], probe["container_path"]]
            or retained_paths["version_stdout_path"].read_text(
                encoding="utf-8",
            ).strip() != probe["version"]):
        raise ValueError(f"timed smoke {label} probe raw content is invalid")


def _validate_smoke_evidence(smoke_evidence, current_evidence, binary_sha256):
    if (not isinstance(smoke_evidence, dict)
            or set(smoke_evidence) != {
                "source_files", "source_snapshot", "compile", "binary",
                "disassembly",
            }):
        raise ValueError("timed smoke evidence schema is invalid")
    if smoke_evidence["source_files"] != current_evidence["source_files"]:
        raise ValueError("timed smoke runner sources differ from full capture")
    source_files = smoke_evidence["source_files"]
    if set(source_files) != {
            "benchmark_source_path", "benchmark_source_sha256",
            "runner_source_path", "runner_source_sha256",
            "telemetry_runner_source_path", "telemetry_runner_source_sha256",
            "production_geometry_manifest_path",
            "production_geometry_manifest_sha256"}:
        raise ValueError("timed smoke source file schema is invalid")
    for path_key, sha_key in (
            ("benchmark_source_path", "benchmark_source_sha256"),
            ("runner_source_path", "runner_source_sha256"),
            ("telemetry_runner_source_path",
             "telemetry_runner_source_sha256"),
            ("production_geometry_manifest_path",
             "production_geometry_manifest_sha256")):
        _validate_path_hash(source_files[path_key], source_files[sha_key], path_key)
    source_snapshot = smoke_evidence["source_snapshot"]
    if set(source_snapshot) != {
            "path", "sha256", "manifest_path", "manifest_sha256"}:
        raise ValueError("timed smoke source snapshot schema is invalid")
    _validate_path_hash(
        source_snapshot["path"], source_snapshot["sha256"], "source snapshot",
    )
    _validate_path_hash(
        source_snapshot["manifest_path"],
        source_snapshot["manifest_sha256"], "snapshot manifest",
    )
    binary = smoke_evidence["binary"]
    if (set(binary) != {"path", "sha256"}
            or binary["sha256"] != binary_sha256):
        raise ValueError("timed smoke binary evidence is invalid")
    _validate_path_hash(binary["path"], binary["sha256"], "binary")

    compile_evidence = smoke_evidence["compile"]
    required_compile = {
        "command_path", "command_sha256", "argv", "returncode",
        "stdout_path", "stdout_sha256", "stderr_path", "stderr_sha256",
        "source_path", "source_sha256", "compiler_container_path",
        "compiler_container_sha256", "compiler_version",
        "compiler_version_argv", "compiler_version_stdout_path",
        "compiler_version_stdout_sha256", "compiler_version_stderr_path",
        "compiler_version_stderr_sha256", "startup_probe",
    }
    if (not isinstance(compile_evidence, dict)
            or set(compile_evidence) != required_compile
            or compile_evidence["returncode"] != 0):
        raise ValueError("timed smoke compile evidence schema is invalid")
    current_compile = current_evidence["compile"]
    for key in (
            "command_sha256", "argv", "returncode", "stdout_sha256",
            "stderr_sha256", "source_sha256", "compiler_container_path",
            "compiler_container_sha256", "compiler_version",
            "compiler_version_argv", "compiler_version_stdout_sha256",
            "compiler_version_stderr_sha256"):
        if compile_evidence[key] != current_compile[key]:
            raise ValueError("timed smoke compile evidence differs from full capture")
    for path_key, sha_key in (
            ("command_path", "command_sha256"),
            ("stdout_path", "stdout_sha256"),
            ("stderr_path", "stderr_sha256"),
            ("source_path", "source_sha256"),
            ("compiler_version_stdout_path",
             "compiler_version_stdout_sha256"),
            ("compiler_version_stderr_path",
             "compiler_version_stderr_sha256")):
        _validate_path_hash(
            compile_evidence[path_key], compile_evidence[sha_key], "compile",
        )
    _validate_container_probe(
        compile_evidence["startup_probe"],
        container_tool_identity(
            current_evidence["compile"]["startup_probe"]
        ), "mxcc",
    )

    disassembly = smoke_evidence["disassembly"]
    if (not isinstance(disassembly, dict) or set(disassembly) != {
            "tool_probe", "argv", "returncode", "stdout_path",
            "stdout_sha256", "stderr_path", "stderr_sha256",
            "empty_kernel_instructions",
            } or disassembly["returncode"] != 0
            or disassembly["empty_kernel_instructions"] != ["endk"]):
        raise ValueError("timed smoke disassembly evidence schema is invalid")
    _validate_container_probe(
        disassembly["tool_probe"],
        container_tool_identity(current_evidence["disassembly"]["tool_probe"]),
        "mxobjdump",
    )
    for path_key, sha_key in (
            ("stdout_path", "stdout_sha256"),
            ("stderr_path", "stderr_sha256")):
        _validate_path_hash(
            disassembly[path_key], disassembly[sha_key], "mxobjdump",
        )
    expected_disassembly_argv = [
        str(DOCKER_PATH), "exec", AUDITED_CONTAINER_ID,
        MXOBJDUMP_CONTAINER_PATH, "--print-code", binary["path"],
    ]
    if disassembly["argv"] != expected_disassembly_argv:
        raise ValueError("timed smoke mxobjdump argv is invalid")
    validate_empty_kernel_disassembly(Path(disassembly["stdout_path"]))


def build_geometry_probe_evidence(raw_path, execution, evidence):
    validation = parse_geometry_probe(raw_path.read_bytes())
    return {
        "schema_version": SCHEMA_VERSION,
        "raw_protocol_version": GEOMETRY_PROBE_PROTOCOL_VERSION,
        "status": "verified",
        "probe_mode": "device_visible_launch_geometry",
        "timing": "none",
        "hw_spec_eligibility": "not_hw_spec_eligible",
        "source": {
            "path": evidence["source_files"]["benchmark_source_path"],
            "sha256": evidence["source_files"][
                "benchmark_source_sha256"
            ],
        },
        "binary": {
            "path": evidence["binary"]["path"],
            "sha256": evidence["binary"]["sha256"],
        },
        "raw": {
            "path": str(raw_path), "sha256": _sha256(raw_path),
        },
        "execution": execution,
        "validation": validation,
    }


def validate_geometry_probe_evidence(
        probe, current_evidence, binary_sha256, deployment_provenance):
    if (not isinstance(probe, dict) or set(probe) != {
            "schema_version", "raw_protocol_version", "status",
            "probe_mode", "timing",
            "hw_spec_eligibility", "source", "binary", "raw",
            "execution", "validation"}):
        raise ValueError("geometry probe evidence schema is invalid")
    if (probe["schema_version"] != SCHEMA_VERSION
            or probe["raw_protocol_version"]
            != GEOMETRY_PROBE_PROTOCOL_VERSION
            or probe["status"] != "verified"
            or probe["probe_mode"] != "device_visible_launch_geometry"
            or probe["timing"] != "none"
            or probe["hw_spec_eligibility"] != "not_hw_spec_eligible"):
        raise ValueError("geometry probe disposition is invalid")
    expected_source = {
        "path": current_evidence["source_files"]["benchmark_source_path"],
        "sha256": current_evidence["source_files"][
            "benchmark_source_sha256"
        ],
    }
    if probe["source"] != expected_source:
        raise ValueError("geometry probe source identity differs from timing")
    _validate_path_hash(
        probe["source"]["path"], probe["source"]["sha256"],
        "geometry probe source",
    )
    expected_binary = {
        "path": current_evidence["binary"]["path"],
        "sha256": binary_sha256,
    }
    if probe["binary"] != expected_binary:
        raise ValueError("geometry probe binary identity differs from timing")
    _validate_path_hash(
        probe["binary"]["path"], probe["binary"]["sha256"],
        "geometry probe binary",
    )
    if (not isinstance(probe["raw"], dict)
            or set(probe["raw"]) != {"path", "sha256"}):
        raise ValueError("geometry probe raw provenance is invalid")
    raw_path = _validate_path_hash(
        probe["raw"]["path"], probe["raw"]["sha256"],
        "geometry probe raw",
    )
    execution = probe["execution"]
    if (not isinstance(execution, dict)
            or set(execution) != CAPTURE_RUN_RECORD_FIELDS):
        raise ValueError("geometry probe execution record schema is invalid")
    if (deployment_provenance["container_id"] != AUDITED_CONTAINER_ID
            or deployment_provenance["device_mapping"][
                "container_device_id"] < 0):
        raise ValueError("geometry probe deployment identity is invalid")
    expected_command = _capture_command(
        deployment_provenance["container_id"],
        Path(execution["container_pid_evidence_path"]),
        Path(probe["binary"]["path"]),
        deployment_provenance["device_mapping"]["container_device_id"],
        ("--geometry-probe",),
    )
    if (execution["command"] != expected_command
            or execution["run_index"] != 0
            or execution["returncode"] != 0
            or execution["status"] != "complete"
            or execution["stdout_path"] != str(raw_path)
            or execution["stdout_sha256"] != probe["raw"]["sha256"]):
        raise ValueError("geometry probe execution provenance is invalid")
    validate_run_record_artifacts([execution])
    per_run_manifest = _load_json_object(
        raw_path.parent / "run.json", "geometry probe execution manifest",
    )
    if per_run_manifest != execution:
        raise ValueError("geometry probe execution manifest changed")
    validation = parse_geometry_probe(raw_path.read_bytes())
    if probe["validation"] != validation:
        raise ValueError("geometry probe validation differs from raw artifact")
    return validation


def validate_smoke_manifest(
        smoke_manifest_path, baseline_commit, binary_sha256,
        deployment_provenance, current_evidence, current_toolchain):
    resolved = smoke_manifest_path.resolve()
    if (resolved.name != "run.json"
            or resolved.parent.parent != PROFILER_OUTPUT_ROOT.resolve()):
        raise ValueError("timed smoke manifest is outside canonical profiler output")
    smoke = _load_json_object(resolved, "timed smoke manifest")
    required = {
        "schema_version", "status", "capture_mode", "timed_smoke",
        "hardware_id", "runner_pid", "runner_command", "baseline_commit",
        "git_provenance", "evidence", "toolchain", "deployment",
        "geometry_probe",
        "telemetry_runner", "device_and_clocks", "matrix_sha256",
        "independent_runs", "cleanup", "sample_count", "artifacts",
    }
    if set(smoke) != required:
        raise ValueError("timed smoke manifest schema is invalid")
    if (smoke["schema_version"] != SCHEMA_VERSION
            or smoke["status"] != "complete"
            or smoke["capture_mode"] != "timed_smoke"
            or smoke["hardware_id"] != HARDWARE_ID
            or smoke["sample_count"] != SMOKE_SAMPLE_COUNT):
        raise ValueError("timed smoke manifest is not a successful smoke")
    if (not isinstance(smoke["runner_pid"], int)
            or smoke["runner_pid"] <= 0
            or not isinstance(smoke["runner_command"], list)
            or not smoke["runner_command"]
            or any(not isinstance(value, str) or not value
                   for value in smoke["runner_command"])
            or not isinstance(smoke["git_provenance"], dict)
            or not smoke["git_provenance"]
            or not isinstance(smoke["cleanup"], list)
            or not smoke["cleanup"]
            or any(not isinstance(record, dict) or not record
                   for record in smoke["cleanup"])):
        raise ValueError("timed smoke execution provenance is invalid")
    if smoke["baseline_commit"] != baseline_commit:
        raise ValueError("timed smoke baseline differs from full capture")
    _validate_smoke_evidence(
        smoke["evidence"], current_evidence, binary_sha256,
    )
    validate_geometry_probe_evidence(
        smoke["geometry_probe"], current_evidence, binary_sha256,
        deployment_provenance,
    )
    if (smoke["matrix_sha256"]
            != _payload_sha256(build_smoke_matrix(
                SMOKE_PAIR_ID, SMOKE_LAUNCH_COUNT,
            ))):
        raise ValueError("timed smoke matrix is not the canonical frozen smoke")
    telemetry_runner = smoke["telemetry_runner"]
    if (not isinstance(telemetry_runner, dict)
            or set(telemetry_runner) != {
                "path", "sha256", "executable", "python_executable",
                "python_executable_sha256",
            } or telemetry_runner["sha256"]
            != current_evidence["source_files"]
            ["telemetry_runner_source_sha256"]
            or telemetry_runner["executable"] is not True):
        raise ValueError("timed smoke telemetry runner hash is invalid")
    _validate_path_hash(
        telemetry_runner["path"], telemetry_runner["sha256"],
        "telemetry runner",
    )
    _validate_path_hash(
        telemetry_runner["python_executable"],
        telemetry_runner["python_executable_sha256"], "python executable",
    )
    if smoke["toolchain"] != current_toolchain:
        raise ValueError("timed smoke toolchain differs from full capture")
    if smoke["deployment"] != deployment_provenance:
        raise ValueError("timed smoke deployment differs from full capture")
    _validate_path_hash(
        smoke["deployment"]["manifest_path"],
        smoke["deployment"]["manifest_sha256"], "deployment manifest",
    )
    _validate_path_hash(
        smoke["deployment"]["docker_inspect_stdout_path"],
        smoke["deployment"]["docker_inspect_stdout_sha256"],
        "docker inspect",
    )
    if (not isinstance(smoke["timed_smoke"], dict)
            or set(smoke["timed_smoke"]) != {
                "status", "eligibility", "pair_id", "launch_count",
                "trials",
            } or smoke["timed_smoke"]["status"] != "not_candidate"
            or smoke["timed_smoke"]["eligibility"] != "not_hw_spec_eligible"
            or smoke["timed_smoke"]["pair_id"] != SMOKE_PAIR_ID
            or smoke["timed_smoke"]["launch_count"] != SMOKE_LAUNCH_COUNT
            or smoke["timed_smoke"]["trials"] != SMOKE_TRIALS):
        raise ValueError("timed smoke eligibility marker is invalid")

    run_records = smoke["independent_runs"]
    if (not isinstance(run_records, list) or len(run_records) != 1
            or not isinstance(run_records[0], dict)
            or set(run_records[0]) != CAPTURE_RUN_RECORD_FIELDS
            or run_records[0]["status"] != "complete"
            or run_records[0]["run_index"] != 1
            or run_records[0]["returncode"] != 0
            or run_records[0]["signal"] is not None
            or run_records[0]["pid_identity_source"] != "initial"
            or not isinstance(run_records[0]["child_pid_identity"], dict)
            or set(run_records[0]["child_pid_identity"]) != {
                "container_pid", "starttime_ticks"}
            or run_records[0]["child_pid_identity"]["container_pid"]
            != run_records[0]["child_container_pid"]):
        raise ValueError("timed smoke must contain one complete run")
    expected_timing_command = _capture_command(
        deployment_provenance["container_id"],
        Path(run_records[0]["container_pid_evidence_path"]),
        Path(current_evidence["binary"]["path"]),
        deployment_provenance["device_mapping"]["container_device_id"],
    )
    if run_records[0]["command"] != expected_timing_command:
        raise ValueError("timed smoke execution command is invalid")
    validate_run_record_artifacts(run_records)
    per_run_manifest = _load_json_object(
        Path(run_records[0]["plan_path"]).parent / "run.json",
        "timed smoke per-run manifest",
    )
    if per_run_manifest != run_records[0]:
        raise ValueError("timed smoke per-run manifest differs from run evidence")
    expected_plan = build_smoke_plan(SMOKE_PAIR_ID, SMOKE_LAUNCH_COUNT)
    if Path(run_records[0]["plan_path"]).read_text(
            encoding="utf-8") != _plan_text(expected_plan):
        raise ValueError("timed smoke raw plan is not canonical")
    parsed_metadata, parsed_samples = parse_capture(
        Path(run_records[0]["stdout_path"]).read_text(encoding="utf-8"),
        expected_plan, 1,
    )
    if len(parsed_samples) != SMOKE_SAMPLE_COUNT:
        raise ValueError("timed smoke raw capture count is invalid")

    device_and_clocks = smoke["device_and_clocks"]
    if (not isinstance(device_and_clocks, dict)
            or set(device_and_clocks) != {
                "device_runtime_metadata", "actual_clock_source",
                "clock_gate", "telemetry",
            } or device_and_clocks["actual_clock_source"]
            != "live_telemetry_watchdog"):
        raise ValueError("timed smoke device evidence schema is invalid")
    metadata = device_and_clocks["device_runtime_metadata"]
    mapping = deployment_provenance["device_mapping"]
    if (not isinstance(metadata, dict)
            or set(metadata) != {
                "device_name", "device_id", "pci_bus_id", "ap_count",
                "wave_size", "runtime_reported_clock_khz_diagnostic",
                "host_clock", "device_timer",
            }
            or metadata["device_id"] != mapping["container_device_id"]
            or metadata["pci_bus_id"] != mapping["pci_bus_id"]
            or metadata["device_name"] != "MetaX C500"
            or metadata["ap_count"] != 104 or metadata["wave_size"] != 64
            or metadata["host_clock"] != "CLOCK_MONOTONIC_RAW"
            or metadata["device_timer"] != "mcEvent"):
        raise ValueError("timed smoke device metadata is invalid")
    if metadata != parsed_metadata:
        raise ValueError("timed smoke device metadata differs from raw capture")
    telemetry = device_and_clocks["telemetry"]
    if not isinstance(telemetry, dict) or set(telemetry) != {
            "preflight", "live", "postflight"}:
        raise ValueError("timed smoke telemetry phase schema is invalid")
    phase_rows = {}
    for phase in ("preflight", "live", "postflight"):
        provenance = telemetry[phase]
        if not isinstance(provenance, dict) or set(provenance) != {
                "path", "sha256", "stderr_path", "stderr_sha256",
                "process"}:
            raise ValueError("timed smoke telemetry provenance is invalid")
        process = provenance["process"]
        if (not isinstance(process, dict)
                or set(process) != {"command", "pid", "pgid"}
                or not isinstance(process["command"], list)
                or not process["command"]
                or not isinstance(process["pid"], int) or process["pid"] <= 0
                or not isinstance(process["pgid"], int)
                or process["pgid"] <= 0):
            raise ValueError("timed smoke telemetry process record is invalid")
        path = _validate_path_hash(
            provenance["path"], provenance["sha256"], phase,
        )
        _validate_path_hash(
            provenance["stderr_path"], provenance["stderr_sha256"], phase,
        )
        phase_rows[phase] = parse_telemetry(
            path, phase, mapping["container_device_id"], mapping["pci_bus_id"],
        )
    if len(phase_rows["preflight"]) != 1 or not phase_rows["postflight"]:
        raise ValueError("timed smoke boundary telemetry count is invalid")
    recomputed_gate = validate_telemetry_capture(
        phase_rows["preflight"], phase_rows["live"],
        phase_rows["postflight"],
        [smoke["geometry_probe"]["execution"], *run_records],
    )
    if device_and_clocks["clock_gate"] != recomputed_gate:
        raise ValueError("timed smoke device-and-clock gate is not reproducible")

    artifacts = smoke["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {
            "samples_csv_sha256", "samples_jsonl_sha256", "summary_sha256"}:
        raise ValueError("timed smoke artifact schema is invalid")
    _validate_path_hash(
        resolved.parent / "samples.csv", artifacts["samples_csv_sha256"],
        "samples CSV",
    )
    samples_jsonl_path = _validate_path_hash(
        resolved.parent / "samples.jsonl",
        artifacts["samples_jsonl_sha256"], "samples JSONL",
    )
    sample_lines = samples_jsonl_path.read_text(encoding="utf-8").splitlines()
    if len(sample_lines) != SMOKE_SAMPLE_COUNT:
        raise ValueError("timed smoke sample artifact count is invalid")
    sample_rows = [json.loads(line) for line in sample_lines]
    if (any(not isinstance(row, dict) for row in sample_rows)
            or sample_rows != parsed_samples):
        raise ValueError("timed smoke sample JSONL rows differ from raw capture")
    summary_path = _validate_path_hash(
        resolved.parent / "smoke_summary.json", artifacts["summary_sha256"],
        "smoke summary",
    )
    summary = _load_json_object(summary_path, "timed smoke summary")
    if (set(summary) != {
            "schema_version", "capture_mode", "status",
            "hw_spec_eligibility", "sample_count", "resolved_event_count",
            "quantized_zero_count"}
            or summary["schema_version"] != SCHEMA_VERSION
            or summary["capture_mode"] != "timed_smoke"
            or summary["status"] != "not_candidate"
            or summary["hw_spec_eligibility"] != "not_hw_spec_eligible"
            or summary["sample_count"] != SMOKE_SAMPLE_COUNT):
        raise ValueError("timed smoke summary content is invalid")
    matrix_payload = _load_json_object(
        resolved.parent / "matrix.json", "timed smoke matrix",
    )
    expected_matrix = build_smoke_matrix(SMOKE_PAIR_ID, SMOKE_LAUNCH_COUNT)
    if matrix_payload != {
            "matrix_sha256": smoke["matrix_sha256"],
            "matrix": expected_matrix}:
        raise ValueError("timed smoke matrix artifact is invalid")
    if (resolved.parent / "candidate.json").exists():
        raise ValueError("timed smoke must not contain a candidate artifact")

    cleanup_by_phase = {}
    for cleanup_record in smoke["cleanup"]:
        phase = cleanup_record["phase"]
        if phase in cleanup_by_phase:
            raise ValueError("timed smoke cleanup phases are duplicated")
        cleanup_by_phase[phase] = cleanup_record
    expected_cleanup_phases = {
        "preflight", "geometry-probe", "run-01", "live", "postflight",
    }
    if set(cleanup_by_phase) != expected_cleanup_phases:
        raise ValueError("timed smoke cleanup phases are incomplete")
    for phase in ("preflight", "postflight"):
        record = cleanup_by_phase[phase]
        if (set(record) != {"phase", "process", "actions"}
                or record["actions"] != []):
            raise ValueError("timed smoke boundary cleanup is invalid")
    run_cleanup = cleanup_by_phase["run-01"]
    if (set(run_cleanup) != {
            "phase", "pid", "pgid", "actions", "remaining_group_members",
            "pid_identity_source", "container_pid_tree",
            "container_pid_tree_attempts", "errors"}
            or run_cleanup["remaining_group_members"] != []
            or run_cleanup["pid_identity_source"] != "initial"
            or run_cleanup["errors"] != []
            or run_cleanup["container_pid_tree"]
            != run_records[0]["container_pid_tree_cleanup"]
            or run_cleanup["container_pid_tree_attempts"]
            != run_records[0]["container_pid_tree_cleanup_attempts"]):
        raise ValueError("timed smoke run cleanup is invalid")
    geometry_cleanup = cleanup_by_phase["geometry-probe"]
    geometry_execution = smoke["geometry_probe"]["execution"]
    if (set(geometry_cleanup) != {
            "phase", "pid", "pgid", "actions", "remaining_group_members",
            "pid_identity_source", "container_pid_tree",
            "container_pid_tree_attempts", "errors"}
            or geometry_cleanup["remaining_group_members"] != []
            or geometry_cleanup["pid_identity_source"] != "initial"
            or geometry_cleanup["errors"] != []
            or geometry_cleanup["container_pid_tree"]
            != geometry_execution["container_pid_tree_cleanup"]
            or geometry_cleanup["container_pid_tree_attempts"]
            != geometry_execution["container_pid_tree_cleanup_attempts"]):
        raise ValueError("geometry probe cleanup is invalid")
    live_cleanup = cleanup_by_phase["live"]
    if (set(live_cleanup) != {"phase", "pid", "pgid", "actions"}
            or not isinstance(live_cleanup["actions"], list)):
        raise ValueError("timed smoke live cleanup is invalid")
    return {
        "path": str(resolved), "sha256": _sha256(resolved),
    }


def final_revalidate_prior_smoke(
        smoke_manifest_path, baseline_commit, binary_sha256,
        deployment_provenance, current_evidence, current_toolchain,
        initially_validated_provenance):
    final_provenance = validate_smoke_manifest(
        smoke_manifest_path, baseline_commit, binary_sha256,
        deployment_provenance, current_evidence, current_toolchain,
    )
    if final_provenance != initially_validated_provenance:
        raise ValueError(
            "prior timed smoke provenance changed during full capture"
        )
    return final_provenance


def _telemetry_command(phase, device_id, device_bus_id, raw_dir, max_samples):
    return [
        sys.executable, str(TELEMETRY_RUNNER_PATH),
        "--phase", phase, "--device-id", str(device_id),
        "--device-bus-id", device_bus_id,
        "--raw-dir", str(raw_dir), "--max-samples", str(max_samples),
    ]


def _stop_process_group(process, pgid, label):
    actions = []
    process.poll()
    if process.poll() is None and os.getpgid(process.pid) != pgid:
        raise RuntimeError(f"{label} process group identity changed")
    if not _process_group_members(pgid):
        process.wait(timeout=PROCESS_STOP_GRACE_SECONDS)
        return actions
    for stop_signal, name in (
            (signal.SIGINT, "SIGINT"),
            (signal.SIGTERM, "SIGTERM"),
            (signal.SIGKILL, "SIGKILL")):
        try:
            os.killpg(pgid, stop_signal)
        except ProcessLookupError:
            if not _process_group_members(pgid):
                process.wait(timeout=PROCESS_STOP_GRACE_SECONDS)
                return actions
            raise
        actions.append(name)
        deadline = time.monotonic() + PROCESS_STOP_GRACE_SECONDS
        while time.monotonic() < deadline:
            process.poll()
            if not _process_group_members(pgid):
                process.wait(timeout=PROCESS_STOP_GRACE_SECONDS)
                return actions
            time.sleep(0.05)
    members = _process_group_members(pgid)
    raise RuntimeError(
        f"{label} process group retained descendants: {members}"
    )


def _process_group_members(pgid):
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8")
        except (FileNotFoundError, ProcessLookupError):
            stat_text = None
        if stat_text is None:
            continue
        closing_parenthesis = stat_text.rfind(")")
        if closing_parenthesis < 0:
            raise ValueError(f"invalid proc stat for PID {entry.name}")
        fields = stat_text[closing_parenthesis + 2:].split()
        if len(fields) < 3:
            raise ValueError(f"short proc stat for PID {entry.name}")
        if fields[0] == "Z":
            continue
        if int(fields[2]) == pgid:
            members.append(int(entry.name))
    return sorted(members)


def _stop_cooperative_process_group(process, pgid, label):
    actions = []
    process.poll()
    if process.poll() is None and os.getpgid(process.pid) != pgid:
        raise RuntimeError(f"{label} process group identity changed")
    if not _process_group_members(pgid):
        process.wait(timeout=PROCESS_STOP_GRACE_SECONDS)
        return []
    if process.poll() is None:
        os.kill(process.pid, signal.SIGINT)
        actions.append("leader_SIGINT")
        deadline = time.monotonic() + PROCESS_STOP_GRACE_SECONDS
        while time.monotonic() < deadline:
            process.poll()
            if not _process_group_members(pgid):
                process.wait(timeout=PROCESS_STOP_GRACE_SECONDS)
                return actions
            time.sleep(0.05)
    return [*actions, *_stop_process_group(process, pgid, label)]


def _parse_nspid_values(status_text):
    lines = status_text.splitlines()
    nspid_lines = [line for line in lines if line.startswith("NSpid:")]
    if len(nspid_lines) != 1:
        raise ValueError("host /proc status has no unique NSpid mapping")
    fields = nspid_lines[0].split()[1:]
    try:
        namespace_pids = [int(field) for field in fields]
    except ValueError as exc:
        raise ValueError("host NSpid mapping is not numeric") from exc
    if not namespace_pids or any(pid <= 0 for pid in namespace_pids):
        raise ValueError("host NSpid mapping contains an invalid PID")
    return namespace_pids


def parse_host_nspid_status(status_text, host_pid, container_pid):
    namespace_pids = _parse_nspid_values(status_text)
    if len(namespace_pids) < 2:
        raise ValueError("host mapping requires nested PID namespace values")
    if namespace_pids[0] != host_pid or namespace_pids[-1] != container_pid:
        raise ValueError("host NSpid values do not match exact PID identities")
    return {
        "namespace_pids": namespace_pids,
        "host_pid": host_pid, "container_pid": container_pid,
    }


def _parse_host_status_uid(status_text):
    uid_lines = [
        line for line in status_text.splitlines()
        if line.startswith("Uid:")
    ]
    if len(uid_lines) != 1:
        raise ValueError("host /proc status has no unique UID mapping")
    fields = uid_lines[0].split()[1:]
    try:
        values = [int(field) for field in fields]
    except ValueError as exc:
        raise ValueError("host /proc UID mapping is not numeric") from exc
    if len(values) != 4 or any(value != AUDITED_HOST_UID for value in values):
        raise ValueError(
            "capture child does not use the audited same-UID identity")
    return values


def _parse_docker_top_host_pids(text):
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or lines[0].split()[0] != "PID":
        raise ValueError("docker top output has no PID header")
    pids = []
    for line in lines[1:]:
        field = line.split(None, 1)[0]
        if not field.isdigit() or int(field) <= 0:
            raise ValueError("docker top output contains an invalid PID")
        pids.append(int(field))
    if not pids or len(pids) != len(set(pids)):
        raise ValueError("docker top output has no unique process set")
    return pids


def validate_host_orchestration(container_init_host_pid):
    if not DOCKER_PATH.is_file() or not os.access(DOCKER_PATH, os.X_OK):
        raise ValueError(f"audited docker executable is absent: {DOCKER_PATH}")
    if (isinstance(container_init_host_pid, bool)
            or not isinstance(container_init_host_pid, int)
            or container_init_host_pid <= 0):
        raise ValueError("container init host PID must be a positive integer")
    if os.getuid() != AUDITED_HOST_UID:
        raise ValueError("capture runner must use the audited host UID")
    if AUDITED_DEVICE_GID not in os.getgroups():
        raise ValueError("capture runner lacks the audited device group")
    host_namespace = os.readlink("/proc/self/ns/pid")
    return {
        "host_pid_namespace": host_namespace,
        "container_init_host_pid": container_init_host_pid,
        "capture_host_uid": AUDITED_HOST_UID,
        "capture_device_gid": AUDITED_DEVICE_GID,
        "capture_container_user": CAPTURE_CONTAINER_USER,
        "procfs_identity_policy": "same_uid_child_plus_docker_top",
        "docker_path": str(DOCKER_PATH),
        "docker_sha256": _sha256(DOCKER_PATH),
    }


def resolve_host_pid(
        container_pid, container_id, evidence_dir):
    docker_top_stdout_path = evidence_dir / "child.docker-top.stdout.txt"
    docker_top_stderr_path = evidence_dir / "child.docker-top.stderr.txt"
    docker_top_command = [
        str(DOCKER_PATH), "top", container_id,
        "-eo", "pid,user,args",
    ]
    docker_top_result = _run_retained_command(
        docker_top_command,
        docker_top_stdout_path, docker_top_stderr_path,
    )
    if docker_top_result.returncode != 0:
        raise RuntimeError("docker top failed during child PID mapping")
    docker_top_text = docker_top_result.stdout.decode("utf-8")
    container_host_pids = _parse_docker_top_host_pids(docker_top_text)
    candidates = []
    for host_pid in container_host_pids:
        entry = Path(f"/proc/{host_pid}")
        try:
            status_text = (entry / "status").read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            status_text = None
        if status_text is None:
            continue
        namespace_pids = _parse_nspid_values(status_text)
        if (namespace_pids[0] == host_pid
                and namespace_pids[-1] == container_pid):
            if len(namespace_pids) < 2:
                raise ValueError(
                    "capture child is not in a nested PID namespace")
            _parse_host_status_uid(status_text)
            candidates.append(
                (host_pid, status_text, namespace_pids))
    if len(candidates) != 1:
        raise ValueError(
            "container PID did not resolve to one exact host PID: "
            f"{[candidate[0] for candidate in candidates]}"
        )
    host_pid, status_text, namespace_pids = candidates[0]
    status_evidence_path = evidence_dir / "child.host-proc-status.txt"
    status_evidence_path.write_text(status_text, encoding="utf-8")
    return {
        "namespace_pids": namespace_pids,
        "host_pid": host_pid, "container_pid": container_pid,
        "host_uid": AUDITED_HOST_UID,
        # This host denies readlink(/proc/<same-uid-child>/ns/pid) under its
        # ptrace policy.  docker top plus retained NSpid, UID and starttime
        # provide the exact auditable identity instead.
        "container_pid_namespace": None,
        "host_status_source_path": f"/proc/{host_pid}/status",
        "host_status_evidence_path": str(status_evidence_path),
        "host_status_evidence_sha256": _sha256(status_evidence_path),
        "docker_top_argv": docker_top_command,
        "docker_top_returncode": docker_top_result.returncode,
        "docker_top_stdout_path": str(docker_top_stdout_path),
        "docker_top_stdout_sha256": _sha256(docker_top_stdout_path),
        "docker_top_stderr_path": str(docker_top_stderr_path),
        "docker_top_stderr_sha256": _sha256(docker_top_stderr_path),
    }


PID_EVIDENCE_PUBLISH_SCRIPT = r"""
import errno
import os
import sys

destination = sys.argv[1]
pid = int(sys.argv[2])
temporary = f"{destination}.tmp.{pid}"
with open(f"/proc/{pid}/stat", encoding="utf-8") as stat_handle:
    payload = (f"{pid}\n" + stat_handle.read()).encode("utf-8")
published = False
try:
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
    )
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise RuntimeError("PID evidence write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, destination)
    published = True
    directory_descriptor = os.open(
        os.path.dirname(destination), os.O_RDONLY | os.O_DIRECTORY,
    )
    try:
        try:
            os.fsync(directory_descriptor)
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EROFS}:
                raise
    finally:
        os.close(directory_descriptor)
finally:
    if not published:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
""".strip()


def _capture_command(
        container_id, pid_evidence_path, binary_path, device_id,
        executable_args=()):
    wrapper = (
        'set -eu; pid="$$"; '
        '"$1" -c "$2" "$3" "$pid"; '
        'kill -STOP "$pid"; '
        'shift 3; exec "$@"'
    )
    return [
        str(DOCKER_PATH), "exec", "-i", "--user",
        CAPTURE_CONTAINER_USER, "--workdir", str(pid_evidence_path.parent),
        "--env", f"HOME={pid_evidence_path.parent}",
        container_id,
        "/bin/sh", "-c", wrapper, "capture-wrapper",
        CONTAINER_PYTHON_PATH, PID_EVIDENCE_PUBLISH_SCRIPT,
        str(pid_evidence_path), str(binary_path), *executable_args,
        "--device-id", str(device_id),
    ]


def _read_container_pid_evidence(pid_path):
    lines = pid_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 2 or not lines[0].isdigit():
        raise ValueError("container PID evidence is invalid")
    container_pid = int(lines[0])
    starttime = _parse_proc_stat_starttime(lines[1], container_pid)
    return {
        "container_pid": container_pid,
        "starttime_ticks": starttime,
    }


def _wait_for_container_pid(pid_path, process):
    deadline = time.monotonic() + TELEMETRY_TIMEOUT_SECONDS
    while True:
        if pid_path.is_file():
            return _read_container_pid_evidence(pid_path)
        if process.poll() is not None:
            if pid_path.is_file():
                return _read_container_pid_evidence(pid_path)
            raise RuntimeError("capture exited before container PID evidence")
        if time.monotonic() >= deadline:
            raise RuntimeError("capture did not publish container PID evidence")
        time.sleep(0.01)


def _wait_for_stopped_host_pid(pid_mapping, pid_evidence):
    host_pid = pid_mapping["host_pid"]
    deadline = time.monotonic() + TELEMETRY_TIMEOUT_SECONDS
    while True:
        try:
            stat_text = Path(f"/proc/{host_pid}/stat").read_text(
                encoding="utf-8",
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "capture child exited before host mapping release") from exc
        closing = stat_text.rfind(")")
        if closing < 0:
            raise ValueError("mapped host PID stat is invalid")
        fields = stat_text[closing + 2:].split()
        if len(fields) < 20:
            raise ValueError("mapped host PID stat is too short")
        if int(fields[19]) != pid_evidence["starttime_ticks"]:
            raise RuntimeError("mapped host PID identity changed before release")
        if fields[0] in {"T", "t"}:
            return
        if fields[0] == "Z":
            raise RuntimeError("capture child became a zombie before release")
        if time.monotonic() >= deadline:
            raise RuntimeError("capture child did not enter the mapping stop")
        time.sleep(0.01)


def _parse_proc_stat_starttime(stat_text, expected_pid):
    closing_parenthesis = stat_text.rfind(")")
    if closing_parenthesis < 0:
        raise ValueError("container PID stat evidence is invalid")
    pid_and_name = stat_text[:closing_parenthesis + 1]
    pid_text = pid_and_name.split(" ", 1)[0]
    fields = stat_text[closing_parenthesis + 2:].split()
    if (not pid_text.isdigit() or int(pid_text) != expected_pid
            or len(fields) < 20 or not fields[19].isdigit()):
        raise ValueError("container PID stat identity is invalid")
    starttime = int(fields[19])
    if starttime <= 0:
        raise ValueError("container PID starttime is invalid")
    return starttime


CONTAINER_PID_CLEANUP_SCRIPT = r"""
import json
import os
import signal
import sys
import time

root_pid = int(sys.argv[1])
expected_starttime = int(sys.argv[2])

def identity(pid):
    try:
        text = open(f"/proc/{pid}/stat", encoding="utf-8").read()
    except (FileNotFoundError, ProcessLookupError):
        return None
    closing = text.rfind(")")
    if closing < 0:
        raise RuntimeError("invalid proc stat")
    fields = text[closing + 2:].split()
    if len(fields) < 20:
        raise RuntimeError("short proc stat")
    if fields[0] == "Z":
        return None
    return int(fields[19])

def child_pids(pid):
    try:
        text = open(
            f"/proc/{pid}/task/{pid}/children", encoding="utf-8",
        ).read()
    except FileNotFoundError:
        return []
    return [int(value) for value in text.split()]

def wait_stopped(pid, expected_starttime):
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        try:
            text = open(f"/proc/{pid}/stat", encoding="utf-8").read()
        except FileNotFoundError:
            return False
        closing = text.rfind(")")
        fields = text[closing + 2:].split()
        if int(fields[19]) != expected_starttime:
            raise RuntimeError("PID identity changed while freezing tree")
        if fields[0] in {"T", "t"}:
            return True
        if fields[0] == "Z":
            return False
        time.sleep(0.01)
    raise RuntimeError("PID did not stop during tree freeze")

def freeze_tree(pid, identities):
    starttime = identity(pid)
    if starttime is None:
        return
    identities[pid] = starttime
    os.kill(pid, signal.SIGSTOP)
    if not wait_stopped(pid, starttime):
        return
    for child_pid in child_pids(pid):
        freeze_tree(child_pid, identities)

root_starttime = identity(root_pid)
if root_starttime is None:
    print(json.dumps({"status": "already_absent", "root_pid": root_pid}))
    raise SystemExit(0)
if root_starttime != expected_starttime:
    raise RuntimeError("container PID identity changed before cleanup")

observed = {}
freeze_error = None
try:
    freeze_tree(root_pid, observed)
except BaseException as exc:
    freeze_error = exc

signal_errors = []
for signal_value in (signal.SIGKILL, signal.SIGCONT):
    for pid, expected in reversed(list(observed.items())):
        try:
            if identity(pid) == expected:
                os.kill(pid, signal_value)
        except ProcessLookupError:
            pass
        except BaseException as exc:
            signal_errors.append(
                f"PID {pid} signal {signal_value} failed: {exc}"
            )
deadline = time.monotonic() + 2.0
while time.monotonic() < deadline:
    alive = [
        pid for pid, starttime in observed.items()
        if identity(pid) == starttime
    ]
    if not alive:
        print(json.dumps({
            "status": "terminated", "root_pid": root_pid,
            "observed_pids": sorted(observed),
        }, sort_keys=True))
        break
    time.sleep(0.05)
else:
    raise RuntimeError("container PID tree survived bounded cleanup")
if freeze_error is not None:
    raise RuntimeError(
        "container PID tree freeze failed after bounded cleanup"
    ) from freeze_error
if signal_errors:
    raise RuntimeError("; ".join(signal_errors))
""".strip()


def _container_pid_cleanup_invocation(
        container_id, pid_evidence, run_dir, label):
    stdout_path = run_dir / f"{label}.container-cleanup.stdout.txt"
    stderr_path = run_dir / f"{label}.container-cleanup.stderr.txt"
    command = [
        str(DOCKER_PATH), "exec", container_id, CONTAINER_PYTHON_PATH,
        "-c", CONTAINER_PID_CLEANUP_SCRIPT,
        str(pid_evidence["container_pid"]),
        str(pid_evidence["starttime_ticks"]),
    ]
    return command, stdout_path, stderr_path


def cleanup_container_pid_tree(
        container_id, pid_evidence, run_dir, label):
    command, stdout_path, stderr_path = _container_pid_cleanup_invocation(
        container_id, pid_evidence, run_dir, label,
    )
    result = _run_retained_command(command, stdout_path, stderr_path)
    try:
        payload = json.loads(result.stdout.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("container cleanup stdout is not one JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("container cleanup stdout is not one JSON object")
    if (set(payload) == {"status", "root_pid"}
            and payload["status"] == "already_absent"):
        status = "already_absent"
        observed_pids = []
    elif (set(payload) == {"status", "root_pid", "observed_pids"}
            and payload["status"] == "terminated"):
        status = "terminated"
        observed_pids = payload["observed_pids"]
        if (not isinstance(observed_pids, list)
                or any(not isinstance(pid, int) or pid <= 0
                       for pid in observed_pids)
                or len(set(observed_pids)) != len(observed_pids)):
            raise ValueError("container cleanup observed PID list is invalid")
    else:
        raise ValueError("container cleanup status is invalid")
    if payload["root_pid"] != pid_evidence["container_pid"]:
        raise ValueError("container cleanup root PID is invalid")
    return {
        "argv": command, "returncode": result.returncode,
        "status": status, "root_pid": payload["root_pid"],
        "observed_pids": observed_pids,
        "stdout_path": str(stdout_path),
        "stdout_sha256": _sha256(stdout_path),
        "stderr_path": str(stderr_path),
        "stderr_sha256": _sha256(stderr_path),
    }


def _failed_container_cleanup_record(
        container_id, pid_evidence, run_dir, label, error):
    command, stdout_path, stderr_path = _container_pid_cleanup_invocation(
        container_id, pid_evidence, run_dir, label,
    )
    return {
        "argv": command, "returncode": None, "status": "failed",
        "root_pid": pid_evidence["container_pid"], "observed_pids": None,
        "stdout_path": str(stdout_path),
        "stdout_sha256": (
            _sha256(stdout_path) if stdout_path.is_file() else None
        ),
        "stderr_path": str(stderr_path),
        "stderr_sha256": (
            _sha256(stderr_path) if stderr_path.is_file() else None
        ),
        "error_type": type(error).__name__, "error": str(error),
    }


def _attempt_container_pid_tree_cleanup(
        container_id, pid_evidence, run_dir, label):
    try:
        record = cleanup_container_pid_tree(
            container_id, pid_evidence, run_dir, label,
        )
        if record["returncode"] != 0:
            raise RuntimeError("container PID tree cleanup exited nonzero")
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            KeyboardInterrupt, SystemExit) as exc:
        return (
            _failed_container_cleanup_record(
                container_id, pid_evidence, run_dir, label, exc,
            ),
            exc,
        )
    return record, None


def validate_container_pid_tree_cleanup(
        record, pid_evidence, require_initial_absence):
    required = {
        "argv", "returncode", "status", "root_pid", "observed_pids",
        "stdout_path", "stdout_sha256", "stderr_path", "stderr_sha256",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("container PID tree cleanup schema is invalid")
    expected_argv = [
        str(DOCKER_PATH), "exec", AUDITED_CONTAINER_ID,
        CONTAINER_PYTHON_PATH, "-c", CONTAINER_PID_CLEANUP_SCRIPT,
        str(pid_evidence["container_pid"]),
        str(pid_evidence["starttime_ticks"]),
    ]
    if (record["argv"] != expected_argv or record["returncode"] != 0
            or record["root_pid"] != pid_evidence["container_pid"]
            or record["status"] not in {"already_absent", "terminated"}
            or not isinstance(record["observed_pids"], list)
            or any(not isinstance(pid, int) or pid <= 0
                   for pid in record["observed_pids"])
            or len(set(record["observed_pids"]))
            != len(record["observed_pids"])):
        raise ValueError("container PID tree cleanup result is invalid")
    if (record["status"] == "already_absent"
            and record["observed_pids"] != []):
        raise ValueError("absent container PID tree has observed PIDs")
    if require_initial_absence and record["status"] != "already_absent":
        raise ValueError("successful capture left a live container PID tree")
    for path_key, sha_key in (
            ("stdout_path", "stdout_sha256"),
            ("stderr_path", "stderr_sha256")):
        path = Path(record[path_key]).resolve()
        if (not path.is_file()
                or SHA256_PATTERN.fullmatch(record[sha_key]) is None
                or _sha256(path) != record[sha_key]):
            raise ValueError("container cleanup raw evidence is invalid")
    stdout_payload = _load_json_object(
        Path(record["stdout_path"]).resolve(), "container cleanup stdout",
    )
    if record["status"] == "already_absent":
        expected_payload = {
            "status": "already_absent",
            "root_pid": record["root_pid"],
        }
    elif record["status"] == "terminated":
        expected_payload = {
            "status": "terminated",
            "root_pid": record["root_pid"],
            "observed_pids": record["observed_pids"],
        }
    else:
        raise ValueError("container cleanup raw status is invalid")
    if stdout_payload != expected_payload:
        raise ValueError("container cleanup raw result differs from record")


def _validate_mapped_host_child_identity(
        status_text, stat_text, pid_mapping, pid_evidence):
    host_pid = pid_mapping["host_pid"]
    parsed = parse_host_nspid_status(
        status_text, host_pid, pid_mapping["container_pid"],
    )
    if parsed["namespace_pids"] != pid_mapping["namespace_pids"]:
        raise RuntimeError("host PID namespace mapping changed after release")
    _parse_host_status_uid(status_text)
    starttime = _parse_proc_stat_starttime(stat_text, host_pid)
    if starttime != pid_evidence["starttime_ticks"]:
        raise RuntimeError("host PID identity changed after mapping")


def _mapped_host_child_exists(pid_mapping, pid_evidence):
    host_pid = pid_mapping["host_pid"]
    try:
        status_text = Path(f"/proc/{host_pid}/status").read_text(
            encoding="utf-8",
        )
        stat_text = Path(f"/proc/{host_pid}/stat").read_text(
            encoding="utf-8",
        )
    except FileNotFoundError:
        return False
    _validate_mapped_host_child_identity(
        status_text, stat_text, pid_mapping, pid_evidence,
    )
    return True


def execute_capture_run(
        run_index, plan, binary_path, device_id, run_dir, cleanup_records,
        deployment_provenance, signal_state=None, run_label=None,
        executable_args=(), stdout_filename="stdout.csv"):
    if run_label is None:
        run_label = f"run-{run_index:02d}"
    plan_text = _plan_text(plan)
    plan_path = run_dir / "plan.txt"
    plan_path.write_text(plan_text, encoding="utf-8")
    stdout_path = run_dir / stdout_filename
    stderr_path = run_dir / "stderr.log"
    pid_evidence_path = run_dir / "child.container-pid.txt"
    if executable_args:
        command = _capture_command(
            deployment_provenance["container_id"], pid_evidence_path,
            binary_path, device_id, executable_args,
        )
    else:
        command = _capture_command(
            deployment_provenance["container_id"], pid_evidence_path,
            binary_path, device_id,
        )
    process = None
    pgid = None
    pid_evidence = None
    pid_mapping = None
    started_ns = None
    ended_ns = None
    returncode = None
    execution_error = None
    try:
        with (plan_path.open("rb") as stdin_handle,
              stdout_path.open("wb") as stdout_handle,
              stderr_path.open("wb") as stderr_handle):
            started_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            process = subprocess.Popen(
                command, stdin=stdin_handle, stdout=stdout_handle,
                stderr=stderr_handle, start_new_session=True,
            )
            pgid = os.getpgid(process.pid)
            pid_evidence = _wait_for_container_pid(pid_evidence_path, process)
            pid_mapping = resolve_host_pid(
                pid_evidence["container_pid"],
                deployment_provenance["container_id"], run_dir,
            )
            _wait_for_stopped_host_pid(pid_mapping, pid_evidence)
            os.kill(pid_mapping["host_pid"], signal.SIGCONT)
            pid_mapping["release_signal"] = "SIGCONT"
            pid_mapping["released_monotonic_raw_ns"] = (
                time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            )
            returncode = process.wait(timeout=CHILD_TIMEOUT_SECONDS)
            ended_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            if returncode != 0:
                raise RuntimeError(
                    f"capture child exited nonzero: {returncode}"
                )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            KeyboardInterrupt, SystemExit) as exc:
        execution_error = exc
    if signal_state is not None:
        signal_state.enter_finalization()

    container_cleanup = None
    container_cleanup_attempts = []
    cleanup_errors = []
    if pid_evidence is None:
        pid_identity_source = None
    else:
        pid_identity_source = "initial"
        container_cleanup, cleanup_error = (
            _attempt_container_pid_tree_cleanup(
                deployment_provenance["container_id"], pid_evidence,
                run_dir, f"{run_label}-before-launcher",
            )
        )
        container_cleanup_attempts.append(container_cleanup)
        if cleanup_error is not None:
            cleanup_errors.append({
                "stage": "container_pid_tree_before_launcher",
                "error_type": type(cleanup_error).__name__,
                "error": str(cleanup_error),
            })
        else:
            try:
                validate_container_pid_tree_cleanup(
                    container_cleanup, pid_evidence,
                    execution_error is None,
                )
            except (OSError, ValueError, RuntimeError,
                    subprocess.SubprocessError, KeyboardInterrupt,
                    SystemExit) as cleanup_validation_error:
                cleanup_errors.append({
                    "stage": (
                        "container_pid_tree_before_launcher_validation"
                    ),
                    "error_type": type(cleanup_validation_error).__name__,
                    "error": str(cleanup_validation_error),
                })

    launcher_actions = []
    launcher_remaining = []
    if process is not None and pgid is not None:
        try:
            launcher_actions = _stop_process_group(
                process, pgid, f"capture run {run_index}",
            )
            launcher_remaining = _process_group_members(pgid)
            if launcher_remaining:
                raise RuntimeError(
                    "capture launcher process group survived cleanup"
                )
        except (OSError, ValueError, RuntimeError,
                subprocess.SubprocessError, KeyboardInterrupt,
                SystemExit) as launcher_cleanup_error:
            cleanup_errors.append({
                "stage": "launcher_process_group",
                "error_type": type(launcher_cleanup_error).__name__,
                "error": str(launcher_cleanup_error),
            })

    if pid_evidence is None and process is not None:
        if pid_evidence_path.is_file():
            try:
                pid_evidence = _read_container_pid_evidence(pid_evidence_path)
                pid_identity_source = "after_launcher_stop"
            except (OSError, ValueError, RuntimeError,
                    subprocess.SubprocessError, KeyboardInterrupt,
                    SystemExit) as pid_recovery_error:
                cleanup_errors.append({
                    "stage": "late_pid_evidence",
                    "error_type": type(pid_recovery_error).__name__,
                    "error": str(pid_recovery_error),
                })
        else:
            cleanup_errors.append({
                "stage": "late_pid_evidence",
                "error_type": "RuntimeError",
                "error": (
                    "container PID evidence remained absent after "
                    "launcher cleanup"
                ),
            })

    if pid_evidence is not None:
        if pid_identity_source == "initial":
            after_launcher_label = f"{run_label}-after-launcher"
            after_launcher_stage = "container_pid_tree_after_launcher"
        elif pid_identity_source == "after_launcher_stop":
            after_launcher_label = f"{run_label}-late"
            after_launcher_stage = "late_container_pid_tree"
        else:
            raise RuntimeError("container PID identity source is invalid")
        container_cleanup, cleanup_error = (
            _attempt_container_pid_tree_cleanup(
                deployment_provenance["container_id"], pid_evidence,
                run_dir, after_launcher_label,
            )
        )
        container_cleanup_attempts.append(container_cleanup)
        if cleanup_error is not None:
            cleanup_errors.append({
                "stage": after_launcher_stage,
                "error_type": type(cleanup_error).__name__,
                "error": str(cleanup_error),
            })
        else:
            try:
                validate_container_pid_tree_cleanup(
                    container_cleanup, pid_evidence, False,
                )
                if (pid_mapping is not None
                        and _mapped_host_child_exists(
                            pid_mapping, pid_evidence,
                        )):
                    raise RuntimeError(
                        "mapped host child survived final container cleanup"
                    )
            except (OSError, ValueError, RuntimeError,
                    subprocess.SubprocessError, KeyboardInterrupt,
                    SystemExit) as cleanup_validation_error:
                cleanup_errors.append({
                    "stage": f"{after_launcher_stage}_validation",
                    "error_type": type(
                        cleanup_validation_error
                    ).__name__,
                    "error": str(cleanup_validation_error),
                })

    if signal_state is None:
        signal_record = None
    else:
        signal_record = signal_state.signal_record()
        if signal_record is not None and execution_error is None:
            execution_error = signal_state.error
        if (signal_record is None and execution_error is None
                and not cleanup_errors):
            signal_state.rearm_after_success()

    cleanup_records.append({
        "phase": run_label,
        "pid": None if process is None else process.pid,
        "pgid": pgid, "actions": launcher_actions,
        "remaining_group_members": launcher_remaining,
        "pid_identity_source": pid_identity_source,
        "container_pid_tree": container_cleanup,
        "container_pid_tree_attempts": container_cleanup_attempts,
        "errors": cleanup_errors,
    })

    if execution_error is not None or cleanup_errors:
        if execution_error is None:
            failure_error = RuntimeError(
                "capture run cleanup failed: "
                + "; ".join(
                    f"{record['stage']}: {record['error']}"
                    for record in cleanup_errors
                )
            )
        else:
            failure_error = execution_error
        if pid_evidence_path.is_file():
            pid_evidence_artifact_path = str(pid_evidence_path)
            pid_evidence_artifact_sha256 = _sha256(pid_evidence_path)
        else:
            pid_evidence_artifact_path = None
            pid_evidence_artifact_sha256 = None
        failed_record = {
            "status": "failed", "run_index": run_index,
            "child_container_pid": (
                None if pid_evidence is None
                else pid_evidence["container_pid"]
            ),
            "child_pid_identity": pid_evidence,
            "child_host_pid": (
                None if pid_mapping is None else pid_mapping["host_pid"]
            ),
            "launcher_pid": None if process is None else process.pid,
            "launcher_pgid": pgid, "pid_mapping": pid_mapping,
            "command": command,
            "started_monotonic_raw_ns": started_ns,
            "ended_monotonic_raw_ns": time.clock_gettime_ns(
                time.CLOCK_MONOTONIC_RAW,
            ),
            "returncode": returncode,
            "error_type": type(failure_error).__name__,
            "error": str(failure_error),
            "signal": signal_record,
            "cleanup_errors": cleanup_errors,
            "pid_identity_source": pid_identity_source,
            "container_pid_tree_cleanup": container_cleanup,
            "container_pid_tree_cleanup_attempts": (
                container_cleanup_attempts
            ),
            "container_pid_evidence_path": pid_evidence_artifact_path,
            "container_pid_evidence_sha256": pid_evidence_artifact_sha256,
            "plan_path": str(plan_path), "plan_sha256": _sha256(plan_path),
            "stdout_path": str(stdout_path),
            "stdout_sha256": _sha256(stdout_path),
            "stderr_path": str(stderr_path),
            "stderr_sha256": _sha256(stderr_path),
        }
        _write_json(run_dir / "run.json", failed_record)
        raise failure_error
    run_record = {
        "status": "complete", "run_index": run_index,
        "child_container_pid": pid_mapping["container_pid"],
        "child_pid_identity": pid_evidence,
        "child_host_pid": pid_mapping["host_pid"],
        "launcher_pid": process.pid,
        "launcher_pgid": pgid, "pid_mapping": pid_mapping,
        "command": command,
        "started_monotonic_raw_ns": started_ns,
        "ended_monotonic_raw_ns": ended_ns,
        "returncode": returncode,
        "signal": None,
        "pid_identity_source": pid_identity_source,
        "container_pid_tree_cleanup": container_cleanup,
        "container_pid_tree_cleanup_attempts": container_cleanup_attempts,
        "container_pid_evidence_path": str(pid_evidence_path),
        "container_pid_evidence_sha256": _sha256(pid_evidence_path),
        "plan_path": str(plan_path), "plan_sha256": _sha256(plan_path),
        "stdout_path": str(stdout_path),
        "stdout_sha256": _sha256(stdout_path),
        "stderr_path": str(stderr_path),
        "stderr_sha256": _sha256(stderr_path),
    }
    _write_json(run_dir / "run.json", run_record)
    return run_record, stdout_path


def validate_run_record_artifacts(run_records):
    for record in run_records:
        if (not isinstance(record, dict)
                or set(record) != CAPTURE_RUN_RECORD_FIELDS):
            raise ValueError("capture run record schema is invalid")
        if record["status"] != "complete":
            raise ValueError("capture run record is not complete")
        if record["signal"] is not None:
            raise ValueError("complete run contains a signal failure")
        for path_key, sha_key in (
                ("container_pid_evidence_path",
                 "container_pid_evidence_sha256"),
                ("plan_path", "plan_sha256"),
                ("stdout_path", "stdout_sha256"),
                ("stderr_path", "stderr_sha256")):
            artifact_path = Path(record[path_key]).resolve()
            if (not artifact_path.is_file()
                    or _sha256(artifact_path) != record[sha_key]):
                raise ValueError("capture run artifact provenance changed")
        published_identity = _read_container_pid_evidence(
            Path(record["container_pid_evidence_path"]).resolve()
        )
        if (published_identity != record["child_pid_identity"]
                or published_identity["container_pid"]
                != record["child_container_pid"]):
            raise ValueError("published container PID identity changed")
        mapping = record["pid_mapping"]
        required_mapping = {
            "namespace_pids", "host_pid", "container_pid", "host_uid",
            "container_pid_namespace", "host_status_source_path",
            "host_status_evidence_path", "host_status_evidence_sha256",
            "docker_top_argv", "docker_top_returncode",
            "docker_top_stdout_path", "docker_top_stdout_sha256",
            "docker_top_stderr_path", "docker_top_stderr_sha256",
            "release_signal", "released_monotonic_raw_ns",
        }
        if not isinstance(mapping, dict) or set(mapping) != required_mapping:
            raise ValueError("host PID mapping schema is invalid")
        status_evidence_path = Path(
            mapping["host_status_evidence_path"]
        ).resolve()
        if (not status_evidence_path.is_file()
                or _sha256(status_evidence_path)
                != mapping["host_status_evidence_sha256"]):
            raise ValueError("host PID mapping evidence changed")
        status_text = status_evidence_path.read_text(encoding="utf-8")
        parsed_mapping = parse_host_nspid_status(
            status_text, mapping["host_pid"], mapping["container_pid"],
        )
        if (parsed_mapping["namespace_pids"] != mapping["namespace_pids"]
                or _parse_host_status_uid(status_text)
                != [mapping["host_uid"]] * 4
                or mapping["host_uid"] != AUDITED_HOST_UID
                or mapping["release_signal"] != "SIGCONT"
                or not isinstance(
                    mapping["released_monotonic_raw_ns"], int)
                or mapping["released_monotonic_raw_ns"] <= 0):
            raise ValueError("host PID mapping identity is invalid")
        expected_top_argv = [
            str(DOCKER_PATH), "top", AUDITED_CONTAINER_ID,
            "-eo", "pid,user,args",
        ]
        if (mapping["docker_top_argv"] != expected_top_argv
                or mapping["docker_top_returncode"] != 0):
            raise ValueError("docker top PID mapping command is invalid")
        for path_key, sha_key in (
                ("docker_top_stdout_path", "docker_top_stdout_sha256"),
                ("docker_top_stderr_path", "docker_top_stderr_sha256")):
            path = Path(mapping[path_key]).resolve()
            if (not path.is_file()
                    or _sha256(path) != mapping[sha_key]):
                raise ValueError("docker top PID mapping evidence changed")
        docker_top_pids = _parse_docker_top_host_pids(
            Path(mapping["docker_top_stdout_path"]).read_text(
                encoding="utf-8",
            )
        )
        if mapping["host_pid"] not in docker_top_pids:
            raise ValueError("mapped host PID is absent from docker top")
        if record["pid_identity_source"] != "initial":
            raise ValueError("complete run PID identity was not published in time")
        attempts = record["container_pid_tree_cleanup_attempts"]
        if (not isinstance(attempts, list) or len(attempts) != 2
                or attempts[-1] != record["container_pid_tree_cleanup"]):
            raise ValueError("complete run cleanup attempts are invalid")
        validate_container_pid_tree_cleanup(
            attempts[0], record["child_pid_identity"], True,
        )
        validate_container_pid_tree_cleanup(
            attempts[1],
            record["child_pid_identity"], False,
        )


def _telemetry_failure_cleanup_record(
        phase, process_record, actions, remaining_group_members, error,
        cleanup_error, output_path, stderr_path, raw_dir):
    record = {
        "phase": phase,
        "status": "failed",
        "process": process_record,
        "actions": actions,
        "remaining_group_members": remaining_group_members,
        "error_type": type(error).__name__,
        "error": str(error),
        "raw": {
            "stdout_path": str(output_path),
            "stderr_path": str(stderr_path),
            "raw_dir": str(raw_dir),
        },
    }
    if cleanup_error is not None:
        record["cleanup_error_type"] = type(cleanup_error).__name__
        record["cleanup_error"] = str(cleanup_error)
    return record


def _run_telemetry_snapshot(
        phase, device_id, device_bus_id, output_path, stderr_path, raw_dir,
        cleanup_records):
    if phase == "preflight":
        max_samples = 1
        timeout_seconds = TELEMETRY_TIMEOUT_SECONDS
    elif phase == "postflight":
        max_samples = POSTFLIGHT_TELEMETRY_MAX_SAMPLES
        timeout_seconds = POSTFLIGHT_TELEMETRY_TIMEOUT_SECONDS
    else:
        raise ValueError("snapshot telemetry phase is invalid")
    command = _telemetry_command(
        phase, device_id, device_bus_id, raw_dir, max_samples,
    )
    process = None
    pgid = None
    try:
        with (output_path.open("wb") as stdout_handle,
              stderr_path.open("wb") as stderr_handle):
            process = subprocess.Popen(
                command, stdout=stdout_handle, stderr=stderr_handle,
                start_new_session=True,
            )
            pgid = process.pid
            try:
                returncode = process.wait(
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"{phase} telemetry timed out"
                ) from exc
            if returncode != 0:
                raise RuntimeError(
                    f"{phase} telemetry exited {returncode}"
                )
    finally:
        telemetry_error = sys.exc_info()[1]
        if process is not None:
            if pgid is None:
                pgid = process.pid
            process_record = {
                "command": command, "pid": process.pid, "pgid": pgid,
            }
            cleanup_actions = []
            try:
                cleanup_actions = _stop_process_group(
                    process, pgid, phase,
                )
                remaining_group_members = _process_group_members(pgid)
                if remaining_group_members:
                    raise RuntimeError(
                        f"{phase} telemetry process group survived cleanup: "
                        f"{remaining_group_members}"
                    )
            except BaseException as cleanup_error:
                try:
                    remaining_group_members = _process_group_members(pgid)
                except (OSError, ValueError, RuntimeError):
                    remaining_group_members = None
                if telemetry_error is None:
                    recorded_error = cleanup_error
                else:
                    recorded_error = telemetry_error
                cleanup_records.append(
                    _telemetry_failure_cleanup_record(
                        phase, process_record, cleanup_actions,
                        remaining_group_members, recorded_error,
                        cleanup_error, output_path, stderr_path, raw_dir,
                    )
                )
                if telemetry_error is None:
                    raise
            else:
                if telemetry_error is None:
                    cleanup_records.append({
                        "phase": phase, "process": process_record,
                        "actions": cleanup_actions,
                    })
                else:
                    cleanup_records.append(
                        _telemetry_failure_cleanup_record(
                            phase, process_record, cleanup_actions,
                            remaining_group_members, telemetry_error, None,
                            output_path, stderr_path, raw_dir,
                        )
                    )
        else:
            if telemetry_error is None:
                raise RuntimeError(
                    f"{phase} telemetry process was not created"
                )
            cleanup_records.append(
                _telemetry_failure_cleanup_record(
                    phase, {
                        "command": command, "pid": None, "pgid": None,
                    }, [], None, telemetry_error, None,
                    output_path, stderr_path, raw_dir,
                )
            )
    return process_record


def parse_telemetry(
        path, expected_phase, device_id, device_bus_id,
        allow_partial_tail=False):
    rows = []
    raw_text = path.read_text(encoding="utf-8")
    lines = raw_text.splitlines()
    if raw_text and not raw_text.endswith("\n"):
        if allow_partial_tail:
            lines = lines[:-1]
        else:
            raise ValueError("telemetry JSONL has a partial final line")
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ValueError(f"empty telemetry line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid telemetry JSON line {line_number}") from exc
        required = {
            "schema_version", "phase", "sample_started_monotonic_raw_ns",
            "sample_ended_monotonic_raw_ns",
            "bus_id", "device_id", "xcore_clock_mhz", "mc_clock_mhz",
            "gpu_utilization_pct", "hbm_throughput", "process_host_pids",
            "sample_index", "mx_smi",
        }
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("telemetry row schema is invalid")
        if row["schema_version"] != TELEMETRY_SCHEMA_VERSION:
            raise ValueError("telemetry schema version is invalid")
        if row["phase"] != expected_phase:
            raise ValueError("telemetry phase is invalid")
        if row["device_id"] != device_id or row["bus_id"] != device_bus_id:
            raise ValueError("telemetry device identity is invalid")
        for clock_name in ("xcore_clock_mhz", "mc_clock_mhz"):
            if (not isinstance(row[clock_name], int)
                    or row[clock_name] <= 0):
                raise ValueError(f"telemetry {clock_name} is invalid")
        utilization = row["gpu_utilization_pct"]
        if (not isinstance(utilization, (int, float))
                or isinstance(utilization, bool)
                or not math.isfinite(float(utilization))
                or utilization < 0.0 or utilization > 100.0):
            raise ValueError("telemetry GPU utilization is invalid")
        hbm = row["hbm_throughput"]
        if (not isinstance(hbm, dict) or set(hbm) != {"value", "unit"}
                or not isinstance(hbm["value"], (int, float))
                or isinstance(hbm["value"], bool)
                or not math.isfinite(float(hbm["value"]))
                or hbm["value"] < 0.0
                or hbm["unit"] != "MBytes/s"):
            raise ValueError("telemetry HBM throughput is invalid")
        pids = row["process_host_pids"]
        if (not isinstance(pids, list)
                or any(not isinstance(pid, int) or pid <= 0 for pid in pids)
                or len(pids) != len(set(pids))):
            raise ValueError("telemetry process_host_pids is invalid")
        sample_started = row["sample_started_monotonic_raw_ns"]
        sample_ended = row["sample_ended_monotonic_raw_ns"]
        if (not isinstance(sample_started, int) or sample_started <= 0
                or not isinstance(sample_ended, int)
                or sample_ended < sample_started):
            raise ValueError("telemetry sample interval is invalid")
        if row["sample_index"] != len(rows) + 1:
            raise ValueError("telemetry sample index is invalid")
        tool = row["mx_smi"]
        required_tool_keys = {
            "path", "sha256", "version", "version_argv",
            "version_started_monotonic_raw_ns",
            "version_ended_monotonic_raw_ns",
            "version_stdout_path", "version_stdout_sha256",
            "version_stderr_path", "version_stderr_sha256",
            "queries",
        }
        if not isinstance(tool, dict) or set(tool) != required_tool_keys:
            raise ValueError("telemetry mx-smi provenance is invalid")
        if tool["path"] != str(MX_SMI_PATH):
            raise ValueError("telemetry used an unapproved mx-smi path")
        if tool["version"] != "2.3.1":
            raise ValueError("telemetry used an unapproved mx-smi version")
        if tool["version_argv"] != [str(MX_SMI_PATH), "--version"]:
            raise ValueError("telemetry mx-smi version argv is invalid")
        version_started = tool["version_started_monotonic_raw_ns"]
        version_ended = tool["version_ended_monotonic_raw_ns"]
        if (not isinstance(version_started, int) or version_started <= 0
                or not isinstance(version_ended, int)
                or version_ended < version_started):
            raise ValueError("telemetry mx-smi version interval is invalid")
        if not MX_SMI_PATH.is_file() or _sha256(MX_SMI_PATH) != tool["sha256"]:
            raise ValueError("telemetry mx-smi executable SHA-256 is invalid")
        for path_key, sha_key in (
                ("version_stdout_path", "version_stdout_sha256"),
                ("version_stderr_path", "version_stderr_sha256")):
            artifact_path = Path(tool[path_key]).resolve()
            if not artifact_path.is_file() or _sha256(artifact_path) != tool[sha_key]:
                raise ValueError("telemetry raw mx-smi artifact is invalid")
        queries = tool["queries"]
        query_options = {
            "clock": "--show-clock", "usage": "--show-usage",
            "hbm_bandwidth": "--show-hbm-bandwidth",
            "process": "--show-process",
        }
        if not isinstance(queries, dict) or set(queries) != set(query_options):
            raise ValueError("telemetry mx-smi query set is invalid")
        previous_query_end = None
        for query_name, option in query_options.items():
            query = queries[query_name]
            required_query_keys = {
                "argv", "started_monotonic_raw_ns",
                "ended_monotonic_raw_ns", "stdout_path", "stdout_sha256",
                "stderr_path", "stderr_sha256",
            }
            if not isinstance(query, dict) or set(query) != required_query_keys:
                raise ValueError("telemetry mx-smi query provenance is invalid")
            if query["argv"] != [
                    str(MX_SMI_PATH), option, "-i", str(device_id)]:
                raise ValueError("telemetry mx-smi query argv is invalid")
            query_started = query["started_monotonic_raw_ns"]
            query_ended = query["ended_monotonic_raw_ns"]
            if (not isinstance(query_started, int) or query_started <= 0
                    or not isinstance(query_ended, int)
                    or query_ended < query_started
                    or (previous_query_end is not None
                        and query_started < previous_query_end)):
                raise ValueError("telemetry mx-smi query interval is invalid")
            previous_query_end = query_ended
            for path_key, sha_key in (
                    ("stdout_path", "stdout_sha256"),
                    ("stderr_path", "stderr_sha256")):
                artifact_path = Path(query[path_key]).resolve()
                if (not artifact_path.is_file()
                        or _sha256(artifact_path) != query[sha_key]):
                    raise ValueError("telemetry raw mx-smi query is invalid")
        if (sample_started != queries["clock"]["started_monotonic_raw_ns"]
                or sample_ended
                != queries["process"]["ended_monotonic_raw_ns"]):
            raise ValueError("telemetry sample does not bind query intervals")
        rows.append(row)
    if not rows:
        raise ValueError("telemetry output is empty")
    for earlier, later in zip(rows, rows[1:]):
        if (later["sample_started_monotonic_raw_ns"]
                < earlier["sample_ended_monotonic_raw_ns"]):
            raise ValueError("telemetry sample intervals overlap or regress")
    return rows


def validate_telemetry_capture(
        preflight_rows, live_rows, postflight_rows, run_records):
    if any(row["process_host_pids"] for row in preflight_rows):
        raise ValueError("device was not idle during preflight")
    if any(row["process_host_pids"] for row in postflight_rows):
        raise ValueError("device was not idle during postflight")
    if any(row["gpu_utilization_pct"] != 0.0 for row in preflight_rows):
        raise ValueError("GPU utilization was nonzero during preflight")
    if postflight_rows[-1]["gpu_utilization_pct"] != 0.0:
        raise ValueError("GPU utilization did not settle during postflight")
    xcore_clocks = {row["xcore_clock_mhz"] for row in live_rows}
    mc_clocks = {row["mc_clock_mhz"] for row in live_rows}
    if mc_clocks != {TARGET_MC_CLOCK_MHZ}:
        raise ValueError("live MC clock did not equal audited target")
    live_start = live_rows[0]["sample_started_monotonic_raw_ns"]
    live_end = live_rows[-1]["sample_ended_monotonic_raw_ns"]
    if not (
            preflight_rows[-1]["sample_ended_monotonic_raw_ns"] < live_start
            and live_end
            < postflight_rows[0]["sample_started_monotonic_raw_ns"]):
        raise ValueError("telemetry phase intervals are not ordered")
    gaps = [
        later["sample_started_monotonic_raw_ns"]
        - earlier["sample_ended_monotonic_raw_ns"]
        for earlier, later in zip(live_rows, live_rows[1:])
    ]
    if gaps and max(gaps) > MAX_TELEMETRY_SAMPLE_GAP_NS:
        raise ValueError("live telemetry exceeded maximum sample gap")
    child_pids = {record["child_host_pid"] for record in run_records}
    observed_pids = set()
    for row in live_rows:
        observed_pids.update(row["process_host_pids"])
    if not observed_pids.issubset(child_pids):
        raise ValueError("unexpected device process appeared during live capture")
    for record in run_records:
        if (live_start > record["started_monotonic_raw_ns"]
                or live_end < record["ended_monotonic_raw_ns"]):
            raise ValueError("live telemetry does not cover a child interval")
        child_start = record["started_monotonic_raw_ns"]
        child_release = record["pid_mapping"][
            "released_monotonic_raw_ns"
        ]
        child_end = record["ended_monotonic_raw_ns"]
        if not any(
                record["child_host_pid"] in row["process_host_pids"]
                and row["mx_smi"]["queries"]["process"]
                ["started_monotonic_raw_ns"]
                <= child_end
                and row["mx_smi"]["queries"]["process"]
                ["ended_monotonic_raw_ns"]
                >= child_start
                for row in live_rows):
            raise ValueError(
                "live telemetry did not observe active host PID in child interval"
            )
        if not any(
                row["xcore_clock_mhz"] == TARGET_XCORE_CLOCK_MHZ
                and row["mx_smi"]["queries"]["clock"]
                ["started_monotonic_raw_ns"] <= child_end
                and row["mx_smi"]["queries"]["clock"]
                ["ended_monotonic_raw_ns"] >= child_release
                for row in live_rows):
            raise ValueError("active live XCORE clock missed audited target")
    hbm_values = [row["hbm_throughput"]["value"] for row in live_rows]
    hbm_units = {row["hbm_throughput"]["unit"] for row in live_rows}
    if len(hbm_units) != 1:
        raise ValueError("live HBM throughput units changed")
    observed_hbm_max = max(hbm_values)
    hbm_candidate_eligible = (
        observed_hbm_max <= MAX_LIVE_HBM_THROUGHPUT_MB_S
    )
    return {
        "actual_xcore_clock_mhz": TARGET_XCORE_CLOCK_MHZ,
        "active_xcore_clock_mhz": TARGET_XCORE_CLOCK_MHZ,
        "actual_mc_clock_mhz": TARGET_MC_CLOCK_MHZ,
        "observed_xcore_clock_mhz": sorted(xcore_clocks),
        "hbm_throughput_observed_min": min(hbm_values),
        "hbm_throughput_observed_max": observed_hbm_max,
        "hbm_throughput_unit": next(iter(hbm_units)),
        "hbm_pollution_assessment": {
            "status": "pass" if hbm_candidate_eligible else "fail",
            "candidate_eligible": hbm_candidate_eligible,
            "threshold_mbytes_s": MAX_LIVE_HBM_THROUGHPUT_MB_S,
            "observed_max_mbytes_s": observed_hbm_max,
            "threshold_fraction_of_nominal_hbm": (
                MAX_LIVE_HBM_POLLUTION_FRACTION
            ),
            "nominal_hbm_mbytes_s": NOMINAL_HBM_THROUGHPUT_MB_S,
            "physical_basis": (
                "empty kernel has no HBM instruction; external HBM traffic "
                "must remain at or below 0.01% of the documented 1.8-TB/s "
                "device envelope"
            ),
        },
        "live_start_monotonic_raw_ns": live_start,
        "live_end_monotonic_raw_ns": live_end,
    }


def _validate_arguments(parser, args):
    if BASELINE_PATTERN.fullmatch(args.baseline_commit) is None:
        parser.error("--baseline-commit must be exactly 40 lowercase hex digits")
    if BUS_ID_PATTERN.fullmatch(args.device_bus_id) is None:
        parser.error("--device-bus-id must be canonical lowercase PCI bus id")
    required_files = (
        ("--source-snapshot", args.source_snapshot, False, False),
        ("--source-snapshot-manifest", args.source_snapshot_manifest,
         False, False),
        ("--compile-command", args.compile_command, False, False),
        ("--compile-stdout", args.compile_stdout, False, True),
        ("--compile-stderr", args.compile_stderr, False, True),
        ("--binary", args.binary, True, False),
        ("--deployment-manifest", args.deployment_manifest, False, False),
        ("--docker-inspect-output", args.docker_inspect_output, False, False),
    )
    for argument, path, executable, allow_empty in required_files:
        if not path.is_file():
            parser.error(f"{argument} must name a regular file")
        if not allow_empty and path.stat().st_size <= 0:
            parser.error(f"{argument} must not be empty")
        if executable and not os.access(path, os.X_OK):
            parser.error(f"{argument} must be executable")
    if args.smoke:
        if args.smoke_manifest is not None:
            parser.error("--smoke does not accept --smoke-manifest")
    else:
        if args.smoke_manifest is None or not args.smoke_manifest.is_file():
            parser.error("full capture requires --smoke-manifest")
    if (not TELEMETRY_RUNNER_PATH.is_file()
            or not os.access(TELEMETRY_RUNNER_PATH, os.X_OK)):
        parser.error("canonical mx-smi telemetry runner is absent")
    if not PRODUCTION_GEOMETRY_MANIFEST_PATH.is_file():
        parser.error("production launch geometry manifest is absent")
    try:
        validate_output_dir_path(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist")


def validate_output_dir_path(output_dir):
    resolved = output_dir.resolve()
    if (resolved.parent != PROFILER_OUTPUT_ROOT.resolve()
            or RUN_NAME_PATTERN.fullmatch(resolved.name) is None):
        raise ValueError(
            "--output-dir must be a direct run child of "
            f"{PROFILER_OUTPUT_ROOT}"
        )
    return resolved


def _write_failure(
        output_dir, error, run_records, cleanup_records,
        signal_state=None):
    if signal_state is None:
        signal_record = None
    else:
        signal_record = signal_state.signal_record()
    _write_json(output_dir / "failure.json", {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "signal": signal_record,
        "runs": run_records,
        "cleanup": cleanup_records,
        "failed_monotonic_raw_ns": time.clock_gettime_ns(
            time.CLOCK_MONOTONIC_RAW,
        ),
    })


def _main(argv, signal_state):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--source-snapshot", required=True, type=Path)
    parser.add_argument("--source-snapshot-manifest", required=True, type=Path)
    parser.add_argument("--compile-command", required=True, type=Path)
    parser.add_argument("--compile-stdout", required=True, type=Path)
    parser.add_argument("--compile-stderr", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--deployment-manifest", required=True, type=Path)
    parser.add_argument("--docker-inspect-output", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=_nonnegative_integer)
    parser.add_argument("--device-bus-id", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-manifest", type=Path)
    args = parser.parse_args(argv)
    _validate_arguments(parser, args)

    source_path = Path(__file__).with_name("empty_kernel_duration.cpp").resolve()
    runner_path = Path(__file__).resolve()
    evidence = validate_source_and_build_evidence(
        args.baseline_commit, args.source_snapshot.resolve(),
        args.source_snapshot_manifest.resolve(), source_path, runner_path,
        args.compile_command.resolve(), args.compile_stdout.resolve(),
        args.compile_stderr.resolve(), args.binary.resolve(),
    )
    git_provenance = validate_git_baseline(args.baseline_commit, runner_path)
    deployment_provenance = validate_deployment_manifest(
        args.deployment_manifest.resolve(), args.docker_inspect_output.resolve(),
        args.baseline_commit, evidence["source_snapshot"]["sha256"],
        args.device_id, args.device_bus_id,
    )
    deployment_provenance["host_orchestration"] = validate_host_orchestration(
        deployment_provenance["container_init_host_pid"],
    )
    output_dir = args.output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    telemetry_dir = output_dir / "telemetry"
    telemetry_dir.mkdir()
    if args.smoke:
        matrix = build_smoke_matrix(
            SMOKE_PAIR_ID, SMOKE_LAUNCH_COUNT,
        )
        capture_run_count = 1
    else:
        matrix = build_matrix()
        capture_run_count = INDEPENDENT_RUNS
    matrix_sha256 = _payload_sha256(matrix)
    _write_json(output_dir / "matrix.json", {
        "matrix_sha256": matrix_sha256, "matrix": matrix,
    })
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir()
    try:
        compiler_probe = probe_container_tool(
            deployment_provenance["container_id"],
            MXCC_CONTAINER_PATH, "mxcc", evidence_dir,
        )
        validate_compiler_startup_probe(
            evidence["compile"], compiler_probe,
        )
        evidence["compile"]["startup_probe"] = compiler_probe
        evidence["disassembly"] = run_and_validate_mxobjdump(
            args.binary.resolve(), evidence_dir,
            deployment_provenance["container_id"],
        )
        toolchain = {
            "mxcc": container_tool_identity(compiler_probe),
            "mxobjdump": container_tool_identity(
                evidence["disassembly"]["tool_probe"]
            ),
            "mx_smi": {
                "path": str(MX_SMI_PATH), "sha256": _sha256(MX_SMI_PATH),
            },
            "docker": {
                "path": str(DOCKER_PATH), "sha256": _sha256(DOCKER_PATH),
            },
            "python": {
                "path": str(Path(sys.executable).resolve()),
                "sha256": _sha256(Path(sys.executable).resolve()),
            },
        }
        if args.smoke:
            prior_smoke_provenance = None
        else:
            prior_smoke_provenance = validate_smoke_manifest(
                args.smoke_manifest.resolve(), args.baseline_commit,
                evidence["binary"]["sha256"], deployment_provenance,
                evidence, toolchain,
            )
    except (OSError, ValueError, RuntimeError,
            subprocess.SubprocessError) as evidence_error:
        signal_state.enter_finalization()
        _write_failure(
            output_dir, evidence_error, [], [], signal_state,
        )
        raise RuntimeError(
            "container toolchain evidence validation failed"
        ) from evidence_error

    immutable_inputs = (
        args.source_snapshot.resolve(), args.source_snapshot_manifest.resolve(),
        args.compile_command.resolve(), args.compile_stdout.resolve(),
        args.compile_stderr.resolve(), args.binary.resolve(),
        args.deployment_manifest.resolve(), args.docker_inspect_output.resolve(),
        source_path, runner_path, TELEMETRY_RUNNER_PATH,
        PRODUCTION_GEOMETRY_MANIFEST_PATH,
        MX_SMI_PATH.resolve(), DOCKER_PATH.resolve(),
        Path(evidence["compile"]["compiler_version_stdout_path"]),
        Path(evidence["compile"]["compiler_version_stderr_path"]),
        Path(sys.executable).resolve(),
        *container_probe_artifact_paths(compiler_probe),
        *container_probe_artifact_paths(
            evidence["disassembly"]["tool_probe"]
        ),
        Path(evidence["disassembly"]["stdout_path"]),
        Path(evidence["disassembly"]["stderr_path"]),
    )
    if prior_smoke_provenance is not None:
        immutable_inputs = (
            *immutable_inputs, args.smoke_manifest.resolve(),
        )
    immutable_hashes = {str(path): _sha256(path) for path in immutable_inputs}

    all_samples = []
    run_records = []
    cleanup_records = []
    reference_metadata = None
    live_process = None
    live_pgid = None
    live_stdout_handle = None
    live_stderr_handle = None
    error = None
    preflight_record = None
    live_record = None
    postflight_record = None
    geometry_probe_evidence = None
    try:
        preflight_path = telemetry_dir / "preflight.jsonl"
        preflight_stderr_path = telemetry_dir / "preflight.stderr.log"
        preflight_raw_dir = telemetry_dir / "preflight_mx_smi_raw"
        preflight_record = _run_telemetry_snapshot(
            "preflight", args.device_id, args.device_bus_id,
            preflight_path, preflight_stderr_path, preflight_raw_dir,
            cleanup_records,
        )
        preflight_rows = parse_telemetry(
            preflight_path, "preflight", args.device_id, args.device_bus_id,
        )
        if any(row["process_host_pids"] for row in preflight_rows):
            raise ValueError("device was not idle during preflight")

        live_path = telemetry_dir / "live.jsonl"
        live_stderr_path = telemetry_dir / "live.stderr.log"
        live_raw_dir = telemetry_dir / "live_mx_smi_raw"
        live_stdout_handle = live_path.open("wb")
        live_stderr_handle = live_stderr_path.open("wb")
        live_command = _telemetry_command(
            "live", args.device_id, args.device_bus_id, live_raw_dir,
            LIVE_TELEMETRY_MAX_SAMPLES,
        )
        live_process = subprocess.Popen(
            live_command, stdout=live_stdout_handle, stderr=live_stderr_handle,
            start_new_session=True,
        )
        live_pgid = os.getpgid(live_process.pid)
        live_record = {
            "command": live_command, "pid": live_process.pid,
            "pgid": live_pgid,
        }
        live_ready_deadline = time.monotonic() + TELEMETRY_TIMEOUT_SECONDS
        while live_path.stat().st_size == 0:
            if live_process.poll() is not None:
                raise RuntimeError("live telemetry exited before first sample")
            if time.monotonic() >= live_ready_deadline:
                raise RuntimeError("live telemetry did not emit its first sample")
            time.sleep(0.05)

        geometry_probe_dir = output_dir / "geometry-probe"
        geometry_probe_dir.mkdir()
        geometry_execution, geometry_raw_path = execute_capture_run(
            0, [], args.binary.resolve(), args.device_id,
            geometry_probe_dir, cleanup_records, deployment_provenance,
            signal_state, run_label="geometry-probe",
            executable_args=("--geometry-probe",),
            stdout_filename="geometry_probe.bin",
        )
        geometry_probe_evidence = build_geometry_probe_evidence(
            geometry_raw_path, geometry_execution, evidence,
        )

        for run_index in range(1, capture_run_count + 1):
            run_dir = output_dir / f"run-{run_index:02d}"
            run_dir.mkdir()
            if args.smoke:
                plan = build_smoke_plan(
                    SMOKE_PAIR_ID, SMOKE_LAUNCH_COUNT,
                )
            else:
                plan = build_plan(run_index)
            run_record, stdout_path = execute_capture_run(
                run_index, plan, args.binary.resolve(), args.device_id,
                run_dir, cleanup_records, deployment_provenance,
                signal_state,
            )
            run_records.append(run_record)
            if run_record["returncode"] != 0:
                raise RuntimeError(
                    "capture child PID "
                    f"{run_record['child_container_pid']} exited "
                    f"{run_record['returncode']}"
                )
            metadata, samples = parse_capture(
                stdout_path.read_text(encoding="utf-8"), plan, run_index,
            )
            if metadata["device_id"] != args.device_id:
                raise RuntimeError("capture child used a different device id")
            if metadata["pci_bus_id"] != args.device_bus_id:
                raise RuntimeError("capture child used a different PCI bus")
            if reference_metadata is None:
                reference_metadata = metadata
            elif metadata != reference_metadata:
                raise RuntimeError("device metadata changed between runs")
            all_samples.extend(samples)

        live_cover_deadline = time.monotonic() + TELEMETRY_TIMEOUT_SECONDS
        final_child_end = run_records[-1]["ended_monotonic_raw_ns"]
        while True:
            live_stdout_handle.flush()
            live_rows = parse_telemetry(
                live_path, "live", args.device_id, args.device_bus_id,
                allow_partial_tail=True,
            )
            if (live_rows[-1]["sample_ended_monotonic_raw_ns"]
                    >= final_child_end):
                break
            if live_process.poll() is not None:
                raise RuntimeError("live telemetry ended before capture coverage")
            if time.monotonic() >= live_cover_deadline:
                raise RuntimeError("live telemetry did not cover final child end")
            time.sleep(0.05)
    except (Exception, KeyboardInterrupt, SystemExit, GeneratorExit) as exc:
        error = exc
    finally:
        signal_state.enter_finalization()
        if live_process is not None and live_pgid is not None:
            try:
                actions = _stop_cooperative_process_group(
                    live_process, live_pgid, "live telemetry",
                )
                cleanup_records.append({
                    "phase": "live", "pid": live_process.pid,
                    "pgid": live_pgid,
                    "actions": actions,
                })
                if live_process.returncode != 0:
                    raise RuntimeError(
                        "live telemetry logger exited nonzero after cleanup"
                    )
            except (OSError, ValueError, RuntimeError,
                    subprocess.SubprocessError) as cleanup_error:
                if error is None:
                    error = cleanup_error
                else:
                    cleanup_records.append({
                        "error": str(cleanup_error), "phase": "live",
                    })
        if live_stdout_handle is not None:
            live_stdout_handle.close()
        if live_stderr_handle is not None:
            live_stderr_handle.close()
        try:
            postflight_path = telemetry_dir / "postflight.jsonl"
            postflight_stderr_path = telemetry_dir / "postflight.stderr.log"
            postflight_raw_dir = telemetry_dir / "postflight_mx_smi_raw"
            postflight_record = _run_telemetry_snapshot(
                "postflight", args.device_id, args.device_bus_id,
                postflight_path, postflight_stderr_path, postflight_raw_dir,
                cleanup_records,
            )
        except (Exception, KeyboardInterrupt,
                SystemExit, GeneratorExit) as postflight_error:
            if error is None:
                error = postflight_error
        if signal_state.signal_record() is not None and error is None:
            error = signal_state.error
        if error is None:
            signal_state.rearm_after_success()
    if error is not None:
        _write_failure(
            output_dir, error, run_records, cleanup_records,
            signal_state,
        )
        raise RuntimeError("C500 retained capture failed") from error

    try:
        postflight_rows = parse_telemetry(
            postflight_path, "postflight", args.device_id,
            args.device_bus_id,
        )
        live_rows = parse_telemetry(
            live_path, "live", args.device_id, args.device_bus_id,
        )
        clock_gate = validate_telemetry_capture(
            preflight_rows, live_rows, postflight_rows,
            [geometry_probe_evidence["execution"], *run_records],
        )
        for path in immutable_inputs:
            if _sha256(path) != immutable_hashes[str(path)]:
                raise RuntimeError(
                    f"capture provenance input changed: {path}"
                )
        validate_run_record_artifacts(run_records)
        validate_geometry_probe_evidence(
            geometry_probe_evidence, evidence, evidence["binary"]["sha256"],
            deployment_provenance,
        )
        if reference_metadata is None:
            raise RuntimeError("no device metadata was captured")
        if not args.smoke:
            prior_smoke_provenance = final_revalidate_prior_smoke(
                args.smoke_manifest.resolve(), args.baseline_commit,
                evidence["binary"]["sha256"], deployment_provenance,
                evidence, toolchain, prior_smoke_provenance,
            )
        sample_checksums = _write_samples(output_dir, all_samples)
        if args.smoke:
            summary = {
                "schema_version": SCHEMA_VERSION,
                "capture_mode": "timed_smoke",
                "status": "not_candidate",
                "hw_spec_eligibility": "not_hw_spec_eligible",
                "sample_count": len(all_samples),
                "resolved_event_count": sum(
                    1 for sample in all_samples
                    if sample["event_status"] == "resolved"
                ),
                "quantized_zero_count": sum(
                    1 for sample in all_samples
                    if sample["event_status"] == "quantized_zero"
                ),
            }
            summary_path = output_dir / "smoke_summary.json"
            candidate_path = None
            _write_json(summary_path, summary)
        else:
            summary = build_summary(all_samples)
            summary_path = output_dir / "summary.json"
            _write_json(summary_path, summary)
            candidate = build_candidate(summary, clock_gate)
            candidate_path = output_dir / "candidate.json"
            _write_json(candidate_path, candidate)
    except (OSError, ValueError, RuntimeError) as finalization_error:
        signal_state.enter_finalization()
        _write_failure(
            output_dir, finalization_error, run_records, cleanup_records,
            signal_state,
        )
        raise RuntimeError("C500 retained capture finalization failed") from finalization_error
    try:
        telemetry_provenance = {}
        for phase, path, stderr_path, process_record in (
                ("preflight", preflight_path, preflight_stderr_path,
                 preflight_record),
                ("live", live_path, live_stderr_path, live_record),
                ("postflight", postflight_path, postflight_stderr_path,
                 postflight_record)):
            telemetry_provenance[phase] = {
                "path": str(path), "sha256": _sha256(path),
                "stderr_path": str(stderr_path),
                "stderr_sha256": _sha256(stderr_path),
                "process": process_record,
            }
        if args.smoke:
            capture_mode = "timed_smoke"
            timed_smoke = {
                "status": "not_candidate",
                "eligibility": "not_hw_spec_eligible",
                "pair_id": SMOKE_PAIR_ID,
                "launch_count": SMOKE_LAUNCH_COUNT,
                "trials": SMOKE_TRIALS,
            }
        else:
            capture_mode = "candidate_capture"
            timed_smoke = {
                "status": "prior_smoke_required_and_validated",
                "eligibility": "not_applicable",
                "provenance": prior_smoke_provenance,
            }
        artifacts = {
            **sample_checksums,
            "summary_sha256": _sha256(summary_path),
        }
        if candidate_path is not None:
            artifacts["candidate_sha256"] = _sha256(candidate_path)
        run_manifest = {
            "schema_version": SCHEMA_VERSION, "status": "complete",
            "capture_mode": capture_mode,
            "timed_smoke": timed_smoke,
            "hardware_id": HARDWARE_ID, "runner_pid": os.getpid(),
            "runner_command": [
                str(runner_path), *(sys.argv[1:] if argv is None else argv),
            ],
            "baseline_commit": args.baseline_commit,
            "git_provenance": git_provenance,
            "evidence": evidence,
            "toolchain": toolchain,
            "deployment": deployment_provenance,
            "geometry_probe": geometry_probe_evidence,
            "telemetry_runner": {
                "path": str(TELEMETRY_RUNNER_PATH),
                "sha256": _sha256(TELEMETRY_RUNNER_PATH),
                "executable": os.access(TELEMETRY_RUNNER_PATH, os.X_OK),
                "python_executable": sys.executable,
                "python_executable_sha256": _sha256(Path(sys.executable)),
            },
            "device_and_clocks": {
                "device_runtime_metadata": reference_metadata,
                "actual_clock_source": "live_telemetry_watchdog",
                "clock_gate": clock_gate,
                "telemetry": telemetry_provenance,
            },
            "matrix_sha256": matrix_sha256,
            "independent_runs": run_records,
            "cleanup": cleanup_records,
            "sample_count": len(all_samples),
            "artifacts": artifacts,
        }
        _write_json(output_dir / "run.json", run_manifest)
    except (OSError, ValueError, RuntimeError) as manifest_error:
        signal_state.enter_finalization()
        _write_failure(
            output_dir, manifest_error, run_records, cleanup_records,
            signal_state,
        )
        raise RuntimeError(
            "C500 completion manifest finalization failed"
        ) from manifest_error


def main(argv=None):
    signal_state = RunnerSignalState()
    signal_state.install()
    try:
        return _main(argv, signal_state)
    finally:
        signal_state.restore()


if __name__ == "__main__":
    main()
