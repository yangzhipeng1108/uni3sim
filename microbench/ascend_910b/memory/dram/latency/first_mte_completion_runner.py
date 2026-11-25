#!/usr/bin/env python3
"""Run the frozen Ascend 910B first-MTE completion sweep."""

import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


BLOCK_DIMS = (1, 8, 24, 48)
DIRECTIONS = (0, 1)
TILE_BYTES = (32, 1024, 2048)
VECTOR_EDGE_DIRECTIONS = (2, 3)
VECTOR_EDGE_TILE_BYTES = (4096, 5120, 8192, 10240, 16384)
SWIGLU_QUEUE_EDGE_DIRECTIONS = (4, 5)
SWIGLU_QUEUE_EDGE_TILE_BYTES = (
    10240, 11264, 11776, 12288, 13824, 15360, 16896, 17920, 18432,
    18944, 19456, 26624,
)
SWIGLU_T1024_BASE_TILE_BYTES = 19 * 512 * 2
SWIGLU_T1024_PAYLOAD_CONTROL_BYTES = 13312 * 2
SWIGLU_T1024_QUEUE_EDGE_GEOMETRIES = (
    (1, SWIGLU_T1024_BASE_TILE_BYTES),
    (24, SWIGLU_T1024_BASE_TILE_BYTES),
    (48, SWIGLU_T1024_BASE_TILE_BYTES),
    (48, SWIGLU_T1024_PAYLOAD_CONTROL_BYTES),
)
SWIGLU_T8_QUEUE_EDGE_BLOCK_DIM = 40
SWIGLU_T8_QUEUE_EDGE_TILE_BYTES = 7 * 1280 * 2
SWIGLU_T8_I28672_QUEUE_EDGE_BLOCK_DIM = 48
SWIGLU_T8_I28672_QUEUE_EDGE_TILE_BYTES = (5632 * 2, 512 * 2)
SWIGLU_T64_QUEUE_EDGE_BLOCK_DIM = 48
SWIGLU_T64_QUEUE_EDGE_TILE_BYTES = (6 * 1536 * 2, 7 * 1536 * 2)
RMSNORM_QUEUE_EDGE_BLOCK_DIMS = (1, 2, 48)
RMSNORM_QUEUE_EDGE_DIRECTIONS = (6, 7)
RMSNORM_QUEUE_EDGE_TILE_BYTES = (4096, 5120, 8192, 10240, 16384)
RMSNORM_KEY0_H2048_EDGE_BLOCK_DIMS = (4, 8)
RMSNORM_KEY0_H2048_EDGE_TILE_BYTES = 4096
RMSNORM_KEY0_MIDWORKER_EDGE_BLOCK_DIM = 16
RMSNORM_KEY0_MIDWORKER_EDGE_TILE_BYTES = (8192, 10240)
RMSNORM_MTE2_ADMISSION_BLOCK_DIMS = (32, 43, 47)
ROPE_KEY24_Q_SLICE_TILE_BYTES = 4096
ROPE_KEY24_UNLOADED_CONTROL = (1, ROPE_KEY24_Q_SLICE_TILE_BYTES)
ROPE_KEY24_REPRESENTATIVE_GEOMETRIES = (
    (32, ROPE_KEY24_Q_SLICE_TILE_BYTES),
    (43, ROPE_KEY24_Q_SLICE_TILE_BYTES),
    (47, ROPE_KEY24_Q_SLICE_TILE_BYTES),
)
ROPE_KEY22_Q_SLICE_TILE_BYTES = 4096 * 2
ROPE_KEY22_Q_SLICE_GEOMETRIES = (
    (1, ROPE_KEY22_Q_SLICE_TILE_BYTES),
    (16, ROPE_KEY22_Q_SLICE_TILE_BYTES),
)
GATHER_INDEX_SCALAR_BLOCK_DIMS = (1, 16, 32, 48)
GATHER_INDEX_SCALAR_TILE_BYTES = (4, 8, 16, 32)
PAIR_ORDERS = (0, 1)
ROPE_KEY22_ACTIVE4_Q_SLICE_CELLS = tuple(
    (4, direction, ROPE_KEY22_Q_SLICE_TILE_BYTES, pair_order)
    for direction in VECTOR_EDGE_DIRECTIONS
    for pair_order in PAIR_ORDERS)
ROPE_KEY22_ACTIVE8_Q_SLICE_CELLS = tuple(
    (8, direction, ROPE_KEY22_Q_SLICE_TILE_BYTES, pair_order)
    for direction in VECTOR_EDGE_DIRECTIONS
    for pair_order in PAIR_ORDERS)
SWIGLU_T1024_QUEUE_EDGE_CELLS = tuple(
    (block_dim, direction, tile_bytes, pair_order)
    for block_dim, tile_bytes in SWIGLU_T1024_QUEUE_EDGE_GEOMETRIES
    for direction in SWIGLU_QUEUE_EDGE_DIRECTIONS
    for pair_order in PAIR_ORDERS)
SWIGLU_T8_QUEUE_EDGE_CELLS = tuple(
    (SWIGLU_T8_QUEUE_EDGE_BLOCK_DIM, direction,
     SWIGLU_T8_QUEUE_EDGE_TILE_BYTES, pair_order)
    for direction in SWIGLU_QUEUE_EDGE_DIRECTIONS
    for pair_order in PAIR_ORDERS)
SWIGLU_T8_I28672_QUEUE_EDGE_CELLS = tuple(
    (SWIGLU_T8_I28672_QUEUE_EDGE_BLOCK_DIM, direction, tile_bytes,
     pair_order)
    for tile_bytes in SWIGLU_T8_I28672_QUEUE_EDGE_TILE_BYTES
    for direction in SWIGLU_QUEUE_EDGE_DIRECTIONS
    for pair_order in PAIR_ORDERS)
SWIGLU_T64_QUEUE_EDGE_CELLS = tuple(
    (SWIGLU_T64_QUEUE_EDGE_BLOCK_DIM, direction, tile_bytes, pair_order)
    for tile_bytes in SWIGLU_T64_QUEUE_EDGE_TILE_BYTES
    for direction in SWIGLU_QUEUE_EDGE_DIRECTIONS
    for pair_order in PAIR_ORDERS)
ROPE_KEY24_Q_SLICE_CELLS = tuple(
    (block_dim, direction, tile_bytes, pair_order)
    for block_dim, tile_bytes in (
        ROPE_KEY24_UNLOADED_CONTROL,
        *ROPE_KEY24_REPRESENTATIVE_GEOMETRIES,
    )
    for direction in VECTOR_EDGE_DIRECTIONS
    for pair_order in PAIR_ORDERS)
ROPE_KEY22_Q_SLICE_CELLS = tuple(
    (block_dim, direction, tile_bytes, pair_order)
    for block_dim, tile_bytes in ROPE_KEY22_Q_SLICE_GEOMETRIES
    for direction in VECTOR_EDGE_DIRECTIONS
    for pair_order in PAIR_ORDERS)
GATHER_INDEX_SCALAR_CELLS = tuple(
    (block_dim, 8, tile_bytes, pair_order)
    for block_dim in GATHER_INDEX_SCALAR_BLOCK_DIMS
    for tile_bytes in GATHER_INDEX_SCALAR_TILE_BYTES
    for pair_order in PAIR_ORDERS)
RMSNORM_MTE2_ADMISSION_CELLS = tuple(
    (block_dim, 6, tile_bytes, pair_order)
    for block_dim in RMSNORM_MTE2_ADMISSION_BLOCK_DIMS
    for tile_bytes in RMSNORM_QUEUE_EDGE_TILE_BYTES
    for pair_order in PAIR_ORDERS)
RMSNORM_ROW_GROUP_EDGE_CELLS = tuple(
    (block_dim, direction, tile_bytes, pair_order)
    for block_dim in RMSNORM_MTE2_ADMISSION_BLOCK_DIMS
    for direction in RMSNORM_QUEUE_EDGE_DIRECTIONS
    for tile_bytes in RMSNORM_QUEUE_EDGE_TILE_BYTES
    for pair_order in PAIR_ORDERS)
RMSNORM_RSTD_COMPLETION_TILE_BYTES = (32, 64, 96, 128, 256)
RMSNORM_RSTD_COMPLETION_CELLS = tuple(
    (block_dim, 7, tile_bytes, pair_order)
    for block_dim in RMSNORM_MTE2_ADMISSION_BLOCK_DIMS
    for tile_bytes in RMSNORM_RSTD_COMPLETION_TILE_BYTES
    for pair_order in PAIR_ORDERS)
CELLS = tuple(
    (block_dim, direction, tile_bytes, pair_order)
    for block_dim in BLOCK_DIMS
    for direction in DIRECTIONS
    for tile_bytes in TILE_BYTES
    for pair_order in PAIR_ORDERS)
VECTOR_EDGE_CELLS = tuple(
    (block_dim, direction, tile_bytes, pair_order)
    for block_dim in BLOCK_DIMS
    for direction in VECTOR_EDGE_DIRECTIONS
    for tile_bytes in VECTOR_EDGE_TILE_BYTES
    for pair_order in PAIR_ORDERS)
SWIGLU_QUEUE_EDGE_CELLS = tuple(
    (1, direction, tile_bytes, pair_order)
    for direction in SWIGLU_QUEUE_EDGE_DIRECTIONS
    for tile_bytes in SWIGLU_QUEUE_EDGE_TILE_BYTES
    for pair_order in PAIR_ORDERS)
RMSNORM_QUEUE_EDGE_CELLS = tuple(
    (block_dim, direction, tile_bytes, pair_order)
    for block_dim in RMSNORM_QUEUE_EDGE_BLOCK_DIMS
    for direction in RMSNORM_QUEUE_EDGE_DIRECTIONS
    for tile_bytes in RMSNORM_QUEUE_EDGE_TILE_BYTES
    for pair_order in PAIR_ORDERS)
RMSNORM_TWO_WORKER_EDGE_CELLS = tuple(
    (2, direction, tile_bytes, pair_order)
    for direction in RMSNORM_QUEUE_EDGE_DIRECTIONS
    for tile_bytes in RMSNORM_QUEUE_EDGE_TILE_BYTES
    for pair_order in PAIR_ORDERS)
RMSNORM_KEY0_H2048_EDGE_CELLS = tuple(
    (block_dim, direction, RMSNORM_KEY0_H2048_EDGE_TILE_BYTES, pair_order)
    for block_dim in RMSNORM_KEY0_H2048_EDGE_BLOCK_DIMS
    for direction in RMSNORM_QUEUE_EDGE_DIRECTIONS
    for pair_order in PAIR_ORDERS)
RMSNORM_KEY0_MIDWORKER_EDGE_CELLS = tuple(
    (RMSNORM_KEY0_MIDWORKER_EDGE_BLOCK_DIM, direction, tile_bytes, pair_order)
    for direction in RMSNORM_QUEUE_EDGE_DIRECTIONS
    for tile_bytes in RMSNORM_KEY0_MIDWORKER_EDGE_TILE_BYTES
    for pair_order in PAIR_ORDERS)
CAMPAIGN_CELLS = {
    "baseline": CELLS,
    "vector_edges": VECTOR_EDGE_CELLS,
    "swiglu_queue_edges": SWIGLU_QUEUE_EDGE_CELLS,
    "swiglu_t1024_queue_edges": SWIGLU_T1024_QUEUE_EDGE_CELLS,
    "swiglu_t8_queue_edges": SWIGLU_T8_QUEUE_EDGE_CELLS,
    "swiglu_t8_i28672_queue_edges": SWIGLU_T8_I28672_QUEUE_EDGE_CELLS,
    "swiglu_t64_queue_edges": SWIGLU_T64_QUEUE_EDGE_CELLS,
    "rmsnorm_queue_edges": RMSNORM_QUEUE_EDGE_CELLS,
    "rmsnorm_two_worker_edges": RMSNORM_TWO_WORKER_EDGE_CELLS,
    "rmsnorm_key0_h2048_mte_edges": RMSNORM_KEY0_H2048_EDGE_CELLS,
    "rmsnorm_key0_midworker_mte_edges": RMSNORM_KEY0_MIDWORKER_EDGE_CELLS,
    "rmsnorm_mte2_admission": RMSNORM_MTE2_ADMISSION_CELLS,
    "rmsnorm_row_group_edges": RMSNORM_ROW_GROUP_EDGE_CELLS,
    "rmsnorm_rstd_completion": RMSNORM_RSTD_COMPLETION_CELLS,
    "rope_key22_first_q_slice_edges": ROPE_KEY22_Q_SLICE_CELLS,
    "rope_key22_active4_replication_edges": (
        ROPE_KEY22_ACTIVE4_Q_SLICE_CELLS),
    "rope_key22_active8_replication_edges": (
        ROPE_KEY22_ACTIVE8_Q_SLICE_CELLS),
    "rope_key24_q_slice_edges": ROPE_KEY24_Q_SLICE_CELLS,
    "gather_index_scalar_edges": GATHER_INDEX_SCALAR_CELLS,
}
CSV_FIELDS = 13
CHILD_STRIDE_BYTES = 32768
SYSTEM_COUNTER_TICK_NS = 20.0
EXPECTED_TILING_SUM = 2027
DEFAULT_SCHEDULE_CYCLES = 1
RMSNORM_KEY0_H2048_SCHEDULE_CYCLES = 4
ROPE_REQUIRED_INDEPENDENT_RUNS = 3
SWIGLU_T1024_REQUIRED_INDEPENDENT_RUNS = 3
RMSNORM_KEY0_H2048_REQUIRED_INDEPENDENT_RUNS = 3
CAMPAIGN_DEFAULT_SCHEDULE_CYCLES = {
    campaign: DEFAULT_SCHEDULE_CYCLES for campaign in CAMPAIGN_CELLS}
CAMPAIGN_DEFAULT_SCHEDULE_CYCLES[
    "rmsnorm_key0_h2048_mte_edges"] = RMSNORM_KEY0_H2048_SCHEDULE_CYCLES
CAMPAIGN_DEFAULT_SCHEDULE_CYCLES[
    "rmsnorm_key0_midworker_mte_edges"] = (
        RMSNORM_KEY0_H2048_SCHEDULE_CYCLES)
CAMPAIGN_DEFAULT_SCHEDULE_CYCLES[
    "rope_key22_active4_replication_edges"] = 8
CAMPAIGN_DEFAULT_SCHEDULE_CYCLES[
    "rope_key22_active8_replication_edges"] = 8
CAMPAIGN_REQUIRED_INDEPENDENT_RUNS = {
    campaign: 1 for campaign in CAMPAIGN_CELLS}
CAMPAIGN_REQUIRED_INDEPENDENT_RUNS[
    "rope_key22_first_q_slice_edges"] = ROPE_REQUIRED_INDEPENDENT_RUNS
CAMPAIGN_REQUIRED_INDEPENDENT_RUNS[
    "rope_key22_active4_replication_edges"] = ROPE_REQUIRED_INDEPENDENT_RUNS
CAMPAIGN_REQUIRED_INDEPENDENT_RUNS[
    "rope_key22_active8_replication_edges"] = ROPE_REQUIRED_INDEPENDENT_RUNS
CAMPAIGN_REQUIRED_INDEPENDENT_RUNS[
    "rope_key24_q_slice_edges"] = ROPE_REQUIRED_INDEPENDENT_RUNS
CAMPAIGN_REQUIRED_INDEPENDENT_RUNS[
    "swiglu_t1024_queue_edges"] = SWIGLU_T1024_REQUIRED_INDEPENDENT_RUNS
CAMPAIGN_REQUIRED_INDEPENDENT_RUNS[
    "swiglu_t8_queue_edges"] = SWIGLU_T1024_REQUIRED_INDEPENDENT_RUNS
CAMPAIGN_REQUIRED_INDEPENDENT_RUNS[
    "swiglu_t8_i28672_queue_edges"] = SWIGLU_T1024_REQUIRED_INDEPENDENT_RUNS
CAMPAIGN_REQUIRED_INDEPENDENT_RUNS[
    "swiglu_t64_queue_edges"] = SWIGLU_T1024_REQUIRED_INDEPENDENT_RUNS
CAMPAIGN_REQUIRED_INDEPENDENT_RUNS[
    "rmsnorm_key0_h2048_mte_edges"] = (
        RMSNORM_KEY0_H2048_REQUIRED_INDEPENDENT_RUNS)
CAMPAIGN_REQUIRED_INDEPENDENT_RUNS[
    "rmsnorm_key0_midworker_mte_edges"] = (
        RMSNORM_KEY0_H2048_REQUIRED_INDEPENDENT_RUNS)
CAMPAIGN_REQUIRED_INDEPENDENT_RUNS[
    "gather_index_scalar_edges"] = 3
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
_ACTIVE_PROCESS = None


def _cell_index(trial_index, order_position, cells=CELLS):
    return (trial_index + order_position) % len(cells)


def _validate_schedule(trials, cells=CELLS):
    if trials <= 0 or trials % len(cells) != 0:
        raise ValueError(
            f"trials must be a positive multiple of {len(cells)}")
    counts = [[0 for _ in cells] for _ in cells]
    for trial_index in range(trials):
        seen = set()
        for order_position in range(len(cells)):
            cell_index = _cell_index(trial_index, order_position, cells)
            if cell_index in seen:
                raise AssertionError("one trial contains a duplicate cell")
            seen.add(cell_index)
            counts[order_position][cell_index] += 1
    expected = trials // len(cells)
    if any(count != expected for row in counts for count in row):
        raise AssertionError("cell/order schedule is not balanced")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _parse_child_row(line, expected, worker):
    fields = line.split(",")
    if len(fields) != CSV_FIELDS:
        raise RuntimeError("child emitted an invalid schema")
    try:
        integers = tuple(int(fields[index]) for index in range(8))
        control_ns = float(fields[8])
        request_ns = float(fields[9])
        sink = int(fields[10])
        destination_head = int(fields[11])
    except ValueError as exc:
        raise RuntimeError("child emitted a non-numeric value") from exc
    (block_dim, actual_worker, direction, tile_bytes, pair_order,
     stride_bytes, control_cycles, request_cycles) = integers
    if (block_dim, direction, tile_bytes, pair_order) != expected:
        raise RuntimeError("child changed requested order")
    if actual_worker != worker:
        raise RuntimeError("child changed worker order")
    if stride_bytes != CHILD_STRIDE_BYTES:
        raise RuntimeError("child changed the frozen stride")
    if control_cycles < 0 or request_cycles <= 0:
        raise RuntimeError("child emitted invalid cycle counts")
    if (not math.isfinite(control_ns) or not math.isfinite(request_ns)
            or control_ns != control_cycles * SYSTEM_COUNTER_TICK_NS
            or request_ns != request_cycles * SYSTEM_COUNTER_TICK_NS):
        raise RuntimeError("child emitted inconsistent elapsed time")
    expected_sink = (
        EXPECTED_TILING_SUM
        + (0x5a if direction in (0, 2, 4, 6) else 0)
        + ((tile_bytes // 4) * 0x5a5a5a5a if direction == 8 else 0))
    expected_destination = 0x3c00 if direction % 2 == 1 else 0
    if sink != expected_sink or destination_head != expected_destination:
        raise RuntimeError("child emitted invalid data validation fields")
    if fields[12] != "pass":
        raise RuntimeError("child validation did not pass")
    return line


def _terminate_process_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _handle_signal(signum, _frame):
    if _ACTIVE_PROCESS is not None:
        _terminate_process_group(_ACTIVE_PROCESS)
    raise SystemExit(128 + signum)


def main():
    global _ACTIVE_PROCESS

    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--campaign", choices=tuple(CAMPAIGN_CELLS),
                        default="baseline")
    parser.add_argument("--trials", type=_positive_integer)
    parser.add_argument(
        "--child-timeout-seconds", type=_positive_integer, default=300)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")
    cells = CAMPAIGN_CELLS[args.campaign]
    trials = (
        CAMPAIGN_DEFAULT_SCHEDULE_CYCLES[args.campaign] * len(cells)
        if args.trials is None else args.trials)
    try:
        _validate_schedule(trials, cells)
    except ValueError as exc:
        parser.error(str(exc))

    print(
        f"runner_pid={os.getpid()} device_id={args.device_id} "
        f"run_id={args.run_id} campaign={args.campaign} trials={trials} "
        f"schedule_cycles={trials // len(cells)} "
        "required_independent_runs="
        f"{CAMPAIGN_REQUIRED_INDEPENDENT_RUNS[args.campaign]} "
        f"binary={args.binary.resolve()} binary_sha256={_sha256(args.binary)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "run_id,trial,order_position,process_pid,block_dim,worker,"
        "direction,tile_bytes,pair_order,stride_bytes,control_cycles,"
        "request_cycles,control_ns,request_ns,sink,destination_head,"
        "validation",
        flush=True,
    )
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    for trial_index in range(trials):
        ordered_cells = tuple(
            cells[_cell_index(trial_index, position, cells)]
            for position in range(len(cells)))
        command = [
            str(args.binary), "--batch",
            *(f"{block_dim}:{direction}:{tile_bytes}:{pair_order}"
              for block_dim, direction, tile_bytes, pair_order
              in ordered_cells),
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            start_new_session=True,
        )
        _ACTIVE_PROCESS = process
        try:
            stdout, stderr = process.communicate(
                timeout=args.child_timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            raise RuntimeError(
                f"child PID {process.pid} timed out after "
                f"{args.child_timeout_seconds} seconds") from exc
        finally:
            if process.poll() is None:
                _terminate_process_group(process)
            _ACTIVE_PROCESS = None
        if stderr:
            print(
                f"child_pid={process.pid} trial={trial_index + 1} "
                f"stderr={stderr.rstrip()}",
                file=sys.stderr,
                flush=True,
            )
        if process.returncode != 0:
            raise RuntimeError(
                f"child PID {process.pid} failed with exit "
                f"{process.returncode}: {' '.join(command)}")
        lines = stdout.strip().splitlines()
        expected_lines = sum(cell[0] for cell in ordered_cells)
        if len(lines) != expected_lines:
            raise RuntimeError(
                f"child PID {process.pid} emitted {len(lines)} data lines; "
                f"expected {expected_lines}")
        line_index = 0
        for order_position, expected in enumerate(ordered_cells):
            block_dim, direction, tile_bytes, pair_order = expected
            for worker in range(block_dim):
                line = lines[line_index]
                line_index += 1
                try:
                    validated_line = _parse_child_row(
                        line, expected, worker)
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"child PID {process.pid}: {exc}") from exc
                print(
                    f"{args.run_id},{trial_index + 1},{order_position},"
                    f"{process.pid},{validated_line}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
