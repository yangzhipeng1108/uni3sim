#!/usr/bin/env python3
"""Run a balanced raw MTE2 GM-to-UB mechanism sweep."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


TILE_BYTES = (1024, 2048, 4096, 8192, 16384, 32768, 65536)
MODES = (0, 1, 2)
CELLS = tuple((tile_bytes, mode)
              for tile_bytes in TILE_BYTES for mode in MODES)
FOOTPRINT_MB = 512
CSV_FIELDS = 13
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
_ACTIVE_PROCESS = None
_CHILD_SIGNALS = frozenset((signal.SIGINT, signal.SIGTERM))


def _cell_index(trial_index, order_position):
    rotation = trial_index % len(CELLS)
    phase = trial_index // len(CELLS)
    if phase % 2 == 0:
        return (rotation + order_position) % len(CELLS)
    return (rotation - order_position) % len(CELLS)


def _validate_schedule(trials):
    if trials <= 0 or trials % (2 * len(CELLS)) != 0:
        raise ValueError(
            f"trials must be a positive multiple of {2 * len(CELLS)}")
    position_counts = [
        [0 for _ in CELLS]
        for _ in CELLS
    ]
    for trial_index in range(trials):
        seen = set()
        for order_position in range(len(CELLS)):
            cell_index = _cell_index(trial_index, order_position)
            if cell_index in seen:
                raise AssertionError("one trial contains a duplicate cell")
            seen.add(cell_index)
            position_counts[order_position][cell_index] += 1
    expected = trials // len(CELLS)
    if any(count != expected
           for position in position_counts for count in position):
        raise AssertionError("cell/order schedule is not balanced")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _process_group_exists(process_group_id):
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


def _signal_process_group(process_group_id, signum):
    try:
        os.killpg(process_group_id, signum)
    except ProcessLookupError:
        # The final descendant may exit between the preceding existence probe
        # and this signal.  That is already the desired cleanup outcome.
        return False
    return True


def _wait_for_process_group_exit(process, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while _process_group_exists(process.pid):
        # Reap the group leader when it exits.  A zombie leader otherwise
        # keeps the process-group identity visible even after every child has
        # stopped.
        process.poll()
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    process.poll()
    return True


def _terminate_process_group(
        process, *, term_timeout_seconds=10, kill_timeout_seconds=10):
    # The group can outlive its leader, especially when a descendant retains
    # stdout/stderr and communicate() times out after the leader has exited.
    # Probe the process group itself rather than returning from leader.poll().
    if not _process_group_exists(process.pid):
        process.poll()
        return
    if not _signal_process_group(process.pid, signal.SIGTERM):
        process.poll()
        return
    if _wait_for_process_group_exit(process, term_timeout_seconds):
        return
    if not _signal_process_group(process.pid, signal.SIGKILL):
        process.poll()
        return
    if not _wait_for_process_group_exit(process, kill_timeout_seconds):
        raise RuntimeError(
            f"child process group {process.pid} survived SIGKILL")


def _handle_signal(signum, _frame):
    if _ACTIVE_PROCESS is not None:
        _terminate_process_group(_ACTIVE_PROCESS)
    raise SystemExit(128 + signum)


def _communicate_child(process, timeout_seconds):
    try:
        return process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        raise RuntimeError(
            f"child PID {process.pid} timed out after "
            f"{timeout_seconds} seconds"
        ) from exc
    finally:
        if _process_group_exists(process.pid):
            _terminate_process_group(process)


def _unblock_child_ownership_signals():
    signal.pthread_sigmask(signal.SIG_UNBLOCK, _CHILD_SIGNALS)


def _spawn_child(command, environment):
    global _ACTIVE_PROCESS

    # Block ownership signals across Popen and registration.  Otherwise a
    # pending SIGINT/SIGTERM can run after the child exists but before the
    # handler can see it, orphaning the new process group.
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _CHILD_SIGNALS)
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            start_new_session=True,
            preexec_fn=_unblock_child_ownership_signals,
        )
        _ACTIVE_PROCESS = process
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    return process


def main():
    global _ACTIVE_PROCESS

    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trials", type=_positive_integer, default=20)
    parser.add_argument(
        "--child-timeout-seconds", type=_positive_integer, default=900)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")
    try:
        _validate_schedule(args.trials)
    except ValueError as exc:
        parser.error(str(exc))

    print(
        f"runner_pid={os.getpid()} device_id={args.device_id} "
        f"run_id={args.run_id} trials={args.trials} "
        f"binary={args.binary.resolve()} "
        f"binary_sha256={_sha256(args.binary)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "run_id,trial,order_position,process_pid,tile_bytes,"
        "footprint_bytes,mode,n_iters,short_cycles,short_wall_ns,"
        "long_cycles,long_wall_ns,tick_ns,per_tile_cycles,"
        "per_tile_ns,bandwidth_gbs,validation",
        flush=True,
    )
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    for trial_index in range(args.trials):
        ordered_cells = []
        for order_position in range(len(CELLS)):
            cell_index = _cell_index(trial_index, order_position)
            ordered_cells.append(CELLS[cell_index])
        command = [str(args.binary), "--batch", str(FOOTPRINT_MB)]
        for tile_bytes, mode in ordered_cells:
            n_iters = FOOTPRINT_MB * 1024 * 1024 // tile_bytes
            if n_iters < 20000:
                n_iters = 20000
            command.extend((str(tile_bytes), str(mode), str(n_iters)))
        process = _spawn_child(command, environment)
        try:
            stdout, stderr = _communicate_child(
                process, args.child_timeout_seconds)
        finally:
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
        output_lines = stdout.strip().splitlines()
        if len(output_lines) != len(CELLS):
            raise RuntimeError(
                f"child PID {process.pid} emitted "
                f"{len(output_lines)} data lines")
        for order_position, output_line in enumerate(output_lines):
            fields = output_line.split(",")
            if len(fields) != CSV_FIELDS:
                raise RuntimeError(
                    f"child PID {process.pid} emitted an invalid raw schema")
            tile_bytes, mode = ordered_cells[order_position]
            expected_mode = (
                "hard_event_completion" if mode == 0
                else "pipeline" if mode == 1
                else "dynamic_params_completion")
            if int(fields[0]) != tile_bytes or fields[2] != expected_mode:
                raise RuntimeError(
                    f"child PID {process.pid} changed the requested order")
            print(
                f"{args.run_id},{trial_index + 1},{order_position},"
                f"{process.pid},{output_line}",
                flush=True,
            )


if __name__ == "__main__":
    main()
