#!/usr/bin/env python3
"""Emit bounded, raw-retained MetaX C500 mx-smi telemetry."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


SCHEMA_VERSION = 1
MX_SMI_PATH = Path("/opt/mxdriver/bin/mx-smi")
EXPECTED_VERSION = "2.3.1"
EXPECTED_VERSION_PATTERN = re.compile(
    r"(?<![0-9.])2\.3\.1(?![0-9.])"
)
SNAPSHOT_TIMEOUT_SECONDS = 10
LIVE_INTERVAL_SECONDS = 0.05
BUS_ID_PATTERN = re.compile(
    r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]"
)
DEVICE_HEADER_PATTERN = re.compile(
    r"^\s*GPU#(?P<device_id>\d+)\s+MXC500\s+"
    r"(?P<bus_id>[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])\s*$",
    re.IGNORECASE | re.MULTILINE,
)
CLOCK_PATTERNS = {
    "xcore_clock_mhz": re.compile(
        r"^\s*\|?\s*XCORE\s*(?:\||:)\s*(\d+)\s*MHz\s*\|?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "mc_clock_mhz": re.compile(
        r"^\s*\|?\s*MC\s*(?:\||:)\s*(\d+)\s*MHz\s*\|?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
}
GPU_USAGE_PATTERN = re.compile(
    r"^\s*\|?\s*GPU\s*(?:\||:)\s*(\d+(?:\.\d+)?)\s*%\s*\|?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
HBM_THROUGHPUT_PATTERN = re.compile(
    r"^\s*\|?\s*throughput\s*(?:\||:)\s*"
    r"(\d+(?:\.\d+)?)\s*MBytes/s\s*\|?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
NO_PROCESS_PATTERN = re.compile(
    r"^[ \t]*\|?[ \t]*"
    r"(?:No process(?: found)?|No running processes(?: found)?)\.?"
    r"[ \t]*\|?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
QUERY_OPTIONS = {
    "clock": "--show-clock",
    "usage": "--show-usage",
    "hbm_bandwidth": "--show-hbm-bandwidth",
    "process": "--show-process",
}
STOP_REQUESTED = False


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_stop(signum, frame):
    del signum
    del frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _validate_device_header(raw_text, device_id, device_bus_id, query_name):
    matches = list(DEVICE_HEADER_PATTERN.finditer(raw_text))
    if len(matches) != 1:
        raise ValueError(f"mx-smi {query_name} output has no unique device header")
    match = matches[0]
    if (int(match.group("device_id")) != device_id
            or match.group("bus_id").lower() != device_bus_id):
        raise ValueError(f"mx-smi {query_name} output identifies another device")


def _unique_match(pattern, raw_text, field_name):
    matches = pattern.findall(raw_text)
    if len(matches) != 1:
        raise ValueError(f"mx-smi output has invalid {field_name}")
    return matches[0]


def _parse_process_pids(raw_text, device_id):
    no_process_matches = NO_PROCESS_PATTERN.findall(raw_text)
    table_lines = []
    for line in raw_text.splitlines():
        if re.search(r"\bPID\b", line, re.IGNORECASE) is None:
            continue
        inner_line = line.strip().strip("|").strip()
        if "|" in inner_line:
            cells = [
                cell.strip() for cell in inner_line.split("|")
            ]
            delimiter = "pipe"
        else:
            cells = inner_line.split()
            delimiter = "whitespace"
        normalized_cells = [cell.upper() for cell in cells]
        if "PID" in normalized_cells:
            table_lines.append((line, cells, normalized_cells, delimiter))
    if len(no_process_matches) > 1 or len(table_lines) > 1:
        raise ValueError("mx-smi process output has no unique PID table")
    if not table_lines:
        if len(no_process_matches) == 1:
            return []
        raise ValueError("mx-smi process output has no unique PID table")
    header_line, header_cells, normalized_cells, delimiter = table_lines[0]
    if "GPU" not in normalized_cells:
        raise ValueError("mx-smi process table has no GPU column")
    gpu_column = normalized_cells.index("GPU")
    pid_column = normalized_cells.index("PID")
    header_index = raw_text.splitlines().index(header_line)
    process_rows = []
    for line in raw_text.splitlines()[header_index + 1:]:
        if delimiter == "pipe":
            if "|" not in line:
                continue
            cells = [
                cell.strip() for cell in line.strip().strip("|").split("|")
            ]
            if len(cells) != len(header_cells):
                continue
        elif delimiter == "whitespace":
            cells = line.strip().strip("|").split()
            if len(cells) <= pid_column:
                continue
        else:
            raise AssertionError("unhandled process table delimiter")
        pid_text = cells[pid_column]
        if pid_text.isdigit():
            gpu_match = re.fullmatch(
                r"(?:GPU#)?(\d+)", cells[gpu_column], re.IGNORECASE,
            )
            if gpu_match is None:
                raise ValueError("mx-smi process table has invalid GPU row")
            process_rows.append((int(gpu_match.group(1)), int(pid_text)))
    process_pids = [
        pid for row_device_id, pid in process_rows
        if row_device_id == device_id
    ]
    if len(process_pids) != len(set(process_pids)):
        raise ValueError("mx-smi process table has invalid PID rows")
    if no_process_matches:
        if process_rows:
            raise ValueError(
                "mx-smi process output mixes PID rows and no-process marker"
            )
        return []
    if not process_rows:
        raise ValueError("mx-smi process table has no PID rows")
    return sorted(process_pids)


def parse_mx_smi_snapshot(
        clock_text, usage_text, hbm_text, process_text,
        device_id, device_bus_id):
    query_texts = {
        "clock": clock_text,
        "usage": usage_text,
        "hbm_bandwidth": hbm_text,
    }
    for query_name, raw_text in query_texts.items():
        _validate_device_header(
            raw_text, device_id, device_bus_id, query_name,
        )
    return {
        "xcore_clock_mhz": int(_unique_match(
            CLOCK_PATTERNS["xcore_clock_mhz"], clock_text,
            "XCORE clock",
        )),
        "mc_clock_mhz": int(_unique_match(
            CLOCK_PATTERNS["mc_clock_mhz"], clock_text, "MC clock",
        )),
        "gpu_utilization_pct": float(_unique_match(
            GPU_USAGE_PATTERN, usage_text, "GPU utilization",
        )),
        "hbm_throughput": {
            "value": float(_unique_match(
                HBM_THROUGHPUT_PATTERN, hbm_text, "HBM throughput",
            )),
            "unit": "MBytes/s",
        },
        "process_pids": _parse_process_pids(process_text, device_id),
    }


def _run_command(argv, stdout_path, stderr_path):
    started_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    result = subprocess.run(
        argv, check=False, capture_output=True,
        timeout=SNAPSHOT_TIMEOUT_SECONDS,
    )
    ended_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    stdout_path.write_bytes(result.stdout)
    stderr_path.write_bytes(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"mx-smi command exited {result.returncode}: {' '.join(argv)}"
        )
    return result.stdout.decode("utf-8"), started_ns, ended_ns


def _positive_integer(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _postflight_is_idle(metrics):
    return (
        metrics["gpu_utilization_pct"] == 0.0
        and not metrics["process_pids"]
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", required=True, choices=("preflight", "live", "postflight"),
    )
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--device-bus-id", required=True)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--max-samples", required=True, type=_positive_integer)
    args = parser.parse_args(argv)
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if BUS_ID_PATTERN.fullmatch(args.device_bus_id) is None:
        parser.error("--device-bus-id must be a canonical lowercase PCI bus id")
    if args.raw_dir.exists():
        parser.error("--raw-dir must not already exist")
    if not MX_SMI_PATH.is_file() or not os.access(MX_SMI_PATH, os.X_OK):
        parser.error(f"required mx-smi executable is absent: {MX_SMI_PATH}")
    args.raw_dir.mkdir(parents=True)

    tool_sha256 = _sha256(MX_SMI_PATH)
    version_stdout_path = args.raw_dir / "version.stdout.txt"
    version_stderr_path = args.raw_dir / "version.stderr.txt"
    version_argv = [str(MX_SMI_PATH), "--version"]
    version_text, version_started_ns, version_ended_ns = _run_command(
        version_argv, version_stdout_path, version_stderr_path,
    )
    if EXPECTED_VERSION_PATTERN.search(version_text) is None:
        raise RuntimeError("mx-smi version is not the audited 2.3.1")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    postflight_reached_idle = args.phase != "postflight"
    for sample_index in range(1, args.max_samples + 1):
        if STOP_REQUESTED:
            break
        query_texts = {}
        query_provenance = {}
        sample_started_ns = None
        sample_ended_ns = None
        for query_name, option in QUERY_OPTIONS.items():
            stdout_path = (
                args.raw_dir
                / f"sample-{sample_index:06d}.{query_name}.stdout.txt"
            )
            stderr_path = (
                args.raw_dir
                / f"sample-{sample_index:06d}.{query_name}.stderr.txt"
            )
            query_argv = [
                str(MX_SMI_PATH), option, "-i", str(args.device_id),
            ]
            raw_text, query_started_ns, query_ended_ns = _run_command(
                query_argv, stdout_path, stderr_path,
            )
            query_texts[query_name] = raw_text
            if sample_started_ns is None:
                sample_started_ns = query_started_ns
            sample_ended_ns = query_ended_ns
            query_provenance[query_name] = {
                "argv": query_argv,
                "started_monotonic_raw_ns": query_started_ns,
                "ended_monotonic_raw_ns": query_ended_ns,
                "stdout_path": str(stdout_path),
                "stdout_sha256": _sha256(stdout_path),
                "stderr_path": str(stderr_path),
                "stderr_sha256": _sha256(stderr_path),
            }
        metrics = parse_mx_smi_snapshot(
            query_texts["clock"], query_texts["usage"],
            query_texts["hbm_bandwidth"], query_texts["process"],
            args.device_id, args.device_bus_id,
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "phase": args.phase,
            "sample_index": sample_index,
            "sample_started_monotonic_raw_ns": sample_started_ns,
            "sample_ended_monotonic_raw_ns": sample_ended_ns,
            "bus_id": args.device_bus_id,
            "device_id": args.device_id,
            "xcore_clock_mhz": metrics["xcore_clock_mhz"],
            "mc_clock_mhz": metrics["mc_clock_mhz"],
            "gpu_utilization_pct": metrics["gpu_utilization_pct"],
            "hbm_throughput": metrics["hbm_throughput"],
            "process_host_pids": metrics["process_pids"],
            "mx_smi": {
                "path": str(MX_SMI_PATH),
                "sha256": tool_sha256,
                "version": EXPECTED_VERSION,
                "version_argv": version_argv,
                "version_started_monotonic_raw_ns": version_started_ns,
                "version_ended_monotonic_raw_ns": version_ended_ns,
                "version_stdout_path": str(version_stdout_path),
                "version_stdout_sha256": _sha256(version_stdout_path),
                "version_stderr_path": str(version_stderr_path),
                "version_stderr_sha256": _sha256(version_stderr_path),
                "queries": query_provenance,
            },
        }
        print(json.dumps(payload, sort_keys=True), flush=True)
        if args.phase == "live":
            time.sleep(LIVE_INTERVAL_SECONDS)
        elif args.phase == "preflight":
            break
        elif args.phase == "postflight":
            if _postflight_is_idle(metrics):
                postflight_reached_idle = True
                break
            time.sleep(LIVE_INTERVAL_SECONDS)
        else:
            raise AssertionError("unreachable telemetry phase")
    if not postflight_reached_idle:
        raise RuntimeError(
            "postflight telemetry did not reach an idle sample"
        )


if __name__ == "__main__":
    main()
