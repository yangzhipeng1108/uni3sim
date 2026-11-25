#!/usr/bin/env python3
"""Audit direct-Fixpipe raw cycles with preregistered conservative gates."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np

from fixpipe_service_runner import (
    BLOCK_DIMS,
    BUILD_ENVIRONMENT_OVERRIDES,
    CALIBRATION_BLOCK_DIM,
    CALIBRATION_GEOMETRY,
    CALIBRATION_DESTINATION_SLOTS,
    CELLS,
    EXPECTED_CANN,
    EXPECTED_CANN_SOURCE_SHA256,
    EXPECTED_PRODUCTION_GEOMETRY_SHA256,
    HOLDOUT_BLOCK_DIM,
    MECHANISM_HOLDOUT_GEOMETRY,
    ORDER_DIRECTIONS,
    PRODUCTION_ROW_MAJOR_GEOMETRY,
    SOURCE_FILENAMES,
    STATIC_INSTRUCTION_SOURCE_MARKERS,
    SYSTEM_COUNTER_HZ,
    SYSTEM_COUNTER_TICK_NS,
    _invocations,
    _ordered_cells,
    _parse_device_identity,
    _parse_idle_telemetry,
    _parse_npu_smi_version,
    _strides,
)


REQUIRED_FIELDS = (
    "run_id",
    "trial",
    "order_direction",
    "order_position",
    "process_pid",
    "family",
    "layout",
    "m",
    "n",
    "src_stride",
    "dst_stride",
    "mode",
    "stream_depth",
    "block_dim",
    "destination_slots",
    "base_invocations",
    "middle_invocations",
    "long_invocations",
    "base_logical_bytes_per_aic",
    "middle_logical_bytes_per_aic",
    "long_logical_bytes_per_aic",
    "base_block_cycles",
    "middle_block_cycles",
    "long_block_cycles",
    "base_host_wall_ns",
    "middle_host_wall_ns",
    "long_host_wall_ns",
    "clock_source",
    "completion",
    "validation",
)
CALIBRATION_RESIDUAL_LIMIT = 0.10
INTERVAL_CONSISTENCY_LIMIT = 0.05
MECHANISM_HOLDOUT_LIMIT = 0.10
SHAPE_HOLDOUT_LIMIT = 0.15
ORDER_HOLDOUT_LIMIT = 0.10
BLOCK_HOLDOUT_LIMIT = 0.15
STRIDE_HOLDOUT_LIMIT = 0.10
REUSE_HOLDOUT_LIMIT = 0.10
RUN_SPAN_LIMIT = 0.05
MAX_DESIGN_CONDITION = 1.0e6
SOURCE_FULL_TILE_ERROR_LIMIT = 0.20
SOURCE_FULL_TILE_TRIAL_SPAN_LIMIT = 0.05
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _relative_error(observed, expected):
    if observed <= 0 or expected <= 0:
        raise ValueError("relative-error operands must be positive")
    return abs(observed - expected) / expected


def _cycles(text, block_dim, label):
    values = tuple(int(value) for value in text.split(";"))
    if len(values) != block_dim or any(value <= 0 for value in values):
        raise ValueError(
            f"{label} must contain {block_dim} positive device-cycle values"
        )
    return values


def _require_mapping(value, label, fields):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"provenance {label} has an invalid schema")


def _require_nonempty_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"provenance {label} must be non-empty text")


def _require_sha256(value, label):
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"provenance {label} must be a lowercase SHA256")


def _argv_option(argv, option):
    positions = [index for index, value in enumerate(argv) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"provenance capture.argv must contain one {option}")
    return argv[positions[0] + 1]


def _load_provenance(path):
    with path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    required = {
        "schema", "run_id", "captured_at_utc", "source_commit",
        "source_identity_mode", "source_snapshot_sha256", "source_files",
        "cann", "build", "device", "idle_telemetry",
        "idle_telemetry_sha256", "clock", "measurement_scope",
        "publication_limit", "production_or_evaluation_duration_consumed",
        "static_instruction_contract_audited", "static_instruction_evidence",
        "production_geometry_evidence", "capture", "npu_smi",
    }
    if set(raw) != required:
        raise ValueError("provenance has an invalid top-level schema")
    if raw["schema"] != "ascend_910b_fixpipe_service_provenance_v2":
        raise ValueError("unsupported Fixpipe provenance schema")
    _require_nonempty_text(raw["run_id"], "run_id")
    _require_nonempty_text(raw["captured_at_utc"], "captured_at_utc")
    try:
        captured_at = datetime.fromisoformat(raw["captured_at_utc"])
    except ValueError as error:
        raise ValueError("provenance captured_at_utc is not ISO-8601") from error
    if captured_at.tzinfo is None or captured_at.utcoffset().total_seconds() != 0:
        raise ValueError("provenance captured_at_utc must use UTC")
    if (
        not isinstance(raw["source_commit"], str)
        or COMMIT_PATTERN.fullmatch(raw["source_commit"]) is None
    ):
        raise ValueError("provenance source_commit must be 40 lowercase hex digits")
    _require_sha256(raw["source_snapshot_sha256"], "source_snapshot_sha256")
    _require_mapping(raw["source_files"], "source_files", SOURCE_FILENAMES)
    for filename in SOURCE_FILENAMES:
        _require_sha256(raw["source_files"][filename], f"source_files.{filename}")
    if raw["production_or_evaluation_duration_consumed"] is not False:
        raise ValueError("provenance consumed forbidden production/evaluation duration")
    if raw["static_instruction_contract_audited"] is not True:
        raise ValueError("Fixpipe provenance requires a static instruction audit")
    evidence = raw["static_instruction_evidence"]
    _require_mapping(
        evidence, "static_instruction_evidence",
        (
            "evidence_kind", "claim", "production_binding", "sources",
            "disassembly_required",
        ),
    )
    if evidence["evidence_kind"] != "source_bound_compiler_intrinsic_chain":
        raise ValueError("Fixpipe static evidence has an invalid kind")
    _require_nonempty_text(evidence["claim"], "static_instruction_evidence.claim")
    _require_nonempty_text(
        evidence["production_binding"],
        "static_instruction_evidence.production_binding",
    )
    if evidence["disassembly_required"] is not False:
        raise ValueError("Fixpipe source audit must not claim disassembly evidence")
    evidence_sources = evidence["sources"]
    if set(evidence_sources) != set(STATIC_INSTRUCTION_SOURCE_MARKERS):
        raise ValueError("Fixpipe static evidence has an invalid source set")
    for label, expected_markers in STATIC_INSTRUCTION_SOURCE_MARKERS.items():
        source = evidence_sources[label]
        _require_mapping(
            source, f"static_instruction_evidence.sources.{label}",
            ("path", "sha256", "required_markers"),
        )
        _require_nonempty_text(
            source["path"],
            f"static_instruction_evidence.sources.{label}.path",
        )
        _require_sha256(
            source["sha256"],
            f"static_instruction_evidence.sources.{label}.sha256",
        )
        if source["required_markers"] != list(expected_markers):
            raise ValueError(
                f"Fixpipe static evidence markers drifted for {label}"
            )
        if label == "benchmark_kernel":
            expected_sha256 = raw[
                "source_files"]["fixpipe_service_kernel.cpp"]
        elif label in set(STATIC_INSTRUCTION_SOURCE_MARKERS) - {
                "benchmark_kernel"}:
            expected_sha256 = raw[
                "cann"]["source_identities"][label]["sha256"]
        else:
            raise AssertionError(
                f"unexpected Fixpipe static evidence label: {label!r}")
        if source["sha256"] != expected_sha256:
            raise ValueError(
                f"Fixpipe static evidence identity drifted for {label}"
            )
    if raw["source_identity_mode"] not in (
        "audited_source_snapshot", "git_commit_and_source_snapshot"
    ):
        raise ValueError("provenance has an invalid source identity mode")
    cann = raw["cann"]
    _require_mapping(
        cann, "cann",
        ("identity", "version_file", "version_file_sha256", "source_identities"),
    )
    if cann["identity"] != EXPECTED_CANN:
        raise ValueError("provenance does not identify frozen CANN 8.5.1")
    _require_nonempty_text(cann["version_file"], "cann.version_file")
    _require_sha256(cann["version_file_sha256"], "cann.version_file_sha256")
    identities = cann["source_identities"]
    if set(identities) != set(EXPECTED_CANN_SOURCE_SHA256):
        raise ValueError("provenance has an invalid CANN source identity set")
    for label, expected in EXPECTED_CANN_SOURCE_SHA256.items():
        _require_mapping(
            identities[label], f"cann.source_identities.{label}",
            ("path", "sha256"),
        )
        _require_nonempty_text(
            identities[label]["path"], f"cann.source_identities.{label}.path"
        )
        if identities[label]["sha256"] != expected:
            raise ValueError(f"provenance {label} SHA256 is not frozen")
    geometry = raw["production_geometry_evidence"]
    _require_mapping(
        geometry, "production_geometry_evidence",
        ("path", "sha256", "git_lfs_oid_sha256", "derivation"),
    )
    _require_nonempty_text(
        geometry["path"], "production_geometry_evidence.path")
    _require_nonempty_text(
        geometry["derivation"], "production_geometry_evidence.derivation")
    if (geometry["sha256"] != EXPECTED_PRODUCTION_GEOMETRY_SHA256
            or geometry["git_lfs_oid_sha256"]
            != EXPECTED_PRODUCTION_GEOMETRY_SHA256):
        raise ValueError(
            "provenance production geometry identity is not frozen")
    build = raw["build"]
    _require_mapping(
        build, "build",
        (
            "directory", "ascend_home_path", "tools", "build_environment",
            "configure_argv", "build_argv", "cmake_cache_sha256", "binary",
            "binary_sha256", "static_artifact",
        ),
    )
    _require_nonempty_text(build["directory"], "build.directory")
    _require_nonempty_text(
        build["ascend_home_path"], "build.ascend_home_path")
    for field in ("configure_argv", "build_argv"):
        if (
            not isinstance(build[field], list)
            or not build[field]
            or any(not isinstance(value, str) or not value for value in build[field])
        ):
            raise ValueError(f"provenance build.{field} must be an exact argv")
    tools = build["tools"]
    if set(tools) != {"cmake", "make", "cxx", "ar", "ranlib"}:
        raise ValueError("provenance build tools have an invalid schema")
    for name in ("cmake", "make", "cxx", "ar", "ranlib"):
        _require_mapping(
            tools[name], f"build.tools.{name}", ("path", "sha256", "version"))
        _require_nonempty_text(tools[name]["path"], f"build.tools.{name}.path")
        if not Path(tools[name]["path"]).is_absolute():
            raise ValueError(f"provenance build tool {name} path must be absolute")
        _require_sha256(
            tools[name]["sha256"], f"build.tools.{name}.sha256")
        _require_nonempty_text(
            tools[name]["version"], f"build.tools.{name}.version")
    build_environment = build["build_environment"]
    _require_mapping(
        build_environment, "build.build_environment",
        ("removed_overrides", "path"))
    if build_environment["removed_overrides"] != list(
            BUILD_ENVIRONMENT_OVERRIDES):
        raise ValueError("provenance build environment override contract changed")
    _require_nonempty_text(build_environment["path"], "build_environment.path")
    _require_sha256(
        build["cmake_cache_sha256"], "build.cmake_cache_sha256")
    _require_nonempty_text(build["binary"], "build.binary")
    _require_sha256(build["binary_sha256"], "build.binary_sha256")
    build_directory = Path(build["directory"])
    if not build_directory.is_absolute():
        raise ValueError("provenance build directory must be absolute")
    if Path(build["binary"]).resolve() != (
            build_directory / "fixpipe_service").resolve():
        raise ValueError("provenance binary is outside the source-bound build")
    static_artifact = build["static_artifact"]
    _require_mapping(
        static_artifact, "build.static_artifact",
        (
            "path", "sha256", "archive_members", "kernel_member",
            "kernel_member_sha256", "kernel_member_format",
        ),
    )
    _require_nonempty_text(static_artifact["path"], "build.static_artifact.path")
    if Path(static_artifact["path"]).resolve() != (
            build_directory / "lib" / "libfixpipe_service_kernel.a").resolve():
        raise ValueError(
            "provenance static artifact is outside the source-bound build")
    _require_sha256(static_artifact["sha256"], "build.static_artifact.sha256")
    if (
        not isinstance(static_artifact["archive_members"], list)
        or not static_artifact["archive_members"]
        or any(
            not isinstance(member, str) or not member
            for member in static_artifact["archive_members"]
        )
    ):
        raise ValueError("provenance static archive members are invalid")
    _require_nonempty_text(
        static_artifact["kernel_member"], "build.static_artifact.kernel_member"
    )
    if static_artifact["kernel_member"] not in static_artifact["archive_members"]:
        raise ValueError("provenance kernel member is absent from archive members")
    _require_sha256(
        static_artifact["kernel_member_sha256"],
        "build.static_artifact.kernel_member_sha256",
    )
    if static_artifact["kernel_member_format"] != "ELF":
        raise ValueError("provenance kernel archive member must be ELF")
    device = raw["device"]
    _require_mapping(
        device, "device",
        (
            "physical_device_id", "logical_device_id", "model",
            "pci_identity", "selection_environment",
        ),
    )
    if (
        not isinstance(device["physical_device_id"], int)
        or isinstance(device["physical_device_id"], bool)
        or device["physical_device_id"] < 0
    ):
        raise ValueError(
            "provenance physical_device_id must be a non-negative integer"
        )
    if (
        not isinstance(device["logical_device_id"], int)
        or isinstance(device["logical_device_id"], bool)
        or device["logical_device_id"] != 0
    ):
        raise ValueError("provenance logical_device_id must be integer 0")
    normalized_model = re.sub(
        r"\s+", "", device["model"].lower().replace("ascend", "")
    )
    if normalized_model != "910b2":
        raise ValueError("provenance device model must be Ascend 910B2")
    _require_nonempty_text(device["pci_identity"], "device.pci_identity")
    selection = device["selection_environment"]
    _require_mapping(
        selection, "device.selection_environment",
        ("ASCEND_DEVICE_ID", "ASCEND_RT_VISIBLE_DEVICES"),
    )
    expected_physical_text = str(device["physical_device_id"])
    expected_logical_text = str(device["logical_device_id"])
    if selection["ASCEND_DEVICE_ID"] != expected_logical_text:
        raise ValueError(
            "provenance ASCEND_DEVICE_ID disagrees with logical_device_id"
        )
    if selection["ASCEND_RT_VISIBLE_DEVICES"] != expected_physical_text:
        raise ValueError(
            "provenance ASCEND_RT_VISIBLE_DEVICES disagrees with "
            "physical_device_id"
        )
    npu_smi = raw["npu_smi"]
    _require_mapping(
        npu_smi, "npu_smi",
        (
            "executable", "executable_sha256", "version", "global", "board",
            "usages",
        ),
    )
    _require_nonempty_text(npu_smi["executable"], "npu_smi.executable")
    _require_sha256(npu_smi["executable_sha256"], "npu_smi.executable_sha256")
    _require_nonempty_text(npu_smi["version"], "npu_smi.version")
    expected_commands = {
        "global": [npu_smi["executable"], "info"],
        "board": [
            npu_smi["executable"], "info", "-t", "board", "-i",
            expected_physical_text,
        ],
        "usages": [
            npu_smi["executable"], "info", "-t", "usages", "-i",
            expected_physical_text,
        ],
    }
    for label in ("global", "board", "usages"):
        telemetry = npu_smi[label]
        _require_mapping(
            telemetry, f"npu_smi.{label}",
            ("command", "stdout", "stdout_sha256"),
        )
        if telemetry["command"] != expected_commands[label]:
            raise ValueError(f"provenance npu_smi.{label} command is not exact")
        _require_nonempty_text(telemetry["stdout"], f"npu_smi.{label}.stdout")
        _require_sha256(
            telemetry["stdout_sha256"], f"npu_smi.{label}.stdout_sha256"
        )
        observed_stdout_sha256 = hashlib.sha256(
            telemetry["stdout"].encode("utf-8")
        ).hexdigest()
        if observed_stdout_sha256 != telemetry["stdout_sha256"]:
            raise ValueError(f"provenance npu_smi.{label} stdout SHA256 changed")
    parsed_device = _parse_device_identity(
        npu_smi["global"]["stdout"], npu_smi["board"]["stdout"],
        device["physical_device_id"],
    )
    if _parse_npu_smi_version(npu_smi["global"]["stdout"]) != npu_smi["version"]:
        raise ValueError("retained telemetry disagrees with npu-smi version")
    for field in ("physical_device_id", "model", "pci_identity"):
        if parsed_device[field] != device[field]:
            raise ValueError(f"retained telemetry disagrees with device.{field}")
    parsed_idle = _parse_idle_telemetry(npu_smi["usages"]["stdout"])
    clock = raw["clock"]
    _require_mapping(
        clock, "clock", ("source", "system_counter_hz", "tick_ns", "host_wall_role")
    )
    if (
        clock["source"] != "GetSystemCycle"
        or clock["system_counter_hz"] != SYSTEM_COUNTER_HZ
        or clock["tick_ns"] != SYSTEM_COUNTER_TICK_NS
        or clock["host_wall_role"] != "health_check_only"
    ):
        raise ValueError("provenance has an invalid device-clock contract")
    idle = raw["idle_telemetry"]
    if set(idle) != {"aic_percent", "aiv_percent", "hbm_bandwidth_percent"}:
        raise ValueError("provenance has incomplete idle telemetry")
    if any(
        not isinstance(idle[field], (int, float))
        or isinstance(idle[field], bool)
        or idle[field] != 0.0
        for field in idle
    ):
        raise ValueError("provenance does not prove an idle selected device")
    if parsed_idle != idle:
        raise ValueError("retained usages telemetry disagrees with idle summary")
    _require_sha256(raw["idle_telemetry_sha256"], "idle_telemetry_sha256")
    if raw["idle_telemetry_sha256"] != npu_smi["usages"]["stdout_sha256"]:
        raise ValueError("idle telemetry SHA256 disagrees with retained usages")
    _require_nonempty_text(raw["measurement_scope"], "measurement_scope")
    _require_nonempty_text(raw["publication_limit"], "publication_limit")
    capture = raw["capture"]
    _require_mapping(
        capture, "capture",
        (
            "raw_output", "raw_sha256", "raw_row_count", "argv",
            "working_directory", "runner_pid",
        ),
    )
    _require_nonempty_text(capture["raw_output"], "capture.raw_output")
    _require_sha256(capture["raw_sha256"], "capture.raw_sha256")
    if (
        not isinstance(capture["raw_row_count"], int)
        or isinstance(capture["raw_row_count"], bool)
        or capture["raw_row_count"] <= 0
    ):
        raise ValueError("provenance capture.raw_row_count must be positive")
    if (
        not isinstance(capture["argv"], list)
        or not capture["argv"]
        or any(not isinstance(value, str) for value in capture["argv"])
    ):
        raise ValueError("provenance capture.argv must be an exact argv list")
    argv = capture["argv"]
    _require_nonempty_text(
        capture["working_directory"], "capture.working_directory"
    )
    working_directory = Path(capture["working_directory"])
    if not working_directory.is_absolute():
        raise ValueError("provenance capture working directory must be absolute")

    def resolve_argv_path(value):
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate.resolve()
        return (working_directory / candidate).resolve()

    if len(argv) < 2 or argv[1] != "run":
        raise ValueError("provenance capture.argv must identify the run command")
    runner_path = resolve_argv_path(argv[0])
    if resolve_argv_path(_argv_option(argv, "--raw-output")) != Path(
        capture["raw_output"]
    ).resolve():
        raise ValueError("provenance argv/raw output binding is inconsistent")
    if resolve_argv_path(_argv_option(argv, "--provenance-output")) != path.resolve():
        raise ValueError("provenance argv/provenance output binding is inconsistent")
    if _argv_option(argv, "--run-id") != raw["run_id"]:
        raise ValueError("provenance argv/run_id binding is inconsistent")
    if _argv_option(argv, "--source-commit") != raw["source_commit"]:
        raise ValueError("provenance argv/source commit binding is inconsistent")
    if (
        _argv_option(argv, "--expected-source-snapshot-sha256")
        != raw["source_snapshot_sha256"]
    ):
        raise ValueError("provenance argv/source snapshot binding is inconsistent")
    if (
        int(_argv_option(argv, "--device-id"))
        != device["physical_device_id"]
    ):
        raise ValueError(
            "provenance argv/physical device binding is inconsistent"
        )
    if resolve_argv_path(_argv_option(argv, "--npu-smi")) != Path(
        npu_smi["executable"]
    ).resolve():
        raise ValueError("provenance argv/npu-smi binding is inconsistent")
    if resolve_argv_path(_argv_option(argv, "--build-directory")) != (
            build_directory.resolve()):
        raise ValueError("provenance argv/build directory binding is inconsistent")
    if resolve_argv_path(_argv_option(argv, "--ascend-home-path")) != Path(
            build["ascend_home_path"]).resolve():
        raise ValueError("provenance argv/Ascend home binding is inconsistent")
    tool_flags = {
        "cmake": "--cmake",
        "make": "--make-program",
        "cxx": "--cxx-compiler",
        "ar": "--archive-tool",
        "ranlib": "--ranlib",
    }
    for name, flag in tool_flags.items():
        if resolve_argv_path(_argv_option(argv, flag)) != Path(
                tools[name]["path"]).resolve():
            raise ValueError(
                f"provenance argv/build tool {name} binding is inconsistent")
    expected_configure_argv = [
        tools["cmake"]["path"], "-S", str(runner_path.parent),
        "-B", str(build_directory), "-G", "Unix Makefiles",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_MAKE_PROGRAM={tools['make']['path']}",
        f"-DCMAKE_CXX_COMPILER={tools['cxx']['path']}",
        f"-DCMAKE_AR={tools['ar']['path']}",
        f"-DCMAKE_RANLIB={tools['ranlib']['path']}",
        f"-DASCEND_HOME_PATH={build['ascend_home_path']}",
        f"-DASCEND_CANN_PACKAGE_PATH={build['ascend_home_path']}",
        "-DSOC_VERSION=Ascend910B2",
    ]
    expected_build_argv = [
        tools["cmake"]["path"], "--build", str(build_directory),
        "--clean-first",
        "--target", "fixpipe_service", "--parallel", "1",
    ]
    if build["configure_argv"] != expected_configure_argv:
        raise ValueError("provenance configure argv is not source-bound")
    if build["build_argv"] != expected_build_argv:
        raise ValueError("provenance build argv is not source-bound")
    if resolve_argv_path(_argv_option(argv, "--cann-version-file")) != Path(
        cann["version_file"]
    ).resolve():
        raise ValueError("provenance argv/CANN version binding is inconsistent")
    source_flags = {
        "fixpipe_interface": "--fixpipe-interface",
        "fixpipe_intf_impl": "--fixpipe-intf-impl",
        "fixpipe_dav_c220_v2_impl": "--fixpipe-dav-c220-v2-impl",
        "production_fa_source": "--production-fa-source",
        "production_fa_kernel": "--production-fa-kernel",
        "fixpipe_struct": "--fixpipe-struct",
    }
    for label, flag in source_flags.items():
        if resolve_argv_path(_argv_option(argv, flag)) != Path(
            identities[label]["path"]
        ).resolve():
            raise ValueError(f"provenance argv/{label} binding is inconsistent")
    if resolve_argv_path(
            _argv_option(argv, "--production-geometry-source")) != Path(
                geometry["path"]).resolve():
        raise ValueError(
            "provenance argv/production geometry binding is inconsistent")
    trial_count = int(_argv_option(argv, "--trials"))
    if trial_count < 6 or trial_count % 2 != 0:
        raise ValueError("provenance argv trial count is invalid")
    if capture["raw_row_count"] != trial_count * len(CELLS):
        raise ValueError("provenance argv trial count disagrees with raw row count")
    if (
        not isinstance(capture["runner_pid"], int)
        or isinstance(capture["runner_pid"], bool)
        or capture["runner_pid"] <= 0
    ):
        raise ValueError("provenance capture.runner_pid must be positive")
    raw["provenance_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return raw


def _expected_cells():
    return {
        (
            family, layout, m, n, dst_stride, mode, depth, block_dim,
            destination_slots,
        )
        for (
            family, _layout_id, layout, m, n, dst_stride, _mode_id, mode,
            depth, block_dim, destination_slots,
        ) in CELLS
    }


def _load_rows(paths, provenance):
    rows = []
    identities = set()
    expected_cells = _expected_cells()
    if len(paths) != 1:
        raise ValueError("report requires exactly one provenance-bound raw CSV")
    path = paths[0]
    capture = provenance["capture"]
    if path.resolve() != Path(capture["raw_output"]).resolve():
        raise ValueError("raw CSV path does not match provenance capture identity")
    if hashlib.sha256(path.read_bytes()).hexdigest() != capture["raw_sha256"]:
        raise ValueError("raw CSV SHA256 does not match provenance capture identity")
    for path in paths:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != REQUIRED_FIELDS:
                raise ValueError(f"{path} has an invalid raw schema")
            for raw in reader:
                if raw["validation"] != "pass":
                    raise ValueError(f"{path} contains failed tile validation")
                if raw["clock_source"] != "GetSystemCycle":
                    raise ValueError(f"{path} did not use the frozen device clock")
                if raw["completion"] != "pipe_fix_completion":
                    raise ValueError(f"{path} changed the completion contract")
                integers = {
                    field: int(raw[field])
                    for field in (
                        "trial", "order_position", "process_pid", "m", "n",
                        "src_stride", "dst_stride", "stream_depth", "block_dim",
                        "destination_slots", "base_invocations",
                        "middle_invocations", "long_invocations",
                        "base_logical_bytes_per_aic",
                        "middle_logical_bytes_per_aic",
                        "long_logical_bytes_per_aic", "base_host_wall_ns",
                        "middle_host_wall_ns", "long_host_wall_ns",
                    )
                }
                if integers["trial"] <= 0 or integers["process_pid"] <= 0:
                    raise ValueError(f"{path} has invalid process identity")
                if raw["order_direction"] not in ORDER_DIRECTIONS:
                    raise ValueError(f"{path} has invalid order direction")
                cell = (
                    raw["family"], raw["layout"], integers["m"], integers["n"],
                    integers["dst_stride"], raw["mode"],
                    integers["stream_depth"], integers["block_dim"],
                    integers["destination_slots"],
                )
                if cell not in expected_cells:
                    raise ValueError(f"{path} contains unexpected cell {cell}")
                expected_src, expected_dst = _strides(
                    raw["layout"], integers["m"], integers["n"],
                    integers["dst_stride"],
                )
                expected_invocations = _invocations(
                    integers["m"], integers["n"], integers["stream_depth"]
                )
                expected_base_bytes = (
                    expected_invocations * integers["m"] * integers["n"] * 4
                )
                if (
                    integers["src_stride"] != expected_src
                    or integers["dst_stride"] != expected_dst
                    or integers["base_invocations"] != expected_invocations
                    or integers["middle_invocations"] != 2 * expected_invocations
                    or integers["long_invocations"] != 4 * expected_invocations
                    or integers["base_logical_bytes_per_aic"]
                    != expected_base_bytes
                    or integers["middle_logical_bytes_per_aic"]
                    != 2 * expected_base_bytes
                    or integers["long_logical_bytes_per_aic"]
                    != 4 * expected_base_bytes
                    or integers["base_host_wall_ns"] <= 0
                    or integers["middle_host_wall_ns"] <= 0
                    or integers["long_host_wall_ns"] <= 0
                ):
                    raise ValueError(f"{path} changed frozen geometry or intervals")
                base_cycles = _cycles(
                    raw["base_block_cycles"], integers["block_dim"],
                    "base_block_cycles",
                )
                middle_cycles = _cycles(
                    raw["middle_block_cycles"], integers["block_dim"],
                    "middle_block_cycles",
                )
                long_cycles = _cycles(
                    raw["long_block_cycles"], integers["block_dim"],
                    "long_block_cycles",
                )
                identity = (
                    raw["run_id"], integers["trial"], integers["order_position"]
                )
                if identity in identities:
                    raise ValueError(f"duplicate raw row identity {identity}")
                identities.add(identity)
                rows.append(
                    {
                        "run_id": raw["run_id"],
                        "trial": integers["trial"],
                        "process_pid": integers["process_pid"],
                        "order_direction": raw["order_direction"],
                        "order_position": integers["order_position"],
                        "family": raw["family"],
                        "layout": raw["layout"],
                        "m": integers["m"],
                        "n": integers["n"],
                        "dst_stride": integers["dst_stride"],
                        "mode": raw["mode"],
                        "stream_depth": integers["stream_depth"],
                        "block_dim": integers["block_dim"],
                        "destination_slots": integers["destination_slots"],
                        "base_invocations": integers["base_invocations"],
                        "middle_invocations": integers["middle_invocations"],
                        "long_invocations": integers["long_invocations"],
                        "base_logical_bytes": integers[
                            "base_logical_bytes_per_aic"
                        ],
                        "middle_logical_bytes": integers[
                            "middle_logical_bytes_per_aic"
                        ],
                        "long_logical_bytes": integers[
                            "long_logical_bytes_per_aic"
                        ],
                        "base_cycles": base_cycles,
                        "middle_cycles": middle_cycles,
                        "long_cycles": long_cycles,
                    }
                )
    if not rows:
        raise ValueError("no Fixpipe raw rows were provided")
    if len(rows) != capture["raw_row_count"]:
        raise ValueError("raw CSV row count does not match provenance capture identity")
    return rows


def _validate_completeness(rows):
    expected_cells = _expected_cells()
    by_trial = {}
    positions = {}
    process_ids = {}
    trials_by_run = {}
    process_ids_by_run = {}
    for row in rows:
        key = (row["run_id"], row["trial"])
        if key not in by_trial:
            by_trial[key] = set()
            positions[key] = set()
            process_ids[key] = set()
        if row["run_id"] not in trials_by_run:
            trials_by_run[row["run_id"]] = set()
            process_ids_by_run[row["run_id"]] = set()
        trials_by_run[row["run_id"]].add(row["trial"])
        process_ids_by_run[row["run_id"]].add(row["process_pid"])
        cell = (
            row["family"], row["layout"], row["m"], row["n"],
            row["dst_stride"],
            row["mode"], row["stream_depth"], row["block_dim"],
            row["destination_slots"],
        )
        if cell in by_trial[key] or row["order_position"] in positions[key]:
            raise ValueError(f"trial {key} duplicates a cell or order position")
        by_trial[key].add(cell)
        positions[key].add(row["order_position"])
        process_ids[key].add(row["process_pid"])
        expected_direction, ordered_cells = _ordered_cells(row["trial"] - 1)
        if row["order_direction"] != expected_direction:
            raise ValueError(f"trial {key} changed the frozen order direction")
        if row["order_position"] < 0 or row["order_position"] >= len(CELLS):
            raise ValueError(f"trial {key} has an invalid order position")
        registry_cell = ordered_cells[row["order_position"]]
        expected_cell = (
            registry_cell[0], registry_cell[2], registry_cell[3],
            registry_cell[4], registry_cell[5], registry_cell[7],
            registry_cell[8], registry_cell[9], registry_cell[10],
        )
        if cell != expected_cell:
            raise ValueError(
                f"trial {key} position {row['order_position']} changed "
                "the preregistered cell mapping"
            )
    for key, cells in by_trial.items():
        if cells != expected_cells or positions[key] != set(range(len(CELLS))):
            raise ValueError(f"trial {key} is incomplete")
        if len(process_ids[key]) != 1:
            raise ValueError(f"trial {key} must have one positive process PID")
    for run_id, trials in trials_by_run.items():
        maximum_trial = max(trials)
        if (
            maximum_trial < 6
            or maximum_trial % 2 != 0
            or trials != set(range(1, maximum_trial + 1))
        ):
            raise ValueError(
                f"run {run_id} trials must be contiguous, even, and at least six"
            )
        if len(process_ids_by_run[run_id]) != len(trials):
            raise ValueError(
                f"run {run_id} must use one independent child process per trial")
    directions = [
        _ordered_cells(key[1] - 1)[0] for key in by_trial
    ]
    if directions.count("forward") != directions.count("reverse"):
        raise ValueError("raw trials do not balance forward and reverse order")
    return len(by_trial)


def _physical_features(m, n):
    return (
        1.0,
        float(m),
        float(((m + 15) // 16) * ((n + 15) // 16)),
        float(m * ((n + 7) // 8)),
    )


def _serial_observations(rows, layout):
    observations = []
    for row in rows:
        if row["layout"] != layout or row["mode"] != "serial":
            continue
        base = row["base_invocations"]
        if (row["middle_invocations"] != 2 * base
                or row["long_invocations"] != 4 * base):
            raise ValueError("Fixpipe interval multipliers changed")
        for block_index in range(row["block_dim"]):
            base_cycles = row["base_cycles"][block_index]
            middle_cycles = row["middle_cycles"][block_index]
            long_cycles = row["long_cycles"][block_index]
            first_slope = (middle_cycles - base_cycles) / base
            second_slope = (long_cycles - middle_cycles) / (2 * base)
            service_cycles = (long_cycles - base_cycles) / (3 * base)
            if first_slope <= 0 or second_slope <= 0 or service_cycles <= 0:
                raise ValueError("Fixpipe interval slopes must be positive")
            observations.append({
                "run_id": row["run_id"],
                "trial": row["trial"],
                "family": row["family"],
                "m": row["m"],
                "n": row["n"],
                "dst_stride": row["dst_stride"],
                "destination_slots": row["destination_slots"],
                "direction": row["order_direction"],
                "block_dim": row["block_dim"],
                "block_index": block_index,
                "features": _physical_features(row["m"], row["n"]),
                "service_cycles": service_cycles,
                "interval_error": _relative_error(
                    first_slope, second_slope),
            })
    return observations


def _source_full_tile_audit(observations):
    """Fit and validate the source-used complete 128-column row-major call.

    CANN standard GQA emits many complete ``M x 128`` score Fixpipe calls.
    The hardware service has a four-phase destination-row effect, so the
    generic byte/fractal affine model is deliberately not reused here.  The
    calibration uses only independent primitive timings: M=1/5/8 establish
    fixed plus per-row service, while a bounded M=4 stride sweep establishes
    the two nontrivial row-address phase penalties.  M=4 aligned calls,
    distant strides, reverse order, and 24-AIC execution remain holdouts.
    """

    points = [
        point for point in observations
        if point["family"] == "production_holdout"
        and point["n"] == 128
        and point["m"] in (1, 4, 5, 8)
    ]
    for point in points:
        point["row_address_phases"] = min(
            point["m"],
            128 // math.gcd(point["dst_stride"], 128),
        )

    base_calibration = [
        point for point in points
        if point["direction"] == "forward"
        and point["block_dim"] == CALIBRATION_BLOCK_DIM
        and point["m"] in (1, 5, 8)
        and point["row_address_phases"] == 1
    ]
    stride_calibration = [
        point for point in points
        if point["direction"] == "forward"
        and point["block_dim"] == CALIBRATION_BLOCK_DIM
        and point["m"] == 4
        and point["row_address_phases"] in (2, 4)
        and point["dst_stride"] <= 1024
    ]
    if not base_calibration or not stride_calibration:
        raise ValueError("source full-tile Fixpipe calibration is incomplete")

    design = np.asarray(
        [[1.0, float(point["m"])] for point in base_calibration],
        dtype=float,
    )
    observed = np.asarray(
        [point["service_cycles"] for point in base_calibration],
        dtype=float,
    )
    base_coefficients, _residuals, rank, _singular = np.linalg.lstsq(
        design, observed, rcond=None)
    call_startup_cycles, row_cycles = (
        float(value) for value in base_coefficients)

    phase_penalty_cycles = {}
    for phase_count in (2, 4):
        residuals = [
            point["service_cycles"]
            - call_startup_cycles - row_cycles * point["m"]
            for point in stride_calibration
            if point["row_address_phases"] == phase_count
        ]
        if not residuals:
            raise ValueError(
                f"source full-tile Fixpipe phase {phase_count} is missing")
        phase_penalty_cycles[phase_count] = statistics.median(residuals)

    def prediction(point):
        phases = point["row_address_phases"]
        if phases == 1:
            penalty = 0.0
        elif phases in phase_penalty_cycles:
            penalty = phase_penalty_cycles[phases]
        else:
            raise ValueError(
                "source full-tile Fixpipe encountered an unmeasured row "
                f"address phase count: {phases}")
        return call_startup_cycles + row_cycles * point["m"] + penalty

    def maximum_error(selected):
        if not selected:
            raise ValueError("source full-tile Fixpipe holdout is empty")
        return max(
            abs(prediction(point) - point["service_cycles"])
            / point["service_cycles"]
            for point in selected
        )

    holdouts = {
        "m4_aligned": [
            point for point in points
            if point["direction"] == "forward"
            and point["block_dim"] == CALIBRATION_BLOCK_DIM
            and point["m"] == 4
            and point["row_address_phases"] == 1
        ],
        "distant_stride": [
            point for point in points
            if point["direction"] == "forward"
            and point["block_dim"] == CALIBRATION_BLOCK_DIM
            and point["m"] == 4
            and point["row_address_phases"] in (2, 4)
            and point["dst_stride"] > 1024
        ],
        "reverse_order": [
            point for point in points
            if point["direction"] == "reverse"
            and point["block_dim"] == CALIBRATION_BLOCK_DIM
        ],
        "block_dim_24": [
            point for point in points
            if point["direction"] == "forward"
            and point["block_dim"] == HOLDOUT_BLOCK_DIM
        ],
    }
    errors = {
        name: maximum_error(selected)
        for name, selected in holdouts.items()
    }
    grouped_trial_blocks = {}
    for point in base_calibration + stride_calibration:
        key = (
            point["m"], point["n"], point["dst_stride"],
            point["direction"], point["block_dim"],
        )
        if key not in grouped_trial_blocks:
            grouped_trial_blocks[key] = {}
        trials = grouped_trial_blocks[key]
        if point["trial"] not in trials:
            trials[point["trial"]] = []
        trials[point["trial"]].append(point["service_cycles"])
    trial_median_groups = [
        [statistics.median(values) for values in trials.values()]
        for trials in grouped_trial_blocks.values()
    ]
    trial_span = max(
        (max(trial_medians) - min(trial_medians))
        / statistics.median(trial_medians)
        for trial_medians in trial_median_groups
    )
    gates = {
        "positive_physical_parameters": (
            call_startup_cycles > 0
            and row_cycles > 0
            and all(value > 0 for value in phase_penalty_cycles.values())
        ),
        "full_rank_base_design": bool(rank == 2),
        "independent_trial_span": (
            trial_span <= SOURCE_FULL_TILE_TRIAL_SPAN_LIMIT),
        **{
            f"{name}_holdout": value <= SOURCE_FULL_TILE_ERROR_LIMIT
            for name, value in errors.items()
        },
    }

    # The public row-major primitive changes from the short-row address-phase
    # regime to a stable row-streaming regime at the source-defined 16-row
    # fractal boundary.  M=16/32/128 are the preregistered independent
    # calibration sweep; the production M=64 geometry remains a strict
    # holdout.  No production-kernel duration participates in either fit.
    large_calibration = [
        point for point in observations
        if point["family"] == "large_row_holdout"
        and point["direction"] == "forward"
        and point["block_dim"] == CALIBRATION_BLOCK_DIM
        and point["m"] in (16, 32, 128)
        and point["n"] == 128
        and point["dst_stride"] == 512
    ]
    if not large_calibration:
        raise ValueError("large-row Fixpipe calibration is incomplete")
    large_design = np.asarray(
        [[1.0, float(point["m"])] for point in large_calibration],
        dtype=float,
    )
    large_observed = np.asarray(
        [point["service_cycles"] for point in large_calibration],
        dtype=float,
    )
    large_coefficients, _residuals, large_rank, _singular = np.linalg.lstsq(
        large_design, large_observed, rcond=None)
    large_call_startup_cycles, large_row_cycles = (
        float(value) for value in large_coefficients)

    def large_prediction(point):
        return large_call_startup_cycles + large_row_cycles * point["m"]

    def large_maximum_error(selected):
        if not selected:
            raise ValueError("large-row Fixpipe holdout is empty")
        return max(
            abs(large_prediction(point) - point["service_cycles"])
            / point["service_cycles"]
            for point in selected
        )

    large_holdouts = {
        "large_calibration": large_calibration,
        "large_reverse_order": [
            point for point in observations
            if point["family"] == "large_row_holdout"
            and point["direction"] == "reverse"
            and point["block_dim"] == CALIBRATION_BLOCK_DIM
            and point["m"] in (16, 32, 128)
            and point["n"] == 128
            and point["dst_stride"] == 512
        ],
        "large_block_dim_24": [
            point for point in observations
            if point["family"] in ("large_row_holdout", "production_holdout")
            and point["block_dim"] == HOLDOUT_BLOCK_DIM
            and point["m"] in (16, 32, 64, 128)
            and point["n"] == 128
            and point["dst_stride"] == 512
        ],
        "mla_m64": [
            point for point in observations
            if point["family"] == "production_holdout"
            and point["block_dim"] == CALIBRATION_BLOCK_DIM
            and point["m"] == 64
            and point["n"] == 128
            and point["dst_stride"] == 512
        ],
    }
    large_errors = {
        name: large_maximum_error(selected)
        for name, selected in large_holdouts.items()
    }
    large_trial_span = _maximum_trial_span(large_calibration)
    large_gates = {
        "large_positive_physical_parameters": (
            large_call_startup_cycles > 0 and large_row_cycles > 0),
        "large_full_rank_design": bool(large_rank == 2),
        "large_independent_trial_span": (
            large_trial_span <= SOURCE_FULL_TILE_TRIAL_SPAN_LIMIT),
        **{
            f"{name}_holdout": value <= SOURCE_FULL_TILE_ERROR_LIMIT
            for name, value in large_errors.items()
        },
    }
    gates.update(large_gates)
    publishable = all(gates.values())
    candidate = None
    if publishable:
        candidate = {
            "formula": (
                "M<16: call_startup + M*row_service + "
                "row_address_phase_penalty[min(M,128/gcd(dst_stride,128))]; "
                "M>=16: large_row_call_startup + M*large_row_service"
            ),
            "parameters_ns": {
                "call_startup": call_startup_cycles * SYSTEM_COUNTER_TICK_NS,
                "row_service": row_cycles * SYSTEM_COUNTER_TICK_NS,
                "row_address_phase_2_penalty": (
                    phase_penalty_cycles[2] * SYSTEM_COUNTER_TICK_NS),
                "row_address_phase_4_penalty": (
                    phase_penalty_cycles[4] * SYSTEM_COUNTER_TICK_NS),
                "large_row_call_startup": (
                    large_call_startup_cycles * SYSTEM_COUNTER_TICK_NS),
                "large_row_service": (
                    large_row_cycles * SYSTEM_COUNTER_TICK_NS),
            },
            "dtype": "fp32",
            "layout": "row_major",
            "column_count": 128,
            "supported_rows": [1, 4, 5, 8, 64],
            "large_row_minimum_rows": 16,
            "destination_stride_alignment_elements": 32,
            "maximum_destination_stride_elements": 8192,
            "service_semantics": "completed_serial_public_fixpipe_call",
        }
    return {
        "gates": gates,
        "maximum_relative_errors": {
            **errors,
            "trial_span": trial_span,
            **large_errors,
            "large_trial_span": large_trial_span,
        },
        "mechanism_publishable": publishable,
        "candidate": candidate,
    }


def _fit_physical_model(points):
    if len(points) < 8:
        raise ValueError("physical Fixpipe calibration has too few points")
    design = np.asarray([point["features"] for point in points], dtype=float)
    observed = np.asarray(
        [point["service_cycles"] for point in points], dtype=float)
    coefficients, _residuals, rank, singular = np.linalg.lstsq(
        design, observed, rcond=None)
    condition = float(singular[0] / singular[-1])
    return tuple(float(value) for value in coefficients), int(rank), condition


def _maximum_model_error(points, coefficients):
    if not points:
        raise ValueError("a preregistered gate has no observations")
    errors = []
    for point in points:
        prediction = float(np.dot(point["features"], coefficients))
        if prediction <= 0:
            return None
        errors.append(_relative_error(point["service_cycles"], prediction))
    return max(errors)


def _maximum_trial_span(points):
    grouped = {}
    for point in points:
        key = (
            point["family"], point["m"], point["n"], point["dst_stride"],
            point["destination_slots"], point["block_dim"],
            point["block_index"],
        )
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(point["service_cycles"])
    if not grouped:
        raise ValueError("trial-span gate has no observations")
    return max(
        (max(values) - min(values)) / statistics.median(values)
        for values in grouped.values()
    )


def _serial_audit(rows, layout):
    observations = _serial_observations(rows, layout)

    def select(family, direction="forward", block_dim=CALIBRATION_BLOCK_DIM):
        return [
            point for point in observations
            if point["family"] == family
            and point["direction"] == direction
            and point["block_dim"] == block_dim
        ]

    calibration = select("synthetic_calibration")
    mechanism_holdout = select("synthetic_holdout")
    production_holdout = select("production_holdout")
    reverse_holdout = select("production_holdout", direction="reverse")
    block_holdout = select(
        "production_holdout", block_dim=HOLDOUT_BLOCK_DIM)
    stride_holdout = select("stride_holdout")
    reuse_holdout = select("reuse_holdout")
    coefficients, rank, condition = _fit_physical_model(calibration)
    calibration_error = _maximum_model_error(calibration, coefficients)
    mechanism_error = _maximum_model_error(
        mechanism_holdout, coefficients)
    production_error = _maximum_model_error(
        production_holdout, coefficients)
    reverse_error = _maximum_model_error(reverse_holdout, coefficients)
    block_error = _maximum_model_error(block_holdout, coefficients)
    stride_error = _maximum_model_error(stride_holdout, coefficients)
    reuse_error = _maximum_model_error(reuse_holdout, coefficients)
    interval_error = max(
        point["interval_error"] for point in observations)
    trial_span = _maximum_trial_span(observations)

    def within(value, limit):
        return value is not None and value <= limit

    gates = {
        "positive_physical_parameters": all(
            value > 0 for value in coefficients),
        "full_rank_predeclared_design": rank == 4,
        "bounded_design_condition": condition <= MAX_DESIGN_CONDITION,
        "predefined_completion": True,
        "three_interval_consistency": (
            interval_error <= INTERVAL_CONSISTENCY_LIMIT),
        "independent_trial_span": trial_span <= RUN_SPAN_LIMIT,
        "synthetic_calibration_residual": within(
            calibration_error, CALIBRATION_RESIDUAL_LIMIT),
        "synthetic_mechanism_holdout": within(
            mechanism_error, MECHANISM_HOLDOUT_LIMIT),
        "production_geometry_holdout": within(
            production_error, SHAPE_HOLDOUT_LIMIT),
        "reverse_order_holdout": within(
            reverse_error, ORDER_HOLDOUT_LIMIT),
        "block_dim_24_holdout": within(
            block_error, BLOCK_HOLDOUT_LIMIT),
        "stride_holdout": within(stride_error, STRIDE_HOLDOUT_LIMIT),
        "destination_reuse_holdout": within(
            reuse_error, REUSE_HOLDOUT_LIMIT),
    }
    publishable = all(gates.values())
    candidate = None
    if publishable:
        names = (
            "call_startup", "row_burst", "l0c_fractal_16x16",
            "destination_sector_32b",
        )
        candidate = {
            "formula": (
                "call_startup + M*row_burst + "
                "ceil(M/16)*ceil(N/16)*l0c_fractal_16x16 + "
                "M*ceil(N/8)*destination_sector_32b"
            ),
            "parameters_cycles": dict(zip(names, coefficients)),
            "parameters_ns": {
                name: value * SYSTEM_COUNTER_TICK_NS
                for name, value in zip(names, coefficients)
            },
            "dtype": "fp32",
            "layout": "row_major",
            "destination_base_alignment_bytes": 512,
            "destination_slot_range": [2, 16],
            "service_semantics": "completed_serial_public_fixpipe_call",
            "supported_geometry": [
                {"m": m, "n": n, "dst_stride": dst_stride}
                for m, n, dst_stride in PRODUCTION_ROW_MAJOR_GEOMETRY
            ],
        }
    return {
        "gates": gates,
        "fit_diagnostics": {
            "rank": rank,
            "condition_number": condition,
        },
        "maximum_relative_errors": {
            "interval": interval_error,
            "trial_span": trial_span,
            "calibration": calibration_error,
            "synthetic_mechanism_holdout": mechanism_error,
            "production_geometry_holdout": production_error,
            "reverse_order_holdout": reverse_error,
            "block_dim_24_holdout": block_error,
            "stride_holdout": stride_error,
            "destination_reuse_holdout": reuse_error,
        },
        "mechanism_publishable": publishable,
        "candidate": candidate,
        "source_full_tile": _source_full_tile_audit(observations),
    }


def _audit(rows, provenance):
    trial_count = _validate_completeness(rows)
    run_ids = {row["run_id"] for row in rows}
    if run_ids != {provenance["run_id"]}:
        raise ValueError("raw run_id does not match provenance")
    serial = _serial_audit(rows, "row_major")
    static_audited = provenance["static_instruction_contract_audited"]
    serial["gates"]["static_instruction_contract_audited"] = static_audited
    if not static_audited:
        serial["mechanism_publishable"] = False
        serial["candidate"] = None
    layouts = {
        "row_major": {
            "serial": serial,
            "mechanism_publishable": serial["mechanism_publishable"],
        },
    }
    mechanism_publishable = (
        static_audited and serial["mechanism_publishable"])
    source_full_tile_publishable = (
        static_audited
        and serial["source_full_tile"]["mechanism_publishable"])
    return {
        "schema": "ascend_910b_fixpipe_service_report_v2",
        "run_id": provenance["run_id"],
        "source_commit": provenance["source_commit"],
        "source_snapshot_sha256": provenance["source_snapshot_sha256"],
        "provenance_sha256": provenance["provenance_sha256"],
        "raw_sha256": provenance["capture"]["raw_sha256"],
        "trial_count": trial_count,
        "layouts": layouts,
        "mechanism_publishable": mechanism_publishable,
        "source_full_tile_publishable": source_full_tile_publishable,
        "scope": (
            "FP32 row-major direct L0C-to-GM public Fixpipe completed serial "
            "service with independent synthetic calibration"),
        "limitations": [
            "no asynchronous queue-depth or credit claim",
            "no production overlap claim",
            "no GM residency claim",
            "512-byte-aligned destination bases only",
            "destination slot domain is two through sixteen",
            "candidate is restricted to the published geometry domain",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", nargs="+", type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    provenance = _load_provenance(args.provenance)
    rows = _load_rows(args.raw_csv, provenance)
    audit = _audit(rows, provenance)
    payload = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.json_output is None:
        print(payload, end="")
    else:
        if args.json_output.exists():
            raise FileExistsError(f"report output exists: {args.json_output}")
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
