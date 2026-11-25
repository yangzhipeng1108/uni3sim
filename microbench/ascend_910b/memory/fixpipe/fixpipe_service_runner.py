#!/usr/bin/env python3
"""Run the preregistered Ascend 910B direct-Fixpipe mechanism campaign."""

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


STANDARD_GQA_S2_BY_M = {
    1: (512, 1024, 2048, 4096),
    4: (
        128, 256, 280, 304, 307, 328, 352, 358, 376, 400, 409, 424,
        448, 460, 472, 496, 511, 512, 520, 544, 562, 568, 592, 613,
        614, 616, 640, 664, 688, 712, 715, 716, 736, 760, 766, 768,
        784, 808, 817, 818, 832, 856, 868, 880, 904, 919, 920, 928,
        952, 970, 976, 1000, 1021, 1022, 1024, 1124, 1226, 1328,
        1430, 1532, 1536, 1634, 1736, 1838, 1940, 2042, 2048, 3072,
        4096, 8192,
    ),
    5: (512, 1024, 2048, 4096),
    8: (512, 1024, 2048, 4096),
}
MM2_SHAPES = ((1, 128), (4, 64), (4, 128), (5, 128), (8, 128))


def _production_row_major_geometry():
    geometry = set()
    for m, s2_values in STANDARD_GQA_S2_BY_M.items():
        for s2 in s2_values:
            dst_stride = (s2 + 31) // 32 * 32
            geometry.add((m, 128, dst_stride))
            tail = s2 % 128
            if tail:
                geometry.add((m, tail, dst_stride))
    for m, n in MM2_SHAPES:
        geometry.add((m, n, (n + 31) // 32 * 32))
    # FiaBlockCubeNonQuantMla emits four M=64,N=128 row-major Fixpipe calls
    # into one 512-column FP32 workspace.  This is a mechanism holdout, not a
    # production duration: only the public L0C->GM primitive is timed.
    geometry.add((64, 128, 512))
    result = tuple(sorted(geometry))
    if len(result) != 115:
        raise AssertionError(
            "FIA Fixpipe registry must contain 115 geometries")
    return result


PRODUCTION_ROW_MAJOR_GEOMETRY = _production_row_major_geometry()
CALIBRATION_GEOMETRY = tuple(
    (m, n, (n + 31) // 32 * 32)
    for m in (2, 3, 7, 16)
    for n in (1, 7, 9, 15, 17, 31, 33, 63, 65, 95, 97, 119, 121)
)
MECHANISM_HOLDOUT_GEOMETRY = tuple(
    (m, n, (n + 31) // 32 * 32)
    for m in (6, 10, 15)
    for n in (3, 13, 27, 55, 87, 111, 123)
)
LARGE_ROW_HOLDOUT_GEOMETRY = tuple(
    (m, 128, 512) for m in (16, 32, 128)
)
SENTINEL_GEOMETRY = (
    (2, 17, 32),
    (7, 63, 64),
    (16, 121, 128),
)
DESTINATION_SLOT_COUNTS = (2, 4, 16)
CALIBRATION_DESTINATION_SLOTS = 16
STREAM_DEPTHS = (1,)
BLOCK_DIMS = (1, 24)
CALIBRATION_BLOCK_DIM = 1
HOLDOUT_BLOCK_DIM = 24
ORDER_DIRECTIONS = ("forward", "reverse")
SYSTEM_COUNTER_HZ = 50_000_000
SYSTEM_COUNTER_TICK_NS = 20.0
TARGET_SHORT_LOGICAL_BYTES_PER_AIC = 512 * 1024
LARGE_ROW_THRESHOLD = 16
MINIMUM_LARGE_ROW_INVOCATIONS = 128
LAYOUT_ID = 0
LAYOUT = "row_major"
MODE_ID = 0
MODE = "serial"
STREAM_DEPTH = 1


def _invocations(m, n, depth):
    tile_bytes = m * n * 4
    count = (TARGET_SHORT_LOGICAL_BYTES_PER_AIC + tile_bytes - 1) // tile_bytes
    if m >= LARGE_ROW_THRESHOLD:
        count = max(count, MINIMUM_LARGE_ROW_INVOCATIONS)
    count = (count + depth - 1) // depth * depth
    if count <= 0 or count % depth != 0:
        raise AssertionError("invocation count must be positive and depth aligned")
    return count


def _cell(family, geometry, block_dim, destination_slots):
    m, n, dst_stride = geometry
    return (
        family, LAYOUT_ID, LAYOUT, m, n, dst_stride, MODE_ID, MODE,
        STREAM_DEPTH, block_dim, destination_slots,
    )


CALIBRATION_CELLS = tuple(
    _cell(
        "synthetic_calibration", geometry, CALIBRATION_BLOCK_DIM,
        CALIBRATION_DESTINATION_SLOTS)
    for geometry in CALIBRATION_GEOMETRY
)
MECHANISM_HOLDOUT_CELLS = tuple(
    _cell(
        "synthetic_holdout", geometry, CALIBRATION_BLOCK_DIM,
        CALIBRATION_DESTINATION_SLOTS)
    for geometry in MECHANISM_HOLDOUT_GEOMETRY
)
LARGE_ROW_HOLDOUT_CELLS = tuple(
    _cell(
        "large_row_holdout", geometry, block_dim,
        min(
            CALIBRATION_DESTINATION_SLOTS,
            _invocations(geometry[0], geometry[1], STREAM_DEPTH),
        ))
    for geometry in LARGE_ROW_HOLDOUT_GEOMETRY
    for block_dim in BLOCK_DIMS
)
STRIDE_HOLDOUT_CELLS = tuple(
    _cell(
        "stride_holdout", (m, n, dst_stride), CALIBRATION_BLOCK_DIM,
        CALIBRATION_DESTINATION_SLOTS)
    for m, n, natural_stride in SENTINEL_GEOMETRY
    for dst_stride in (128, 512, 2048, 8192)
    if dst_stride != natural_stride
)
REUSE_HOLDOUT_CELLS = tuple(
    _cell(
        "reuse_holdout", geometry, CALIBRATION_BLOCK_DIM,
        destination_slots)
    for geometry in SENTINEL_GEOMETRY
    for destination_slots in DESTINATION_SLOT_COUNTS
    if destination_slots != CALIBRATION_DESTINATION_SLOTS
)
PRODUCTION_HOLDOUT_CELLS = tuple(
    _cell(
        "production_holdout", geometry, block_dim,
        CALIBRATION_DESTINATION_SLOTS)
    for geometry in PRODUCTION_ROW_MAJOR_GEOMETRY
    for block_dim in BLOCK_DIMS
)
CELLS = (
    CALIBRATION_CELLS
    + MECHANISM_HOLDOUT_CELLS
    + LARGE_ROW_HOLDOUT_CELLS
    + STRIDE_HOLDOUT_CELLS
    + REUSE_HOLDOUT_CELLS
    + PRODUCTION_HOLDOUT_CELLS
)
if len(CELLS) != len(set(CELLS)):
    raise AssertionError("Fixpipe campaign contains duplicate cells")
CSV_FIELDS = 24
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
EXPECTED_CANN = {
    "package": "Ascend-cann-toolkit",
    "version": "8.5.1",
    "innerversion": "V100R001C25SPC002B220",
}
EXPECTED_CANN_SOURCE_SHA256 = {
    "fixpipe_interface": (
        "7118fe548d4463b4e9746e61285a396ca9aa25269af97fd049fd4ef193a42ef3"
    ),
    "fixpipe_intf_impl": (
        "20d126f20de9e3e3c3cf62080ccc04a0e1500dfcf88ee3f82f51c21a7f1c9bdb"
    ),
    "fixpipe_dav_c220_v2_impl": (
        "bf77b939a3c91df743f0e11c162b429a7bc871252b48c349527912b715bc0be9"
    ),
    "production_fa_source": (
        "acce194dcf01f370d864ba7987f0c5a9d5927f7a69fb47426afa7789a6f856ef"
    ),
    "production_fa_kernel": (
        "7cde05730d68bb02a2c45cd6b25605417ee61d42dfd87f21ccf0948c014b7e21"
    ),
    "fixpipe_struct": (
        "6e209373437175e4639ebaab69b952cc467b47c477780770134dff91bc8646a3"
    ),
}
EXPECTED_PRODUCTION_GEOMETRY_SHA256 = (
    "59d714d9ed0e83d18cc49039abaddb56592d40d47f2de388cfce44d0193d1ba7"
)
STATIC_INSTRUCTION_SOURCE_MARKERS = {
    "benchmark_kernel": (
        "extern \"C\" __global__ __aicore__ void fixpipe_service(",
        "AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>",
        "AscendC::PipeBarrier<PIPE_FIX>();",
    ),
    "fixpipe_intf_impl": (
        "// L0C->GM",
        "FixpipeL0C2GMImpl<PrimT<T>, PrimT<U>, config>",
        "(__cc__ PrimT<U>*)src.GetPhyAddr()",
    ),
    "fixpipe_dav_c220_v2_impl": (
        "FixpipeL0C2GMImpl(__gm__ T *dst, __cc__ U *src, "
        "const FixpipeParamsV220 &intriParams)",
        "if constexpr (config.format == CO2Layout::ROW_MAJOR)",
        "PipeBarrier<PIPE_FIX>();",
        "FixpipeL0cToOut<T, U, config>",
        "copy_matrix_cc_to_gm",
    ),
    "production_fa_source": (
        "template <CubeFormat OutFormat=CubeFormat::ND, "
        "CubeFormat BFormat=CubeFormat::ND>",
        "template <CubeFormat OutFormat=CubeFormat::ND, "
        "CubeFormat AFormat=CubeFormat::ND>",
        "Fixpipe<MM_OUT_T, T, CFG_NZ>",
        "Fixpipe<MM_OUT_T, T, CFG_ROW_MAJOR>",
        "SetFlag<HardEvent::FIX_M>",
    ),
    "production_fa_kernel": (
        "CubeBlockType matmulService;",
        "matmulService.ComputeMm1(info);",
        "matmulService.ComputeMm2(info);",
    ),
    "fixpipe_struct": (
        "constexpr FixpipeConfig CFG_ROW_MAJOR = {CO2Layout::ROW_MAJOR};",
        "struct FixpipeParamsV220 {",
    ),
}
SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "fixpipe_service_contract.h",
    "fixpipe_service_kernel.cpp",
    "fixpipe_service_main.cpp",
    "fixpipe_service_runner.py",
    "fixpipe_service_report.py",
)
UTILIZATION_PATTERNS = {
    "aic_percent": (
        r"(?:AI\s*Core|AICore)(?:\s*(?:Usage|Utilization|Rate))*\s*\(%\)",
    ),
    "aiv_percent": (
        r"(?:AI\s*Vector|AIVector|Vector\s*Core)"
        r"(?:\s*(?:Usage|Utilization|Rate))*\s*\(%\)",
    ),
    "hbm_bandwidth_percent": (
        r"(?:HBM\s*(?:Bandwidth|BW))"
        r"(?:\s*(?:Usage|Utilization|Rate))*\s*\(%\)",
    ),
}
_ACTIVE_PROCESS = None
LOGICAL_DEVICE_ID = 0
BUILD_ENVIRONMENT_OVERRIDES = (
    "CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "AR",
    "RANLIB", "LD", "NM", "OBJCOPY", "CMAKE_GENERATOR",
    "CMAKE_TOOLCHAIN_FILE", "ASCEND_HOME_PATH", "ASCEND_CANN_PACKAGE_PATH",
    "SOC_VERSION",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_snapshot_sha256(benchmark_root):
    digest = hashlib.sha256()
    for filename in SOURCE_FILENAMES:
        path = benchmark_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"snapshot source is missing: {path}")
        payload = path.read_bytes()
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _positive_integer(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _strides(layout, m, n, dst_stride):
    src_stride = (m + 15) // 16 * 16
    if layout != "row_major":
        raise ValueError(f"unsupported layout: {layout}")
    if dst_stride < n or dst_stride % 32 != 0:
        raise ValueError(
            "row-major destination stride must be a 32-element-aligned "
            "value no smaller than N")
    return src_stride, dst_stride


def _ordered_cells(trial_index):
    if trial_index < 0:
        raise ValueError("trial_index must be non-negative")
    if trial_index % 2 == 0:
        return "forward", CELLS
    return "reverse", tuple(reversed(CELLS))


def _validate_schedule(trials):
    if trials < 6 or trials % 2 != 0:
        raise ValueError("trials must be an even integer greater than or equal to 6")
    directions = []
    for trial_index in range(trials):
        direction, cells = _ordered_cells(trial_index)
        if len(cells) != len(CELLS) or set(cells) != set(CELLS):
            raise AssertionError("trial does not contain the frozen Fixpipe cells")
        directions.append(direction)
    if directions.count("forward") != directions.count("reverse"):
        raise AssertionError("forward/reverse trial counts are not balanced")


def _build_command(binary, cells):
    command = [str(binary), "--batch"]
    for (
            _family, layout_id, layout, m, n, dst_stride, mode_id, _mode,
            depth, block_dim, destination_slots) in cells:
        src_stride, checked_dst_stride = _strides(
            layout, m, n, dst_stride)
        command.extend(
            (
                str(layout_id), str(mode_id), str(depth), str(m), str(n),
                str(src_stride), str(checked_dst_stride), str(block_dim),
                str(destination_slots), str(_invocations(m, n, depth)),
            )
        )
    return command


def _run_checked(command, *, environment=None, timeout=30):
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True,
        env=environment, timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command returned {completed.returncode}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout


def _parse_idle_telemetry(text):
    if not isinstance(text, str) or not text.strip():
        raise TypeError("npu-smi usage telemetry must be non-empty text")
    parsed = {}
    for field, patterns in UTILIZATION_PATTERNS.items():
        values = []
        for pattern in patterns:
            matches = re.findall(
                pattern + r"[^0-9+-]*([0-9]+(?:\.[0-9]+)?)",
                text, flags=re.IGNORECASE,
            )
            values.extend(Decimal(value) for value in matches)
        distinct = set(values)
        if len(distinct) != 1:
            raise ValueError(
                f"npu-smi telemetry must contain one unambiguous {field}, "
                f"found {sorted(str(value) for value in distinct)}"
            )
        value = distinct.pop()
        if value != 0:
            raise RuntimeError(f"selected NPU is busy: {field}={value}")
        parsed[field] = float(value)
    return parsed


def _parse_device_identity(global_text, board_text, physical_device_id):
    rows = re.findall(
        r"^\s*\|\s*([0-9]+)\s+((?:Ascend\s*)?910B2)\s*\|.*$",
        global_text, flags=re.IGNORECASE | re.MULTILINE,
    )
    selected = [
        model for index, model in rows if int(index) == physical_device_id
    ]
    if len(selected) != 1:
        raise ValueError(
            "npu-smi must identify physical device "
            f"{physical_device_id} exactly once as 910B2"
        )
    board_ids = re.findall(
        r"^\s*NPU\s+ID\s*:\s*([0-9]+)\s*$", board_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if board_ids != [str(physical_device_id)]:
        raise ValueError("board telemetry does not match the selected NPU")
    pci = re.findall(
        r"^\s*(?:PCIe?\s+Bus\s+Info|PCIe?\s+Bus-Id|Bus-Id)\s*:\s*"
        r"([0-9a-fA-F:.]+)\s*$", board_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if len(pci) != 1:
        raise ValueError("board telemetry must contain one PCI identity")
    return {
        "physical_device_id": physical_device_id,
        "model": selected[0],
        "pci_identity": pci[0],
    }


def _parse_npu_smi_version(global_text):
    tool_versions = re.findall(
        r"npu-smi\s+([0-9]+(?:\.[0-9]+)+)", global_text,
        flags=re.IGNORECASE,
    )
    header_versions = re.findall(
        r"Version\s*:\s*([0-9]+(?:\.[0-9]+)+)", global_text,
        flags=re.IGNORECASE,
    )
    if (
        len(tool_versions) != 1
        or len(header_versions) != 1
        or tool_versions[0] != header_versions[0]
    ):
        raise ValueError("npu-smi global telemetry has ambiguous version identity")
    return tool_versions[0]


def _npu_smi_commands(npu_smi, physical_device_id):
    if physical_device_id < 0:
        raise ValueError("physical device ID must be non-negative")
    executable = str(npu_smi)
    return (
        [executable, "info"],
        [
            executable, "info", "-t", "board", "-i",
            str(physical_device_id),
        ],
        [
            executable, "info", "-t", "usages", "-i",
            str(physical_device_id),
        ],
    )


def _device_selection_environment(physical_device_id):
    if physical_device_id < 0:
        raise ValueError("physical device ID must be non-negative")
    return {
        "ASCEND_DEVICE_ID": str(LOGICAL_DEVICE_ID),
        "ASCEND_RT_VISIBLE_DEVICES": str(physical_device_id),
    }


def _parse_cann_version(text):
    if not isinstance(text, str) or not text.strip():
        raise TypeError("CANN version text must be non-empty")
    source_fields = {
        "package": "package_name",
        "version": "version",
        "innerversion": "innerversion",
    }
    parsed = {}
    for field, expected in EXPECTED_CANN.items():
        source_field = source_fields[field]
        matches = re.findall(
            rf"^\s*{source_field}\s*=\s*([^\s]+)\s*$", text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if matches != [expected]:
            raise ValueError(
                f"CANN {field} must be exactly {expected}, found {matches!r}"
            )
        parsed[field] = expected
    return parsed


def _validate_cann_sources(paths):
    identities = {}
    for label, expected in EXPECTED_CANN_SOURCE_SHA256.items():
        path = paths[label]
        if not path.is_file():
            raise FileNotFoundError(f"required {label} is missing: {path}")
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(
                f"{label} SHA256 mismatch: expected {expected}, got {observed}"
            )
        identities[label] = {
            "path": str(path.resolve()),
            "sha256": observed,
        }
    return identities


def _validate_static_instruction_contract(
        benchmark_root, source_paths, source_identities):
    """Bind the public API call to the dav_c220 L0C-to-GM intrinsic chain."""

    paths = {
        "benchmark_kernel": benchmark_root / "fixpipe_service_kernel.cpp",
        "fixpipe_intf_impl": source_paths["fixpipe_intf_impl"],
        "fixpipe_dav_c220_v2_impl": source_paths[
            "fixpipe_dav_c220_v2_impl"],
        "production_fa_source": source_paths["production_fa_source"],
        "production_fa_kernel": source_paths["production_fa_kernel"],
        "fixpipe_struct": source_paths["fixpipe_struct"],
    }
    evidence_sources = {}
    for label, markers in STATIC_INSTRUCTION_SOURCE_MARKERS.items():
        path = paths[label]
        if not path.is_file():
            raise FileNotFoundError(
                f"static instruction source is missing: {label}={path}")
        source = path.read_text(encoding="utf-8")
        missing = tuple(marker for marker in markers if marker not in source)
        if missing:
            raise ValueError(
                f"static instruction source {label} lost required markers: "
                f"{missing}")
        evidence_sources[label] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "required_markers": list(markers),
        }
    for label in set(STATIC_INSTRUCTION_SOURCE_MARKERS) - {"benchmark_kernel"}:
        if (evidence_sources[label]["sha256"]
                != source_identities[label]["sha256"]):
            raise ValueError(
                f"static instruction source identity drifted: {label}")
    return {
        "evidence_kind": "source_bound_compiler_intrinsic_chain",
        "claim": (
            "the benchmark public GlobalTensor<-L0C Fixpipe calls lower "
            "through the dav_c220 L0C-to-GM implementation to "
            "copy_matrix_cc_to_gm and complete on PIPE_FIX"
        ),
        "production_binding": (
            "the fixed standard-GQA FIA Cube source uses the same "
            "CFG_ROW_MAJOR FixpipeParamsV220 public route"
        ),
        "sources": evidence_sources,
        "disassembly_required": False,
    }


def _validate_static_artifact(path, archive_tool):
    if path.name != "libfixpipe_service_kernel.a":
        raise ValueError(
            "static artifact must be named libfixpipe_service_kernel.a"
        )
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("static artifact must be a non-empty regular file")
    archive_executable = str(archive_tool.resolve())
    members = _run_checked([archive_executable, "t", str(path)]).splitlines()
    if not members or not any("fixpipe_service_kernel" in member for member in members):
        raise ValueError("static artifact lacks the Fixpipe kernel object")
    kernel_members = [
        member for member in members if "fixpipe_service_kernel" in member
    ]
    if len(kernel_members) != 1:
        raise ValueError("static artifact must contain one Fixpipe kernel object")
    extracted = subprocess.run(
        [archive_executable, "p", str(path), kernel_members[0]], check=False,
        capture_output=True,
    )
    if extracted.returncode != 0:
        raise RuntimeError(
            f"ar failed to extract {kernel_members[0]} with "
            f"return code {extracted.returncode}"
        )
    if not extracted.stdout.startswith(b"\x7fELF"):
        raise ValueError("Fixpipe archive member is not an ELF object")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "archive_members": members,
        "kernel_member": kernel_members[0],
        "kernel_member_sha256": hashlib.sha256(extracted.stdout).hexdigest(),
        "kernel_member_format": "ELF",
    }


def _build_environment():
    environment = os.environ.copy()
    for name in BUILD_ENVIRONMENT_OVERRIDES:
        if name in environment:
            del environment[name]
    return environment


def _build_tool_identity(path, environment):
    resolved = path.resolve()
    if not path.is_absolute() or not resolved.is_file() \
            or not os.access(resolved, os.X_OK):
        raise ValueError("build tools must be absolute executable files")
    version_text = _run_checked(
        [str(resolved), "--version"], environment=environment, timeout=30)
    if not version_text.strip():
        raise ValueError(f"build tool returned an empty version: {resolved}")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "version": version_text.splitlines()[0],
    }


def _cmake_cache_value(cache_text, name):
    matches = re.findall(
        rf"^{re.escape(name)}:[^=]+=(.*)$", cache_text, flags=re.MULTILINE)
    if len(matches) != 1 or not matches[0]:
        raise ValueError(f"source-bound CMake cache lacks exact {name}")
    return matches[0]


def _validate_frozen_build_identity(build):
    binary = Path(build["binary"])
    if _sha256(binary) != build["binary_sha256"]:
        raise RuntimeError("source-bound executable changed after build")
    observed_static = _validate_static_artifact(
        Path(build["static_artifact"]["path"]),
        Path(build["tools"]["ar"]["path"]))
    if observed_static != build["static_artifact"]:
        raise RuntimeError("source-bound static artifact changed after build")


def _build_source_bound_artifacts(
        benchmark_root, build_directory, ascend_home_path, tool_paths):
    if build_directory.exists():
        raise FileExistsError(
            f"source-bound build directory already exists: {build_directory}"
        )
    if not ascend_home_path.is_dir():
        raise ValueError("Ascend home path must name an existing directory")
    resolved_source = benchmark_root.resolve()
    resolved_build = build_directory.resolve()
    resolved_ascend = ascend_home_path.resolve()
    environment = _build_environment()
    tools = {
        name: _build_tool_identity(tool_paths[name], environment)
        for name in ("cmake", "make", "cxx", "ar", "ranlib")
    }
    configure_argv = [
        tools["cmake"]["path"], "-S", str(resolved_source),
        "-B", str(resolved_build), "-G", "Unix Makefiles",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_MAKE_PROGRAM={tools['make']['path']}",
        f"-DCMAKE_CXX_COMPILER={tools['cxx']['path']}",
        f"-DCMAKE_AR={tools['ar']['path']}",
        f"-DCMAKE_RANLIB={tools['ranlib']['path']}",
        f"-DASCEND_HOME_PATH={resolved_ascend}",
        f"-DASCEND_CANN_PACKAGE_PATH={resolved_ascend}",
        "-DSOC_VERSION=Ascend910B2",
    ]
    build_argv = [
        tools["cmake"]["path"], "--build", str(resolved_build), "--clean-first",
        "--target", "fixpipe_service", "--parallel", "1",
    ]
    _run_checked(configure_argv, environment=environment, timeout=600)
    _run_checked(build_argv, environment=environment, timeout=1800)
    cache_path = resolved_build / "CMakeCache.txt"
    if not cache_path.is_file():
        raise RuntimeError("source-bound build did not produce CMakeCache.txt")
    cache_text = cache_path.read_text(encoding="utf-8")
    expected_cache = {
        "CMAKE_HOME_DIRECTORY": str(resolved_source),
        "CMAKE_MAKE_PROGRAM": tools["make"]["path"],
        "CMAKE_CXX_COMPILER": tools["cxx"]["path"],
        "CMAKE_AR": tools["ar"]["path"],
        "CMAKE_RANLIB": tools["ranlib"]["path"],
        "ASCEND_HOME_PATH": str(resolved_ascend),
        "ASCEND_CANN_PACKAGE_PATH": str(resolved_ascend),
        "SOC_VERSION": "Ascend910B2",
    }
    for name, expected in expected_cache.items():
        if _cmake_cache_value(cache_text, name) != expected:
            raise ValueError(f"source-bound CMake cache changed {name}")
    binary = resolved_build / "fixpipe_service"
    static_path = resolved_build / "lib" / "libfixpipe_service_kernel.a"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError("source-bound build did not produce the executable")
    static_artifact = _validate_static_artifact(
        static_path, Path(tools["ar"]["path"]))
    return {
        "directory": str(resolved_build),
        "ascend_home_path": str(resolved_ascend),
        "tools": tools,
        "build_environment": {
            "removed_overrides": list(BUILD_ENVIRONMENT_OVERRIDES),
            "path": environment["PATH"],
        },
        "configure_argv": configure_argv,
        "build_argv": build_argv,
        "cmake_cache_sha256": _sha256(cache_path),
        "binary": str(binary),
        "binary_sha256": _sha256(binary),
        "static_artifact": static_artifact,
    }


def _validate_profiler_output(path, repo_root, label):
    resolved = path.resolve()
    root = (repo_root / "profiler_output").resolve()
    if root not in resolved.parents:
        raise ValueError(f"{label} must be below profiler_output/")
    relative = resolved.relative_to(root)
    if "ascend_910b" not in relative.parts:
        raise ValueError(
            f"{label} must be scoped below profiler_output/.../ascend_910b/"
        )
    if path.exists():
        raise FileExistsError(f"{label} already exists: {path}")


def _validate_git_or_snapshot(repo_root, source_commit, snapshot_sha256):
    git_marker = repo_root / ".git"
    if not git_marker.exists():
        return "audited_source_snapshot"
    head = _run_checked(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"]
    ).strip()
    if head != source_commit:
        raise ValueError(
            f"source commit mismatch: requested {source_commit}, checkout {head}"
        )
    relative_paths = [
        str((Path("microbench") / "ascend_910b" / "memory" / "fixpipe"
             / filename))
        for filename in SOURCE_FILENAMES
    ]
    _run_checked(
        ["git", "-C", str(repo_root), "diff", "--exit-code", source_commit,
         "--", *relative_paths]
    )
    if not SHA256_PATTERN.fullmatch(snapshot_sha256):
        raise ValueError("snapshot SHA256 must contain 64 lowercase hex digits")
    return "git_commit_and_source_snapshot"


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


def _run_campaign(args, benchmark_root, repo_root):
    global _ACTIVE_PROCESS

    if args.device_id < 0:
        raise ValueError("physical device ID must be non-negative")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        raise ValueError("run ID must match [A-Za-z0-9._-]{1,96}")
    if COMMIT_PATTERN.fullmatch(args.source_commit) is None:
        raise ValueError("source commit must contain 40 lowercase hex digits")
    if SHA256_PATTERN.fullmatch(args.expected_source_snapshot_sha256) is None:
        raise ValueError("expected snapshot SHA256 must contain 64 lowercase hex digits")
    if not args.cann_version_file.is_file():
        raise ValueError("CANN version file must name a regular file")
    if not args.npu_smi.is_file() or not os.access(args.npu_smi, os.X_OK):
        raise ValueError("npu-smi must name an executable regular file")
    _validate_profiler_output(args.raw_output, repo_root, "raw output")
    _validate_profiler_output(
        args.provenance_output, repo_root, "provenance output"
    )
    if args.raw_output.resolve() == args.provenance_output.resolve():
        raise ValueError("raw and provenance outputs must be distinct paths")
    _validate_profiler_output(
        args.build_directory, repo_root, "source-bound build directory")
    _validate_schedule(args.trials)
    observed_snapshot = _source_snapshot_sha256(benchmark_root)
    if observed_snapshot != args.expected_source_snapshot_sha256:
        raise ValueError(
            "source snapshot SHA256 mismatch: expected "
            f"{args.expected_source_snapshot_sha256}, got {observed_snapshot}"
        )
    identity_mode = _validate_git_or_snapshot(
        repo_root, args.source_commit, observed_snapshot
    )
    source_paths = {
        "fixpipe_interface": args.fixpipe_interface,
        "fixpipe_intf_impl": args.fixpipe_intf_impl,
        "fixpipe_dav_c220_v2_impl": args.fixpipe_dav_c220_v2_impl,
        "production_fa_source": args.production_fa_source,
        "production_fa_kernel": args.production_fa_kernel,
        "fixpipe_struct": args.fixpipe_struct,
    }
    cann_sources = _validate_cann_sources(source_paths)
    if not args.production_geometry_source.is_file():
        raise FileNotFoundError(
            "production geometry source is missing: "
            f"{args.production_geometry_source}")
    production_geometry_sha256 = _sha256(args.production_geometry_source)
    if production_geometry_sha256 != EXPECTED_PRODUCTION_GEOMETRY_SHA256:
        raise ValueError(
            "production geometry source SHA256 mismatch: expected "
            f"{EXPECTED_PRODUCTION_GEOMETRY_SHA256}, got "
            f"{production_geometry_sha256}")
    static_instruction_evidence = _validate_static_instruction_contract(
        benchmark_root, source_paths, cann_sources)
    cann_text = args.cann_version_file.read_text(encoding="utf-8")
    cann_version = _parse_cann_version(cann_text)
    tool_paths = {
        "cmake": args.cmake,
        "make": args.make_program,
        "cxx": args.cxx_compiler,
        "ar": args.archive_tool,
        "ranlib": args.ranlib,
    }
    build = _build_source_bound_artifacts(
        benchmark_root, args.build_directory, args.ascend_home_path,
        tool_paths)
    if _source_snapshot_sha256(benchmark_root) != observed_snapshot:
        raise RuntimeError("source snapshot changed during source-bound build")
    if _validate_cann_sources(source_paths) != cann_sources:
        raise RuntimeError("CANN source identities changed during build")
    if _sha256(args.production_geometry_source) != production_geometry_sha256:
        raise RuntimeError("production geometry source changed during build")
    _validate_frozen_build_identity(build)
    binary = Path(build["binary"])

    environment = os.environ.copy()
    physical_device_id = args.device_id
    environment.update(_device_selection_environment(physical_device_id))
    resolved_npu_smi = args.npu_smi.resolve()
    global_command, board_command, usages_command = _npu_smi_commands(
        resolved_npu_smi, physical_device_id
    )
    global_telemetry = _run_checked(global_command, environment=environment)
    board_telemetry = _run_checked(board_command, environment=environment)
    usage_telemetry = _run_checked(usages_command, environment=environment)
    device_identity = _parse_device_identity(
        global_telemetry, board_telemetry, physical_device_id
    )
    device_identity["logical_device_id"] = LOGICAL_DEVICE_ID
    npu_smi_version = _parse_npu_smi_version(global_telemetry)
    device_identity["selection_environment"] = {
        "ASCEND_DEVICE_ID": environment["ASCEND_DEVICE_ID"],
        "ASCEND_RT_VISIBLE_DEVICES": environment["ASCEND_RT_VISIBLE_DEVICES"],
    }
    idle = _parse_idle_telemetry(usage_telemetry)

    provenance = {
        "schema": "ascend_910b_fixpipe_service_provenance_v2",
        "run_id": args.run_id,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "source_identity_mode": identity_mode,
        "source_snapshot_sha256": observed_snapshot,
        "source_files": {
            filename: _sha256(benchmark_root / filename)
            for filename in SOURCE_FILENAMES
        },
        "cann": {
            "identity": cann_version,
            "version_file": str(args.cann_version_file.resolve()),
            "version_file_sha256": _sha256(args.cann_version_file),
            "source_identities": cann_sources,
        },
        "build": build,
        "device": device_identity,
        "npu_smi": {
            "executable": str(resolved_npu_smi),
            "executable_sha256": _sha256(args.npu_smi),
            "version": npu_smi_version,
            "global": {
                "command": global_command,
                "stdout": global_telemetry,
                "stdout_sha256": hashlib.sha256(
                    global_telemetry.encode("utf-8")
                ).hexdigest(),
            },
            "board": {
                "command": board_command,
                "stdout": board_telemetry,
                "stdout_sha256": hashlib.sha256(
                    board_telemetry.encode("utf-8")
                ).hexdigest(),
            },
            "usages": {
                "command": usages_command,
                "stdout": usage_telemetry,
                "stdout_sha256": hashlib.sha256(
                    usage_telemetry.encode("utf-8")
                ).hexdigest(),
            },
        },
        "idle_telemetry": idle,
        "idle_telemetry_sha256": hashlib.sha256(
            usage_telemetry.encode("utf-8")
        ).hexdigest(),
        "clock": {
            "source": "GetSystemCycle",
            "system_counter_hz": SYSTEM_COUNTER_HZ,
            "tick_ns": SYSTEM_COUNTER_TICK_NS,
            "host_wall_role": "health_check_only",
        },
        "measurement_scope": (
            "FP32 row-major direct L0C-to-GM public Fixpipe completed serial "
            "service with independent synthetic calibration and untouched "
            "standard-GQA geometry holdout"
        ),
        "publication_limit": (
            "does not establish asynchronous queue depth, credit, production "
            "overlap, or GM residency; any parameter remains restricted to "
            "512-byte-aligned destination bases, two-to-sixteen destination "
            "slots, the preregistered geometry, and all report gates"
        ),
        "production_or_evaluation_duration_consumed": False,
        "static_instruction_contract_audited": True,
        "static_instruction_evidence": static_instruction_evidence,
        "production_geometry_evidence": {
            "path": str(args.production_geometry_source.resolve()),
            "sha256": production_geometry_sha256,
            "git_lfs_oid_sha256": EXPECTED_PRODUCTION_GEOMETRY_SHA256,
            "derivation": (
                "the preregistered 115 (M,N,dstStride) cells are derived "
                "only from duration_consumed=false FIA source "
                "schedule observations"
            ),
        },
    }

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    raw_temporary = args.raw_output.with_suffix(args.raw_output.suffix + ".tmp")
    if raw_temporary.exists():
        raise FileExistsError(f"temporary raw output exists: {raw_temporary}")
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with raw_temporary.open("x", encoding="utf-8") as raw_stream:
        print(
            "run_id,trial,order_direction,order_position,process_pid,family,"
            "layout,m,n,src_stride,dst_stride,mode,stream_depth,block_dim,"
            "destination_slots,base_invocations,middle_invocations,"
            "long_invocations,base_logical_bytes_per_aic,"
            "middle_logical_bytes_per_aic,long_logical_bytes_per_aic,"
            "base_block_cycles,middle_block_cycles,long_block_cycles,"
            "base_host_wall_ns,middle_host_wall_ns,long_host_wall_ns,"
            "clock_source,completion,validation",
            file=raw_stream,
        )
        for trial_index in range(args.trials):
            direction, cells = _ordered_cells(trial_index)
            _validate_frozen_build_identity(build)
            command = _build_command(binary, cells)
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=environment, start_new_session=True,
            )
            _ACTIVE_PROCESS = process
            try:
                stdout, stderr = process.communicate(
                    timeout=args.child_timeout_seconds
                )
            except subprocess.TimeoutExpired as error:
                _terminate_process_group(process)
                raise RuntimeError(
                    f"child PID {process.pid} timed out after "
                    f"{args.child_timeout_seconds} seconds"
                ) from error
            finally:
                if process.poll() is None:
                    _terminate_process_group(process)
                _ACTIVE_PROCESS = None
            if stderr:
                print(
                    f"child_pid={process.pid} trial={trial_index + 1} "
                    f"stderr={stderr.rstrip()}", file=sys.stderr, flush=True,
                )
            if process.returncode != 0:
                raise RuntimeError(
                    f"child PID {process.pid} failed with exit "
                    f"{process.returncode}: {' '.join(command)}"
                )
            _validate_frozen_build_identity(build)
            lines = stdout.strip().splitlines()
            if len(lines) != len(cells):
                raise RuntimeError(
                    f"child PID {process.pid} emitted {len(lines)} data lines"
                )
            for order_position, line in enumerate(lines):
                fields = line.split(",")
                if len(fields) != CSV_FIELDS:
                    raise RuntimeError("child emitted an invalid raw schema")
                (
                    family, layout_id, layout, m, n, dst_stride, _mode_id,
                    mode, depth, block_dim, destination_slots,
                ) = cells[order_position]
                src_stride, checked_dst_stride = _strides(
                    layout, m, n, dst_stride)
                if (
                    fields[0] != layout
                    or int(fields[1]) != m
                    or int(fields[2]) != n
                    or int(fields[3]) != src_stride
                    or int(fields[4]) != checked_dst_stride
                    or fields[5] != mode
                    or int(fields[6]) != depth
                    or int(fields[7]) != block_dim
                    or int(fields[8]) != destination_slots
                    or int(fields[9]) != _invocations(m, n, depth)
                    or fields[21] != "GetSystemCycle"
                    or fields[22] != "pipe_fix_completion"
                    or fields[23] != "pass"
                ):
                    raise RuntimeError("child changed the requested Fixpipe cell")
                if layout_id != LAYOUT_ID:
                    raise AssertionError("layout registry contains an invalid ID")
                print(
                    f"{args.run_id},{trial_index + 1},{direction},"
                    f"{order_position},{process.pid},{family},{line}",
                    file=raw_stream,
                )
                row_count += 1
        raw_stream.flush()
        os.fsync(raw_stream.fileno())
    expected_rows = args.trials * len(CELLS)
    if row_count != expected_rows:
        raise AssertionError(
            f"raw row count {row_count} does not match {expected_rows}"
        )
    _validate_frozen_build_identity(build)
    raw_temporary.replace(args.raw_output)
    provenance["capture"] = {
        "raw_output": str(args.raw_output.resolve()),
        "raw_sha256": _sha256(args.raw_output),
        "raw_row_count": row_count,
        "argv": list(sys.argv),
        "working_directory": str(Path.cwd().resolve()),
        "runner_pid": os.getpid(),
    }

    temporary = args.provenance_output.with_suffix(
        args.provenance_output.suffix + ".tmp"
    )
    if temporary.exists():
        raise FileExistsError(f"temporary provenance path exists: {temporary}")
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(args.provenance_output)


def main():
    benchmark_root = Path(__file__).resolve().parent
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("snapshot")
    run = subparsers.add_parser("run")
    run.add_argument("--build-directory", required=True, type=Path)
    run.add_argument("--ascend-home-path", required=True, type=Path)
    run.add_argument("--cmake", required=True, type=Path)
    run.add_argument("--make-program", required=True, type=Path)
    run.add_argument("--cxx-compiler", required=True, type=Path)
    run.add_argument("--archive-tool", required=True, type=Path)
    run.add_argument("--ranlib", required=True, type=Path)
    run.add_argument(
        "--device-id", required=True, type=int,
        help="physical NPU ID for npu-smi and visibility remapping",
    )
    run.add_argument("--run-id", required=True)
    run.add_argument("--source-commit", required=True)
    run.add_argument("--expected-source-snapshot-sha256", required=True)
    run.add_argument("--cann-version-file", required=True, type=Path)
    run.add_argument("--fixpipe-interface", required=True, type=Path)
    run.add_argument("--fixpipe-intf-impl", required=True, type=Path)
    run.add_argument(
        "--fixpipe-dav-c220-v2-impl", required=True, type=Path)
    run.add_argument("--production-fa-source", required=True, type=Path)
    run.add_argument("--production-fa-kernel", required=True, type=Path)
    run.add_argument("--fixpipe-struct", required=True, type=Path)
    run.add_argument(
        "--production-geometry-source", required=True, type=Path)
    run.add_argument("--npu-smi", required=True, type=Path)
    run.add_argument("--raw-output", required=True, type=Path)
    run.add_argument("--provenance-output", required=True, type=Path)
    run.add_argument("--trials", required=True, type=_positive_integer)
    run.add_argument(
        "--child-timeout-seconds", type=_positive_integer, default=1800
    )
    args = parser.parse_args()
    if args.command == "snapshot":
        print(_source_snapshot_sha256(benchmark_root))
    elif args.command == "run":
        _run_campaign(args, benchmark_root, repo_root)
    else:
        raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
