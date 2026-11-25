"""Derive discrete raster candidates from duration-free driver parameters.

This does not publish launch data or predict performance. The source entry
scope is supplied by retained static reviews, never by evaluation error.
"""
import ast
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import struct

from src.kernel.launch_config.launch_program import GpuCtaRasterProgram
from src.kernel.launch_config.tiling_spec import parse_nvjet_gemm_symbol


def _checked_payload(path, expected_sha):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError(f"raster evidence content identity changed: {path}")
    return payload


def load_nvjet_parameter_records(campaign):
    """Validate capture/source identities and the known marker's C0 origin."""
    campaign = Path(campaign)
    raw = json.loads((campaign / "raw.json").read_text())
    if (raw["build_exit_code"] != 0 or raw["run_exit_code"] != 0
            or raw["production_duration_consumed"] is not False
            or raw["fixed221_shape_capture"] is not True):
        raise ValueError("raster parameter capture did not complete")
    for filename, key in (
            ("nvjet_launch_params.cu", "source_sha256"),
            ("nvjet_launch_params_modal_runner.py", "runner_sha256")):
        _checked_payload(campaign / "source" / filename, raw[key])
    stdout = gzip.decompress((campaign / raw["stdout_path"]).read_bytes())
    if hashlib.sha256(stdout).hexdigest() != raw["stdout_sha256"]:
        raise ValueError("raster launch records changed")
    marker_sass = gzip.decompress(
        (campaign / "parameter_marker.sass.txt.gz").read_bytes())
    if hashlib.sha256(marker_sass).hexdigest() != raw["marker_sass_stdout_sha256"]:
        raise ValueError("raster marker compiler evidence changed")
    rows = [json.loads(line) for line in stdout.decode().splitlines()]
    versions = [r["cublas_version"] for r in rows if "cublas_version" in r]
    if versions != [120901]:
        raise ValueError("raster source scope requires cuBLAS12.9.1")
    identity = json.loads((campaign / "shape_input_identity.json").read_text())
    shape_payload = _checked_payload(campaign / "source/shapes.csv", identity["sha256"])
    shape_rows = list(csv.reader(shape_payload.decode().splitlines()))
    if len(shape_rows) != 222 or shape_rows[0] != ["case_id", "m", "n", "k"]:
        raise ValueError("raster shape-only input is not fixed221")
    planned = {}
    for row in shape_rows[1:]:
        if len(row) != 4 or row[0] in planned:
            raise ValueError("raster shape rows are malformed or duplicated")
        values = tuple(int(value) for value in row[1:])
        if any(value <= 0 for value in values):
            raise ValueError("raster shape extents must be positive")
        planned[row[0]] = values
    declared = [r for r in rows if "m" in r]
    observed = {r["case_id"]: (r["m"], r["n"], r["k"]) for r in declared}
    if len(declared) != 221 or observed != planned:
        raise ValueError("executed shape declarations differ from pinned input")
    markers = [r for r in rows if "kernel" in r
               and r["case_id"] == "parameter_marker"
               and "parameter_bytes_hex" in r]
    if len(markers) != 1:
        raise ValueError("raster capture needs one parameter marker")
    marker = markers[0]
    if marker["parameter_fields"] != [
            {"offset": 0, "size": 8}, {"offset": 8, "size": 4}]:
        raise ValueError("parameter marker declaration changed")
    marker_bytes = bytes.fromhex(marker["parameter_bytes_hex"])
    if len(marker_bytes) != 12 or struct.unpack_from("<I", marker_bytes, 8)[0] != 0x51515151:
        raise ValueError("parameter marker value changed")
    sass = marker_sass.decode()
    stores = re.findall(r"STG[^\n]*,\s*(R\d+)\s*;", sass)
    if len(stores) != 1:
        raise ValueError("parameter marker store is ambiguous")
    loads = re.findall(
        rf"LDC\s+{stores[0]},\s*c\[0x0\]\[0x([0-9a-f]+)\]", sass)
    if len(loads) != 1:
        raise ValueError("parameter marker constant offset is ambiguous")
    origin = int(loads[0], 16) - 8
    records = [r for r in rows if "kernel" in r
               and r["kernel"].startswith("nvjet_")]
    if (len(records) != 221 or len({r["case_id"] for r in records}) != 221
            or {r["case_id"] for r in records} != set(planned)
            or rows[-1]["cases_checked"] != 221
            or rows[-1]["marker_and_zero_output_check"] is not True
            or rows[-1]["production_duration_collected"] is not False):
        raise ValueError("fixed221 launch parameter capture is incomplete")
    records = [{**r, "captured_shape": planned[r["case_id"]]} for r in records]
    return records, origin


def source_bound_nvjet_entries(review):
    """Require the retained entries to contain the observed integer decoder."""
    review = Path(review)
    entries = set()
    required = tuple(range(0x214, 0x240, 4)) + tuple(range(0x340, 0x350, 4))
    for row in json.loads(review.read_text()):
        source = review.parent / f"{row['tile_m']}x{row['tile_n']}_source.sass.gz"
        text = gzip.decompress(source.read_bytes()).decode()
        entry = row["canonical_provenance"]["ncu_kernel_name"]
        if entry not in re.sub(r"\s+", "", text[:500]):
            raise ValueError(f"retained source entry name changed: {source}")
        if any(f"c[0x0][{hex(offset)}]" not in text for offset in required):
            raise ValueError(f"source entry lacks the raster scalar layout: {source}")
        if "ATOMG.E.ADD" not in text or "UIMAD.WIDE.U32" not in text:
            raise ValueError(f"source entry lacks the persistent integer decoder: {source}")
        entries.add(entry)
    return frozenset(entries)


def _scalars(record, origin):
    payload = bytes.fromhex(record["parameter_bytes_hex"])
    if len(payload) != record["parameter_bytes"]:
        raise ValueError("driver parameter byte length changed")
    result = {}
    for address in tuple(range(0x214, 0x240, 4)) + tuple(range(0x340, 0x350, 4)):
        offset = address - origin
        if offset < 0 or not any(
                field["offset"] <= offset
                and offset + 4 <= field["offset"] + field["size"]
                for field in record["parameter_fields"]):
            raise ValueError("raster scalar lies outside observed parameter fields")
        result[address] = struct.unpack_from("<I", payload, offset)[0]
    return result


def _divide(value, fields, negative, magic, shift):
    divisor = (-fields[negative]) & 0xffffffff
    if divisor <= 0:
        raise ValueError("used raster divisor is not positive")
    if fields[magic] == 0:
        quotient, remainder = value, 0
    else:
        quotient = ((value * fields[magic]) >> 32) >> fields[shift]
        remainder = (value + quotient * fields[negative]) & 0xffffffff
    if quotient != value // divisor or remainder != value % divisor:
        raise ValueError("observed raster fast-division fields disagree")
    return quotient, remainder


def derive_nvjet_raster_programs(records, origin, expected_rows, source_entries):
    """Return typed source programs plus explicit unbound reasons by case ID."""
    indexed = {r["case_id"]: r for r in records}
    if len(indexed) != len(records) or set(indexed) != set(expected_rows):
        raise ValueError("raster capture and frozen physical cases differ")
    pending = []
    status = {}
    axes_by_entry = {}
    for case, record in indexed.items():
        expected = expected_rows[case]
        attr = expected["op_info"]["Attr"]
        if (expected["op_info"]["Type"] != "GEMM"
                or attr["dtype"] != "fp16"
                or tuple(record["captured_shape"]) != (
                    attr["B"] * attr["M"], attr["N"], attr["K"])):
            raise ValueError("captured shape/dtype differs from the fixed GEMM input")
        metadata = expected["kernels"][0]["Metadata"]
        header = expected["kernels"][0]["Sections"]["Launch Statistics"]["Header"]
        matches = (
            record["kernel"] == metadata["Kernel Name"]
            and tuple(record["grid"]) == ast.literal_eval(metadata["Grid Size"])
            and tuple(record["block"]) == ast.literal_eval(metadata["Block Size"])
            and record["dynamic_shared_bytes"] == int(header["Dynamic Shared Memory Per Block [byte/block]"])
            and record["static_shared_bytes"] == int(header["Static Shared Memory Per Block [byte/block]"])
            and record["registers_per_thread"] == int(header["Registers Per Thread [register/thread]"]))
        if not matches:
            status[case] = "unbound:canonical_launch_signature_mismatch"
            continue
        symbol = parse_nvjet_gemm_symbol(record["kernel"])
        if symbol is None:
            raise ValueError("selected record has no complete nvjet symbol")
        if symbol.split_k:
            status[case] = "unbound:split_k_raster_not_projected"
            continue
        if record["kernel"] not in source_entries:
            status[case] = "unbound:static_raster_entry_not_reviewed"
            continue
        mc = math.ceil(attr["B"] * attr["M"] / (symbol.tile_m * symbol.cluster_m_tiles))
        nc = math.ceil(attr["N"] / (symbol.tile_n * symbol.cluster_n_tiles))
        fields = _scalars(record, origin)
        if fields[0x214] != mc * nc:
            raise ValueError("raster job count differs from complete output geometry")
        possible = []
        if fields[0x34c] + 1 == nc:
            possible.append("m")
        if fields[0x34c] + 1 == mc:
            possible.append("n")
        if not possible:
            raise ValueError("raster other-axis extent differs from output geometry")
        if len(possible) == 1:
            entry = record["kernel"]
            if entry in axes_by_entry and axes_by_entry[entry] != possible[0]:
                raise ValueError("one static raster entry has inconsistent output axes")
            axes_by_entry[entry] = possible[0]
        pending.append((case, record, mc, nc, fields, possible))
    programs = {}
    for case, record, mc, nc, fields, possible in pending:
        if len(possible) == 1:
            axis = possible[0]
        elif record["kernel"] in axes_by_entry:
            axis = axes_by_entry[record["kernel"]]
        else:
            status[case] = "unbound:source_axis_not_distinguished"
            continue
        count, other = (mc, nc) if axis == "m" else (nc, mc)
        width = (-fields[0x228]) & 0xffffffff
        if (width <= 0 or fields[0x218] != count // width
                or ((-fields[0x21c]) & 0xffffffff) != width * other
                or ((-fields[0x340]) & 0xffffffff) != mc * nc):
            raise ValueError("raster group/axis integers do not conserve output jobs")
        decoded = []
        for job in range(mc * nc):
            batch, remainder = _divide(job, fields, 0x340, 0x344, 0x348)
            if batch != 0:
                raise ValueError("flattened GEMM unexpectedly has a matrix-batch raster")
            band, inner = _divide(remainder, fields, 0x21c, 0x220, 0x224)
            division = ((0x234, 0x238, 0x23c) if band >= fields[0x218]
                        else (0x228, 0x22c, 0x230))
            ungrouped, grouped = _divide(inner, fields, *division)
            grouped += band * width
            if band % 2:
                ungrouped = fields[0x34c] - ungrouped
            mi, ni = ((grouped, ungrouped) if axis == "m"
                      else (ungrouped, grouped))
            decoded.append((mi, ni, 0))
        program = GpuCtaRasterProgram(
            "cluster_m_group" if axis == "m" else "cluster_n_group",
            0, True, 1, width)
        if tuple(decoded) != program.schedule(mc, nc):
            raise ValueError("shared grouped program differs from observed source arithmetic")
        if len(set(decoded)) != mc * nc or set(decoded) != {
                (mi, ni, 0) for mi in range(mc) for ni in range(nc)}:
            raise ValueError("raster program does not cover every output cluster exactly once")
        programs[case] = program
        status[case] = "bound:integer_map_and_complete_cluster_coverage_verified"
    return programs, status
