#!/usr/bin/env python3
"""Decode non-duration CANN 8.5.1 SwiGlu launch/tiling facts."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct


EXPECTED_EXECUTIONS = 6
EXPECTED_RUNTIME_FUNCTION_NAME = "SwiGlu_3_high_performance_27"
EXPECTED_REGISTRATION_KEY = "SwiGlu"
EXPECTED_CLASS = "SwiGluTilingData"
EXPECTED_SIZE_BYTES = 80
EXPECTED_FIELDS = (
    ("is32BAligned", "uint32_t", 0, 1, 4),
    ("isDoubleBuffer", "uint32_t", 4, 1, 4),
    ("rowLen", "uint64_t", 8, 1, 8),
    ("colLen", "uint64_t", 16, 1, 8),
    ("baseRowLen", "uint32_t", 24, 1, 4),
    ("baseColLen", "uint32_t", 28, 1, 4),
    ("activateLeft", "uint32_t", 32, 1, 4),
    ("biasIsEmpty", "uint32_t", 36, 1, 4),
    ("quantScaleIsEmpty", "uint32_t", 40, 1, 4),
    ("activateScaleIsEmpty", "uint32_t", 44, 1, 4),
    ("swiColLen", "uint64_t", 48, 1, 8),
    ("perRowLen", "uint64_t", 56, 1, 8),
    ("modRowLen", "uint64_t", 64, 1, 8),
    ("usedCoreNum", "uint32_t", 72, 1, 4),
    ("SwiGluTilingDataPH", "uint8_t", 76, 4, 4),
)
INTEGER_FORMATS = {
    "uint8_t": "B",
    "uint32_t": "I",
    "uint64_t": "Q",
}
UNSIGNED_DECIMAL = re.compile(r"[0-9]+")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_uint(text, context, *, positive=False):
    if not isinstance(text, str) or UNSIGNED_DECIMAL.fullmatch(text) is None:
        raise ValueError(f"{context} must be an unsigned decimal integer")
    value = int(text)
    if positive and value <= 0:
        raise ValueError(f"{context} must be positive")
    return value


def _parse_optional_text(text, context):
    if text == "none":
        return None
    if not isinstance(text, str) or not text or "\n" in text or "\r" in text:
        raise ValueError(f"{context} must be non-empty single-line text or none")
    return text


def _parse_optional_uint(text, context):
    if text == "none":
        return None
    return _parse_uint(text, context)


def _parse_stub_identity(values, path):
    kind = values["stub_identity_kind"]
    if kind not in {
            "runtime_function_name", "elf_symbol", "elf_object_offset",
            "unresolved"}:
        raise ValueError(f"unknown stub identity kind in {path}: {kind!r}")
    runtime_name = _parse_optional_text(
        values["stub_runtime_function_name"], "runtime function name")
    runtime_status = _parse_uint(
        values["stub_runtime_name_status"], "runtime name status")
    elf_membership = values["stub_elf_membership"]
    if elf_membership not in {"mapped_elf", "not_mapped_elf"}:
        raise ValueError(
            f"invalid stub ELF membership in {path}: {elf_membership!r}")
    object_path_text = _parse_optional_text(
        values["stub_elf_object_path"], "stub ELF object path")
    object_base = _parse_optional_uint(
        values["stub_elf_object_base"], "stub ELF object base")
    object_offset = _parse_optional_uint(
        values["stub_elf_object_offset"], "stub ELF object offset")
    elf_symbol = _parse_optional_text(
        values["stub_elf_symbol"], "stub ELF symbol")
    symbol_address = _parse_optional_uint(
        values["stub_elf_symbol_address"], "stub ELF symbol address")
    symbol_offset = _parse_optional_uint(
        values["stub_elf_symbol_offset"], "stub ELF symbol offset")
    stub_address = _parse_uint(
        values["stub_address"], "stub address", positive=True)
    if kind == "runtime_function_name":
        if runtime_status != 0 or runtime_name is None:
            raise ValueError(
                "runtime_function_name identity requires a successful "
                "aclrtGetFunctionName result")
    elif runtime_name is not None or runtime_status == 0:
        raise ValueError(
            "non-runtime stub identity contradicts aclrtGetFunctionName")
    if elf_membership == "mapped_elf":
        if object_path_text is None or object_base is None or object_offset is None:
            raise ValueError("mapped ELF identity is missing object coordinates")
        object_path = Path(object_path_text)
        if not object_path.is_absolute() or not object_path.is_file():
            raise ValueError(
                f"stub ELF object path is not an absolute file: {object_path}")
        if stub_address != object_base + object_offset:
            raise ValueError("stub ELF object coordinates do not reconstruct address")
    else:
        if any(value is not None for value in (
                object_path_text, object_base, object_offset, elf_symbol,
                symbol_address, symbol_offset)):
            raise ValueError("non-ELF stub identity contains ELF coordinates")
    if elf_symbol is None:
        if symbol_address is not None or symbol_offset is not None:
            raise ValueError("stub ELF symbol coordinates lack a symbol")
    else:
        if symbol_address is None or symbol_offset is None:
            raise ValueError("stub ELF symbol is missing coordinates")
        if stub_address != symbol_address + symbol_offset:
            raise ValueError("stub ELF symbol coordinates do not reconstruct address")
    if kind == "elf_symbol" and (
            elf_membership != "mapped_elf" or elf_symbol is None):
        raise ValueError("elf_symbol identity requires a mapped exact symbol")
    if kind == "elf_object_offset" and (
            elf_membership != "mapped_elf" or elf_symbol is not None):
        raise ValueError(
            "elf_object_offset identity requires a symbol-less mapped ELF")
    if kind == "unresolved" and (
            runtime_name is not None or elf_membership != "not_mapped_elf"):
        raise ValueError("unresolved identity contains a resolved identity")
    return {
        "kind": kind,
        "stub_address": stub_address,
        "runtime_name_status": runtime_status,
        "runtime_function_name": runtime_name,
        "elf_membership": elf_membership,
        "elf_object_path": object_path_text,
        "elf_object_base": object_base,
        "elf_object_offset": object_offset,
        "elf_symbol": elf_symbol,
        "elf_symbol_address": symbol_address,
        "elf_symbol_offset": symbol_offset,
    }


def parse_reflection(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 3 + len(EXPECTED_FIELDS):
        raise ValueError("SwiGlu reflection line count changed")
    if lines[0] != "SCHEMA\t1":
        raise ValueError("SwiGlu reflection schema must be 1")
    root = lines[1].split("\t")
    expected_root = [
        "ROOT", EXPECTED_REGISTRATION_KEY, EXPECTED_CLASS,
        str(EXPECTED_SIZE_BYTES), "1", str(len(EXPECTED_FIELDS)),
    ]
    if root != expected_root:
        raise ValueError(f"unexpected SwiGlu reflection root: {root!r}")
    structure = lines[2].split("\t")
    expected_structure = [
        "STRUCT", "$", EXPECTED_CLASS, "0", str(EXPECTED_SIZE_BYTES),
        str(len(EXPECTED_FIELDS)),
    ]
    if structure != expected_structure:
        raise ValueError(
            f"unexpected SwiGlu reflection structure: {structure!r}")
    reflected_fields = []
    covered = []
    for line, expected in zip(lines[3:], EXPECTED_FIELDS):
        columns = line.split("\t")
        expected_columns = [
            "FIELD", expected[0], expected[1], str(expected[2]),
            str(expected[3]), str(expected[4]),
        ]
        if columns != expected_columns:
            raise ValueError(
                "unexpected SwiGlu reflected field: "
                f"expected={expected_columns!r}, actual={columns!r}")
        reflected_fields.append({
            "name": expected[0],
            "dtype": expected[1],
            "offset_bytes": expected[2],
            "count": expected[3],
            "size_bytes": expected[4],
        })
        covered.extend(range(expected[2], expected[2] + expected[4]))
    if sorted(covered) != list(range(EXPECTED_SIZE_BYTES)):
        raise ValueError("reflected SwiGlu fields do not exactly cover 80 bytes")
    return {
        "registration_key": EXPECTED_REGISTRATION_KEY,
        "class_name": EXPECTED_CLASS,
        "data_size_bytes": EXPECTED_SIZE_BYTES,
        "fields": reflected_fields,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
    }


def _parse_metadata(path):
    values = {}
    for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or "=" not in line:
            raise ValueError(
                f"invalid capture metadata at {path}:{line_number}")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise ValueError(
                f"duplicate or empty key at {path}:{line_number}")
        values[key] = value
    return values


def _safe_child(raw_directory, name, context):
    if not name or Path(name).name != name:
        raise ValueError(f"{context} must be a plain file name")
    path = raw_directory / name
    if not path.is_file():
        raise FileNotFoundError(f"{context} is missing: {path}")
    return path


def parse_capture_records(raw_directory):
    metadata_paths = sorted(raw_directory.glob("capture_*.meta"))
    if not metadata_paths:
        raise ValueError("SwiGlu interposer emitted no metadata")
    records = []
    for path in metadata_paths:
        values = _parse_metadata(path)
        common = {
            "schema_version", "record_type", "sequence", "api", "pid",
            "tid",
        }
        record_type = values["record_type"]
        if record_type == "workspace":
            if values["schema_version"] != "1":
                raise ValueError(f"unsupported workspace schema in {path}")
            expected = common | {"workspace_size", "execute_stream"}
            if set(values) != expected:
                raise ValueError(
                    f"workspace metadata fields changed: {sorted(values)}")
            if values["api"] != "aclnnSwiGlu":
                raise ValueError("workspace record is not aclnnSwiGlu")
            record = {
                "record_type": record_type,
                "sequence": _parse_uint(
                    values["sequence"], "workspace sequence", positive=True),
                "api": values["api"],
                "pid": _parse_uint(values["pid"], "workspace pid", positive=True),
                "tid": _parse_uint(values["tid"], "workspace tid", positive=True),
                "workspace_size_bytes": _parse_uint(
                    values["workspace_size"], "workspace size"),
                "execute_stream": _parse_uint(
                    values["execute_stream"], "execute stream"),
                "metadata_path": str(path.resolve()),
                "metadata_sha256": _sha256(path),
            }
        elif record_type == "launch_args":
            if values["schema_version"] != "2":
                raise ValueError(f"unsupported launch schema in {path}")
            expected = common | {
                "execute_api", "stub_identity_kind", "stub_address",
                "stub_runtime_name_status", "stub_runtime_function_name",
                "stub_elf_membership", "stub_elf_object_path",
                "stub_elf_object_base", "stub_elf_object_offset",
                "stub_elf_symbol", "stub_elf_symbol_address",
                "stub_elf_symbol_offset", "block_dim", "args_size",
                "args_file", "placeholder_count", "placeholder_entry_size",
                "placeholder_file", "execute_stream", "launch_stream",
            }
            if set(values) != expected:
                raise ValueError(
                    f"launch metadata fields changed: {sorted(values)}")
            if (values["api"] != "aclrtLaunchKernelWithHostArgs"
                    or values["execute_api"] != "aclnnSwiGlu"):
                raise ValueError("launch record is outside aclnnSwiGlu")
            args_path = _safe_child(
                raw_directory, values["args_file"], "host args")
            placeholder_path = _safe_child(
                raw_directory, values["placeholder_file"], "placeholder map")
            args = args_path.read_bytes()
            args_size = _parse_uint(
                values["args_size"], "host args size", positive=True)
            if len(args) != args_size:
                raise ValueError("host args file size differs from metadata")
            entry_size = _parse_uint(
                values["placeholder_entry_size"], "placeholder entry size",
                positive=True)
            if entry_size != 8:
                raise ValueError("aclrtPlaceHolderInfo must be eight bytes")
            placeholder_count = _parse_uint(
                values["placeholder_count"], "placeholder count")
            placeholder_bytes = placeholder_path.read_bytes()
            if len(placeholder_bytes) != placeholder_count * entry_size:
                raise ValueError("placeholder file size differs from metadata")
            placeholders = [
                {
                    "address_offset": address_offset,
                    "data_offset": data_offset,
                }
                for address_offset, data_offset in struct.iter_unpack(
                    "<II", placeholder_bytes)
            ]
            for placeholder in placeholders:
                if placeholder["address_offset"] + 8 > args_size:
                    raise ValueError("placeholder address slot exceeds host args")
                if placeholder["data_offset"] >= args_size:
                    raise ValueError("placeholder data offset exceeds host args")
            record = {
                "record_type": record_type,
                "sequence": _parse_uint(
                    values["sequence"], "launch sequence", positive=True),
                "api": values["api"],
                "execute_api": values["execute_api"],
                "pid": _parse_uint(values["pid"], "launch pid", positive=True),
                "tid": _parse_uint(values["tid"], "launch tid", positive=True),
                "stub_identity": _parse_stub_identity(values, path),
                "block_dim": _parse_uint(
                    values["block_dim"], "block dim", positive=True),
                "args_size_bytes": args_size,
                "args_path": str(args_path.resolve()),
                "args_sha256": _sha256(args_path),
                "placeholder_count": placeholder_count,
                "placeholder_path": str(placeholder_path.resolve()),
                "placeholder_sha256": _sha256(placeholder_path),
                "placeholders": placeholders,
                "execute_stream": _parse_uint(
                    values["execute_stream"], "execute stream"),
                "launch_stream": _parse_uint(
                    values["launch_stream"], "launch stream"),
                "metadata_path": str(path.resolve()),
                "metadata_sha256": _sha256(path),
            }
        else:
            raise ValueError(f"unknown capture record type: {record_type!r}")
        records.append(record)
    sequences = [record["sequence"] for record in records]
    if sequences != list(range(sequences[0], sequences[0] + len(records))):
        raise ValueError(f"capture sequences are not contiguous: {sequences}")
    return records


def _decode_payload(payload, layout):
    if len(payload) != layout["data_size_bytes"]:
        raise ValueError("serialized SwiGlu payload is not exactly 80 bytes")
    decoded = {}
    for field in layout["fields"]:
        dtype = field["dtype"]
        if dtype not in INTEGER_FORMATS:
            raise ValueError(f"unsupported reflected dtype: {dtype!r}")
        values = struct.unpack_from(
            "<" + INTEGER_FORMATS[dtype] * field["count"], payload,
            field["offset_bytes"])
        decoded[field["name"]] = values[0] if len(values) == 1 else list(values)
    return decoded


def _matches_contract(decoded, contract):
    boolean_fields = (
        "is32BAligned", "isDoubleBuffer", "activateLeft", "biasIsEmpty",
        "quantScaleIsEmpty", "activateScaleIsEmpty",
    )
    if any(decoded[name] not in (0, 1) for name in boolean_fields):
        return False
    if decoded["rowLen"] != contract["tok"]:
        return False
    if decoded["colLen"] != contract["intermediate"]:
        return False
    if decoded["baseRowLen"] <= 0 or decoded["baseColLen"] <= 0:
        return False
    if decoded["usedCoreNum"] > 48:
        return False
    expected_aligned = (
        decoded["colLen"] * contract["dtype_bytes"] % 32 == 0)
    return decoded["is32BAligned"] == int(expected_aligned)


def _candidate_payloads(record, layout, contract):
    args = Path(record["args_path"]).read_bytes()
    data_offsets = sorted({
        placeholder["data_offset"] for placeholder in record["placeholders"]
    })
    candidates = []
    for index, data_offset in enumerate(data_offsets):
        boundary = (
            data_offsets[index + 1]
            if index + 1 < len(data_offsets) else len(args))
        if boundary < data_offset + layout["data_size_bytes"]:
            continue
        payload = args[data_offset:data_offset + layout["data_size_bytes"]]
        decoded = _decode_payload(payload, layout)
        if _matches_contract(decoded, contract):
            address_offsets = sorted(
                placeholder["address_offset"]
                for placeholder in record["placeholders"]
                if placeholder["data_offset"] == data_offset)
            candidates.append({
                "data_offset_bytes": data_offset,
                "bounded_region_end_bytes": boundary,
                "bounded_region_size_bytes": boundary - data_offset,
                "address_offsets_bytes": address_offsets,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "decoded": decoded,
            })
    return candidates


def _tile_schedule(decoded, dtype_bytes):
    if dtype_bytes <= 0 or 32 % dtype_bytes != 0:
        raise ValueError("dtype size must divide one 32-byte data block")
    row_len = decoded["rowLen"]
    col_len = decoded["colLen"]
    base_row = decoded["baseRowLen"]
    base_col = decoded["baseColLen"]
    base_row_tiles, tail_row = divmod(row_len, base_row)
    base_col_tiles, tail_col = divmod(col_len, base_col)
    alignment_elements = 32 // dtype_bytes

    def cal_len(rows, cols):
        if decoded["is32BAligned"] == 1:
            return rows * cols
        aligned_cols = (
            (cols + alignment_elements - 1) // alignment_elements
            * alignment_elements)
        return rows * aligned_cols

    tile_classes = []
    class_inputs = (
        ("base_row_base_col", base_row, base_col,
         base_row_tiles * base_col_tiles),
        ("base_row_tail_col", base_row, tail_col,
         base_row_tiles if tail_col > 0 else 0),
        ("tail_row_base_col", tail_row, base_col,
         base_col_tiles if tail_row > 0 else 0),
        ("tail_row_tail_col", tail_row, tail_col,
         1 if tail_row > 0 and tail_col > 0 else 0),
    )
    for name, rows, cols, count in class_inputs:
        if count > 0:
            tile_classes.append({
                "name": name,
                "rows": rows,
                "columns": cols,
                "cur_cal_len_elements": cal_len(rows, cols),
                "tile_count": count,
            })
    if not tile_classes:
        raise ValueError("decoded tiling produces no executable tile")
    return {
        "row_tiles": base_row_tiles + (1 if tail_row > 0 else 0),
        "column_tiles": base_col_tiles + (1 if tail_col > 0 else 0),
        "total_tiles": sum(item["tile_count"] for item in tile_classes),
        "tile_classes": tile_classes,
        "maximum_cur_cal_len_elements": max(
            item["cur_cal_len_elements"] for item in tile_classes),
    }


def _stable_kernel_identity(launch_records, cann_root):
    raw_identities = [record["stub_identity"] for record in launch_records]
    kinds = {identity["kind"] for identity in raw_identities}
    if len(kinds) != 1:
        raise ValueError(
            f"live SwiGlu stub identity kind varies: {sorted(kinds)}")
    kind = next(iter(kinds))
    if kind != "runtime_function_name":
        raise ValueError(
            "live SwiGlu kernel identity must come from aclrtGetFunctionName; "
            "ELF coordinates are diagnostic-only")
    names = {
        identity["runtime_function_name"] for identity in raw_identities}
    if names != {EXPECTED_RUNTIME_FUNCTION_NAME}:
        raise ValueError(
            "live SwiGlu runtime function name changed: "
            f"expected={EXPECTED_RUNTIME_FUNCTION_NAME!r}, "
            f"actual={sorted(names)!r}")
    return {
        "kind": kind,
        "name": EXPECTED_RUNTIME_FUNCTION_NAME,
        "resolver": "aclrtGetFunctionName",
    }, EXPECTED_RUNTIME_FUNCTION_NAME


def build_report(records, layout, contract, cann_root=None):
    workspace_records = [
        record for record in records if record["record_type"] == "workspace"]
    launch_records = [
        record for record in records if record["record_type"] == "launch_args"]
    if (len(workspace_records) != EXPECTED_EXECUTIONS
            or len(launch_records) != EXPECTED_EXECUTIONS):
        raise ValueError(
            "SwiGlu capture must contain exactly six execute/launch pairs: "
            f"workspace={len(workspace_records)}, launch={len(launch_records)}")
    for workspace, launch in zip(workspace_records, launch_records):
        if workspace["sequence"] + 1 != launch["sequence"]:
            raise ValueError("each SwiGlu workspace record must precede its launch")
        if (workspace["pid"] != launch["pid"]
                or workspace["tid"] != launch["tid"]):
            raise ValueError("SwiGlu execute and launch changed process/thread")
        if (workspace["execute_stream"] != launch["execute_stream"]
                or launch["execute_stream"] != launch["launch_stream"]):
            raise ValueError("SwiGlu execute and launch stream identities differ")
    live_block_dims = {record["block_dim"] for record in launch_records}
    if len(live_block_dims) != 1:
        raise ValueError("live SwiGlu block_dim varies across executions")
    live_block_dim = next(iter(live_block_dims))
    expected_block_dim = contract["block_dim"]
    if (expected_block_dim is not None
            and live_block_dim != expected_block_dim):
        raise ValueError("live SwiGlu block_dim differs from canonical launch")
    kernel_identity, kernel_symbol = _stable_kernel_identity(
        launch_records, cann_root)
    candidate_records = []
    for record in launch_records:
        candidates = _candidate_payloads(record, layout, contract)
        if len(candidates) != 1:
            raise ValueError(
                "SwiGlu tiling payload is not uniquely identified from live "
                f"placeholders: sequence={record['sequence']}, "
                f"candidate_count={len(candidates)}")
        candidate_records.append(candidates[0])
    first = candidate_records[0]
    if any(candidate["decoded"] != first["decoded"]
           for candidate in candidate_records[1:]):
        raise ValueError("SwiGlu decoded tiling fields vary across executions")
    if any(candidate["payload_sha256"] != first["payload_sha256"]
           for candidate in candidate_records[1:]):
        raise ValueError("SwiGlu serialized tiling varies across executions")
    if any(candidate["data_offset_bytes"] != first["data_offset_bytes"]
           for candidate in candidate_records[1:]):
        raise ValueError("SwiGlu tiling data offset varies across executions")
    decoded = first["decoded"]
    return {
        "schema_version": 3,
        "artifact_type": "ascend_910b_swiglu_tiling_observation",
        "duration_consumed": False,
        "target_kernel_submitted": False,
        "case_contract": contract,
        "execution_count": EXPECTED_EXECUTIONS,
        "workspace_size_bytes": sorted({
            record["workspace_size_bytes"] for record in workspace_records}),
        "block_dim": live_block_dim,
        "kernel_symbol": kernel_symbol,
        "kernel_identity": kernel_identity,
        "tiling_used_core_num": decoded["usedCoreNum"],
        "tiling_used_core_num_matches_launch": (
            decoded["usedCoreNum"] == live_block_dim),
        "launch_api": "aclrtLaunchKernelWithHostArgs",
        "execute_api": "aclnnSwiGlu",
        "stream_identity_equal": True,
        "placeholder_maps": [
            record["placeholders"] for record in launch_records],
        "tiling_payload_selection": {
            "status": "unique_reflected_contract_match",
            "data_offset_bytes": first["data_offset_bytes"],
            "address_offsets_bytes": first["address_offsets_bytes"],
            "serialized_size_bytes": layout["data_size_bytes"],
            "bounded_region_size_bytes": first["bounded_region_size_bytes"],
            "payload_sha256": first["payload_sha256"],
        },
        "decoded_tiling": decoded,
        "source_derived_tile_schedule": _tile_schedule(
            decoded, contract["dtype_bytes"]),
        "reflection": {
            "registration_key": layout["registration_key"],
            "class_name": layout["class_name"],
            "data_size_bytes": layout["data_size_bytes"],
            "path": layout["path"],
            "sha256": layout["sha256"],
        },
        "limitations": [
            "the active aclnnSwiGlu marker records host launch arguments and "
            "suppresses the target kernel submission; torch and CANN "
            "initialization outside that marker may submit other device kernels",
            "observation records launch and tiling facts only; no duration, "
            "cycle, utilization, or evaluation error is consumed",
            "decoded fields are valid only for the hashed CANN 8.5.1 build, "
            "BF16 SwiGlu key-27 source chain, and stated workload contract",
            "usedCoreNum is retained verbatim; key 27 emits zero while the "
            "runtime launch block_dim is retained separately, so usedCoreNum "
            "is not used as launch geometry",
        ],
    }


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--reflection-tsv", type=Path, required=True)
    parser.add_argument("--contract-json", type=Path, required=True)
    parser.add_argument("--cann-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    contract = json.loads(args.contract_json.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise TypeError("SwiGlu contract JSON must be a mapping")
    records = parse_capture_records(args.capture_dir / "raw")
    layout = parse_reflection(args.reflection_tsv)
    report = build_report(records, layout, contract, args.cann_root)
    _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
