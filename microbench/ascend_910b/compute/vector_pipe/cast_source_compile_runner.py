#!/usr/bin/env python3
"""CPU-only diagnostic recompile of installed TBE Cast, with exact code gate.

Saving generated CCE/IR changes diagnostics only. Generated source cannot bind
the production object unless its selected function bytes match exactly.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_bytes(path, name, *, required=True):
    raw = path.read_bytes()
    if raw[:6] != b"\x7fELF\x02\x01":
        raise ValueError("need a little-endian ELF64 object")
    header = struct.unpack_from("<HHIQQQIHHHHHH", raw, 16)
    section_offset, section_stride, section_count = header[5], header[10], header[11]
    if section_stride != 64 or section_count == 0:
        raise ValueError("unsupported ELF section table")
    sections = [struct.unpack_from("<IIQQQQIIQQ", raw, section_offset + index * section_stride)
                for index in range(section_count)]
    matches = []
    for section in sections:
        if section[1] != 2:
            continue
        strings = sections[section[6]]
        string_data = raw[strings[4]:strings[4] + strings[5]]
        if section[9] != 24:
            raise ValueError("unsupported symbol table stride")
        for offset in range(section[4], section[4] + section[5], 24):
            string_offset, info, _, target, value, size = struct.unpack_from("<IBBHQQ", raw, offset)
            symbol = string_data[string_offset:].split(b"\0", 1)[0].decode()
            if symbol != name or info & 15 != 2:
                continue
            if target >= len(sections) or size <= 0:
                raise ValueError("invalid selected function section")
            code = sections[target]
            begin = code[4] + value - code[3]
            if begin < code[4] or begin + size > code[4] + code[5]:
                raise ValueError("selected function exceeds its section")
            matches.append(raw[begin:begin + size])
    if not matches and not required:
        return None
    if len(matches) != 1:
        raise ValueError("expected exactly one selected function in " + str(path))
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--binding-json", type=Path, required=True)
    parser.add_argument("--selected-object", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, selected, output = args.source.resolve(), args.selected_object.resolve(), args.output.resolve()
    binding = json.loads(args.binding_json.read_text())
    if output.exists():
        raise FileExistsError("refusing to replace previous diagnostic compilation")
    if sha256(source) != args.expected_source_sha256:
        raise ValueError("source differs from frozen installed Cast")
    if sha256(selected) != binding["selected_object_sha256"]:
        raise ValueError("selected object differs from runtime binding")
    reference = function_bytes(selected, binding["runtime_kernel_function"])
    text = source.read_text()
    anchor = 'config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}'
    if text.count(anchor) != 1:
        raise ValueError("installed Cast build-config anchor changed")
    replacement = anchor[:-1] + ', "build_args": {"save_temp_cce_file": True, "dump_cce_code": True, "dump_ir": True}}'
    output.mkdir(parents=True)
    patched = output / "diagnostic_cast.py"
    patched.write_text(text.replace(anchor, replacement))
    os.chdir(output)
    from tbe.common import platform
    from tbe.common.buildcfg import build_config
    platform.set_current_compile_soc_info("Ascend910B2", core_type="VectorCore")
    if int(platform.get_soc_spec("CORE_NUM")) != binding["compile_info"]["_base_info"]["100"][0]:
        raise ValueError("actual target disagrees with selected compiler metadata")
    spec = importlib.util.spec_from_file_location("midlayer_diagnostic_cast", patched)
    if spec is None or spec.loader is None:
        raise ValueError("could not load frozen diagnostic source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    name = selected.stem
    with build_config(kernel_meta_parent_dir=str(output)):
        module.cast({"shape": [-1], "range": [(1, None)], "dtype": "int32", "format": "ND"},
                    {"shape": [-1], "range": [(1, None)], "dtype": "int64", "format": "ND"},
                    None, kernel_name=name)
    candidates = []
    for path in sorted(output.rglob("*.o")):
        code = function_bytes(path, binding["runtime_kernel_function"], required=False)
        if code is None:
            continue
        candidates.append({"path": str(path.relative_to(output)), "sha256": sha256(path),
                           "selected_function_size": len(code),
                           "selected_function_sha256": hashlib.sha256(code).hexdigest(),
                           "matches_runtime_function_bytes": code == reference})
    matched = bool(candidates) and all(row["matches_runtime_function_bytes"] for row in candidates)
    (output / "receipt.json").write_text(json.dumps({"schema_version": 1, "baseline": args.baseline,
        "CPU_only": True, "device_execution": False, "duration_consumed": False,
        "source_sha256": sha256(source), "selected_object_sha256": sha256(selected),
        "diagnostic_flags_only": ["save_temp_cce_file", "dump_cce_code", "dump_ir"],
        "selected_function_size": len(reference), "candidates": candidates,
        "generated_source_bound_to_runtime_object": matched,
        "production_rate_or_serving_acceptance": False}, indent=2) + "\n")
    if not matched:
        raise ValueError("recompiled function does not exactly match the selected runtime code")


if __name__ == "__main__":
    main()
