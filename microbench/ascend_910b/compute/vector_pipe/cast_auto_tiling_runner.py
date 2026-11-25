#!/usr/bin/env python3
"""CPU-only selected TBE Cast partition query using the installed public API.

No tensors, device initialization, kernel execution or timing. Compile-info is
read unchanged from the selected-object binding. The AutoTiling entry follows
the same-version TBE elewise schedule, not the unrelated registered Cast key.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding-json", type=Path, required=True)
    parser.add_argument("--observed-projection", type=Path, required=True)
    parser.add_argument("--expected-wrapper-sha256", required=True)
    parser.add_argument("--soc-version", choices=("Ascend910B2",), required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to replace a previous tiling receipt")
    if len(args.baseline) != 40 or any(c not in "0123456789abcdef" for c in args.baseline):
        parser.error("--baseline must name the frozen commit")
    binding = json.loads(args.binding_json.read_text())
    observed = json.loads(args.observed_projection.read_text())
    compile_info = binding["compile_info"]
    key = str(binding["runtime_tiling_key"])
    if (compile_info["_pattern"] != "ElemWise" or compile_info["_int64_mode"] is not True
            or compile_info["_vars"][key] != ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]):
        raise ValueError("this runner requires the exact selected int64-mode ElemWise layout")
    from tbe.common.utils import op_tiling
    wrapper = Path(op_tiling.__file__)
    if sha256(wrapper) != args.expected_wrapper_sha256:
        raise ValueError("installed public tiling wrapper differs from the frozen source")
    from tbe.common import platform
    if binding["object_core_type"] != "VectorCore":
        raise ValueError("this selected Cast requires the explicit VectorCore target")
    platform.set_current_compile_soc_info(args.soc_version, core_type="VectorCore")
    platform_info = {"soc_version": args.soc_version, "core_type": "VectorCore",
                     "core_num": platform.get_soc_spec("CORE_NUM"),
                     "ub_size_bytes": platform.get_soc_spec("UB_SIZE")}
    if int(platform_info["core_num"]) != compile_info["_base_info"]["100"][0]:
        raise ValueError("installed VectorCore target disagrees with selected-object compileInfo")
    expected = {}
    for row in observed["rows"]:
        if row["launch"] == "target":
            count = row["elements"]
            if count in expected:
                raise ValueError("duplicate observed target shape")
            expected[count] = row
    if not expected or observed["duration_consumed"] is not False:
        raise ValueError("need non-duration target observations")
    rows = []
    for count, reference in expected.items():
        # The public wrapper mutates compile-info/descriptors. Give each call
        # its own copy, leaving the selected source metadata unchanged.
        result = op_tiling.do_op_tiling("AutoTiling", copy.deepcopy(compile_info),
            [{"shape": [count], "dtype": "int32"}],
            [{"shape": [count], "dtype": "int64"}], timer=None)
        payload = result["tiling_data"]
        if not isinstance(payload, bytes) or len(payload) < 24:
            raise ValueError("AutoTiling returned an unknown inline variable extent")
        returned_key = int(result["tiling_key"])
        blocks = result["block_dim"]
        values = struct.unpack_from("<qqq", payload)
        fields = dict(zip(compile_info["_vars"][key], values))
        same = (returned_key == binding["runtime_tiling_key"]
                and blocks == reference["block_dim"] and fields == reference["tiling_variables"])
        rows.append({"elements": count, "block_dim": blocks, "tiling_key": returned_key,
                     "tiling_data_hex": payload.hex(), "tiling_variables": fields,
                     "workspaces": result["workspaces"], "clear_atomic": result["clear_atomic"],
                     "matches_observed_host_arguments": same})
        if not same:
            raise ValueError("public AutoTiling disagrees with captured runtime partition: " + json.dumps(rows[-1]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "baseline": args.baseline,
        "CPU_only": True, "device_execution": False, "duration_consumed": False,
        "entry": "tbe.common.utils.op_tiling.do_op_tiling(AutoTiling, timer=None)",
        "explicit_target_platform": platform_info,
        "wrapper": {"path": str(wrapper), "sha256": sha256(wrapper)},
        "binding": {"path": str(args.binding_json), "sha256": sha256(args.binding_json)},
        "observations": {"path": str(args.observed_projection), "sha256": sha256(args.observed_projection)},
        "rows": rows, "all_observed_partitions_match": True,
        "rule_fitted_from_observations": False, "selected_ISA_bound": False,
        "production_rate_or_serving_acceptance": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
