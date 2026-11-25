"""Query CANN's real CPU tiler with tensor descriptors, never device tensors.

This diagnostic emits candidate discrete launch facts only. It does not call
ACL initialization, allocate device memory, run a kernel, or record duration.
The input schema/attribute order follows the installed CANN8.5.1 FIA prototype.
"""

import argparse
import hashlib
import json
from pathlib import Path
import resource


def descriptor(name, shape, dtype, *, values=None):
    value = {"name": name, "shape": shape, "ori_shape": shape,
        "format": "ND", "ori_format": "ND", "dtype": dtype}
    if values is not None:
        value["const_value"] = values
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-lens", nargs="+", type=int, required=True)
    parser.add_argument("--kv-lens", nargs="+", type=int, required=True)
    parser.add_argument("--kv-layout", choices=("paged", "contiguous"), default="paged")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (len(args.q_lens) != len(args.kv_lens) or not args.q_lens
            or any(q < 0 for q in args.q_lens) or not any(args.q_lens)
            or any(k < 0 for k in args.kv_lens) or not any(args.kv_lens)):
        raise ValueError("FIA query requires nonnegative Q/KV with nonempty total input")
    if args.output.exists():
        raise FileExistsError(args.output)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    from tbe.common.platform import set_current_compile_soc_info
    from tbe.common.utils.op_tiling import do_op_tiling

    sdk = Path("/usr/local/Ascend/cann-8.5.1")
    version = (sdk / "opp/version.info").read_text()
    set_current_compile_soc_info("Ascend910B2")
    rows, batch = sum(args.q_lens), len(args.q_lens)
    pages = [(k + 127) // 128 for k in args.kv_lens]
    cumulative, total = [], 0
    for q in args.q_lens:
        total += q
        cumulative.append(total)
    inputs = [None] * 31
    inputs[0] = descriptor("query", [rows, 32, 128], "bfloat16")
    kv_shape = [sum(pages), 128, 1024] if args.kv_layout == "paged" else [sum(args.kv_lens), 8, 128]
    inputs[1] = [descriptor("key", kv_shape, "bfloat16")]
    inputs[2] = [descriptor("value", kv_shape, "bfloat16")]
    inputs[4] = descriptor("atten_mask", [2048, 2048], "int8")
    inputs[5] = descriptor("actual_seq_lengths", [batch], "int64", values=cumulative)
    kv_values = args.kv_lens
    if args.kv_layout == "contiguous":
        kv_values, kv_total = [], 0
        for length in args.kv_lens:
            kv_total += length
            kv_values.append(kv_total)
    inputs[6] = descriptor("actual_seq_lengths_kv", [batch], "int64", values=kv_values)
    if args.kv_layout == "paged":
        inputs[14] = descriptor("block_table", [batch, max(pages)], "int32")
    outputs = [descriptor("attention_out", [rows, 32, 128], "bfloat16"),
        descriptor("softmax_lse", [1], "float32")]
    # All sixteen slots are required by the native attribute reader, even
    # for values with defaults in the public prototype. inner_precise=0 is
    # the torch_npu API default; all other defaults follow fusion_ops.h.
    attrs = [{"name": name, "dtype": dtype, "value": value}
        for name, dtype, value in (
            ("num_heads", "int", 32), ("scale", "float", 128 ** -0.5),
            ("pre_tokens", "int", 2147483647), ("next_tokens", "int", 2147483647),
            ("input_layout", "str", "TND"), ("num_key_value_heads", "int", 8),
            ("sparse_mode", "int", 3), ("inner_precise", "int", 0),
            ("block_size", "int", 128), ("antiquant_mode", "int", 0),
            ("softmax_lse_flag", "bool", False), ("key_antiquant_mode", "int", 0),
            ("value_antiquant_mode", "int", 0), ("query_quant_mode", "int", 0),
            ("pse_type", "int", 0), ("out_dtype", "int", 0))]
    result = dict(do_op_tiling("FusedInferAttentionScore", {}, inputs, outputs, attrs=attrs))
    result["tiling_data_hex"] = result["tiling_data"].hex()
    del result["tiling_data"]
    files = (
        Path(__file__).resolve(),
        sdk / "python/site-packages/tbe/common/utils/op_tiling.py",
        sdk / "opp/built-in/op_proto/inc/fusion_ops.h",
        sdk / "opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/aarch64/libophost_transformer.so",
        sdk / "aarch64-linux/lib64/libregister.so")
    record = {"artifact_type": "cpu_host_tiling_query_candidate",
        "device_execution": False, "duration_consumed": False,
        "soc_version": "Ascend910B2", "sdk_version_info": version,
        "scope": "synthetic BF16 TND/H32/KV8/D128 descriptors with explicit KV layout; no actual provider execution claim",
        "kv_layout": args.kv_layout,
        "q_lens": args.q_lens, "kv_lens": args.kv_lens,
        "inputs": inputs, "outputs": outputs, "attrs": attrs, "compile_info": {},
        "source_files": [{"path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in files],
        "result": result}
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "tiling_key": result["tiling_key"],
        "block_dim": result["block_dim"], "tiling_bytes": len(result["tiling_data_hex"]) // 2}), flush=True)


if __name__ == "__main__":
    main()
