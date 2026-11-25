#!/usr/bin/env python3
"""Extract non-duration launch geometry from real H200 vLLM nsys GPU traces.

Input: the ``cuda_gpu_trace`` CSV exports of the DeepSeek-V3 (32-layer) TP4
decode-uniform real run on target host (vLLM 0.24.0, torch 2.11, triton 3.6,
cuBLASLt 12.9).  Only the launch-geometry columns (GrdX..BlkZ, Reg/Trd,
StcSMem, DymSMem), the kernel ``Name`` and the ``Device`` identity are read;
the ``Start``/``Duration``/throughput columns are never indexed, so no timing
value can reach the output.  Rows without grid dimensions (memcpy/memset) are
skipped.

The output groups every distinct kernel name with each distinct geometry tuple
it was launched with and the number of launches per window (a launch count,
not a time), and labels the operator the kernel implements from its symbol.

Standard library only so it runs with the remote system ``python3``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

GEOMETRY_COLUMNS = (
    "GrdX", "GrdY", "GrdZ", "BlkX", "BlkY", "BlkZ", "Reg/Trd",
    "StcSMem (MB)", "DymSMem (MB)",
)
IDENTITY_COLUMNS = ("Device", "Name")
FORBIDDEN_OUTPUT_TOKENS = ("duration", "Duration", "Start (ns)", "Throughput")

# Ordered (label, regex) rules; first match wins; every kernel must match one.
OPERATOR_RULES = (
    ("deepgemm_fp8_block_gemm", r"^void deep_gemm::sm90_fp8_gemm_1d2d_impl<"),
    ("deepgemm_fp8_block_gemm_swap_ab",
     r"^void deep_gemm::fp8_gemm_kernel_swapAB<"),
    ("moe_router_gemm_bf16_fp32_out", r"^void router_gemm_kernel_float_output<"),
    ("cublaslt_bf16_gemm_nvjet", r"^nvjet_"),
    ("cublaslt_splitk_reduce", r"cublasLt::splitKreduce_kernel<"),
    ("triton_fused_moe_gemm", r"^fused_moe_kernel$"),
    ("moe_align_block_size", r"vllm::moe::moe_align_block_size_kernel<"),
    ("moe_count_and_sort_expert_tokens",
     r"vllm::moe::count_and_sort_expert_tokens_kernel<"),
    ("moe_sum_reduce", r"vllm::moe::moe_sum_kernel<|moe_sum"),
    ("fa3_mla_attention_fwd_sm90",
     r"^void cutlass::device_kernel<flash::enable_sm90_or_later<"
     r"flash::FlashAttnFwdSm90<"),
    ("fa3_attention_split_combine",
     r"^void cutlass::device_kernel<flash::FlashAttnFwdCombine<"),
    ("fa3_prepare_varlen_num_blocks", r"flash::prepare_varlen_num_blocks_kernel<"),
    ("flashinfer_trtllm_allreduce_fusion",
     r"flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel"),
    ("nccl_collective", r"^ncclDevKernel_|^ncclKernel_"),
    ("vllm_custom_allreduce", r"vllm::cross_device_reduce"),
    ("per_token_group_quant_fp8", r"per_token_group_quant_8bit_kernel<"),
    ("scale_1x128_fp8_quant", r"scale_1x128|per_token_group_quant_fp8_colmajor"),
    ("mla_kv_cache_write", r"vllm::concat_and_cache_mla_kernel<"),
    ("silu_and_mul_activation", r"vllm::act_and_mul_kernel<"),
    ("rmsnorm", r"rms_norm|rmsnorm|RMSNorm"),
    ("rotary_embedding", r"rotary_embedding|rope"),
    ("topk_router", r"topk|grouped_topk|TopK"),
    # Inductor-fused kernels are labelled from the aten ops their name lists:
    # mul+silu+slice is SiLU-and-mul; neg+slice+stack(+index) is the
    # rotate-half rotary embedding.  These two labels are inferences from the
    # fused-op list, recorded as such in ``label_basis``.
    ("silu_and_mul_activation_inductor", r"^triton_poi_fused_mul_silu_slice_"),
    ("rotary_embedding_inductor",
     r"^triton_poi_fused_add_clone_copy_expand_index_mul_neg_slice_split_"
     r"stack_unsqueeze_view_"),
    ("torch_inductor_triton_pointwise", r"^triton_poi_"),
    ("torch_inductor_triton_reduction", r"^triton_red_|^triton_per_"),
    ("torch_softmax", r"at::native::<unnamed>::cunn_SoftMax"),
    ("torch_sampler_random", r"at::native::<unnamed>::distribution_"),
    ("torch_reduce", r"^void at::native::reduce_kernel<"),
    ("torch_gather_index", r"at::native::(index_elementwise_kernel|"
     r"vectorized_gather_kernel|_scatter_gather_elementwise_kernel)"),
    ("torch_elementwise", r"^void at::native::(vectorized_|unrolled_|)"
     r"elementwise_kernel<"),
    ("vllm_slot_mapping", r"^_compute_slot_mapping_kernel$"),
    ("triton_other", r"^_[a-z_0-9]+_kernel$"),
)


def label(name: str) -> str:
    for operator, pattern in OPERATOR_RULES:
        if re.search(pattern, name):
            return operator
    raise ValueError(f"kernel name matches no operator rule: {name[:200]!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _int(text: str, context: str) -> int:
    value = int(text)
    if value < 0:
        raise ValueError(f"{context} is negative")
    return value


def extract(paths: list[Path]) -> dict[str, object]:
    kernels: dict[str, dict[tuple, dict[str, int]]] = {}
    devices: set[str] = set()
    windows = []
    for path in paths:
        window = path.name.replace("_cuda_gpu_trace.csv", "")
        with path.open(newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            index = {name: header.index(name)
                     for name in GEOMETRY_COLUMNS + IDENTITY_COLUMNS}
            kernel_rows = 0
            skipped = 0
            for row in reader:
                if row[index["GrdX"]] == "":
                    skipped += 1
                    continue
                kernel_rows += 1
                name = row[index["Name"]]
                device = row[index["Device"]]
                devices.add(device)
                geometry = (
                    tuple(_int(row[index[c]], c) for c in GEOMETRY_COLUMNS[:7])
                    + (row[index["StcSMem (MB)"]], row[index["DymSMem (MB)"]])
                )
                if name not in kernels:
                    kernels[name] = {}
                if geometry not in kernels[name]:
                    kernels[name][geometry] = {}
                counts = kernels[name][geometry]
                counts[window] = (counts[window] if window in counts else 0) + 1
        windows.append({
            "window": window,
            "source_csv": str(path),
            "source_csv_sha256": sha256_file(path),
            "kernel_launch_rows": kernel_rows,
            "non_kernel_rows_skipped": skipped,
        })

    table = []
    for name in sorted(kernels):
        geometries = []
        for geometry, counts in sorted(kernels[name].items()):
            (gx, gy, gz, bx, by, bz, regs, static_mb, dynamic_mb) = geometry
            geometries.append({
                "grid": [gx, gy, gz],
                "block": [bx, by, bz],
                "ctas": gx * gy * gz,
                "threads_per_cta": bx * by * bz,
                "registers_per_thread": regs,
                "static_shared_memory_mb_nsys": static_mb,
                "dynamic_shared_memory_mb_nsys": dynamic_mb,
                "launch_count_by_window": dict(sorted(counts.items())),
            })
        table.append({
            "kernel_name": name,
            "operator": label(name),
            "distinct_geometry_count": len(geometries),
            "geometries": geometries,
        })
    operators = {}
    for entry in table:
        if entry["operator"] not in operators:
            operators[entry["operator"]] = []
        operators[entry["operator"]].append(entry["kernel_name"])
    return {
        "windows": windows,
        "devices": sorted(devices),
        "distinct_kernel_count": len(table),
        "operator_index": {k: sorted(v) for k, v in sorted(operators.items())},
        "kernels": table,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = {
        "schema_version": 1,
        "artifact_type": "production_kernel_launch_geometry",
        "hardware_id": "nvidia_h200",
        "workload": "deepseek_v3_32layer_tp4_decode_uniform_real_vllm_run",
        "software_stack": {
            "vllm": "0.24.0+cu129", "torch": "2.11", "triton": "3.6",
            "cublaslt": "12.9",
        },
        "production_or_evaluation_duration_consumed": False,
        "duration_columns_read": False,
        "columns_read": list(GEOMETRY_COLUMNS + IDENTITY_COLUMNS),
        "shared_memory_unit_note": (
            "nsys cuda_gpu_trace reports StcSMem/DymSMem in MB with three "
            "decimals; values are kept verbatim as strings (resolution "
            "0.001 MB) because the CSV does not carry exact bytes"),
        "label_basis": (
            "operator labels are regex rules over the demangled kernel symbol "
            "(OPERATOR_RULES in the extractor); *_inductor labels are "
            "inferred from the aten op list in the Inductor kernel name"),
        "extractor": "microbench/nvidia_h200/operators/launch_geometry/"
                     "nsys_trace_launch_geometry.py",
        "extractor_sha256": sha256_file(Path(__file__).resolve()),
        **extract(args.trace_csv),
    }
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    # The two boundary flags above name "duration" only to deny it; every
    # extracted section must not mention a timing column at all.
    extracted = json.dumps(
        {key: payload[key] for key in ("windows", "kernels", "columns_read")})
    for token in FORBIDDEN_OUTPUT_TOKENS:
        if token in extracted:
            raise ValueError(f"output would contain forbidden token {token!r}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(json.dumps({"output": str(args.output),
                      "kernels": payload["distinct_kernel_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
