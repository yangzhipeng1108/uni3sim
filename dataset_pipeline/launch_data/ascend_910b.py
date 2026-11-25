#!/usr/bin/env python3
"""Package legal Huawei 910B launch facts.

The FlashAttention output contains only versioned discrete launch facts.  Its
generator reads the static workload contract, Attr fields, kernel symbol and
grid/block/task fields; it never reads duration, cycle, pipe, traffic or
counter fields.  Those entries are valid production launch-selection inputs.

Outputs:
  launch_data/ascend_910b/gemm_launch_selection.csv.zst
  launch_data/ascend_910b/gemm_launch_selection.manifest.json
  launch_data/ascend_910b/flash_attn_launch_facts.json
  launch_data/ascend_910b/swiglu_launch_facts.json

Usage:
  python scripts/launch_data/ascend_910b_launch_generator.py --write
"""
import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path

from dataset_pipeline.launch_data.gemm_workload_candidates import (
    HARDWARE_IDS as GEMM_HARDWARE_IDS,
    PHYSICAL_OPERATION_FIELDS,
)
from dataset_pipeline.kernel_dataset.config import (
    file_sha256,
    require_non_empty_string,
    require_positive_int,
)
from dataset_pipeline.launch_data.json_validators import (
    canonical_sequence_sha256,
    parse_dim3_csv,
    require_integer,
    require_sha256_hex,
    sorted_json_sha256,
)
from src.kernel.launch_config.gemm_workload import (
    SERVING_MODEL_PARALLELISM as GEMM_MODEL_PARALLELISM,
    SERVING_SCHEDULER_CONTRACT as EXPECTED_SCHEDULER_CONTRACT,
)
from src.kernel.launch_config.flash_attn_workload import (
    flash_attn_workload_key,
)
from src.kernel.launch_config.ascend_910b.gemm_launch import (
    GEMM_EXACT_DOMAIN_SHA256,
    GEMM_EXACT_PHYSICAL_OPERATION_COUNT,
    GEMM_EXACT_SEMANTIC_KEY_COUNT,
    MATMUL_V2_KERNEL_PATTERN,
    MATMUL_V2_TILING_PAYLOAD_SIZE_BYTES,
    MATMUL_V3_KERNEL_PATTERN,
    MATMUL_V3_TILING_PAYLOAD_SIZE_BYTES,
    decode_matmul_tiling_payload,
    matmul_tiling_abi_provenance,
)
from src.kernel.launch_config.gemm_workload import (
    GEMM_WORKLOAD_KEY_FIELDS,
    validate_gemm_workload_key,
)
from dataset_pipeline.launch_data.canonical_projection import (
    build_gemm_launch_selection_artifact,
    compact_json_text,
    compact_ascend_910b_flash_attn,
    compact_ascend_910b_swiglu,
    expanded_evidence_sha256,
    project_ascend_910b_gemm,
    write_gemm_launch_selection_artifact,
)
from src.kernel.launch_config.gemm_launch_selection import (
    GEMM_LAUNCH_CSV_FILENAME,
    GEMM_LAUNCH_MANIFEST_FILENAME,
    decompress_gemm_launch_csv,
    load_gemm_launch_selection_files,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "kernel_dataset" / "ascend_910b"
OUT_DIR = ROOT / "launch_data" / "ascend_910b"
FLASH_ATTN_OUT = OUT_DIR / "flash_attn_launch_facts.json"
SWIGLU_OUT = OUT_DIR / "swiglu_launch_facts.json"
GEMM_CSV_OUT = OUT_DIR / GEMM_LAUNCH_CSV_FILENAME
GEMM_MANIFEST_OUT = OUT_DIR / GEMM_LAUNCH_MANIFEST_FILENAME
FLASH_ATTN_DATASET = DATASET_DIR / "flash_attn.json"
FLASH_ATTN_WORKLOAD_CONTRACT = (
    ROOT / "configs" / "dataset" / "kernel" / "flash_attn.yaml")
FLASH_ATTN_SOURCE_SCHEDULE_REFERENCE = (
    ROOT / "microbench" / "ascend_910b" / "references"
    / "fused_infer_attention_score_cann_8_5_1.json")
FLASH_ATTN_RUNTIME_PREFIX = (
    "FusedInferAttentionScore_3b093497fc536d61a77a7a3293a524da_")
FLASH_ATTN_RUNTIME_SUFFIX = "_mix_aic"
FLASH_ATTN_METADATA_TILING_KEYS = frozenset({
    "103000000010022221",
    "103000000010222221",
    "103000000010422221",
    "103000000010522221",
    "103000000010622221",
    "105000000000222220",
    "105000000000322220",
    "5000000000000200203",
    "5000000000010200203",
})
FLASH_ATTN_ATTRIBUTE_KEYS = {
    "B", "S", "Q_LEN", "Q_REQUESTED", "KV_LEN", "KV_REQUESTED",
    "HQ", "HKV", "D", "SP", "SP_REQUESTED", "Q_LENS", "KV_LENS",
    "split_policy", "kv_layout", "kv_dist", "causal", "layer",
}
FLASH_ATTN_SOURCE_SCHEDULE_REFERENCE_FIELDS = {
    "schema_version", "hardware_id", "software_revision",
    "duration_consumed", "canonical_dataset_sha256",
    "capture_source_archives", "source_evidence", "entry_fields",
    "base_fields", "inner_split_fields", "workspace_element_fields",
    "worker_exclusive_end_fields", "flash_decode_fields", "entries",
}
FLASH_ATTN_SOURCE_SCHEDULE_ENTRY_FIELDS = [
    "capture_baseline_commit", "capture_directory", "manifest_sha256",
    "tiling_data_sha256", "serialized_tiling_sha256",
    "serialized_tiling_size_bytes", "runtime_variant", "base",
    "inner_split", "workspace_elements", "worker_exclusive_ends",
    "flash_decode",
]
FLASH_ATTN_SOURCE_SCHEDULE_BASE_FIELDS = [
    "batch", "kv_heads", "group_size", "q_len", "kv_len", "head_dim",
    "used_core_num", "output_layout", "softmax_lse_flag",
]
FLASH_ATTN_SOURCE_SCHEDULE_INNER_SPLIT_FIELDS = [
    "m_base_size", "s2_base_size",
]
FLASH_ATTN_SOURCE_SCHEDULE_WORKSPACE_FIELDS = [
    "mm1", "mm2", "flash_decode_accum", "flash_decode_lse",
]
FLASH_ATTN_SOURCE_SCHEDULE_WORKER_END_FIELDS = ["b_n2", "g_s1", "s2"]
FLASH_ATTN_SOURCE_SCHEDULE_FLASH_DECODE_FIELDS = [
    "used_vector_core_num", "head_task_count", "g_s1_base_size",
    "head_b_n2", "head_g_s1", "head_s2_split_count",
    "worker_s2_split_start", "head_g_s1_split_count",
    "head_g_s1_last_size", "vector_worker_head_end",
    "vector_worker_head_split_end",
]
FLASH_ATTN_SOURCE_SCHEDULE_CAPTURE_ARCHIVES = {
    "4a63ffd15052598ec8355709dcfa7b726646e092": (
        "50d97d879c391f4d4a8ba28a626e13af8ce807238a360a32b6f1a9bb55cbfe53"),
    "5e8d6d677cef7b606f1da41b37b922fe0eaf6d77": (
        "cc6f74d676f22541c7ad2f0dad60a874a4e6599de64fb99046eb85bffe6f5fd2"),
    "70140e6c7b0c43c059b70a2cfa220771d4a05d60": (
        "f7b7a040ffe3933c1208a4612fa136cdd3ca8b87fbfe460091f8ac5dcf291547"),
    "9516b048e10dfe84ac0f44646df3eb3e858e9ce3": (
        "f8b07b7ffc2c88cfb1c76ab02ddc84620384fa31bebc2365a6fc780a69271618"),
    "12d5612d4f0b1d1d8a4a79c723651ffd4b2fa078": (
        "789feb723f48c8059b1c25a9cdbaf2d685d574c0e35858bf5805abf0fc733b39"),
    "e6e3ff386e496eeeb370b082319ef14282f2bfac": (
        "2a31f8e900b2fa6d4f4fb2e0be3bd70162a47ff2283a6a97da3f042e9d4bb2f3"),
}
FLASH_ATTN_SOURCE_SCHEDULE_EVIDENCE = {
    "opp/version.info": (
        "49437edb9a235b28a87d07d2a2ad189022b6d246dc7c4cbcd013d2b7ecaf838b"),
    "fused_infer_attention_score_v3.cpp": (
        "9315d547e7468e28c5e38656ec01b8c9ef876bf53715cd166a545c7232305961"),
    "fia_kernel_nonquant.h": (
        "7cde05730d68bb02a2c45cd6b25605417ee61d42dfd87f21ccf0948c014b7e21"),
    "fia_kernel_nonquant_mla.h": (
        "18f001420f2bb7fe7a0109e3d0e39787cc56857172754df171a009d577942516"),
    "fia_public_define.h": (
        "1339e51f2bb1b2e6002e9a1b1b342523b4bb4007c86242d83b681d605ba52fb4"),
    "fia_block_vec_nonquant.h": (
        "259f5700812eafe32bcffd075f59d90c82cb71458f62bd47cb3d6f17f171e762"),
    "fia_block_vec_nonquant_mla.h": (
        "62a072eaa46f9ed2bf6b0da21ed7bd6ce5eaa288f784aa76307fa640cae3986e"),
    "fia_block_cube_nonquant_gqa.h": (
        "acce194dcf01f370d864ba7987f0c5a9d5927f7a69fb47426afa7789a6f856ef"),
    "fia_block_cube_nonquant_mla.h": (
        "c2d054e58a94f72c1a01f6f70bd77c21eba12939f01fabfe263d062a2088212f"),
    "fia_block_vec_flashdecode.h": (
        "04ebb6eb9dd85da8cf75d57d466be0c6ec8980b4cf366294dca5cb7495faef45"),
}
FLASH_ATTN_SOURCE_SCHEDULE_VARIANTS = frozenset({
    "103000000010022221",
    "103000000010222221",
    "103000000010422221",
    "103000000010522221",
    "103000000010622221",
    "105000000000222220",
    "105000000000322220",
})
FLASH_ATTN_FLASH_DECODE_VARIANTS = frozenset({
    "103000000010522221",
    "105000000000322220",
})

SWIGLU_REFLECTION_SHA256 = (
    "2988a095118fd9df6b28226569c580946b3b7b6b578c1ceaf1ee481bfc4d9415")
SWIGLU_DECODED_TILING_KEYS = {
    "SwiGluTilingDataPH", "activateLeft", "activateScaleIsEmpty",
    "baseColLen", "baseRowLen", "biasIsEmpty", "colLen", "is32BAligned",
    "isDoubleBuffer", "modRowLen", "perRowLen", "quantScaleIsEmpty",
    "rowLen", "swiColLen", "usedCoreNum",
}
SWIGLU_TILE_SCHEDULE_KEYS = {
    "column_tiles", "maximum_cur_cal_len_elements", "row_tiles",
    "tile_classes", "total_tiles",
}
SWIGLU_PAYLOAD_SELECTION_KEYS = {
    "address_offsets_bytes", "bounded_region_size_bytes", "data_offset_bytes",
    "payload_sha256", "serialized_size_bytes", "status",
}
SWIGLU_CANN_EVIDENCE = {
    "opp/version.info": (
        "49437edb9a235b28a87d07d2a2ad189022b6d246dc7c4cbcd013d2b7ecaf838b"),
    "aarch64-linux/include/acl/acl_rt.h": (
        "8cce153ef9eb5ed49b0f5bfb5608de706e8d466c836ad2eb50745ec2e8629bc9"),
    "aarch64-linux/include/aclnn/acl_meta.h": (
        "be93460abfc7aa14113d7b9a3536fe2dd45d5148a357f55e15847655c9d5ffed"),
    "aarch64-linux/include/aclnnop/aclnn_swi_glu.h": (
        "20e999b2ccbeb985adf2142760491fa2730cd8cb69723dd3bf5c2c6ebdcef1e3"),
    "aarch64-linux/lib64/libacl_rt.so": (
        "b6d7e6a70c4677b385c45f47aad25006a7ecbaf041a41838f26962b708a557ec"),
    "aarch64-linux/lib64/libopapi_nn.so": (
        "d8cde8eabd9f4ceffba6032eea6c514b3e25413af0462efe4a3823e56aa905be"),
    "aarch64-linux/lib64/libregister.so": (
        "1a4df97416ed6983f4a33be4d3cb10802ce6bd715d5bed229eb9066841369360"),
    "opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/aarch64/"
    "libophost_nn.so": (
        "cb2f73db214b817780aa8f1cb075a91026ef8efe495996965e72bc3967e1451b"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/swi_glu/"
    "swi_glu.cpp": (
        "584fcf498038a6b31adf480c4fb4a600dd4cf59657a1c0a0bf725c3df3717cf3"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/swi_glu/"
    "swi_glu_bf16.hpp": (
        "d18f872867f22c9f7032a44c22a09ceb84d78bdb9b4a94288062b6d1e498e933"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/swi_glu/"
    "swi_glu_single.hpp": (
        "ace487f5c4f3929e0cc1e02dfc54b062294a3018d0cb102baa920ea83e8dd096"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/swi_glu/"
    "glu_tiling_kernel.hpp": (
        "2b6abc813b6e372b0d36e81f79fd618c180b0c0e4de2643056fd7aad775c2053"),
    "opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/ops_nn/swi_glu/"
    "SwiGlu_3_high_performance.o": (
        "37c8b298668e992dff17c758f55de13e7d94f94361e73dae492b122979d10ab3"),
    "opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910b/ops_nn/"
    "swi_glu.json": (
        "922668421363c521a8fc97def191e9618d8b36073bf1d3ae10127876e182d9b1"),
}

SWIGLU_EXACT_CAPTURE_SCHEMA_VERSION = 2
SWIGLU_EXACT_FACT_SCHEMA_VERSION = 4
SWIGLU_EXACT_CAPTURE_ARTIFACT_TYPE = (
    "ascend_910b_swiglu_bulk_tiling_observation")
SWIGLU_EXACT_FACT_ARTIFACT_TYPE = "ascend_910b_swiglu_launch_facts"
SWIGLU_EXACT_RUNTIME_SYMBOL = "SwiGlu_3_high_performance_27"
SWIGLU_EXACT_INTERMEDIATES = (
    6144, 9728, 12288, 12800, 14336, 25600, 28672, 53248,
)
SWIGLU_EXACT_M_MIN = 1
SWIGLU_EXACT_M_MAX = 8192
SWIGLU_EXACT_WORKLOAD_COUNT = 65536
SWIGLU_EXACT_SHARD_FIELDS = {
    "application_batch_size", "application_process_count", "artifact_type",
    "baseline_commit", "cann_provenance", "capture_state",
    "compile_command", "compile_count", "compiler", "duration_consumed",
    "environment", "full_workload_count", "interposer",
    "launch_import_audit", "post_telemetry", "postflight_identity",
    "pre_telemetry", "python", "reflection", "result_count", "results",
    "schema_version", "selected_range", "selected_workload_count",
    "source_provenance", "target_kernel_submitted", "workload_frontier",
}
SWIGLU_EXACT_RESULT_FIELDS = {
    "application_command", "application_log", "contract", "launch_result",
    "limitations", "records", "report", "semantic_key", "workload_key",
}
SWIGLU_EXACT_CONTRACT_FIELDS = {
    "batch", "block_dim", "capture_input", "case_key", "dim", "dtype",
    "dtype_bytes", "element_count", "head_dim", "hidden", "intermediate",
    "kernel_name", "layer", "layout", "num_heads", "num_kv_heads", "tok",
}
SWIGLU_EXACT_LAUNCH_RESULT_FIELDS = {
    "block_dim", "decoded_tiling", "kernel_identity", "kernel_symbol",
    "serialized_tiling_sha256", "source_derived_tile_schedule",
    "target_kernel_submitted",
}
SWIGLU_EXACT_REPORT_FIELDS = {
    "artifact_type", "block_dim", "case_contract", "decoded_tiling",
    "duration_consumed", "execute_api", "execution_count", "kernel_identity",
    "kernel_symbol", "launch_api", "limitations", "placeholder_maps",
    "reflection", "schema_version", "source_derived_tile_schedule",
    "stream_identity_equal", "target_kernel_submitted",
    "tiling_payload_selection", "tiling_used_core_num",
    "tiling_used_core_num_matches_launch", "workspace_size_bytes",
}
SWIGLU_EXACT_SOURCE_FILES = (
    "microbench/ascend_910b/operators/swiglu/tiling_capture_interposer.cpp",
    "microbench/ascend_910b/operators/swiglu/tiling_capture_runner.py",
    "microbench/ascend_910b/operators/swiglu/tiling_capture_report.py",
    "microbench/ascend_910b/operators/swiglu/tiling_capture_application_runner.py",
    "dataset_pipeline/launch_data/gemm_workload_candidates.py",
    "src/kernel/launch_config/gemm_workload.py",
    "src/kernel/launch_config/selection_primitives.py",
)
SWIGLU_EXACT_CANN_FILES = (
    "opp/version.info",
    "aarch64-linux/include/acl/acl_rt.h",
    "aarch64-linux/include/aclnn/acl_meta.h",
    "aarch64-linux/include/aclnnop/aclnn_swi_glu.h",
    "aarch64-linux/lib64/libacl_rt.so",
    "aarch64-linux/lib64/libnnopbase.so",
    "aarch64-linux/lib64/libopapi_nn.so",
    "aarch64-linux/lib64/libregister.so",
    "opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/aarch64/libophost_nn.so",
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/swi_glu/swi_glu.cpp",
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/swi_glu/swi_glu_bf16.hpp",
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/swi_glu/swi_glu_single.hpp",
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/swi_glu/glu_tiling_kernel.hpp",
    "opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/ops_nn/swi_glu/SwiGlu_3_high_performance.o",
    "opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910b/ops_nn/swi_glu.json",
)
SWIGLU_EXACT_LAUNCH_IMPORT_FILES = (
    "aarch64-linux/lib64/libnnopbase.so",
    "aarch64-linux/lib64/libopapi_nn.so",
    "aarch64-linux/lib64/libregister.so",
    "opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/aarch64/libophost_nn.so",
)
SWIGLU_EXACT_INTERCEPTED_LAUNCH_APIS = (
    "aclrtLaunchKernel", "aclrtLaunchKernelV2",
    "aclrtLaunchKernelWithConfig", "aclrtLaunchKernelWithHostArgs",
)
SWIGLU_EXACT_REFLECTION_FIELDS = (
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
SWIGLU_EXACT_WORKSPACE_RECORD_FIELDS = {
    "api", "execute_stream", "metadata_path", "metadata_sha256", "pid",
    "record_type", "sequence", "tid", "workspace_size_bytes",
}
SWIGLU_EXACT_LAUNCH_RECORD_FIELDS = {
    "api", "args_path", "args_sha256", "args_size_bytes", "block_dim",
    "execute_api", "execute_stream", "launch_stream", "metadata_path",
    "metadata_sha256", "pid", "placeholder_count", "placeholder_path",
    "placeholder_sha256", "placeholders", "record_type", "sequence",
    "stub_identity", "tid",
}

GEMM_CAPTURE_SCHEMA_VERSION = 3
GEMM_FACT_SCHEMA_VERSION = 4
GEMM_CAPTURE_ARTIFACT_TYPE = "ascend_910b_gemm_dry_launch_bulk"
GEMM_CANDIDATE_SCOPE = "static_complete_legal_launch_selector_domain"
GEMM_COVERAGE_CLAIM = "complete_exact_legal_domain"
GEMM_FAMILY_CONTRACTS = {
    "matmul_v2": {
        "kernel_pattern": MATMUL_V2_KERNEL_PATTERN,
        "host_args_size_bytes": 200,
        "placeholder": {"address_offset": 24, "data_offset": 40},
        "schem_mode": 0,
        "tiling_payload_size_bytes": MATMUL_V2_TILING_PAYLOAD_SIZE_BYTES,
    },
    "matmul_v3": {
        "kernel_pattern": MATMUL_V3_KERNEL_PATTERN,
        "host_args_size_bytes": 360,
        "placeholder": {"address_offset": 56, "data_offset": 72},
        "schem_mode": 1,
        "tiling_payload_size_bytes": MATMUL_V3_TILING_PAYLOAD_SIZE_BYTES,
    },
}
GEMM_CANDIDATE_FIELDS = {
    "artifact_type", "coverage_scope", "cross_hardware",
    "dataset_model_inventory", "duration_consumed", "hardware_domains",
    "limitations", "model_domains", "physical_operation_count",
    "physical_operation_fields", "scheduler_contract", "schema_version",
    "semantic_key_count", "source_config", "source_config_sha256",
    "workload_key_fields", "workload_keys",
}
GEMM_SHARD_FIELDS = {
    "artifact_type", "baseline_commit", "build_and_abi_audit",
    "cann_abi_provenance", "cann_version", "cube_math_type", "device_id",
    "device_kernel_submitted", "duration_consumed", "execute_api_called",
    "full_input_operation_count", "full_input_workload_key_count",
    "hardware_id", "limitations", "mode", "operation_count", "operations",
    "operator", "runner_version", "schema_version",
    "selected_operation_count", "selected_operation_range",
    "source_provenance", "workload_candidate", "workload_key_count",
    "workload_key_fields", "workload_mappings",
}
GEMM_OPERATION_FIELDS = {
    "cube_math_type", "dtype", "k", "m", "n", "operation_id", "result",
    "workload_keys",
}
GEMM_RESULT_FIELDS = {
    "cube_math_type", "device_kernel_submitted", "dry_launch", "dtype",
    "duration_consumed", "execute_api_called", "executor_lifecycle",
    "executor_tiling", "layout", "operation_id", "operator",
    "process_peak_rss_kib", "schema_version", "shape",
    "workload_keys", "workspace_size_bytes",
}
GEMM_LAUNCH_APIS = [
    "aclrtLaunchKernel", "aclrtLaunchKernelV2", "aclrtLaunchKernelWithConfig",
    "aclrtLaunchKernelWithHostArgs",
]
GEMM_SOURCE_FILES = {
    "dataset_pipeline/launch_data/gemm_workload_candidates.py",
    "microbench/ascend_910b/operators/gemm/launch_guard_interposer.cpp",
    "microbench/ascend_910b/operators/gemm/tiling_query_helper.cpp",
    "microbench/ascend_910b/operators/gemm/tiling_query_runner.py",
    "src/kernel/launch_config/gemm_workload.py",
}
GEMM_CAPTURE_LIMITATIONS = [
    "the complete host-argument, placeholder, and tiling payload buffers are "
    "retained; the capture runner identifies the pinned CANN 8.5.1 MatMulV2 "
    "or MatMulV3 family but does not semantically decode MatMulV3 tiling",
    "the result is specific to CANN 8.5.1, whose family metadata, source "
    "artifacts, and libnnopbase import table are audited before dry launches",
    "the MatMulV2 final eight payload bytes and the complete MatMulV3 payload "
    "are retained verbatim without unsupported semantic interpretation",
    "the helper does not call aclSetAclOpExecutorRepeatable; under CANN 8.5.1 "
    "acl_meta.h and opdev/op_executor.h semantics, the successful "
    "non-repeatable aclnnMatmul CommonOpExecutorRun path consumes the executor, "
    "so the helper clears its borrowed pointer and does not destroy it again",
]


def _load(name):
    p = DATASET_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"required kernel dataset is missing: {p}")
    with open(p) as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"kernel dataset must be a non-empty object: {p}")
    return data


def _reference_named_values(value, fields, context):
    if not isinstance(value, list) or len(value) != len(fields):
        raise ValueError(
            f"{context} must contain exactly {len(fields)} ordered fields")
    return dict(zip(fields, value))


def _reference_int_list(value, context, *, length, positive=False):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{context} must contain exactly {length} integers")
    return [
        require_integer(item, f"{context}[{index}]", positive=positive)
        for index, item in enumerate(value)
    ]


def _flash_attn_source_schedules(dataset):
    """Load exact CANN payload fields without extrapolating missing cases."""

    reference = json.loads(FLASH_ATTN_SOURCE_SCHEDULE_REFERENCE.read_text())
    if (not isinstance(reference, dict)
            or set(reference) != FLASH_ATTN_SOURCE_SCHEDULE_REFERENCE_FIELDS):
        raise ValueError(
            "910B FlashAttention source-schedule reference schema changed")
    if reference["schema_version"] != 1:
        raise ValueError(
            "910B FlashAttention source-schedule reference version changed")
    if reference["hardware_id"] != "ascend_910b":
        raise ValueError(
            "910B FlashAttention source-schedule hardware changed")
    if reference["software_revision"] != "cann_8.5.1":
        raise ValueError(
            "910B FlashAttention source-schedule software changed")
    if reference["duration_consumed"] is not False:
        raise ValueError(
            "910B FlashAttention source schedule consumed duration")
    if reference["canonical_dataset_sha256"] != file_sha256(
            FLASH_ATTN_DATASET):
        raise ValueError(
            "910B FlashAttention source schedule targets a stale dataset")
    if reference["capture_source_archives"] != (
            FLASH_ATTN_SOURCE_SCHEDULE_CAPTURE_ARCHIVES):
        raise ValueError(
            "910B FlashAttention capture source provenance changed")
    if reference["source_evidence"] != FLASH_ATTN_SOURCE_SCHEDULE_EVIDENCE:
        raise ValueError(
            "910B FlashAttention CANN source evidence changed")
    expected_field_lists = {
        "entry_fields": FLASH_ATTN_SOURCE_SCHEDULE_ENTRY_FIELDS,
        "base_fields": FLASH_ATTN_SOURCE_SCHEDULE_BASE_FIELDS,
        "inner_split_fields": FLASH_ATTN_SOURCE_SCHEDULE_INNER_SPLIT_FIELDS,
        "workspace_element_fields": (
            FLASH_ATTN_SOURCE_SCHEDULE_WORKSPACE_FIELDS),
        "worker_exclusive_end_fields": (
            FLASH_ATTN_SOURCE_SCHEDULE_WORKER_END_FIELDS),
        "flash_decode_fields": (
            FLASH_ATTN_SOURCE_SCHEDULE_FLASH_DECODE_FIELDS),
    }
    for name, expected in expected_field_lists.items():
        if reference[name] != expected:
            raise ValueError(
                f"910B FlashAttention source-schedule {name} changed")

    entries = reference["entries"]
    if not isinstance(entries, dict) or len(entries) != 200:
        raise ValueError(
            "910B FlashAttention source schedule must contain 200 exact "
            "source-reflected captures")
    schedules = {}
    for case_id, packed in sorted(entries.items()):
        if not isinstance(case_id, str) or not case_id:
            raise TypeError(
                "910B FlashAttention captured case id must be a string")
        if case_id not in dataset:
            raise KeyError(
                f"910B FlashAttention captured case is not canonical: "
                f"{case_id!r}")
        entry = _reference_named_values(
            packed, FLASH_ATTN_SOURCE_SCHEDULE_ENTRY_FIELDS,
            f"source_schedule.{case_id}")
        baseline = entry["capture_baseline_commit"]
        if baseline not in FLASH_ATTN_SOURCE_SCHEDULE_CAPTURE_ARCHIVES:
            raise ValueError(
                f"source_schedule.{case_id} has an unknown capture baseline")
        directory = entry["capture_directory"]
        if (not isinstance(directory, str)
                or not directory.startswith("fa_tiling_")):
            raise ValueError(
                f"source_schedule.{case_id} has an invalid capture directory")
        for name in (
                "manifest_sha256", "tiling_data_sha256",
                "serialized_tiling_sha256"):
            require_sha256_hex(
                entry[name], f"source_schedule.{case_id}.{name}")
        if entry["serialized_tiling_size_bytes"] != 1560:
            raise ValueError(
                f"source_schedule.{case_id} reflected payload size changed")
        runtime_variant = entry["runtime_variant"]
        if runtime_variant not in FLASH_ATTN_SOURCE_SCHEDULE_VARIANTS:
            raise ValueError(
                f"source_schedule.{case_id} has an unsupported variant")

        base = _reference_named_values(
            entry["base"], FLASH_ATTN_SOURCE_SCHEDULE_BASE_FIELDS,
            f"source_schedule.{case_id}.base")
        for name in (
                "batch", "kv_heads", "group_size", "q_len", "kv_len",
                "head_dim", "used_core_num"):
            base[name] = require_integer(
                base[name], f"source_schedule.{case_id}.base.{name}",
                positive=True)
        for name in ("output_layout", "softmax_lse_flag"):
            base[name] = require_integer(
                base[name], f"source_schedule.{case_id}.base.{name}")
        if base["softmax_lse_flag"] not in (0, 1):
            raise ValueError(
                f"source_schedule.{case_id}.base.softmax_lse_flag must be "
                "zero or one")

        record = dataset[case_id]
        attributes = record["op_info"]["Attr"]
        kernel = record["kernels"][0]
        metadata = kernel["Metadata"]
        ascend = kernel["Ascend"]
        expected_base = {
            "batch": attributes["B"],
            "kv_heads": attributes["HKV"],
            "group_size": attributes["HQ"] // attributes["HKV"],
            # The payload stores the allocated Q/KV tensor extents.  Varlen
            # cases carry smaller actual per-batch lengths separately in the
            # canonical Q_LENS/KV_LENS workload contract.
            "q_len": attributes["Q_REQUESTED"],
            "kv_len": attributes["KV_REQUESTED"],
            "head_dim": attributes["D"],
            "used_core_num": ascend["block_dim"],
        }
        for name, expected in expected_base.items():
            if base[name] != expected:
                raise ValueError(
                    f"source_schedule.{case_id}.base.{name} disagrees with "
                    "the canonical case")
        if base["group_size"] * base["kv_heads"] != attributes["HQ"]:
            raise ValueError(
                f"source_schedule.{case_id} has a fractional GQA group")
        symbol = metadata["Kernel Name"]
        if runtime_variant not in symbol:
            raise ValueError(
                f"source_schedule.{case_id} runtime variant disagrees with "
                "the canonical symbol")

        inner_split = _reference_named_values(
            entry["inner_split"],
            FLASH_ATTN_SOURCE_SCHEDULE_INNER_SPLIT_FIELDS,
            f"source_schedule.{case_id}.inner_split")
        for name in FLASH_ATTN_SOURCE_SCHEDULE_INNER_SPLIT_FIELDS:
            inner_split[name] = require_integer(
                inner_split[name],
                f"source_schedule.{case_id}.inner_split.{name}",
                positive=True)
        workspace = _reference_named_values(
            entry["workspace_elements"],
            FLASH_ATTN_SOURCE_SCHEDULE_WORKSPACE_FIELDS,
            f"source_schedule.{case_id}.workspace_elements")
        for name in FLASH_ATTN_SOURCE_SCHEDULE_WORKSPACE_FIELDS:
            workspace[name] = require_integer(
                workspace[name],
                f"source_schedule.{case_id}.workspace_elements.{name}",
                positive=True)

        worker_ends = _reference_named_values(
            entry["worker_exclusive_ends"],
            FLASH_ATTN_SOURCE_SCHEDULE_WORKER_END_FIELDS,
            f"source_schedule.{case_id}.worker_exclusive_ends")
        used_core_num = base["used_core_num"]
        for name in FLASH_ATTN_SOURCE_SCHEDULE_WORKER_END_FIELDS:
            worker_ends[name] = _reference_int_list(
                worker_ends[name],
                f"source_schedule.{case_id}.worker_exclusive_ends.{name}",
                length=used_core_num)
        coordinates = list(zip(
            worker_ends["b_n2"], worker_ends["g_s1"], worker_ends["s2"]))
        previous = (0, 0, 0)
        for coordinate in coordinates:
            if coordinate <= previous:
                raise ValueError(
                    f"source_schedule.{case_id} worker endpoints are not "
                    "strictly ordered")
            previous = coordinate
        if coordinates[-1] != (
                base["batch"] * base["kv_heads"], 0, 0):
            raise ValueError(
                f"source_schedule.{case_id} does not cover all BN2 heads")

        packed_flash_decode = entry["flash_decode"]
        flash_decode = None
        if packed_flash_decode is not None:
            flash_decode = _reference_named_values(
                packed_flash_decode,
                FLASH_ATTN_SOURCE_SCHEDULE_FLASH_DECODE_FIELDS,
                f"source_schedule.{case_id}.flash_decode")
            for name in (
                    "used_vector_core_num", "head_task_count",
                    "g_s1_base_size"):
                flash_decode[name] = require_integer(
                    flash_decode[name],
                    f"source_schedule.{case_id}.flash_decode.{name}",
                    positive=True)
            fd_head_count = flash_decode["head_task_count"]
            for name in (
                    "head_b_n2", "head_g_s1", "head_s2_split_count",
                    "head_g_s1_split_count", "head_g_s1_last_size"):
                flash_decode[name] = _reference_int_list(
                    flash_decode[name],
                    f"source_schedule.{case_id}.flash_decode.{name}",
                    length=fd_head_count,
                    positive=name in (
                        "head_s2_split_count", "head_g_s1_split_count",
                        "head_g_s1_last_size"))
            flash_decode["worker_s2_split_start"] = _reference_int_list(
                flash_decode["worker_s2_split_start"],
                f"source_schedule.{case_id}.flash_decode."
                "worker_s2_split_start",
                length=used_core_num)
            used_vector_core_num = flash_decode["used_vector_core_num"]
            if used_vector_core_num > 2 * used_core_num:
                raise ValueError(
                    f"source_schedule.{case_id} FlashDecode exceeds AIV "
                    "launch capacity")
            for name in (
                    "vector_worker_head_end",
                    "vector_worker_head_split_end"):
                flash_decode[name] = _reference_int_list(
                    flash_decode[name],
                    f"source_schedule.{case_id}.flash_decode.{name}",
                    length=used_vector_core_num)
        if ((runtime_variant in FLASH_ATTN_FLASH_DECODE_VARIANTS)
                != (flash_decode is not None)):
            raise ValueError(
                f"source_schedule.{case_id} FlashDecode payload disagrees "
                "with its runtime variant")

        schedules[case_id] = {
            "capture_baseline_commit": baseline,
            "capture_directory": directory,
            "manifest_sha256": entry["manifest_sha256"],
            "tiling_data_sha256": entry["tiling_data_sha256"],
            "serialized_tiling_sha256": entry[
                "serialized_tiling_sha256"],
            "serialized_tiling_size_bytes": entry[
                "serialized_tiling_size_bytes"],
            "base": base,
            "inner_split": inner_split,
            "workspace_elements": workspace,
            "worker_exclusive_ends": worker_ends,
            "flash_decode": flash_decode,
        }
    return schedules


def _flash_attn_source_program(schedule, attributes):
    """Expand one reflected FIA cursor into its exact ordered task program."""

    if schedule is None or schedule["flash_decode"] is not None:
        return None
    base = schedule["base"]
    inner_split = schedule["inner_split"]
    batch = base["batch"]
    kv_heads = base["kv_heads"]
    group_size = base["group_size"]
    m_base_size = inner_split["m_base_size"]
    s2_base_size = inner_split["s2_base_size"]
    q_lens = attributes["Q_LENS"]
    kv_lens = attributes["KV_LENS"]
    if (not isinstance(q_lens, list) or len(q_lens) != batch
            or not isinstance(kv_lens, list) or len(kv_lens) != batch):
        raise ValueError(
            "FlashAttention source program requires exact per-batch lengths")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in q_lens + kv_lens):
        raise TypeError(
            "FlashAttention source program lengths must be positive integers")
    worker_ends = schedule["worker_exclusive_ends"]
    endpoints = tuple(zip(
        worker_ends["b_n2"], worker_ends["g_s1"], worker_ends["s2"]))

    def advance(coordinate):
        b_n2, g_s1, s2 = coordinate
        if b_n2 >= batch * kv_heads:
            raise ValueError(
                "FlashAttention source program advanced past the BN2 domain")
        batch_index = b_n2 // kv_heads
        g_s1_count = math.ceil(
            q_lens[batch_index] * group_size / m_base_size)
        s2_count = math.ceil(kv_lens[batch_index] / s2_base_size)
        if g_s1 >= g_s1_count or s2 >= s2_count:
            raise ValueError(
                "FlashAttention source program cursor is outside lengths")
        if s2 + 1 < s2_count:
            return b_n2, g_s1, s2 + 1
        if g_s1 + 1 < g_s1_count:
            return b_n2, g_s1 + 1, 0
        return b_n2 + 1, 0, 0

    groups = []
    cursor = (0, 0, 0)
    for aic_group, endpoint in enumerate(endpoints):
        tasks = []
        while cursor != endpoint:
            if cursor > endpoint:
                raise ValueError(
                    "FlashAttention source endpoint bisects an invalid cursor")
            b_n2, g_s1, s2 = cursor
            batch_index = b_n2 // kv_heads
            actual_m = min(
                m_base_size,
                q_lens[batch_index] * group_size - g_s1 * m_base_size)
            actual_s2 = min(
                s2_base_size, kv_lens[batch_index] - s2 * s2_base_size)
            if actual_m <= 0 or actual_s2 <= 0:
                raise ValueError(
                    "FlashAttention source program task has no actual work")
            next_cursor = advance(cursor)
            same_head_as_previous = bool(
                tasks
                and tasks[-1]["b_n2"] == b_n2
                and tasks[-1]["g_s1"] == g_s1)
            same_head_as_next = (
                next_cursor != endpoint
                and next_cursor[0:2] == (b_n2, g_s1))
            tasks.append({
                "b_n2": b_n2,
                "g_s1": g_s1,
                "s2": s2,
                "actual_m": actual_m,
                "actual_s2": actual_s2,
                "is_head_first": not same_head_as_previous,
                "is_head_last": not same_head_as_next,
            })
            cursor = next_cursor
        if not tasks:
            raise ValueError(
                f"FlashAttention AIC group {aic_group} has no source tasks")
        groups.append({
            "aic_group": aic_group,
            "aiv_workers": [2 * aic_group, 2 * aic_group + 1],
            "tasks": tasks,
        })
    if cursor != (batch * kv_heads, 0, 0):
        raise ValueError(
            "FlashAttention source program does not cover the BN2 domain")
    return {
        "program_kind": "cann_fia_1_2_three_handoff_v1",
        "aic_groups": groups,
    }


def build_flash_attn_launch_facts(dataset):
    """Extract exact non-duration launch identity for every canonical case."""

    canonical_dataset = json.loads(FLASH_ATTN_DATASET.read_text())
    if dataset != canonical_dataset:
        raise ValueError(
            "FlashAttention launch facts require the exact canonical dataset")
    source_schedules = _flash_attn_source_schedules(dataset)
    entries = {}
    for case_id, record in sorted(dataset.items()):
        if not isinstance(case_id, str) or not case_id:
            raise TypeError("FlashAttention case id must be a non-empty string")
        if not isinstance(record, dict):
            raise TypeError(f"FlashAttention case {case_id!r} must be a mapping")
        attributes = record["op_info"]["Attr"]
        if not isinstance(attributes, dict) or set(attributes) != FLASH_ATTN_ATTRIBUTE_KEYS:
            raise ValueError(
                f"FlashAttention case {case_id!r} attributes changed")
        kernels = record["kernels"]
        if not isinstance(kernels, list) or len(kernels) != 1:
            raise ValueError(
                f"FlashAttention case {case_id!r} requires one fused kernel")
        kernel = kernels[0]
        metadata = kernel["Metadata"]
        ascend = kernel["Ascend"]
        runtime_symbol = require_non_empty_string(
            metadata, "Kernel Name", f"{case_id}.Metadata")
        if (not runtime_symbol.startswith(FLASH_ATTN_RUNTIME_PREFIX)
                or not runtime_symbol.endswith(FLASH_ATTN_RUNTIME_SUFFIX)):
            raise ValueError(
                f"FlashAttention case {case_id!r} has an unknown runtime symbol")
        runtime_variant = runtime_symbol[
            len(FLASH_ATTN_RUNTIME_PREFIX):-len(FLASH_ATTN_RUNTIME_SUFFIX)]
        if re.fullmatch(r"[0-9]+", runtime_variant) is None:
            raise ValueError(
                f"FlashAttention case {case_id!r} has invalid runtime variant")
        if runtime_variant not in FLASH_ATTN_METADATA_TILING_KEYS:
            raise ValueError(
                f"FlashAttention case {case_id!r} runtime variant has no "
                "exact static metadata tilingKey evidence")
        if (require_non_empty_string(metadata, "Op Type", f"{case_id}.Metadata") != "mix"
                or require_non_empty_string(
                    ascend, "op_type", f"{case_id}.Ascend") != "mix"):
            raise ValueError(
                f"FlashAttention case {case_id!r} must be a mixed task")
        block_dim = require_positive_int(
            ascend, "block_dim", f"{case_id}.Ascend")
        mix_block_dim = require_positive_int(
            ascend, "mix_block_dim", f"{case_id}.Ascend")
        if mix_block_dim != 2 * block_dim:
            raise ValueError(
                f"FlashAttention case {case_id!r} must use AIC:AIV=1:2")
        grid = parse_dim3_csv(metadata["Grid Size"], f"{case_id}.Metadata.Grid Size")
        block = parse_dim3_csv(metadata["Block Size"], f"{case_id}.Metadata.Block Size")
        if grid != [block_dim, 1, 1] or block != [1, 1, 1]:
            raise ValueError(
                f"FlashAttention case {case_id!r} launch dimensions disagree")

        shape = {
            "batch": require_positive_int(attributes, "B", f"{case_id}.Attr"),
            "q_len": require_positive_int(attributes, "Q_LEN", f"{case_id}.Attr"),
            "kv_len": require_positive_int(attributes, "KV_LEN", f"{case_id}.Attr"),
            "head_dim": require_positive_int(attributes, "D", f"{case_id}.Attr"),
            "num_query_heads": require_positive_int(
                attributes, "HQ", f"{case_id}.Attr"),
            "num_kv_heads": require_positive_int(
                attributes, "HKV", f"{case_id}.Attr"),
        }
        execution_variant = "decode" if shape["q_len"] == 1 else "prefill"
        selection_attributes = {
            "is_decode": execution_variant == "decode",
            "causal": attributes["causal"],
            "kv_layout": attributes["kv_layout"],
            "kv_dist": attributes["kv_dist"],
            "q_lens": attributes["Q_LENS"],
            "kv_lens": attributes["KV_LENS"],
        }
        key = flash_attn_workload_key(
            shape, "bf16", execution_variant, selection_attributes)
        if key in entries:
            raise ValueError(
                f"duplicate FlashAttention workload launch key: {key}")
        if case_id in source_schedules:
            source_schedule = source_schedules[case_id]
        else:
            # The reference schema defines schedule coverage as optional at
            # this input boundary.  Normalize absence to an explicit null in
            # every generated entry; downstream consumers never probe keys.
            source_schedule = None
        source_program = _flash_attn_source_program(
            source_schedule, attributes)
        entries[key] = {
            "source_case_id": case_id,
            "runtime_symbol": runtime_symbol,
            "runtime_variant": runtime_variant,
            # Exact Python integer parsing of the hash-pinned CANN metadata
            # proves equality.  Keep the identifier as a decimal string so a
            # JSON consumer cannot round it through IEEE-754.
            "metadata_tiling_key": runtime_variant,
            "task_kind": "mixed",
            "block_dim": block_dim,
            "mix_block_dim": mix_block_dim,
            "grid": grid,
            "block": block,
            # Only directly captured, reflected CANN payloads publish a
            # schedule.  The other canonical cases retain an explicit null;
            # no shape interpolation or variant-wide extrapolation is legal.
            "source_schedule": source_schedule,
            # The ordered program contains only source cursor facts.  It has
            # no operations, traffic, duration, or empirical correction.
            "source_program": source_program,
        }
    program_entries = tuple(
        entry for entry in entries.values()
        if entry["source_program"] is not None)
    flash_decode_entries = tuple(
        entry for entry in entries.values()
        if (entry["source_schedule"] is not None
            and entry["source_schedule"]["flash_decode"] is not None))
    if len(program_entries) != 177 or len(flash_decode_entries) != 23:
        raise ValueError(
            "FlashAttention source-program coverage changed: "
            f"programs={len(program_entries)}, "
            f"flash_decode={len(flash_decode_entries)}")
    return {
        "generated_by": "scripts/launch_data/ascend_910b_launch_generator.py",
        "hardware_id": "ascend_910b",
        "schema_version": 3,
        "software_revision": "cann_8.5.1",
        "source": "kernel_dataset/ascend_910b/flash_attn.json.zst",
        "source_sha256": file_sha256(FLASH_ATTN_DATASET),
        "workload_contract": "configs/dataset/kernel/flash_attn.yaml",
        "workload_contract_sha256": file_sha256(FLASH_ATTN_WORKLOAD_CONTRACT),
        "source_schedule_reference": (
            "microbench/ascend_910b/references/"
            "fused_infer_attention_score_cann_8_5_1.json"),
        "source_schedule_reference_sha256": file_sha256(
            FLASH_ATTN_SOURCE_SCHEDULE_REFERENCE),
        "duration_consumed": False,
        "static_evidence": {
            "cann_kernel_metadata_sha256": (
                "e57163518dc1619f52737d44f35e03ac2c1023c133ea6af9ad819edf0ec8f64d"),
            "tiling_key_header_sha256": (
                "5bbe5bab7caa58f65d43a261c83ee9e26f282634299397b44d2d56216e5d4e3e"),
            "v3_dispatch_sha256": (
                "9315d547e7468e28c5e38656ec01b8c9ef876bf53715cd166a545c7232305961"),
            "split_fuse_dispatch_sha256": (
                "6ae6d2f747fa2fc30ad322ca8701ad193ef0ab3f6d64a46c41fb8f05c0ea17a3"),
            "fia_kernel_nonquant_sha256": (
                "7cde05730d68bb02a2c45cd6b25605417ee61d42dfd87f21ccf0948c014b7e21"),
            "fia_public_define_sha256": (
                "1339e51f2bb1b2e6002e9a1b1b342523b4bb4007c86242d83b681d605ba52fb4"),
            "fia_block_vec_nonquant_sha256": (
                "259f5700812eafe32bcffd075f59d90c82cb71458f62bd47cb3d6f17f171e762"),
            "fia_block_cube_nonquant_gqa_sha256": (
                "acce194dcf01f370d864ba7987f0c5a9d5927f7a69fb47426afa7789a6f856ef"),
            "fia_block_vec_flashdecode_sha256": (
                "04ebb6eb9dd85da8cf75d57d466be0c6ec8980b4cf366294dca5cb7495faef45"),
        },
        "workload_key_fields": [
            "dtype", "execution_variant", "batch", "q_len", "kv_len",
            "head_dim", "num_query_heads", "num_kv_heads", "causal",
            "kv_layout", "kv_dist", "q_lens", "kv_lens",
        ],
        "entries": entries,
    }


def _exact_mapping(value, expected_keys, context):
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a mapping")
    if set(value) != expected_keys:
        raise ValueError(
            f"{context} schema changed: actual={sorted(value)}")
    return value


def _reject_swiglu_measurement_fields(value, context):
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{context} field names must be strings")
            if re.search(
                    r"(?:^|_)(?:duration|durations|cycle|cycles|timing)(?:_|$)",
                    key.lower()) is not None:
                if key != "duration_consumed" or item is not False:
                    raise ValueError(
                        f"{context} contains forbidden measurement field {key!r}")
            _reject_swiglu_measurement_fields(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_swiglu_measurement_fields(item, f"{context}[{index}]")


def _swiglu_exact_keys():
    return [
        (tok, intermediate)
        for tok in range(SWIGLU_EXACT_M_MIN, SWIGLU_EXACT_M_MAX + 1)
        for intermediate in SWIGLU_EXACT_INTERMEDIATES
    ]


def _swiglu_exact_provenance(shard):
    baseline = shard["baseline_commit"]
    if (not isinstance(baseline, str)
            or re.fullmatch(r"[0-9a-f]{40}", baseline) is None):
        raise ValueError("Ascend SwiGLU capture baseline is not a commit")
    source = _exact_mapping(
        shard["source_provenance"],
        {"archive_path", "archive_sha256", "git_archive_commit",
         "verified_sources"},
        "Ascend SwiGLU source provenance")
    require_sha256_hex(
        source["archive_sha256"], "Ascend SwiGLU source archive SHA256")
    if (not isinstance(source["archive_path"], str)
            or not Path(source["archive_path"]).is_absolute()
            or source["git_archive_commit"] != baseline):
        raise ValueError("Ascend SwiGLU source archive baseline changed")
    verified_sources = source["verified_sources"]
    if not isinstance(verified_sources, list):
        raise TypeError("Ascend SwiGLU verified sources must be a list")
    source_hashes = {}
    for item in verified_sources:
        identity = _exact_mapping(
            item, {"path", "relative_path", "sha256"},
            "Ascend SwiGLU verified source")
        relative = identity["relative_path"]
        if relative in source_hashes:
            raise ValueError(f"duplicate Ascend SwiGLU source: {relative!r}")
        require_sha256_hex(
            identity["sha256"], f"Ascend SwiGLU source {relative} SHA256")
        if (not isinstance(identity["path"], str)
                or not identity["path"].endswith(f"/{relative}")):
            raise ValueError("Ascend SwiGLU source path identity changed")
        source_hashes[relative] = identity["sha256"]
    if tuple(source_hashes) != SWIGLU_EXACT_SOURCE_FILES:
        raise ValueError("Ascend SwiGLU verified source-file set changed")

    cann = _exact_mapping(
        shard["cann_provenance"], {"cann_root", "files"},
        "Ascend SwiGLU CANN provenance")
    if cann["cann_root"] != "/usr/local/Ascend/cann-8.5.1":
        raise ValueError("Ascend SwiGLU CANN root changed")
    if not isinstance(cann["files"], list):
        raise TypeError("Ascend SwiGLU CANN files must be a list")
    cann_hashes = {}
    for item in cann["files"]:
        identity = _exact_mapping(
            item, {"path", "relative_path", "sha256"},
            "Ascend SwiGLU CANN file")
        relative = identity["relative_path"]
        if relative in cann_hashes:
            raise ValueError(f"duplicate Ascend SwiGLU CANN file: {relative!r}")
        require_sha256_hex(
            identity["sha256"], f"Ascend SwiGLU CANN {relative} SHA256")
        if identity["path"] != f"{cann['cann_root']}/{relative}":
            raise ValueError("Ascend SwiGLU CANN file path changed")
        cann_hashes[relative] = identity["sha256"]
    expected_cann_hashes = dict(SWIGLU_CANN_EVIDENCE)
    expected_cann_hashes["aarch64-linux/lib64/libnnopbase.so"] = (
        "6dc3b4fcfcf57b10be0018e89f4edbc079f02b4bb0aaadb109df350939fd4c99")
    if (tuple(cann_hashes) != SWIGLU_EXACT_CANN_FILES
            or cann_hashes != expected_cann_hashes):
        raise ValueError("Ascend SwiGLU CANN evidence file set changed")

    reflection = _exact_mapping(
        shard["reflection"],
        {"class_name", "data_size_bytes", "fields", "path",
         "registration_key", "sha256"},
        "Ascend SwiGLU reflection")
    require_sha256_hex(
        reflection["sha256"], "Ascend SwiGLU reflection SHA256")
    expected_fields = [
        {"name": name, "dtype": dtype, "offset_bytes": offset,
         "count": count, "size_bytes": size}
        for name, dtype, offset, count, size
        in SWIGLU_EXACT_REFLECTION_FIELDS
    ]
    if (not isinstance(reflection["path"], str)
            or not Path(reflection["path"]).is_absolute()
            or reflection["registration_key"] != "SwiGlu"
            or reflection["class_name"] != "SwiGluTilingData"
            or reflection["data_size_bytes"] != 80
            or reflection["sha256"] != SWIGLU_REFLECTION_SHA256
            or reflection["fields"] != expected_fields):
        raise ValueError("Ascend SwiGLU reflection schema changed")

    audit = _exact_mapping(
        shard["launch_import_audit"],
        {"all_imported_launch_apis_intercepted", "audited_files",
         "imported_launch_apis", "intercepted_launch_apis", "tool"},
        "Ascend SwiGLU launch-import audit")
    tool = _exact_mapping(
        audit["tool"], {"path", "sha256"},
        "Ascend SwiGLU launch-import audit tool")
    require_sha256_hex(
        tool["sha256"], "Ascend SwiGLU launch-import audit tool SHA256")
    if (not isinstance(tool["path"], str)
            or "readelf" not in Path(tool["path"]).name):
        raise ValueError("Ascend SwiGLU launch-import audit tool changed")
    audited = audit["audited_files"]
    if not isinstance(audited, list):
        raise TypeError("Ascend SwiGLU audited files must be a list")
    imported_union = set()
    audited_relatives = []
    for item in audited:
        identity = _exact_mapping(
            item, {"imported_launch_apis", "path", "relative_path", "sha256"},
            "Ascend SwiGLU audited launch-import file")
        relative = identity["relative_path"]
        imported = identity["imported_launch_apis"]
        if (not isinstance(imported, list)
                or imported != sorted(set(imported))
                or any(not isinstance(api, str) for api in imported)):
            raise ValueError("Ascend SwiGLU imported launch APIs changed")
        if (relative not in cann_hashes
                or identity["path"] != f"{cann['cann_root']}/{relative}"
                or identity["sha256"] != cann_hashes[relative]):
            raise ValueError("Ascend SwiGLU launch audit differs from CANN evidence")
        audited_relatives.append(relative)
        imported_union.update(imported)
    if (tuple(audited_relatives) != SWIGLU_EXACT_LAUNCH_IMPORT_FILES
            or imported_union != {"aclrtLaunchKernelWithHostArgs"}
            or audit["imported_launch_apis"]
            != ["aclrtLaunchKernelWithHostArgs"]
            or tuple(audit["intercepted_launch_apis"])
            != SWIGLU_EXACT_INTERCEPTED_LAUNCH_APIS
            or audit["all_imported_launch_apis_intercepted"] is not True):
        raise ValueError("Ascend SwiGLU launch-import audit changed")
    return {
        "baseline_commit": baseline,
        "cann_provenance": cann,
        "launch_import_audit": audit,
        "reflection": reflection,
        "source_provenance": source,
        "workload_frontier": shard["workload_frontier"],
    }


def _swiglu_parse_uint(text, context, *, positive=False):
    if not isinstance(text, str) or re.fullmatch(r"[0-9]+", text) is None:
        raise ValueError(f"{context} must be an unsigned decimal integer")
    value = int(text)
    if positive and value <= 0:
        raise ValueError(f"{context} must be positive")
    return value


def _swiglu_parse_optional_text(text, context):
    if text == "none":
        return None
    if not isinstance(text, str) or not text or "\n" in text or "\r" in text:
        raise ValueError(f"{context} must be non-empty single-line text or none")
    return text


def _swiglu_parse_optional_uint(text, context):
    if text == "none":
        return None
    return _swiglu_parse_uint(text, context)


def _swiglu_metadata(path):
    values = {}
    for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or "=" not in line:
            raise ValueError(f"invalid SwiGLU metadata at {path}:{line_number}")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise ValueError(f"duplicate SwiGLU metadata at {path}:{line_number}")
        values[key] = value
    return values


def _swiglu_stub_identity(values, context):
    expected = {
        "runtime_function_name": ("runtime_function_name", 0),
    }
    kind = values["stub_identity_kind"]
    runtime_name = _swiglu_parse_optional_text(
        values["stub_runtime_function_name"], f"{context} runtime name")
    runtime_status = _swiglu_parse_uint(
        values["stub_runtime_name_status"], f"{context} runtime status")
    if ((kind, runtime_status) != expected["runtime_function_name"]
            or runtime_name != SWIGLU_EXACT_RUNTIME_SYMBOL):
        raise ValueError("Ascend SwiGLU raw runtime kernel identity changed")
    membership = values["stub_elf_membership"]
    object_path = _swiglu_parse_optional_text(
        values["stub_elf_object_path"], f"{context} ELF object")
    object_base = _swiglu_parse_optional_uint(
        values["stub_elf_object_base"], f"{context} ELF object base")
    object_offset = _swiglu_parse_optional_uint(
        values["stub_elf_object_offset"], f"{context} ELF object offset")
    elf_symbol = _swiglu_parse_optional_text(
        values["stub_elf_symbol"], f"{context} ELF symbol")
    symbol_address = _swiglu_parse_optional_uint(
        values["stub_elf_symbol_address"], f"{context} ELF symbol address")
    symbol_offset = _swiglu_parse_optional_uint(
        values["stub_elf_symbol_offset"], f"{context} ELF symbol offset")
    stub_address = _swiglu_parse_uint(
        values["stub_address"], f"{context} stub address", positive=True)
    if membership == "mapped_elf":
        if (object_path is None or not Path(object_path).is_absolute()
                or object_base is None or object_offset is None
                or stub_address != object_base + object_offset):
            raise ValueError("Ascend SwiGLU raw ELF object identity changed")
    elif membership == "not_mapped_elf":
        if any(item is not None for item in (
                object_path, object_base, object_offset, elf_symbol,
                symbol_address, symbol_offset)):
            raise ValueError("Ascend SwiGLU unresolved ELF identity changed")
    else:
        raise ValueError("Ascend SwiGLU raw ELF membership changed")
    if elf_symbol is None:
        if symbol_address is not None or symbol_offset is not None:
            raise ValueError("Ascend SwiGLU raw ELF symbol coordinates changed")
    elif (symbol_address is None or symbol_offset is None
          or stub_address != symbol_address + symbol_offset):
        raise ValueError("Ascend SwiGLU raw ELF symbol identity changed")
    return {
        "kind": kind, "stub_address": stub_address,
        "runtime_name_status": runtime_status,
        "runtime_function_name": runtime_name,
        "elf_membership": membership, "elf_object_path": object_path,
        "elf_object_base": object_base, "elf_object_offset": object_offset,
        "elf_symbol": elf_symbol, "elf_symbol_address": symbol_address,
        "elf_symbol_offset": symbol_offset,
    }


def _swiglu_bulk_artifact_files(shard_path):
    artifact_path = shard_path.parent / "artifact.sha256"
    lines = artifact_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty SwiGLU artifact manifest: {artifact_path}")
    files = {}
    for line_number, line in enumerate(lines, 1):
        match = re.fullmatch(r"([0-9a-f]{64})  \./([^\s]+)", line)
        if match is None:
            raise ValueError(
                f"invalid SwiGLU artifact line {artifact_path}:{line_number}")
        digest, relative_text = match.groups()
        relative_path = Path(relative_text)
        if (relative_path.is_absolute() or ".." in relative_path.parts
                or relative_text in files):
            raise ValueError(
                f"unsafe or duplicate SwiGLU artifact path: {relative_text}")
        path = shard_path.parent / relative_path
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"SwiGLU artifact hash changed: {relative_text}")
        files[relative_text] = digest
    actual_files = {
        path.relative_to(shard_path.parent).as_posix()
        for path in shard_path.parent.rglob("*")
        if path.is_file() and path != artifact_path
    }
    if set(files) != actual_files or files["manifest.json"] != file_sha256(shard_path):
        raise ValueError("SwiGLU artifact inventory changed")
    return files, file_sha256(artifact_path)


def _swiglu_case_raw_artifacts(artifact_files):
    """Index each case's raw files once instead of rescanning the full shard."""

    cases = {}
    for relative in artifact_files:
        parts = Path(relative).parts
        if len(parts) < 3 or parts[0] != "cases" or parts[2] != "raw":
            continue
        if len(parts) != 4:
            raise ValueError(
                f"nested Ascend SwiGLU raw artifact path: {relative}")
        case_key = parts[1]
        if case_key in cases:
            case_files = cases[case_key]
        else:
            case_files = set()
            cases[case_key] = case_files
        case_files.add(relative)
    return {
        case_key: frozenset(case_files)
        for case_key, case_files in cases.items()
    }


def _swiglu_result_artifact(shard_path, result, name, artifact_files):
    identity = _exact_mapping(
        result[name], {"path", "sha256"}, f"SwiGLU result {name}")
    digest = require_sha256_hex(identity["sha256"], f"SwiGLU {name} SHA256")
    filenames = {
        "contract": "case_contract.json",
        "records": "records.json",
        "report": "report.json",
    }
    expected_relative = (
        Path("cases") / result["workload_key"] / filenames[name]).as_posix()
    if expected_relative not in artifact_files or artifact_files[expected_relative] != digest:
        raise ValueError(f"SwiGLU {name} artifact identity changed")
    return shard_path.parent / expected_relative


def _swiglu_raw_file(case_dir, name, artifact_files, context):
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError(f"{context} must be a plain file name")
    relative = (case_dir.relative_to(case_dir.parents[1]) / "raw" / name).as_posix()
    path = case_dir / "raw" / name
    if relative not in artifact_files or not path.is_file():
        raise FileNotFoundError(f"{context} is missing: {path}")
    return path, relative


def _swiglu_parse_raw_records(
        case_dir, artifact_files, expected_case_raw_files):
    raw_dir = case_dir / "raw"
    metadata_paths = sorted(raw_dir.glob("capture_*.meta"))
    if len(metadata_paths) != 12:
        raise ValueError("Ascend SwiGLU raw capture must contain 12 metadata files")
    records = []
    referenced = set()
    common_fields = {
        "schema_version", "record_type", "sequence", "api", "pid", "tid",
    }
    for metadata_path in metadata_paths:
        values = _swiglu_metadata(metadata_path)
        record_type = values["record_type"]
        _, metadata_relative = _swiglu_raw_file(
            case_dir, metadata_path.name, artifact_files,
            "Ascend SwiGLU raw metadata")
        referenced.add(metadata_relative)
        common = {
            "record_type": record_type,
            "sequence": _swiglu_parse_uint(
                values["sequence"], "Ascend SwiGLU raw sequence", positive=True),
            "api": values["api"],
            "pid": _swiglu_parse_uint(
                values["pid"], "Ascend SwiGLU raw pid", positive=True),
            "tid": _swiglu_parse_uint(
                values["tid"], "Ascend SwiGLU raw tid", positive=True),
            "metadata_path": str(metadata_path.resolve()),
            "metadata_sha256": artifact_files[metadata_relative],
        }
        if record_type == "workspace":
            expected_fields = common_fields | {"workspace_size", "execute_stream"}
            if set(values) != expected_fields or values["schema_version"] != "1":
                raise ValueError("Ascend SwiGLU raw workspace metadata changed")
            if values["api"] != "aclnnSwiGlu":
                raise ValueError("Ascend SwiGLU raw workspace API changed")
            records.append({
                **common,
                "workspace_size_bytes": _swiglu_parse_uint(
                    values["workspace_size"], "Ascend SwiGLU workspace size"),
                "execute_stream": _swiglu_parse_uint(
                    values["execute_stream"], "Ascend SwiGLU execute stream"),
            })
            continue
        if record_type != "launch_args":
            raise ValueError(f"unknown Ascend SwiGLU raw record: {record_type!r}")
        expected_fields = common_fields | {
            "execute_api", "stub_identity_kind", "stub_address",
            "stub_runtime_name_status", "stub_runtime_function_name",
            "stub_elf_membership", "stub_elf_object_path",
            "stub_elf_object_base", "stub_elf_object_offset",
            "stub_elf_symbol", "stub_elf_symbol_address",
            "stub_elf_symbol_offset", "block_dim", "args_size", "args_file",
            "placeholder_count", "placeholder_entry_size", "placeholder_file",
            "execute_stream", "launch_stream",
        }
        if set(values) != expected_fields or values["schema_version"] != "2":
            raise ValueError("Ascend SwiGLU raw launch metadata changed")
        if (values["api"] != "aclrtLaunchKernelWithHostArgs"
                or values["execute_api"] != "aclnnSwiGlu"):
            raise ValueError("Ascend SwiGLU raw launch API changed")
        args_path, args_relative = _swiglu_raw_file(
            case_dir, values["args_file"], artifact_files,
            "Ascend SwiGLU host args")
        placeholder_path, placeholder_relative = _swiglu_raw_file(
            case_dir, values["placeholder_file"], artifact_files,
            "Ascend SwiGLU placeholder map")
        referenced.update((args_relative, placeholder_relative))
        args_size = _swiglu_parse_uint(
            values["args_size"], "Ascend SwiGLU host args size", positive=True)
        args = args_path.read_bytes()
        if len(args) != args_size:
            raise ValueError("Ascend SwiGLU host args size differs from metadata")
        placeholder_count = _swiglu_parse_uint(
            values["placeholder_count"], "Ascend SwiGLU placeholder count")
        entry_size = _swiglu_parse_uint(
            values["placeholder_entry_size"],
            "Ascend SwiGLU placeholder entry size", positive=True)
        placeholder_bytes = placeholder_path.read_bytes()
        if entry_size != 8 or len(placeholder_bytes) != placeholder_count * 8:
            raise ValueError("Ascend SwiGLU placeholder bytes changed")
        placeholders = [
            {"address_offset": address, "data_offset": data}
            for address, data in struct.iter_unpack("<II", placeholder_bytes)
        ]
        if (placeholder_count == 0
                or any(item["address_offset"] + 8 > args_size
                       or item["data_offset"] >= args_size
                       for item in placeholders)):
            raise ValueError("Ascend SwiGLU placeholder bounds changed")
        records.append({
            **common, "execute_api": values["execute_api"],
            "stub_identity": _swiglu_stub_identity(values, str(metadata_path)),
            "block_dim": _swiglu_parse_uint(
                values["block_dim"], "Ascend SwiGLU block_dim", positive=True),
            "args_size_bytes": args_size, "args_path": str(args_path.resolve()),
            "args_sha256": artifact_files[args_relative],
            "placeholder_count": placeholder_count,
            "placeholder_path": str(placeholder_path.resolve()),
            "placeholder_sha256": artifact_files[placeholder_relative],
            "placeholders": placeholders,
            "execute_stream": _swiglu_parse_uint(
                values["execute_stream"], "Ascend SwiGLU execute stream"),
            "launch_stream": _swiglu_parse_uint(
                values["launch_stream"], "Ascend SwiGLU launch stream"),
        })
    if referenced != expected_case_raw_files or len(referenced) != 24:
        raise ValueError("Ascend SwiGLU raw artifact inventory changed")
    sequences = [record["sequence"] for record in records]
    if sequences != list(range(sequences[0], sequences[0] + 12)):
        raise ValueError("Ascend SwiGLU raw sequences are not contiguous")
    return records


def _swiglu_compare_records(serialized, raw_records):
    if not isinstance(serialized, list) or len(serialized) != len(raw_records):
        raise ValueError("Ascend SwiGLU records.json must contain 12 records")
    path_fields = {"metadata_path", "args_path", "placeholder_path"}
    for expected, actual in zip(raw_records, serialized):
        if expected["record_type"] == "workspace":
            fields = SWIGLU_EXACT_WORKSPACE_RECORD_FIELDS
        elif expected["record_type"] == "launch_args":
            fields = SWIGLU_EXACT_LAUNCH_RECORD_FIELDS
        else:
            raise ValueError(
                f"unknown Ascend SwiGLU record: {expected['record_type']!r}")
        actual = _exact_mapping(actual, fields, "Ascend SwiGLU records.json record")
        for name in fields:
            if name in path_fields:
                if Path(actual[name]).name != Path(expected[name]).name:
                    raise ValueError("Ascend SwiGLU records.json path changed")
            elif actual[name] != expected[name]:
                raise ValueError(f"Ascend SwiGLU records.json {name} changed")


def _swiglu_decode_payload(payload):
    if len(payload) != 80:
        raise ValueError("Ascend SwiGLU serialized payload must be 80 bytes")
    formats = {"uint8_t": "B", "uint32_t": "I", "uint64_t": "Q"}
    decoded = {}
    for name, dtype, offset, count, _ in SWIGLU_EXACT_REFLECTION_FIELDS:
        values = struct.unpack_from("<" + formats[dtype] * count, payload, offset)
        decoded[name] = values[0] if count == 1 else list(values)
    return decoded


def _swiglu_raw_payloads(records, contract):
    workspace_records = [
        record for record in records if record["record_type"] == "workspace"]
    launch_records = [
        record for record in records if record["record_type"] == "launch_args"]
    if len(workspace_records) != 6 or len(launch_records) != 6:
        raise ValueError("Ascend SwiGLU raw evidence requires six launch pairs")
    for workspace, launch in zip(workspace_records, launch_records):
        if (workspace["sequence"] + 1 != launch["sequence"]
                or workspace["pid"] != launch["pid"]
                or workspace["tid"] != launch["tid"]
                or workspace["execute_stream"] != launch["execute_stream"]
                or launch["execute_stream"] != launch["launch_stream"]):
            raise ValueError("Ascend SwiGLU raw workspace/launch pair changed")
    block_dims = {record["block_dim"] for record in launch_records}
    if len(block_dims) != 1:
        raise ValueError("Ascend SwiGLU raw block_dim varies")
    candidates = []
    for record in launch_records:
        args = Path(record["args_path"]).read_bytes()
        data_offsets = sorted({
            item["data_offset"] for item in record["placeholders"]})
        matches = []
        for index, data_offset in enumerate(data_offsets):
            boundary = (data_offsets[index + 1]
                        if index + 1 < len(data_offsets) else len(args))
            if boundary < data_offset + 80:
                continue
            payload = args[data_offset:data_offset + 80]
            decoded = _swiglu_decode_payload(payload)
            if (decoded["rowLen"] != contract["tok"]
                    or decoded["colLen"] != contract["intermediate"]
                    or decoded["baseRowLen"] <= 0
                    or decoded["baseColLen"] <= 0
                    or decoded["usedCoreNum"] > 48
                    or any(decoded[name] not in (0, 1) for name in (
                        "is32BAligned", "isDoubleBuffer", "activateLeft",
                        "biasIsEmpty", "quantScaleIsEmpty",
                        "activateScaleIsEmpty"))
                    or decoded["is32BAligned"]
                    != int(decoded["colLen"] * contract["dtype_bytes"] % 32 == 0)):
                continue
            matches.append({
                "address_offsets_bytes": sorted(
                    item["address_offset"] for item in record["placeholders"]
                    if item["data_offset"] == data_offset),
                "bounded_region_size_bytes": boundary - data_offset,
                "data_offset_bytes": data_offset,
                "decoded": decoded,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            })
        if len(matches) != 1:
            raise ValueError("Ascend SwiGLU raw payload is not uniquely identified")
        candidates.append(matches[0])
    if any(candidate != candidates[0] for candidate in candidates[1:]):
        raise ValueError("Ascend SwiGLU raw payload varies across executions")
    return next(iter(block_dims)), workspace_records, launch_records, candidates[0]


def _swiglu_tile_schedule(decoded, dtype_bytes):
    row_base, row_tail = divmod(decoded["rowLen"], decoded["baseRowLen"])
    col_base, col_tail = divmod(decoded["colLen"], decoded["baseColLen"])
    alignment = 32 // dtype_bytes

    def calculation_length(rows, columns):
        if decoded["is32BAligned"] == 1:
            return rows * columns
        return rows * ((columns + alignment - 1) // alignment * alignment)

    classes = []
    for name, rows, columns, count in (
            ("base_row_base_col", decoded["baseRowLen"],
             decoded["baseColLen"], row_base * col_base),
            ("base_row_tail_col", decoded["baseRowLen"], col_tail,
             row_base if col_tail > 0 else 0),
            ("tail_row_base_col", row_tail, decoded["baseColLen"],
             col_base if row_tail > 0 else 0),
            ("tail_row_tail_col", row_tail, col_tail,
             1 if row_tail > 0 and col_tail > 0 else 0)):
        if count > 0:
            classes.append({
                "name": name, "rows": rows, "columns": columns,
                "cur_cal_len_elements": calculation_length(rows, columns),
                "tile_count": count,
            })
    if not classes:
        raise ValueError("Ascend SwiGLU decoded tiling has no tiles")
    return {
        "row_tiles": row_base + int(row_tail > 0),
        "column_tiles": col_base + int(col_tail > 0),
        "total_tiles": sum(item["tile_count"] for item in classes),
        "tile_classes": classes,
        "maximum_cur_cal_len_elements": max(
            item["cur_cal_len_elements"] for item in classes),
    }


def _swiglu_exact_entry(
        shard_path, result, artifact_files, artifact_sha256, reflection,
        expected_case_raw_files):
    result = _exact_mapping(
        result, SWIGLU_EXACT_RESULT_FIELDS, "Ascend SwiGLU shard result")
    _reject_swiglu_measurement_fields(result, "Ascend SwiGLU shard result")
    semantic = _exact_mapping(
        result["semantic_key"], {"dim", "dtype", "intermediate", "layout", "tok"},
        "Ascend SwiGLU semantic key")
    tok = require_positive_int(semantic, "tok", "Ascend SwiGLU semantic key")
    intermediate = require_positive_int(
        semantic, "intermediate", "Ascend SwiGLU semantic key")
    if (semantic["dtype"] != "bf16" or semantic["layout"] != "ND"
            or semantic["dim"] != -1):
        raise ValueError("Ascend SwiGLU semantic key changed")
    expected_workload_key = f"VECTOR_SILU_T{tok}_I{intermediate}_B1"
    if result["workload_key"] != expected_workload_key:
        raise ValueError("Ascend SwiGLU workload key differs from semantic key")
    application_log = _exact_mapping(
        result["application_log"], {"path", "sha256"},
        "Ascend SwiGLU application log")
    application_log_sha256 = require_sha256_hex(
        application_log["sha256"], "Ascend SwiGLU application log SHA256")
    application_log_relative = (
        Path("logs") / Path(application_log["path"]).name).as_posix()
    if (application_log_relative not in artifact_files
            or artifact_files[application_log_relative]
            != application_log_sha256):
        raise ValueError("Ascend SwiGLU application log identity changed")
    command = result["application_command"]
    if (not isinstance(command, list) or len(command) != 6
            or re.fullmatch(
                r"python(?:3(?:\.[0-9]+)?)?", Path(command[0]).name) is None
            or not command[1].endswith(
                "/microbench/ascend_910b/operators/swiglu/"
                "tiling_capture_application_runner.py")
            or command[2] != "--workload-batch"
            or re.fullmatch(
                r"application_batch_[0-9]{6}\.json",
                Path(command[3]).name) is None
            or command[4] != "--device-id"
            or re.fullmatch(r"[0-9]+", command[5]) is None):
        raise ValueError("Ascend SwiGLU application command changed")
    contract_path = _swiglu_result_artifact(
        shard_path, result, "contract", artifact_files)
    records_path = _swiglu_result_artifact(
        shard_path, result, "records", artifact_files)
    report_path = _swiglu_result_artifact(
        shard_path, result, "report", artifact_files)
    contract = _exact_mapping(
        json.loads(contract_path.read_text(encoding="utf-8")),
        SWIGLU_EXACT_CONTRACT_FIELDS, "Ascend SwiGLU contract")
    if (contract["case_key"] != expected_workload_key
            or contract["capture_input"] != "workload_manifest"
            or contract["block_dim"] is not None
            or contract["kernel_name"] is not None
            or contract["dtype"] != "bf16" or contract["layout"] != "ND"
            or contract["dim"] != -1 or contract["tok"] != tok
            or contract["intermediate"] != intermediate
            or contract["element_count"] != tok * intermediate
            or contract["dtype_bytes"] != 2 or contract["batch"] != 1):
        raise ValueError("Ascend SwiGLU case contract changed")
    for name in (
            "dtype_bytes", "element_count", "head_dim", "hidden",
            "intermediate", "num_heads", "num_kv_heads", "tok"):
        require_positive_int(contract, name, "Ascend SwiGLU contract")
    if (contract["layer"] is not None
            or any(contract[name] != 1 for name in (
                "head_dim", "hidden", "num_heads", "num_kv_heads"))):
        raise ValueError("Ascend SwiGLU exact-domain contract must not name a layer")
    raw_records = _swiglu_parse_raw_records(
        contract_path.parent, artifact_files, expected_case_raw_files)
    serialized_records = json.loads(records_path.read_text(encoding="utf-8"))
    _swiglu_compare_records(serialized_records, raw_records)
    raw_block_dim, workspace_records, launch_records, raw_candidate = (
        _swiglu_raw_payloads(raw_records, contract))
    report_value = json.loads(report_path.read_text(encoding="utf-8"))
    _reject_swiglu_measurement_fields(report_value, "Ascend SwiGLU report")
    report = _exact_mapping(
        report_value,
        SWIGLU_EXACT_REPORT_FIELDS, "Ascend SwiGLU report")
    launch = _exact_mapping(
        result["launch_result"], SWIGLU_EXACT_LAUNCH_RESULT_FIELDS,
        "Ascend SwiGLU launch result")
    identity = _exact_mapping(
        launch["kernel_identity"], {"kind", "name", "resolver"},
        "Ascend SwiGLU kernel identity")
    if identity != {
            "kind": "runtime_function_name",
            "name": SWIGLU_EXACT_RUNTIME_SYMBOL,
            "resolver": "aclrtGetFunctionName",
    }:
        raise ValueError("Ascend SwiGLU requires the exact runtime kernel identity")
    payload_selection = _exact_mapping(
        report["tiling_payload_selection"], SWIGLU_PAYLOAD_SELECTION_KEYS,
        "Ascend SwiGLU tiling payload selection")
    if (launch["kernel_symbol"] != SWIGLU_EXACT_RUNTIME_SYMBOL
            or launch["target_kernel_submitted"] is not False
            or report["schema_version"] != 3
            or report["artifact_type"]
            != "ascend_910b_swiglu_tiling_observation"
            or report["duration_consumed"] is not False
            or report["target_kernel_submitted"] is not False
            or report["execution_count"] != 6
            or report["execute_api"] != "aclnnSwiGlu"
            or report["launch_api"] != "aclrtLaunchKernelWithHostArgs"
            or report["stream_identity_equal"] is not True
            or report["kernel_identity"] != identity
            or report["kernel_symbol"] != SWIGLU_EXACT_RUNTIME_SYMBOL
            or report["case_contract"] != contract
            or report["block_dim"] != launch["block_dim"]
            or report["decoded_tiling"] != launch["decoded_tiling"]
            or report["source_derived_tile_schedule"]
            != launch["source_derived_tile_schedule"]
            or payload_selection["payload_sha256"]
            != launch["serialized_tiling_sha256"]
            or report["limitations"] != result["limitations"]):
        raise ValueError("Ascend SwiGLU report and shard result disagree")
    report_reflection = _exact_mapping(
        report["reflection"],
        {"class_name", "data_size_bytes", "path", "registration_key", "sha256"},
        "Ascend SwiGLU report reflection")
    if report_reflection != {
            "registration_key": reflection["registration_key"],
            "class_name": reflection["class_name"],
            "data_size_bytes": reflection["data_size_bytes"],
            "path": reflection["path"],
            "sha256": reflection["sha256"],
    }:
        raise ValueError("Ascend SwiGLU report reflection changed")
    raw_identity = {
        "kind": "runtime_function_name",
        "name": SWIGLU_EXACT_RUNTIME_SYMBOL,
        "resolver": "aclrtGetFunctionName",
    }
    if (any(record["stub_identity"]["kind"] != "runtime_function_name"
            or record["stub_identity"]["runtime_function_name"]
            != SWIGLU_EXACT_RUNTIME_SYMBOL for record in launch_records)
            or identity != raw_identity
            or raw_block_dim != report["block_dim"]
            or report["workspace_size_bytes"] != sorted({
                record["workspace_size_bytes"] for record in workspace_records})
            or report["placeholder_maps"]
            != [record["placeholders"] for record in launch_records]
            or report["tiling_used_core_num"]
            != raw_candidate["decoded"]["usedCoreNum"]
            or report["tiling_used_core_num_matches_launch"]
            is not (raw_candidate["decoded"]["usedCoreNum"] == raw_block_dim)):
        raise ValueError("Ascend SwiGLU raw launch evidence differs from report")
    expected_payload_selection = {
        "status": "unique_reflected_contract_match",
        "data_offset_bytes": raw_candidate["data_offset_bytes"],
        "address_offsets_bytes": raw_candidate["address_offsets_bytes"],
        "serialized_size_bytes": 80,
        "bounded_region_size_bytes": raw_candidate["bounded_region_size_bytes"],
        "payload_sha256": raw_candidate["payload_sha256"],
    }
    if payload_selection != expected_payload_selection:
        raise ValueError("Ascend SwiGLU raw payload selection differs from report")
    block_dim = require_positive_int(launch, "block_dim", "Ascend SwiGLU launch")
    if block_dim > 48:
        raise ValueError("Ascend SwiGLU block_dim exceeds physical AIVs")
    decoded = _exact_mapping(
        launch["decoded_tiling"], SWIGLU_DECODED_TILING_KEYS,
        "Ascend SwiGLU decoded tiling")
    for name in SWIGLU_DECODED_TILING_KEYS - {"SwiGluTilingDataPH"}:
        value = decoded[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TypeError(f"Ascend SwiGLU decoded {name} must be non-negative")
    padding = decoded["SwiGluTilingDataPH"]
    if (not isinstance(padding, list) or len(padding) != 4
            or any(type(value) is not int or not 0 <= value <= 255
                   for value in padding)):
        raise ValueError("Ascend SwiGLU decoded padding changed")
    base_row_len = require_positive_int(
        decoded, "baseRowLen", "Ascend SwiGLU decoded tiling")
    base_col_len = require_positive_int(
        decoded, "baseColLen", "Ascend SwiGLU decoded tiling")
    if (decoded["rowLen"] != tok or decoded["colLen"] != intermediate
            or decoded["isDoubleBuffer"] not in (0, 1)):
        raise ValueError("Ascend SwiGLU decoded shape differs from semantic key")
    if decoded != raw_candidate["decoded"]:
        raise ValueError("Ascend SwiGLU decoded tiling differs from raw payload")
    schedule = _exact_mapping(
        launch["source_derived_tile_schedule"], SWIGLU_TILE_SCHEDULE_KEYS,
        "Ascend SwiGLU tile schedule")
    for name in ("row_tiles", "column_tiles", "total_tiles"):
        require_positive_int(schedule, name, "Ascend SwiGLU tile schedule")
    raw_schedule = _swiglu_tile_schedule(decoded, contract["dtype_bytes"])
    expected_block_dim = min(48, schedule["total_tiles"])
    if schedule != raw_schedule or block_dim != expected_block_dim:
        raise ValueError("Ascend SwiGLU tile schedule is inconsistent")
    row_tiles = schedule["row_tiles"]
    column_tiles = schedule["column_tiles"]
    total_tiles = schedule["total_tiles"]
    payload_sha256 = require_sha256_hex(
        launch["serialized_tiling_sha256"], "Ascend SwiGLU tiling payload")
    if payload_sha256 != raw_candidate["payload_sha256"]:
        raise ValueError("Ascend SwiGLU launch payload hash differs from raw bytes")
    return (tok, intermediate), {
        "artifact_manifest_sha256": artifact_sha256,
        "base_col_len": base_col_len,
        "base_row_len": base_row_len,
        "block_dim": block_dim,
        "column_tiles": column_tiles,
        "kernel_identity_kind": "runtime_function_name",
        "payload_sha256": payload_sha256,
        "queue_depth": 2 if decoded["isDoubleBuffer"] == 1 else 1,
        "report_sha256": file_sha256(report_path),
        "row_tiles": row_tiles,
        "runtime_symbol": SWIGLU_EXACT_RUNTIME_SYMBOL,
        "source_case_id": expected_workload_key,
        "total_tiles": total_tiles,
    }


def build_swiglu_launch_facts(observation_root):
    """Aggregate a gap-free exact BF16 SwiGLU domain without timing data."""

    observation_root = Path(observation_root).resolve()
    if not observation_root.is_dir():
        raise FileNotFoundError(
            f"Ascend SwiGLU observation root is missing: {observation_root}")
    shards = []
    for path in sorted(observation_root.rglob("manifest.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if (isinstance(value, dict) and "artifact_type" in value
                and value["artifact_type"] == SWIGLU_EXACT_CAPTURE_ARTIFACT_TYPE):
            shards.append((path, value))
    if not shards:
        raise ValueError("Ascend SwiGLU observation root contains no bulk shards")
    expected_keys = _swiglu_exact_keys()
    common = None
    ranges = []
    entries = {}
    for shard_path, value in shards:
        shard = _exact_mapping(
            value, SWIGLU_EXACT_SHARD_FIELDS, "Ascend SwiGLU bulk shard")
        _reject_swiglu_measurement_fields(shard, "Ascend SwiGLU bulk shard")
        if (shard["schema_version"] != SWIGLU_EXACT_CAPTURE_SCHEMA_VERSION
                or shard["capture_state"]
                != "complete_non_duration_observation"
                or shard["duration_consumed"] is not False
                or shard["target_kernel_submitted"] is not False
                or shard["compile_count"] != 1
                or shard["full_workload_count"]
                != SWIGLU_EXACT_WORKLOAD_COUNT):
            raise ValueError("Ascend SwiGLU bulk shard identity changed")
        for name in (
                "application_batch_size", "application_process_count",
                "compile_count", "full_workload_count"):
            _gemm_positive_integer(
                shard[name], f"Ascend SwiGLU shard {name}")
        frontier = _exact_mapping(shard["workload_frontier"], {
            "artifact_type", "coverage_claim", "duration_consumed",
            "encoded_swiglu_keys", "path", "schema_version", "sha256",
            "source_config", "source_config_sha256", "workload_keys",
        }, "Ascend SwiGLU workload frontier")
        if (frontier["schema_version"] != 2
                or frontier["artifact_type"] != "gemm_workload_candidate"
                or frontier["coverage_claim"]
                != "complete_exact_swiglu_launch_domain"
                or frontier["duration_consumed"] is not False
                or len(frontier["workload_keys"]) != SWIGLU_EXACT_WORKLOAD_COUNT
                or len(frontier["encoded_swiglu_keys"])
                != SWIGLU_EXACT_WORKLOAD_COUNT):
            raise ValueError("Ascend SwiGLU workload frontier changed")
        if (not isinstance(frontier["source_config"], str)
                or Path(frontier["source_config"]).is_absolute()
                or ".." in Path(frontier["source_config"]).parts):
            raise ValueError("Ascend SwiGLU workload source config changed")
        require_sha256_hex(frontier["sha256"], "Ascend SwiGLU candidate SHA256")
        require_sha256_hex(
            frontier["source_config_sha256"],
            "Ascend SwiGLU source config SHA256")
        frontier_keys = [
            tuple(json.loads(key)) for key in frontier["encoded_swiglu_keys"]]
        if any(len(key) != 2 or type(key[0]) is not int or type(key[1]) is not int
               for key in frontier_keys):
            raise TypeError("Ascend SwiGLU frontier keys must be integer pairs")
        if frontier_keys != expected_keys:
            raise ValueError("Ascend SwiGLU workload frontier is not exact numeric order")
        expected_workload_names = [
            f"VECTOR_SILU_T{tok}_I{intermediate}_B1"
            for tok, intermediate in expected_keys]
        if frontier["workload_keys"] != expected_workload_names:
            raise ValueError("Ascend SwiGLU workload names are not exact numeric order")
        postflight = _exact_mapping(shard["postflight_identity"], {
            "cann_provenance_sha256", "source_provenance_sha256",
            "workload_candidate_sha256",
        }, "Ascend SwiGLU postflight identity")
        if (postflight["source_provenance_sha256"]
                != sorted_json_sha256(shard["source_provenance"])
                or postflight["cann_provenance_sha256"]
                != sorted_json_sha256(shard["cann_provenance"])
                or postflight["workload_candidate_sha256"]
                != sorted_json_sha256(frontier)):
            raise ValueError("Ascend SwiGLU postflight identity rehash changed")
        shard_common = _swiglu_exact_provenance(shard)
        if common is None:
            common = shard_common
        elif shard_common != common:
            raise ValueError("Ascend SwiGLU bulk shards disagree on provenance")
        selected = _exact_mapping(
            shard["selected_range"], {"end_index", "start_index"},
            "Ascend SwiGLU selected range")
        start = selected["start_index"]
        end = selected["end_index"]
        if (isinstance(start, bool) or not isinstance(start, int)
                or isinstance(end, bool) or not isinstance(end, int)
                or start < 0 or end <= start or end > len(expected_keys)):
            raise ValueError("Ascend SwiGLU shard range is invalid")
        results = shard["results"]
        if not isinstance(results, list):
            raise TypeError("Ascend SwiGLU shard results must be a list")
        selected_count = _gemm_positive_integer(
            shard["selected_workload_count"],
            "Ascend SwiGLU selected workload count")
        result_count = _gemm_positive_integer(
            shard["result_count"], "Ascend SwiGLU result count")
        if (selected_count != end - start
                or result_count != end - start
                or len(results) != end - start):
            raise ValueError("Ascend SwiGLU shard counts are inconsistent")
        ranges.append((start, end))
        artifact_files, artifact_sha256 = _swiglu_bulk_artifact_files(shard_path)
        raw_artifacts_by_case = _swiglu_case_raw_artifacts(artifact_files)
        expected_case_names = {
            f"VECTOR_SILU_T{tok}_I{intermediate}_B1"
            for tok, intermediate in expected_keys[start:end]
        }
        if set(raw_artifacts_by_case) != expected_case_names:
            raise ValueError("Ascend SwiGLU raw case inventory changed")
        interposer = _exact_mapping(
            shard["interposer"], {"path", "sha256"},
            "Ascend SwiGLU interposer")
        interposer_sha256 = require_sha256_hex(
            interposer["sha256"], "Ascend SwiGLU interposer SHA256")
        if (artifact_files["build/libmidlayer_swiglu_tiling_capture.so"]
                != interposer_sha256):
            raise ValueError("Ascend SwiGLU interposer artifact changed")
        for offset, result in enumerate(results):
            expected_case_name = (
                f"VECTOR_SILU_T{expected_keys[start + offset][0]}_"
                f"I{expected_keys[start + offset][1]}_B1")
            key, entry = _swiglu_exact_entry(
                shard_path, result, artifact_files, artifact_sha256,
                shard_common["reflection"],
                raw_artifacts_by_case[expected_case_name])
            if key != expected_keys[start + offset]:
                raise ValueError("Ascend SwiGLU shard results are not numeric order")
            if key in entries:
                raise ValueError(f"duplicate Ascend SwiGLU result: {key}")
            entries[key] = entry
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError("Ascend SwiGLU shard ranges contain a gap or overlap")
        cursor = end
    if cursor != len(expected_keys) or set(entries) != set(expected_keys):
        raise ValueError("Ascend SwiGLU shards do not provide complete coverage")
    if common is None:
        raise AssertionError("Ascend SwiGLU common provenance is missing")
    frontier = common["workload_frontier"]
    source = _gemm_candidate_manifest_path(frontier["path"])
    source_snapshot_sha256 = sorted_json_sha256(common["source_provenance"])
    cann_provenance_sha256 = sorted_json_sha256(common["cann_provenance"])
    return {
        "artifact_type": SWIGLU_EXACT_FACT_ARTIFACT_TYPE,
        "candidate_manifest_sha256": frontier["sha256"],
        "cann_provenance_sha256": cann_provenance_sha256,
        "capture_baseline_commit": common["baseline_commit"],
        "coverage_claim": "complete_exact_legal_domain",
        "duration_consumed": False,
        "entries": {_tuple_key(key): entries[key] for key in expected_keys},
        "exact_domain": {
            "dtype": "bf16", "layout": "ND", "dim": -1,
            "m_min": SWIGLU_EXACT_M_MIN, "m_max": SWIGLU_EXACT_M_MAX,
            "intermediate_values": list(SWIGLU_EXACT_INTERMEDIATES),
        },
        "generated_by": "scripts/launch_data/ascend_910b_launch_generator.py",
        "hardware_id": "ascend_910b",
        "schema_version": SWIGLU_EXACT_FACT_SCHEMA_VERSION,
        "semantic_key_count": SWIGLU_EXACT_WORKLOAD_COUNT,
        "software_revision": "cann_8.5.1",
        "source": source,
        "source_snapshot_sha256": source_snapshot_sha256,
        "target_kernel_submitted": False,
        "workload_key_fields": ["tok", "intermediate"],
    }


def _gemm_nonnegative_integer(value, context):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if value < 0:
        raise ValueError(f"{context} must be non-negative")
    return value


def _gemm_positive_integer(value, context):
    if _gemm_nonnegative_integer(value, context) == 0:
        raise ValueError(f"{context} must be positive")
    return value


def _gemm_sequence(value, context, *, nonempty=False):
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    if nonempty and not value:
        raise ValueError(f"{context} must not be empty")
    return value


def _gemm_repository_relative_path(value, context):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"{context} must be a canonical repository-relative path")
    return value


def _gemm_candidate_manifest_path(value):
    """Normalize the runner's remote path to its repository-owned identity."""

    if not isinstance(value, str) or not value:
        raise ValueError("Ascend GEMM candidate path must be non-empty")
    path = Path(value)
    if ".." in path.parts:
        raise ValueError("Ascend GEMM candidate path must not traverse parents")
    marker = ("profiler_output", "launch_data")
    starts = [
        index for index in range(len(path.parts) - 1)
        if path.parts[index:index + 2] == marker
    ]
    if len(starts) != 1:
        raise ValueError(
            "Ascend GEMM candidate path must identify exactly one "
            "profiler_output/launch_data subtree")
    relative = Path(*path.parts[starts[0]:]).as_posix()
    if not relative.endswith(".json"):
        raise ValueError("Ascend GEMM candidate path must identify JSON")
    return relative


def _reject_gemm_measurement_fields(value, context):
    """Reject timing payloads while allowing explicit non-consumption flags."""

    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{context} field names must be strings")
            lowered = key.lower()
            if re.search(
                    r"(?:^|_)(?:duration|durations|cycle|cycles|timing)(?:_|$)",
                    lowered) is not None:
                allowed = (
                    key in {"duration_consumed", "timing_consumed"}
                    and item is False
                )
                if not allowed:
                    raise ValueError(
                        f"{context} contains forbidden measurement field {key!r}")
            _reject_gemm_measurement_fields(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_gemm_measurement_fields(item, f"{context}[{index}]")


def _validate_gemm_candidate_manifest(value):
    manifest = _exact_mapping(
        value, GEMM_CANDIDATE_FIELDS, "Ascend GEMM candidate manifest")
    _reject_gemm_measurement_fields(manifest, "Ascend GEMM candidate manifest")
    if (manifest["schema_version"] != 2
            or manifest["artifact_type"] != "gemm_workload_candidate"
            or manifest["duration_consumed"] is not False):
        raise ValueError("Ascend GEMM candidate manifest identity changed")
    _gemm_repository_relative_path(
        manifest["source_config"], "Ascend GEMM candidate source_config")
    require_sha256_hex(
        manifest["source_config_sha256"],
        "Ascend GEMM candidate source_config SHA256")
    if manifest["workload_key_fields"] != list(GEMM_WORKLOAD_KEY_FIELDS):
        raise ValueError("Ascend GEMM candidate workload-key schema changed")
    if manifest["physical_operation_fields"] != list(PHYSICAL_OPERATION_FIELDS):
        raise ValueError("Ascend GEMM physical-operation schema changed")
    scheduler = _exact_mapping(
        manifest["scheduler_contract"], set(EXPECTED_SCHEDULER_CONTRACT),
        "Ascend GEMM scheduler contract")
    if (scheduler != EXPECTED_SCHEDULER_CONTRACT
            or any(type(scheduler[key]) is not type(expected)
                   for key, expected in EXPECTED_SCHEDULER_CONTRACT.items())):
        raise ValueError("Ascend GEMM scheduler contract changed")
    coverage = _exact_mapping(
        manifest["coverage_scope"], {
            "fused_moe_included", "kind", "operator_trace_consumed",
            "request_trajectory_consumed", "scheduler_alignment_consumed",
            "serving_end_to_end_executability_claimed", "timing_consumed",
        }, "Ascend GEMM coverage scope")
    if coverage != {
            "kind": GEMM_CANDIDATE_SCOPE,
            "request_trajectory_consumed": False,
            "scheduler_alignment_consumed": False,
            "operator_trace_consumed": False,
            "timing_consumed": False,
            "fused_moe_included": False,
            "serving_end_to_end_executability_claimed": False,
    }:
        raise ValueError("Ascend GEMM candidate coverage scope changed")
    model_ids = set(GEMM_MODEL_PARALLELISM)
    limitations = _gemm_sequence(
        manifest["limitations"], "Ascend GEMM limitations")
    for index, value in enumerate(limitations):
        limitation = _exact_mapping(
            value, {"scope", "affected_hardware_ids", "affected_model_ids",
                    "reason"}, f"Ascend GEMM limitation {index}")
        require_non_empty_string(limitation, "scope", f"Ascend GEMM limitation {index}")
        require_non_empty_string(limitation, "reason", f"Ascend GEMM limitation {index}")
        affected_hardware = _gemm_sequence(
            limitation["affected_hardware_ids"],
            f"Ascend GEMM limitation {index} hardware", nonempty=True)
        affected_models = _gemm_sequence(
            limitation["affected_model_ids"],
            f"Ascend GEMM limitation {index} models", nonempty=True)
        if (any(not isinstance(item, str) for item in affected_hardware)
                or any(not isinstance(item, str) for item in affected_models)
                or len(affected_hardware) != len(set(affected_hardware))
                or not set(affected_hardware).issubset(GEMM_HARDWARE_IDS)
                or len(affected_models) != len(set(affected_models))
                or not set(affected_models).issubset(model_ids)):
            raise ValueError("Ascend GEMM candidate limitation scope changed")
    inventory = _exact_mapping(
        manifest["dataset_model_inventory"], set(GEMM_HARDWARE_IDS),
        "Ascend GEMM dataset model inventory")
    expected_models = sorted(model_ids)
    if any(inventory[hardware_id] != expected_models
           for hardware_id in GEMM_HARDWARE_IDS):
        raise ValueError("Ascend GEMM dataset model inventory changed")
    models = _gemm_sequence(
        manifest["model_domains"], "Ascend GEMM model domains", nonempty=True)
    seen_models = set()
    for index, value in enumerate(models):
        model = _exact_mapping(value, {
            "expert_parallel_size", "model_config_sha256", "model_id",
            "model_path", "model_type", "physical_operation_count",
            "positions", "semantic_key_count", "tensor_parallel_size",
        }, f"Ascend GEMM model domain {index}")
        model_id = require_non_empty_string(
            model, "model_id", f"Ascend GEMM model domain {index}")
        if model_id not in model_ids or model_id in seen_models:
            raise ValueError("Ascend GEMM candidate model set changed")
        seen_models.add(model_id)
        require_non_empty_string(model, "model_path", f"Ascend GEMM model {model_id}")
        require_non_empty_string(model, "model_type", f"Ascend GEMM model {model_id}")
        require_sha256_hex(
            model["model_config_sha256"], f"Ascend GEMM model {model_id}")
        expected_tp, expected_ep = GEMM_MODEL_PARALLELISM[model_id]
        tensor_parallel_size = _gemm_positive_integer(
            model["tensor_parallel_size"],
            f"Ascend GEMM model {model_id} tensor parallel size")
        expert_parallel_size = _gemm_positive_integer(
            model["expert_parallel_size"],
            f"Ascend GEMM model {model_id} expert parallel size")
        if (tensor_parallel_size != expected_tp
                or expert_parallel_size != expected_ep):
            raise ValueError("Ascend GEMM candidate model parallelism changed")
        semantic_count = _gemm_positive_integer(
            model["semantic_key_count"], f"Ascend GEMM model {model_id} keys")
        _gemm_positive_integer(
            model["physical_operation_count"],
            f"Ascend GEMM model {model_id} operations")
        positions = _gemm_sequence(
            model["positions"], f"Ascend GEMM model {model_id} positions",
            nonempty=True)
        position_count = 0
        position_names = set()
        for position_index, position_value in enumerate(positions):
            position = _exact_mapping(position_value, {
                "dtype", "has_bias", "k", "m_max", "m_min", "n",
                "position", "semantic_key_count", "variant_contract",
            }, f"Ascend GEMM model {model_id} position {position_index}")
            name = require_non_empty_string(
                position, "position", f"Ascend GEMM model {model_id} position")
            if name in position_names:
                raise ValueError("Ascend GEMM candidate repeats a model position")
            position_names.add(name)
            minimum_m = _gemm_positive_integer(
                position["m_min"], "Ascend GEMM position m_min")
            if (position["dtype"] != "bf16"
                    or position["has_bias"] is not False
                    or minimum_m != 1
                    or position["variant_contract"] not in (
                        "gemm", "m1_gemv_else_gemm")):
                raise ValueError("Ascend GEMM candidate position changed")
            maximum_m = _gemm_positive_integer(
                position["m_max"], "Ascend GEMM position m_max")
            _gemm_positive_integer(position["n"], "Ascend GEMM position N")
            _gemm_positive_integer(position["k"], "Ascend GEMM position K")
            if position["semantic_key_count"] != maximum_m:
                raise ValueError("Ascend GEMM position key count changed")
            position_count += maximum_m
        if position_count != semantic_count:
            raise ValueError("Ascend GEMM model key count is inconsistent")
    if seen_models != model_ids:
        raise ValueError("Ascend GEMM candidate model set is incomplete")
    workload_keys = _gemm_sequence(
        manifest["workload_keys"], "Ascend GEMM workload keys", nonempty=True)
    if workload_keys != sorted(set(workload_keys)):
        raise ValueError("Ascend GEMM workload keys must be sorted and unique")
    physical_operations = set()
    for workload_key in workload_keys:
        values = _gemm_workload_values(workload_key)
        if (values["dtype"] != "bf16" or values["b"] != 1
                or values["a_layout"] != "row_major"
                or values["b_layout"] != "row_major"
                or values["c_layout"] != "row_major"
                or values["trans_a"] is not False
                or values["trans_b"] is not True
                or values["epilogue"] != "none"
                or values["has_bias"] is not False):
            raise ValueError("Ascend GEMM candidate workload semantics changed")
        physical_operations.add(tuple(
            values[name] for name in PHYSICAL_OPERATION_FIELDS))
    semantic_count = _gemm_positive_integer(
        manifest["semantic_key_count"], "Ascend GEMM semantic key count")
    physical_count = _gemm_positive_integer(
        manifest["physical_operation_count"],
        "Ascend GEMM physical operation count")
    if (semantic_count != GEMM_EXACT_SEMANTIC_KEY_COUNT
            or physical_count != GEMM_EXACT_PHYSICAL_OPERATION_COUNT
            or semantic_count != len(workload_keys)
            or physical_count != len(physical_operations)):
        raise ValueError("Ascend GEMM exact-domain counts are inconsistent")
    domain_sha256 = canonical_sequence_sha256(workload_keys)
    if domain_sha256 != GEMM_EXACT_DOMAIN_SHA256:
        raise ValueError("Ascend GEMM exact workload-key set changed")
    hardware_domains = _exact_mapping(
        manifest["hardware_domains"], set(GEMM_HARDWARE_IDS),
        "Ascend GEMM hardware domains")
    expected_domain = {
        "status": "exact",
        "domain_sha256": domain_sha256,
        "semantic_key_count": semantic_count,
        "physical_operation_count": physical_count,
    }
    for hardware_id in GEMM_HARDWARE_IDS:
        domain = _exact_mapping(
            hardware_domains[hardware_id], set(expected_domain),
            f"Ascend GEMM {hardware_id} domain")
        if domain != expected_domain:
            raise ValueError(f"Ascend GEMM {hardware_id} domain changed")
    cross_hardware = _exact_mapping(manifest["cross_hardware"], {
        "domain_sha256", "hardware_ids", "physical_operation_count",
        "semantic_key_count", "status",
    }, "Ascend GEMM cross-hardware domain")
    if cross_hardware != {
            "status": "exact_equal",
            "hardware_ids": list(GEMM_HARDWARE_IDS),
            "domain_sha256": domain_sha256,
            "semantic_key_count": semantic_count,
            "physical_operation_count": physical_count,
    }:
        raise ValueError("Ascend GEMM cross-hardware domain changed")
    return manifest


def _gemm_workload_values(workload_key):
    validate_gemm_workload_key(workload_key)
    values = json.loads(workload_key)
    return dict(zip(GEMM_WORKLOAD_KEY_FIELDS, values))


def _gemm_artifact_path(shard_path, relative_path, context):
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{context} path must be non-empty")
    path = Path(relative_path)
    if (path.is_absolute() or ".." in path.parts
            or path.as_posix() != relative_path):
        raise ValueError(f"{context} path must be shard-relative")
    resolved = (shard_path.parent / path).resolve()
    if not resolved.is_relative_to(shard_path.parent.resolve()):
        raise ValueError(f"{context} path escapes its shard")
    if not resolved.is_file():
        raise FileNotFoundError(f"{context} is missing: {resolved}")
    return resolved


def _gemm_captured_bytes(shard_path, value, fields, context):
    artifact = _exact_mapping(value, fields, context)
    path = _gemm_artifact_path(shard_path, artifact["path"], context)
    try:
        serialized = bytes.fromhex(artifact["serialized_hex"])
    except ValueError as error:
        raise ValueError(f"{context} is not hexadecimal") from error
    if "size_bytes" in fields:
        _gemm_positive_integer(artifact["size_bytes"], f"{context} size")
    if (("size_bytes" in fields and len(serialized) != artifact["size_bytes"])
            or path.read_bytes() != serialized
            or file_sha256(path) != artifact["sha256"]):
        raise ValueError(f"{context} bytes, size, path, or SHA256 disagree")
    require_sha256_hex(artifact["sha256"], f"{context} SHA256")
    return artifact, serialized


def _gemm_operation_entry(operation, result_path, shard_path):
    context = f"Ascend GEMM {operation['operation_id']}"
    if file_sha256(result_path) != operation["result"]["sha256"]:
        raise ValueError(f"{context} result SHA256 changed")
    result = _exact_mapping(
        json.loads(result_path.read_text(encoding="utf-8")),
        GEMM_RESULT_FIELDS, f"{context} result")
    _reject_gemm_measurement_fields(result, f"{context} result")
    if (result["schema_version"] != GEMM_CAPTURE_SCHEMA_VERSION
            or result["operator"] != "aclnnMatmul"):
        raise ValueError(f"{context} result identity changed")
    if result["operation_id"] != operation["operation_id"]:
        raise ValueError(f"{context} result operation identity changed")
    if (result["duration_consumed"] is not False
            or result["device_kernel_submitted"] is not False
            or result["execute_api_called"] is not True):
        raise ValueError(f"{context} is not a non-duration dry launch")
    if (result["dtype"] != "bf16" or result["cube_math_type"] != 0
            or result["executor_lifecycle"]
            != "consumed_by_aclnnMatmul"):
        raise ValueError(f"{context} execution contract changed")
    workspace_size = _gemm_nonnegative_integer(
        result["workspace_size_bytes"], f"{context} workspace size")
    _gemm_positive_integer(
        result["process_peak_rss_kib"], f"{context} process peak RSS")
    shape = _exact_mapping(
        result["shape"], {"m", "n", "k"}, f"{context} shape")
    for name in ("m", "n", "k"):
        if operation[name] != shape[name]:
            raise ValueError(f"{context} shard and result shapes differ")
    expected_layout = {
        "lhs": {
            "kind": "row_major", "view_shape": [shape["m"], shape["k"]],
            "strides": [shape["k"], 1],
            "storage_shape": [shape["m"], shape["k"]],
        },
        "rhs": {
            "kind": "transposed_view",
            "view_shape": [shape["k"], shape["n"]],
            "strides": [1, shape["k"]],
            "storage_shape": [shape["n"], shape["k"]],
        },
        "output": {
            "kind": "row_major", "view_shape": [shape["m"], shape["n"]],
            "strides": [shape["n"], 1],
            "storage_shape": [shape["m"], shape["n"]],
        },
    }
    if result["layout"] != expected_layout:
        raise ValueError(f"{context} layout contract changed")
    if result["workload_keys"] != operation["workload_keys"]:
        raise ValueError(f"{context} workload-key mapping changed")
    dry_launch = _exact_mapping(result["dry_launch"], {
        "captured_launch_count", "expected_launch_count",
        "interposed_launch_apis", "launch", "nnopbase_imported_launch_apis",
    }, f"{context} dry launch")
    launch = _exact_mapping(dry_launch["launch"], {
        "api", "block_dim", "configuration", "device_kernel_submitted",
        "duration_consumed", "execute_api", "host_args", "kernel_stub",
        "kernel_family", "metadata_path", "metadata_sha256", "placeholders",
        "stream",
    }, f"{context} launch")
    expected_launch_count = _gemm_positive_integer(
        dry_launch["expected_launch_count"], f"{context} expected launches")
    captured_launch_count = _gemm_positive_integer(
        dry_launch["captured_launch_count"], f"{context} captured launches")
    if (expected_launch_count != 1
            or captured_launch_count != 1
            or dry_launch["interposed_launch_apis"] != GEMM_LAUNCH_APIS
            or dry_launch["nnopbase_imported_launch_apis"]
            != ["aclrtLaunchKernelWithHostArgs"]
            or launch["api"] != "aclrtLaunchKernelWithHostArgs"
            or launch["execute_api"] != "aclnnMatmul"
            or launch["device_kernel_submitted"] is not False
            or launch["duration_consumed"] is not False):
        raise ValueError(f"{context} launch contract changed")
    kernel_symbol = require_non_empty_string(launch, "kernel_stub", context)
    kernel_family = require_non_empty_string(launch, "kernel_family", context)
    if kernel_family not in GEMM_FAMILY_CONTRACTS:
        raise ValueError(f"{context} has unsupported MatMul family")
    family_contract = GEMM_FAMILY_CONTRACTS[kernel_family]
    if family_contract["kernel_pattern"].fullmatch(kernel_symbol) is None:
        raise ValueError(f"{context} kernel symbol and family disagree")
    block_dim = require_positive_int(launch, "block_dim", context)
    _gemm_nonnegative_integer(launch["stream"], f"{context} stream")
    metadata_path = _gemm_artifact_path(
        shard_path, launch["metadata_path"], f"{context} launch metadata")
    if file_sha256(metadata_path) != launch["metadata_sha256"]:
        raise ValueError(f"{context} launch metadata SHA256 changed")
    require_sha256_hex(
        launch["metadata_sha256"], f"{context} launch metadata SHA256")
    _, host_bytes = _gemm_captured_bytes(
        shard_path, launch["host_args"],
        {"path", "serialized_hex", "sha256", "size_bytes"},
        f"{context} host arguments")
    if len(host_bytes) != family_contract["host_args_size_bytes"]:
        raise ValueError(f"{context} family-specific host arguments changed")
    placeholders, placeholder_bytes = _gemm_captured_bytes(
        shard_path, launch["placeholders"], {
            "count", "entries", "entry_size_bytes", "path",
            "serialized_hex", "sha256",
        }, f"{context} placeholders")
    placeholder_count = _gemm_positive_integer(
        placeholders["count"], f"{context} placeholder count")
    placeholder_entry_size = _gemm_positive_integer(
        placeholders["entry_size_bytes"],
        f"{context} placeholder entry size")
    expected_placeholder = family_contract["placeholder"]
    if (placeholder_count != 1
            or placeholder_entry_size != 8
            or placeholders["entries"] != [expected_placeholder]
            or placeholder_bytes != struct.pack(
                "<II", expected_placeholder["address_offset"],
                expected_placeholder["data_offset"])):
        raise ValueError(f"{context} placeholder contract changed")
    configuration, configuration_bytes = _gemm_captured_bytes(
        shard_path, launch["configuration"], {
            "attribute_count", "attribute_entry_size_bytes", "path",
            "decoded_attributes", "present", "serialized_hex", "sha256",
        }, f"{context} launch configuration")
    attribute_count = _gemm_nonnegative_integer(
        configuration["attribute_count"],
        f"{context} configuration attribute count")
    attribute_entry_size = _gemm_positive_integer(
        configuration["attribute_entry_size_bytes"],
        f"{context} configuration attribute size")
    if (configuration["present"] is not True
            or attribute_count != 2
            or attribute_entry_size != 20
            or configuration_bytes
            != struct.pack(
                "<II12sII12s", 1, family_contract["schem_mode"], bytes(12),
                2, 0, bytes(12))
            or configuration["decoded_attributes"] != [
                {"name": "SCHEM_MODE", "value": family_contract["schem_mode"]},
                {"name": "LOCAL_MEMORY_SIZE", "value": 0},
            ]):
        raise ValueError(f"{context} launch configuration changed")
    tiling_payload = _exact_mapping(result["executor_tiling"], {
        "available", "decoded_fields", "kernel_family", "offset_bytes",
        "pointer_slot_offset_bytes", "serialized_hex", "sha256",
        "size_bytes", "source",
    }, f"{context} executor tiling")
    if (tiling_payload["available"] is not True
            or tiling_payload["decoded_fields"] is not None
            or tiling_payload["source"]
            != "aclrtLaunchKernelWithHostArgs_placeholder_tail"
            or tiling_payload["kernel_family"] != kernel_family
            or tiling_payload["pointer_slot_offset_bytes"]
            != expected_placeholder["address_offset"]
            or tiling_payload["offset_bytes"]
            != expected_placeholder["data_offset"]
            or tiling_payload["size_bytes"]
            != family_contract["tiling_payload_size_bytes"]):
        raise ValueError(f"{context} executor tiling contract changed")
    try:
        payload = bytes.fromhex(tiling_payload["serialized_hex"])
    except ValueError as error:
        raise ValueError(f"{context} executor tiling is not hexadecimal") from error
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    if (len(payload) != family_contract["tiling_payload_size_bytes"]
            or payload_sha256 != tiling_payload["sha256"]):
        raise ValueError(f"{context} executor tiling SHA256 changed")
    require_sha256_hex(
        tiling_payload["sha256"], f"{context} executor tiling SHA256")
    if host_bytes[expected_placeholder["data_offset"]:] != payload:
        raise ValueError(f"{context} placeholder tiling payload changed")
    decoded_family, decoded = decode_matmul_tiling_payload(
        payload, shape, block_dim, kernel_symbol)
    if decoded_family != kernel_family:
        raise AssertionError("Ascend GEMM decoder family changed")
    return {
        "block": [1, 1, 1],
        "cube_math_type": result["cube_math_type"],
        "grid": [block_dim, 1, 1],
        "kernel_family": kernel_family,
        "kernel_symbol": kernel_symbol,
        "operation_id": operation["operation_id"],
        "opaque_tail_hex": decoded["opaque_tail_hex"],
        "payload_hex": payload.hex(),
        "payload_sha256": payload_sha256,
        "payload_size_bytes": len(payload),
        "tiling": decoded["tiling"],
        "workspace_size_bytes": workspace_size,
    }


def _gemm_source_snapshot_sha256(source_provenance, baseline):
    if not isinstance(source_provenance, dict):
        raise TypeError("Ascend GEMM source_provenance must be a mapping")
    required = {"mode", "baseline_commit", "marker_path", "marker_sha256",
                "source_files"}
    if set(source_provenance) != required:
        raise ValueError("Ascend GEMM source_provenance schema changed")
    if source_provenance["baseline_commit"] != baseline:
        raise ValueError("Ascend GEMM source and capture baselines differ")
    source_files = source_provenance["source_files"]
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("Ascend GEMM source provenance has no source files")
    relative_paths = set()
    normalized_files = []
    for index, source_file in enumerate(source_files):
        item = _exact_mapping(
            source_file, {"relative_path", "sha256"},
            f"Ascend GEMM source file {index}")
        relative_path = require_non_empty_string(
            item, "relative_path", f"Ascend GEMM source file {index}")
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Ascend GEMM source path must be repository-relative")
        if relative_path in relative_paths:
            raise ValueError("Ascend GEMM source provenance repeats a file")
        relative_paths.add(relative_path)
        source_sha256 = require_sha256_hex(
            item["sha256"], f"Ascend GEMM source file {relative_path}")
        normalized_files.append({
            "relative_path": relative_path,
            "sha256": source_sha256,
        })
    if relative_paths != GEMM_SOURCE_FILES:
        raise ValueError("Ascend GEMM source provenance file set changed")
    if source_provenance["mode"] == "git_checkout":
        if (source_provenance["marker_path"] is not None
                or source_provenance["marker_sha256"] is not None):
            raise ValueError("Ascend GEMM Git source has a snapshot marker")
        marker_path = None
        marker_sha256 = None
    elif source_provenance["mode"] == "audited_snapshot":
        marker = Path(require_non_empty_string(
            source_provenance, "marker_path", "Ascend GEMM source"))
        if marker.name != ".source_commit":
            raise ValueError("Ascend GEMM source marker path changed")
        marker_path = ".source_commit"
        marker_sha256 = require_sha256_hex(
            source_provenance["marker_sha256"],
            "Ascend GEMM source marker")
    else:
        raise ValueError("Ascend GEMM source provenance mode changed")
    normalized = {
        "mode": source_provenance["mode"],
        "baseline_commit": baseline,
        "marker_path": marker_path,
        "marker_sha256": marker_sha256,
        "source_files": normalized_files,
    }
    return expanded_evidence_sha256(normalized)


def build_gemm_launch_facts(candidate_root):
    """Package complete CANN 8.5.1 GEMM dry-launch shards as schema v4."""

    candidate_root = Path(candidate_root).resolve()
    if not candidate_root.is_dir():
        raise FileNotFoundError(
            f"Ascend GEMM candidate root is missing: {candidate_root}")
    shards = []
    for path in sorted(candidate_root.rglob("result.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if (isinstance(value, dict) and "artifact_type" in value
                and value["artifact_type"] == GEMM_CAPTURE_ARTIFACT_TYPE):
            shards.append((path, value))
    if not shards:
        raise ValueError("Ascend GEMM candidate root contains no bulk shards")

    common = None
    ranges = []
    operations = {}
    mappings = {}
    for shard_path, shard in shards:
        shard = _exact_mapping(
            shard, GEMM_SHARD_FIELDS, "Ascend GEMM bulk shard")
        _reject_gemm_measurement_fields(shard, "Ascend GEMM bulk shard")
        shard_cube_math_type = _gemm_nonnegative_integer(
            shard["cube_math_type"], "Ascend GEMM shard cube_math_type")
        runner_version = _gemm_positive_integer(
            shard["runner_version"], "Ascend GEMM runner version")
        if (shard["schema_version"] != GEMM_CAPTURE_SCHEMA_VERSION
                or shard["hardware_id"] != "ascend_910b"
                or shard["cann_version"] != "8.5.1"
                or shard["mode"] != "bulk_candidate"
                or shard["operator"] != "aclnnMatmul"
                or shard_cube_math_type != 0
                or runner_version != 5):
            raise ValueError("Ascend GEMM bulk shard identity changed")
        if (shard["duration_consumed"] is not False
                or shard["device_kernel_submitted"] is not False
                or shard["execute_api_called"] is not True):
            raise ValueError("Ascend GEMM bulk shard consumed duration")
        if shard["workload_key_fields"] != list(GEMM_WORKLOAD_KEY_FIELDS):
            raise ValueError("Ascend GEMM workload-key schema changed")
        _gemm_nonnegative_integer(
            shard["device_id"], "Ascend GEMM capture device")
        limitations = _gemm_sequence(
            shard["limitations"], "Ascend GEMM capture limitations",
            nonempty=True)
        if limitations != GEMM_CAPTURE_LIMITATIONS:
            raise ValueError("Ascend GEMM capture limitations changed")
        cann_abi_provenance = shard["cann_abi_provenance"]
        if cann_abi_provenance != matmul_tiling_abi_provenance():
            raise ValueError("Ascend GEMM CANN ABI provenance changed")
        audit = _exact_mapping(shard["build_and_abi_audit"], {
            "helper_compile_count", "helper_path", "helper_sha256",
            "interposed_launch_apis", "interposer_compile_count",
            "interposer_path", "interposer_sha256",
            "nnopbase_imported_launch_apis",
        }, "Ascend GEMM build and ABI audit")
        helper_compile_count = _gemm_positive_integer(
            audit["helper_compile_count"], "Ascend GEMM helper compile count")
        interposer_compile_count = _gemm_positive_integer(
            audit["interposer_compile_count"],
            "Ascend GEMM interposer compile count")
        if (helper_compile_count != 1
                or interposer_compile_count != 1
                or audit["interposed_launch_apis"] != GEMM_LAUNCH_APIS
                or audit["nnopbase_imported_launch_apis"]
                != ["aclrtLaunchKernelWithHostArgs"]):
            raise ValueError("Ascend GEMM build and ABI audit changed")
        for kind in ("helper", "interposer"):
            artifact_path = _gemm_artifact_path(
                shard_path, audit[f"{kind}_path"],
                f"Ascend GEMM {kind} artifact")
            if (require_sha256_hex(
                    audit[f"{kind}_sha256"],
                    f"Ascend GEMM {kind} SHA256") != file_sha256(artifact_path)):
                raise ValueError(f"Ascend GEMM {kind} artifact SHA256 changed")
        candidate = _exact_mapping(shard["workload_candidate"], {
            "candidate_scope", "coverage_claim", "final_exact_workload_universe",
            "input_manifest", "path", "sha256",
        }, "Ascend GEMM workload candidate")
        if (candidate["candidate_scope"] != GEMM_CANDIDATE_SCOPE
                or candidate["coverage_claim"]
                != GEMM_COVERAGE_CLAIM
                or candidate["final_exact_workload_universe"] is not True):
            raise ValueError("Ascend GEMM candidate scope changed")
        candidate_sha256 = require_sha256_hex(
            candidate["sha256"], "Ascend GEMM candidate SHA256")
        input_manifest = _validate_gemm_candidate_manifest(
            candidate["input_manifest"])
        if expanded_evidence_sha256(input_manifest) != candidate_sha256:
            raise ValueError("Ascend GEMM embedded candidate SHA256 changed")
        candidate_manifest_path = _gemm_candidate_manifest_path(
            candidate["path"])
        baseline = require_non_empty_string(
            shard, "baseline_commit", "Ascend GEMM shard")
        if re.fullmatch(r"[0-9a-f]{40}", baseline) is None:
            raise ValueError("Ascend GEMM capture baseline is not a commit")
        source_snapshot_sha256 = _gemm_source_snapshot_sha256(
            shard["source_provenance"], baseline)
        full_operation_count = _gemm_positive_integer(
            shard["full_input_operation_count"],
            "Ascend GEMM full operation count")
        full_workload_count = _gemm_positive_integer(
            shard["full_input_workload_key_count"],
            "Ascend GEMM full workload count")
        if (full_workload_count != input_manifest["semantic_key_count"]
                or full_operation_count
                != input_manifest["physical_operation_count"]):
            raise ValueError(
                "Ascend GEMM full input counts differ from the manifest")
        shard_common = {
            "capture_baseline_commit": baseline,
            "candidate_manifest_path": candidate_manifest_path,
            "candidate_manifest_sha256": candidate_sha256,
            "source_snapshot_sha256": source_snapshot_sha256,
            "input_manifest": input_manifest,
            "full_input_operation_count": full_operation_count,
            "full_input_workload_key_count": full_workload_count,
            "cann_abi_provenance": cann_abi_provenance,
            "domain_sha256": input_manifest["hardware_domains"][
                "ascend_910b"]["domain_sha256"],
        }
        if common is None:
            common = shard_common
        elif shard_common != common:
            raise ValueError(
                "Ascend GEMM bulk shards disagree on source or candidate")
        selected_range = _exact_mapping(
            shard["selected_operation_range"], {"end_index", "start_index"},
            "Ascend GEMM selected operation range")
        start = selected_range["start_index"]
        end = selected_range["end_index"]
        if (isinstance(start, bool) or not isinstance(start, int)
                or isinstance(end, bool) or not isinstance(end, int)
                or start < 0 or end <= start):
            raise ValueError("Ascend GEMM bulk shard range is invalid")
        operations_value = _gemm_sequence(
            shard["operations"], "Ascend GEMM shard operations")
        mappings_value = _gemm_sequence(
            shard["workload_mappings"], "Ascend GEMM workload mappings")
        selected_count = _gemm_positive_integer(
            shard["selected_operation_count"],
            "Ascend GEMM selected operation count")
        operation_count = _gemm_positive_integer(
            shard["operation_count"], "Ascend GEMM operation count")
        workload_count = _gemm_positive_integer(
            shard["workload_key_count"], "Ascend GEMM workload count")
        if (selected_count != end - start
                or operation_count != end - start
                or len(operations_value) != end - start
                or workload_count != len(mappings_value)):
            raise ValueError("Ascend GEMM bulk shard counts are inconsistent")
        ranges.append((start, end))
        result_identity_by_operation = {}
        declared_workloads_by_operation = {}
        shape_by_operation = {}
        for offset, operation_value in enumerate(operations_value):
            operation = _exact_mapping(
                operation_value, GEMM_OPERATION_FIELDS,
                f"Ascend GEMM operation {start + offset}")
            operation_id = operation["operation_id"]
            if operation_id != f"operation_{start + offset:06d}":
                raise ValueError("Ascend GEMM operation sequence changed")
            if operation_id in operations:
                raise ValueError(f"duplicate Ascend GEMM operation {operation_id}")
            operation_cube_math_type = _gemm_nonnegative_integer(
                operation["cube_math_type"],
                f"Ascend GEMM {operation_id} cube_math_type")
            if operation["dtype"] != "bf16" or operation_cube_math_type != 0:
                raise ValueError("Ascend GEMM operation contract changed")
            result_identity = _exact_mapping(
                operation["result"], {"path", "sha256"},
                f"Ascend GEMM {operation_id} result")
            require_sha256_hex(
                result_identity["sha256"],
                f"Ascend GEMM {operation_id} result SHA256")
            result_path = _gemm_artifact_path(
                shard_path, result_identity["path"],
                f"Ascend GEMM {operation_id} result")
            operations[operation_id] = _gemm_operation_entry(
                operation, result_path, shard_path)
            result_identity_by_operation[operation_id] = result_identity
            workload_keys = operation["workload_keys"]
            if (not isinstance(workload_keys, list) or not workload_keys
                    or len(workload_keys) != len(set(workload_keys))):
                raise ValueError(
                    "Ascend GEMM operation workload keys are invalid")
            declared_workloads_by_operation[operation_id] = set(workload_keys)
            shape_by_operation[operation_id] = {
                name: operation[name] for name in ("m", "n", "k")}
        mapped_workloads_by_operation = {
            operation_id: set() for operation_id in declared_workloads_by_operation}
        for mapping_index, mapping_value in enumerate(mappings_value):
            mapping = _exact_mapping(
                mapping_value, {"operation_id", "result", "workload_key"},
                f"Ascend GEMM mapping {mapping_index}")
            workload_key = mapping["workload_key"]
            operation_id = mapping["operation_id"]
            if workload_key in mappings:
                raise ValueError("duplicate Ascend GEMM workload mapping")
            if operation_id not in operations:
                raise ValueError("Ascend GEMM workload maps outside its shard")
            if mapping["result"] != result_identity_by_operation[operation_id]:
                raise ValueError("Ascend GEMM mapping result identity changed")
            values = _gemm_workload_values(workload_key)
            entry = operations[operation_id]
            operation_shape = shape_by_operation[operation_id]
            if (values["dtype"] != "bf16" or values["b"] != 1
                    or values["m"] != operation_shape["m"]
                    or values["n"] != operation_shape["n"]
                    or values["k"] != operation_shape["k"]
                    or values["a_layout"] != "row_major"
                    or values["b_layout"] != "row_major"
                    or values["c_layout"] != "row_major"
                    or values["trans_a"] is not False
                    or values["trans_b"] is not True
                    or values["epilogue"] != "none"
                    or values["has_bias"] is not False):
                raise ValueError("Ascend GEMM workload semantics changed")
            mappings[workload_key] = entry
            mapped_workloads_by_operation[operation_id].add(workload_key)
        if mapped_workloads_by_operation != declared_workloads_by_operation:
            raise ValueError("Ascend GEMM shard workload mappings are incomplete")

    if common is None:
        raise AssertionError("Ascend GEMM candidate common identity missing")
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError("Ascend GEMM shard ranges contain a gap or overlap")
        cursor = end
    if (cursor != common["full_input_operation_count"]
            or len(operations) != common["full_input_operation_count"]
            or len(mappings) != common["full_input_workload_key_count"]):
        raise ValueError("Ascend GEMM shards do not provide complete coverage")
    candidate_keys = common["input_manifest"]["workload_keys"]
    if candidate_keys != sorted(mappings) or len(candidate_keys) != len(mappings):
        raise ValueError("Ascend GEMM mappings do not cover the candidate")
    entries = {key: mappings[key] for key in sorted(mappings)}
    return {
        "schema_version": GEMM_FACT_SCHEMA_VERSION,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "generated_by": "scripts/launch_data/ascend_910b_launch_generator.py",
        "source": common["candidate_manifest_path"],
        "candidate_manifest_path": common["candidate_manifest_path"],
        "candidate_manifest_sha256": common["candidate_manifest_sha256"],
        "source_snapshot_sha256": common["source_snapshot_sha256"],
        "capture_baseline_commit": common["capture_baseline_commit"],
        "tiling_abi": common["cann_abi_provenance"],
        "domain_sha256": common["domain_sha256"],
        "semantic_key_count": common["full_input_workload_key_count"],
        "physical_operation_count": common["full_input_operation_count"],
        "workload_key_fields": list(GEMM_WORKLOAD_KEY_FIELDS),
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------
def _tuple_key(values):
    return "[" + ", ".join(str(value) for value in values) + "]"




def _write_json_if_changed(path, payload):
    text = compact_json_text(payload)
    if path.exists() and path.read_text() == text:
        return False
    path.write_text(text)
    return True


def emit(flash_attn):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    compact = compact_ascend_910b_flash_attn(flash_attn)
    flash_changed = _write_json_if_changed(FLASH_ATTN_OUT, compact)
    print(
        f"[gen] {'wrote' if flash_changed else 'verified'} {FLASH_ATTN_OUT}",
        file=sys.stderr)
    print(f"[gen]   FlashAttention entries: {len(flash_attn['entries'])}",
          file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the generated launch data after validation",
    )
    parser.add_argument(
        "--swiglu-observation-root",
        type=Path,
        help=(
            "directory containing gap-free exact-domain SwiGLU bulk shards"),
    )
    parser.add_argument(
        "--gemm-candidate-root",
        type=Path,
        help="directory containing a complete Ascend GEMM dry-launch capture",
    )
    parser.add_argument(
        "--gemm-expanded-evidence",
        type=Path,
        help=(
            "validated complete Ascend GEMM dry-launch evidence JSON; this "
            "is projected into the single canonical selector CSV"),
    )
    args = parser.parse_args(argv)

    flash_attn_json = _load("flash_attn.json")
    flash_attn = build_flash_attn_launch_facts(flash_attn_json)
    if not flash_attn["entries"]:
        raise ValueError(
            "FlashAttention dataset produced no launch-fact entries")
    if args.write:
        emit(flash_attn)
    summary = {
        "flash_attn_entries": len(flash_attn["entries"]),
    }
    if args.swiglu_observation_root is not None:
        swiglu = build_swiglu_launch_facts(args.swiglu_observation_root)
        summary["swiglu_entries"] = len(swiglu["entries"])
        if args.write:
            compact = compact_ascend_910b_swiglu(swiglu)
            changed = _write_json_if_changed(SWIGLU_OUT, compact)
            print(
                f"[gen] {'wrote' if changed else 'verified'} {SWIGLU_OUT}",
                file=sys.stderr)
            print(
                f"[gen]   SwiGLU entries: {len(swiglu['entries'])}",
                file=sys.stderr)
    if (args.gemm_candidate_root is not None
            and args.gemm_expanded_evidence is not None):
        raise ValueError("choose one Ascend GEMM evidence input")
    if args.gemm_candidate_root is not None:
        gemm = build_gemm_launch_facts(args.gemm_candidate_root)
    elif args.gemm_expanded_evidence is not None:
        gemm = json.loads(args.gemm_expanded_evidence.read_text(encoding="utf-8"))
        _reject_gemm_measurement_fields(
            gemm, "Ascend GEMM expanded launch evidence")
        if (gemm["schema_version"] != 4
                or gemm["hardware_id"] != "ascend_910b"
                or gemm["software_revision"] != "cann_8.5.1"
                or gemm["domain_sha256"] != GEMM_EXACT_DOMAIN_SHA256
                or gemm["semantic_key_count"]
                != GEMM_EXACT_SEMANTIC_KEY_COUNT
                or gemm["physical_operation_count"]
                != GEMM_EXACT_PHYSICAL_OPERATION_COUNT):
            raise ValueError("Ascend GEMM expanded launch evidence identity changed")
        if not isinstance(gemm["entries"], dict) or len(gemm["entries"]) != (
                GEMM_EXACT_SEMANTIC_KEY_COUNT):
            raise ValueError("Ascend GEMM expanded launch evidence coverage changed")
    else:
        gemm = None
    if gemm is not None:
        summary["gemm_entries"] = len(gemm["entries"])
        if args.write:
            entries = project_ascend_910b_gemm(gemm)
            evidence_sha256 = expanded_evidence_sha256(gemm)
            csv_text, manifest = build_gemm_launch_selection_artifact(
                entries,
                hardware_id="ascend_910b",
                software_revision="cann_8.5.1",
                generated_by=(
                    "scripts/launch_data/ascend_910b_launch_generator.py"),
                evidence_sha256=evidence_sha256,
                evidence_sources=[
                    {
                        "kind": "expanded_evidence",
                        "sha256": evidence_sha256,
                    },
                    {
                        "kind": "workload_candidate_manifest",
                        "sha256": gemm["candidate_manifest_sha256"],
                    },
                    {
                        "kind": "source_snapshot",
                        "sha256": gemm["source_snapshot_sha256"],
                    },
                ],
                capture_baseline_commit=gemm["capture_baseline_commit"],
                domain_sha256=gemm["domain_sha256"],
                repository_root=ROOT,
            )
            previous_csv = (
                decompress_gemm_launch_csv(
                    GEMM_CSV_OUT.read_bytes()).decode("utf-8")
                if GEMM_CSV_OUT.is_file() else None)
            manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            previous_manifest = (
                GEMM_MANIFEST_OUT.read_text(encoding="utf-8")
                if GEMM_MANIFEST_OUT.is_file() else None)
            write_gemm_launch_selection_artifact(
                OUT_DIR, csv_text, manifest)
            changed = (
                previous_csv != csv_text or previous_manifest != manifest_text)
            print(
                f"[gen] {'wrote' if changed else 'verified'} {GEMM_CSV_OUT}",
                file=sys.stderr)
            print(
                f"[gen]   GEMM entries: {len(gemm['entries'])}",
                file=sys.stderr)
    elif (args.swiglu_observation_root is None and not args.write
          and (GEMM_CSV_OUT.is_file() or GEMM_MANIFEST_OUT.is_file())):
        selection = load_gemm_launch_selection_files(
            GEMM_CSV_OUT,
            GEMM_MANIFEST_OUT,
            hardware_id="ascend_910b",
            software_revision="cann_8.5.1",
        )
        summary["gemm_entries"] = len(selection)
    print(json.dumps(summary, sort_keys=True))
    return 0
