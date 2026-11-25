"""Huawei 910B GEMM launch selection from exact CANN launch facts."""

import hashlib
import json
import math
import re
import struct
from functools import lru_cache
from typing import Mapping, Optional

from src.kernel.launch_config import (
    CannGemmLaunchProgram,
    CannGemmTilePipelineProgram,
    LaunchConfig,
    TilingSpec,
)
from src.kernel.launch_config.cann_gemm_schedule import (
    derive_cann_matmul_v3_l2_region_schedule,
)
from src.kernel.launch_config.gemm_workload import gemm_launch_selection_key
from src.kernel.launch_config.gemm_launch_selection import (
    CannGemmLaunchProgramSelection,
    GemmLaunchSelection,
    load_gemm_launch_selection,
)
from src.kernel.launch_config.selection_primitives import (
    launch_policy, require_dtype, require_shape,
    require_variant)


MATMUL_V2_KERNEL_PATTERN = re.compile(
    r"MatMulV2_ND_ND_FP16_FP16_false_true_all_([0-9]+)")
MATMUL_V3_KERNEL_NAME = (
    "MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance")
MATMUL_V3_KERNEL_PATTERN = re.compile(
    re.escape(MATMUL_V3_KERNEL_NAME) + r"_([0-9]+)")
MATMUL_V2_TILING_PAYLOAD_SIZE_BYTES = 160
MATMUL_V2_TILING_FIELD_COUNT = 38
MATMUL_V2_TILING_PREFIX_SIZE_BYTES = 152
MATMUL_V2_TILING_OPAQUE_TAIL_SIZE_BYTES = 8
MATMUL_V2_KERNEL_TILING_KEY_COUNT = 441
MATMUL_V2_KERNEL_TILING_KEYS_SHA256 = (
    "85a1cd131480205c4cbb489800047b385269a9b80f5ba5b558072eee9503b2d7")
MATMUL_V3_TILING_PAYLOAD_SIZE_BYTES = 288
MATMUL_V3_TCUBE_TILING_SIZE_BYTES = 200
MATMUL_V3_LEGACY_EXTENSION_FIELD_COUNT = 17
MATMUL_V3_TILING_DECODED_SIZE_BYTES = 268
MATMUL_V3_TILING_OPAQUE_TAIL_SIZE_BYTES = 20
MATMUL_V3_SUPPORTED_TILING_KEY = 65536
MATMUL_V3_KERNEL_TILING_KEY_COUNT = 20
MATMUL_V3_KERNEL_TILING_KEYS_SHA256 = (
    "1fc2464b7a7493a6fff4eb865c8e837f6b7ef4a2676be6b0c6ed4ba285b3ae35")
if (MATMUL_V3_TILING_DECODED_SIZE_BYTES
        + MATMUL_V3_TILING_OPAQUE_TAIL_SIZE_BYTES
        != MATMUL_V3_TILING_PAYLOAD_SIZE_BYTES):
    raise RuntimeError("pinned MatMulV3 tiling payload layout is inconsistent")
GEMM_EXACT_SEMANTIC_KEY_COUNT = 140544
GEMM_EXACT_PHYSICAL_OPERATION_COUNT = 140544
GEMM_EXACT_DOMAIN_SHA256 = (
    "8a935c5226fc5b719fac1ab0e8dd78df130b9deeaa0725c9c936e5eeb9d14bee")
_CUBE_EXTENT = 16
_BF16_BYTES = 2
_FP32_ACCUMULATOR_BYTES = 4
_L0A_BYTES = 65_536
_L0B_BYTES = 65_536
_L0C_BYTES = 131_072
_OPERAND_BUFFER_DEPTH = 2
_ASW_BASE_M = 128
_ASW_BASE_N = 256
# CANN 8.5.1's legacy MatMul tiler emits two AL0/BL0 operand buffers.  The
# source's L0C capacity guard independently reduces only CL0 to one buffer
# when a second accumulator tile does not fit.
_MATMUL_V2_AL0_BL0_PIPELINE_DEPTH = 2
_PREFERRED_OUTPUT_EXTENT = 256
_MAX_K_TILE = 128
_DETERMINISTIC_SPLIT_K_THRESHOLD = 27392
_DETERMINISTIC_SPLIT_K_BASE = 128
_DETERMINISTIC_SPLIT_K_STEP_M = 3
_DETERMINISTIC_SPLIT_K_STEP_N = 1
_DETERMINISTIC_SPLIT_K_DEPTH_A1 = 9
_DETERMINISTIC_SPLIT_K_DEPTH_B1 = 6
_DETERMINISTIC_SPLIT_K_L2_NUMERATOR = 7
_DETERMINISTIC_SPLIT_K_L2_DENOMINATOR = 10
_DETERMINISTIC_SPLIT_K_SINGLE_CORE_N_MINIMUM = 512
# CANN 8.5.1's deterministic split-K kernel selects NBuffer33 only on its
# non-L2-split MK path when the logical M extent stays within two 128-row base
# tiles.  The L2-split implementation constructs ordinary MK/NK MatmulImpl
# objects even for the same small M geometry.  NBuffer33 owns three A1, two
# B1 and two L0A/L0B credits; those credits must therefore not be inferred
# from M alone.
_DETERMINISTIC_SPLIT_K_NBUFFER33_MAX_ROWS = 2 * _DETERMINISTIC_SPLIT_K_BASE

# CANN 8.5.1 legacy key-65632 launch-only captures.  These are discrete
# source-program facts: no duration or accuracy result participates.  The
# single-core Split-K kernel assigns one complete output macro block to every
# AIC, serially replays the listed K chunk, and hands the final FP32 workspace
# stripe to its two paired AIVs.  Effective M is the flattened B*M extent.
_SINGLE_CORE_SPLIT_K_PROGRAMS = {
    256: (22, 128, 768, 512, 8, 8, 1, 1),
    512: (22, 256, 768, 512, 8, 8, 2, 1),
    2048: (24, 256, 2816, 512, 8, 8, 2, 1),
    4096: (24, 512, 2816, 384, 9, 6, 3, 1),
    8192: (24, 1024, 2816, 384, 9, 6, 3, 1),
    16384: (24, 2048, 2816, 384, 9, 6, 3, 1),
    32768: (24, 4096, 2816, 384, 9, 6, 3, 1),
}


def derive_cann_gemm_operand_pipeline_depth(
        tile_m: int, tile_n: int, tile_k: int) -> int:
    """Return the common legal A/B ping-pong depth for one Cube macro tile.

    This is deliberately a GEMM tile property, not a MatMulV2/V3 selector
    property.  Both operands reside in independent L0A/L0B banks; every
    retained macro tile must support the two-slot ping-pong program used by
    the analytical Forward.  The source-defined legacy V2 tiler proves that
    program directly, while V3 and analytically selected tiles derive the
    same conservative depth from the published bank capacities.  Do not
    promote spare capacity beyond two slots without a source scheduling
    contract.
    """

    for name, value in (("tile_m", tile_m), ("tile_n", tile_n),
                        ("tile_k", tile_k)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"910B GEMM {name} must be an integer")
        if value <= 0:
            raise ValueError(f"910B GEMM {name} must be positive")
    a_tile_bytes = tile_m * tile_k * _BF16_BYTES
    b_tile_bytes = tile_k * tile_n * _BF16_BYTES
    bank_depth = min(_L0A_BYTES // a_tile_bytes, _L0B_BYTES // b_tile_bytes)
    if bank_depth < _OPERAND_BUFFER_DEPTH:
        raise ValueError(
            "910B GEMM macro tile cannot sustain the required two-slot "
            "L0A/L0B operand program")
    return _OPERAND_BUFFER_DEPTH


def derive_cann_matmul_v3_asw_macro_tile(
        n: int, base_k: int, l1_size_bytes: int) -> tuple[int, int, int]:
    """Derive CANN ASW's macro-tile orientation from its L1 fit rule.

    The two legal orientations have the same output and operand-buffer
    footprint.  CANN selects the M-major orientation when one complete
    N-by-baseK BF16 slab fits strictly below half of per-AIC L1; otherwise it
    retains the N-major orientation.  This is a capacity decision shared by
    analytical and observed-launch projection, not a kernel-name dispatch.
    """

    for name, value in (
            ("n", n), ("base_k", base_k),
            ("l1_size_bytes", l1_size_bytes)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"910B GEMM ASW {name} must be an integer")
        if value <= 0:
            raise ValueError(f"910B GEMM ASW {name} must be positive")
    complete_n_slab_bytes = n * base_k * _BF16_BYTES
    if complete_n_slab_bytes < l1_size_bytes // 2:
        return _ASW_BASE_N, _ASW_BASE_M, base_k
    return _ASW_BASE_M, _ASW_BASE_N, base_k


_MATMUL_V2_BASE_TILING_KEYS = (
    98304, 98305, 98312, 98313, 98324, 98325, 98336, 98337, 98344,
    98345, 98356, 98357, 98370, 98371, 98378, 98379, 98384, 98385,
    98386, 98387, 98388, 98389, 98432, 98433, 98440, 98441, 98452,
    98453, 98464, 98465, 98472, 98473, 98484, 98485, 98498, 98499,
    98506, 98507, 98512, 98513, 98514, 98515, 98516, 98517, 98816,
    98817, 98824, 98825, 98836, 98837, 98848, 98849, 98856, 98857,
    98868, 98869, 98882, 98883, 98890, 98891, 98896, 98897, 98898,
    98899, 98900, 98901, 98944, 98945, 98952, 98953, 98964, 98965,
    98976, 98977, 98984, 98985, 98996, 98997, 99010, 99011, 99018,
    99019, 99024, 99025, 99026, 99027, 99028, 99029,
)
_MATMUL_V2_TILING_KEY_OFFSETS = (0, 1048576, 2097152, 3145728, 16777216)
MATMUL_V2_KERNEL_TILING_KEYS = frozenset(
    {base + offset
     for base in _MATMUL_V2_BASE_TILING_KEYS
     for offset in _MATMUL_V2_TILING_KEY_OFFSETS}
    | {360489})
_matmul_v2_tiling_keys_payload = json.dumps(
    sorted(MATMUL_V2_KERNEL_TILING_KEYS),
    separators=(",", ":"), ensure_ascii=True).encode("utf-8")
if (len(MATMUL_V2_KERNEL_TILING_KEYS) != MATMUL_V2_KERNEL_TILING_KEY_COUNT
        or hashlib.sha256(_matmul_v2_tiling_keys_payload).hexdigest()
        != MATMUL_V2_KERNEL_TILING_KEYS_SHA256):
    raise RuntimeError("pinned MatMulV2 kernel tiling-key set is inconsistent")
del _matmul_v2_tiling_keys_payload

MATMUL_V3_KERNEL_TILING_KEYS = frozenset({
    0, 2, 32, 48, 96, 4098, 8240, 65536, 65537, 65538, 65568,
    65584, 65600, 65616, 65632, 69634, 73730, 73776, 131074, 1114112,
})
_matmul_v3_tiling_keys_payload = json.dumps(
    sorted(MATMUL_V3_KERNEL_TILING_KEYS),
    separators=(",", ":"), ensure_ascii=True).encode("utf-8")
if (len(MATMUL_V3_KERNEL_TILING_KEYS) != MATMUL_V3_KERNEL_TILING_KEY_COUNT
        or hashlib.sha256(_matmul_v3_tiling_keys_payload).hexdigest()
        != MATMUL_V3_KERNEL_TILING_KEYS_SHA256):
    raise RuntimeError("pinned MatMulV3 kernel tiling-key set is inconsistent")
del _matmul_v3_tiling_keys_payload


def matmul_v2_tiling_abi_provenance():
    """Return the pinned CANN sources that define this payload projection."""

    return {
        "abi": "cann_8.5.1_matmul_v2_tiling_data",
        "cann_version": "8.5.1",
        "decoded_prefix": {
            "encoding": "little_endian_signed_int32",
            "field_count": MATMUL_V2_TILING_FIELD_COUNT,
            "size_bytes": MATMUL_V2_TILING_PREFIX_SIZE_BYTES,
        },
        "opaque_tail_size_bytes": MATMUL_V2_TILING_OPAQUE_TAIL_SIZE_BYTES,
        "kernel_tiling_key_count": MATMUL_V2_KERNEL_TILING_KEY_COUNT,
        "kernel_tiling_keys_sha256": MATMUL_V2_KERNEL_TILING_KEYS_SHA256,
        "source_artifacts": [
            {"path": "/usr/local/Ascend/cann-8.5.1/opp/version.info",
             "sha256": "49437edb9a235b28a87d07d2a2ad189022b6d246dc7c4cbcd013d2b7ecaf838b"},
            {"path": ("/usr/local/Ascend/cann-8.5.1/python/site-packages/"
                      "tbe/common/utils/op_tiling.py"),
             "sha256": "60ce20a9d26ecb7a7bb761362d7670466a748376099ddda0d5790e0d49956d31"},
            {"path": ("/usr/local/Ascend/cann-8.5.1/python/site-packages/"
                      "tbe/dsl/unify_schedule/gemm_tiling_util.py"),
             "sha256": "05e033838e714b2648f75f09ae193b73206c825712a55fa5d70c9b7ff1029b39"},
            {"path": ("/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/"
                      "ai_core/tbe/impl/ops_legacy/dynamic/"
                      "batch_matmul_v2.py"),
             "sha256": "fc2725710d154cdf6cae28cb42ea9838c4efa6ad3a086826fc19e4181d8961e6"},
            {"path": ("/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/"
                      "ai_core/tbe/kernel/ascend910b/ops_legacy/mat_mul/"
                      "MatMulV2_ND_ND_FP16_FP16_false_true_all.json"),
             "sha256": "95885de8b6e57ae6b828581f001dc90fa84b9a1341d1cd4a9fdccf81d92197f0"},
            {"path": ("/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/"
                      "ai_core/tbe/kernel/ascend910b/ops_legacy/mat_mul/"
                      "MatMulV2_ND_ND_FP16_FP16_false_true_all.o"),
             "sha256": "87bed199882fee1f7b890bce46fd78059c0d465344757e1227bc2c25071b5d0b"},
        ],
    }


_MATMUL_V3_SOURCE_ARTIFACTS = (
    ("/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/"
     "kernel/ascend910b/ops_nn/mat_mul_v3/"
     "MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance.json",
     "0d252989ae68724145b139d747dc49d9416d8a929cf422d0790531a7904008a7"),
    ("/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/"
     "kernel/ascend910b/ops_nn/mat_mul_v3/"
     "MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance.o",
     "53f6b25ef79582e2e49f80ec9d68df1ef58f05856d4b9ba04d0fd2ce39d883b9"),
    ("/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/impl/"
     "ops_nn/ascendc/mat_mul_v3/arch35/mat_mul_tiling_data.h",
     "f4ea1cc98a05dc5f4b313b3beb41c131fa440a314affd7fa713aa9f2ae88780a"),
    ("/usr/local/Ascend/cann-8.5.1/aarch64-linux/ascendc/include/"
     "highlevel_api/kernel_tiling/kernel_tiling.h",
     "0f50fe7d81d4e01f80b62c19d816ad28151ac1502f708b70454efc8599ddaf44"),
    ("/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/impl/"
     "ops_nn/ascendc/mat_mul_v3/arch35/mat_mul_tiling_key.h",
     "9ccbd8a9548310b5b4551e25128f0641b531c38ec3292f81dea4329454b0cf46"),
    ("/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/impl/"
     "ops_nn/dynamic/mat_mul_v3.py",
     "89a51f41fbe3d02720172e0d5a2225b21f6409f36fd80001591244dd7040d32c"),
    ("/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/impl/"
     "ops_transformer/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/"
     "matmul_v3_asw_full_load_tiling.cpp",
     "15a2e1ebf77c79e730dc2c4ef18bb49ed1681eca3b62d987b0dc413646bb88ad"),
    ("/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/impl/"
     "ops_transformer/ascendc/3rd/mat_mul_v3/op_host/op_tiling/arch35/"
     "matmul_v3_base_tiling_advanced.h",
     "74b677d7476fbfe499bd981a1e5343363af2653b6b56d438680165e8cced5eba"),
)


def matmul_tiling_abi_provenance():
    """Return the exact family-aware ABI identity emitted by runner v5."""

    v2 = matmul_v2_tiling_abi_provenance()
    v2_family = {
        "kernel_name": "MatMulV2_ND_ND_FP16_FP16_false_true_all",
        "kernel_symbol_prefix": "MatMulV2_ND_ND_FP16_FP16_false_true_all_",
        "tiling_payload_size_bytes": MATMUL_V2_TILING_PAYLOAD_SIZE_BYTES,
        "kernel_tiling_key_count": MATMUL_V2_KERNEL_TILING_KEY_COUNT,
        "kernel_tiling_keys_sha256": MATMUL_V2_KERNEL_TILING_KEYS_SHA256,
        "kernel_tiling_keys_canonicalization": (
            "numeric_sort_compact_json_no_trailing_newline"),
        "decoded_prefix": v2["decoded_prefix"],
        "opaque_tail_size_bytes": MATMUL_V2_TILING_OPAQUE_TAIL_SIZE_BYTES,
        "source_artifacts": v2["source_artifacts"],
    }
    version_artifact = v2["source_artifacts"][0]
    v3_artifacts = [
        {"path": path, "sha256": sha256}
        for path, sha256 in _MATMUL_V3_SOURCE_ARTIFACTS
    ]
    v3_family = {
        "kernel_name": MATMUL_V3_KERNEL_NAME,
        "kernel_symbol_prefix": MATMUL_V3_KERNEL_NAME + "_",
        "tiling_payload_size_bytes": MATMUL_V3_TILING_PAYLOAD_SIZE_BYTES,
        "kernel_tiling_key_count": MATMUL_V3_KERNEL_TILING_KEY_COUNT,
        "kernel_tiling_keys_sha256": MATMUL_V3_KERNEL_TILING_KEYS_SHA256,
        "kernel_tiling_keys_canonicalization": (
            "numeric_sort_compact_json_no_trailing_newline"),
        "decoded_prefix": {
            "encoding": "little_endian_signed_int32",
            "size_bytes": MATMUL_V3_TILING_DECODED_SIZE_BYTES,
            "tcube_tiling_size_bytes": 200,
            "tcube_projected_field_count": 15,
            "legacy_extension_offset_bytes": 200,
            "legacy_extension_field_count": 17,
        },
        "opaque_tail_size_bytes": MATMUL_V3_TILING_OPAQUE_TAIL_SIZE_BYTES,
        "source_artifacts": [version_artifact, *v3_artifacts],
    }
    return {
        "abi": "cann_8.5.1_matmul_tiling_families",
        "cann_version": "8.5.1",
        "kernel_families": {
            "matmul_v2": v2_family,
            "matmul_v3": v3_family,
        },
        "source_artifacts": [*v2["source_artifacts"], *v3_artifacts],
    }


def _positive_abi_integer(value, context):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if value <= 0:
        raise ValueError(f"{context} must be positive")
    return value


def _nonnegative_abi_integer(value, context):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if value < 0:
        raise ValueError(f"{context} must be non-negative")
    return value


def decode_matmul_v2_tiling_payload(
        payload: bytes, shape: Mapping[str, int], block_dim: int,
        kernel_symbol: str):
    """Project only source-defined tile and split fields from a payload."""

    if not isinstance(payload, bytes):
        raise TypeError("Ascend GEMM tiling payload must be bytes")
    if len(payload) != MATMUL_V2_TILING_PAYLOAD_SIZE_BYTES:
        raise ValueError(
            "Ascend GEMM tiling payload must contain exactly "
            f"{MATMUL_V2_TILING_PAYLOAD_SIZE_BYTES} bytes")
    match = MATMUL_V2_KERNEL_PATTERN.fullmatch(kernel_symbol)
    if match is None:
        raise ValueError(
            "Ascend GEMM requires an evidence-backed MatMulV2 kernel")
    tiling_key = _positive_abi_integer(
        int(match.group(1)), "Ascend GEMM tiling key")
    if tiling_key not in MATMUL_V2_KERNEL_TILING_KEYS:
        raise ValueError(
            "Ascend GEMM tiling key is absent from the pinned 441-kernel JSON")
    block_dim = _positive_abi_integer(block_dim, "Ascend GEMM block_dim")
    k = _positive_abi_integer(shape["k"], "Ascend GEMM shape.k")
    m = _positive_abi_integer(shape["m"], "Ascend GEMM shape.m")
    n = _positive_abi_integer(shape["n"], "Ascend GEMM shape.n")
    words = struct.unpack_from(
        f"<{MATMUL_V2_TILING_FIELD_COUNT}i", payload)
    if words[:3] != (k, m, n):
        raise ValueError(
            "Ascend GEMM tiling payload K/M/N differs from its workload")
    aligned_extents = tuple(
        (dimension + 15) // 16 for dimension in (k, m, n))
    if words[3:6] != aligned_extents:
        raise ValueError(
            "Ascend GEMM tiling payload aligned K/M/N extents changed")
    if words[6] != 1:
        raise ValueError("Ascend GEMM batch_single_core changed")
    for index, name in (
            (7, "m_single_core"), (8, "n_single_core"),
            (9, "batch_dim"), (10, "n_dim"), (11, "m_dim"),
            (12, "k_dim"), (13, "m_al1"), (14, "n_bl1"),
            (15, "cub_n1"), (16, "m_l0"), (17, "k_l0"),
            (18, "n_ub_l0_time"), (21, "kal1_factor"),
            (22, "kbl1_factor"), (23, "kal1_16"), (24, "kbl1_16")):
        _positive_abi_integer(words[index], f"Ascend GEMM tiling {name}")
    for index, name in (
            (33, "al1_db"), (34, "bl1_db"), (35, "l0c_db"),
            (36, "l2_cache_flag"), (37, "close_k_shift")):
        _nonnegative_abi_integer(
            words[index], f"Ascend GEMM tiling {name}")
    if words[9] * words[10] * words[11] * words[12] != block_dim:
        raise ValueError(
            "Ascend GEMM tiling block dimensions do not reproduce block_dim")
    if words[30] != 0:
        raise ValueError("Ascend GEMM tiling payload unexpectedly enables bias")
    if words[32] != 1:
        raise ValueError("Ascend GEMM tiling payload is not BF16")
    tile_m = words[16] * _CUBE_EXTENT
    tile_n = words[15] * words[18] * _CUBE_EXTENT
    tile_k = words[17] * _CUBE_EXTENT
    worker_grid_n = words[10]
    worker_grid_m = words[11]
    worker_grid_k = words[12]
    worker_m = words[7] * words[13] * tile_m
    worker_n = words[8] * words[14] * tile_n
    a_k_schedule_span = words[21] * words[23] * _CUBE_EXTENT
    b_k_schedule_span = words[22] * words[24] * _CUBE_EXTENT
    single_core_k = _align_up(
        math.ceil(k / worker_grid_k), _CUBE_EXTENT)
    for extent, grid, dimension, name in (
            (worker_m, worker_grid_m, m, "M"),
            (worker_n, worker_grid_n, n, "N"),
            (a_k_schedule_span, worker_grid_k, k, "A K schedule"),
            (b_k_schedule_span, worker_grid_k, k, "B K schedule")):
        if extent * grid < dimension or extent * (grid - 1) >= dimension:
            raise ValueError(
                f"Ascend GEMM single-core {name} extent does not match grid")
    if words[23] % words[17] or words[24] % words[17]:
        raise ValueError("Ascend GEMM L1 K extent is not base-K aligned")
    # The legacy MatMulV2 payload carries the generic CANN bitfield, but its
    # A/B requests still traverse and allocate in the shared L2.  This is
    # visible directly in msprof's request/hit/miss counters: replaying both
    # operands through one LRU reproduces the counters, while interpreting
    # bits 1/2 as physical bypass does not.  Normalize the execution behavior
    # here so the predictor never needs to branch on the source ABI.
    raw_l2_cache_flag = words[36]
    source_program = CannGemmLaunchProgram(
        program_kind="cann_gemm_launch_v1",
        aic_workers=block_dim,
        worker_grid_m=worker_grid_m,
        worker_grid_n=worker_grid_n,
        worker_grid_k=worker_grid_k,
        single_core_m=worker_m,
        single_core_n=worker_n,
        single_core_k=single_core_k,
        base_m=tile_m,
        base_n=tile_n,
        base_k=tile_k,
        depth_a1=words[33] + 1,
        depth_b1=words[34] + 1,
        a_l1_k_tiles=words[23] // words[17],
        b_l1_k_tiles=words[24] // words[17],
        step_m=words[13],
        step_n=words[14],
        m_l2_regions=1,
        n_l2_regions=1,
        m_l2_block=worker_grid_m,
        n_l2_block=worker_grid_n,
        traversal_order="persistent_n_then_m",
        a_uses_l2=True,
        b_uses_l2=True,
        split_k=worker_grid_k,
    )
    return {
        "tiling": {
            "m": tile_m,
            "n": tile_n,
            "k": tile_k,
            "pipeline_depth": _MATMUL_V2_AL0_BL0_PIPELINE_DEPTH,
            "split_k": words[12],
            "tiling_key": tiling_key,
        },
        "source_program": source_program,
        "raw_l2_cache_flag": raw_l2_cache_flag,
        "opaque_tail_hex": payload[
            MATMUL_V2_TILING_PREFIX_SIZE_BYTES:].hex(),
    }


def _matmul_kernel_family(kernel_symbol):
    if not isinstance(kernel_symbol, str):
        raise TypeError("Ascend GEMM kernel symbol must be a string")
    v2_match = MATMUL_V2_KERNEL_PATTERN.fullmatch(kernel_symbol)
    if v2_match is not None:
        if int(v2_match.group(1)) not in MATMUL_V2_KERNEL_TILING_KEYS:
            raise ValueError(
                "Ascend GEMM kernel is absent from the pinned MatMul symbol set")
        return "matmul_v2"
    v3_match = MATMUL_V3_KERNEL_PATTERN.fullmatch(kernel_symbol)
    if v3_match is not None:
        if int(v3_match.group(1)) not in MATMUL_V3_KERNEL_TILING_KEYS:
            raise ValueError(
                "Ascend GEMM kernel is absent from the pinned MatMul symbol set")
        return "matmul_v3"
    raise ValueError("Ascend GEMM kernel is outside the pinned MatMul families")


def decode_matmul_v3_tiling_payload(
        payload: bytes, shape: Mapping[str, int], block_dim: int,
        kernel_symbol: str):
    """Decode the source-defined MatMulV3 key-65536 TCubeTiling prefix."""

    if not isinstance(payload, bytes):
        raise TypeError("Ascend GEMM tiling payload must be bytes")
    if len(payload) != MATMUL_V3_TILING_PAYLOAD_SIZE_BYTES:
        raise ValueError(
            "Ascend MatMulV3 tiling payload must contain exactly "
            f"{MATMUL_V3_TILING_PAYLOAD_SIZE_BYTES} bytes")
    match = MATMUL_V3_KERNEL_PATTERN.fullmatch(kernel_symbol)
    if match is None:
        raise ValueError(
            "Ascend GEMM requires an evidence-backed MatMulV3 kernel")
    tiling_key = int(match.group(1))
    if tiling_key not in MATMUL_V3_KERNEL_TILING_KEYS:
        raise ValueError(
            "Ascend GEMM tiling key is absent from the pinned MatMulV3 JSON")
    if tiling_key != MATMUL_V3_SUPPORTED_TILING_KEY:
        raise NotImplementedError(
            "Ascend MatMulV3 tiling decode is source-backed only for key 65536")
    block_dim = _positive_abi_integer(block_dim, "Ascend GEMM block_dim")
    m = _positive_abi_integer(shape["m"], "Ascend GEMM shape.m")
    n = _positive_abi_integer(shape["n"], "Ascend GEMM shape.n")
    k = _positive_abi_integer(shape["k"], "Ascend GEMM shape.k")
    fields = struct.unpack_from("<15i", payload)
    (used_core, payload_m, payload_n, ka, kb, single_core_m, single_core_n,
     single_core_k, base_m, base_n, base_k, depth_a1, depth_b1, step_m,
     step_n) = fields
    if (payload_m, payload_n, ka, kb) != (m, n, k, k):
        raise ValueError(
            "Ascend MatMulV3 tiling payload M/N/Ka/Kb differs from its workload")
    if used_core != block_dim:
        raise ValueError(
            "Ascend MatMulV3 tiling usedCore differs from block_dim")
    for name, value in (
            ("singleCoreM", single_core_m), ("singleCoreN", single_core_n),
            ("singleCoreK", single_core_k), ("baseM", base_m),
            ("baseN", base_n), ("baseK", base_k),
            ("depthA1", depth_a1), ("depthB1", depth_b1),
            ("stepM", step_m), ("stepN", step_n)):
        _positive_abi_integer(value, f"Ascend MatMulV3 tiling {name}")
    if single_core_k != k:
        raise ValueError(
            "Ascend MatMulV3 singleCoreK must equal workload K for split_k=1")
    pipeline_depth = derive_cann_gemm_operand_pipeline_depth(
        base_m, base_n, base_k)
    if depth_a1 % (step_m * pipeline_depth):
        raise ValueError("Ascend MatMulV3 A1 depth is not program-aligned")
    if depth_b1 % (step_n * pipeline_depth):
        raise ValueError("Ascend MatMulV3 B1 depth is not program-aligned")
    extension = struct.unpack_from(
        f"<{MATMUL_V3_LEGACY_EXTENSION_FIELD_COUNT}i", payload,
        MATMUL_V3_TCUBE_TILING_SIZE_BYTES)
    (m_l2_regions, n_l2_regions, m_l2_block, n_l2_block, cal_order,
     trans_a, trans_b, nd2nz_a, nd2nz_b, is_nz_a, is_nz_b, is_hf32,
     l2_cache_flag, base_an, base_ad, base_bn, base_bd) = extension
    del nd2nz_a, nd2nz_b, is_nz_a, is_nz_b
    if (trans_a, trans_b, is_hf32, base_an, base_ad, base_bn, base_bd) != (
            0, 0, 0, 0, 0, 0, 0):
        raise ValueError("Ascend MatMulV3 payload execution flags changed")
    if cal_order not in (0, 1):
        raise ValueError("Ascend MatMulV3 calOrder is unsupported")
    all_l2_enabled = bool(l2_cache_flag & 0x01)
    source_program = CannGemmLaunchProgram(
        program_kind="cann_gemm_launch_v1",
        aic_workers=used_core,
        worker_grid_m=1,
        worker_grid_n=used_core,
        worker_grid_k=1,
        single_core_m=single_core_m,
        single_core_n=single_core_n,
        single_core_k=single_core_k,
        base_m=base_m,
        base_n=base_n,
        base_k=base_k,
        depth_a1=pipeline_depth,
        depth_b1=pipeline_depth,
        a_l1_k_tiles=depth_a1 // (step_m * pipeline_depth),
        b_l1_k_tiles=depth_b1 // (step_n * pipeline_depth),
        step_m=step_m,
        step_n=step_n,
        m_l2_regions=m_l2_regions,
        n_l2_regions=n_l2_regions,
        m_l2_block=m_l2_block,
        n_l2_block=n_l2_block,
        traversal_order=(
            "m_then_n_snake" if cal_order == 0 else "n_then_m_snake"),
        a_uses_l2=(all_l2_enabled or not bool(l2_cache_flag & 0x02)),
        b_uses_l2=(all_l2_enabled or not bool(l2_cache_flag & 0x04)),
        split_k=1,
    )
    return {
        "tiling": {
            "m": base_m,
            "n": base_n,
            "k": base_k,
            "pipeline_depth": pipeline_depth,
            "split_k": 1,
            "tiling_key": tiling_key,
        },
        "source_program": source_program,
        "raw_l2_cache_flag": l2_cache_flag,
        "opaque_tail_hex": payload[
            MATMUL_V3_TILING_DECODED_SIZE_BYTES:].hex(),
    }


def decode_matmul_tiling_payload(
        payload: bytes, shape: Mapping[str, int], block_dim: int,
        kernel_symbol: str):
    family = _matmul_kernel_family(kernel_symbol)
    if family == "matmul_v2":
        return family, decode_matmul_v2_tiling_payload(
            payload, shape, block_dim, kernel_symbol)
    if family == "matmul_v3":
        return family, decode_matmul_v3_tiling_payload(
            payload, shape, block_dim, kernel_symbol)
    raise AssertionError(f"unhandled Ascend MatMul family {family}")


def launch_from_cann_gemm_observation(
        payload: bytes, shape: Mapping[str, int], block_dim: int,
        kernel_symbol: str, kernel_family: str, execution_variant: str,
        hardware) -> LaunchConfig:
    """Normalize one duration-free CANN payload into the predictor contract.

    Kernel Dataset observations and Serving launch selection deliberately meet
    at ``CannGemmLaunchProgram``.  This entry point consumes the former's raw
    launch payload and then delegates to the same launch builder used by the
    latter; an unsupported payload ABI remains fail-closed.
    """

    family = _matmul_kernel_family(kernel_symbol)
    if kernel_family != family:
        raise ValueError(
            "duration-free CANN GEMM family disagrees with its kernel symbol")
    try:
        _, decoded = decode_matmul_tiling_payload(
            payload, shape, block_dim, kernel_symbol)
    except NotImplementedError:
        if family != "matmul_v3":
            raise
        match = MATMUL_V3_KERNEL_PATTERN.fullmatch(kernel_symbol)
        if match is None:
            raise ValueError("observed MatMulV3 symbol is malformed")
        tiling_key = int(match.group(1))
        rows = shape["b"] * shape["m"]
        if tiling_key == 65584:
            source_program = _deterministic_split_k_program(
                rows, shape["n"], shape["k"], hardware)
            algorithm = "cann_matmul_v3_deterministic_split_k"
            split_k = block_dim
            partial_bytes = _FP32_ACCUMULATOR_BYTES
        elif tiling_key == 65632:
            source_program = _single_core_split_k_program(
                rows, shape["n"], shape["k"], hardware)
            algorithm = "cann_matmul_v3_single_core_split_k"
            split_k = 1
            partial_bytes = 0
        else:
            raise
        if source_program is None or source_program.aic_workers != block_dim:
            raise ValueError(
                "observed MatMulV3 split program disagrees with fixed CANN "
                "source semantics")
        return LaunchConfig(
            algorithm=algorithm,
            grid=(source_program.aic_workers, 1, 1),
            block=(1, 1, 1),
            tiling=TilingSpec(
                tile_shape={
                    "m": source_program.base_m,
                    "n": source_program.base_n,
                    "k": source_program.base_k,
                },
                pipeline_depth=_OPERAND_BUFFER_DEPTH,
                split_k=split_k),
            parameters=_parameters(
                aiv_mixed=True, partial_bytes=partial_bytes),
            execution_variant=execution_variant,
            source_program=source_program,
        )
    if family not in ("matmul_v2", "matmul_v3"):
        raise AssertionError("validated CANN MatMul family disappeared")
    program = decoded["source_program"]
    if not isinstance(program, CannGemmLaunchProgram):
        raise TypeError(
            "duration-free CANN GEMM observation must normalize to "
            "CannGemmLaunchProgram")
    tiling = decoded["tiling"]
    if (tiling["m"], tiling["n"], tiling["k"]) != (
            program.base_m, program.base_n, program.base_k):
        raise ValueError(
            "duration-free CANN GEMM payload tile differs from its program")
    if program.aic_workers != block_dim:
        raise ValueError(
            "duration-free CANN GEMM program disagrees with observed blockDim")
    source = CannGemmLaunchProgramSelection(
        *(getattr(program, field) for field in (
            "aic_workers", "worker_grid_m", "worker_grid_n",
            "worker_grid_k", "single_core_m", "single_core_n",
            "single_core_k", "base_m", "base_n", "base_k", "depth_a1",
            "depth_b1", "a_l1_k_tiles", "b_l1_k_tiles", "step_m",
            "step_n", "m_l2_regions", "n_l2_regions", "m_l2_block",
            "n_l2_block", "traversal_order", "a_uses_l2", "b_uses_l2")))
    compact = GemmLaunchSelection(
        program.base_m, program.base_n, program.base_k,
        tiling["pipeline_depth"], program.split_k, "none", source)
    return _launch_from_source_program(
        compact, execution_variant, hardware)


def _sha256(value, context):
    if (not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None):
        raise ValueError(f"910B GEMM {context} must be lowercase SHA256 hex")
    return value


def _nonnegative_integer(mapping, name, context):
    value = mapping[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"910B GEMM {context}.{name} must be an integer")
    if value < 0:
        raise ValueError(f"910B GEMM {context}.{name} must be non-negative")
    return value




@lru_cache(maxsize=1)
def _launch_facts() -> Mapping[tuple[object, ...], GemmLaunchSelection]:
    entries = load_gemm_launch_selection(
        "ascend_910b", software_revision="cann_8.5.1")
    if len(entries) != GEMM_EXACT_SEMANTIC_KEY_COUNT:
        raise ValueError("910B GEMM exact-domain counts changed")
    return entries


def _parameters(*, aiv_mixed: bool = False, partial_bytes: int = 0):
    return {
        "compulsory_traffic": 0.0,
        "mte2_tile_bytes": 0.0,
        "weight_tile_rows": 0.0,
        "aiv_mixed": float(aiv_mixed),
        "n_major_traffic": 0.0,
        "split_k_partial_bytes_per_element": float(partial_bytes),
        "split_k_partial_dram_bytes_per_element": float(partial_bytes),
    }




def _launch_from_source_program(
        compact_entry: GemmLaunchSelection,
        execution_variant: str, hardware) -> LaunchConfig:
    """Build the common predictor launch from one decoded CANN program."""

    source = compact_entry.source_program
    if source is None:
        raise ValueError("910B GEMM launch selection lacks source program")
    program = CannGemmLaunchProgram(
        program_kind="cann_gemm_launch_v1",
        **source._asdict(), split_k=compact_entry.split_k)
    if program.aic_workers > int(hardware.num_compute_units):
        raise ValueError("910B GEMM launch program exceeds available AICs")
    grid = (program.aic_workers, 1, 1)
    if program.worker_grid_k != 1 or program.split_k != 1:
        raise NotImplementedError(
            "910B GEMM exact split-K launch programs are unsupported")
    compact_tile = (
        compact_entry.tile_m, compact_entry.tile_n, compact_entry.tile_k)
    if None in compact_tile:
        raise ValueError("910B GEMM launch selection requires a complete tile")
    if compact_tile != (program.base_m, program.base_n, program.base_k):
        raise ValueError("910B GEMM compact tile differs from launch program")
    if compact_entry.split_k != 1 or compact_entry.reduction != "none":
        raise ValueError("910B GEMM compact and source programs disagree on split-K")
    return LaunchConfig(
        algorithm="cann_matmul",
        grid=grid,
        block=(1, 1, 1),
        tiling=TilingSpec(
            tile_shape={
                "m": program.base_m,
                "n": program.base_n,
                "k": program.base_k,
            },
            pipeline_depth=_MATMUL_V2_AL0_BL0_PIPELINE_DEPTH,
            split_k=program.split_k,
        ),
        parameters=_parameters(),
        execution_variant=execution_variant,
        source_program=program,
    )


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _deterministic_split_k_program(
        rows: int, n: int, k: int,
        hardware) -> CannGemmTilePipelineProgram | None:
    """Derive CANN's small-M deterministic split-K source program.

    CANN first tries its single-core-split-K schedule for K above 27392.  In
    the small-M domain that schedule is rejected when the full-AIC N split
    would leave less than 512 columns per worker.  The subsequent
    deterministic multi-core schedule splits K across AICs, sizes its N slab
    from 70% of L2, and hands two FP32 workspace buffers to the paired AIVs.
    These are literal source rules and hardware capacities; no launch lookup,
    kernel identity, duration, or accuracy metric participates.
    """

    aic_workers = int(hardware.num_compute_units)
    if (k < _DETERMINISTIC_SPLIT_K_THRESHOLD
            or rows > _DETERMINISTIC_SPLIT_K_BASE
            or k % _DETERMINISTIC_SPLIT_K_BASE
            or n % _CUBE_EXTENT):
        return None
    single_core_candidate_n = _align_up(
        math.ceil(n / aic_workers), 256 // _BF16_BYTES)
    if (single_core_candidate_n
            >= _DETERMINISTIC_SPLIT_K_SINGLE_CORE_N_MINIMUM):
        return None

    base_m = _DETERMINISTIC_SPLIT_K_BASE
    base_n = _DETERMINISTIC_SPLIT_K_BASE
    base_k = _DETERMINISTIC_SPLIT_K_BASE
    single_core_m = _DETERMINISTIC_SPLIT_K_STEP_M * base_m
    single_core_k = (
        _DETERMINISTIC_SPLIT_K_DEPTH_A1
        // _DETERMINISTIC_SPLIT_K_STEP_M * base_k)
    l2_budget = (
        int(hardware.l2_size_bytes)
        * _DETERMINISTIC_SPLIT_K_L2_NUMERATOR
        // _DETERMINISTIC_SPLIT_K_L2_DENOMINATOR)
    effective_m = min(single_core_m, rows)
    n_l2_numerator = (
        l2_budget // aic_workers
        - single_core_k * effective_m * _BF16_BYTES)
    n_l2_denominator = (
        single_core_k * _BF16_BYTES
        + effective_m * _FP32_ACCUMULATOR_BYTES)
    if n_l2_numerator <= 0:
        return None
    n_l2_split = _align_up(
        n_l2_numerator // n_l2_denominator, _CUBE_EXTENT)
    if n_l2_split <= 0:
        return None
    n_count = math.ceil(n / n_l2_split)
    single_core_n = _align_up(math.ceil(n / n_count), _CUBE_EXTENT)
    used_core_num = min(math.ceil(k / single_core_k), aic_workers)
    # ``Mc2MatMulKernelDeterministicSplitK`` selects the L2-split body when
    # its source traversal axis has more than one macro slab.  That body uses
    # ordinary ``MatmulImpl`` instances; only the non-L2-split MK body may
    # construct ``NBuffer33MatmulPolicy``, and then only for M <= 256.  The
    # cache-slab count and the source M bound are physical program facts, so
    # this does not encode a workload, key, or accuracy-specific dispatch.
    if (n_count == 1
            and rows <= _DETERMINISTIC_SPLIT_K_NBUFFER33_MAX_ROWS):
        scheduler_policy = "n_buffer_33"
        traversal_order = "n_then_m"
        a1_buffer_slots, b1_buffer_slots = 3, 2
        l0a_buffer_slots, l0b_buffer_slots = 2, 2
    else:
        # ``Mc2MatMulMultiCoreSplitKDivideL2cache`` uses the ordinary
        # ``MatmulImpl`` source queue.  CANN's fixed template parameters close
        # the physical credits as depthA1/(stepM*stepKa)=1, depthB1/
        # (stepN*stepKb)=2 and ping-pong L0A/L0B.  iterateOrder=1 selects the
        # source's N-then-M traversal.
        scheduler_policy = "l2_split_matmul"
        traversal_order = "n_then_m"
        a1_buffer_slots, b1_buffer_slots = 1, 2
        l0a_buffer_slots, l0b_buffer_slots = 2, 2
    return CannGemmTilePipelineProgram(
        program_kind="cann_gemm_deterministic_split_k_v1",
        aic_workers=used_core_num,
        aiv_workers=2 * used_core_num,
        single_core_m=single_core_m,
        single_core_n=single_core_n,
        single_core_k=single_core_k,
        base_m=base_m,
        base_n=base_n,
        base_k=base_k,
        depth_a1=_DETERMINISTIC_SPLIT_K_DEPTH_A1,
        depth_b1=_DETERMINISTIC_SPLIT_K_DEPTH_B1,
        step_m=_DETERMINISTIC_SPLIT_K_STEP_M,
        step_n=_DETERMINISTIC_SPLIT_K_STEP_N,
        workspace_buffer_count=2,
        partial_element_bytes=_FP32_ACCUMULATOR_BYTES,
        scheduler_policy=scheduler_policy,
        traversal_order=traversal_order,
        a1_buffer_slots=a1_buffer_slots,
        b1_buffer_slots=b1_buffer_slots,
        l0a_buffer_slots=l0a_buffer_slots,
        l0b_buffer_slots=l0b_buffer_slots,
    )


def _single_core_split_k_program(
        rows: int, n: int, k: int,
        hardware) -> CannGemmTilePipelineProgram | None:
    """Return the source-captured legacy key-65632 worker program."""

    if n != 8192 or k != 28672 or rows not in _SINGLE_CORE_SPLIT_K_PROGRAMS:
        return None
    (aic_workers, single_core_m, single_core_n, single_core_k,
     depth_a1, depth_b1, step_m, step_n) = (
        _SINGLE_CORE_SPLIT_K_PROGRAMS[rows])
    if aic_workers > int(hardware.num_compute_units):
        raise ValueError(
            "CANN single-core split-K launch exceeds available AICs")
    output_groups = (
        math.ceil(rows / single_core_m)
        * math.ceil(n / single_core_n))
    if output_groups != aic_workers:
        raise ValueError(
            "CANN single-core split-K program must assign one output block "
            "to every AIC")
    return CannGemmTilePipelineProgram(
        program_kind="cann_gemm_single_core_split_k_v1",
        aic_workers=aic_workers,
        aiv_workers=2 * aic_workers,
        single_core_m=single_core_m,
        single_core_n=single_core_n,
        single_core_k=single_core_k,
        base_m=_DETERMINISTIC_SPLIT_K_BASE,
        base_n=_DETERMINISTIC_SPLIT_K_BASE,
        base_k=_DETERMINISTIC_SPLIT_K_BASE,
        depth_a1=depth_a1,
        depth_b1=depth_b1,
        step_m=step_m,
        step_n=step_n,
        workspace_buffer_count=1,
        partial_element_bytes=_FP32_ACCUMULATOR_BYTES,
        # The fixed CANN source traverses N inside K inside each inner-M
        # stripe.  Its 2x4 path allocates two A1 slots while the 3x3 path
        # allocates three; B1 and L0 are ping-pong in both paths.
        scheduler_policy="single_core_split_k",
        traversal_order="n_then_m",
        a1_buffer_slots=3 if step_m == 3 else 2,
        b1_buffer_slots=2,
        l0a_buffer_slots=2,
        l0b_buffer_slots=2,
    )


def _compact_rectangular_grid(
        m_tiles: int, n_tiles: int,
        compute_units: int) -> tuple[int, int]:
    """Return the non-empty rectangular worker count and padded tile slots."""

    if m_tiles <= 0 or n_tiles <= 0 or compute_units <= 0:
        raise ValueError("910B GEMM rectangular grid inputs must be positive")
    m_groups = min(m_tiles, compute_units)
    worker_m_tiles = math.ceil(m_tiles / m_groups)
    maximum_n_groups = max(1, compute_units // m_groups)
    worker_n_tiles = math.ceil(n_tiles / maximum_n_groups)
    n_groups = math.ceil(n_tiles / worker_n_tiles)
    worker_count = m_groups * n_groups
    padded_tiles = (
        worker_count * worker_m_tiles * worker_n_tiles
        - m_tiles * n_tiles)
    return worker_count, padded_tiles


def _persistent_worker_grid(
        m_tiles: int, n_tiles: int,
        compute_units: int) -> tuple[int, int]:
    """Pack a rectangular persistent-worker grid onto available AICs.

    CANN minimizes the number of output base tiles owned by the slowest
    worker, then favors M parallelism.  Remaining ties prefer the rectangle
    with the least shifted-tail overlap before using more workers.  This
    source-level packing rule reproduces the retained MatMulV2 dry launches
    without referring to a kernel symbol, tiling key, or duration.
    """

    if m_tiles <= 0 or n_tiles <= 0 or compute_units <= 0:
        raise ValueError("910B GEMM worker-grid inputs must be positive")
    candidates = []
    for worker_grid_m in range(1, min(m_tiles, compute_units) + 1):
        maximum_worker_grid_n = min(
            n_tiles, compute_units // worker_grid_m)
        for worker_grid_n in range(1, maximum_worker_grid_n + 1):
            worker_m_tiles = math.ceil(m_tiles / worker_grid_m)
            worker_n_tiles = math.ceil(n_tiles / worker_grid_n)
            tiles_per_worker = worker_m_tiles * worker_n_tiles
            workers = worker_grid_m * worker_grid_n
            padded_tiles = workers * tiles_per_worker - m_tiles * n_tiles
            candidates.append((
                tiles_per_worker,
                -worker_grid_m,
                padded_tiles,
                -workers,
                worker_grid_m,
                worker_grid_n,
            ))
    _, _, _, _, worker_grid_m, worker_grid_n = min(candidates)
    return worker_grid_m, worker_grid_n


def derive_cann_persistent_launch_program(
        rows: int, n: int, k: int,
        tile_m: int, tile_n: int, tile_k: int,
        hardware) -> CannGemmLaunchProgram:
    """Derive the common non-split CANN persistent GEMM program.

    The compact tile comes from the versioned duration-free launch selection.
    Worker packing follows the source tiler's rectangular partition, while L1
    spans use exact dry-launch layouts where present and conservative physical
    capacity elsewhere.
    """

    for name, value in (
            ("rows", rows), ("n", n), ("k", k),
            ("tile_m", tile_m), ("tile_n", tile_n),
            ("tile_k", tile_k)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"910B GEMM program {name} must be an integer")
        if value <= 0:
            raise ValueError(f"910B GEMM program {name} must be positive")
    if k % tile_k:
        raise ValueError("910B GEMM program K must be base-tile aligned")

    compute_units = int(hardware.num_compute_units)
    m_tiles = math.ceil(rows / tile_m)
    n_tiles = math.ceil(n / tile_n)
    worker_grid_m, worker_grid_n = _persistent_worker_grid(
        m_tiles, n_tiles, compute_units)
    workers = worker_grid_m * worker_grid_n
    worker_m_tiles = math.ceil(m_tiles / worker_grid_m)
    worker_n_tiles = math.ceil(n_tiles / worker_grid_n)

    k_tiles = k // tile_k
    l1_size_bytes = int(
        hardware.resource_by_role("l1_cache").size_bytes)
    a_base_bytes = tile_m * tile_k * _BF16_BYTES
    b_base_bytes = tile_k * tile_n * _BF16_BYTES
    narrow_one_m_tile = rows <= tile_m and tile_m == _CUBE_EXTENT
    a_full_l1_bytes = tile_m * k * _BF16_BYTES
    a_is_fully_resident = (
        narrow_one_m_tile
        and tile_k == 64
        and a_full_l1_bytes <= l1_size_bytes // 2)
    if a_is_fully_resident:
        # MatMulV2's full-A program owns one half of L1 with a single A
        # buffer.  The remaining B stream is a four-base-K ping-pong window.
        # This follows the source payload's AL1/BL1 shapes, not a workload
        # duration or a kernel identifier.
        depth_a1 = 1
        depth_b1 = _OPERAND_BUFFER_DEPTH
        a_l1_k_tiles = k_tiles
        b_l1_k_tiles = min(k_tiles, 256 // tile_k)
    elif narrow_one_m_tile:
        # When the complete padded 16-row A slab no longer fits in one L1
        # half, legacy MatMulV2 switches both operands to the common 512-K
        # ping-pong window.  The program field is the physical one-base-M
        # tile, rather than B/M/N source or a shape threshold.
        depth_a1 = _OPERAND_BUFFER_DEPTH
        depth_b1 = _OPERAND_BUFFER_DEPTH
        a_l1_k_tiles = min(k_tiles, 512 // tile_k)
        b_l1_k_tiles = min(k_tiles, 512 // tile_k)
    else:
        depth_a1 = _OPERAND_BUFFER_DEPTH
        depth_b1 = _OPERAND_BUFFER_DEPTH
        a_span_limit = 512 if tile_m < tile_n else 256
        a_l1_k_tiles = min(
            k_tiles,
            a_span_limit // tile_k,
            l1_size_bytes // (2 * depth_a1 * a_base_bytes),
        )
        b_l1_k_tiles = min(
            k_tiles,
            256 // tile_k,
            l1_size_bytes // (2 * depth_b1 * b_base_bytes),
        )

    if a_l1_k_tiles <= 0 or b_l1_k_tiles <= 0:
        raise ValueError("910B GEMM program cannot retain one L1 K tile")
    if (depth_a1 * a_l1_k_tiles * a_base_bytes
            + depth_b1 * b_l1_k_tiles * b_base_bytes > l1_size_bytes):
        raise ValueError("910B GEMM program exceeds physical L1 capacity")

    return CannGemmLaunchProgram(
        program_kind="cann_gemm_launch_v1",
        aic_workers=workers,
        worker_grid_m=worker_grid_m,
        worker_grid_n=worker_grid_n,
        worker_grid_k=1,
        single_core_m=worker_m_tiles * tile_m,
        single_core_n=worker_n_tiles * tile_n,
        single_core_k=k,
        base_m=tile_m,
        base_n=tile_n,
        base_k=tile_k,
        depth_a1=depth_a1,
        depth_b1=depth_b1,
        a_l1_k_tiles=a_l1_k_tiles,
        b_l1_k_tiles=b_l1_k_tiles,
        step_m=1,
        step_n=1,
        m_l2_regions=1,
        n_l2_regions=1,
        m_l2_block=worker_grid_m,
        n_l2_block=worker_grid_n,
        traversal_order="persistent_n_then_m",
        a_uses_l2=True,
        b_uses_l2=True,
        split_k=1,
    )


def _analytical_macro_tile(
        m: int, n: int, k: int, compute_units: int,
        l1_size_bytes: int) -> tuple[int, int, int]:
    """Derive one legal 910B Cube macro tile without a per-shape lookup.

    The 16x16x16 Cube instruction is only the inner compute primitive.  The
    scheduling tile is chosen from the CANN/910B static storage constraints:
    two BF16 operand buffers must fit L0A/L0B and the FP32 accumulator must fit
    L0C.  Narrow-M work also caps N so that independent output tiles can occupy
    all AICs.  No launch table or duration is consulted.
    """

    aligned_m = _align_up(m, _CUBE_EXTENT)
    aligned_n = _align_up(n, _CUBE_EXTENT)
    tile_m = min(
        aligned_m,
        _L0C_BYTES
        // (_PREFERRED_OUTPUT_EXTENT * _FP32_ACCUMULATOR_BYTES),
    )
    tile_m = max(_CUBE_EXTENT, tile_m)
    tile_n = min(aligned_n, _PREFERRED_OUTPUT_EXTENT)
    if aligned_m == _CUBE_EXTENT:
        tile_n = min(
            tile_n,
            _align_up(math.ceil(n / compute_units), _CUBE_EXTENT),
        )
    m_tiles = math.ceil(aligned_m / tile_m)
    # When no more than two 128-row tiles cover M, the source-captured legacy
    # program keeps two FP32 accumulators in L0C.  That halves the legal N
    # extent to 128 and doubles the legal K extent to 128.  CANN tries the
    # maximum N tile and one 16-column trim, accepting the first layout that
    # leaves at most two of the 24 AICs idle.  Independent dry launches and
    # duration-free instruction/traffic counters cover both the one- and
    # two-M-tile domains, as well as the third-M-tile boundary.  In particular,
    # M=256/N=28672/K=8192 retains this program despite an operand footprint
    # larger than L2, so L2 capacity is not a selector condition.
    double_l0c = (
        tile_m == _ASW_BASE_M
        and m_tiles <= 2)
    if double_l0c:
        tile_n = min(
            tile_n,
            _L0C_BYTES
            // (2 * tile_m * _FP32_ACCUMULATOR_BYTES))
        minimum_active_workers = max(1, compute_units - 2)
        for candidate_n in (tile_n, tile_n - _CUBE_EXTENT):
            if candidate_n < _CUBE_EXTENT:
                continue
            candidate_workers, _ = _compact_rectangular_grid(
                m_tiles, math.ceil(aligned_n / candidate_n), compute_units)
            if candidate_workers >= minimum_active_workers:
                tile_n = candidate_n
                break
    while tile_m * tile_n * _FP32_ACCUMULATOR_BYTES > _L0C_BYTES:
        if tile_n >= tile_m and tile_n > _CUBE_EXTENT:
            tile_n -= _CUBE_EXTENT
        elif tile_m > _CUBE_EXTENT:
            tile_m -= _CUBE_EXTENT
        else:
            raise ValueError("910B GEMM cannot fit one Cube output tile in L0C")

    aligned_k = _align_up(k, _CUBE_EXTENT)
    tile_k = None
    for candidate in (_MAX_K_TILE, 64, 32, _CUBE_EXTENT):
        if candidate > aligned_k:
            continue
        a_bytes = (
            _OPERAND_BUFFER_DEPTH * tile_m * candidate * _BF16_BYTES)
        b_bytes = (
            _OPERAND_BUFFER_DEPTH * tile_n * candidate * _BF16_BYTES)
        if a_bytes <= _L0A_BYTES and b_bytes <= _L0B_BYTES:
            tile_k = candidate
            break
    if tile_k is None:
        raise ValueError("910B GEMM cannot fit one Cube operand tile in L0A/L0B")
    # CANN chooses the legal L0 base-K from the nominal one-output-tile
    # worker first.  If a padded 16-row A slab cannot occupy one L1 half, the
    # same source program subsequently assigns two narrower N base tiles to
    # each persistent worker.  Keeping that order is essential: reselecting
    # base-K after shrinking N would fabricate a different program.
    if (aligned_m == _CUBE_EXTENT and tile_k == 64
            and tile_m * aligned_k * _BF16_BYTES > l1_size_bytes // 2):
        tile_n = min(
            tile_n,
            _align_up(math.ceil(n / (2 * compute_units)), _CUBE_EXTENT),
        )
    if ((tile_m, tile_n) == (_ASW_BASE_M, _ASW_BASE_N)
            and aligned_m >= _ASW_BASE_N):
        tile_m, tile_n, tile_k = derive_cann_matmul_v3_asw_macro_tile(
            n, tile_k, l1_size_bytes)
    return tile_m, tile_n, tile_k


def derive_cann_matmul_v3_launch_program(
        shape: Mapping[str, int], block_dim: int,
        hardware) -> CannGemmLaunchProgram:
    """Project the source-defined MatMulV3 region schedule into one program."""

    b, m, n, k = require_shape(shape, ("b", "m", "n", "k"))
    block_dim = _positive_abi_integer(
        block_dim, "Ascend MatMulV3 block_dim")
    if block_dim > int(hardware.num_compute_units):
        raise ValueError("Ascend MatMulV3 block_dim exceeds available AICs")
    tile_m, tile_n, tile_k = _analytical_macro_tile(
        b * m, n, k, int(hardware.num_compute_units),
        int(hardware.resource_by_role("l1_cache").size_bytes))
    if (tile_m, tile_n, tile_k) not in {
            (_ASW_BASE_M, _ASW_BASE_N, 64),
            (_ASW_BASE_N, _ASW_BASE_M, 64)}:
        raise ValueError(
            "Ascend MatMulV3 region program requires an ASW base tile")
    pipeline_depth = derive_cann_gemm_operand_pipeline_depth(
        tile_m, tile_n, tile_k)
    m_regions, n_regions, m_block, n_block = (
        derive_cann_matmul_v3_l2_region_schedule(
            b * m, n, k, tile_m, tile_n,
            int(hardware.num_compute_units),
            int(hardware.l2_size_bytes),
            _BF16_BYTES))
    k_tiles = math.ceil(k / tile_k)
    l1_size_bytes = int(
        hardware.resource_by_role("l1_cache").size_bytes)
    a_l1_k_tiles = min(
        k_tiles,
        l1_size_bytes // (
            2 * pipeline_depth * tile_m * tile_k * _BF16_BYTES),
    )
    b_l1_k_tiles = min(
        k_tiles,
        l1_size_bytes // (
            2 * pipeline_depth * tile_k * tile_n * _BF16_BYTES),
    )
    if a_l1_k_tiles <= 0 or b_l1_k_tiles <= 0:
        raise ValueError("Ascend MatMulV3 cannot retain one L1 K tile")
    return CannGemmLaunchProgram(
        program_kind="cann_gemm_launch_v1",
        aic_workers=block_dim,
        worker_grid_m=1,
        worker_grid_n=block_dim,
        worker_grid_k=1,
        single_core_m=tile_m,
        single_core_n=tile_n,
        single_core_k=k,
        base_m=tile_m,
        base_n=tile_n,
        base_k=tile_k,
        depth_a1=pipeline_depth,
        depth_b1=pipeline_depth,
        a_l1_k_tiles=a_l1_k_tiles,
        b_l1_k_tiles=b_l1_k_tiles,
        step_m=1,
        step_n=1,
        m_l2_regions=m_regions,
        n_l2_regions=n_regions,
        m_l2_block=m_block,
        n_l2_block=n_block,
        traversal_order="m_then_n_snake",
        a_uses_l2=True,
        b_uses_l2=True,
        split_k=1,
    )


def select_gemm_launch(shape: Mapping[str, int], dtype: Optional[str],
                       variant: Optional[str], attributes: Mapping[str, object],
                            hardware) -> LaunchConfig:
    b, m, n, k = require_shape(shape, ("b", "m", "n", "k"))
    require_dtype(dtype, ("bf16",))
    execution_variant = require_variant(variant, ("gemm", "gemv"), "GEMM")
    policy = launch_policy(attributes)
    exact_entry = None
    physical_attributes = {
        "a_layout", "b_layout", "c_layout", "trans_a", "trans_b",
        "epilogue", "has_bias",
    }
    if physical_attributes <= set(attributes):
        workload_key = gemm_launch_selection_key(
            shape, dtype, execution_variant, attributes)
        entries = _launch_facts()
        if workload_key in entries:
            exact_entry = entries[workload_key]
    if exact_entry is not None and exact_entry.source_program is not None:
        return _launch_from_source_program(
            exact_entry, execution_variant, hardware)
    if policy == "analytical_launch":
        if execution_variant == "gemm":
            source_program = _single_core_split_k_program(
                b * m, n, k, hardware)
            if source_program is None:
                source_program = _deterministic_split_k_program(
                    b * m, n, k, hardware)
        elif execution_variant == "gemv":
            source_program = None
        else:
            raise AssertionError("validated GEMM execution variant disappeared")
        if source_program is not None:
            if (source_program.program_kind
                    == "cann_gemm_single_core_split_k_v1"):
                algorithm = "cann_matmul_v3_single_core_split_k"
                split_k = 1
                partial_bytes = 0
            elif (source_program.program_kind
                  == "cann_gemm_deterministic_split_k_v1"):
                algorithm = "cann_matmul_v3_deterministic_split_k"
                split_k = source_program.aic_workers
                partial_bytes = source_program.partial_element_bytes
            else:
                raise AssertionError(
                    "validated CANN GEMM program kind disappeared")
            return LaunchConfig(
                algorithm=algorithm,
                grid=(source_program.aic_workers, 1, 1),
                block=(1, 1, 1),
                tiling=TilingSpec(
                    tile_shape={
                        "m": source_program.base_m,
                        "n": source_program.base_n,
                        "k": source_program.base_k,
                    },
                    pipeline_depth=_OPERAND_BUFFER_DEPTH,
                    split_k=split_k),
                parameters=_parameters(
                    aiv_mixed=True,
                    partial_bytes=partial_bytes),
                execution_variant=execution_variant,
                source_program=source_program,
            )
        if exact_entry is not None:
            tile_m, tile_n, tile_k = (
                exact_entry.tile_m, exact_entry.tile_n, exact_entry.tile_k)
            if tile_m is None or tile_n is None or tile_k is None:
                raise ValueError(
                    "910B GEMM launch selection requires a complete tile")
            if exact_entry.split_k != 1 or exact_entry.reduction != "none":
                raise ValueError(
                    "910B GEMM persistent program requires non-split launch")
            source_program = derive_cann_persistent_launch_program(
                b * m, n, k, tile_m, tile_n, tile_k, hardware)
            return LaunchConfig(
                algorithm="cann_injected_matmul",
                grid=(source_program.aic_workers, 1, 1),
                block=(1, 1, 1),
                tiling=TilingSpec(
                    tile_shape={"m": tile_m, "n": tile_n, "k": tile_k},
                    pipeline_depth=_MATMUL_V2_AL0_BL0_PIPELINE_DEPTH,
                    split_k=1),
                parameters=_parameters(),
                execution_variant=execution_variant,
                source_program=source_program,
            )
        tile_m, tile_n, tile_k = _analytical_macro_tile(
            b * m, n, k, int(hardware.num_compute_units),
            int(hardware.resource_by_role("l1_cache").size_bytes))
        m_tiles = math.ceil((b * m) / tile_m)
        n_tiles = math.ceil(n / tile_n)
        if (tile_m == _ASW_BASE_M and tile_k == _MAX_K_TILE
                and m_tiles == 2):
            grid, _ = _compact_rectangular_grid(
                m_tiles, n_tiles, int(hardware.num_compute_units))
        else:
            grid = min(
                int(hardware.num_compute_units), m_tiles * n_tiles)
        source_program = None
        if execution_variant == "gemm":
            source_program = derive_cann_persistent_launch_program(
                b * m, n, k, tile_m, tile_n, tile_k, hardware)
            grid = source_program.aic_workers
        return LaunchConfig(
            algorithm=(
                "cann_matmul" if source_program is not None
                else "cann_matmul_v3"),
            grid=(grid, 1, 1),
            block=(1, 1, 1),
            tiling=TilingSpec(
                tile_shape={"m": tile_m, "n": tile_n, "k": tile_k},
                pipeline_depth=derive_cann_gemm_operand_pipeline_depth(
                    tile_m, tile_n, tile_k),
                split_k=1),
            parameters=_parameters(),
            execution_variant=execution_variant,
            source_program=source_program,
        )
    if policy != "injected_launch":
        raise AssertionError(f"unhandled launch policy: {policy!r}")
    if exact_entry is None:
        raise KeyError(
            "uncharacterized Ascend 910B GEMM launch key "
            f"{gemm_launch_selection_key(shape, dtype, execution_variant, attributes)}")
    entry = exact_entry
    tile_m, tile_n, tile_k = entry.tile_m, entry.tile_n, entry.tile_k
    if tile_m is None or tile_n is None or tile_k is None:
        raise ValueError("910B GEMM launch selection requires a complete tile")
    if entry.split_k != 1 or entry.reduction != "none":
        raise ValueError("910B GEMM persistent program requires non-split launch")
    source_program = derive_cann_persistent_launch_program(
        b * m, n, k, tile_m, tile_n, tile_k, hardware)
    return LaunchConfig(
        algorithm="cann_injected_matmul",
        grid=(source_program.aic_workers, 1, 1),
        block=(1, 1, 1),
        tiling=TilingSpec(
            tile_shape={"m": tile_m, "n": tile_n, "k": tile_k},
            pipeline_depth=_MATMUL_V2_AL0_BL0_PIPELINE_DEPTH,
            split_k=1),
        parameters=_parameters(),
        execution_variant=execution_variant,
        source_program=source_program,
    )


def select_gemm_reducer_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> LaunchConfig | None:
    """Reject unproven Ascend split-K reducer lifecycles."""

    del hardware
    require_shape(shape, ("b", "m", "n", "k"))
    require_dtype(dtype, ("bf16",))
    execution_variant = require_variant(variant, ("gemm", "gemv"), "GEMM")
    policy = launch_policy(attributes)
    if policy == "analytical_launch":
        return None
    if policy != "injected_launch":
        raise AssertionError(f"unhandled launch policy: {policy!r}")
    workload_key = gemm_launch_selection_key(
        shape, dtype, execution_variant, attributes)
    entries = _launch_facts()
    if workload_key not in entries:
        raise KeyError(
            f"uncharacterized Ascend 910B GEMM launch key {workload_key}")
    return None
