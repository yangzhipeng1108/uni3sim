"""MetaX C500 GEMM/GEMV launch selection from exact host-selector facts."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Optional

from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.kernel.launch_config.gemm_workload import (
    GEMM_WORKLOAD_KEY_FIELDS,
    gemm_launch_selection_key,
    validate_gemm_workload_key,
)
from src.kernel.launch_config.gemm_launch_selection import (
    GemmLaunchSelection,
    ARTIFACT_MANIFEST_FIELDS,
    METAX_LAUNCH_FIELDS,
    METAX_RESULT_FIELDS,
    METAX_SCHEDULER_FIELDS,
    PROHIBITED_TIMING_FIELD_PARTS,
    load_gemm_launch_selection,
)
from src.kernel.launch_config.selection_primitives import (
    ceil_div,
    launch_policy,
    require_dtype,
    require_exact_fields as _exact_fields,
    require_mapping as _mapping,
    require_non_negative_integer,
    require_positive_integer,
    require_sequence as _sequence,
    require_shape,
    require_variant,
)
from src.kernel.launch_config.tiling_spec import parse_complete_gemm_tile


_EXPECTED_SEMANTIC_KEYS = 140544
_EXPECTED_PHYSICAL_OPERATIONS = 140544
_ROOT = Path(__file__).resolve().parents[4]
_DOMAIN_CONFIG = (
    _ROOT / "configs" / "evaluation" / "experiments"
    / "gemm_launch_domain.json")
_EXPECTED_WORKLOAD_MANIFEST_SHA256 = (
    "60c39f865cf0acff07c3906ab8167f2bba20bfad90dad2eb65e1827c371c9af4")
_MAIN_ALGORITHM = "mcblas_injected_gemm:gemm"
_REDUCER_ALGORITHM = "logical_gemm_aux:gemm_reduce"
_PROJECTION_FIELDS = {
    "execution_variant", "main_ordinal", "tile", "pipeline_depth",
    "split_k", "reducer_ordinal", "partial_buffer",
}
HARDWARE_DOMAIN_FIELDS = {
    "status", "domain_sha256", "semantic_key_count",
    "physical_operation_count",
}
_ALLOWED_TIMING_BOUNDARY_FIELDS = {
    "duration_consumed", "timing_consumed",
}
_OPERATION_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_MAIN_GEMM_PATTERN = re.compile(
    r"^mcblas__Mck_bf16gemm_tn_[0-9]+x[0-9]+x[0-9]+_")
_UNALIGNED_GEMM_PATTERN = re.compile(
    r"^_ZN8cpp_gemm12unalign_gemmL6LaunchINS0_19unalign_gemm_kernelI")
_GEMV_IDENTITIES = ("b16gemvt_kernel", "gemv")
_MANGLED_REDUCER_PREFIX = (
    "_ZN6mcblas20DevDstElementwiseAddI15__maca_bfloat16ffLi0EEEvPT_"
    "PKS2_PKT0_"
)
_DEMANGLED_REDUCER_PREFIX = (
    "void mcblas::DevDstElementwiseAdd<__maca_bfloat16, float, float, 0>("
)

# Serving graph-replay GEMM surface overlay.  The canonical launch CSV was
# collected with the dry-launch probe stack, whose host-side mcBLAS heuristic
# table differs from the canonical vLLM-metax 0.24.0 serving deployment:
# inside FULL/PIECEWISE graph replay the deployment selects different
# programs for the same physical (m, n, k).  The overlay records the
# deployment's discrete launch identity (tile, thread block, split-K,
# pipeline stage class) per token count m for the serving token GEMM cells,
# built exclusively from launch-identity runs on the deployed serving
# container (kernel name/grid/block per forward over every PIECEWISE token
# count and every FULL decode bucket; qwen3-4b/qwen3-8b/llama-3.1-8b TP1 and
# qwen3-32b TP2); it consumes no duration evidence.  Earlier mcTracer/nsys
# extractions came from a different software stack and are not used.
# Keying follows the deployment's measured semantics: PIECEWISE (mixed)
# steps select programs by the step's exact scheduled token count
# (``graph_logical_tokens``), while FULL decode steps select by the padded
# capture bucket (the physical m); both agree at equal m.  It applies only
# to queries that carry the serving graph token domain
# (``graph_runtime_mode`` in FULL/PIECEWISE); every other query - kernel
# evaluation, eager-mode (NONE) serving steps, lm_head, GEMV - keeps the
# canonical CSV contract, and an overlay miss inside the gated domain falls
# through to the CSV unchanged.
_GRAPH_SURFACE_REFERENCE = (
    "microbench/metax_c500/references/"
    "mcblas_serving_graph_gemm_surface_vllm_metax_0_24_0.json")
_GRAPH_SURFACE_SHA256 = (
    "3b1dec7b6b85066ee31ae8fb25111a1394c459e7d1bb733aaa3cb48bdc156d64")
_GRAPH_SURFACE_SCHEMA = "mcblas_serving_graph_gemm_surface_v1"
_GRAPH_SURFACE_TOP_FIELDS = {
    "schema", "hardware_id", "software_revision", "duration_consumed",
    "provenance", "cells",
}
_GRAPH_SURFACE_CELL_FIELDS = {"n", "k", "entries", "evidence"}
_GRAPH_SURFACE_ENTRY_FIELDS = {
    "m", "tile", "block", "split_k", "stage4", "program",
}
_GRAPH_SURFACE_PROGRAM_PATTERN = re.compile(
    r"^mcblas__Mck_bf16gemm_tn_(\d+)x(\d+)x(\d+)_(\w+?)_(\d+)t(_4stage)?"
    r"_bf16_bf16_fp32_sb_0_(\d+)$")
_GRAPH_RUNTIME_MODES = ("FULL", "PIECEWISE")


def _positive_dim3(value: object, context: str) -> tuple[int, int, int]:
    if (not isinstance(value, list) or len(value) != 3
            or any(isinstance(dimension, bool)
                   or not isinstance(dimension, int)
                   for dimension in value)):
        raise TypeError(f"{context} must contain three integers")
    if any(dimension <= 0 for dimension in value):
        raise ValueError(f"{context} dimensions must be positive")
    return tuple(value)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_duration_fields(value: object, context: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{context} contains a non-string field")
            lowered_key = key.lower()
            if (any(part in lowered_key
                    for part in PROHIBITED_TIMING_FIELD_PARTS)
                    and key not in _ALLOWED_TIMING_BOUNDARY_FIELDS):
                raise ValueError(
                    f"{context} contains forbidden timing field {key!r}")
            if (key in _ALLOWED_TIMING_BOUNDARY_FIELDS
                    and item is not False):
                raise ValueError(f"{context}.{key} must be false")
            _reject_duration_fields(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_duration_fields(item, f"{context}[{index}]")


def _validate_manifest(manifest: Mapping[str, object]) -> None:
    _exact_fields(manifest, ARTIFACT_MANIFEST_FIELDS, "workload_manifest")
    _reject_duration_fields(manifest, "workload_manifest")
    if (manifest["schema_version"] != 2
            or manifest["artifact_type"] != "gemm_workload_candidate"
            or manifest["source_config"]
            != "configs/evaluation/experiments/gemm_launch_domain.json"
            or manifest["source_config_sha256"]
            != _sha256(_DOMAIN_CONFIG.read_bytes())
            or manifest["duration_consumed"] is not False
            or manifest["semantic_key_count"] != _EXPECTED_SEMANTIC_KEYS
            or manifest["physical_operation_count"]
            != _EXPECTED_PHYSICAL_OPERATIONS):
        raise ValueError("workload manifest identity differs")
    if tuple(_sequence(
            manifest["workload_key_fields"],
            "workload_manifest.workload_key_fields")) \
            != GEMM_WORKLOAD_KEY_FIELDS:
        raise ValueError("workload manifest key fields differ")
    if tuple(_sequence(
            manifest["physical_operation_fields"],
            "workload_manifest.physical_operation_fields")) \
            != ("m", "n", "k", "dtype", "has_bias"):
        raise ValueError("workload manifest physical-operation fields differ")

    config = _mapping(json.loads(_DOMAIN_CONFIG.read_bytes()), "domain config")
    scheduler_expected = _mapping(
        config["scheduler_contract"], "domain config scheduler_contract")
    scheduler = _mapping(
        manifest["scheduler_contract"],
        "workload_manifest.scheduler_contract")
    _exact_fields(
        scheduler, METAX_SCHEDULER_FIELDS,
        "workload_manifest.scheduler_contract")
    if scheduler != scheduler_expected:
        raise ValueError("workload manifest scheduler_contract differs")

    workload_keys = _sequence(
        manifest["workload_keys"], "workload_manifest.workload_keys")
    if (len(workload_keys) != _EXPECTED_SEMANTIC_KEYS
            or workload_keys != sorted(set(workload_keys))):
        raise ValueError("workload manifest keys are not complete and exact")
    for index, semantic_key in enumerate(workload_keys):
        if not isinstance(semantic_key, str):
            raise TypeError(
                f"workload_manifest.workload_keys[{index}] must be a string")
        validate_gemm_workload_key(semantic_key)

    hardware_domains = _mapping(
        manifest["hardware_domains"], "workload_manifest.hardware_domains")
    metax_domain = _mapping(
        hardware_domains["metax_c500"],
        "workload_manifest.hardware_domains.metax_c500")
    _exact_fields(
        metax_domain, HARDWARE_DOMAIN_FIELDS,
        "workload_manifest.hardware_domains.metax_c500")
    if (metax_domain["status"] != "exact"
            or metax_domain["semantic_key_count"] != _EXPECTED_SEMANTIC_KEYS
            or metax_domain["physical_operation_count"]
            != _EXPECTED_PHYSICAL_OPERATIONS):
        raise ValueError("MetaX C500 GEMM workload domain is incomplete")

    manifest_digest = _sha256(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    if manifest_digest != _EXPECTED_WORKLOAD_MANIFEST_SHA256:
        raise ValueError("workload manifest differs from the frozen exact domain")


def _expected_operation(semantic_key: str):
    validate_gemm_workload_key(semantic_key)
    values = json.loads(semantic_key)
    fields = dict(zip(GEMM_WORKLOAD_KEY_FIELDS, values, strict=True))
    operation_key = json.dumps([
        fields["b"], fields["m"], fields["n"], fields["k"],
        "bfloat16", "row_major_linear_tn", "MCBLAS_GEMM_DEFAULT",
    ], separators=(",", ":"))
    operation_id = "sha256:" + _sha256(operation_key.encode("utf-8"))
    workload = {
        "batch": fields["b"],
        "m": fields["m"],
        "n": fields["n"],
        "k": fields["k"],
        "dtype": "bfloat16",
        "layout": "row_major_linear_tn",
        "algorithm": "MCBLAS_GEMM_DEFAULT",
        "algorithm_value": -1,
    }
    return operation_id, operation_key, workload


def _classify_main_identity(name: str, context: str):
    if (_MAIN_GEMM_PATTERN.match(name) is not None
            or _UNALIGNED_GEMM_PATTERN.match(name) is not None):
        parsed = parse_complete_gemm_tile(name)
        if parsed is None:
            raise ValueError(f"{context} main kernel has no parsed tile")
        tile_m, tile_n, tile_k, pipeline_depth = parsed
        if tile_k is None:
            raise ValueError(f"{context} main kernel has no K tile")
        return (
            "gemm",
            {"m": tile_m, "n": tile_n, "k": tile_k},
            pipeline_depth,
        )
    lowered = name.lower()
    if any(identity in lowered for identity in _GEMV_IDENTITIES):
        return "gemv", None, None
    raise NotImplementedError(
        f"{context} has unsupported mcBLAS main kernel {name!r}")


def _is_source_backed_reducer(name: str) -> bool:
    return (
        name.startswith(_MANGLED_REDUCER_PREFIX)
        or name.startswith(_DEMANGLED_REDUCER_PREFIX)
    )


def _validate_workload(value: object, operation_key: str, context: str):
    workload = _mapping(value, context)
    expected_fields = {
        "batch", "m", "n", "k", "dtype", "layout", "algorithm",
        "algorithm_value",
    }
    if set(workload) != expected_fields:
        raise ValueError(f"{context} fields mismatch")
    for field in ("batch", "m", "n", "k"):
        require_positive_integer(workload, field)
    if workload["batch"] != 1:
        raise ValueError(f"{context}.batch must be one")
    if (workload["dtype"] != "bfloat16"
            or workload["layout"] != "row_major_linear_tn"
            or workload["algorithm"] != "MCBLAS_GEMM_DEFAULT"
            or workload["algorithm_value"] != -1):
        raise ValueError(f"{context} differs from the selector ABI")
    rebuilt_key = json.dumps([
        workload["batch"], workload["m"], workload["n"], workload["k"],
        workload["dtype"], workload["layout"], workload["algorithm"],
    ], separators=(",", ":"))
    if rebuilt_key != operation_key:
        raise ValueError(f"{context} disagrees with operation_key")
    return workload


@lru_cache(maxsize=1)
def _data() -> Mapping[tuple[object, ...], GemmLaunchSelection]:
    result = load_gemm_launch_selection(
        "metax_c500", software_revision="maca_3.7.2_mcblas_10")
    if len(result) != _EXPECTED_SEMANTIC_KEYS:
        raise ValueError(
            "MetaX C500 GEMM launch selection must contain 140544 rows")
    return result


def _launch(value: object, expected_ordinal: int, context: str):
    item = _mapping(value, context)
    if set(item) != METAX_LAUNCH_FIELDS:
        raise ValueError(f"{context} fields mismatch: {sorted(item)}")
    if item["record_type"] != "launch" or item["ordinal"] != expected_ordinal:
        raise ValueError(f"{context} has an invalid launch ordinal")
    if item["api"] not in ("mcLaunchKernel", "mcModuleLaunchKernel"):
        raise ValueError(f"{context} uses an unsupported launch API")
    if item["identity_source"] not in (
            "mcModuleGetFunction", "mcFuncGetName", "dladdr"):
        raise ValueError(f"{context} has invalid identity provenance")
    name = item["kernel_identity"]
    if not isinstance(name, str) or not name:
        raise ValueError(f"{context} kernel_identity must be non-empty")
    grid = _positive_dim3(item["grid"], f"{context}.grid")
    block = _positive_dim3(item["block"], f"{context}.block")
    dynamic_smem = require_non_negative_integer(
        item, "dynamic_shared_memory_bytes")
    if (item["static_shared_memory_bytes"] is not None
            or item["static_shared_memory_status"]
            != "not_exposed_by_launch_api"):
        raise ValueError(f"{context} invents static shared memory")
    return item, name, grid, block, dynamic_smem


def _validate_operation(operation_id: str, value: object, context: str):
    if _OPERATION_ID_PATTERN.fullmatch(operation_id) is None:
        raise ValueError(f"{context} has invalid operation id")
    operation = _mapping(value, context)
    if set(operation) != {
            "operation_key", "workload", "selector_result", "projection"}:
        raise ValueError(f"{context} fields mismatch")
    operation_key = operation["operation_key"]
    if not isinstance(operation_key, str) or not operation_key:
        raise ValueError(f"{context}.operation_key must be non-empty")
    if "sha256:" + _sha256(operation_key.encode("utf-8")) != operation_id:
        raise ValueError(f"{context} operation id/hash mismatch")
    workload = _validate_workload(
        operation["workload"], operation_key, f"{context}.workload")
    result = _mapping(operation["selector_result"], f"{context}.selector_result")
    if set(result) != METAX_RESULT_FIELDS:
        raise ValueError(f"{context}.selector_result fields mismatch")
    if result["gemm_executed"] is not False:
        raise ValueError(f"{context} executed GEMM")
    if result["duration_consumed"] is not False:
        raise ValueError(f"{context} consumed duration evidence")
    attempts = require_positive_integer(result, "selector_attempt_count")
    initial = require_positive_integer(result, "initial_expected_launch_count")
    validated = require_positive_integer(result, "validated_launch_count")
    stability = require_positive_integer(result, "stability_capture_count")
    launches_value = result["launches"]
    if not isinstance(launches_value, list) or len(launches_value) not in (1, 2):
        raise ValueError(f"{context} launch sequence is unsupported")
    launches = []
    for ordinal, launch_value in enumerate(launches_value, start=1):
        launch, _, _, _, _ = _launch(
            launch_value, ordinal, f"{context}.launches[{ordinal - 1}]")
        launches.append(launch)
    if initial != 1 or validated != len(launches):
        raise ValueError(f"{context} launch count evidence is inconsistent")
    if len(launches) == 1:
        if attempts != 1 or stability != 1:
            raise ValueError(f"{context} single-launch stability is invalid")
    elif attempts != 3 or stability != 2:
        raise ValueError(f"{context} multi-launch stability is invalid")

    _, main_name, main_grid, _, _ = _launch(
        launches[0], 1, f"{context}.launches[0]")
    execution_variant, tile, pipeline_depth = _classify_main_identity(
        main_name, context)
    split_k = main_grid[2]
    if len(launches) == 1:
        if split_k != 1:
            raise NotImplementedError(f"{context} split-K main lacks reducer")
        reducer_ordinal = None
        partial_buffer = None
    else:
        if execution_variant != "gemm" or split_k <= 1:
            raise ValueError(f"{context} reducer lacks a split tensor main")
        _, reducer_name, _, _, _ = _launch(
            launches[1], 2, f"{context}.launches[1]")
        if not _is_source_backed_reducer(reducer_name):
            raise NotImplementedError(f"{context} reducer ABI is unsupported")
        if launches[1]["api"] != "mcLaunchKernel":
            raise ValueError(f"{context} reducer launch API differs")
        reducer_ordinal = 2
        partial_buffer = {
            "contract": "mcblas_devdstelementwiseadd_bf16_fp32_partial",
            "element_bytes": 4,
            "dram_bytes_per_element": 4,
        }
    expected_projection = {
        "execution_variant": execution_variant,
        "main_ordinal": 1,
        "tile": tile,
        "pipeline_depth": pipeline_depth,
        "split_k": split_k,
        "reducer_ordinal": reducer_ordinal,
        "partial_buffer": partial_buffer,
    }
    projection = _mapping(operation["projection"], f"{context}.projection")
    if set(projection) != _PROJECTION_FIELDS:
        raise ValueError(f"{context}.projection fields mismatch")
    if projection != expected_projection:
        raise ValueError(f"{context} projection differs from raw launches")
    return operation, workload, launches, projection


@lru_cache(maxsize=1)
def _graph_surface() -> Mapping[tuple[int, int, int], GemmLaunchSelection]:
    """Load the pinned serving graph-replay GEMM surface overlay."""
    path = _ROOT / _GRAPH_SURFACE_REFERENCE
    raw = path.read_bytes()
    if _sha256(raw) != _GRAPH_SURFACE_SHA256:
        raise ValueError(
            "graph surface reference differs from the pinned artifact")
    document = _mapping(json.loads(raw), "graph surface reference")
    _exact_fields(document, _GRAPH_SURFACE_TOP_FIELDS, "graph surface")
    _reject_duration_fields(document, "graph surface")
    if (document["schema"] != _GRAPH_SURFACE_SCHEMA
            or document["hardware_id"] != "metax_c500"
            or document["software_revision"] != "vllm_metax_0.24.0_maca_3.8.2.2"
            or document["duration_consumed"] is not False):
        raise ValueError("graph surface reference identity differs")
    surface: dict[tuple[int, int, int], GemmLaunchSelection] = {}
    for cell_value in _sequence(document["cells"], "graph surface.cells"):
        cell = _mapping(cell_value, "graph surface cell")
        _exact_fields(cell, _GRAPH_SURFACE_CELL_FIELDS, "graph surface cell")
        n = require_positive_integer(cell, "n")
        k = require_positive_integer(cell, "k")
        entries = _sequence(cell["entries"], "graph surface cell entries")
        for entry_value in entries:
            entry = _mapping(entry_value, "graph surface entry")
            _exact_fields(
                entry, _GRAPH_SURFACE_ENTRY_FIELDS, "graph surface entry")
            m = require_positive_integer(entry, "m")
            split_k = require_positive_integer(entry, "split_k")
            block = require_positive_integer(entry, "block")
            tile = _positive_dim3(entry["tile"], "graph surface entry.tile")
            stage4 = entry["stage4"]
            if not isinstance(stage4, bool):
                raise TypeError("graph surface entry.stage4 must be boolean")
            program = entry["program"]
            if not isinstance(program, str):
                raise TypeError("graph surface entry.program must be string")
            parsed = _GRAPH_SURFACE_PROGRAM_PATTERN.match(program)
            if parsed is None:
                raise ValueError(
                    f"graph surface entry program {program!r} is not a "
                    "C500 BF16 mcBLAS GEMM identity")
            if ((int(parsed.group(2)), int(parsed.group(1)),
                 int(parsed.group(3))) != tuple(tile)
                    or bool(parsed.group(6)) != stage4):
                raise ValueError(
                    f"graph surface entry {program!r} disagrees with its "
                    "tile/stage fields")
            # sb_0_0 marks the no-split program class; sb_0_2 the split-K
            # class.  The split factor itself lives in the launch grid.
            if (split_k == 1) != (int(parsed.group(7)) == 0):
                raise ValueError(
                    f"graph surface entry {program!r} split class differs "
                    "from its split_k field")
            if _mcblas_main_block_threads(*tile) != block:
                raise ValueError(
                    f"graph surface entry {program!r} thread block differs "
                    "from the tile class contract")
            if int(parsed.group(5)) != block:
                raise ValueError(
                    f"graph surface entry {program!r} thread block differs "
                    "from its program name")
            key = (n, k, m)
            if key in surface:
                raise ValueError(
                    f"graph surface has duplicate entry for {key}")
            surface[key] = GemmLaunchSelection(
                tile_m=tile[0], tile_n=tile[1], tile_k=tile[2],
                pipeline_depth=None,
                split_k=split_k,
                reduction=("fp32_external" if split_k > 1 else "none"),
            )
    return surface


def _graph_surface_entry(
        shape: Mapping[str, int], dtype: Optional[str], variant: Optional[str],
        attributes: Mapping[str, object]) -> Optional[GemmLaunchSelection]:
    if variant != "gemm" or dtype != "bf16":
        return None
    if "graph_runtime_mode" not in attributes:
        return None
    mode = attributes["graph_runtime_mode"]
    if not isinstance(mode, str) or mode not in _GRAPH_RUNTIME_MODES:
        raise ValueError(
            f"unsupported serving graph runtime mode {mode!r}")
    b, m, n, k = require_shape(shape, ("b", "m", "n", "k"))
    if b != 1:
        return None
    if mode == "PIECEWISE":
        # The deployment launches mixed-step GEMMs with the exact scheduled
        # token count, not the padded capture bucket: per-(cell, exact
        # tokens) program identity is a pure function (zero conflicts over
        # ~1M kernels in 9 TP1 nsys captures, stable across runs), and every
        # bucket-keyed mixed-phase observation reproduces at its recorded
        # logical token count.  FULL decode graphs are captured at the
        # padded bucket, so the physical m remains the key there.
        if "graph_logical_tokens" not in attributes:
            raise ValueError(
                "PIECEWISE graph GEMM query lacks graph_logical_tokens")
        m = require_positive_integer(attributes, "graph_logical_tokens")
    surface = _graph_surface()
    key = (n, k, m)
    if key not in surface:
        return None
    return key, surface[key]


def _operation(
        shape: Mapping[str, int], dtype: Optional[str], variant: Optional[str],
        attributes: Mapping[str, object]):
    overlay = _graph_surface_entry(shape, dtype, variant, attributes)
    if overlay is not None:
        (n, k, m), selection = overlay
        return (
            f"mcblas_serving_graph_gemm_surface:(n={n},k={k},m={m})",
            selection,
        )
    key = gemm_launch_selection_key(shape, dtype, variant, attributes)
    entries = _data()
    if key not in entries:
        raise NotImplementedError(
            f"uncharacterized MetaX C500 GEMM launch key {key}")
    return key, entries[key]




def _mcblas_main_block_threads(tile_m: int, tile_n: int, tile_k: int) -> int:
    """Return the mcBLAS thread-block size encoded by the selected tile class.

    Dry-launch probes (`LD_PRELOAD`-interposed `mcblasGemmEx` calls, no
    kernel execution/timing) against real MetaX C500 hardware for every
    distinct `(tile_m, tile_n, tile_k)` class this LUT selects show the
    launch's thread-block size depends only on the selected logical tile,
    never on the workload shape or its origin -- exactly analogous to the
    A800 `_cublaslt_main_block_threads` table this mirrors.
    """

    threads_by_tile = {
        (64, 128, 256): 256,
        (128, 128, 128): 256,
        (32, 64, 256): 256,
        (256, 256, 64): 512,
        (192, 256, 64): 512,
        (128, 256, 64): 512,
        (32, 256, 128): 256,
        (64, 256, 128): 256,
        (48, 64, 256): 256,
        (64, 64, 256): 256,
        (80, 64, 256): 256,
        (96, 64, 256): 256,
        (160, 256, 64): 512,
        (224, 256, 64): 512,
        (80, 128, 128): 256,
        (64, 64, 64): 256,
        (80, 64, 128): 256,
        (128, 128, 64): 256,
        (64, 64, 128): 256,
        (16, 16, 64): 256,
        (64, 32, 64): 128,
    }
    tile = (tile_m, tile_n, tile_k)
    if tile not in threads_by_tile:
        raise ValueError(f"unsupported mcBLAS logical tile {tile!r}")
    return threads_by_tile[tile]


def _observed_main_launch(
        shape: Mapping[str, int], dtype: Optional[str], variant: Optional[str],
        attributes: Mapping[str, object]) -> LaunchConfig:
    operation_id, projection = _operation(
        shape, dtype, variant, attributes)
    context = f"MetaX C500 GEMM operation {operation_id}"
    split_k = projection.split_k
    execution_variant = projection.execution_variant
    if projection.tile_m is None:
        if (execution_variant != "gemv"
                or projection.pipeline_depth is not None or split_k != 1):
            raise ValueError(f"{context} has invalid tile-free GEMV facts")
        tiling = None
        _, _, n, _ = require_shape(shape, ("b", "m", "n", "k"))
        block_size = 256 if n > 256 else 512
        divisor = 16 if n > 256 else 64
        grid = (ceil_div(n, divisor), 1, 1)
    else:
        tile_shape = {
            "m": projection.tile_m,
            "n": projection.tile_n,
            "k": projection.tile_k,
        }
        tiling = TilingSpec(
            tile_shape=tile_shape,
            pipeline_depth=projection.pipeline_depth,
            split_k=split_k,
        )
        b, m, n, _ = require_shape(shape, ("b", "m", "n", "k"))
        grid = (
            ceil_div(b * m, tile_shape["m"]),
            ceil_div(n, tile_shape["n"]), split_k)
        block_size = _mcblas_main_block_threads(
            tile_shape["m"], tile_shape["n"], tile_shape["k"])
    if projection.reduction == "fp32_external":
        partial_bytes = 4
    elif projection.reduction == "none":
        partial_bytes = 0
    else:
        raise AssertionError("unhandled MetaX C500 GEMM reduction contract")
    partial_dram_bytes = partial_bytes
    return LaunchConfig(
        algorithm=_MAIN_ALGORITHM,
        grid=grid,
        block=(block_size, 1, 1),
        tiling=tiling,
        parameters={
            "compulsory_traffic": 0.0,
            "weight_tile_rows": 0.0,
            "mte2_tile_bytes": 0.0,
            "aiv_mixed": 0.0,
            "n_major_traffic": 0.0,
            "split_k_partial_bytes_per_element": float(partial_bytes),
            "split_k_partial_dram_bytes_per_element": float(
                partial_dram_bytes),
        },
        execution_variant=execution_variant,
    )


def select_gemm_reducer_launch(
        shape: Mapping[str, int], dtype: Optional[str], variant: Optional[str],
        attributes: Mapping[str, object], hardware) -> LaunchConfig:
    """Return the measured reducer paired with one exact split-K main."""

    del hardware
    if launch_policy(attributes) != "injected_launch":
        raise NotImplementedError(
            "analytical MetaX C500 split-K has no source-backed reducer")
    require_dtype(dtype, ("bf16",))
    b, m, n, _ = require_shape(shape, ("b", "m", "n", "k"))
    operation_id, projection = _operation(
        shape, dtype, variant, attributes)
    context = f"MetaX C500 GEMM operation {operation_id}"
    split_k = projection.split_k
    if split_k <= 1:
        raise ValueError(f"{context} does not require a reducer")
    if projection.reduction != "fp32_external":
        raise NotImplementedError(f"{context} reducer ABI is unsupported")
    partial_bytes = partial_dram_bytes = 4
    if partial_bytes != 4 or partial_dram_bytes != 4:
        raise ValueError(f"{context} reducer does not consume FP32 partials")
    # mcBLAS DevDstElementwiseAdd assigns one 512-thread workgroup to each
    # 32x16 output tile, matching the cuBLASLt splitKreduce tiling.
    return LaunchConfig(
        algorithm=_REDUCER_ALGORITHM,
        grid=(ceil_div(n, 32), ceil_div(b * m, 16), 1),
        block=(512, 1, 1),
        parameters={
            "split_k_partial_bytes_per_element": float(partial_bytes),
            "split_k_partial_dram_bytes_per_element": float(
                partial_dram_bytes),
        },
        execution_variant="gemm_reduce",
    )


def _analytical_launch(
        b: int, m: int, n: int, k: int,
        execution_variant: str) -> LaunchConfig:
    m_total = b * m
    if m_total == 1 or execution_variant == "gemv":
        block = 256 if n > 256 else 512
        divisor = 16 if n > 256 else 64
        tile = (
            {"m": 4, "n": 4, "k": 8}
            if n > 256 else {"m": 64, "n": 8, "k": 8})
        return LaunchConfig(
            algorithm=("mcblas_gemv_wave" if n > 256 else "mcblas_gemv"),
            grid=(ceil_div(n, divisor), 1, 1),
            block=(block, 1, 1),
            tiling=TilingSpec(tile_shape=tile, split_k=1),
            parameters={
                "compulsory_traffic": 0.0,
                "weight_tile_rows": 0.0,
                "mte2_tile_bytes": 0.0,
                "aiv_mixed": 0.0,
                "n_major_traffic": 0.0,
                "split_k_partial_bytes_per_element": 0.0,
                "split_k_partial_dram_bytes_per_element": 0.0,
            },
            execution_variant="gemv",
        )
    tile = {
        "m": 32 if m_total <= 32 else (64 if m_total <= 64 else 128),
        "n": 64 if n <= 64 else 128,
        "k": 32 if k <= 32 else 64,
    }
    return LaunchConfig(
        algorithm="mcblas_analytical_gemm",
        grid=(ceil_div(m_total, tile["m"]), ceil_div(n, tile["n"]), 1),
        block=(256, 1, 1),
        tiling=TilingSpec(tile_shape=tile, split_k=1),
        parameters={
            "compulsory_traffic": 0.0,
            "weight_tile_rows": 0.0,
            "mte2_tile_bytes": 0.0,
            "aiv_mixed": 0.0,
            "n_major_traffic": 0.0,
            "split_k_partial_bytes_per_element": 0.0,
            "split_k_partial_dram_bytes_per_element": 0.0,
        },
        execution_variant="gemm",
    )


def select_gemm_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> LaunchConfig:
    del hardware
    b, m, n, k = require_shape(shape, ("b", "m", "n", "k"))
    require_dtype(dtype, ("bf16",))
    execution_variant = require_variant(variant, ("gemm", "gemv"), "GEMM")
    policy = launch_policy(attributes)
    if policy == "analytical_launch":
        return _analytical_launch(b, m, n, k, execution_variant)
    if policy != "injected_launch":
        raise ValueError(f"unhandled launch policy: {policy!r}")
    return _observed_main_launch(shape, dtype, variant, attributes)
