"""Strict loader for canonical GEMM launch-selection CSV artifacts."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
import re
from typing import Mapping, NamedTuple

import yaml
import zstandard

from src.kernel.hardware_ids import canonical_hardware_id
from src.kernel.launch_config.gemm_workload import (
    GEMM_LAUNCH_SELECTION_KEY_FIELDS,
    GEMM_SEMANTIC_CONTEXT_FIELDS,
)

from src.kernel.launch_config.selection_primitives import (
    require_sha256 as _require_sha256,
)


GEMM_LAUNCH_FIELDS = (
    "tile_m", "tile_n", "tile_k", "pipeline_depth", "split_k",
    "reduction",
)
NVIDIA_A800_GEMM_RASTER_FIELDS = (
    "cta_raster_convention", "cta_raster_log_group", "cta_raster_serpentine",
    "cta_raster_k_slices",
)
ASCEND_910B_GEMM_PROGRAM_FIELDS = (
    "aic_workers", "worker_grid_m", "worker_grid_n", "worker_grid_k",
    "single_core_m", "single_core_n", "single_core_k",
    "base_m", "base_n", "base_k", "depth_a1", "depth_b1",
    "a_l1_k_tiles", "b_l1_k_tiles", "step_m", "step_n",
    "m_l2_regions", "n_l2_regions", "m_l2_block", "n_l2_block",
    "traversal_order", "a_uses_l2", "b_uses_l2",
)
# H200 rows observed from a profiler launch record that states the grid, the
# block and the separate split-K reducer launch but neither the tile nor the
# split count (a cuBLASLt ``nvjet`` kernel keeps both in its parameter block).
# Tile, pipeline and split columns stay null; the observed geometry is kept
# verbatim so a consumer can use it and must declare everything it cannot.
NVIDIA_H200_GEMM_OBSERVED_LAUNCH_FIELDS = (
    "launch_observability", "grid_x", "grid_y", "grid_z", "block_threads",
    "reducer_grid_x", "reducer_grid_y", "reducer_block_x", "reducer_block_y",
    "partial_dtype", "output_dtype",
)
# v6 typed-call extension (H200 only, backward compatible): real beta,
# output storage dtype, and batched-GEMM program axes become physical key
# fields so typed requests never alias ordinary rows.  Empty beta means 0;
# empty output_dtype means the input dtype; batch 1 with empty strides is an
# ordinary single-matrix call.
NVIDIA_H200_GEMM_V6_EXTENSION_FIELDS = (
    "beta", "output_dtype_key", "batch_count",
    "stride_a", "stride_b", "stride_c",
)
# The v6 physical selection key is the 13 shared GEMM workload key fields
# followed by the typed-call extension.
NVIDIA_H200_GEMM_V6_SELECTION_KEY_FIELDS = (
    *GEMM_LAUNCH_SELECTION_KEY_FIELDS,
    *NVIDIA_H200_GEMM_V6_EXTENSION_FIELDS,
)
GEMM_LAUNCH_CSV_FIELDS = (
    *GEMM_LAUNCH_SELECTION_KEY_FIELDS,
    *GEMM_LAUNCH_FIELDS,
)
GEMM_LAUNCH_CSV_FILENAME = "gemm_launch_selection.csv.zst"
GEMM_LAUNCH_MANIFEST_FILENAME = "gemm_launch_selection.manifest.json"
GEMM_LAUNCH_SCHEMA_VERSION = 2
NVIDIA_A800_GEMM_LAUNCH_SCHEMA_VERSION = 3
ASCEND_910B_GEMM_LAUNCH_SCHEMA_VERSION = 3
# v5 is the tiled-grid era of the H200 artifact (its rows carry no typed
# extension); v6 adds the typed-call key block.
NVIDIA_H200_GEMM_OBSERVED_SCHEMA_VERSION = 5
NVIDIA_H200_GEMM_LAUNCH_SCHEMA_VERSION = 6
GEMM_LAUNCH_ARTIFACT_TYPE = "gemm_launch_selection"
GEMM_BOOLEAN_ENCODING = "lowercase_true_false"
GEMM_NULL_ENCODING = "empty_string"
GEMM_LAUNCH_COMPRESSION = "zstd"
GEMM_LAUNCH_COMPRESSION_LEVEL = 1
_GEMM_LAUNCH_SOFTWARE_REVISIONS = {
    "nvidia_a800": "cuda_12.2",
    "nvidia_h200": "cuda_12.9",
    "metax_c500": "maca_3.7.2_mcblas_10",
    "ascend_910b": "cann_8.5.1",
}


def gemm_launch_software_revision(hardware_id: str) -> str:
    """Return the software revision a hardware's canonical GEMM artifact binds.

    The launch artifact's manifest records this revision, so every caller that
    loads one needs it; it is declared here, once, rather than restated.  A
    hardware without a canonical artifact has no revision and raises KeyError,
    the boundary the callers have always had.
    """

    return _GEMM_LAUNCH_SOFTWARE_REVISIONS[hardware_id]



class GemmLaunchRecordLayout(NamedTuple):
    """What one hardware's canonical GEMM launch record carries.

    Every record has the shared tile/pipeline/split/reduction columns.
    ``extension`` names the one hardware-specific block that follows them,
    ``extension_fields`` its CSV columns, and ``manifest_fields`` any extra
    manifest fields its capture declares:

    * ``"none"`` -- no extension;
    * ``"cta_raster"`` -- the cuBLASLt CTA raster program;
    * ``"cann_source_program"`` -- the CANN MatMul source program;
    * ``"observed_grid"`` -- a grid/block-only profiler observation whose
      tile, pipeline and split stay null.

    A consumer branches on the declared extension, so a new hardware whose
    records use an existing extension needs only a layout entry here.
    """

    schema_version: int
    extension: str
    extension_fields: tuple
    manifest_fields: frozenset


_NVIDIA_H200_MANIFEST_FIELDS = frozenset({
    "capture_runtime", "consumption_scope"})
_BASE_GEMM_LAUNCH_RECORD_LAYOUT = GemmLaunchRecordLayout(
    GEMM_LAUNCH_SCHEMA_VERSION, "none", (), frozenset())
_GEMM_LAUNCH_RECORD_LAYOUTS = {
    "nvidia_a800": GemmLaunchRecordLayout(
        NVIDIA_A800_GEMM_LAUNCH_SCHEMA_VERSION, "cta_raster",
        NVIDIA_A800_GEMM_RASTER_FIELDS, frozenset()),
    "nvidia_h200": GemmLaunchRecordLayout(
        NVIDIA_H200_GEMM_LAUNCH_SCHEMA_VERSION, "observed_grid",
        NVIDIA_H200_GEMM_OBSERVED_LAUNCH_FIELDS,
        _NVIDIA_H200_MANIFEST_FIELDS),
    "metax_c500": _BASE_GEMM_LAUNCH_RECORD_LAYOUT,
    "ascend_910b": GemmLaunchRecordLayout(
        ASCEND_910B_GEMM_LAUNCH_SCHEMA_VERSION, "cann_source_program",
        ASCEND_910B_GEMM_PROGRAM_FIELDS, frozenset()),
}


def gemm_launch_record_layout(hardware_id: str) -> GemmLaunchRecordLayout:
    """Return the declared GEMM launch record layout of one hardware.

    A hardware without a declared layout has no launch-record extension,
    which is the shared base layout.
    """

    canonical = canonical_hardware_id(hardware_id)
    if canonical in _GEMM_LAUNCH_RECORD_LAYOUTS:
        return _GEMM_LAUNCH_RECORD_LAYOUTS[canonical]
    return _BASE_GEMM_LAUNCH_RECORD_LAYOUT


_ROOT = Path(__file__).resolve().parents[3]
_LAUNCH_DATA_ROOT = _ROOT / "launch_data"
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_MANIFEST_FIELDS = {
    "schema_version", "artifact_type", "hardware_id", "hardware_revision",
    "software_revision", "generated_by", "evidence_sha256",
    "evidence_sources", "capture_baseline_commit", "domain_sha256",
    "csv_sha256", "csv_compressed_sha256", "compression", "row_count",
    "selection_key_fields", "launch_fields",
    "semantic_context_fields", "boolean_encoding", "null_encoding",
    "duration_consumed",
}
_HARDWARE_REVISION_FIELDS = {
    "config_path", "launch_contract", "launch_contract_sha256",
}
# H200 v4 manifests additionally record the capture runtime and the stack
# that may consume the rows, because the capture's CUDA/cuBLAS version is not
# the serving stack's (see ``consumption_scope``).
_CAPTURE_RUNTIME_FIELDS = {
    "torch_version", "torch_cuda_version", "vllm_version",
}
_EVIDENCE_SOURCE_FIELDS = {"kind", "sha256"}
_LAUNCH_SCHEDULING_FIELDS = (
    "parallel_unit",
    "parallel_width",
    "num_compute_units",
    "max_warps_per_unit",
    "max_blocks_per_unit",
    "max_regs_per_unit",
    "max_smem_per_unit_bytes",
    "max_smem_per_block_bytes",
    "max_threads_per_unit",
)
_LAUNCH_CACHE_ROLES = ("l1_cache", "l2_cache")
_ALLOWED_REDUCTIONS = {
    "none", "in_place", "fp32_external", "output_dtype_external",
}
_ALLOWED_LAUNCH_OBSERVABILITY = {"grid_block_only", "tiled_grid"}
_ALLOWED_OBSERVED_DTYPES = {"fp32", "bf16", "fp16"}
_ALLOWED_CTA_RASTER_CONVENTIONS = {
    "none", "cublaslt_m_group", "cutlass_n_group",
}
_PRELOADED_HARDWARE_CONTRACTS: dict[tuple[str, str], dict[str, object]] = {}


class CannGemmLaunchProgramSelection(NamedTuple):
    """Source-defined CANN execution geometry stored in the 910B CSV."""

    aic_workers: int
    worker_grid_m: int
    worker_grid_n: int
    worker_grid_k: int
    single_core_m: int
    single_core_n: int
    single_core_k: int
    base_m: int
    base_n: int
    base_k: int
    depth_a1: int
    depth_b1: int
    a_l1_k_tiles: int
    b_l1_k_tiles: int
    step_m: int
    step_n: int
    m_l2_regions: int
    n_l2_regions: int
    m_l2_block: int
    n_l2_block: int
    traversal_order: str
    a_uses_l2: bool
    b_uses_l2: bool


class CublasCtaRasterSelection(NamedTuple):
    """Source-decoded CTA-to-tile raster program stored in the A800 CSV.

    ``convention`` names the vendor mapping the kernel applies to its launch
    grid, ``log_group`` the log2 width of the grouped axis, and
    ``serpentine`` whether alternate groups reverse the ungrouped axis.
    """

    convention: str
    log_group: int
    serpentine: bool
    k_slices: int


class ObservedGridLaunchSelection(NamedTuple):
    """Recorded main/reducer geometry, with observability stated per row.

    ``grid``/``block`` are the main kernel's launch dimensions and
    ``reducer_grid``/``reducer_block`` those of the separate split-K reducer
    the same call launched; ``partial_dtype`` is the reducer's partial type
    and ``output_dtype`` the GEMM output type.
    """

    observability: str
    grid: tuple[int, int, int]
    block_threads: int
    reducer_grid: tuple[int, int] | None
    reducer_block: tuple[int, int] | None
    partial_dtype: str | None
    output_dtype: str


class GemmLaunchSelection(NamedTuple):
    """Physical launch choice reconstructed from one canonical CSV row."""

    tile_m: int | None
    tile_n: int | None
    tile_k: int | None
    pipeline_depth: int | None
    split_k: int | None
    reduction: str
    source_program: CannGemmLaunchProgramSelection | None = None
    cta_raster: CublasCtaRasterSelection | None = None
    observed_grid: ObservedGridLaunchSelection | None = None

    @property
    def execution_variant(self) -> str:
        if self.observed_grid is not None:
            return "gemm"
        return "gemv" if self.tile_m is None else "gemm"


def gemm_launch_fields(hardware_id: str) -> tuple[str, ...]:
    """Return the canonical CSV launch columns for one hardware target."""

    return (*GEMM_LAUNCH_FIELDS,
            *gemm_launch_record_layout(hardware_id).extension_fields)


def gemm_launch_csv_fields(hardware_id: str) -> tuple[str, ...]:
    return (*GEMM_LAUNCH_SELECTION_KEY_FIELDS, *gemm_launch_fields(hardware_id))


def gemm_launch_csv_fields_for_schema(
        hardware_id: str, schema_version: int,
        ) -> tuple[str, ...]:
    """CSV columns one hardware/schema-version artifact is allowed to carry.

    v4/v5-era H200 artifacts predate the typed-call extension; their CSV
    stops at the observed-grid block.  v6 candidates must declare the full
    block.
    """

    layout = gemm_launch_record_layout(hardware_id)
    extension_fields = layout.extension_fields
    if (hardware_id == "nvidia_h200"
            and schema_version != layout.schema_version):
        extension_fields = extension_fields[
            :len(NVIDIA_H200_GEMM_OBSERVED_LAUNCH_FIELDS)]
    key_fields = (GEMM_LAUNCH_SELECTION_KEY_FIELDS
                  if (hardware_id != "nvidia_h200"
                      or schema_version != layout.schema_version)
                  else NVIDIA_H200_GEMM_V6_SELECTION_KEY_FIELDS)
    return (*key_fields, *GEMM_LAUNCH_FIELDS, *extension_fields)


def gemm_launch_schema_version(hardware_id: str) -> int:
    """Return the schema version a canonical on-disk artifact must carry.

    The H200 canonical artifact is now the v6 typed-call extension; the
    v5 router observation is the historical predecessor.  Both v5 and v6
    artifacts remain loadable, but new canonical data must be v6.
    ``gemm_launch_record_layout`` keeps returning the v6 layout for
    in-memory candidate construction.
    """

    if canonical_hardware_id(hardware_id) == "nvidia_h200":
        return NVIDIA_H200_GEMM_LAUNCH_SCHEMA_VERSION
    return gemm_launch_record_layout(hardware_id).schema_version


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def compress_gemm_launch_csv(csv_bytes: bytes) -> bytes:
    """Encode canonical launch CSV bytes with the fixed storage contract."""

    if not isinstance(csv_bytes, bytes):
        raise TypeError("GEMM launch CSV content must be bytes")
    return zstandard.ZstdCompressor(
        level=GEMM_LAUNCH_COMPRESSION_LEVEL).compress(csv_bytes)


def decompress_gemm_launch_csv(compressed_bytes: bytes) -> bytes:
    """Decode one canonical launch CSV zstd frame without format fallback."""

    if not isinstance(compressed_bytes, bytes):
        raise TypeError("compressed GEMM launch CSV must be bytes")
    try:
        return zstandard.ZstdDecompressor().decompress(compressed_bytes)
    except zstandard.ZstdError as error:
        raise ValueError(
            f"GEMM launch CSV is not a valid zstd frame: {error}") from error




def _require_non_empty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _launch_cache_capacities(
        components: object,
        ) -> dict[str, int | None]:
    if not isinstance(components, list) or not components:
        raise ValueError("hardware launch contract requires components")
    capacities: dict[str, int | None] = {}
    pending = list(components)
    while pending:
        component = pending.pop()
        if not isinstance(component, Mapping):
            raise TypeError("hardware launch component must be a mapping")
        if "role" in component and component["role"] in _LAUNCH_CACHE_ROLES:
            role = component["role"]
            if role in capacities:
                raise ValueError(
                    f"hardware launch contract repeats role {role!r}")
            size_bytes = (
                component["size_bytes"] if "size_bytes" in component
                else None)
            if (size_bytes is not None
                    and (isinstance(size_bytes, bool)
                         or not isinstance(size_bytes, int)
                         or size_bytes <= 0)):
                raise ValueError(
                    f"hardware launch {role} size must be positive or null")
            capacities[role] = size_bytes
        if "children" in component:
            children = component["children"]
            if not isinstance(children, list):
                raise TypeError("hardware component children must be a list")
            pending.extend(children)
    if set(capacities) != set(_LAUNCH_CACHE_ROLES):
        raise ValueError(
            "hardware launch contract requires unique L1 and L2 resources")
    return {
        role: capacities[role]
        for role in _LAUNCH_CACHE_ROLES
    }


def gemm_launch_hardware_contract(
        config: object, *, hardware_id: str,
        ) -> dict[str, object]:
    """Project only hardware fields that can change a GEMM launch."""

    expected_hardware = canonical_hardware_id(hardware_id)
    if not isinstance(config, Mapping):
        raise TypeError("hardware launch config must be a mapping")
    if "hardware_id" not in config or config["hardware_id"] != expected_hardware:
        raise ValueError("hardware launch config identity differs")
    if "scheduling" not in config or not isinstance(
            config["scheduling"], Mapping):
        raise ValueError("hardware launch config requires scheduling")
    scheduling = config["scheduling"]
    missing = set(_LAUNCH_SCHEDULING_FIELDS) - set(scheduling)
    if missing:
        raise ValueError(
            f"hardware launch scheduling fields are missing: {sorted(missing)}")
    if (not isinstance(scheduling["parallel_unit"], str)
            or not scheduling["parallel_unit"]):
        raise ValueError("hardware launch parallel_unit must be non-empty")
    for field in _LAUNCH_SCHEDULING_FIELDS[1:]:
        value = scheduling[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"hardware launch scheduling.{field} must be positive")
    if "components" not in config:
        raise ValueError("hardware launch config requires components")
    return {
        "schema_version": 1,
        "hardware_id": expected_hardware,
        "scheduling": {
            field: scheduling[field]
            for field in _LAUNCH_SCHEDULING_FIELDS
        },
        "cache_capacity_bytes": _launch_cache_capacities(
            config["components"]),
    }


def gemm_launch_hardware_contract_sha256(contract: object) -> str:
    """Hash the canonical structured launch contract, not YAML bytes."""

    payload = json.dumps(
        contract, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def load_gemm_launch_hardware_contract(
        config_path: Path, *, hardware_id: str,
        ) -> dict[str, object]:
    """Load and project one hardware YAML into its launch-only contract."""

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return gemm_launch_hardware_contract(config, hardware_id=hardware_id)


def _parse_positive_integer(value: str, context: str) -> int:
    if not value or not value.isascii() or not value.isdecimal():
        raise ValueError(f"{context} must be a canonical positive integer")
    parsed = int(value)
    if parsed <= 0 or str(parsed) != value:
        raise ValueError(f"{context} must be a canonical positive integer")
    return parsed


def _parse_non_negative_integer(value: str, context: str) -> int:
    if not value or not value.isascii() or not value.isdecimal():
        raise ValueError(f"{context} must be a canonical non-negative integer")
    parsed = int(value)
    if parsed < 0 or str(parsed) != value:
        raise ValueError(f"{context} must be a canonical non-negative integer")
    return parsed


def _parse_optional_positive_integer(value: str, context: str) -> int | None:
    if value == "":
        return None
    return _parse_positive_integer(value, context)


def _parse_boolean(value: str, context: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{context} must use lowercase true or false")


def _validate_manifest(
        manifest: object, hardware_id: str, software_revision: str,
        csv_bytes: bytes, compressed_csv_bytes: bytes, manifest_path: Path,
        hardware_contract: dict[str, object] | None = None) -> int:
    layout = gemm_launch_record_layout(hardware_id)
    expected_fields = set(_MANIFEST_FIELDS) | layout.manifest_fields
    if not isinstance(manifest, dict) or set(manifest) != expected_fields:
        actual = sorted(manifest) if isinstance(manifest, dict) else type(manifest)
        raise ValueError(
            f"{manifest_path} fields differ: expected={sorted(expected_fields)}, "
            f"actual={actual}")
    if "capture_runtime" in layout.manifest_fields:
        runtime = manifest["capture_runtime"]
        if (not isinstance(runtime, dict)
                or set(runtime) != _CAPTURE_RUNTIME_FIELDS):
            raise ValueError(f"{manifest_path} capture_runtime fields differ")
        for field in sorted(_CAPTURE_RUNTIME_FIELDS):
            _require_non_empty_string(
                runtime[field], f"{manifest_path}.capture_runtime.{field}")
        _require_non_empty_string(
            manifest["consumption_scope"],
            f"{manifest_path}.consumption_scope")
    supported_versions = {gemm_launch_schema_version(hardware_id)}
    if hardware_id == "nvidia_h200":
        # v4 contains only the original router rows, with the same columns;
        # v5 is the tiled-grid era (its rows carry no typed extension);
        # v6 is the candidate typed-call extension.
        supported_versions.update(
            {4, NVIDIA_H200_GEMM_OBSERVED_SCHEMA_VERSION,
             NVIDIA_H200_GEMM_LAUNCH_SCHEMA_VERSION})
    if manifest["schema_version"] not in supported_versions:
        raise ValueError(
            f"{manifest_path} artifacts require schema v5 or later")
    if (manifest["artifact_type"] != GEMM_LAUNCH_ARTIFACT_TYPE
            or manifest["hardware_id"] != hardware_id
            or manifest["software_revision"] != software_revision
            or manifest["compression"] != GEMM_LAUNCH_COMPRESSION
            or manifest["boolean_encoding"] != GEMM_BOOLEAN_ENCODING
            or manifest["null_encoding"] != GEMM_NULL_ENCODING
            or manifest["duration_consumed"] is not False):
        raise ValueError(f"{manifest_path} identity or evidence boundary differs")
    expected_key_fields = list(GEMM_LAUNCH_SELECTION_KEY_FIELDS)
    if hardware_id == "nvidia_h200":
        if manifest["schema_version"] == NVIDIA_H200_GEMM_LAUNCH_SCHEMA_VERSION:
            # v6 candidates must declare the full typed-call key extension.
            expected_key_fields = list(NVIDIA_H200_GEMM_V6_SELECTION_KEY_FIELDS)
        elif manifest["schema_version"] == NVIDIA_H200_GEMM_OBSERVED_SCHEMA_VERSION:
            # v5-era artifacts predate the typed extension; their keys stay
            # at the shared 13-field identity.
            expected_key_fields = list(GEMM_LAUNCH_SELECTION_KEY_FIELDS)
        else:
            # v4-era artifacts predate the typed extension; their keys stay
            # at the shared 13-field identity.  Anything else must not
            # smuggle a v6 typed key identity: fail closed rather than guess.
            if manifest["schema_version"] != 4:
                raise ValueError(
                    f"{manifest_path} selection key fields differ for schema "
                    f"v{manifest['schema_version']}")
    if manifest["selection_key_fields"] != expected_key_fields:
        raise ValueError(f"{manifest_path} selection key fields differ")
    expected_launch_fields = list(gemm_launch_fields(hardware_id))
    if (hardware_id == "nvidia_h200"
            and manifest["schema_version"] in supported_versions
            and manifest["schema_version"] != NVIDIA_H200_GEMM_LAUNCH_SCHEMA_VERSION):
        # v4/v5-era artifacts predate the typed extension; their launch_fields
        # stop at the observed-grid block.  v6 candidates must declare the
        # full extension block.
        expected_launch_fields = list(
            gemm_launch_csv_fields_for_schema(
                hardware_id, manifest["schema_version"])[
                    len(expected_key_fields):])
    if manifest["launch_fields"] != expected_launch_fields:
        raise ValueError(f"{manifest_path} launch fields differ")
    if manifest["semantic_context_fields"] != list(
            GEMM_SEMANTIC_CONTEXT_FIELDS):
        raise ValueError(f"{manifest_path} semantic context fields differ")
    expected_generator = (
        f"scripts/launch_data/{hardware_id}_launch_generator.py")
    if manifest["generated_by"] != expected_generator:
        raise ValueError(f"{manifest_path} generated_by differs")
    _require_sha256(manifest["evidence_sha256"], "evidence_sha256")
    _require_sha256(manifest["domain_sha256"], "domain_sha256")
    if manifest["csv_sha256"] != _sha256_bytes(csv_bytes):
        raise ValueError(f"{manifest_path} csv_sha256 does not match CSV bytes")
    if manifest["csv_compressed_sha256"] != _sha256_bytes(
            compressed_csv_bytes):
        raise ValueError(
            f"{manifest_path} csv_compressed_sha256 does not match zstd bytes")
    baseline = manifest["capture_baseline_commit"]
    if not isinstance(baseline, str) or _COMMIT_PATTERN.fullmatch(baseline) is None:
        raise ValueError(f"{manifest_path} capture baseline must be a commit")
    row_count = manifest["row_count"]
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise ValueError(f"{manifest_path} row_count must be positive")
    hardware_revision = manifest["hardware_revision"]
    if (not isinstance(hardware_revision, dict)
            or set(hardware_revision) != _HARDWARE_REVISION_FIELDS):
        raise ValueError(f"{manifest_path} hardware_revision fields differ")
    expected_config_path = f"configs/hardware/{hardware_id}.yaml"
    if hardware_revision["config_path"] != expected_config_path:
        raise ValueError(f"{manifest_path} hardware config path differs")
    launch_contract = hardware_contract
    if launch_contract is None:
        launch_contract = load_gemm_launch_hardware_contract(
            _ROOT / expected_config_path, hardware_id=hardware_id)
    if hardware_revision["launch_contract"] != launch_contract:
        raise ValueError(f"{manifest_path} hardware launch contract differs")
    expected_contract_sha256 = gemm_launch_hardware_contract_sha256(
        launch_contract)
    if (hardware_revision["launch_contract_sha256"]
            != expected_contract_sha256):
        raise ValueError(
            f"{manifest_path} hardware launch contract SHA256 differs")
    evidence_sources = manifest["evidence_sources"]
    if not isinstance(evidence_sources, list) or not evidence_sources:
        raise ValueError(f"{manifest_path} evidence_sources must be non-empty")
    seen_kinds = set()
    for index, source in enumerate(evidence_sources):
        context = f"{manifest_path}.evidence_sources[{index}]"
        if not isinstance(source, dict) or set(source) != _EVIDENCE_SOURCE_FIELDS:
            raise ValueError(f"{context} fields differ")
        kind = _require_non_empty_string(source["kind"], f"{context}.kind")
        if kind in seen_kinds:
            raise ValueError(f"{manifest_path} repeats evidence kind {kind!r}")
        seen_kinds.add(kind)
        _require_sha256(source["sha256"], f"{context}.sha256")
    primary_kinds = {"expanded_evidence"}
    if hardware_id == "nvidia_h200" and manifest["schema_version"] == 5:
        primary_kinds.add("compact_rank0_launch_facts")
    if (evidence_sources[0]["kind"] not in primary_kinds
            or evidence_sources[0]["sha256"] != manifest["evidence_sha256"]):
        raise ValueError(
            f"{manifest_path} primary evidence source SHA256 differs")
    return row_count


def _parse_csv(
        csv_bytes: bytes, expected_row_count: int, csv_path: Path,
        hardware_id: str, schema_version: int,
        ) -> dict[tuple[object, ...], GemmLaunchSelection]:
    layout = gemm_launch_record_layout(hardware_id)
    try:
        csv_text = csv_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{csv_path} must be UTF-8") from error
    if "\r" in csv_text or not csv_text.endswith("\n"):
        raise ValueError(f"{csv_path} must use LF endings and end with a newline")
    # Every canonical field is either an ASCII token or an integer, so quoted
    # CSV fields cannot be part of the canonical serialization.  Splitting
    # the verified unquoted form avoids constructing csv.reader state for all
    # 140k Ascend launch rows during every cold evaluator process.
    if '"' in csv_text:
        raise ValueError(f"{csv_path} must not contain quoted CSV fields")
    lines = csv_text.splitlines()
    if not lines:
        raise ValueError(f"{csv_path} is empty")
    header = lines[0].split(",")
    csv_fields = gemm_launch_csv_fields_for_schema(hardware_id, schema_version)
    if header != list(csv_fields):
        raise ValueError(f"{csv_path} header differs from the canonical schema")
    result: dict[tuple[object, ...], GemmLaunchSelection] = {}
    previous_key: tuple[object, ...] | None = None
    positive_values: dict[str, int] = {}
    optional_positive_values: dict[str, int | None] = {}

    def positive(value: str, field: str, line_number: int) -> int:
        if value not in positive_values:
            positive_values[value] = _parse_positive_integer(
                value, f"{csv_path}:{line_number}.{field}")
        return positive_values[value]

    def optional_positive(value: str, field: str, line_number: int) -> int | None:
        if value not in optional_positive_values:
            optional_positive_values[value] = _parse_optional_positive_integer(
                value, f"{csv_path}:{line_number}.{field}")
        return optional_positive_values[value]

    for line_number, line in enumerate(lines[1:], start=2):
        context = f"{csv_path}:{line_number}"
        row = line.split(",")
        if len(row) != len(csv_fields):
            raise ValueError(f"{context} has {len(row)} fields")
        dtype, variant = row[0], row[1]
        if dtype not in ("bf16", "fp16", "fp32"):
            raise ValueError(f"{context} has unsupported dtype {dtype!r}")
        if variant not in ("gemm", "gemv", "batched_gemm"):
            raise ValueError(f"{context} has unsupported variant {variant!r}")
        b = positive(row[2], "b", line_number)
        m = positive(row[3], "m", line_number)
        n = positive(row[4], "n", line_number)
        k = positive(row[5], "k", line_number)
        a_layout, b_layout, c_layout = row[6:9]
        if (a_layout, b_layout, c_layout) != (
                "row_major", "row_major", "row_major"):
            raise ValueError(f"{context} has unsupported GEMM layouts")
        trans_a = _parse_boolean(row[9], f"{context}.trans_a")
        trans_b = _parse_boolean(row[10], f"{context}.trans_b")
        epilogue = row[11]
        if epilogue not in ("none", "bias"):
            raise ValueError(f"{context} has unsupported epilogue {epilogue!r}")
        has_bias = _parse_boolean(row[12], f"{context}.has_bias")
        if has_bias != (epilogue == "bias"):
            raise ValueError(f"{context} has_bias conflicts with epilogue")
        v6_extension = None
        if hardware_id == "nvidia_h200" and schema_version >= 6:
            v6_extension = _parse_h200_v6_extension(
                row[len(NVIDIA_H200_GEMM_V6_SELECTION_KEY_FIELDS)
                    - len(NVIDIA_H200_GEMM_V6_EXTENSION_FIELDS):
                    len(NVIDIA_H200_GEMM_V6_SELECTION_KEY_FIELDS)],
                context)
        launch_offset = (len(GEMM_LAUNCH_SELECTION_KEY_FIELDS)
                         if schema_version < 6
                         else len(NVIDIA_H200_GEMM_V6_SELECTION_KEY_FIELDS))
        tile_m = optional_positive(row[launch_offset], "tile_m", line_number)
        tile_n = optional_positive(row[launch_offset + 1], "tile_n", line_number)
        tile_k = optional_positive(row[launch_offset + 2], "tile_k", line_number)
        tile = (tile_m, tile_n, tile_k)
        if ((tile_m is None or tile_n is None or tile_k is None)
                and (tile_m is not None or tile_n is not None
                     or tile_k is not None)):
            raise ValueError(f"{context} tile must be fully present or fully null")
        pipeline_depth = optional_positive(row[launch_offset + 3], "pipeline_depth", line_number)
        if tile[0] is None and pipeline_depth is not None:
            raise ValueError(f"{context} untiled launch must not declare a pipeline")
        reduction = row[launch_offset + 5]
        if reduction not in _ALLOWED_REDUCTIONS:
            raise ValueError(f"{context} has unsupported reduction {reduction!r}")
        observed_grid = None
        if layout.extension == "observed_grid":
            observed_grid = _parse_observed_grid(
                row[launch_offset + len(GEMM_LAUNCH_FIELDS):],
                context, variant)
            if schema_version == 4 and observed_grid.observability != "grid_block_only":
                raise ValueError("H200 tiled-grid observations require schema v5")
        if (observed_grid is not None
                and observed_grid.observability == "grid_block_only"):
            # v5-era grid-block-only rows (the published router artifact)
            # are bound to the FP32-external reduction contract.  v6
            # grid-block-only rows may carry their real reduction facts
            # (including "none") so a capture that cannot see the tile can
            # still record what it observed without inventing tiling.  In
            # both eras unknown tile/pipeline/split facts must stay null;
            # only the reducer geometry/partial dtype that were actually
            # observed may be present, and they must agree with the row's
            # reduction kind.
            if (any(value is not None for value in tile)
                    or pipeline_depth is not None or row[launch_offset + 4] != ""):
                raise ValueError(
                    f"{context} grid-block-only launch must leave tile, "
                    "pipeline and split null")
            if schema_version < 6 and reduction != "fp32_external":
                raise ValueError(
                    f"{context} grid-block-only launch must reduce FP32 "
                    "partials externally")
            if reduction in ("fp32_external", "output_dtype_external"):
                if observed_grid.reducer_grid is None:
                    raise ValueError(
                        f"{context} grid-block-only external reduction lacks "
                        "the observed reducer geometry")
            elif observed_grid.reducer_grid is not None:
                raise ValueError(
                    f"{context} grid-block-only launch declares a reducer "
                    "without an external reduction")
            split_k = None
        else:
            split_k = positive(row[launch_offset + 4], "split_k", line_number)
            if ((split_k == 1) != (reduction == "none")
                    or tile[0] is None and split_k != 1):
                raise ValueError(f"{context} split/reduction contract differs")
            if observed_grid is not None:
                if tile[0] is None or pipeline_depth is None:
                    raise ValueError(f"{context} tiled-grid launch lacks tiling")
                external = reduction in ("fp32_external", "output_dtype_external")
                if external != (observed_grid.reducer_grid is not None):
                    raise ValueError(f"{context} observed reducer differs from reduction")
                if observed_grid.output_dtype != dtype:
                    raise ValueError(f"{context} tiled-grid output dtype differs")
                expected_partial = (
                    "fp32" if reduction == "fp32_external"
                    else dtype if reduction == "output_dtype_external" else None)
                if observed_grid.partial_dtype != expected_partial:
                    raise ValueError(f"{context} partial dtype differs from reduction")
        if (observed_grid is not None
                and observed_grid.observability == "grid_block_only"
                and schema_version >= 6 and reduction == "none"):
            # A v6 grid-block-only row may record its true reduction kind
            # (including "none"); without a reducer launch the row must not
            # carry reducer geometry or an FP32 partial dtype, because both
            # facts describe the split-K reduction the launch did not have.
            if (observed_grid.reducer_grid is not None
                    or observed_grid.partial_dtype is not None):
                raise ValueError(
                    f"{context} grid-block-only launch declares a reducer "
                    "without an external reduction")
        if (observed_grid is not None and v6_extension is not None
                and observed_grid.observability == "grid_block_only"):
            # A grid-only observation records the true output storage of
            # the launch; the v6 typed key must state the same storage or
            # the row silently mis-describes the captured call.
            _, out_dtype_key, _, *_ = v6_extension
            observed_output = observed_grid.output_dtype
            effective_key_dtype = out_dtype_key or dtype
            if effective_key_dtype != observed_output:
                raise ValueError(
                    f"{context} grid-only output_dtype_key differs from the "
                    "observed output dtype")
        key = (
            dtype, variant, b, m, n, k, a_layout, b_layout, c_layout,
            trans_a, trans_b, epilogue, has_bias,
        )
        if v6_extension is not None:
            key = (*key, *v6_extension)
        if previous_key is not None and key <= previous_key:
            raise ValueError(f"{context} physical keys are not strictly sorted")
        previous_key = key
        cta_raster = None
        if layout.extension == "cta_raster":
            raster_values = row[len(GEMM_LAUNCH_CSV_FIELDS):]
            convention = raster_values[0]
            if convention not in _ALLOWED_CTA_RASTER_CONVENTIONS:
                raise ValueError(
                    f"{context}.cta_raster_convention is unsupported")
            log_group = _parse_non_negative_integer(
                raster_values[1], f"{context}.cta_raster_log_group")
            serpentine = _parse_boolean(
                raster_values[2], f"{context}.cta_raster_serpentine")
            if convention == "none" and (log_group or serpentine):
                raise ValueError(
                    f"{context} declares a raster program without a "
                    f"convention")
            if convention == "cutlass_n_group" and serpentine:
                raise ValueError(
                    f"{context} CUTLASS raster has no serpentine reversal")
            if tile[0] is None and convention != "none":
                raise ValueError(
                    f"{context} untiled launch must not declare a raster")
            k_slices = positive(
                raster_values[3], "cta_raster_k_slices", line_number)
            if k_slices > split_k:
                raise ValueError(
                    f"{context} raster K slices exceed the retained K split")
            cta_raster = CublasCtaRasterSelection(
                convention, log_group, serpentine, k_slices)
        source_program = None
        if layout.extension == "cann_source_program":
            program_values = row[len(GEMM_LAUNCH_CSV_FIELDS):]
            integer_fields = ASCEND_910B_GEMM_PROGRAM_FIELDS[:20]
            parsed_program = []
            for program_index, field in enumerate(integer_fields):
                value = program_values[program_index]
                if value not in positive_values:
                    positive_values[value] = _parse_positive_integer(
                        value, f"{csv_path}:{line_number}.{field}")
                parsed_program.append(positive_values[value])
            traversal_order = program_values[20]
            if traversal_order not in {
                    "persistent_n_then_m", "m_then_n_snake",
                    "n_then_m_snake"}:
                raise ValueError(f"{context}.traversal_order is unsupported")
            source_program = CannGemmLaunchProgramSelection(
                *parsed_program, traversal_order,
                _parse_boolean(program_values[21], f"{context}.a_uses_l2"),
                _parse_boolean(program_values[22], f"{context}.b_uses_l2"))
            if source_program.worker_grid_m * source_program.worker_grid_n * (
                    source_program.worker_grid_k) != source_program.aic_workers:
                raise ValueError(f"{context} program grid differs from workers")
            if source_program.worker_grid_k != split_k:
                raise ValueError(f"{context} program split-K differs from launch")
            if (source_program.base_m, source_program.base_n,
                    source_program.base_k) != tile:
                raise ValueError(f"{context} program base tile differs from launch")
        result[key] = GemmLaunchSelection(
            tile[0], tile[1], tile[2], pipeline_depth, split_k, reduction,
            source_program, cta_raster, observed_grid)
    if len(result) != expected_row_count:
        raise ValueError(
            f"{csv_path} row count {len(result)} differs from manifest "
            f"{expected_row_count}")
    return result


def _parse_h200_v6_extension(values: list[str], context: str) -> tuple:
    """Parse the v6 typed-call key extension (H200 only).

    ``beta`` is empty (0) or a positive decimal; ``output_dtype_key`` is
    empty (input dtype) or one of bf16/fp16/fp32; ``batch_count`` is a
    positive integer (1 for ordinary calls); strides are empty or positive
    integers and must be present exactly when batch_count > 1.  Physically
    equivalent spellings are normalized at this single boundary so two
    different lexical forms of the same beta (for example ``.5`` and
    ``0.5``) never alias distinct LUT rows.
    """
    if len(values) != len(NVIDIA_H200_GEMM_V6_EXTENSION_FIELDS):
        raise AssertionError("H200 v6 extension columns disappeared")
    beta_text = values[0]
    if beta_text != "":
        try:
            beta_value = float(beta_text)
        except ValueError:
            raise ValueError(
                f"{context}.beta must be a decimal string") from None
        _require_h200_decimal(beta_text, f"{context}.beta")
        if not (beta_value > 0.0):
            raise ValueError(f"{context}.beta must be positive")
        normalized = f"{beta_value:g}"
        beta_text = "" if normalized == "0" else normalized
    out_dtype = values[1]
    if out_dtype not in ("", "bf16", "fp16", "fp32"):
        raise ValueError(f"{context}.output_dtype_key is unsupported")
    batch = _parse_positive_integer(values[2], f"{context}.batch_count")
    strides = []
    for field, value in zip(("stride_a", "stride_b", "stride_c"),
                            values[3:]):
        if value == "":
            strides.append(None)
        else:
            strides.append(_parse_positive_integer(
                value, f"{context}.{field}"))
    if batch > 1 and any(s is None for s in strides):
        raise ValueError(f"{context} batched call requires all strides")
    if batch == 1 and any(s is not None for s in strides):
        raise ValueError(f"{context} ordinary call must not declare strides")
    return (beta_text, out_dtype, batch, *strides)


def _require_h200_decimal(value: str, context: str) -> None:
    if not value or not value.isascii():
        raise ValueError(f"{context} must be a decimal string")
    seen_dot = False
    for char in value:
        if char == "." and not seen_dot:
            seen_dot = True
            continue
        if not char.isdigit():
            raise ValueError(f"{context} must be a decimal string")


def _parse_observed_grid(
        values: list[str], context: str, variant: str,
        ) -> ObservedGridLaunchSelection:
    if len(values) != len(NVIDIA_H200_GEMM_OBSERVED_LAUNCH_FIELDS):
        raise AssertionError("H200 observed launch columns disappeared")
    observability = values[0]
    if observability not in _ALLOWED_LAUNCH_OBSERVABILITY:
        raise ValueError(f"{context}.launch_observability is unsupported")
    if variant not in ("gemm", "gemv", "batched_gemm"):
        raise ValueError(f"{context} observed launch must be a GEMM/GEMV key")
    if observability == "grid_block_only" and variant not in ("gemm", "batched_gemm"):
        raise ValueError(f"{context} grid-block-only launch must be a GEMM")
    main = tuple(
        _parse_positive_integer(value, f"{context}.{field}")
        for field, value in zip(
            NVIDIA_H200_GEMM_OBSERVED_LAUNCH_FIELDS[1:5], values[1:5]))
    reducer_values = values[5:9]
    if all(value == "" for value in reducer_values):
        reducer_grid = reducer_block = None
    elif any(value == "" for value in reducer_values):
        raise ValueError(f"{context} reducer geometry is only partly present")
    else:
        reducer = tuple(_parse_positive_integer(value, f"{context}.reducer")
                        for value in reducer_values)
        reducer_grid, reducer_block = reducer[:2], reducer[2:]
    partial_dtype = None if values[9] == "" else values[9]
    output_dtype = values[10]
    if ((partial_dtype is not None and partial_dtype not in _ALLOWED_OBSERVED_DTYPES)
            or output_dtype not in _ALLOWED_OBSERVED_DTYPES):
        raise ValueError(f"{context} observed partial/output dtype differs")
    if observability == "grid_block_only" and (
            partial_dtype is not None or reducer_grid is not None) and (
            partial_dtype != "fp32" or reducer_grid is None):
        raise ValueError(f"{context} grid-block-only needs an FP32 reducer")
    return ObservedGridLaunchSelection(
        observability, main[:3], main[3],
        reducer_grid, reducer_block, partial_dtype, output_dtype)


def load_gemm_launch_selection_files(
        csv_path: Path, manifest_path: Path, *, hardware_id: str,
        software_revision: str,
        hardware_contract: dict[str, object] | None = None,
        ) -> Mapping[tuple[object, ...], GemmLaunchSelection]:
    """Load and strictly validate one explicit CSV/manifest pair."""

    expected_hardware = canonical_hardware_id(hardware_id)
    csv_path = Path(csv_path)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    compressed_csv_bytes = csv_path.read_bytes()
    csv_bytes = decompress_gemm_launch_csv(compressed_csv_bytes)
    row_count = _validate_manifest(
        manifest, expected_hardware, software_revision, csv_bytes,
        compressed_csv_bytes,
        manifest_path, hardware_contract)
    return _parse_csv(csv_bytes, row_count, csv_path, expected_hardware,
                      manifest["schema_version"])


@lru_cache(maxsize=None)
def load_gemm_launch_selection(
        hardware_id: str, *, software_revision: str,
        ) -> Mapping[tuple[object, ...], GemmLaunchSelection]:
    """Load one hardware's canonical GEMM selector with process-local caching."""

    expected_hardware = canonical_hardware_id(hardware_id)
    directory = _LAUNCH_DATA_ROOT / expected_hardware
    contract_key = (expected_hardware, software_revision)
    hardware_contract = None
    if contract_key in _PRELOADED_HARDWARE_CONTRACTS:
        hardware_contract = _PRELOADED_HARDWARE_CONTRACTS[contract_key]
    return load_gemm_launch_selection_files(
        directory / GEMM_LAUNCH_CSV_FILENAME,
        directory / GEMM_LAUNCH_MANIFEST_FILENAME,
        hardware_id=expected_hardware,
        software_revision=software_revision,
        hardware_contract=hardware_contract,
    )


def preload_gemm_launch_selection(
        hardware_id: str, *, hardware_config: object | None = None,
        ) -> Mapping[tuple[object, ...], GemmLaunchSelection]:
    """Load a supported immutable selector before evaluation workers fork."""

    expected_hardware = canonical_hardware_id(hardware_id)
    if expected_hardware not in _GEMM_LAUNCH_SOFTWARE_REVISIONS:
        raise ValueError(
            "GEMM launch preloading is unsupported for hardware "
            f"{expected_hardware!r}")
    software_revision = _GEMM_LAUNCH_SOFTWARE_REVISIONS[expected_hardware]
    if hardware_config is not None:
        contract_key = (expected_hardware, software_revision)
        hardware_contract = gemm_launch_hardware_contract(
            hardware_config, hardware_id=expected_hardware)
        if contract_key in _PRELOADED_HARDWARE_CONTRACTS:
            if _PRELOADED_HARDWARE_CONTRACTS[contract_key] != hardware_contract:
                raise ValueError(
                    "preloaded GEMM launch contract differs within one "
                    "interpreter")
        else:
            _PRELOADED_HARDWARE_CONTRACTS[contract_key] = hardware_contract
    return load_gemm_launch_selection(
        expected_hardware,
        software_revision=software_revision,
    )



# Schema field sets of the canonical A800 GEMM launch artifact.  The
# artifact's loader owns them; both the launch package and the dataset
# projection validate the same records, so the sets are declared once.
CANONICAL_OPERATION_FIELDS = {
    "operation", "queried_device", "algorithms", "selection", "projection",
    "main_launch", "reducer_launch", "device_execution",
}
OPERATION_FIELDS = {
    "m", "n", "k", "dtype", "layout", "trans_a", "trans_b", "has_bias",
}
ALGORITHM_FIELDS = {
    "rank", "algorithm_id", "tile_id", "tile", "logical_tile_m",
    "logical_tile_n", "stages_id", "stages", "inner_shape_id", "split_k",
    "reduction_scheme", "cta_swizzle", "custom_option", "workspace_bytes",
    "waves_count",
}
ALGORITHM_INTEGER_FIELDS = (
    "rank", "algorithm_id", "tile_id", "logical_tile_m", "logical_tile_n",
    "stages_id", "inner_shape_id", "split_k", "reduction_scheme",
    "cta_swizzle", "custom_option", "workspace_bytes",
)
SELECTION_FIELDS = {"contract", "selected_rank", "reason"}
DEVICE_EXECUTION_FIELDS = {
    "evidence_kind", "evidence_version", "activity_record_count",
    "executed_symbols", "all_replacements_observed", "dropped_record_count",
}
COVERAGE_SCOPE_FIELDS = {
    "kind", "request_trajectory_consumed", "scheduler_alignment_consumed",
    "operator_trace_consumed", "timing_consumed", "fused_moe_included",
    "serving_end_to_end_executability_claimed",
}
MODEL_DOMAIN_FIELDS = {
    "model_id", "model_path", "model_config_sha256", "model_type",
    "tensor_parallel_size", "expert_parallel_size", "positions",
    "semantic_key_count", "physical_operation_count",
}
POSITION_DOMAIN_FIELDS = {
    "position", "m_min", "m_max", "semantic_key_count", "n", "k",
    "dtype", "has_bias", "variant_contract",
}
HARDWARE_DOMAIN_FIELDS = {
    "status", "domain_sha256", "semantic_key_count",
    "physical_operation_count",
}
CROSS_HARDWARE_FIELDS = {
    "status", "hardware_ids", "domain_sha256", "semantic_key_count",
    "physical_operation_count",
}
LIMITATION_FIELDS = {
    "scope", "affected_hardware_ids", "affected_model_ids", "reason",
}

# The projection manifest field set (distinct from the loader's
# ``_MANIFEST_FIELDS`` above) is identical for every hardware's canonical GEMM
# launch artifact, so it is declared once; the remaining sets below are the
# MetaX C500 artifact's own record shapes.
ARTIFACT_MANIFEST_FIELDS = {
    "schema_version", "artifact_type", "source_config",
    "source_config_sha256", "scheduler_contract", "coverage_scope",
    "duration_consumed",
    "physical_operation_fields", "workload_key_fields", "limitations",
    "dataset_model_inventory", "model_domains", "workload_keys",
    "semantic_key_count", "physical_operation_count", "hardware_domains",
    "cross_hardware",
}
METAX_CANONICAL_FIELDS = {
    "schema_version", "artifact_type", "hardware_id", "software_revision",
    "maca_version", "mcblas_version", "source", "generated_by",
    "candidate_sha256", "capture_baseline_commit", "source_sha256",
    "workload_manifest_sha256", "workload_manifest", "workload_key_fields",
    "selector_contract", "duration_consumed", "gemm_executed",
    "semantic_key_count", "physical_operation_count", "workloads",
    "operations",
}
METAX_SOURCE_PATHS = (
    "microbench/metax_c500/operators/gemm/dry_launch_runner.py",
    "microbench/metax_c500/operators/gemm/dry_launch_probe.cpp",
    "microbench/metax_c500/operators/gemm/dry_launch_interposer.cpp",
    "dataset_pipeline/launch_data/gemm_workload_candidates.py",
    "src/kernel/launch_config/gemm_workload.py",
    "src/kernel/launch_config/selection_primitives.py",
)
METAX_RESULT_FIELDS = {
    "gemm_executed", "duration_consumed", "selector_attempt_count",
    "initial_expected_launch_count", "validated_launch_count",
    "stability_capture_count", "launches",
}
METAX_LAUNCH_FIELDS = {
    "record_type", "ordinal", "api", "identity_source", "kernel_identity",
    "grid", "block", "dynamic_shared_memory_bytes",
    "static_shared_memory_bytes", "static_shared_memory_status",
}
METAX_SCHEDULER_FIELDS = {
    "backend", "scheduler_version", "scheduler_source_commit",
    "max_batched_tokens", "max_requests", "enable_chunked_prefill",
    "mtp_num_speculative_tokens",
}
PROHIBITED_TIMING_FIELD_PARTS = (
    "duration", "timing", "cycle", "latency", "elapsed",
)
