"""Structured tiling selected by a hardware launch implementation."""

from dataclasses import dataclass, field
import math
import re
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple


METAX_MLA_SCHEDULER_PARTITIONS = 104
METAX_MLA_KV_BLOCK = 16
METAX_MLA_SCHEDULER_FIXED_BLOCKS = 5


@dataclass(frozen=True)
class MetaxMlaSegment:
    """One request fragment emitted by the fixed FlashMLA scheduler."""

    request_index: int
    first_kv_block: int
    kv_blocks: int
    split_index: int


@dataclass(frozen=True)
class MetaxMlaSchedulerWork:
    """Exact output of the public 104-partition metadata scheduler."""

    payload_blocks: int
    kv_blocks_per_request: Tuple[int, ...]
    segments_by_partition: Tuple[Tuple[MetaxMlaSegment, ...], ...]
    splits_per_request: Tuple[int, ...]

    @property
    def kv_block_count(self) -> int:
        return sum(self.kv_blocks_per_request)

    @property
    def segment_count(self) -> int:
        return sum(len(segments) for segments in self.segments_by_partition)

    @property
    def same_worker_transitions(self) -> int:
        return sum(
            max(0, len(segments) - 1)
            for segments in self.segments_by_partition)

    @property
    def unsplit_requests(self) -> int:
        return sum(count == 1 for count in self.splits_per_request)


def derive_mla_scheduler_work(
        kv_lengths: Sequence[int]) -> MetaxMlaSchedulerWork:
    """Execute the source-defined FlashMLA metadata scheduling loop."""

    if (not isinstance(kv_lengths, (list, tuple))
            or not kv_lengths
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in kv_lengths)):
        raise ValueError("FlashMLA kv_lengths must be positive integers")
    kv_blocks_per_request = tuple(
        math.ceil(length / METAX_MLA_KV_BLOCK) for length in kv_lengths)
    total_weighted_blocks = sum(
        blocks + METAX_MLA_SCHEDULER_FIXED_BLOCKS
        for blocks in kv_blocks_per_request)
    payload_blocks = (
        math.ceil(total_weighted_blocks / METAX_MLA_SCHEDULER_PARTITIONS)
        + METAX_MLA_SCHEDULER_FIXED_BLOCKS)

    request_index = 0
    first_kv_block = 0
    split_index = 0
    split_counts = [0] * len(kv_blocks_per_request)
    partitions = []
    for _ in range(METAX_MLA_SCHEDULER_PARTITIONS):
        remaining_payload = payload_blocks
        segments = []
        while request_index < len(kv_blocks_per_request):
            request_blocks = kv_blocks_per_request[request_index]
            remaining_blocks = request_blocks - first_kv_block
            if remaining_payload >= (
                    remaining_blocks + METAX_MLA_SCHEDULER_FIXED_BLOCKS):
                segments.append(MetaxMlaSegment(
                    request_index=request_index,
                    first_kv_block=first_kv_block,
                    kv_blocks=remaining_blocks,
                    split_index=split_index,
                ))
                split_counts[request_index] += 1
                remaining_payload -= (
                    remaining_blocks + METAX_MLA_SCHEDULER_FIXED_BLOCKS)
                request_index += 1
                first_kv_block = 0
                split_index = 0
            else:
                assigned_blocks = (
                    remaining_payload - METAX_MLA_SCHEDULER_FIXED_BLOCKS)
                if assigned_blocks > 0:
                    segments.append(MetaxMlaSegment(
                        request_index=request_index,
                        first_kv_block=first_kv_block,
                        kv_blocks=assigned_blocks,
                        split_index=split_index,
                    ))
                    split_counts[request_index] += 1
                    first_kv_block += assigned_blocks
                    split_index += 1
                break
        partitions.append(tuple(segments))
    if request_index != len(kv_blocks_per_request):
        raise ValueError("FlashMLA scheduler did not consume every request")
    if first_kv_block != 0 or split_index != 0:
        raise ValueError("FlashMLA scheduler ended inside a request")
    work = MetaxMlaSchedulerWork(
        payload_blocks=payload_blocks,
        kv_blocks_per_request=kv_blocks_per_request,
        segments_by_partition=tuple(partitions),
        splits_per_request=tuple(split_counts),
    )
    if work.kv_block_count != sum(
            segment.kv_blocks
            for segments in work.segments_by_partition
            for segment in segments):
        raise AssertionError("FlashMLA scheduler lost KV blocks")
    return work


def metax_mla_combine_block_m(
        batch: int, query_length: int, query_heads: int) -> int:
    """Return FlashMLA's source-selected output rows per combine CTA."""

    for name, value in (
            ("batch", batch), ("query_length", query_length),
            ("query_heads", query_heads)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"FlashMLA {name} must be an integer")
        if value <= 0:
            raise ValueError(f"FlashMLA {name} must be positive")
    rows_per_request = query_length * query_heads
    total_rows = batch * rows_per_request
    if rows_per_request == 2:
        return 2
    if rows_per_request < 16 and rows_per_request % 4 == 0:
        return 4
    if total_rows >= 1024 and rows_per_request % 16 == 0:
        return 16
    if rows_per_request % 4 == 0:
        return 4
    return 1


def find_gemm_swizzle_convention(
        m_tiles: int, n_tiles: int, grid_x: int, grid_y: int):
    """Return the represented M/N swizzle, or ``None`` for a worker grid.

    A ``None`` result is not itself a fallback.  Launch packages must stamp an
    explicit logical-order fact when a physical grid represents padded or
    persistent workers rather than output tiles; consumers otherwise reject
    the launch.
    """

    values = {
        "m_tiles": m_tiles,
        "n_tiles": n_tiles,
        "grid_x": grid_x,
        "grid_y": grid_y,
    }
    invalid_types = {
        name: value for name, value in values.items()
        if isinstance(value, bool) or not isinstance(value, int)
    }
    if invalid_types:
        raise TypeError(
            "GEMM swizzle fields must be integers: "
            f"{invalid_types}")
    non_positive = {
        name: value for name, value in values.items() if value <= 0
    }
    if non_positive:
        raise ValueError(
            "GEMM swizzle fields must be positive: "
            f"{non_positive}")
    for width in range(1, 33):
        if m_tiles * width == grid_x and math.ceil(n_tiles / width) == grid_y:
            return "M", width
    for width in range(1, 33):
        if n_tiles * width == grid_x and math.ceil(m_tiles / width) == grid_y:
            return "N", width
    return None


def gemm_logical_workgroups(
        batch: int, m: int, n: int, tile_m: int, tile_n: int,
        split_k: int) -> int:
    """Return logical GEMM tile jobs independently of a worker-grid swizzle."""

    values = {
        "batch": batch,
        "m": m,
        "n": n,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "split_k": split_k,
    }
    invalid_types = {
        name: value for name, value in values.items()
        if isinstance(value, bool) or not isinstance(value, int)
    }
    if invalid_types:
        raise TypeError(
            "GEMM logical-workgroup fields must be integers: "
            f"{invalid_types}")
    non_positive = {
        name: value for name, value in values.items() if value <= 0
    }
    if non_positive:
        raise ValueError(
            "GEMM logical-workgroup fields must be positive: "
            f"{non_positive}")
    rows = batch * m
    return (
        ((rows + tile_m - 1) // tile_m)
        * ((n + tile_n - 1) // tile_n)
        * split_k
    )


def flash_attention_packed_varlen_scheduled_workgroups(
        batch: int, q_len: int, tile_q: int, query_heads: int,
        query_heads_per_workgroup: int, splits: int) -> int:
    """Return the padded logical jobs scheduled by a packed FA queue."""

    values = {
        "batch": batch,
        "q_len": q_len,
        "tile_q": tile_q,
        "query_heads": query_heads,
        "query_heads_per_workgroup": query_heads_per_workgroup,
        "splits": splits,
    }
    invalid_types = {
        name: value for name, value in values.items()
        if isinstance(value, bool) or not isinstance(value, int)
    }
    if invalid_types:
        raise TypeError(
            "packed-varlen FLASH_ATTN schedule fields must be integers: "
            f"{invalid_types}")
    non_positive = {
        name: value for name, value in values.items() if value <= 0
    }
    if non_positive:
        raise ValueError(
            "packed-varlen FLASH_ATTN schedule fields must be positive: "
            f"{non_positive}")
    return (
        batch
        * ((q_len + tile_q - 1) // tile_q)
        * ((query_heads + query_heads_per_workgroup - 1)
           // query_heads_per_workgroup)
        * splits
    )


# cuBLASLt 12.9 Hopper ``nvjet`` kernel symbols (H200 WP0 launch capture and
# kernel campaign): ``nvjet_<types>_<TM>x<TN>_<K>x<STAGES>_<CX>x<CY>_<raster>
# _bz[_coopA|_coopB][_splitK]_<layout>``.  ``TM``/``TN`` are the cuBLAS
# column-major tile of ``C^T = W * X^T`` for a PyTorch linear, so ``TM`` runs
# along the output features (Midlayer ``n``) and ``TN`` along the tokens
# (Midlayer ``m``); WP0 cross-checked ``TILE_ID``, ``STAGES_ID`` K, the
# cluster shape and the launch cluster attribute for every captured launch.
_NVJET_GEMM_SYMBOL = re.compile(
    r"nvjet_(?P<types>[a-z]{3})_(?P<tile_cublas_m>\d+)x(?P<tile_cublas_n>\d+)"
    r"_(?P<tile_k>\d+)x(?P<stages>\d+)_(?P<cluster_m>\d+)x(?P<cluster_n>\d+)"
    r"_(?P<raster>[vh])_bz(?P<cooperative>_coop[AB])?(?P<split_k>_splitK)?"
    r"_(?P<layout>[TN]{3})")


@dataclass(frozen=True)
class NvjetGemmSymbol:
    """Discrete launch facts spelled by one cuBLASLt ``nvjet`` symbol."""

    types: str
    tile_m: int
    tile_n: int
    tile_k: int
    pipeline_depth: int
    cluster_n_tiles: int
    cluster_m_tiles: int
    raster: str
    cooperative: Optional[str]
    split_k: bool
    layout: str


def parse_nvjet_gemm_symbol(runtime_name: str) -> Optional[NvjetGemmSymbol]:
    """Return the ``nvjet`` launch facts, ``None`` for other families.

    A name that starts as an ``nvjet`` symbol but does not match the
    complete grammar fails instead of falling through to the generic tile
    heuristics, which would misread its ``<K>x<STAGES>`` field as a tile.
    """

    if not isinstance(runtime_name, str):
        raise TypeError("GEMM runtime name must be a string")
    if not runtime_name.startswith("nvjet_"):
        return None
    match = _NVJET_GEMM_SYMBOL.fullmatch(runtime_name)
    if match is None:
        raise ValueError(
            f"unrecognized cuBLASLt nvjet GEMM symbol: {runtime_name!r}")
    cooperative = match.group("cooperative")
    return NvjetGemmSymbol(
        types=match.group("types"),
        tile_m=int(match.group("tile_cublas_n")),
        tile_n=int(match.group("tile_cublas_m")),
        tile_k=int(match.group("tile_k")),
        pipeline_depth=int(match.group("stages")),
        cluster_n_tiles=int(match.group("cluster_m")),
        cluster_m_tiles=int(match.group("cluster_n")),
        raster=match.group("raster"),
        cooperative=None if cooperative is None else cooperative[1:],
        split_k=match.group("split_k") is not None,
        layout=match.group("layout"),
    )


def parse_gemm_tile(runtime_name: str):
    """Extract ``(tile_m, tile_n, tile_k)`` from a runtime implementation."""

    if not isinstance(runtime_name, str):
        raise TypeError("GEMM runtime name must be a string")
    if not runtime_name:
        raise ValueError("GEMM runtime name must not be empty")
    nvjet = parse_nvjet_gemm_symbol(runtime_name)
    if nvjet is not None:
        return nvjet.tile_m, nvjet.tile_n, nvjet.tile_k
    for match in re.finditer(r"(\d+)x(\d+)x(\d+)", runtime_name):
        first, second, third = (int(match.group(i)) for i in range(1, 4))
        if 16 <= first <= 512 and 16 <= second <= 512 and 8 <= third <= 256:
            return second, first, third
    for match in re.finditer(r"(\d+)x(\d+)", runtime_name):
        first, second = int(match.group(1)), int(match.group(2))
        if 16 <= first <= 512 and 16 <= second <= 512:
            stage = re.search(r"stages_(\d+)x\d+", runtime_name)
            if stage:
                return second, first, int(stage.group(1))
            cutlass = re.search(r"_(\d+)x\d+_tn", runtime_name)
            if cutlass:
                return second, first, int(cutlass.group(1))
            return second, first, None
    config = re.search(r"GemmConfig<(\d+),\s*(\d+),\s*(\d+)>", runtime_name)
    if config:
        values = tuple(int(config.group(i)) for i in range(1, 4))
        if all(8 <= value <= 512 for value in values):
            return values
    mangled_config = re.search(
        r"GemmConfigILi(\d+)ELi(\d+)ELi(\d+)EE", runtime_name)
    if mangled_config:
        values = tuple(int(mangled_config.group(i)) for i in range(1, 4))
        if all(8 <= value <= 512 for value in values):
            return values
    return None, None, None


def parse_pipeline_stages(runtime_name: str):
    """Extract ``(tile_k, pipeline_depth)`` from an observed stage suffix."""

    if not isinstance(runtime_name, str):
        raise TypeError("runtime name must be a string")
    if not runtime_name:
        raise ValueError("runtime name must not be empty")
    match = re.search(r"stages_(\d+)x(\d+)", runtime_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    # CUTLASS 80 encodes the retained six-stage programs as ``_<K>x6_tn``
    # rather than spelling out ``stages``.  ``64x6`` is the pre-existing
    # source-bound class.  The ``32x6`` contract is narrower: the recorded
    # FP16 128x256 CUTLASS program has six 24-KiB operand panels, which fit
    # in its SM80 shared-memory allocation and expose a finite producer ring.
    # Keep every other opaque suffix fail-closed until it has equivalent
    # static evidence.
    match = re.search(r"_(64)x(6)_tn(?:_|$)", runtime_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(
        r"cutlass_80_tensorop_f16_s16816gemm_relu_f16_"
        r"128x256_(32)x(6)_tn(?:_|$)", runtime_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def parse_complete_gemm_tile(runtime_name: str):
    """Return ``(tile_m, tile_n, tile_k, depth)`` or ``None``."""

    nvjet = parse_nvjet_gemm_symbol(runtime_name)
    if nvjet is not None:
        return (nvjet.tile_m, nvjet.tile_n, nvjet.tile_k,
                nvjet.pipeline_depth)
    tile_m, tile_n, tile_k = parse_gemm_tile(runtime_name)
    if tile_m is None or tile_n is None:
        return None
    stage_k, depth = parse_pipeline_stages(runtime_name)
    if (depth is None
            and (tile_m, tile_n, tile_k) == (128, 256, 64)
            and re.search(r"_64x3_tn(?:_|$)", runtime_name)):
        depth = 3
    return tile_m, tile_n, tile_k if tile_k is not None else stage_k, depth


def parse_flash_attention_trait(runtime_name: str):
    """Extract ``(head_dim, query_tile, kv_tile, warps)`` from FA traits.

    This parser only recovers the four positional template integers.  Whether
    a particular software revision gives those positions the corresponding
    semantic meanings remains the responsibility of the version-bound launch
    or observation package that consumes the result.
    """

    if not isinstance(runtime_name, str):
        raise TypeError("FLASH_ATTN runtime name must be a string")
    if not runtime_name:
        raise ValueError("FLASH_ATTN runtime name must not be empty")
    patterns = (
        (r"kernel_traits<\(int\)(\d+),\s*\(int\)(\d+),\s*"
         r"\(int\)(\d+),\s*\(int\)(\d+)"),
        r"Flash_fwd_kernel_traitsILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E",
        r"kernel_traits<(\d+),\s*(\d+),\s*(\d+),\s*(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, runtime_name)
        if match:
            return tuple(int(match.group(index)) for index in range(1, 5))
    return None


def parse_flash_attention_tile(runtime_name: str):
    """Extract ``(query_tile, kv_tile)`` from known FA runtime templates."""

    trait = parse_flash_attention_trait(runtime_name)
    if trait is not None:
        return trait[1], trait[2]
    # Some retained demangled CUDA names stop after the third positional
    # integer.  Their tile remains parseable even though the warp count is not.
    match = re.search(
        r"kernel_traits<\(int\)(\d+),\s*\(int\)(\d+),\s*\(int\)(\d+)",
        runtime_name,
    )
    if match is not None:
        return int(match.group(2)), int(match.group(3))
    return None


def _positive_int_map(values: Mapping[str, int]) -> Mapping[str, int]:
    if not isinstance(values, Mapping):
        raise TypeError("tile_shape must be a mapping")
    invalid_keys = [
        name for name in values if not isinstance(name, str) or not name
    ]
    if invalid_keys:
        raise TypeError(
            f"tile dimension keys must be non-empty strings: {invalid_keys}")
    invalid_types = {
        name: value for name, value in values.items()
        if isinstance(value, bool) or not isinstance(value, int)
    }
    if invalid_types:
        raise TypeError(f"tile dimensions must be integers: {invalid_types}")
    normalized = dict(values)
    invalid = {name: value for name, value in normalized.items() if value <= 0}
    if invalid:
        raise ValueError(f"tile dimensions must be positive: {invalid}")
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class TilingSpec:
    """Optional compute-kernel tile, split, and pipeline configuration.

    ``tile_shape`` uses semantic axis names (for example ``m``, ``n``, ``k``)
    instead of a positional tuple so different kernel families cannot silently
    reinterpret the same value.
    """

    tile_shape: Mapping[str, int] = field(default_factory=dict)
    pipeline_depth: Optional[int] = None
    split_k: Optional[int] = None
    num_kv_splits: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tile_shape", _positive_int_map(self.tile_shape))
        for name in ("pipeline_depth", "split_k", "num_kv_splits"):
            value = getattr(self, name)
            if (value is not None
                    and (isinstance(value, bool) or not isinstance(value, int))):
                raise TypeError(f"{name} must be an integer or null")
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
