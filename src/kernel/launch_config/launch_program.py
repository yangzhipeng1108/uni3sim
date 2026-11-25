"""Typed source programs retained by production launch selection."""

from dataclasses import dataclass
from typing import Optional, Tuple


class LaunchProgram:
    """Marker base for immutable, source-derived launch programs."""


@dataclass(frozen=True)
class CannGemmLaunchProgram(LaunchProgram):
    """Normalized CANN GEMM worker and tile traversal program.

    MatMulV2 and MatMulV3 have different tiling payload ABIs, but both lower
    to this execution contract before they reach the predictor.  Every extent
    is expressed in matrix elements; the predictor therefore has no reason to
    inspect a kernel symbol or tiling key.
    """

    program_kind: str
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
    split_k: int

    def __post_init__(self) -> None:
        if self.program_kind != "cann_gemm_launch_v1":
            raise ValueError("CannGemmLaunchProgram has an unknown kind")
        for name in (
                "aic_workers", "worker_grid_m", "worker_grid_n",
                "worker_grid_k", "single_core_m", "single_core_n",
                "single_core_k", "base_m", "base_n", "base_k",
                "depth_a1", "depth_b1", "a_l1_k_tiles", "b_l1_k_tiles",
                "step_m", "step_n",
                "m_l2_regions", "n_l2_regions", "m_l2_block",
                "n_l2_block", "split_k"):
            _require_non_negative_int(
                f"CannGemmLaunchProgram.{name}", getattr(self, name),
                positive=True)
        for name in ("a_uses_l2", "b_uses_l2"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(
                    f"CannGemmLaunchProgram.{name} must be boolean")
        if self.traversal_order not in (
                "persistent_n_then_m", "m_then_n_snake",
                "n_then_m_snake"):
            raise ValueError("CANN GEMM traversal order is unsupported")
        if (self.worker_grid_m * self.worker_grid_n * self.worker_grid_k
                != self.aic_workers):
            raise ValueError("CANN GEMM worker grid does not match AIC count")
        if self.worker_grid_k != self.split_k:
            raise ValueError("CANN GEMM worker K grid differs from split_k")
    def operand_uses_l2(self, operand: str) -> bool:
        """Return the source-selected L2 policy for one matrix operand."""

        cache_fields = {"a": self.a_uses_l2, "b": self.b_uses_l2}
        if operand not in cache_fields:
            raise ValueError("CANN GEMM L2 operand is unsupported")
        return cache_fields[operand]


def _require_non_negative_int(owner: str, value: int, *, positive=False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{owner} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{owner} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{owner} cannot be negative")
    return value


@dataclass(frozen=True)
class CannGemmTilePipelineProgram(LaunchProgram):
    """Compact CANN GEMM tile program shared by AIC and AIV domains.

    The program retains source-selected scheduling geometry, not a kernel
    identity or an observed duration.  A predictor can therefore execute the
    same analytical tile-pipeline semantics for ordinary and split-K GEMMs
    while accounting for the latter's FP32 workspace handoff.
    """

    program_kind: str
    aic_workers: int
    aiv_workers: int
    single_core_m: int
    single_core_n: int
    single_core_k: int
    base_m: int
    base_n: int
    base_k: int
    depth_a1: int
    depth_b1: int
    step_m: int
    step_n: int
    workspace_buffer_count: int
    partial_element_bytes: int
    scheduler_policy: str
    traversal_order: str
    a1_buffer_slots: int
    b1_buffer_slots: int
    l0a_buffer_slots: int
    l0b_buffer_slots: int

    def __post_init__(self) -> None:
        if self.program_kind not in {
                "cann_gemm_deterministic_split_k_v1",
                "cann_gemm_single_core_split_k_v1",
        }:
            raise ValueError(
                "CannGemmTilePipelineProgram has an unknown program kind")
        for name in (
                "aic_workers", "aiv_workers", "single_core_m",
                "single_core_n", "single_core_k", "base_m", "base_n",
                "base_k", "depth_a1", "depth_b1", "step_m", "step_n",
                "workspace_buffer_count", "partial_element_bytes"):
            _require_non_negative_int(
                f"CannGemmTilePipelineProgram.{name}", getattr(self, name),
                positive=True)
        if self.aiv_workers != 2 * self.aic_workers:
            raise ValueError(
                "CANN deterministic split-K requires the source 1:2 "
                "AIC/AIV mapping")
        if self.single_core_m % self.base_m:
            raise ValueError("CANN GEMM single_core_m must be tiled by base_m")
        if self.single_core_n % self.base_n:
            raise ValueError("CANN GEMM single_core_n must be tiled by base_n")
        if self.single_core_k % self.base_k:
            raise ValueError("CANN GEMM single_core_k must be tiled by base_k")
        if self.program_kind == "cann_gemm_deterministic_split_k_v1":
            expected_workspace_buffers = 2
        elif self.program_kind == "cann_gemm_single_core_split_k_v1":
            expected_workspace_buffers = 1
        else:
            raise AssertionError("validated CANN GEMM program kind disappeared")
        if self.workspace_buffer_count != expected_workspace_buffers:
            raise ValueError(
                "CANN GEMM source program has an invalid workspace buffer "
                "count")
        if self.partial_element_bytes != 4:
            raise ValueError(
                "CANN deterministic split-K partials must be FP32")
        if self.scheduler_policy not in {
                "l2_split_matmul", "n_buffer_33",
                "single_core_split_k", "source_policy_unmodeled"}:
            raise ValueError("CANN GEMM scheduler policy is unsupported")
        if self.traversal_order not in {"n_then_m", "unmodeled"}:
            raise ValueError("CANN GEMM tile traversal order is unsupported")
        for name in (
                "a1_buffer_slots", "b1_buffer_slots", "l0a_buffer_slots",
                "l0b_buffer_slots"):
            _require_non_negative_int(
                f"CannGemmTilePipelineProgram.{name}", getattr(self, name))
        if self.scheduler_policy == "n_buffer_33":
            if self.program_kind != "cann_gemm_deterministic_split_k_v1":
                raise ValueError(
                    "NBuffer33 requires CANN deterministic split-K")
            if self.traversal_order != "n_then_m":
                raise ValueError("NBuffer33 traversal must be N then M")
            if (self.a1_buffer_slots, self.b1_buffer_slots,
                    self.l0a_buffer_slots, self.l0b_buffer_slots) != (
                        3, 2, 2, 2):
                raise ValueError(
                    "NBuffer33 requires source-defined 3A1/2B1/2L0 slots")
        elif self.scheduler_policy == "l2_split_matmul":
            if self.program_kind != "cann_gemm_deterministic_split_k_v1":
                raise ValueError(
                    "L2-split Matmul requires CANN deterministic split-K")
            if self.traversal_order != "n_then_m":
                raise ValueError(
                    "L2-split Matmul traversal must be N then M")
            if (self.a1_buffer_slots, self.b1_buffer_slots,
                    self.l0a_buffer_slots, self.l0b_buffer_slots) != (
                        1, 2, 2, 2):
                raise ValueError(
                    "L2-split Matmul requires source-defined "
                    "1A1/2B1/2L0 slots")
        elif self.scheduler_policy == "single_core_split_k":
            if self.program_kind != "cann_gemm_single_core_split_k_v1":
                raise ValueError(
                    "single-core split-K policy requires its CANN program")
            if self.traversal_order != "n_then_m":
                raise ValueError(
                    "single-core split-K traversal must be N then M")
            expected_a1_slots = 3 if self.step_m == 3 else 2
            if (self.a1_buffer_slots, self.b1_buffer_slots,
                    self.l0a_buffer_slots, self.l0b_buffer_slots) != (
                        expected_a1_slots, 2, 2, 2):
                raise ValueError(
                    "single-core split-K requires its source-defined "
                    "A1/B1/L0 slots")
        elif (self.traversal_order != "unmodeled"
              or any((self.a1_buffer_slots, self.b1_buffer_slots,
                      self.l0a_buffer_slots, self.l0b_buffer_slots))):
            raise ValueError(
                "unmodeled CANN scheduler cannot claim buffer ownership")


@dataclass(frozen=True)
class FiaCrossDomainTask(LaunchProgram):
    """One source-ordered FIA MM1/Vec1/MM2/Vec2 main task."""

    b_n2: int
    g_s1: int
    s2: int
    actual_m: int
    actual_s2: int
    is_head_first: bool
    is_head_last: bool
    is_s2_split_core: bool

    def __post_init__(self) -> None:
        for name in ("b_n2", "g_s1", "s2"):
            _require_non_negative_int(
                f"FiaCrossDomainTask.{name}", getattr(self, name))
        for name in ("actual_m", "actual_s2"):
            _require_non_negative_int(
                f"FiaCrossDomainTask.{name}", getattr(self, name),
                positive=True)
        for name in (
                "is_head_first", "is_head_last", "is_s2_split_core"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"FiaCrossDomainTask.{name} must be boolean")


@dataclass(frozen=True)
class FiaCrossDomainAicGroupProgram(LaunchProgram):
    """Ordered FIA tasks owned by one AIC and its two physical AIVs."""

    aic_group: int
    aiv_workers: Tuple[int, int]
    tasks: Tuple[FiaCrossDomainTask, ...]

    def __post_init__(self) -> None:
        _require_non_negative_int(
            "FiaCrossDomainAicGroupProgram.aic_group", self.aic_group)
        if (not isinstance(self.aiv_workers, tuple)
                or len(self.aiv_workers) != 2):
            raise TypeError(
                "FiaCrossDomainAicGroupProgram.aiv_workers must be a pair")
        if any(
                isinstance(value, bool) or not isinstance(value, int)
                or value < 0 for value in self.aiv_workers):
            raise ValueError(
                "FiaCrossDomainAicGroupProgram AIV workers must be "
                "non-negative integers")
        if self.aiv_workers != (
                2 * self.aic_group, 2 * self.aic_group + 1):
            raise ValueError(
                "FiaCrossDomainAicGroupProgram must use the source 1:2 "
                "AIC/AIV mapping")
        if not isinstance(self.tasks, tuple) or not self.tasks:
            raise TypeError(
                "FiaCrossDomainAicGroupProgram.tasks must be a non-empty "
                "tuple")
        if any(not isinstance(value, FiaCrossDomainTask)
               for value in self.tasks):
            raise TypeError(
                "FiaCrossDomainAicGroupProgram.tasks must contain "
                "FiaCrossDomainTask")


@dataclass(frozen=True)
class FiaCrossDomainLaunchProgram(LaunchProgram):
    """Complete CANN FIA 1:2 three-handoff source program."""

    program_kind: str
    aic_groups: Tuple[FiaCrossDomainAicGroupProgram, ...]

    def __post_init__(self) -> None:
        if self.program_kind != "cann_fia_1_2_three_handoff_v1":
            raise ValueError(
                "FiaCrossDomainLaunchProgram has an unknown program kind")
        if not isinstance(self.aic_groups, tuple) or not self.aic_groups:
            raise TypeError(
                "FiaCrossDomainLaunchProgram.aic_groups must be a non-empty "
                "tuple")
        if any(not isinstance(value, FiaCrossDomainAicGroupProgram)
               for value in self.aic_groups):
            raise TypeError(
                "FiaCrossDomainLaunchProgram.aic_groups must contain "
                "FiaCrossDomainAicGroupProgram")
        actual_groups = tuple(group.aic_group for group in self.aic_groups)
        expected_groups = tuple(range(len(self.aic_groups)))
        if actual_groups != expected_groups:
            raise ValueError(
                "FiaCrossDomainLaunchProgram AIC groups must be consecutive")


@dataclass(frozen=True)
class GpuCtaRasterProgram(LaunchProgram):
    """Vendor CTA-to-tile raster map retained by a GPU GEMM launch.

    A GPU GEMM launch names one output tile and K slice per CTA, but the CTA
    index is not that tile: each library applies its own reversible map first.
    ``convention`` names the map, ``log_group`` the log2 width of its grouped
    axis, and ``serpentine`` whether alternate groups walk the ungrouped axis
    backwards.  Every field is a discrete launch fact; nothing here depends on
    a measured duration or traffic.

    Three maps reach this contract.  ``cublaslt_m_group`` keeps one CTA per
    output tile and bands ``blockIdx.y``; the two ``cutlass_*_group`` maps
    instead widen one grid axis by the group and fold the other, so the grid
    shape states the group itself and a launch that exposes such a grid
    decodes its own map.  ``none`` is the direct grid raster.

    ``cluster_m_group`` and ``cluster_n_group`` enumerate persistent output
    cluster jobs. Their observed ``group_size`` is an integer, including
    non-power-of-two widths, and the final band keeps its complete remainder.
    The predictor supplies cluster tile counts rather than CTA tile counts;
    the physical worker grid remains on LaunchConfig.
    """

    convention: str
    log_group: int
    serpentine: bool
    k_slices: int
    group_size: Optional[int] = None

    def __post_init__(self) -> None:
        if self.convention not in (
                "none", "cublaslt_m_group", "cutlass_n_group",
                "cutlass_m_group", "cluster_m_group", "cluster_n_group"):
            raise ValueError("GPU CTA raster convention is unsupported")
        _require_non_negative_int(
            "GpuCtaRasterProgram.k_slices", self.k_slices, positive=True)
        _require_non_negative_int(
            "GpuCtaRasterProgram.log_group", self.log_group)
        if self.log_group > 15:
            raise ValueError("GPU CTA raster group exceeds the launch domain")
        if not isinstance(self.serpentine, bool):
            raise TypeError("GpuCtaRasterProgram.serpentine must be boolean")
        if self.is_cluster_grouped:
            if self.group_size is None:
                raise ValueError("cluster raster requires observed group_size")
            _require_non_negative_int(
                "GpuCtaRasterProgram.group_size", self.group_size, positive=True)
            if self.log_group != 0:
                raise ValueError("cluster raster uses group_size, not log_group")
        elif self.group_size is not None:
            raise ValueError("CTA raster does not accept cluster group_size")
        if self.convention == "none" and (
                self.log_group or self.serpentine):
            raise ValueError(
                "a direct CTA raster declares no group or serpentine")
        if self.convention.startswith("cutlass_") and self.serpentine:
            raise ValueError(
                "a grouped grid CTA raster has no serpentine reversal")

    @property
    def is_cluster_grouped(self) -> bool:
        return self.convention in ("cluster_m_group", "cluster_n_group")

    def grid(self, m_tiles: int, n_tiles: int) -> Tuple[int, int, int]:
        """Return the logical raster grid; persistent workers are separate."""

        for name, value in (
                ("M tiles", m_tiles), ("N tiles", n_tiles)):
            _require_non_negative_int(
                f"GPU CTA raster {name}", value, positive=True)
        group = 1 << self.log_group
        if self.convention == "cutlass_n_group":
            return (m_tiles * group, -(-n_tiles // group), self.k_slices)
        if self.convention == "cutlass_m_group":
            return (n_tiles * group, -(-m_tiles // group), self.k_slices)
        return (n_tiles, m_tiles, self.k_slices)

    def schedule(self, m_tiles: int,
                 n_tiles: int) -> Tuple[Tuple[int, int, int], ...]:
        """Return the (M tile, N tile, K slice) order of CTA dispatch.

        CUDA enumerates ``blockIdx.x`` before ``blockIdx.y`` before
        ``blockIdx.z``, so walking the grid in that order and applying the map
        yields the dispatch order itself.
        """

        grid_x, grid_y, grid_z = self.grid(m_tiles, n_tiles)
        group = 1 << self.log_group
        schedule = []
        if self.is_cluster_grouped:
            group = self.group_size
            along_m = self.convention == "cluster_m_group"
            extent = m_tiles if along_m else n_tiles
            other_extent = n_tiles if along_m else m_tiles
            for slice_index in range(grid_z):
                for band, start in enumerate(range(0, extent, group)):
                    width = min(group, extent - start)
                    other_order = (
                        range(other_extent - 1, -1, -1)
                        if self.serpentine and band % 2 else range(other_extent))
                    for other in other_order:
                        for local in range(width):
                            m_index, n_index = (
                                (start + local, other) if along_m
                                else (other, start + local))
                            schedule.append((m_index, n_index, slice_index))
            return tuple(schedule)
        for slice_index in range(grid_z):
            for grid_row in range(grid_y):
                for grid_column in range(grid_x):
                    tile = self._tile(
                        grid_column, grid_row, grid_x, grid_y, group)
                    if tile is None:
                        continue
                    m_index, n_index = tile
                    if m_index < m_tiles and n_index < n_tiles:
                        schedule.append((m_index, n_index, slice_index))
        expected = m_tiles * n_tiles * self.k_slices
        if len(schedule) != expected:
            raise ValueError(
                "GPU CTA raster does not cover its output tiles exactly: "
                f"{len(schedule)} of {expected}")
        return tuple(schedule)

    def _tile(self, grid_column: int, grid_row: int, grid_x: int,
              grid_y: int, group: int):
        if self.convention == "none":
            return grid_row, grid_column
        if self.convention == "cutlass_n_group":
            return grid_column >> self.log_group, (
                (grid_row << self.log_group) + (grid_column & (group - 1)))
        if self.convention == "cutlass_m_group":
            return (
                (grid_row << self.log_group) + (grid_column & (group - 1)),
                grid_column >> self.log_group)
        # cuBLASLt bands blockIdx.y into groups of ``group`` rows and walks
        # each band M-fastest.  A band that would run past the grid halves
        # until it fits, and an exact three-row remainder keeps a base-three
        # band instead of collapsing to a single row.
        width = group
        residual_three = False
        while width > 1:
            mask = width - 1
            if (grid_row | mask) < grid_y:
                break
            if mask == 3 and (grid_row | mask) == grid_y:
                residual_three = True
                break
            width >>= 1
        mask = width - 1
        linear = (grid_row & mask) * grid_x + grid_column
        if residual_three:
            n_index = linear // 3
            m_index = (grid_row & ~mask) + linear % 3
        else:
            n_index = linear >> (width.bit_length() - 1)
            m_index = (grid_row & ~mask) + (linear & mask)
        if self.serpentine and (grid_row & group) == group:
            n_index = grid_x - 1 - n_index
        return m_index, n_index
