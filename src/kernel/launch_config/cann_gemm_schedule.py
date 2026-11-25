"""Source-derived CANN GEMM scheduling primitives shared with prediction."""

from src.kernel.prediction_cache import runtime_lru_cache

from dataclasses import dataclass
import math

from src.kernel.launch_config.launch_program import CannGemmLaunchProgram


_CANN_REFERENCE_L2_BYTES = 192 * 1024 * 1024
_CANN_L2_REGION_LIMIT_BYTES = 100 * 1024 * 1024
_BF16_BYTES = 2
_CUBE_EDGE = 16
_L2_CACHE_LINE_BYTES = 512


@dataclass(frozen=True)
class CannGemmDescriptorTrace:
    """Physical requests emitted by persistent CANN GEMM workers."""

    access_keys: tuple[int, ...]
    access_sizes: tuple[int, ...]
    accounted_accesses: tuple[bool, ...]
    descriptor_count: int
    bypass_dram_bytes: int

    @property
    def l2_requested_bytes(self) -> int:
        """Return operand read bytes, excluding output replacement events."""

        return sum(
            size for size, accounted in zip(
                self.access_sizes, self.accounted_accesses)
            if accounted)

    @property
    def output_cache_access_bytes(self) -> int:
        """Return cache-line bytes touched by source-ordered output writes."""

        return sum(
            size for size, accounted in zip(
                self.access_sizes, self.accounted_accesses)
            if not accounted)

    @property
    def output_cache_access_count(self) -> int:
        """Return the number of source-ordered output cache accesses."""

        return sum(not accounted for accounted in self.accounted_accesses)

    def operand_accounting_mask(self, operand: str) -> tuple[bool, ...]:
        """Select one operand while retaining every replacement event."""

        tags = {"a": 0, "b": 1}
        if operand not in tags:
            raise ValueError("CANN GEMM trace operand must be 'a' or 'b'")
        tag = tags[operand]
        return tuple(
            accounted and (key >> 61) == tag
            for key, accounted in zip(
                self.access_keys, self.accounted_accesses)
        )


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def cann_gemm_persistent_descriptor_trace(
        program: CannGemmLaunchProgram, m: int, n: int, k: int,
        element_bytes: int) -> CannGemmDescriptorTrace:
    """Expand a V2 persistent-worker program into physical A/B L1 requests.

    The CANN schedule assigns each AIC one M/N rectangle.  Non-factorial tails
    are shifted inward, so the final rectangles overlap earlier workers while
    retaining full descriptor extents.  Each descriptor is decomposed into
    16-by-baseK physical objects so the shared LRU can recognize that overlap.
    """

    if not isinstance(program, CannGemmLaunchProgram):
        raise TypeError("CANN GEMM descriptor trace requires a launch program")
    values = {"m": m, "n": n, "k": k, "element_bytes": element_bytes}
    invalid_types = {
        name: value for name, value in values.items()
        if isinstance(value, bool) or not isinstance(value, int)
    }
    if invalid_types:
        raise TypeError(
            f"CANN GEMM descriptor trace inputs must be integers: "
            f"{invalid_types}")
    invalid_values = {
        name: value for name, value in values.items() if value <= 0
    }
    if invalid_values:
        raise ValueError(
            f"CANN GEMM descriptor trace inputs must be positive: "
            f"{invalid_values}")
    if program.traversal_order != "persistent_n_then_m":
        raise ValueError(
            "CANN persistent descriptor trace requires worker-local N/M order")
    for extent, base, name in (
            (program.single_core_m, program.base_m, "M"),
            (program.single_core_n, program.base_n, "N")):
        if extent % base:
            raise ValueError(
                f"CANN GEMM worker {name} extent is not base-tile aligned")
    if (program.single_core_k % program.base_k
            or program.a_l1_k_tiles * program.base_k
                > program.single_core_k
            or program.b_l1_k_tiles * program.base_k
                > program.single_core_k):
        raise ValueError("CANN GEMM worker K program is not base-tile aligned")

    aligned_m = _align_up(m, _CUBE_EDGE)
    aligned_n = _align_up(n, _CUBE_EDGE)
    aligned_k = _align_up(k, _CUBE_EDGE)
    for extent, grid, dimension, name in (
            (program.single_core_m, program.worker_grid_m, aligned_m, "M"),
            (program.single_core_n, program.worker_grid_n, aligned_n, "N"),
            (program.single_core_k, program.worker_grid_k, aligned_k, "K")):
        if extent * grid < dimension:
            raise ValueError(
                f"CANN GEMM worker {name} extent does not cover its dimension")

    # CANN may keep all AICs active when the final two worker rectangles map
    # to the same shifted-in tail.  That deliberate overlap is represented by
    # the min(..., dimension - extent) starts below; it is not an empty worker
    # and must remain in the physical descriptor stream.

    worker_outputs: list[tuple[tuple[int, int, int, int], ...]] = []
    worker_k_starts: list[int] = []
    mn_workers = program.worker_grid_m * program.worker_grid_n
    for worker in range(program.aic_workers):
        k_worker, mn_worker = divmod(worker, mn_workers)
        n_worker, m_worker = divmod(mn_worker, program.worker_grid_m)
        m_start = min(
            m_worker * program.single_core_m,
            aligned_m - program.single_core_m)
        n_start = min(
            n_worker * program.single_core_n,
            aligned_n - program.single_core_n)
        m_start = max(0, m_start)
        n_start = max(0, n_start)
        outputs = tuple(
            (m_start + local_m, n_start + local_n, local_m, local_n)
            for local_n in range(0, program.single_core_n, program.base_n)
            for local_m in range(0, program.single_core_m, program.base_m)
        )
        worker_outputs.append(outputs)
        worker_k_starts.append(max(0, min(
            k_worker * program.single_core_k,
            aligned_k - program.single_core_k)))

    access_keys: list[int] = []
    access_sizes: list[int] = []
    accounted_accesses: list[bool] = []
    descriptor_count = 0
    bypass_dram_bytes = 0

    def append_descriptor(
            operand: str, outer_start: int, outer_extent: int,
            k_start: int, k_extent: int) -> None:
        nonlocal descriptor_count, bypass_dram_bytes
        if operand == "a":
            physical_outer = aligned_m
            operand_tag = 0
        elif operand == "b":
            physical_outer = aligned_n
            operand_tag = 1 << 61
        else:
            raise ValueError(f"unsupported GEMM operand: {operand!r}")
        outer_extent = min(outer_extent, physical_outer - outer_start)
        descriptor_count += 1
        descriptor_bytes = outer_extent * k_extent * element_bytes
        if not program.operand_uses_l2(operand):
            bypass_dram_bytes += descriptor_bytes
            return
        for outer in range(
                outer_start, outer_start + outer_extent, _CUBE_EDGE):
            for inner_k in range(
                    k_start, k_start + k_extent, program.base_k):
                current_k = min(program.base_k, aligned_k - inner_k)
                key = operand_tag | (outer // _CUBE_EDGE << 32) | (
                    inner_k // program.base_k)
                access_keys.append(key)
                access_sizes.append(_CUBE_EDGE * current_k * element_bytes)
                accounted_accesses.append(True)

    def append_output_tile(m_start: int, n_start: int) -> None:
        """Insert the cache lines allocated by one completed FixPipe tile."""

        current_m = min(program.base_m, m - m_start)
        current_n = min(program.base_n, n - n_start)
        if current_m <= 0 or current_n <= 0:
            return
        output_tag = 2 << 61
        for row in range(m_start, m_start + current_m):
            byte_start = (row * n + n_start) * element_bytes
            byte_end = byte_start + current_n * element_bytes
            first_line = byte_start // _L2_CACHE_LINE_BYTES
            final_line = (byte_end - 1) // _L2_CACHE_LINE_BYTES
            for line in range(first_line, final_line + 1):
                access_keys.append(output_tag | line)
                access_sizes.append(_L2_CACHE_LINE_BYTES)
                accounted_accesses.append(False)

    maximum_outputs = max(len(outputs) for outputs in worker_outputs)
    a_k_extent = program.a_l1_k_tiles * program.base_k
    b_k_extent = program.b_l1_k_tiles * program.base_k
    full_a_resident = a_k_extent == program.single_core_k
    for output_index in range(maximum_outputs):
        for relative_k in range(0, program.single_core_k, program.base_k):
            for worker, outputs in enumerate(worker_outputs):
                if output_index >= len(outputs):
                    continue
                m_start, n_start, local_m, local_n = outputs[output_index]
                k_start = worker_k_starts[worker] + relative_k
                if ((not full_a_resident or local_n == 0)
                        and local_m
                            % (program.base_m * program.step_m) == 0
                        and relative_k % a_k_extent == 0):
                    append_descriptor(
                        "a", m_start, program.base_m * program.step_m,
                        k_start, min(a_k_extent, aligned_k - k_start))
                if (local_n % (program.base_n * program.step_n) == 0
                        and relative_k % b_k_extent == 0):
                    append_descriptor(
                        "b", n_start, program.base_n * program.step_n,
                        k_start, min(b_k_extent, aligned_k - k_start))
        if program.split_k == 1:
            for outputs in worker_outputs:
                if output_index >= len(outputs):
                    continue
                m_start, n_start, _, _ = outputs[output_index]
                append_output_tile(m_start, n_start)

    return CannGemmDescriptorTrace(
        access_keys=tuple(access_keys),
        access_sizes=tuple(access_sizes),
        accounted_accesses=tuple(accounted_accesses),
        descriptor_count=descriptor_count,
        bypass_dram_bytes=bypass_dram_bytes,
    )


@runtime_lru_cache(maxsize=8)
def cann_gemm_source_tile_assignments(
        m_tiles: int, n_tiles: int, m_l2_regions: int, n_l2_regions: int,
        m_l2_block: int, n_l2_block: int,
        traversal_order: str,
        aic_workers: int) -> tuple[tuple[int, int, int], ...]:
    """Expand CANN's region schedule into ``(worker, M, N)`` assignments."""

    for name, value in (
            ("m_tiles", m_tiles), ("n_tiles", n_tiles),
            ("m_l2_regions", m_l2_regions),
            ("n_l2_regions", n_l2_regions),
            ("m_l2_block", m_l2_block), ("n_l2_block", n_l2_block),
            ("aic_workers", aic_workers)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"CANN GEMM source order {name} must be an integer")
        if value <= 0:
            raise ValueError(f"CANN GEMM source order {name} must be positive")
    if traversal_order not in ("m_then_n_snake", "n_then_m_snake"):
        raise ValueError("CANN GEMM source order traversal is unsupported")

    result = []
    tiles_per_region = m_l2_block * n_l2_block
    rounds = math.ceil(tiles_per_region / aic_workers)
    tail_workers = tiles_per_region % aic_workers
    skew_period = math.lcm(m_l2_block, n_l2_block)
    region_count = m_l2_regions * n_l2_regions
    for region in range(region_count):
        block_start = region * tail_workers % aic_workers
        for round_index in range(rounds):
            for worker in range(aic_workers):
                relative_worker = (
                    worker + aic_workers - block_start) % aic_workers
                worker_rounds = (
                    rounds if tail_workers == 0
                    or relative_worker < tail_workers else rounds - 1)
                if round_index >= worker_rounds:
                    continue
                local = relative_worker + round_index * aic_workers
                if traversal_order == "m_then_n_snake":
                    local_m = local % m_l2_block
                    local_n = (local + local // skew_period) % n_l2_block
                    m_region = region // n_l2_regions
                    n_position = region % n_l2_regions
                    n_region = (
                        n_l2_regions - n_position - 1
                        if m_region % 2 else n_position)
                elif traversal_order == "n_then_m_snake":
                    local_n = local % n_l2_block
                    local_m = (local + local // skew_period) % m_l2_block
                    n_region = region // m_l2_regions
                    m_position = region % m_l2_regions
                    m_region = (
                        m_l2_regions - m_position - 1
                        if n_region % 2 else m_position)
                else:
                    raise ValueError(
                        "CANN GEMM source order traversal is unsupported")
                m_index = m_region * m_l2_block + local_m
                n_index = n_region * n_l2_block + local_n
                if m_index < m_tiles and n_index < n_tiles:
                    result.append((worker, m_index, n_index))
    expected = {
        (m_index, n_index)
        for m_index in range(m_tiles)
        for n_index in range(n_tiles)
    }
    actual = {(m_index, n_index) for _, m_index, n_index in result}
    if len(result) != len(expected) or actual != expected:
        raise ValueError("CANN GEMM L2 regions do not cover output tiles once")
    return tuple(result)


def cann_gemm_source_tile_order(
        m_tiles: int, n_tiles: int, m_l2_regions: int, n_l2_regions: int,
        m_l2_block: int, n_l2_block: int,
        traversal_order: str, aic_workers: int) -> tuple[tuple[int, int], ...]:
    """Expand CANN's region/round/worker mapping into output-tile order."""

    return tuple(
        (m_index, n_index)
        for _, m_index, n_index in cann_gemm_source_tile_assignments(
            m_tiles, n_tiles, m_l2_regions, n_l2_regions,
            m_l2_block, n_l2_block, traversal_order, aic_workers)
    )


def derive_cann_matmul_v3_l2_region_schedule(
        m: int, n: int, k: int, tile_m: int, tile_n: int,
        compute_units: int, l2_size_bytes: int,
        element_bytes: int = _BF16_BYTES) -> tuple[int, int, int, int]:
    """Reproduce CANN 8.5.1's non-transposed MatMulV3 L2 regions."""

    for name, value in (
            ("m", m), ("n", n), ("k", k), ("tile_m", tile_m),
            ("tile_n", tile_n), ("compute_units", compute_units),
            ("l2_size_bytes", l2_size_bytes),
            ("element_bytes", element_bytes)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"CANN MatMulV3 L2 {name} must be an integer")
        if value <= 0:
            raise ValueError(f"CANN MatMulV3 L2 {name} must be positive")
    if compute_units != 24:
        raise ValueError(
            "CANN MatMulV3 L2 schedule is source-backed for 24 AICs")

    m_tiles = math.ceil(m / tile_m)
    n_tiles = math.ceil(n / tile_n)
    region_limit_bytes = (
        l2_size_bytes * _CANN_L2_REGION_LIMIT_BYTES
        // _CANN_REFERENCE_L2_BYTES)
    total_bytes = (m * k + k * n + m * n) * element_bytes
    if total_bytes <= region_limit_bytes:
        return 1, 1, m_tiles, n_tiles

    if tile_n >= tile_m:
        outer_value, inner_value = n, m
        outer_tile, inner_tile = tile_n, tile_m
        outer_is_n = True
        inner_bad = False
    else:
        outer_value, inner_value = m, n
        outer_tile, inner_tile = tile_m, tile_n
        outer_is_n = False
        inner_bad = True
    maximum_conflict = min(compute_units, 6)
    minimum_conflict = min(compute_units, 3)
    inner_maximum_conflict = (
        minimum_conflict if inner_bad else maximum_conflict)
    outer_minimum_units = compute_units // maximum_conflict
    inner_minimum_units = compute_units // inner_maximum_conflict
    selected = None
    outer_conflict = 0
    inner_conflict = 0
    for outer_units in range(
            compute_units, outer_minimum_units - 1, -1):
        for inner_units in range(
                compute_units, inner_minimum_units - 1, -1):
            outer_regions = max(
                outer_value // (outer_tile * outer_units), 1)
            inner_regions = max(
                inner_value // (inner_tile * inner_units), 1)
            outer_extent = _align_up(
                math.ceil(outer_value / outer_regions), outer_tile)
            inner_extent = _align_up(
                math.ceil(inner_value / inner_regions), inner_tile)
            region_bytes = (
                outer_extent * k + k * inner_extent
                + outer_extent * inner_extent) * element_bytes
            if region_bytes > region_limit_bytes:
                continue
            outer_tail = math.ceil(
                (((outer_value + outer_extent - 1) % outer_extent) + 1)
                / outer_tile)
            inner_tail = math.ceil(
                (((inner_value + inner_extent - 1) % inner_extent) + 1)
                / inner_tile)
            if (outer_tail * maximum_conflict < compute_units
                    or inner_tail * inner_maximum_conflict < compute_units):
                continue
            candidate_outer_conflict = math.ceil(
                compute_units / outer_tail)
            candidate_inner_conflict = math.ceil(
                compute_units / inner_tail)
            if (selected is None
                    or (outer_conflict >= candidate_outer_conflict
                        and inner_conflict >= candidate_inner_conflict)):
                selected = (
                    outer_regions, inner_regions,
                    outer_extent // outer_tile,
                    inner_extent // inner_tile)
                outer_conflict = candidate_outer_conflict
                inner_conflict = candidate_inner_conflict
    if selected is None:
        raise ValueError("CANN MatMulV3 cannot derive a legal L2 region")
    _, _, outer_block, inner_block = selected
    # The source search uses provisional region counts only to choose the
    # physical block extents.  Once those extents are fixed, the serialized
    # launch program records the exact number of blocks needed to cover each
    # tile dimension.  Retaining the provisional divisor can manufacture
    # empty tail regions (for example 36 regions for 256 M tiles with an
    # eight-tile block), which changes source order and LRU residency.
    outer_regions = math.ceil(
        (n_tiles if outer_is_n else m_tiles) / outer_block)
    inner_regions = math.ceil(
        (m_tiles if outer_is_n else n_tiles) / inner_block)
    if outer_is_n:
        return inner_regions, outer_regions, inner_block, outer_block
    return outer_regions, inner_regions, outer_block, inner_block
