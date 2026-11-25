"""
Python wrapper for the C LRU simulation library (lru_sim.so).

Provides the validated combined GEMM/MoE simulations used by predictors.
Typical performance: ~100M cache ops/s on single core.
"""

from src.kernel.prediction_cache import runtime_lru_cache, RuntimeCache
import ctypes
import dataclasses
import functools
import os
import sys
import numpy as np

# Load shared library. build/lru_sim.so is pinned for provenance digests on
# every platform, so on Windows the loadable PE image is built alongside it as
# lru_sim.pyd and preferred here.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if sys.platform == "win32":
    _lib_names = ("lru_sim.pyd", "lru_sim.dll", "lru_sim.so")
else:
    _lib_names = ("lru_sim.so", "lru_sim.dll", "lru_sim.pyd")
_lib_candidates = [
    os.path.join(_project_root, "build", name) for name in _lib_names
]
_lib_path = next(
    (candidate for candidate in _lib_candidates if os.path.exists(candidate)),
    _lib_candidates[0],
)
_lib = ctypes.CDLL(_lib_path)


_lib.simulate_lru_trace.restype = None
_lib.simulate_lru_trace.argtypes = [
    ctypes.POINTER(ctypes.c_uint64),  # physical object keys [access_count]
    ctypes.POINTER(ctypes.c_int32),   # object sizes [access_count]
    ctypes.c_int32,                   # access_count
    ctypes.c_int64,                   # capacity_bytes
    ctypes.POINTER(ctypes.c_int64),   # out[requested_bytes, miss_bytes]
]

_lib.simulate_lru_trace_accounted.restype = None
_lib.simulate_lru_trace_accounted.argtypes = [
    ctypes.POINTER(ctypes.c_uint64),  # physical object keys [access_count]
    ctypes.POINTER(ctypes.c_int32),   # object sizes [access_count]
    ctypes.POINTER(ctypes.c_uint8),   # accesses included in returned traffic
    ctypes.c_int32,                   # access_count
    ctypes.c_int64,                   # capacity_bytes
    ctypes.POINTER(ctypes.c_int64),   # out[requested_bytes, miss_bytes]
]

_lib.simulate_lru_trace_stateful.restype = None
_lib.simulate_lru_trace_stateful.argtypes = [
    ctypes.POINTER(ctypes.c_uint64),  # physical object keys [access_count]
    ctypes.POINTER(ctypes.c_int32),   # object sizes [access_count]
    ctypes.POINTER(ctypes.c_uint8),   # 0=read, 1=write
    ctypes.POINTER(ctypes.c_uint8),   # accesses included in returned traffic
    ctypes.c_int32,                   # access_count
    ctypes.c_int64,                   # capacity_bytes
    ctypes.c_int64,                   # anonymous initial dirty occupancy
    ctypes.POINTER(ctypes.c_int64),   # out[request, miss, writeback, dirty, anonymous]
]

_lib.simulate_lru_trace_stateful_dual_accounted.restype = None
_lib.simulate_lru_trace_stateful_dual_accounted.argtypes = [
    ctypes.POINTER(ctypes.c_uint64),  # physical object keys [access_count]
    ctypes.POINTER(ctypes.c_int32),   # object sizes [access_count]
    ctypes.POINTER(ctypes.c_uint8),   # 0=read, 1=write
    ctypes.POINTER(ctypes.c_uint8),   # first accounting mask
    ctypes.POINTER(ctypes.c_uint8),   # second accounting mask
    ctypes.c_int32,                   # access_count
    ctypes.c_int64,                   # capacity_bytes
    ctypes.c_int64,                   # anonymous initial dirty occupancy
    ctypes.POINTER(ctypes.c_int64),   # out[first req, first miss, second req, second miss, dirty, anonymous]
]

_lib.simulate_cann_gemm_persistent_stateful_dual_accounted.restype = None
_lib.simulate_cann_gemm_persistent_stateful_dual_accounted.argtypes = [
    ctypes.c_int32,  # aic_workers
    ctypes.c_int32,  # worker_grid_m
    ctypes.c_int32,  # worker_grid_n
    ctypes.c_int32,  # worker_grid_k
    ctypes.c_int32,  # single_core_m
    ctypes.c_int32,  # single_core_n
    ctypes.c_int32,  # single_core_k
    ctypes.c_int32,  # base_m
    ctypes.c_int32,  # base_n
    ctypes.c_int32,  # base_k
    ctypes.c_int32,  # a_l1_k_tiles
    ctypes.c_int32,  # b_l1_k_tiles
    ctypes.c_int32,  # step_m
    ctypes.c_int32,  # step_n
    ctypes.c_int64,  # rows
    ctypes.c_int64,  # n
    ctypes.c_int64,  # k
    ctypes.c_int32,  # element bytes
    ctypes.c_int64,  # L2 replacement capacity
    ctypes.c_int64,  # anonymous initial dirty occupancy
    ctypes.POINTER(ctypes.c_int64),
]


def simulate_lru_trace(access_keys, access_sizes,
                       capacity_bytes: int) -> tuple[int, int]:
    """Replay one ordered physical-object stream through the shared LRU.

    A key identifies one physical cache object and therefore must always have
    the same byte size.  Hardware launch adapters own event construction;
    this cache solver is intentionally unaware of GEMM variants and callers.
    """

    keys = np.asarray(access_keys)
    sizes = np.asarray(access_sizes)
    if keys.ndim != 1 or sizes.ndim != 1 or keys.shape != sizes.shape:
        raise TypeError("LRU keys and sizes must be equal-length 1-D arrays")
    if keys.size == 0:
        raise ValueError("LRU access trace cannot be empty")
    if keys.dtype.kind not in "iu" or sizes.dtype.kind not in "iu":
        raise TypeError("LRU keys and sizes must contain integers")
    if np.any(keys < 0):
        raise ValueError("LRU access keys must be non-negative")
    if np.any(sizes <= 0):
        raise ValueError("LRU access sizes must be positive")
    if (isinstance(capacity_bytes, bool)
            or not isinstance(capacity_bytes, int)):
        raise TypeError("LRU capacity must be an integer")
    if capacity_bytes <= 0:
        raise ValueError("LRU capacity must be positive")
    if keys.size > np.iinfo(np.int32).max:
        raise ValueError("LRU access trace exceeds the C ABI length")
    if np.any(sizes > np.iinfo(np.int32).max):
        raise ValueError("LRU access size exceeds the C ABI range")

    object_sizes = {}
    for key, size in zip(keys.tolist(), sizes.tolist()):
        if key in object_sizes and object_sizes[key] != size:
            raise ValueError("one LRU object key cannot change byte size")
        object_sizes[key] = size

    keys = np.ascontiguousarray(keys, dtype=np.uint64)
    sizes = np.ascontiguousarray(sizes, dtype=np.int32)
    output = np.zeros(2, dtype=np.int64)
    _lib.simulate_lru_trace(
        keys.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(keys.size),
        ctypes.c_int64(capacity_bytes),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    return int(output[0]), int(output[1])


def simulate_lru_trace_accounted(
        access_keys, access_sizes, accounted_accesses,
        capacity_bytes: int) -> tuple[int, int]:
    """Replay every access while reporting traffic for a selected subset.

    All entries allocate and update LRU recency.  ``accounted_accesses`` only
    controls which requests and misses enter the returned byte totals.  This
    lets a write-allocate stream evict read operands without misclassifying
    the writes as DRAM reads.
    """

    keys = np.asarray(access_keys)
    sizes = np.asarray(access_sizes)
    accounted = np.asarray(accounted_accesses)
    if (keys.ndim != 1 or sizes.ndim != 1 or accounted.ndim != 1
            or keys.shape != sizes.shape or keys.shape != accounted.shape):
        raise TypeError(
            "LRU keys, sizes, and accounting mask must be equal-length "
            "1-D arrays")
    if keys.size == 0:
        raise ValueError("LRU access trace cannot be empty")
    if keys.dtype.kind not in "iu" or sizes.dtype.kind not in "iu":
        raise TypeError("LRU keys and sizes must contain integers")
    if accounted.dtype.kind not in "biu":
        raise TypeError("LRU accounting mask must contain booleans")
    if np.any(keys < 0):
        raise ValueError("LRU access keys must be non-negative")
    if np.any(sizes <= 0):
        raise ValueError("LRU access sizes must be positive")
    if np.any((accounted != 0) & (accounted != 1)):
        raise ValueError("LRU accounting mask must contain only 0 or 1")
    if (isinstance(capacity_bytes, bool)
            or not isinstance(capacity_bytes, int)):
        raise TypeError("LRU capacity must be an integer")
    if capacity_bytes <= 0:
        raise ValueError("LRU capacity must be positive")
    if keys.size > np.iinfo(np.int32).max:
        raise ValueError("LRU access trace exceeds the C ABI length")
    if np.any(sizes > np.iinfo(np.int32).max):
        raise ValueError("LRU access size exceeds the C ABI range")

    object_sizes = {}
    for key, size in zip(keys.tolist(), sizes.tolist()):
        if key in object_sizes and object_sizes[key] != size:
            raise ValueError("one LRU object key cannot change byte size")
        object_sizes[key] = size

    keys = np.ascontiguousarray(keys, dtype=np.uint64)
    sizes = np.ascontiguousarray(sizes, dtype=np.int32)
    accounted = np.ascontiguousarray(accounted, dtype=np.uint8)
    output = np.zeros(2, dtype=np.int64)
    _lib.simulate_lru_trace_accounted(
        keys.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        accounted.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(keys.size),
        ctypes.c_int64(capacity_bytes),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    return int(output[0]), int(output[1])


def simulate_lru_trace_stateful(
        access_keys, access_sizes, access_kinds, accounted_accesses,
        capacity_bytes: int, initial_dirty_bytes: int,
        ) -> tuple[int, int, int, int, int]:
    """Replay an ordered read/write stream through a dirty write-back LRU.

    Initial dirty occupancy represents older anonymous cache content at the
    LRU tail. It can be displaced but never hit by this trace. Returned
    values are selected request bytes, selected read-miss bytes, all dirty
    victim bytes, all dirty bytes still resident after replay, and the
    anonymous portion of those resident dirty bytes.
    """

    keys = np.asarray(access_keys)
    sizes = np.asarray(access_sizes)
    kinds = np.asarray(access_kinds)
    accounted = np.asarray(accounted_accesses)
    if (keys.ndim != 1 or sizes.ndim != 1 or kinds.ndim != 1
            or accounted.ndim != 1 or keys.shape != sizes.shape
            or keys.shape != kinds.shape or keys.shape != accounted.shape):
        raise TypeError(
            "stateful LRU inputs must be equal-length 1-D arrays")
    if keys.size == 0:
        raise ValueError("LRU access trace cannot be empty")
    if (keys.dtype.kind not in "iu" or sizes.dtype.kind not in "iu"
            or kinds.dtype.kind not in "biu"
            or accounted.dtype.kind not in "biu"):
        raise TypeError("stateful LRU inputs must contain integers")
    if np.any(keys < 0):
        raise ValueError("LRU access keys must be non-negative")
    if np.any(sizes <= 0):
        raise ValueError("LRU access sizes must be positive")
    if np.any((kinds != 0) & (kinds != 1)):
        raise ValueError("LRU access kind must be 0 (read) or 1 (write)")
    if np.any((accounted != 0) & (accounted != 1)):
        raise ValueError("LRU accounting mask must contain only 0 or 1")
    for name, value in (
            ("capacity", capacity_bytes),
            ("initial dirty occupancy", initial_dirty_bytes)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"LRU {name} must be an integer")
    if capacity_bytes <= 0:
        raise ValueError("LRU capacity must be positive")
    if not 0 <= initial_dirty_bytes <= capacity_bytes:
        raise ValueError("LRU initial dirty occupancy must fit in capacity")
    if keys.size > np.iinfo(np.int32).max:
        raise ValueError("LRU access trace exceeds the C ABI length")
    if np.any(sizes > np.iinfo(np.int32).max):
        raise ValueError("LRU access size exceeds the C ABI range")

    object_sizes = {}
    for key, size in zip(keys.tolist(), sizes.tolist()):
        if key in object_sizes and object_sizes[key] != size:
            raise ValueError("one LRU object key cannot change byte size")
        object_sizes[key] = size

    keys = np.ascontiguousarray(keys, dtype=np.uint64)
    sizes = np.ascontiguousarray(sizes, dtype=np.int32)
    kinds = np.ascontiguousarray(kinds, dtype=np.uint8)
    accounted = np.ascontiguousarray(accounted, dtype=np.uint8)
    output = np.zeros(5, dtype=np.int64)
    _lib.simulate_lru_trace_stateful(
        keys.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        kinds.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        accounted.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(keys.size),
        ctypes.c_int64(capacity_bytes),
        ctypes.c_int64(initial_dirty_bytes),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    return tuple(int(value) for value in output)


def simulate_lru_trace_stateful_dual_accounted(
        access_keys, access_sizes, access_kinds,
        first_accounted_accesses, second_accounted_accesses,
        capacity_bytes: int, initial_dirty_bytes: int,
        ) -> tuple[int, int, int, int, int, int, int]:
    """Replay one dirty LRU trace while pricing two exact access subsets.

    Both accounting masks observe the same cache state.  The returned values
    are first request/miss bytes, second request/miss bytes, then the shared
    dirty-victim, terminal-dirty, and terminal-anonymous-dirty byte counts.
    """

    keys = np.asarray(access_keys)
    sizes = np.asarray(access_sizes)
    kinds = np.asarray(access_kinds)
    first_accounted = np.asarray(first_accounted_accesses)
    second_accounted = np.asarray(second_accounted_accesses)
    if (keys.ndim != 1 or sizes.ndim != 1 or kinds.ndim != 1
            or first_accounted.ndim != 1 or second_accounted.ndim != 1
            or keys.shape != sizes.shape or keys.shape != kinds.shape
            or keys.shape != first_accounted.shape
            or keys.shape != second_accounted.shape):
        raise TypeError(
            "dual stateful LRU inputs must be equal-length 1-D arrays")
    if keys.size == 0:
        raise ValueError("LRU access trace cannot be empty")
    if (keys.dtype.kind not in "iu" or sizes.dtype.kind not in "iu"
            or kinds.dtype.kind not in "biu"
            or first_accounted.dtype.kind not in "biu"
            or second_accounted.dtype.kind not in "biu"):
        raise TypeError("dual stateful LRU inputs must contain integers")
    if np.any(keys < 0):
        raise ValueError("LRU access keys must be non-negative")
    if np.any(sizes <= 0):
        raise ValueError("LRU access sizes must be positive")
    if np.any((kinds != 0) & (kinds != 1)):
        raise ValueError("LRU access kind must be 0 (read) or 1 (write)")
    if np.any((first_accounted != 0) & (first_accounted != 1)):
        raise ValueError("first LRU accounting mask must contain only 0 or 1")
    if np.any((second_accounted != 0) & (second_accounted != 1)):
        raise ValueError("second LRU accounting mask must contain only 0 or 1")
    for name, value in (
            ("capacity", capacity_bytes),
            ("initial dirty occupancy", initial_dirty_bytes)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"LRU {name} must be an integer")
    if capacity_bytes <= 0:
        raise ValueError("LRU capacity must be positive")
    if not 0 <= initial_dirty_bytes <= capacity_bytes:
        raise ValueError("LRU initial dirty occupancy must fit in capacity")
    if keys.size > np.iinfo(np.int32).max:
        raise ValueError("LRU access trace exceeds the C ABI length")
    if np.any(sizes > np.iinfo(np.int32).max):
        raise ValueError("LRU access size exceeds the C ABI range")

    object_sizes = {}
    for key, size in zip(keys.tolist(), sizes.tolist()):
        if key in object_sizes and object_sizes[key] != size:
            raise ValueError("one LRU object key cannot change byte size")
        object_sizes[key] = size

    keys = np.ascontiguousarray(keys, dtype=np.uint64)
    sizes = np.ascontiguousarray(sizes, dtype=np.int32)
    kinds = np.ascontiguousarray(kinds, dtype=np.uint8)
    first_accounted = np.ascontiguousarray(first_accounted, dtype=np.uint8)
    second_accounted = np.ascontiguousarray(second_accounted, dtype=np.uint8)
    output = np.zeros(7, dtype=np.int64)
    _lib.simulate_lru_trace_stateful_dual_accounted(
        keys.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        kinds.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        first_accounted.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        second_accounted.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(keys.size),
        ctypes.c_int64(capacity_bytes),
        ctypes.c_int64(initial_dirty_bytes),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    return tuple(int(value) for value in output)


@functools.lru_cache(maxsize=512)
def _simulate_cann_gemm_persistent_stateful_dual_accounted_validated(
        aic_workers, worker_grid_m, worker_grid_n, worker_grid_k,
        single_core_m, single_core_n, single_core_k, base_m, base_n, base_k,
        a_l1_k_tiles, b_l1_k_tiles, step_m, step_n, rows, n, k,
        element_bytes, capacity_bytes, initial_dirty_bytes):
    """Memoize the pure native replay after public contract validation."""
    output = np.zeros(7, dtype=np.int64)
    _lib.simulate_cann_gemm_persistent_stateful_dual_accounted(
        ctypes.c_int32(aic_workers),
        ctypes.c_int32(worker_grid_m),
        ctypes.c_int32(worker_grid_n),
        ctypes.c_int32(worker_grid_k),
        ctypes.c_int32(single_core_m),
        ctypes.c_int32(single_core_n),
        ctypes.c_int32(single_core_k),
        ctypes.c_int32(base_m),
        ctypes.c_int32(base_n),
        ctypes.c_int32(base_k),
        ctypes.c_int32(a_l1_k_tiles),
        ctypes.c_int32(b_l1_k_tiles),
        ctypes.c_int32(step_m),
        ctypes.c_int32(step_n),
        ctypes.c_int64(rows),
        ctypes.c_int64(n),
        ctypes.c_int64(k),
        ctypes.c_int32(element_bytes),
        ctypes.c_int64(capacity_bytes),
        ctypes.c_int64(initial_dirty_bytes),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    return tuple(int(value) for value in output)


def simulate_cann_gemm_persistent_stateful_dual_accounted(
        *, aic_workers: int, worker_grid_m: int, worker_grid_n: int,
        worker_grid_k: int, single_core_m: int, single_core_n: int,
        single_core_k: int, base_m: int, base_n: int, base_k: int,
        a_l1_k_tiles: int, b_l1_k_tiles: int, step_m: int, step_n: int,
        rows: int, n: int, k: int, element_bytes: int,
        capacity_bytes: int, initial_dirty_bytes: int,
        ) -> tuple[int, int, int, int, int, int, int]:
    """Replay a typed CANN persistent GEMM program without Python events.

    The native implementation preserves the descriptor and output-cache access
    ordering of ``cann_gemm_persistent_descriptor_trace`` while deriving each
    event directly from its source-defined launch program.
    """

    values = {
        "aic workers": aic_workers,
        "worker M grid": worker_grid_m,
        "worker N grid": worker_grid_n,
        "worker K grid": worker_grid_k,
        "single-core M": single_core_m,
        "single-core N": single_core_n,
        "single-core K": single_core_k,
        "base M": base_m,
        "base N": base_n,
        "base K": base_k,
        "A L1 K tiles": a_l1_k_tiles,
        "B L1 K tiles": b_l1_k_tiles,
        "M step": step_m,
        "N step": step_n,
        "rows": rows,
        "N": n,
        "K": k,
        "element bytes": element_bytes,
        "capacity": capacity_bytes,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"CANN persistent {name} must be an integer")
        if value <= 0:
            raise ValueError(f"CANN persistent {name} must be positive")
    if (isinstance(initial_dirty_bytes, bool)
            or not isinstance(initial_dirty_bytes, int)):
        raise TypeError("CANN persistent initial dirty occupancy must be an integer")
    if not 0 <= initial_dirty_bytes <= capacity_bytes:
        raise ValueError(
            "CANN persistent initial dirty occupancy must fit in capacity")
    int32_values = (
        aic_workers, worker_grid_m, worker_grid_n, worker_grid_k,
        single_core_m, single_core_n, single_core_k,
        base_m, base_n, base_k, a_l1_k_tiles, b_l1_k_tiles,
        step_m, step_n, element_bytes,
    )
    if any(value > np.iinfo(np.int32).max for value in int32_values):
        raise ValueError("CANN persistent program exceeds the C ABI range")
    return _simulate_cann_gemm_persistent_stateful_dual_accounted_validated(
        aic_workers, worker_grid_m, worker_grid_n, worker_grid_k,
        single_core_m, single_core_n, single_core_k, base_m, base_n, base_k,
        a_l1_k_tiles, b_l1_k_tiles, step_m, step_n, rows, n, k,
        element_bytes, capacity_bytes, initial_dirty_bytes,
    )


# ─── Combined swizzle search for GEMM ───

_lib.simulate_gemm_swizzle_search.restype = None
_lib.simulate_gemm_swizzle_search.argtypes = [
    ctypes.c_int32,                   # grid_x
    ctypes.c_int32,                   # grid_y
    ctypes.c_int32,                   # conv (0=M, 1=N)
    ctypes.c_int32,                   # W
    ctypes.c_int32,                   # mt
    ctypes.c_int32,                   # nt
    ctypes.c_int32,                   # max_sw
    ctypes.c_int32,                   # k_steps
    ctypes.c_int32,                   # a_tile_bytes
    ctypes.c_int32,                   # b_tile_bytes
    ctypes.c_int32,                   # c_tile_bytes
    ctypes.c_int32,                   # wave_size
    ctypes.c_int64,                   # l2_capacity_bytes
    ctypes.POINTER(ctypes.c_int64),  # out[2]
]


def simulate_gemm_swizzle_search(grid_x: int, grid_y: int,
                                  conv: str, W: int,
                                  mt: int, nt: int, max_sw: int,
                                  k_steps: int,
                                  a_tile_bytes: int, b_tile_bytes: int,
                                  c_tile_bytes: int,
                                  wave_size: int,
                                  l2_capacity: int) -> tuple:
    """Combined swizzle search + LRU simulation for GEMM.

    Tries all swizzle values 0..max_sw, builds schedule in C, runs LRU sim,
    and returns (best_l2, best_dram). Raises if no valid schedule exists.
    """
    if conv == "M":
        conv_int = 0
    elif conv == "N":
        conv_int = 1
    else:
        raise ValueError(f"Unknown swizzle convention: {conv}")

    out = np.zeros(2, dtype=np.int64)
    _lib.simulate_gemm_swizzle_search(
        ctypes.c_int32(grid_x),
        ctypes.c_int32(grid_y),
        ctypes.c_int32(conv_int),
        ctypes.c_int32(W),
        ctypes.c_int32(mt),
        ctypes.c_int32(nt),
        ctypes.c_int32(max_sw),
        ctypes.c_int32(k_steps),
        ctypes.c_int32(a_tile_bytes),
        ctypes.c_int32(b_tile_bytes),
        ctypes.c_int32(c_tile_bytes),
        ctypes.c_int32(wave_size),
        ctypes.c_int64(l2_capacity),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    if out[1] == np.iinfo(np.int64).max:
        raise ValueError(
            "simulate_gemm_swizzle_search: no valid swizzle for "
            f"grid=({grid_x},{grid_y}) mt={mt} nt={nt}")
    return int(out[0]), int(out[1])


_lib.simulate_gemm_lru.restype = None
_lib.simulate_gemm_lru.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # schedule_mn [n_ctas, 2]
    ctypes.c_int32,                   # n_ctas
    ctypes.c_int32,                   # k_steps
    ctypes.c_int32,                   # a_tile_bytes
    ctypes.c_int32,                   # b_tile_bytes
    ctypes.c_int32,                   # c_tile_bytes
    ctypes.c_int32,                   # wave_size
    ctypes.c_int64,                   # l2_capacity_bytes
    ctypes.POINTER(ctypes.c_int64),  # out[2]
]

_lib.simulate_gemm_n_major_lru.restype = None
_lib.simulate_gemm_n_major_lru.argtypes = [
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int64),
]


def _simulate_gemm_schedule(schedule: np.ndarray, k_steps: int,
                            a_tile_bytes: int, b_tile_bytes: int,
                            c_tile_bytes: int, wave_size: int,
                            l2_capacity: int) -> tuple[int, int]:
    """Run the validated C LRU engine for one explicit logical schedule."""

    if (not isinstance(schedule, np.ndarray) or schedule.ndim != 2
            or schedule.shape[1] != 2 or schedule.shape[0] <= 0):
        raise TypeError(
            "GEMM schedule must be a non-empty NumPy array with shape [N, 2]")
    schedule = np.ascontiguousarray(schedule, dtype=np.int32)
    output = np.zeros(2, dtype=np.int64)
    _lib.simulate_gemm_lru(
        schedule.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(schedule.shape[0]),
        ctypes.c_int32(k_steps),
        ctypes.c_int32(a_tile_bytes),
        ctypes.c_int32(b_tile_bytes),
        ctypes.c_int32(c_tile_bytes),
        ctypes.c_int32(wave_size),
        ctypes.c_int64(l2_capacity),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    return int(output[0]), int(output[1])


_lib.simulate_gemm_lru_tails.restype = None
_lib.simulate_gemm_lru_tails.argtypes = [
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int64),
]


@dataclasses.dataclass(frozen=True)
class GemmSourceOrderTraffic:
    """Per-stream traffic from one shared-L2 GEMM replay."""

    l2_bytes: int
    dram_bytes: int
    a_l2_bytes: int
    a_dram_bytes: int
    b_l2_bytes: int
    b_dram_bytes: int
    output_l2_bytes: int
    dirty_writeback_bytes: int = 0
    predecessor_dirty_writeback_bytes: int = 0


def simulate_gemm_source_order_traffic(
        schedule_mn, m: int, n: int, k: int, tile_m: int, tile_n: int,
        tile_k: int, element_bytes: int, wave_size: int,
        l2_capacity: int,
        residency_profile: tuple[tuple[int, float], ...] = (),
        ) -> GemmSourceOrderTraffic:
    """Replay one explicit GEMM order and preserve operand miss identity.

    The replacement decision behind this replay is the shared miss core; the
    order and the tile extents are this family's launch facts.
    """

    schedule = np.ascontiguousarray(schedule_mn, dtype=np.int32)
    if schedule.ndim != 2 or schedule.shape[1] != 2 or schedule.shape[0] <= 0:
        raise TypeError("GEMM source order must have shape [N, 2]")
    values = (m, n, k, tile_m, tile_n, tile_k, element_bytes, wave_size,
              l2_capacity)
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in values):
        raise TypeError("GEMM source-order LRU inputs must be integers")
    if any(value <= 0 for value in values):
        raise ValueError("GEMM source-order LRU inputs must be positive")
    occupancy, fraction = _residency_profile_arrays(residency_profile)
    output = np.zeros(7, dtype=np.int64)
    _lib.simulate_gemm_lru_tails(
        schedule.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(schedule.shape[0]),
        ctypes.c_int32(m), ctypes.c_int32(n), ctypes.c_int32(k),
        ctypes.c_int32(tile_m), ctypes.c_int32(tile_n),
        ctypes.c_int32(tile_k), ctypes.c_int32(element_bytes),
        ctypes.c_int32(wave_size), ctypes.c_int64(l2_capacity),
        occupancy.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        fraction.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int32(occupancy.size),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    return GemmSourceOrderTraffic(*(int(value) for value in output))


def simulate_gemm_source_order(
        schedule_mn, m: int, n: int, k: int, tile_m: int, tile_n: int,
        tile_k: int, element_bytes: int, wave_size: int,
        l2_capacity: int) -> tuple[int, int]:
    """Replay one explicit GEMM output-tile order with exact edge tiles."""

    traffic = simulate_gemm_source_order_traffic(
        schedule_mn, m, n, k, tile_m, tile_n, tile_k, element_bytes,
        wave_size, l2_capacity)
    return traffic.l2_bytes, traffic.dram_bytes


@runtime_lru_cache(maxsize=512)
def _simulate_gemm_n_major_cached(
        mt: int, nt: int, k_steps: int, a_tile_bytes: int,
        b_tile_bytes: int, c_tile_bytes: int, wave_size: int,
        l2_capacity: int) -> tuple[int, int]:
    """Replay one validated source schedule for its exact physical inputs."""

    # The established generic key encoding stores M/N/K tile identities in
    # 16-bit fields.  Preserve its collision behavior outside that addressable
    # range instead of making the compact native N-major path subtly differ.
    if mt > 1 << 16 or nt > 1 << 16 or k_steps > 1 << 16:
        schedule = np.empty((mt * nt, 2), dtype=np.int32)
        schedule[:, 0] = np.tile(np.arange(mt, dtype=np.int32), nt)
        schedule[:, 1] = np.repeat(np.arange(nt, dtype=np.int32), mt)
        return _simulate_gemm_schedule(
            schedule, k_steps, a_tile_bytes, b_tile_bytes, c_tile_bytes,
            wave_size, l2_capacity)
    output = np.zeros(2, dtype=np.int64)
    _lib.simulate_gemm_n_major_lru(
        ctypes.c_int32(mt), ctypes.c_int32(nt), ctypes.c_int32(k_steps),
        ctypes.c_int32(a_tile_bytes), ctypes.c_int32(b_tile_bytes),
        ctypes.c_int32(c_tile_bytes), ctypes.c_int32(wave_size),
        ctypes.c_int64(l2_capacity),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    return int(output[0]), int(output[1])


def simulate_gemm_n_major(mt: int, nt: int, k_steps: int,
                          a_tile_bytes: int, b_tile_bytes: int,
                          c_tile_bytes: int, wave_size: int,
                          l2_capacity: int) -> tuple[int, int]:
    """Simulate the logical N-major tile schedule used by MetaX mcBLAS.

    mcBLAS launch grids encode implementation work distribution, including
    padding and persistent workgroups, rather than a CUTLASS-style bijection
    from ``grid.(x, y)`` to output tiles. Keep that measured grid for launch
    resources while simulating traffic over the complete logical tile set.
    """

    values = {
        "mt": mt,
        "nt": nt,
        "k_steps": k_steps,
        "a_tile_bytes": a_tile_bytes,
        "b_tile_bytes": b_tile_bytes,
        "c_tile_bytes": c_tile_bytes,
        "wave_size": wave_size,
        "l2_capacity": l2_capacity,
    }
    invalid_types = {
        name: value for name, value in values.items()
        if isinstance(value, bool) or not isinstance(value, int)
    }
    if invalid_types:
        raise TypeError(
            f"GEMM N-major simulation inputs must be integers: {invalid_types}")
    invalid_values = {
        name: value for name, value in values.items() if value <= 0
    }
    if invalid_values:
        raise ValueError(
            f"GEMM N-major simulation inputs must be positive: {invalid_values}")

    return _simulate_gemm_n_major_cached(
        mt, nt, k_steps, a_tile_bytes, b_tile_bytes, c_tile_bytes,
        wave_size, l2_capacity)


def simulate_gemm_launch_order(grid_x: int, grid_y: int, convention: str,
                               width: int, mt: int, nt: int, k_steps: int,
                               a_tile_bytes: int, b_tile_bytes: int,
                               c_tile_bytes: int, wave_size: int,
                               l2_capacity: int) -> tuple[int, int]:
    """Simulate the logical tile order encoded by a 2-D launch grid.

    CUDA/MACA enumerate ``blockIdx.x`` before ``blockIdx.y``.  ``convention``
    and ``width`` describe the exact reversible mapping from those axes to the
    logical M/N tile coordinates.  Unlike the legacy swizzle search, this uses
    the observed launch order instead of choosing an order that minimizes DRAM
    traffic.
    """

    values = {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "width": width,
        "mt": mt,
        "nt": nt,
        "k_steps": k_steps,
        "a_tile_bytes": a_tile_bytes,
        "b_tile_bytes": b_tile_bytes,
        "c_tile_bytes": c_tile_bytes,
        "wave_size": wave_size,
        "l2_capacity": l2_capacity,
    }
    invalid_types = {
        name: value for name, value in values.items()
        if isinstance(value, bool) or not isinstance(value, int)
    }
    if invalid_types:
        raise TypeError(
            f"GEMM launch-order inputs must be integers: {invalid_types}")
    invalid_values = {
        name: value for name, value in values.items() if value <= 0
    }
    if invalid_values:
        raise ValueError(
            f"GEMM launch-order inputs must be positive: {invalid_values}")
    if convention not in ("M", "N"):
        raise ValueError(
            f"GEMM launch-order convention must be 'M' or 'N': "
            f"{convention!r}")

    schedule = np.empty((grid_x * grid_y, 2), dtype=np.int32)
    count = 0
    for grid_y_index in range(grid_y):
        for grid_x_index in range(grid_x):
            if convention == "M":
                m_index = grid_x_index // width
                n_index = grid_y_index * width + grid_x_index % width
            elif convention == "N":
                n_index = grid_x_index // width
                m_index = grid_y_index * width + grid_x_index % width
            else:
                raise AssertionError(
                    f"unvalidated GEMM launch convention: {convention!r}")
            if m_index < mt and n_index < nt:
                schedule[count] = (m_index, n_index)
                count += 1
    if count != mt * nt:
        raise ValueError(
            "GEMM launch grid does not map bijectively to logical tiles: "
            f"grid=({grid_x}, {grid_y}), convention={convention}, "
            f"width={width}, tiles=({mt}, {nt}), mapped={count}")
    return _simulate_gemm_schedule(
        schedule[:count], k_steps, a_tile_bytes, b_tile_bytes, c_tile_bytes,
        wave_size, l2_capacity)


_lib.gemm_swizzle_best_schedule.restype = ctypes.c_int32
_lib.gemm_swizzle_best_schedule.argtypes = [
    ctypes.c_int32, ctypes.c_int32,   # grid_x, grid_y
    ctypes.c_int32, ctypes.c_int32,   # conv, W
    ctypes.c_int32, ctypes.c_int32,   # mt, nt
    ctypes.c_int32,                    # max_sw
    ctypes.c_int32,                    # k_steps
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,  # a/b/c tile bytes
    ctypes.c_int32,                    # wave_size
    ctypes.c_int64,                    # l2_capacity_bytes
    ctypes.POINTER(ctypes.c_int32),   # out_sched [mt*nt, 2]
]


def gemm_swizzle_best_schedule(grid_x: int, grid_y: int, conv: str, W: int,
                               mt: int, nt: int, max_sw: int, k_steps: int,
                               a_tile_bytes: int, b_tile_bytes: int,
                               c_tile_bytes: int, wave_size: int,
                               l2_capacity: int) -> np.ndarray:
    """Return the best-swizzle CTA schedule [n_ctas, 2] chosen by the GEMM
    traffic predictor's swizzle search, so the stack-distance hit curve can be
    computed on exactly that schedule. Raises if no valid swizzle exists."""
    if conv == "M":
        conv_int = 0
    elif conv == "N":
        conv_int = 1
    else:
        raise ValueError(f"Unknown swizzle convention: {conv}")
    out_sched = np.zeros((mt * nt, 2), dtype=np.int32)
    count = _lib.gemm_swizzle_best_schedule(
        ctypes.c_int32(grid_x), ctypes.c_int32(grid_y),
        ctypes.c_int32(conv_int), ctypes.c_int32(W),
        ctypes.c_int32(mt), ctypes.c_int32(nt),
        ctypes.c_int32(max_sw), ctypes.c_int32(k_steps),
        ctypes.c_int32(a_tile_bytes), ctypes.c_int32(b_tile_bytes),
        ctypes.c_int32(c_tile_bytes), ctypes.c_int32(wave_size),
        ctypes.c_int64(l2_capacity),
        out_sched.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    if count < 0:
        raise ValueError(
            f"gemm_swizzle_best_schedule: no valid swizzle for "
            f"grid=({grid_x},{grid_y}) mt={mt} nt={nt}")
    return out_sched[:count]


_lib.simulate_moe_full.restype = None
_lib.simulate_moe_full.argtypes = [
    ctypes.c_int32,                   # mt
    ctypes.c_int32,                   # nt
    ctypes.c_int32,                   # gsm
    ctypes.c_int32,                   # k_steps
    ctypes.c_int32,                   # a_tile_bytes
    ctypes.c_int32,                   # b_tile_bytes
    ctypes.c_int32,                   # c_tile_bytes
    ctypes.c_int32,                   # wave_size
    ctypes.c_int64,                   # l2_capacity_bytes
    ctypes.POINTER(ctypes.c_int32),  # tok_per_expert
    ctypes.c_int32,                   # E
    ctypes.c_int32,                   # BM
    ctypes.c_int32,                   # EM
    ctypes.POINTER(ctypes.c_int64),  # out[2]
]

_lib.simulate_moe_ab.restype = None
_lib.simulate_moe_ab.argtypes = [
    ctypes.c_int32,                   # mt
    ctypes.c_int32,                   # nt
    ctypes.c_int32,                   # gsm
    ctypes.c_int32,                   # k_steps
    ctypes.c_int32,                   # a_tile_bytes
    ctypes.c_int32,                   # b_tile_bytes
    ctypes.c_int32,                   # wave_size
    ctypes.c_int64,                   # l2_capacity_bytes
    ctypes.c_int64,                   # near_bytes
    ctypes.POINTER(ctypes.c_int32),  # tok_per_expert
    ctypes.c_int32,                   # E
    ctypes.c_int32,                   # BM
    ctypes.POINTER(ctypes.c_int64),  # out[6]
]


_moe_full_cache: dict = RuntimeCache()


def simulate_moe_full(mt: int, nt: int, gsm: int,
                      k_steps: int,
                      a_tile_bytes: int, b_tile_bytes: int, c_tile_bytes: int,
                      wave_size: int, l2_capacity: int,
                      tok_per_expert: list, E: int,
                      BM: int, EM: int) -> tuple:
    """Combined MoE schedule build + expert assignment + LRU simulation.

    Returns (l2_total, dram_total).

    Memoized: the underlying C sim is a pure function of these arguments, so
    identical inputs return an identical (l2_total, dram_total). Datasets can
    contain many fused_moe_kernel instances with the SAME tiling geometry (e.g.
    the C500 mcProfiler records ~3× more fused_moe_kernel copies per op than
    A800's NCU → 68% of MoE sim calls were exact duplicates), and each such call
    re-ran the full grid×k-step LRU walk. Caching cuts that redundant work with
    bit-identical results. Key includes every argument (tok_per_expert as a
    tuple), so it never collides across shapes or hardware (l2_capacity /
    wave_size carry the HW dependence).
    """
    scalar_values = {
        "mt": mt,
        "nt": nt,
        "gsm": gsm,
        "k_steps": k_steps,
        "a_tile_bytes": a_tile_bytes,
        "b_tile_bytes": b_tile_bytes,
        "c_tile_bytes": c_tile_bytes,
        "wave_size": wave_size,
        "l2_capacity": l2_capacity,
        "E": E,
        "BM": BM,
        "EM": EM,
    }
    invalid_types = {
        name: value for name, value in scalar_values.items()
        if isinstance(value, bool) or not isinstance(value, (int, np.integer))
    }
    if invalid_types:
        raise TypeError(
            "MoE LRU scalar inputs must be integers: "
            f"{invalid_types}")
    invalid_values = {
        name: value for name, value in scalar_values.items() if value <= 0
    }
    if invalid_values:
        raise ValueError(
            "MoE LRU scalar inputs must be positive: "
            f"{invalid_values}")
    if not isinstance(tok_per_expert, (list, tuple, np.ndarray)):
        raise TypeError(
            "MoE tokens_per_expert must be a list, tuple, or array")
    counts = tuple(tok_per_expert)
    if len(counts) != E:
        raise ValueError(
            "MoE tokens_per_expert must contain one count per expert")
    invalid_counts = [
        value for value in counts
        if isinstance(value, bool) or not isinstance(value, (int, np.integer))
    ]
    if invalid_counts:
        raise TypeError(
            "MoE tokens_per_expert values must be integers: "
            f"{invalid_counts[:3]}")
    if any(value < 0 for value in counts):
        raise ValueError("MoE tokens_per_expert cannot contain negatives")
    populated_m_tiles = sum(
        (int(value) + BM - 1) // BM for value in counts if value > 0)
    if populated_m_tiles <= 0:
        raise ValueError("MoE routing must contain a populated M tile")
    if populated_m_tiles > mt:
        raise ValueError(
            "MoE populated M tiles exceed the launch geometry")
    if (EM + BM - 1) // BM != mt:
        raise ValueError(
            "MoE padded rows disagree with the launch M-tile geometry")
    if a_tile_bytes % BM or c_tile_bytes % BM:
        raise ValueError("MoE A/C tile bytes must be divisible by BM")

    normalized_counts = tuple(int(value) for value in counts)
    key = (mt, nt, gsm, k_steps, a_tile_bytes, b_tile_bytes, c_tile_bytes,
           wave_size, l2_capacity, normalized_counts, E, BM, EM)
    if key in _moe_full_cache:
        return _moe_full_cache[key]

    tok_arr = np.array(normalized_counts, dtype=np.int32)
    tok_arr = np.ascontiguousarray(tok_arr)
    out = np.zeros(2, dtype=np.int64)

    _lib.simulate_moe_full(
        ctypes.c_int32(mt),
        ctypes.c_int32(nt),
        ctypes.c_int32(gsm),
        ctypes.c_int32(k_steps),
        ctypes.c_int32(a_tile_bytes),
        ctypes.c_int32(b_tile_bytes),
        ctypes.c_int32(c_tile_bytes),
        ctypes.c_int32(wave_size),
        ctypes.c_int64(l2_capacity),
        tok_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(E),
        ctypes.c_int32(BM),
        ctypes.c_int32(EM),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    result = (int(out[0]), int(out[1]))
    _moe_full_cache[key] = result
    return result


def moe_operand_latency_classes(mt, nt, gsm, k_steps,
                                a_tile_bytes, b_tile_bytes,
                                tok_per_expert, E, BM,
                                l2_capacity, near_bytes, wave_size):
    """Reuse-distance latency classification for MoE A/B operand streams.

    Grouped-GEMM streams two operands per k-step with distinct physical roles:
      - A (activation): indirect-addressed (sorted_token_ids pointer-chase), a
        dependency chain whose L2 latency is EXPOSED on the critical path.
      - B (weight): prefetchable, latency HIDDEN, bandwidth-bound only.

    Runs the exact triton grouped-GEMM CTA schedule through TWO nested LRU
    caches (full = l2_capacity, near = near_bytes = L2/2) and classifies every
    operand access by where it is served:
      near : hit in near cache               -> tau_local
      far  : miss near but hit full          -> tau_remote
      miss : miss full (evicted/first touch) -> DRAM (tau_local + tau_dram_extra)

    near/full nesting is exact for LRU (near-resident ⊂ full-resident).
    near_bytes is a hardware quantity (L2 size / 2 = near-half capacity), not a
    fitted parameter. Returns {'A': (near, far, miss), 'B': (near, far, miss)}.

    Delegates to the compiled C simulator (O(1)/access, handles arbitrarily
    large grids — no size cap needed).
    """
    import numpy as _np
    tok_arr = _np.ascontiguousarray(_np.array(tok_per_expert, dtype=_np.int32))
    out = _np.zeros(6, dtype=_np.int64)
    _lib.simulate_moe_ab(
        ctypes.c_int32(mt), ctypes.c_int32(nt), ctypes.c_int32(gsm),
        ctypes.c_int32(k_steps),
        ctypes.c_int32(a_tile_bytes), ctypes.c_int32(b_tile_bytes),
        ctypes.c_int32(wave_size),
        ctypes.c_int64(int(l2_capacity)), ctypes.c_int64(int(near_bytes)),
        tok_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(E), ctypes.c_int32(BM),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    return {'A': (int(out[0]), int(out[1]), int(out[2])),
            'B': (int(out[3]), int(out[4]), int(out[5]))}


_lib.simulate_cann_grouped_moe_operands.restype = None
_lib.simulate_cann_grouped_moe_operands.argtypes = [
    ctypes.c_int32,                   # projection_k
    ctypes.c_int32,                   # projection_n
    ctypes.c_int32,                   # tile_m
    ctypes.c_int32,                   # tile_n
    ctypes.c_int32,                   # tile_k
    ctypes.c_int32,                   # element_bytes
    ctypes.c_int32,                   # worker_count
    ctypes.c_int64,                   # l2_capacity_bytes
    ctypes.c_int32,                   # B L2 cache disabled
    ctypes.POINTER(ctypes.c_int32),  # tokens_per_expert
    ctypes.c_int32,                   # expert_count
    ctypes.POINTER(ctypes.c_int64),   # residency profile occupancy
    ctypes.POINTER(ctypes.c_double),  # residency profile DRAM fraction
    ctypes.c_int32,                   # residency profile point count
    ctypes.POINTER(ctypes.c_int64),  # out[worker_count * (8 + 2 * tile_m) + 2]
]


_cann_grouped_moe_cache: dict = RuntimeCache()


def simulate_cann_grouped_moe_operands(
        projection_k: int, projection_n: int,
        tile_m: int, tile_n: int, tile_k: int, element_bytes: int,
        worker_count: int, l2_capacity: int,
        tokens_per_expert, *, b_l2_cache_disabled: bool = False,
        residency_profile: tuple[tuple[int, float], ...] = ()) -> dict:
    """Classify source-ordered CANN GroupedMatmul A/B L2 residency.

    The result reports full-tile hit/miss counts and a compact per-AIC tile
    shape histogram plus total operand bytes at the L2 ingress and downstream
    DRAM interface.  Joint hit/hit, hit/miss, miss/hit and miss/miss pair
    counts preserve the exact source order needed by pair-level services.
    Tail operands and output writes participate in replacement but are
    deliberately excluded from the exact full-tile timing counts.  When the
    version-bound source disables B's L2 allocation, B still traverses the
    ingress interface but every B descriptor is a DRAM miss and it does not
    evict A or output objects.
    """

    scalar_values = {
        "projection_k": projection_k,
        "projection_n": projection_n,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "element_bytes": element_bytes,
        "worker_count": worker_count,
        "l2_capacity": l2_capacity,
    }
    invalid_types = {
        name: value for name, value in scalar_values.items()
        if isinstance(value, bool) or not isinstance(value, (int, np.integer))
    }
    if invalid_types:
        raise TypeError(
            "CANN GroupedMatmul LRU scalar inputs must be integers: "
            f"{invalid_types}")
    invalid_values = {
        name: value for name, value in scalar_values.items() if value <= 0
    }
    if invalid_values:
        raise ValueError(
            "CANN GroupedMatmul LRU scalar inputs must be positive: "
            f"{invalid_values}")
    if not isinstance(b_l2_cache_disabled, bool):
        raise TypeError(
            "CANN GroupedMatmul B cache policy must be a boolean")
    if not isinstance(tokens_per_expert, (list, tuple, np.ndarray)):
        raise TypeError(
            "CANN GroupedMatmul tokens_per_expert must be a sequence")
    counts = tuple(tokens_per_expert)
    if not counts:
        raise ValueError(
            "CANN GroupedMatmul tokens_per_expert cannot be empty")
    if any(isinstance(value, bool)
           or not isinstance(value, (int, np.integer)) for value in counts):
        raise TypeError(
            "CANN GroupedMatmul tokens_per_expert values must be integers")
    if any(value < 0 for value in counts):
        raise ValueError(
            "CANN GroupedMatmul tokens_per_expert cannot contain negatives")
    if not any(counts):
        raise ValueError(
            "CANN GroupedMatmul routing must contain a populated expert")
    normalized_counts = tuple(int(value) for value in counts)
    key = tuple(int(value) for value in scalar_values.values()) + (
        b_l2_cache_disabled, normalized_counts, residency_profile)
    if key in _cann_grouped_moe_cache:
        return _cann_grouped_moe_cache[key]
    occupancy, fraction = _residency_profile_arrays(residency_profile)
    count_array = np.ascontiguousarray(
        np.array(normalized_counts, dtype=np.int32))
    tile_class_stride = 2 * tile_m
    legacy_values = worker_count * 8
    class_values = worker_count * tile_class_stride
    output = np.zeros(legacy_values + class_values + 2, dtype=np.int64)
    _lib.simulate_cann_grouped_moe_operands(
        ctypes.c_int32(projection_k),
        ctypes.c_int32(projection_n),
        ctypes.c_int32(tile_m),
        ctypes.c_int32(tile_n),
        ctypes.c_int32(tile_k),
        ctypes.c_int32(element_bytes),
        ctypes.c_int32(worker_count),
        ctypes.c_int64(l2_capacity),
        ctypes.c_int32(int(b_l2_cache_disabled)),
        count_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(len(normalized_counts)),
        occupancy.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        fraction.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int32(occupancy.size),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    tail_n = projection_n - ((projection_n - 1) // tile_n) * tile_n
    tile_classes_by_worker = []
    for worker in range(worker_count):
        classes = []
        base = legacy_values + worker * tile_class_stride
        for n_class, logical_n in enumerate((tile_n, tail_n)):
            if n_class == 1 and logical_n == tile_n:
                continue
            for logical_m in range(1, tile_m + 1):
                count = int(output[base + n_class * tile_m + logical_m - 1])
                if count:
                    classes.append((logical_m, logical_n, count))
        tile_classes_by_worker.append(tuple(classes))
    result = {
        "a_hits_by_worker": tuple(
            int(output[worker * 8]) for worker in range(worker_count)),
        "a_misses_by_worker": tuple(
            int(output[worker * 8 + 1]) for worker in range(worker_count)),
        "b_hits_by_worker": tuple(
            int(output[worker * 8 + 2]) for worker in range(worker_count)),
        "b_misses_by_worker": tuple(
            int(output[worker * 8 + 3]) for worker in range(worker_count)),
        "pair_hit_hit_by_worker": tuple(
            int(output[worker * 8 + 4]) for worker in range(worker_count)),
        "pair_hit_miss_by_worker": tuple(
            int(output[worker * 8 + 5]) for worker in range(worker_count)),
        "pair_miss_hit_by_worker": tuple(
            int(output[worker * 8 + 6]) for worker in range(worker_count)),
        "pair_miss_miss_by_worker": tuple(
            int(output[worker * 8 + 7]) for worker in range(worker_count)),
        "tile_classes_by_worker": tuple(tile_classes_by_worker),
        "l2_read_bytes": int(output[legacy_values + class_values]),
        "dram_read_bytes": int(output[legacy_values + class_values + 1]),
    }
    _cann_grouped_moe_cache[key] = result
    return result


_lib.simulate_gemm_tile_program.restype = None
_lib.simulate_gemm_tile_program.argtypes = [
    ctypes.POINTER(ctypes.c_int32),   # schedule [n_ctas, 3] = (m, n, k slice)
    ctypes.c_int32,                   # n_ctas
    ctypes.c_int32,                   # m extent
    ctypes.c_int32,                   # n extent
    ctypes.c_int32,                   # k extent
    ctypes.c_int32,                   # tile m
    ctypes.c_int32,                   # tile n
    ctypes.c_int32,                   # tile k
    ctypes.c_int32,                   # A element bytes
    ctypes.c_int32,                   # B element bytes
    ctypes.c_int32,                   # C element bytes
    ctypes.c_int32,                   # K split
    ctypes.c_int32,                   # uniform rounded K-slice boundaries
    ctypes.c_int32,                   # accumulate in place
    ctypes.c_int32,                   # resident CTA wave width
    ctypes.c_int64,                   # replacement capacity
    ctypes.POINTER(ctypes.c_int64),   # residency profile occupancy
    ctypes.POINTER(ctypes.c_double),  # residency profile DRAM fraction
    ctypes.c_int32,                   # residency profile point count
    ctypes.POINTER(ctypes.c_int64),   # out[6]
]


@dataclasses.dataclass(frozen=True)
class TileProgramTraffic:
    """Per-operand traffic from one dense tile program replay."""

    a_l2_bytes: int
    a_dram_bytes: int
    b_l2_bytes: int
    b_dram_bytes: int
    output_l2_bytes: int
    output_dram_bytes: int

    @property
    def l2_bytes(self) -> int:
        return self.a_l2_bytes + self.b_l2_bytes + self.output_l2_bytes

    @property
    def dram_bytes(self) -> int:
        return self.a_dram_bytes + self.b_dram_bytes + self.output_dram_bytes


def simulate_gemm_tile_program(
        schedule_mnz, *, m_extent: int, n_extent: int, k_extent: int,
        tile_m: int, tile_n: int, tile_k: int,
        a_element_bytes: int, b_element_bytes: int, c_element_bytes: int,
        split_k: int, accumulate_in_place: bool, wave_size: int,
        capacity_bytes: int,
        residency_profile: tuple[tuple[int, float], ...] = (),
        uniform_split_k_tiles: bool = False,
        ) -> TileProgramTraffic:
    """Price one dense tile program through the shared L2-to-DRAM miss core.

    ``schedule_mnz`` is the CTA-to-tile schedule the hardware launch package
    derives from its own raster facts, in dispatch order.  Every remaining
    input is either a retained launch fact or a hardware numeric parameter;
    the solver behind this call is shared with every other family.
    """

    schedule = np.ascontiguousarray(schedule_mnz, dtype=np.int32)
    if schedule.ndim != 2 or schedule.shape[1] != 3 or schedule.shape[0] <= 0:
        raise TypeError("tile program schedule must have shape [N, 3]")
    values = {
        "m extent": m_extent,
        "n extent": n_extent,
        "k extent": k_extent,
        "tile m": tile_m,
        "tile n": tile_n,
        "tile k": tile_k,
        "A element bytes": a_element_bytes,
        "B element bytes": b_element_bytes,
        "C element bytes": c_element_bytes,
        "K split": split_k,
        "wave width": wave_size,
        "capacity": capacity_bytes,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"tile program {name} must be an integer")
        if value <= 0:
            raise ValueError(f"tile program {name} must be positive")
    if not isinstance(accumulate_in_place, bool):
        raise TypeError("tile program accumulation mode must be a boolean")
    if not isinstance(uniform_split_k_tiles, bool):
        raise TypeError("tile program uniform K slices must be a boolean")
    occupancy, fraction = _residency_profile_arrays(residency_profile)
    output = np.zeros(6, dtype=np.int64)
    _lib.simulate_gemm_tile_program(
        schedule.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(schedule.shape[0]),
        ctypes.c_int32(int(m_extent)), ctypes.c_int32(int(n_extent)),
        ctypes.c_int32(int(k_extent)),
        ctypes.c_int32(int(tile_m)), ctypes.c_int32(int(tile_n)),
        ctypes.c_int32(int(tile_k)),
        ctypes.c_int32(int(a_element_bytes)),
        ctypes.c_int32(int(b_element_bytes)),
        ctypes.c_int32(int(c_element_bytes)),
        ctypes.c_int32(int(split_k)),
        ctypes.c_int32(int(uniform_split_k_tiles)),
        ctypes.c_int32(1 if accumulate_in_place else 0),
        ctypes.c_int32(int(wave_size)),
        ctypes.c_int64(int(capacity_bytes)),
        occupancy.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        fraction.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int32(occupancy.size),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    return TileProgramTraffic(*(int(value) for value in output))


def _residency_profile_arrays(residency_profile):
    """Validate one hardware residency profile and return its C arrays."""

    if not isinstance(residency_profile, tuple):
        raise TypeError("residency profile must be a tuple of points")
    occupancy = []
    fraction = []
    for point in residency_profile:
        if not isinstance(point, tuple) or len(point) != 2:
            raise TypeError(
                "residency profile points must be (occupancy, fraction)")
        bytes_value, served = point
        if (isinstance(bytes_value, bool)
                or not isinstance(bytes_value, (int, np.integer))):
            raise TypeError("residency profile occupancy must be an integer")
        if bytes_value <= 0:
            raise ValueError("residency profile occupancy must be positive")
        if isinstance(served, bool) or not isinstance(
                served, (int, float, np.floating)):
            raise TypeError("residency profile fraction must be a number")
        if not 0.0 <= float(served) <= 1.0:
            raise ValueError("residency profile fraction must be in [0, 1]")
        if occupancy and bytes_value <= occupancy[-1]:
            raise ValueError("residency profile occupancy must ascend")
        if fraction and float(served) < fraction[-1]:
            raise ValueError(
                "residency profile fraction must be non-decreasing")
        occupancy.append(int(bytes_value))
        fraction.append(float(served))
    return (
        np.ascontiguousarray(occupancy, dtype=np.int64),
        np.ascontiguousarray(fraction, dtype=np.float64),
    )


_lib.simulate_moe_tile_program.restype = None
_lib.simulate_moe_tile_program.argtypes = [
    ctypes.c_int32,                   # M tiles
    ctypes.c_int32,                   # N tiles
    ctypes.c_int32,                   # group size over M
    ctypes.c_int32,                   # K extent
    ctypes.c_int32,                   # K block
    ctypes.c_int32,                   # N extent
    ctypes.c_int32,                   # N block
    ctypes.c_int32,                   # element bytes
    ctypes.c_int32,                   # C bytes per routed row
    ctypes.c_int32,                   # M block rows
    ctypes.c_int32,                   # resident CTA wave width
    ctypes.POINTER(ctypes.c_int32),   # tokens per expert
    ctypes.c_int32,                   # expert count
    ctypes.c_int64,                   # replacement capacity
    ctypes.POINTER(ctypes.c_int64),   # residency profile occupancy
    ctypes.POINTER(ctypes.c_double),  # residency profile DRAM fraction
    ctypes.c_int32,                   # residency profile point count
    ctypes.POINTER(ctypes.c_int64),   # out[6]
]

_moe_tile_program_cache: dict = RuntimeCache()


def simulate_moe_tile_program(
        *, m_tiles: int, n_tiles: int, group_size_m: int, k_extent: int,
        block_k: int, n_extent: int, block_n: int, element_bytes: int,
        c_row_bytes: int,
        block_m: int, wave_size: int, tokens_per_expert,
        capacity_bytes: int,
        residency_profile: tuple[tuple[int, float], ...] = (),
        ) -> TileProgramTraffic:
    """Price one grouped MoE tile program through the shared miss core.

    Geometry and routing are launch facts; capacity and replacement grading
    are hardware numeric parameters.  The solver is the one every other
    family uses.
    """

    values = {
        "M tiles": m_tiles,
        "N tiles": n_tiles,
        "group size": group_size_m,
        "K extent": k_extent,
        "K block": block_k,
        "N extent": n_extent,
        "N block": block_n,
        "element bytes": element_bytes,
        "C row bytes": c_row_bytes,
        "M block": block_m,
        "wave width": wave_size,
        "capacity": capacity_bytes,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"MoE tile program {name} must be an integer")
        if value <= 0:
            raise ValueError(f"MoE tile program {name} must be positive")
    if not isinstance(tokens_per_expert, (list, tuple, np.ndarray)):
        raise TypeError("MoE tokens_per_expert must be a sequence")
    counts = tuple(int(value) for value in tokens_per_expert)
    if not counts:
        raise ValueError("MoE tokens_per_expert cannot be empty")
    if any(value < 0 for value in counts):
        raise ValueError("MoE tokens_per_expert cannot contain negatives")
    if not any(counts):
        raise ValueError("MoE routing must contain a populated expert")
    # An expert with no routed token contributes no M tile, so it produces
    # neither an access nor a resident object: the replay walks only the
    # populated tiles, and an expert index reaches the stream solely as the
    # distinct label separating one expert's weight objects from another's.
    # Dropping the empty experts therefore leaves the access sequence, the
    # object identities, and the resident-object bound unchanged, while
    # collapsing every routing that differs only in which experts stayed
    # empty onto one physical program.
    populated_counts = tuple(value for value in counts if value)
    key = (tuple(int(value) for value in values.values()), populated_counts,
           residency_profile)
    if key in _moe_tile_program_cache:
        return _moe_tile_program_cache[key]
    counts = populated_counts
    occupancy, fraction = _residency_profile_arrays(residency_profile)
    count_array = np.ascontiguousarray(np.array(counts, dtype=np.int32))
    output = np.zeros(6, dtype=np.int64)
    _lib.simulate_moe_tile_program(
        ctypes.c_int32(int(m_tiles)), ctypes.c_int32(int(n_tiles)),
        ctypes.c_int32(int(group_size_m)), ctypes.c_int32(int(k_extent)),
        ctypes.c_int32(int(block_k)),
        ctypes.c_int32(int(n_extent)), ctypes.c_int32(int(block_n)),
        ctypes.c_int32(int(element_bytes)),
        ctypes.c_int32(int(c_row_bytes)), ctypes.c_int32(int(block_m)),
        ctypes.c_int32(int(wave_size)),
        count_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(len(counts)),
        ctypes.c_int64(int(capacity_bytes)),
        occupancy.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        fraction.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int32(occupancy.size),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    result = TileProgramTraffic(*(int(value) for value in output))
    _moe_tile_program_cache[key] = result
    return result


def simulate_attention_prefill_residency(
        *, batch: int, query_heads: int, kv_heads: int, head_dim: int,
        element_bytes: int, query_tile: int, kv_panel: int,
        query_lengths, kv_lengths, wave_size: int, capacity_bytes: int,
        query_tile_is_slowest: bool = True,
) -> tuple[int, int]:
    """Replay one causal attention prefill program through the shared core.

    The launch retains the dispatch order its grid states -- ``(query_heads,
    batch, query_tiles)`` with x fastest, so the query tile is the *slowest*
    dimension -- and causal masking states which K/V panels each query tile
    reads.  That is the whole access sequence; the replacement decision
    behind it is the same core the dense and grouped families use.

    The order is what makes this program different from a dense GEMM, and the
    two MACA prefill programs order it differently.  ``query_tile_is_slowest``
    selects which: the grid the launch states is ``(query_heads, batch,
    query_tiles)`` for one and ``(query_heads, query_tiles, batch)`` for the
    other, x fastest in both.  Two query tiles of one request are therefore
    ``query_heads * batch`` programs apart in the first and only
    ``query_heads`` apart in the second, and every intervening program's K/V
    panels sit between one tile's reference to a panel and the next tile's.
    Whether the panel survives that distance is a capacity question, and it is
    why the same geometry retains under one order and not the other.

    Returns the read path's ``(l2_request_bytes, dram_bytes)``; the caller
    owns the write side.
    """

    values = {
        "batch": batch, "query heads": query_heads, "kv heads": kv_heads,
        "head dimension": head_dim, "element bytes": element_bytes,
        "query tile": query_tile, "K/V panel": kv_panel,
        "wave width": wave_size, "capacity": capacity_bytes,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"attention prefill {name} must be an integer")
        if value <= 0:
            raise ValueError(f"attention prefill {name} must be positive")
    if query_heads % kv_heads != 0:
        raise ValueError(
            "attention prefill query heads must be a multiple of kv heads")
    q_lengths = tuple(int(value) for value in query_lengths)
    k_lengths = tuple(int(value) for value in kv_lengths)
    if len(q_lengths) != batch or len(k_lengths) != batch:
        raise ValueError(
            "attention prefill request lengths must cover every batch entry")
    if any(value <= 0 for value in q_lengths + k_lengths):
        raise ValueError("attention prefill request lengths must be positive")
    if any(q > k for q, k in zip(q_lengths, k_lengths, strict=True)):
        raise ValueError(
            "attention prefill query length cannot exceed its K/V length")

    # The grid's tile dimension covers the longest request; a program past its
    # own request's tile count returns before issuing any load.
    tiles_per_request = tuple(
        -(-length // query_tile) for length in q_lengths)
    query_tiles = max(tiles_per_request)
    group = query_heads // kv_heads
    panel_bytes = 2 * kv_panel * head_dim * element_bytes

    def panels_read(entry: int, tile: int) -> int:
        """Bottom-right causal reach of one tile, in K/V panels."""
        if tile >= tiles_per_request[entry]:
            return 0
        rows = min((tile + 1) * query_tile, q_lengths[entry])
        attended = k_lengths[entry] - q_lengths[entry] + rows
        if attended <= 0:
            return 0
        return min(-(-attended // kv_panel),
                   -(-k_lengths[entry] // kv_panel))

    keys: list[int] = []
    sizes: list[int] = []
    if query_tile_is_slowest:
        programs = [
            (tile, entry, head)
            for tile in range(query_tiles)
            for entry in range(batch)
            for head in range(query_heads)
        ]
    else:
        programs = [
            (tile, entry, head)
            for entry in range(batch)
            for tile in range(query_tiles)
            for head in range(query_heads)
        ]
    width = min(len(programs), wave_size)
    for start in range(0, len(programs), width):
        wave = programs[start:start + width]
        for tile, entry, head in wave:
            if tile >= tiles_per_request[entry]:
                continue
            rows = (min((tile + 1) * query_tile, q_lengths[entry])
                    - tile * query_tile)
            keys.append((1 << 62) | (entry << 40) | (head << 24) | tile)
            sizes.append(rows * head_dim * element_bytes)
        # A wave's programs advance their K/V loop together, so a panel that
        # several of them read is fetched once for the wave.  Causal masking
        # gives the programs different trip counts; the longest one sets the
        # wave's step count and shorter ones simply stop contributing.
        reach = [panels_read(entry, tile) for tile, entry, _head in wave]
        for step in range(max(reach)):
            for (_tile, entry, head), stop in zip(wave, reach, strict=True):
                if step >= stop:
                    continue
                keys.append((entry << 40) | ((head // group) << 24) | step)
                sizes.append(panel_bytes)
    requested, missed = simulate_lru_trace(
        np.array(keys, dtype=np.uint64),
        np.array(sizes, dtype=np.int32),
        capacity_bytes)
    return int(requested), int(missed)
