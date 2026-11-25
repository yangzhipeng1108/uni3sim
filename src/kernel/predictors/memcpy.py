"""Complete-spec device and host memory-copy prediction."""
import math

from src.kernel.kernel_base import KernelPredictor
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.memory_access_builder import transfer_memory_access_program
from src.kernel.memory_path_solver import logical_byte_copy_payload
from src.kernel.predictors.common import prediction_from_ns


_NATIVE_D2D_FLASH_ATTENTION_COPY = "native_d2d_flash_attention_copy"
_NATIVE_D2D_MEDIUM_FIELDS = (
    "memcpy_dtod_native_flash_attention_copy_medium_min_bytes",
    "memcpy_dtod_native_flash_attention_copy_medium_max_bytes",
    "memcpy_dtod_native_flash_attention_copy_medium_floor_ns",
)


def _native_d2d_flash_attention_copy_floor_ns(
        kernel: KernelSpec, hardware, generic_floor_ns: float,
        payload_bytes: int) -> float:
    """Return the bounded C500 native-copy task floor when source-selected.

    The class is restricted to the source-declared native FlashAttention copy
    ABI.  Its medium-size plateau comes from an independent contiguous BF16
    D2D microbenchmark, after directional payload service is removed; it is
    not inferred from a serving or kernel workload duration.
    """

    if kernel.transfer.algorithm != _NATIVE_D2D_FLASH_ATTENTION_COPY:
        return generic_floor_ns
    subsystem = hardware.memory_subsystem
    missing = tuple(name for name in _NATIVE_D2D_MEDIUM_FIELDS
                    if subsystem[name] is None)
    if missing:
        raise ValueError(
            "native FlashAttention D2D copy floor is incomplete: "
            f"{list(missing)!r}")
    minimum_bytes, maximum_bytes, medium_floor_ns = tuple(
        subsystem[name] for name in _NATIVE_D2D_MEDIUM_FIELDS)
    if (isinstance(minimum_bytes, bool)
            or not isinstance(minimum_bytes, int)
            or minimum_bytes <= 0):
        raise TypeError("native FlashAttention D2D minimum bytes is invalid")
    if (isinstance(maximum_bytes, bool)
            or not isinstance(maximum_bytes, int)
            or maximum_bytes < minimum_bytes):
        raise ValueError("native FlashAttention D2D byte range is invalid")
    if (isinstance(medium_floor_ns, bool)
            or not isinstance(medium_floor_ns, (int, float))
            or not math.isfinite(float(medium_floor_ns))
            or float(medium_floor_ns) <= 0):
        raise ValueError("native FlashAttention D2D medium floor is invalid")
    if minimum_bytes <= payload_bytes <= maximum_bytes:
        return max(generic_floor_ns, float(medium_floor_ns))
    return generic_floor_ns


def _stream_copy_key(transfer) -> str:
    direction = transfer.direction.lower()
    if direction in ("d2d", "dtod", "device_to_device"):
        return "d2d"
    host = {"stream_copy_pinned": "pinned",
            "stream_copy_pageable": "pageable"}[transfer.algorithm]
    if direction in ("h2d", "host_to_device"):
        return f"h2d_{host}"
    if direction in ("d2h", "device_to_host") and host == "pinned":
        return "d2h_pinned"
    raise ValueError(
        f"no measured stream copy for {direction!r} from {host} memory")


def _cost(kernel: KernelSpec, hardware, include_latency: bool) -> float:
    transfer = kernel.transfer
    payload_bytes = logical_byte_copy_payload(
        transfer_memory_access_program(kernel, hardware))
    if transfer.algorithm == "fpga_staging":
        if include_latency:
            raise NotImplementedError("FPGA staging controller/DMA startup is unbound; only explicit DDR bandwidth work is available")
        seconds = payload_bytes / hardware.resource_by_role("ddr").capacity
        if transfer.direction == "hbm_to_ddr":
            seconds = max(seconds, payload_bytes / hardware.resource_by_role("dram").capacity)
            seconds += hardware.fpga_execution["hbm_latency_ns"] / 1e9
        return seconds * 1e9
    if transfer.algorithm.startswith("stream_copy"):
        # Measured cost of one eager cudaMemcpyAsync between kernels on the
        # same stream (latency and payload together).
        return hardware.stream_memcpy_service_ns(
            _stream_copy_key(transfer), payload_bytes)
    direction = transfer.direction.lower()
    if direction in ("d2d", "dtod", "device_to_device"):
        read_bandwidth = hardware.dram_capacity_for_access(
            "read",
            active_units=hardware.num_compute_units,
            sm_freq_hz=hardware.nominal_sm_freq_hz,
            dram_freq_hz=hardware.nominal_dram_freq_hz,
        )
        write_bandwidth = hardware.dram_capacity_for_access(
            "write",
            active_units=hardware.num_compute_units,
            sm_freq_hz=hardware.nominal_sm_freq_hz,
            dram_freq_hz=hardware.nominal_dram_freq_hz,
        )
        transfer_time_ns = payload_bytes * (
            1.0 / read_bandwidth + 1.0 / write_bandwidth) * 1e9
        if include_latency:
            configured_latency = hardware.memory_subsystem[
                "memcpy_dtod_floor_ns"]
            if configured_latency is None:
                raise KeyError(
                    "MEMCPY requires memory_subsystem.memcpy_dtod_floor_ns")
            latency = _native_d2d_flash_attention_copy_floor_ns(
                kernel, hardware, float(configured_latency), payload_bytes)
        else:
            latency = 0.0
    elif direction in ("h2d", "d2h", "host_to_device", "device_to_host"):
        bandwidth = hardware.interconnect["pcie_bw_bytes_s"]
        if bandwidth is None:
            raise KeyError("MEMCPY requires interconnect.pcie_bw_bytes_s")
        transfer_time_ns = payload_bytes / bandwidth * 1e9
        if include_latency:
            latency = hardware.interconnect["pcie_latency_ns"]
            if latency is None:
                raise KeyError("MEMCPY requires interconnect.pcie_latency_ns")
        else:
            latency = 0.0
    else:
        raise ValueError(f"unknown MEMCPY direction: {transfer.direction!r}")
    return transfer_time_ns + float(latency)


class MemcpyPredictor(KernelPredictor):
    kind = KernelKind.MEMCPY

    def midlayer(self, kernel: KernelSpec, hardware):
        return prediction_from_ns(_cost(kernel, hardware, True), "memcpy")

    def roofline(self, kernel: KernelSpec, hardware):
        return prediction_from_ns(_cost(kernel, hardware, False), "memcpy")
