"""Zero-fit analytical communication Roofline and backend dispatch."""

from collections import Counter
import math

from src.kernel.kernel_base import KernelPredictor
from src.kernel.kernel_spec import KernelKind, KernelSpec, KernelPrediction
from src.kernel.predictors.common import prediction_from_ns
from src.kernel.predictors.protobridge.adapter import predict_protobridge


BILATERAL_BANDWIDTH_ROLES = (
    "communication_sender_endpoint", "communication_receiver_endpoint", "communication_fabric")
NODE_BANDWIDTH_ROLES = ("communication_sender_endpoint", "communication_receiver_endpoint",
    "communication_node_tx", "communication_node_rx")


def direct_switch_collective_logical_work(kernel: KernelSpec):
    """Endpoint byte loads of an explicitly routed, nonblocking switch call.

    Two GPUs sharing one NIC share its physical TX/RX capacity. Even traffic
    between those GPUs follows the declared switch program; no unbound fast
    local path is inserted. Full-duplex TX and RX are independent services.
    """
    transfer = kernel.transfer
    if kernel.kind is not KernelKind.COMMUNICATION or transfer is None or not transfer.traffic_matrix_bytes:
        raise ValueError("direct-switch work requires its explicit traffic-matrix collective")
    tx = Counter({nic: 0 for nic in transfer.participant_nic_ids})
    rx = Counter({nic: 0 for nic in transfer.participant_nic_ids})
    matrix = transfer.traffic_matrix_bytes
    for source, nic in enumerate(transfer.participant_nic_ids):
        tx[nic] += sum(matrix[source])
        rx[nic] += sum(row[source] for row in matrix)
    return {"payload_bytes": transfer.bytes, "nic_tx_bytes": dict(tx), "nic_rx_bytes": dict(rx),
        "max_nic_load_bytes": max((*tx.values(), *rx.values())),
        "limitations": ("declared_direct_switch_traffic_not_a_discovered_collective_library_algorithm",
            "endpoint_capacity_lower_bound_nonblocking_switch_no_protocol_efficiency_inferred",
            "one_collective_startup_separate_from_bandwidth_service_not_per_rank_or_packet")}


class CommunicationPredictor(KernelPredictor):
    kind = KernelKind.COMMUNICATION

    def _analytical_roofline(self, kernel: KernelSpec, hardware):
        transfer = kernel.transfer
        if transfer.traffic_matrix_bytes:
            if not hardware.roofline_only:
                raise NotImplementedError("direct-switch traffic requires its declared aggregate hardware profile")
            work = direct_switch_collective_logical_work(kernel)
            startup_ns = hardware.interconnect["roce_collective_startup_ns"]
            if (isinstance(startup_ns, bool) or not isinstance(startup_ns, (int, float))
                    or not math.isfinite(startup_ns) or startup_ns < 0):
                raise ValueError("direct-switch collective requires an explicit nonnegative startup latency")
            bandwidth = hardware.analytical_link_bandwidth_bytes_s(transfer.link)
            return prediction_from_ns(startup_ns + work["max_nic_load_bytes"] / bandwidth * 1e9,
                "direct_switch_collective")
        from src.kernel.memory_communication import has_communication_source
        if has_communication_source(kernel):
            from src.kernel.memory_access_builder import communication_memory_access_program
            from src.kernel.memory_path_solver import collective_source_work
            payload, ranks, operation = collective_source_work(
                communication_memory_access_program(kernel, hardware))
            if operation != transfer.operation:
                raise ValueError("collective source operation does not match the call")
        else:
            # Fused residual/RMSNorm has extra operands and remains outside
            # this source domain; preserve its existing support.
            payload, ranks = transfer.bytes, len(transfer.participants)
        bandwidth = hardware.analytical_link_bandwidth_bytes_s(transfer.link)
        if transfer.destination_hardware_id is not None:
            pcie = hardware.interconnect["pcie_bw_bytes_s"]
            if pcie is None:
                raise ValueError("bilateral transport requires sender PCIe bandwidth")
            # Streaming GPUDirect path: HBM read -> PCIe -> network ->
            # receiver PCIe -> HBM write. Stages pipeline; do not add the
            # same payload's bandwidth service six times. Startup is unbound,
            # so this backend deliberately reports a bandwidth lower bound.
            sender = min(pcie, hardware.resource_by_role("dram").capacity)
            # Preserve each resource's byte service, not just the bottleneck
            # duration. Serving can share the aggregate fabric while every
            # physical GPU retains its independent endpoint constraint.
            if transfer.participant_nic_ids:
                if transfer.destination_hardware_id != hardware.hardware_id:
                    raise NotImplementedError("node-port transport needs a separate receiver-port bound for heterogeneous hardware")
                services = dict(zip(NODE_BANDWIDTH_ROLES, (payload / sender,
                    payload / transfer.destination_bandwidth_bytes_s,
                    payload / bandwidth, payload / bandwidth), strict=True))
            else:
                services = dict(zip(BILATERAL_BANDWIDTH_ROLES, (
                    payload / sender, payload / transfer.destination_bandwidth_bytes_s,
                    payload / bandwidth), strict=True))
            duration = max(services.values())
            return KernelPrediction(duration, {"communication": duration}, 0.0,
                "communication", services)
        if transfer.operation == "all_reduce":
            network_bytes = (
                2.0 * (ranks - 1) / ranks * payload)
        elif transfer.operation == "all_gather":
            network_bytes = (ranks - 1) * payload
        elif transfer.operation in ("all_gatherv", "reduce_scatterv"):
            # Explicit direct pairwise broadcast/reduce scenario. Each root
            # sends its segment to N-1 peers (or receives N-1 contributions).
            # Full-duplex endpoint load is the maximum TX or RX byte demand;
            # no equal-size padding or all-reduce/broadcast phase is invented.
            sizes = transfer.partition_bytes
            network_bytes = max((ranks - 1) * max(sizes), payload - min(sizes))
        elif transfer.operation == "send_recv":
            network_bytes = payload
        elif transfer.operation == "all_to_all":
            network_bytes = (
                (ranks - 1) / ranks * payload)
        else:
            raise ValueError(
                f"unknown COMMUNICATION operation: {transfer.operation!r}")
        data_ns = network_bytes / bandwidth * 1e9
        if transfer.link == "roce":
            # A full-duplex NIC has independent TX/RX capacity. Each ring
            # participant sends and receives the same logical byte volume;
            # max NIC load, rather than rank count or switch port count,
            # sets the bandwidth term. IDs describe physical placement.
            if not hardware.roofline_only:
                raise NotImplementedError("RoCE requires a declared aggregate NIC model")
            loads = Counter(transfer.participant_nic_ids)
            if max(loads.values()) > hardware.interconnect["roce_devices_per_nic"]:
                raise ValueError("RoCE NIC placement exceeds declared devices per NIC")
            nic_by_rank = dict(zip(transfer.participants, transfer.participant_nic_ids))
            if transfer.operation in ("all_reduce", "all_gather"):
                ring = transfer.ring_order
                if any(nic_by_rank[rank] == nic_by_rank[ring[(i + 1) % ranks]]
                       for i, rank in enumerate(ring)):
                    raise NotImplementedError("RoCE ring includes an unknown same-NIC local transfer path")
            elif transfer.operation in ("send_recv", "all_to_all"):
                if len(loads) != ranks:
                    raise NotImplementedError("RoCE transfer includes an unknown same-NIC local transfer path")
            else:
                raise AssertionError("validated RoCE operation disappeared")
            network_ns = data_ns * max(loads.values())
            signal_ns = hardware.interconnect["roce_signal_latency_ns"]
            if signal_ns is not None:
                # One-way propagation per dependent algorithm round. Startup
                # and staging DDR are distinct services, not included here.
                if transfer.operation == "all_reduce":
                    rounds = 2 * (ranks - 1)
                elif transfer.operation in ("all_gather", "all_to_all"):
                    rounds = ranks - 1
                else:
                    rounds = 1
                network_ns += rounds * signal_ns
            return prediction_from_ns(network_ns, "communication")
        if (transfer.implementation
                == "sglang_flashinfer_allreduce_residual_rmsnorm"):
            # One fused kernel: all-reduce, residual add and RMSNorm.
            service_ns = hardware.sglang_flashinfer_allreduce_fusion_ns(
                ranks, payload)
            return prediction_from_ns(
                max(service_ns, data_ns), "communication")
        if transfer.implementation == "sglang_custom_all_reduce_v2":
            # SGLang CustomAllReduceV2 below its max size; above it SGLang
            # falls back to NCCL, priced by the bandwidth term.
            service_ns = hardware.sglang_custom_all_reduce_service_ns(
                ranks, payload)
            if service_ns is not None:
                return prediction_from_ns(
                    max(service_ns, data_ns), "communication")
            return prediction_from_ns(data_ns, "communication")
        if transfer.operation == "all_reduce" and transfer.link == "nvlink":
            # Pinned vLLM runs small NVLink all-reduces through its IPC
            # custom kernel; its measured service includes the cross-rank
            # synchronisation that a bandwidth-only term omits.
            service_ns = hardware.custom_all_reduce_service_ns(
                ranks, payload)
            if service_ns is not None:
                return prediction_from_ns(
                    max(service_ns, data_ns), "communication")
        return prediction_from_ns(data_ns, "communication")

    def _predict_selected_backend(self, kernel: KernelSpec, hardware):
        backend = kernel.transfer.backend
        if backend == "analytical_roofline_v1":
            return self._analytical_roofline(kernel, hardware)
        elif backend == "protobridge":
            return predict_protobridge(kernel, hardware)
        else:
            raise ValueError(
                f"unknown COMMUNICATION backend: {backend!r}")

    def midlayer(self, kernel: KernelSpec, hardware):
        return self._predict_selected_backend(kernel, hardware)

    def roofline(self, kernel: KernelSpec, hardware):
        return self._predict_selected_backend(kernel, hardware)
