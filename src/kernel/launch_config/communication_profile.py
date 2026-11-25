"""Hardware-owned inputs for versioned communication profiles."""

from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from src.kernel.hardware_ids import canonical_hardware_id
from src.kernel.launch_config.ascend_910b import (
    DEVELOPMENT_PROTOBRIDGE_INPUTS as ASCEND_DEVELOPMENT_INPUTS,
)
from src.kernel.launch_config.metax_c500 import (
    DEVELOPMENT_PROTOBRIDGE_INPUTS as METAX_DEVELOPMENT_INPUTS,
)
from src.kernel.launch_config.nvidia_a800 import (
    DEVELOPMENT_PROTOBRIDGE_INPUTS as NVIDIA_DEVELOPMENT_INPUTS,
)
from src.kernel.launch_config.nvidia_h200.collective_rules import (
    RULE_VERSION as H200_COLLECTIVE_RULE_VERSION, OBSERVED_TOPOLOGY as H200_OBSERVED_TOPOLOGY,
)


# H200 scale-up (NVLink) inputs measured on a two-rank H200 pair: NCCL
# 2.27.3 Ring forced to LL / LL128 / Simple, every all-reduce a CUDA-graph
# node (microbench/nvidia_h200/communication/nccl_protocol_sweep/).  The
# derivation and its N=2 check table are in
# microbench/nvidia_h200/references/nccl_ring_protocol_ws2_protobridge_inputs_sm90.json
# (nccl_ring_protocol_protobridge_inputs.py).
# * per_link_bytes_per_us / num_lanes: H200 NVLink4 spec, 18 x 25 GB/s
#   one-way (configs/hardware/nvidia_h200.yaml, 450 GB/s one-way).
# * Two-rank Simple all-reduce ring bandwidth: measured large-message slope.
# * Startups: solved so the two-rank all-reduce reproduces the H200
#   measurement at the protocol's smallest selected size (LL 256 B, Simple
#   2 MiB); LL128 is never selected at N=2 and is solved at its smallest
#   forced size (256 B).
# Still A800 values, unmeasured on H200: the per-phase fixed costs (a
# two-rank scan identifies only startup + 2 * phase_fixed, so they matter
# only for N > 2), t_min_switch_us, spray_chunk_bytes and wire_efficiency. The
# selector now admits only exact automatic two-rank observations from the
# same raw campaign and pair identity; other messages and rank counts are
# unbound. The campaign topology records NV18 between devices4/5; selector
# topology names that observed pair, not all single-node NVLink pairs. The 4- and
# 8-rank Simple bandwidths are not measured and are omitted, so those ranks
# use the spec-derived default ring bandwidth.
_H200_SCALE_UP_INPUTS: Mapping[str, object] = MappingProxyType({
    **dict(NVIDIA_DEVELOPMENT_INPUTS),
    "hardware_id": "nvidia_h200",
    "selector_rule_version": H200_COLLECTIVE_RULE_VERSION,
    "selector_topology": H200_OBSERVED_TOPOLOGY,
    "per_link_bytes_per_us": 25_000.0,
    "num_lanes": 18,
    "small_startup_us": 0.4399832935621344,
    "ll128_startup_us": 11.350483296855288,
    "simple_startup_us": 7.944375612765548,
    "small_phase_fixed_us": 2.59,
    "ll128_phase_fixed_us": 0.659,
    "simple_phase_fixed_us": 6.321,
    "simple_allreduce_ring_bw_by_participant": ((2, 220733.0447194697),),
    "provenance_note": (
        "H200 two-rank NCCL 2.27.3 Ring scan (startups and the 2-rank Simple "
        "ring bandwidth) plus NVLink4 spec lanes; algorithm/protocol selection "
        "uses exact automatic two-rank observations only. phase_fixed, "
        "t_min_switch_us, spray_chunk_bytes, wire_efficiency and "
        "4/8-rank Simple bandwidths remain A800-projected or unmeasured on "
        "H200; "),
})

# H200 scale-out point-to-point timing is projected from the linked
# upstream ProtoBridge make_h200_roce_theory preset
# (third_party/protobridge submodule, surrogate/theory/whitebox_surrogate_v2.py), a
# spec-derived and explicitly UNCALIBRATED 400GbE RoCEv2 model anchored to
# ns-3 rather than measured hardware: 50 GB/s per-link wire rate, 8 NICs,
# 0.4 ring efficiency (ring_bw = 0.4 * 8 * 50000 = 160000 B/us), ~1 us
# per-hop propagation, ~1.5 us switch pipeline, ~40 us RDMA startup, and
# 0.95 wire efficiency for Ethernet/IP/UDP/IB headers.  It registers only a
# send_recv profile (empty collective algorithm tuples) so collective
# resolution for nvidia_h200 stays bound to the scale-up record above.
# Development-only until a captured scale-out profile exists.
_H200_SCALE_OUT_PROJECTED_INPUTS: Mapping[str, object] = MappingProxyType({
    "hardware_id": "nvidia_h200",
    "communication_library": "NCCL",
    "library_version": "2.27.3",
    "selector_topology": "multi_node_rocev2_400gbe",
    "selector_rule_version": (
        "protobridge-h200-rocev2-theory-uncalibrated-20260923-v1"),
    "all_reduce_algorithms": (),
    "all_gather_algorithms": (),
    "default_all_reduce_algorithm": "ring",
    "default_all_gather_algorithm": "ring",
    "link": "infiniband",
    "per_link_bytes_per_us": 50_000.0,
    "num_lanes": 8,
    "link_latency_us": 1.0,
    "small_startup_us": 40.0,
    "ll128_startup_us": 40.0,
    "simple_startup_us": 40.0,
    "small_phase_fixed_us": 1.0,
    "ll128_phase_fixed_us": 1.0,
    "simple_phase_fixed_us": 1.0,
    "simple_allreduce_ring_bw_by_participant": (),
    "spray_chunk_bytes": 131_072,
    "t_min_switch_us": 1.5,
    "wire_efficiency": 0.95,
    "provenance_note": (
        "scale-out projection anchored to the vendored upstream ProtoBridge "
        "make_h200_roce_theory preset (400GbE RoCEv2, explicitly "
        "UNCALIBRATED, ns-3-anchored, not measured hardware); "),
    "bisection_bw_bytes_per_us": None,
    "hop_count": 1.0,
})

_DEVELOPMENT_INPUTS: Tuple[Mapping[str, object], ...] = tuple(
    MappingProxyType(dict(inputs))
    for inputs in (
        NVIDIA_DEVELOPMENT_INPUTS,
        METAX_DEVELOPMENT_INPUTS,
        ASCEND_DEVELOPMENT_INPUTS,
        _H200_SCALE_UP_INPUTS,
        _H200_SCALE_OUT_PROJECTED_INPUTS,
    )
)


def development_protobridge_inputs() -> Tuple[Mapping[str, object], ...]:
    """Return immutable development profile inputs for all supported hardware."""

    return _DEVELOPMENT_INPUTS


def development_protobridge_inputs_for(
    hardware_id: str,
    link: Optional[str] = None,
) -> Mapping[str, object]:
    """Resolve one hardware profile input record or fail loud."""

    normalized = canonical_hardware_id(hardware_id)
    if link is None:
        normalized_link = None
    elif isinstance(link, str) and link.strip():
        normalized_link = link.strip().lower()
    else:
        raise ValueError(
            "development ProtoBridge link must be non-empty or null")
    matches = tuple(
        inputs for inputs in _DEVELOPMENT_INPUTS
        if inputs["hardware_id"] == normalized
        and (normalized_link is None or inputs["link"] == normalized_link)
    )
    if len(matches) != 1:
        raise NotImplementedError(
            "no unique development ProtoBridge inputs for "
            f"{normalized!r} link={normalized_link!r}")
    return matches[0]
