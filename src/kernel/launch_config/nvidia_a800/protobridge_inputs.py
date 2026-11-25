"""NVIDIA A800 development ProtoBridge inputs."""

DEVELOPMENT_PROTOBRIDGE_INPUTS = {
    "hardware_id": "nvidia_a800",
    "communication_library": "NCCL",
    "library_version": "2.27.3",
    "selector_topology": "single_node_nvlink",
    "selector_rule_version": "nccl-2.27.3-a800-nvlink-profile-20260826-v1",
    "all_reduce_algorithms": ("ring",),
    "all_gather_algorithms": ("ring",),
    "default_all_reduce_algorithm": "ring",
    "default_all_gather_algorithm": "ring",
    "link": "nvlink",
    "per_link_bytes_per_us": 25_000.0,
    "num_lanes": 8,
    "link_latency_us": 0.5,
    # Independent NCCL Ring AllReduce microbenchmark, A800 NV8, NCCL 2.27.3.
    # Each is ``collective startup + ring_phase_count * phase_fixed`` in us.
    "small_startup_us": 10.383,
    "ll128_startup_us": 25.738,
    "simple_startup_us": 23.859,
    "small_phase_fixed_us": 2.590,
    "ll128_phase_fixed_us": 0.659,
    "simple_phase_fixed_us": 6.321,
    # Independent NCCL Ring AllReduce GPU-kernel microbenchmark.  The
    # effective transfer rate depends on the number of ranks using the ring.
    "simple_allreduce_ring_bw_by_participant": (
        (2, 86_000.0),
        (4, 103_000.0),
        (8, 113_000.0),
    ),
    "spray_chunk_bytes": 4096,
    # NVSwitch pipeline latency and wire efficiency for the scale-up fabric;
    # explicit record fields so scale-out records can carry their own values.
    "t_min_switch_us": 1.1,
    "wire_efficiency": 1.0,
    "provenance_note": "",
    "bisection_bw_bytes_per_us": None,
    "hop_count": 1.0,
}
