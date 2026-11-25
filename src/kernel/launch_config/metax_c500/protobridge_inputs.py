"""MetaX C500 development ProtoBridge inputs."""

DEVELOPMENT_PROTOBRIDGE_INPUTS = {
    "hardware_id": "metax_c500",
    "communication_library": "MCCL",
    "library_version": "2.16.5",
    "selector_topology": "single_node_metaxlink",
    "selector_rule_version": "mccl-2.16.5-c500-metaxlink-profile-20260827-v1",
    "all_reduce_algorithms": ("ring", "tree", "fc2"),
    "all_gather_algorithms": ("ring", "fc2"),
    "default_all_reduce_algorithm": "fc2",
    "default_all_gather_algorithm": "fc2",
    "link": "metaxlink",
    "per_link_bytes_per_us": 64_000.0,
    "num_lanes": 3,
    "link_latency_us": 0.5,
    # Independent MCCL FC2 AllReduce kernel-span microbenchmark, C500
    # two-rank MetaXLink, MCCL 2.16.5 on MACA 3.7.2, thirteen synthetic
    # message sizes from 2 KiB to 8 MiB.  Each is the span of the collective
    # on the rank that reached it last, so it excludes the rank skew the
    # execution timeline already represents by serialising ranks.  Reference
    # microbench/metax_c500/references/
    # mccl_collective_protocol_policy_maca_3_7_2.json; an independent
    # all_gather sweep reproduces the small band within 3.8%.
    "small_startup_us": 6.392,
    "ll128_startup_us": 10.520,
    # Not measured.  The sweep reaches one simple-band cell and the model
    # over-predicts it by 34% with the startup at zero, so no non-negative
    # startup reproduces it and the residual is in that band's bandwidth
    # term.  This stays a development placeholder and is reported as such.
    "simple_startup_us": 10.0,
    # Not separable at two ranks: an FC2 collective carries one phase there,
    # so the sweep constrains startup and phase only as their sum and the
    # startup above carries the measured difference.  Development values.
    "small_phase_fixed_us": 0.5,
    "ll128_phase_fixed_us": 0.5,
    "simple_phase_fixed_us": 0.5,
    "simple_allreduce_ring_bw_by_participant": (),
    "spray_chunk_bytes": 4096,
    "t_min_switch_us": 1.1,
    "wire_efficiency": 1.0,
    "provenance_note": "",
    "bisection_bw_bytes_per_us": 64_000.0,
    "hop_count": 1.0,
}
