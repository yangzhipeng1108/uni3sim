"""Huawei Ascend 910B development ProtoBridge inputs."""

DEVELOPMENT_PROTOBRIDGE_INPUTS = {
    "hardware_id": "ascend_910b",
    "communication_library": "HCCL",
    "library_version": "8.5.1",
    "selector_topology": "single_node_full_hccs",
    "selector_rule_version": "hccl-8.5.1-910b-full-hccs-profile-20260826-v1",
    "all_reduce_algorithms": ("mesh",),
    "all_gather_algorithms": ("mesh",),
    "default_all_reduce_algorithm": "mesh",
    "default_all_gather_algorithm": "mesh",
    "link": "hccs",
    "per_link_bytes_per_us": 28_000.0,
    "num_lanes": 7,
    "link_latency_us": 0.2,
    # These three are still development placeholders, and a standalone
    # measurement is not the number that belongs here.  The HCCL FullMesh
    # sweep in microbench/ascend_910b/references/
    # hccl_fullmesh_collective_startup_cann_8_5_1.json puts one collective at
    # 127-137 us back-to-back from 4 KiB to 128 KiB, flat across both protocol
    # boundaries, and the repeat-depth sweep shows that time is device-bound.
    # Charging it here is nevertheless wrong: a decode step executes on one
    # device serially, so its device time cannot exceed the measured token
    # cadence, and 126 us per collective drives the predicted decode step past
    # that bound on both TP=2 models at QPS 1 -- for qwen3-30b-a3b it would
    # spend 61% of the whole step budget inside the collectives alone.  What
    # the model needs is the marginal cost of a collective issued inside a
    # stream that already has compute in it, which the same sweep bounds from
    # the other side: its isolated pattern costs 220 us against 127 us
    # back-to-back, so roughly 93 us of the standalone figure is launch and
    # rank-skew that surrounding compute absorbs.  Measuring that marginal
    # cost needs a compute-plus-collective probe that does not exist yet.
    "small_startup_us": 60.0,
    "ll128_startup_us": 60.0,
    "simple_startup_us": 60.0,
    "small_phase_fixed_us": 0.2,
    "ll128_phase_fixed_us": 0.2,
    "simple_phase_fixed_us": 0.2,
    "simple_allreduce_ring_bw_by_participant": (),
    "spray_chunk_bytes": 512 * 1024,
    "t_min_switch_us": 1.1,
    "wire_efficiency": 1.0,
    # The independent two-rank HCCL 8/16 MiB back-to-back AllReduce sweep gives
    # (16-8) MiB / (940.3892-487.4020) us = 18,518.42 B/us. The 4/8/16 MiB
    # points form one linear large-payload region; retain the 28,000 B/us
    # physical peer path and apply its measured efficiency only to two ranks.
    # Source: hccl_fullmesh_collective_startup_cann_8_5_1.json, not Serving.
    "large_message_min_bytes": 4 * 1024 * 1024,
    "large_message_wire_efficiency": 0.6613722040206498,
    "provenance_note": "",
    "bisection_bw_bytes_per_us": 28_000.0,
    "hop_count": 1.0,
}
