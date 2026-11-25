"""Huawei Ascend 910B HCCL collective selection rules."""

from src.kernel.launch_config.collective_selection import (
    CollectiveRuleInterval as _Band,
    CollectiveRuleSet as _RuleSet,
    CollectiveRuleTier as _Tier,
)


_MAX_COLLECTIVE_BYTES = 128 * 1024 * 1024
_RANKS = (2, 4, 8)


def _hccl_mesh(*bands):
    # HCCL Mesh exposes its executor, not a protocol, so every band is
    # protocol-unobservable.
    return (_Tier(minimum_bytes=2, alignment_bytes=2, intervals=tuple(
        _Band(lower, upper, "mesh", None, executor, "unobservable")
        for lower, upper, executor in bands)),)


# HCCL 8.5.1 collective selection (the profiled Mesh executor) on one 910B
# full-HCCS node.
COLLECTIVE_RULES = _RuleSet(
    hardware_id="ascend_910b",
    label="910B",
    library="HCCL",
    library_version="8.5.1",
    topology="single_node_full_hccs",
    rule_version="hccl-8.5.1-910b-full-hccs-profile-20260826-v1",
    participant_counts=_RANKS,
    max_message_bytes=_MAX_COLLECTIVE_BYTES,
    tiers={
        **{("all_reduce", ranks): _hccl_mesh(
            (2, 262_144, "AllReduceMeshSmallCountExecutor"),
            (262_145, _MAX_COLLECTIVE_BYTES,
             "AllReduceMeshOpbaseLoopExecutor"))
           for ranks in _RANKS},
        **{("all_gather", ranks): _hccl_mesh(
            (2, _MAX_COLLECTIVE_BYTES, "AllGatherMeshOpbaseExecutor"))
           for ranks in _RANKS},
    },
    evidence_run_ids={
        ("all_reduce", 2): (
            "20260826_hccl_ar_w2_pow2_v1",
            "20260826_hccl_ar_w2_boundary_v1",
        ),
        ("all_reduce", 4): (
            "20260826_hccl_ar_w4_pow2_v1",
            "20260826_hccl_ar_w4_boundary_v1",
        ),
        ("all_reduce", 8): (
            "20260826_hccl_ar_w8_pow2_v1",
            "20260826_hccl_ar_w8_boundary_v1",
        ),
        ("all_gather", 2): ("20260826_hccl_ag_w2_pow2_v1",),
        ("all_gather", 4): ("20260826_hccl_ag_w4_pow2_v1",),
        ("all_gather", 8): ("20260826_hccl_ag_w8_pow2_v1",),
    },
    # The executor bands cover every admissible message; nothing is bracketed.
    bracket_messages={
        (operation, ranks): (
            "910B HCCL executor band missing for operation={operation!r}, "
            "participant_count={participant_count}, "
            "message_bytes={message_bytes}")
        for operation in ("all_reduce", "all_gather") for ranks in _RANKS
    },
)
