"""MetaX C500 MCCL collective selection rules."""

from src.kernel.launch_config.collective_selection import (
    CollectiveRuleInterval as _Band,
    CollectiveRuleSet as _RuleSet,
    CollectiveRuleTier as _Tier,
)


_MAX_COLLECTIVE_BYTES = 128 * 1024 * 1024


def _two_rank(*, fc2_minimum_bytes, fc2_alignment_bytes,
              largest_observed_ring_ll_bytes):
    # MCCL takes FC2 for an aligned message of at least the FC2 minimum; its
    # LL protocol is observable below 128 KiB and unobservable above.  Every
    # other message runs the ring, whose LL protocol is observed up to the
    # largest profiled LL message and unresolved beyond it.
    return (
        _Tier(minimum_bytes=fc2_minimum_bytes,
              alignment_bytes=fc2_alignment_bytes, intervals=(
                  _Band(fc2_minimum_bytes, 131_071, "fc2", "ll", None,
                        "selected"),
                  _Band(131_072, _MAX_COLLECTIVE_BYTES, "fc2", None, None,
                        "unobservable"),
              )),
        _Tier(minimum_bytes=2, alignment_bytes=2, intervals=(
            _Band(2, largest_observed_ring_ll_bytes, "ring", "ll", None,
                  "selected"),
            _Band(largest_observed_ring_ll_bytes + 1, _MAX_COLLECTIVE_BYTES,
                  "ring", None, None, "unresolved"),
        )),
    )


def _selected(*bands):
    return (_Tier(minimum_bytes=2, alignment_bytes=2, intervals=tuple(
        _Band(lower, upper, algorithm, protocol, None, "selected")
        for lower, upper, algorithm, protocol in bands)),)


# MCCL 2.16.5 collective selection (profiled algorithm and available
# protocol) on one C500 MetaXLink node.
COLLECTIVE_RULES = _RuleSet(
    hardware_id="metax_c500",
    label="C500",
    library="MCCL",
    library_version="2.16.5",
    topology="single_node_metaxlink",
    rule_version="mccl-2.16.5-c500-metaxlink-profile-20260827-v1",
    participant_counts=(2, 4),
    max_message_bytes=_MAX_COLLECTIVE_BYTES,
    tiers={
        ("all_reduce", 2): _two_rank(
            fc2_minimum_bytes=32, fc2_alignment_bytes=32,
            largest_observed_ring_ll_bytes=131_088),
        ("all_gather", 2): _two_rank(
            fc2_minimum_bytes=16, fc2_alignment_bytes=16,
            largest_observed_ring_ll_bytes=131_074),
        # The LL/SIMPLE crossing between 524,290 and 540,672 bytes remains
        # bracketed.
        ("all_reduce", 4): _selected(
            (2, 32_767, "tree", "ll"),
            (32_768, 524_290, "ring", "ll"),
            (540_672, 16_777_215, "ring", "simple"),
            (16_777_216, 33_554_431, "tree", "simple"),
            (33_554_432, 67_108_863, "ring", "simple"),
            (67_108_864, _MAX_COLLECTIVE_BYTES, "tree", "simple")),
        # The LL/SIMPLE crossing between 155,648 and 159,744 bytes remains
        # bracketed.
        ("all_gather", 4): _selected(
            (2, 155_648, "ring", "ll"),
            (159_744, _MAX_COLLECTIVE_BYTES, "ring", "simple")),
    },
    evidence_run_ids={
        ("all_reduce", 2): (
            "20260826_mctracer_ar_w2_pow2_v1",
            "20260826_mctracer_ar_w2_boundary_v2",
            "20260827_mctracer_ar_w2_aligned_boundary_v1",
        ),
        ("all_reduce", 4): (
            "20260826_mctracer_ar_w4_pow2_v1",
            "20260826_mctracer_ar_w4_boundary_v1",
            "20260827_mctracer_ar_w4_aligned_boundary_v1",
        ),
        ("all_gather", 2): (
            "20260826_mctracer_ag_w2_pow2_v1",
            "20260826_mctracer_ag_w2_pow2_v4",
            "20260826_mctracer_ag_w2_boundary_v2",
            "20260827_mctracer_ag_w2_aligned_boundary_v1",
        ),
        ("all_gather", 4): (
            "20260826_mctracer_ag_w4_pow2_v1",
            "20260826_mctracer_ag_w4_boundary_v1",
            "20260827_mctracer_ag_w4_aligned_boundary_v1",
        ),
    },
    bracket_messages={
        # Two-rank tiers cover every admissible message.
        ("all_reduce", 2): (
            "C500 two-rank AllReduce band missing at "
            "message_bytes={message_bytes}"),
        ("all_gather", 2): (
            "C500 two-rank AllGather band missing at "
            "message_bytes={message_bytes}"),
        ("all_reduce", 4): (
            "C500 four-rank AllReduce LL/SIMPLE boundary remains bracketed "
            "at message_bytes={message_bytes}"),
        ("all_gather", 4): (
            "C500 four-rank AllGather LL/SIMPLE boundary remains bracketed "
            "at message_bytes={message_bytes}"),
    },
)
