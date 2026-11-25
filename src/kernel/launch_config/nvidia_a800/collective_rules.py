"""NVIDIA A800 NCCL collective selection rules."""

from src.kernel.launch_config.collective_selection import (
    CollectiveRuleInterval as _Band,
    CollectiveRuleSet as _RuleSet,
    CollectiveRuleTier as _Tier,
)


_MAX_COLLECTIVE_BYTES = 128 * 1024 * 1024

# NCCL v2.27.3-1 computes protocol cost as latency_us +
# nBytes/(1000*bandwidth_GB_s) in graph/tuning.cc.  For this two-rank
# single-node A800 AllGather topology, its static ring table is LL 7.2/160.0,
# LL128 15.9/294.4, and SIMPLE 11.8/320.0.  getAlgoInfo() uses two times the
# per-rank input for AllGather; LL128 is dominated, and LL ties SIMPLE at a
# per-rank input of 736,000 bytes.  topoGetAlgoInfo() visits LL first and only
# replaces the winner on a strict cost decrease, so the tie remains LL.  BF16
# permits only even byte counts, making 736,002 the first SIMPLE input.
_ALL_GATHER_TWO_RANK_LL_MAX_BYTES = 736_000

# The same NCCL cost calculation gives the two-rank AllReduce ring costs as
# LL 7.8 + nBytes/(1000*80.0) and SIMPLE
# 15.2 + nBytes/(1000*160.0).  LL128 and the tree candidates are dominated at
# their crossing.  LL and SIMPLE tie at 1,184,000 bytes; the strict comparison
# and protocol traversal order retain LL at the tie, so the next BF16 byte
# count is the first SIMPLE input.
_ALL_REDUCE_TWO_RANK_LL_MAX_BYTES = 1_184_000


def _nccl_ring(*bands):
    # Intervals exclude boundary brackets that remain unresolved to an exact
    # byte; a message between two bands raises the bracket message.
    return (_Tier(minimum_bytes=2, alignment_bytes=2, intervals=tuple(
        _Band(lower, upper, "ring", protocol, None, "selected")
        for lower, upper, protocol in bands)),)


# NCCL 2.27.3 collective selection on one A800 NVLink node, excluding vLLM
# custom AllReduce.  Every band is directly profiled.
COLLECTIVE_RULES = _RuleSet(
    hardware_id="nvidia_a800",
    label="A800",
    library="NCCL",
    library_version="2.27.3",
    topology="single_node_nvlink",
    rule_version="nccl-2.27.3-a800-nvlink-profile-20260826-v1",
    participant_counts=(2, 4, 8),
    max_message_bytes=_MAX_COLLECTIVE_BYTES,
    tiers={
        ("all_reduce", 2): _nccl_ring(
            (2, _ALL_REDUCE_TWO_RANK_LL_MAX_BYTES, "ll"),
            (_ALL_REDUCE_TWO_RANK_LL_MAX_BYTES + 2,
             _MAX_COLLECTIVE_BYTES, "simple")),
        ("all_reduce", 4): _nccl_ring(
            (2, 1_572_864, "ll"),
            (1_835_008, 3_932_160, "ll128"),
            (4_194_304, _MAX_COLLECTIVE_BYTES, "simple")),
        ("all_reduce", 8): _nccl_ring(
            (2, 2_359_296, "ll"),
            (2_621_440, 15_728_640, "ll128"),
            (16_777_216, _MAX_COLLECTIVE_BYTES, "simple")),
        ("all_gather", 2): _nccl_ring(
            (2, _ALL_GATHER_TWO_RANK_LL_MAX_BYTES, "ll"),
            (_ALL_GATHER_TWO_RANK_LL_MAX_BYTES + 2,
             _MAX_COLLECTIVE_BYTES, "simple")),
        ("all_gather", 4): _nccl_ring(
            (2, 524_288, "ll"),
            (589_824, _MAX_COLLECTIVE_BYTES, "simple")),
        ("all_gather", 8): _nccl_ring(
            (2, 393_216, "ll"),
            (425_984, 1_179_648, "ll128"),
            (1_310_720, _MAX_COLLECTIVE_BYTES, "simple")),
    },
    evidence_run_ids={
        ("all_reduce", 2): (
            "20260826_nccl_profiler_ar_pow2_v2",
            "20260826_nccl_profiler_ar_w2_boundary_v1",
            "nccl-v2.27.3-1-72d2432094d6-tuning.cc",
        ),
        ("all_reduce", 4): (
            "20260826_nccl_profiler_ar_w4_pow2_v1",
            "20260826_nccl_profiler_ar_w4_boundary_v1",
        ),
        ("all_reduce", 8): (
            "20260826_nccl_profiler_ar_w8_pow2_v1",
            "20260826_nccl_profiler_ar_w8_boundary_v1",
        ),
        ("all_gather", 2): (
            "20260826_nccl_profiler_ag_pow2_v1",
            "20260826_nccl_profiler_ag_w2_boundary_v1",
            "nccl-v2.27.3-1-72d2432094d6-tuning.cc",
        ),
        ("all_gather", 4): (
            "20260826_nccl_profiler_ag_w4_pow2_v1",
            "20260826_nccl_profiler_ag_w4_boundary_v1",
        ),
        ("all_gather", 8): (
            "20260826_nccl_profiler_ag_w8_pow2_v1",
            "20260826_nccl_profiler_ag_w8_boundary_v1",
        ),
    },
    bracket_messages={
        key: ("A800 NCCL protocol boundary remains bracketed for "
              "operation={operation!r}, participant_count="
              "{participant_count}, message_bytes={message_bytes}")
        for key in (("all_reduce", 2), ("all_reduce", 4), ("all_reduce", 8),
                    ("all_gather", 2), ("all_gather", 4), ("all_gather", 8))
    },
)
