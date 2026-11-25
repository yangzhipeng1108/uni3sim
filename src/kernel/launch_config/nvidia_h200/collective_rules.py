"""Exact automatic NCCL choices observed on one two-rank H200 pair.

Canonical raw: profiler_output/serving/nvidia_h200/host_profiles/
20260927_221848/profiler_output/microbench/nvidia_h200/nccl_ring_protocol_ws2/.
report.json binds BF16, NCCL2.27.3, two ranks and per-rank input bytes;
nccl_auto.*.250814.log and .250815.log record automatic choices with neither
NCCL_ALGO nor NCCL_PROTO forced. Existing reference raw_report owns the hash:
microbench/nvidia_h200/references/nccl_ring_protocol_ws2_protobridge_inputs_sm90.json.

Each singleton below is a directly logged message size, not an interpolated
threshold. No duration, forced-protocol measurement or A800 choice is used.
These are algorithm/protocol facts, not physical CTA geometry or resources.
The same campaign's SHA256SUMS-pinned provenance/nvidia-smi-topo.txt records
NV18 between devices4/5. Bind that exact observed pair identity rather than
claiming that all H200 NVLink pairs select these choices.
"""

from src.kernel.launch_config.collective_selection import (
    CollectiveRuleInterval, CollectiveRuleSet, CollectiveRuleTier,
)


RULE_VERSION = "nccl-2.27.3-h200-observed-pair-ws2-auto-points-20260927-v1"
OBSERVED_TOPOLOGY = "single_node_h200_observed_pair_4_5_20260927"


_OBSERVED_POINTS = tuple(
    CollectiveRuleInterval(1 << exponent, 1 << exponent, "ring", protocol, None, "selected")
    for protocol, exponents in (("ll", range(8, 21)), ("simple", range(21, 29)))
    for exponent in exponents)


COLLECTIVE_RULES = CollectiveRuleSet(
    hardware_id="nvidia_h200", label="H200 two-rank observed pair",
    library="NCCL", library_version="2.27.3", topology=OBSERVED_TOPOLOGY,
    rule_version=RULE_VERSION, participant_counts=(2,), max_message_bytes=1 << 28,
    tiers={(operation, 2): (CollectiveRuleTier(256, 2, _OBSERVED_POINTS),)
        for operation in ("all_reduce", "all_gather")},
    evidence_run_ids={(operation, 2): ("20260927_221848_nccl_ring_protocol_ws2_auto",)
        for operation in ("all_reduce", "all_gather")},
    bracket_messages={(operation, 2): (
        "H200 automatic NCCL choice is unobserved for operation={operation!r}, "
        "participant_count={participant_count}, message_bytes={message_bytes}; "
        "only exact two-rank logged message sizes are bound")
        for operation in ("all_reduce", "all_gather")},
)
