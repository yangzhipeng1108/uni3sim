"""Canonical hardware identifiers and the sole legacy-read alias map."""

from typing import Dict


CANONICAL_HARDWARE_IDS = frozenset({
    "nvidia_a800",
    "nvidia_h200",
    "hygon_bw1100",
    "nvidia_rtx_pro_5500",
    "intel_agmf039r47",
    "metax_c500",
    "ascend_910b",
    "ucb_gemmini",
    "groq_lpu",
    "samsung_hbm_pim",
})

# The three hardware the canonical evaluations cover, in report order.  This
# is the declaration every consumer used to restate as its own literal; the
# full registry above also holds the non-evaluated cards.
EVALUATED_HARDWARE_IDS = ("nvidia_a800", "metax_c500", "ascend_910b")


LEGACY_HARDWARE_ID_ALIASES: Dict[str, str] = {
    "a800": "nvidia_a800",
    "a800-sxm4-80gb": "nvidia_a800",
    "c500": "metax_c500",
    "c500-64gb": "metax_c500",
    "910b": "ascend_910b",
    "910b-64gb": "ascend_910b",
    "huawei_910b": "ascend_910b",
    "gemmini": "ucb_gemmini",
    "lpu": "groq_lpu",
    "hbm-pim-16gb": "samsung_hbm_pim",
    "pim": "samsung_hbm_pim",
    "dataflow": "ucb_gemmini",
}


def canonical_hardware_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("hardware_id must be a non-empty string")
    normalized = value.strip().lower().replace(" ", "_")
    if normalized in LEGACY_HARDWARE_ID_ALIASES:
        return LEGACY_HARDWARE_ID_ALIASES[normalized]
    elif normalized in CANONICAL_HARDWARE_IDS:
        return normalized
    else:
        raise ValueError(f"unknown hardware_id: {value!r}")
