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
    # Token 工厂选型新增（2026-11）。B/C 类配置：屋顶线与容量级，
    # 不具备 EVALUATED 三张卡的 microbench 标定与 launch_config 逐 kernel 建模。
    "nvidia_h20",        # TF-DESIGN-03 基准卡；官方 ID 表原本没有
    "nvidia_b200",       # 官方规格，屋顶线级
    "nvidia_b300",       # 官方规格，屋顶线级（带宽为推算）
    "ascend_950pr",      # 已填公开规格，时钟待 CANN platform.ini
    "ascend_950dt",      # 已填公开规格，时钟待 CANN platform.ini
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
    # Token 工厂选型新增别名
    "h20": "nvidia_h20",
    "h20-96gb": "nvidia_h20",
    "h20-sxm5-96gb": "nvidia_h20",
    "b200": "nvidia_b200",
    "blackwell_b200": "nvidia_b200",
    "b300": "nvidia_b300",
    "blackwell_b300": "nvidia_b300",
    "950pr": "ascend_950pr",
    "ascend_950_pr": "ascend_950pr",
    "950dt": "ascend_950dt",
    "ascend_950_dt": "ascend_950dt",
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
