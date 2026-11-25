#!/usr/bin/env python3
"""Run the exact 32-AIV RMSNorm key-0 first-MTE3 edge campaign."""

import first_mte_completion_runner as base


CAMPAIGN = "rmsnorm_key0_h2048_first_mte3_32"
CELLS = (
    (32, 7, 4096, pair_order)
    for pair_order in base.PAIR_ORDERS
)
CELLS = tuple(CELLS)
SCHEDULE_CYCLES = 16
REQUIRED_INDEPENDENT_RUNS = 3

base.CAMPAIGN_CELLS[CAMPAIGN] = CELLS
base.CAMPAIGN_DEFAULT_SCHEDULE_CYCLES[CAMPAIGN] = SCHEDULE_CYCLES
base.CAMPAIGN_REQUIRED_INDEPENDENT_RUNS[CAMPAIGN] = REQUIRED_INDEPENDENT_RUNS


if __name__ == "__main__":
    base.main()
