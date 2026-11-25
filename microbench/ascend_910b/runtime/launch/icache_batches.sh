#!/bin/bash
# Independent AIV instruction-cache cold-start measurement.
#
# One binary, one kernel entry, one launch per --batch cell.  The WARM run
# issues the same opcode 32 times in a row, so every launch after the first
# re-executes code the AIV instruction cache already holds.  The COLD run
# interleaves it with a second opcode whose body lies elsewhere in the same
# text section, so consecutive launches execute disjoint code.  Geometry,
# element count and iteration count are identical in both runs, so the only
# difference is which code the AIV must fetch.  msprof reports both the task
# duration and aiv_icache_miss_rate, so the PMU arbitrates whether the
# interleaving actually evicted anything.
set -e
BIN=${MIDLAYER_ASCEND_VECTOR_OPCODE_BIN:?set MIDLAYER_ASCEND_VECTOR_OPCODE_BIN}
A_OP=${A_OP:-11}      # mul_serialized
B_OP=${B_OP:-27}      # rmsnorm_half_interval_reduce
ELEMS=${ELEMS:-2048}
BD=${BD:-48}
ITERS=${ITERS:-2}     # minimum even iteration count: expose the launch, not the loop

warm=""
cold=""
for i in $(seq 1 32); do
  warm="$warm $A_OP $ELEMS $BD $ITERS"
done
for i in $(seq 1 16); do
  cold="$cold $A_OP $ELEMS $BD $ITERS $B_OP $ELEMS $BD $ITERS"
done
echo "=== WARM"
$BIN --batch $warm > /dev/null
echo "=== COLD"
$BIN --batch $cold > /dev/null
