#!/bin/bash
# AIV instruction-cache cold-start cost, measured once per process.
#
# Interleaving two opcodes of the same binary does NOT evict the AIV
# instruction cache (both bodies stay resident, PMU 0.000000), so the only
# reproducible cold state available from this bench is a process's first
# launch.  Each invocation below contributes one cold launch followed by
# warm repeats of the identical cell, so the pair is measured at fixed
# geometry, element count and iteration count.
BIN=${MIDLAYER_ASCEND_VECTOR_OPCODE_BIN:?set MIDLAYER_ASCEND_VECTOR_OPCODE_BIN}
for i in $(seq 1 20); do
  $BIN --batch 21 1024 48 1 21 1024 48 1 21 1024 48 1 > /dev/null 2>&1 || true
done
