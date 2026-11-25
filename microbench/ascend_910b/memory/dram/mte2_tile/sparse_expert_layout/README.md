# GroupedMatmul sparse-expert B ingress

This non-production Ascend 910B microbenchmark isolates the CANN 8.5.1 BF16 GroupedMatmul W1 B operand's physical-address topology. It preserves the target's `K=6144`, `N=32768`, `baseN=256`, `stepKb=4`, 24 AIC workers, three populated one-M-tile experts, source `count/cur_block` assignment, B1/L0B ping-pong, `CopyND2NZOnTheFly`'s sixteen C0-width copies, and B L2-cache-disable hint. A is prefetched before timing, C is never stored, and the benchmark contains no routing, output, activation, or production FusedMoE code.

Layout `0` maps the three active source programs to adjacent expert slots `[1,2,3]`; layout `1` uses the canonical target's sparse slots `[1,4,6]`. The allocation deliberately holds eight full expert spans, so the comparison changes only physical GM addresses. The host records the max AIC completion time using a 1x/4x differential and validates that all 24 workers complete the exact same 384 B1 groups per source pattern.

The result is mechanism evidence only. It may become a Forward service only if repeated fresh-process captures establish a stable, non-zero sparse-minus-adjacent critical-path effect on independent neighbouring non-production K/N geometries. It must not be used as an OP residual or as a duration-derived per-case correction.
