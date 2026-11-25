# 910B FIA drained ND2NZ descriptor: AIC-concurrency service under two source-address layouts

- Owner/baseline: `wt-910b-gating-device-pair`, `5141eb9514e99205b7e5060c8273882329f77509`. Device 0 on `<capture-host-0118b599d5e7>`, container `<ascend-container>`.
- This registration is written before any measurement of this probe is taken.

## Why this cell exists

`round16`'s runtime counter probe (report §5.100.44) measured, on the production FIA decode operator with per-core work held strictly constant while the active AIC block count went 8 → 16: `aic_cube_time` 1.00x, `aic_mte1_time` 0.99x, `aic_fixpipe_time` 1.01x — and `aic_mte2_time` **1.70x**. The published concurrency law `aic_mte2_fia_gqa_drained_descriptor_count_law_cann_8_5_1.json` grows only **1.257x** over that same step (1240.5 → 1558.9 ns per descriptor at `columns=128`). Per 64 KiB of KV, production reads **1172 ns at 8 blocks (0.95x the probe) and 1997 ns at 16 blocks (1.28x the probe)**: the two agree at low concurrency and diverge as concurrency rises.

Two candidate differences were considered. AIV co-issue was **rejected before this probe was written**, from the same counters: the AIV side moves 83.7 KiB against 2.00 MiB of KV at 8 blocks and 129.5 KiB against 4.00 MiB at 16 blocks, i.e. 3-4% of the bytes, too small to carry a 28% divergence. The remaining difference is the one this probe tests: the published law is measured with **worker-disjoint source regions and a non-power-of-two skew chosen to keep consecutive descriptors out of the same L2 sets**, while production FIA cores read **one shared, densely strided KV tensor** that the launch splits across them.

## Probe

The AIC descriptor program, its slot lifetime, its MTE2-to-MTE1 completion edge, its single minimal `LoadData2D` consumer and its timing boundary are character-identical to the published `fia_gqa_drain_count_law` probe. The only added degree of freedom is the source offset:

- `address_mode = 0` (**disjoint**): the published layout. Worker `w` reads its own `kElementsPerWorker` region, descriptor `i` at `i * (rows * stride + 8192)`.
- `address_mode = 1` (**packed**): one shared region, no skew, no per-worker partition. Descriptor `i` of worker `w` reads tile `i * workers + w`, each tile `rows * stride` elements, so the descriptors issued concurrently at one step are adjacent in memory.

It omits Q, P, MMAD, Fixpipe, softmax, AIV work, output composition and cross-domain handoff, so it cannot reproduce a production Attention duration, and it consumes no production or evaluation duration.

## Frozen timing boundary

Identical to the published probe: `GetSystemCycle()` immediately before the first descriptor's `WaitFlag<HardEvent::MTE1_MTE2>` to `GetSystemCycle()` immediately after the last descriptor's slot release, one `PipeBarrier<PIPE_ALL>` before the closing read, divided by `count`, in `ns_per_descriptor_per_aic` at the 20.0 ns system-counter tick. Per-launch statistic is the median over the launch's workers; the cell value is the median over three fresh processes. One untimed full-size warmup launch on a disjoint pass precedes the timed launch. No other span definition, aggregation or estimator is used, and coefficients from different aggregations are never mixed.

## Frozen cells

- Geometry: `rows=256, columns=128, source_row_stride_elements=1024` (the D128 GQA K/V geometry of the published law).
- `active_aic_groups` in `{8, 16, 22, 24}` — the `block_dim` values the canonical 910B FA dataset launches.
- `address_mode` in `{disjoint, packed}`; `descriptor_count` 64; three fresh processes per cell.

## Acceptance and decision, declared before collection

1. Every worker validates and stderr is empty for every launch; any failure rejects the campaign without resampling.
2. Process-median stability: `(max - min) / median` over the three process medians must be `<= 5%` per cell.
3. **Instrument cross-check**: the `disjoint` cells must reproduce the published `aic_mte2_fia_gqa_drained_descriptor_count_law` values at the same groups within **10%**. Failure means this probe is not measuring the same thing as the published campaign and rejects it outright, rather than being resolved by preferring either number.
4. **Decision on the carrier.** Production's per-core MTE2 time grows **1.70x** over blocks 8 → 16 at constant per-core work, against the published disjoint law's 1.257x.
   - The locality candidate is **confirmed** if `packed` grows over blocks 8 → 16 by a factor of **at least 1.55x** (within 10% of production's 1.70x) while `disjoint` reproduces its published ~1.26x.
   - It is **rejected** if `packed` grows by less than 1.40x, i.e. it closes under a third of the gap between 1.257x and 1.70x.
   - Between 1.40x and 1.55x the result is reported as partial and no parameter is published from it.
5. No cell, group count, geometry or threshold here was chosen by looking at evaluation APE, MAPE or any production duration. The growth factor the decision brackets is a **ratio measured on production counters**, used as a falsification target; the service level a passing campaign would publish comes from this probe alone, never from production.

## Publication boundary

A passing campaign publishes a drained per-descriptor ND2NZ service over the measured `(geometry, active_aic_groups, address_mode)` domain only. It authorizes no interpolation across geometry, group count or layout, replaces no existing per-stage service, and by itself justifies no Forward change: wiring it in requires a separate step that establishes which layout each production launch presents, from the launch derivation rather than from error.

## Amendment: packed-layout publication grid (frozen before its own collection)

The decision campaign above is complete: `locality_confirmed`, packed 1.687x against production's 1.70x, with the disjoint cells reproducing the published law within 1.1%. This amendment adds the measurements needed to make the packed layout *usable* in place of the disjoint one, and it changes nothing about the cells already collected or the decision already recorded.

A further production cross-check, computed after the decision campaign and before this amendment: per 64 KiB of KV, production reads 1172 ns at 8 blocks and 1997 ns at 16. Against the packed cells those are **0.91x and 0.92x** — one consistent ratio at both concurrencies — while against the disjoint cells they are 0.94x and 1.28x. A constant sub-unity ratio is what partial L2 residency predicts, and Forward already carries that separately as `mte2_dram_miss_fraction`; an inconsistent ratio is not explainable that way. This is stated as corroboration of the layout choice, not as a parameter: nothing derived from production duration is published.

Question: does the packed layout satisfy the same count-independence law over the same domain the disjoint campaign published, so that a packed per-descriptor service can replace the disjoint one cell for cell?

Cells: `address_mode = packed`; geometry A `rows=256, columns=128, source_row_stride_elements=1024` and geometry B `rows=256, columns=64, source_row_stride_elements=512`; `active_aic_groups` in `{8, 16, 22, 24}`; `descriptor_count` in `{8, 16, 32, 64, 128}`; three fresh processes per cell. Timing boundary, aggregation and warmup are unchanged.

Gates, identical in form to the published disjoint campaign:

1. Every worker validates and stderr is empty for every launch; any failure rejects the campaign without resampling.
2. Process-median stability `<= 5%` per cell. A failing cell is excluded and named, exactly as the disjoint campaign excluded its two d64 cells.
3. Count-independence `<= 8%` span across `descriptor_count` 8..128 within each `(geometry, active_aic_groups)` cell. A cell failing this gate publishes no count-free packed service and Forward keeps its existing behaviour there.

A passing cell publishes a count-free packed per-descriptor service over that exact `(geometry, active_aic_groups)` domain. It still authorizes no interpolation across geometry, group count or layout. Which layout a production launch presents remains a separate source question, to be answered from the FIA launch derivation and not from any error signal.

## Amendment 2: the production descriptor-count domain (frozen before its own collection)

Amendment 1 returned a partly negative result and, in doing so, located the production domain precisely. Two facts, both established before this amendment was written:

1. **Count-independence does not extend to the packed layout at high concurrency.** Over `descriptor_count` 8..128 the packed span is 4.9% and 5.2% at 8 and 16 groups but **18.6% and 21.4%** at 22 and 24 groups (d128), and 24.1% / 21.9% (d64): service *falls* as count rises. Only 3 of 8 cells pass the 8% count gate, so no count-free packed service is publishable at 22 or 24 groups. That result stands as recorded.
2. **Production does not use that count range at all.** Instrumenting `aic_mte2_fia_gqa_finite_drain_supports` on the production FIA decode path gives `(rows=256, columns=128, stride=1024, groups, descriptor_count=2)` for every probed decode shape, and the count law's licensed range is 8..128, so the drained-program path returns unsupported and the launch is priced through `aic_mte2_nd2nz_strided_serial_time_ns` instead. That serial service is **flat in concurrency**: 549.9, 557.4, 565.1, 567.7 ns per descriptor at 8, 16, 22, 24 groups, i.e. **1.014x** over the 8 → 16 step where both measured layouts grow 1.244x (disjoint) and 1.687x (packed) and production grows 1.70x.

Question: at the descriptor count production actually issues, what is the drained per-descriptor service under each layout, and how does it scale with concurrency?

Cells: `descriptor_count` in `{1, 2, 4}`; `address_mode` in `{disjoint, packed}`; geometry A `rows=256, columns=128, source_row_stride_elements=1024` and geometry B `rows=256, columns=64, source_row_stride_elements=512`; `active_aic_groups` in `{8, 16, 22, 24}`; three fresh processes per cell. Timing boundary, aggregation and warmup unchanged.

Gates: worker validation and empty stderr as before; process-median stability `<= 5%` per cell, failing cells excluded and named. No count-independence gate applies here — this amendment measures the count dependence rather than assuming it away, and each `(geometry, groups, count, layout)` cell is reported on its own.

Declared interpretation, fixed before collection: a cell publishes an exact measured drained service at that `(geometry, groups, count, layout)` point and nothing else. The layout factor for a cell is the packed service divided by the disjoint service **at the same count, geometry and group count, both arms from this one instrument** — never across campaigns. Whether Forward may consume any of it remains gated on the separate source question of which layout a production launch presents, and on that launch's descriptor geometry matching a measured cell exactly. No threshold, cell or count here was chosen by looking at evaluation error.

## Amendment 3: launch-level aggregation at the production descriptor counts (frozen before its own collection)

Amendment 2 measured the right cells and could publish only 4 of 24 of them. The cause is in the aggregation, not in the mechanism: each process contributes **one** timed launch, and at `descriptor_count` 1..4 that launch spans only 3-4 us, so a single launch's worker-median carries enough run-to-run variance that three process medians spread up to 14.1%. The published disjoint campaign never met this because its cells span 8..128 descriptors.

This amendment changes the aggregation and nothing else. It does **not** change the timed region, the span definition, the per-worker statistic, the cells, the geometries, the group counts or the descriptor counts, and it does not lengthen or merge the descriptor programs — each timed launch remains exactly the same short `count`-descriptor drained program that Amendment 2 measured.

**Declared aggregation for this amendment**: each process performs `LAUNCHES = 15` timed launches after its untimed warmup. Launch `l` reads a source region offset by `l * count` descriptors from the pass base, so no launch revisits another's addresses and every launch is DRAM-cold in the same sense as before. The per-launch statistic stays the median over that launch's workers. **The process value is the median over its 15 launch values**, and the cell value remains the median over three fresh processes. The stability gate stays `(max - min) / median <= 5%` over the three process medians.

This layer is declared here because the existing registration fixes a single per-launch statistic; adding it is a change of estimator and may not be introduced silently. It is applied uniformly to every cell of this amendment, to both layouts, and the Amendment 2 values are retained as collected rather than being superseded or re-aggregated.

Declared interpretation, fixed before collection: a cell whose three process medians agree within 5% publishes an exact measured drained per-descriptor service at that `(geometry, active_aic_groups, descriptor_count, address_mode)` point. The layout factor remains the packed value divided by the disjoint value at the same point, both arms from this instrument. If the tightened aggregation still leaves the 22 and 24 group cells outside the gate, that is reported as a negative and no parameter is published for them; the gate is not relaxed and passing cells are not selected out of a failing set.
