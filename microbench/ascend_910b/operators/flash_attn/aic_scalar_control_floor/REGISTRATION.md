# 910B FIA AIC scalar control floor registration

- Owner/baseline: `wt-910b-fa-latency`, `1b6bf72064b2ba1011c3959aa7267943c19cde70` plus this round's merged FA commits. Device 5 on `<capture-host-0118b599d5e7>`.
- Motivation, stated as observation only: in the canonical 334-row set the 910B FA residual's under-predicted body (short Q 28.29%, short KV 20.65%) shows measured duration rising with the number of active AIC blocks at essentially constant per-block work — at `S512 D128 contiguous`, `B2` and `B4` both launch `block_dim=16` and differ by 1.9% despite `B4` carrying twice the work, while `B1` at `block_dim=8` is 65% faster than `B2` at 16. Over the same cases `aic_scalar_ratio` is 0.52–0.68, against 0.25 where the model is accurate. `configs/hardware/ascend_910b.yaml` still carries `fma_pipe` (AIC scalar) as `throughput_per_clk: 16  # [NEEDS_MICROBENCH] ... placeholder`. No production duration is consumed by this probe, and nothing here presumes which of the two hypotheses below is true.

## The two hypotheses this probe separates

- **H-envelope**: the published `fia_mixed_task_outer_envelope` (`5213.005 + 55.681 * active_aiv_workers` ns) under-measures the real mix-task floor, because it was taken on a task with no AIC program. Over `block_dim` 8→24 it accounts for only ≈1.8 µs, while the production floors over the same range rise by an order of magnitude more.
- **H-program**: the growth is per-source-iteration AIC scalar control work, which merely correlates with block count across the dataset.

These make opposite predictions once group count and program length are swept independently, so the probe sweeps both.

## Frozen timing boundary

Two spans are recorded per launch and never mixed.

- **Internal span**, per AIC: from the `GetSystemCycle()` read placed immediately before the first control iteration's `WaitFlag<HardEvent::MTE1_MTE2>`, to the `GetSystemCycle()` read placed immediately after a `PipeBarrier<PIPE_ALL>` following the last iteration's slot release. At `iters = 0` this is the empty control floor. The per-launch internal statistic is the **maximum** over the launch's AIC cores, because the task cannot finish before its slowest core.
- **Host span**: `aclrtEventElapsedTime` across the launch and its stream synchronize, on an otherwise idle stream.
- **Envelope** is defined as host span minus internal span, using those two definitions only.

The cell value for each of the three quantities is the median over fresh processes; per launch the internal statistic is the max over cores as above. The 20.0 ns system-counter tick converts cycles to ns. No other span definition, aggregation or estimator is used anywhere in this campaign, and coefficients from different aggregations are never mixed or selected between.

## Probe

Mix `KERNEL_TYPE_MIX_AIC_1_2`, matching the FIA actor topology. Each AIV executes one `SyncAll`, the FIA-mandatory prelude, and then idles. Each AIC executes `iters` iterations of the source's L1 slot lifetime with the descriptor removed: `WaitFlag<HardEvent::MTE1_MTE2>` on a two-slot ping-pong, integer address arithmetic for one descriptor, `SetFlag<HardEvent::MTE2_MTE1>`, `WaitFlag<HardEvent::MTE2_MTE1>`, `SetFlag<HardEvent::MTE1_MTE2>`. There is no `DataCopy`, no `LoadData`, no `Mmad`, no `Fixpipe`, no Vector arithmetic and no production memory traffic, so this cannot reproduce a production Attention duration and is not a partial FIA reimplementation. It measures the control skeleton alone.

## Frozen cells

- `active_aic_groups` in `{1, 4, 8, 16, 22, 24}` — the launch `block_dim` values the canonical 910B FA dataset actually uses, plus 1 and 4 as low anchors.
- `iters` in `{0, 16, 64, 256}`.
- Full cross product, three fresh processes per cell, one untimed warmup launch per process.

## Acceptance, declared before collection

1. Every core validates and stderr is empty for every launch. Any failure rejects the campaign without resampling.
2. Process-median stability: for each cell, `(max - min) / median` over the three process medians must be `<= 5%`, separately for the internal span and the host span.
3. Separation gate: the campaign is interpretable only if the `iters` slope is resolvable, i.e. at `active_aic_groups = 24` the internal span at `iters = 256` exceeds the internal span at `iters = 0` by at least 10%. Otherwise the control body is too cheap to measure at this length and no per-iteration parameter is published.
4. Cross-check against published evidence: at `iters = 0` the measured envelope must be compared against `5213.005 + 55.681 * (2 * active_aic_groups)` ns. Agreement within 15% supports H-envelope being already correctly captured; a larger, group-dependent excess is evidence against it. **Either outcome is reported as measured.** Whichever way it falls, no Forward change is made on the basis of this campaign alone, and no coefficient is chosen by evaluation error.

## Amendment: serialized scalar GM read cell (frozen before its own collection)

The campaign above is complete. This amendment adds one further frozen question with the identical timing boundary, aggregation and process count; it changes nothing already collected.

Question. The measured AIC scalar floor is 8.0 us at `block_dim` 8 rising to 48 us at `block_dim` 24, and the binary bound in the accuracy note shows it cannot be instruction issue, so it is a stall. The one stall source with an already published independent parameter is the serialized scalar GM read, `aic_scalar_global_get_value_serialized` at 93.317 ns per value. Eighty-six such reads would make 8 us and about 514 would make 48 us, both plausible for tiling parsing. What is not known is whether that per-read cost is constant or grows with the number of AIC cores issuing concurrently; only a growing cost can produce a floor that rises with `block_dim`.

Cell: `gm_reads` in `{0, 32, 128, 512}` crossed with `active_aic_groups` in `{1, 8, 16, 22, 24}`, `iters = 0`, three fresh processes. Each read is an `int64` `GetValue` whose index is derived from the previous returned value, so the chain is serialized and cannot be reordered or elided; the source region is worker-disjoint and larger than the shared L2 so reads are not served from a warm line.

Declared interpretation, fixed before collection. Let `c(g)` be the per-read cost at `g` groups, taken as the `gm_reads = 512` slope. If `c(24) / c(1)` is below 1.25, the per-read cost does not scale with concurrency, serialized scalar GM reads cannot produce a floor that grows with `block_dim`, and this candidate is rejected. If it is at or above 1.25, the growth is real and `c(g)` becomes the candidate mechanism, still requiring an independent count of reads per block before anything reaches Forward. Either outcome is reported as measured; no Forward change follows from this cell alone and no coefficient is chosen by evaluation error.

## Amendment: empty mix-task duration in the msprof timing domain (frozen before its own collection)

The two cells above are complete. This amendment adds one further frozen question. It uses a different timing instrument, declared here in full, and changes nothing already collected.

Motivation. The canonical 910B evaluation target is `msprof_timeline_op_median`, that is the `Task Duration(us)` column of the msprof timeline `op_summary` CSV. The published `fia_mixed_task_outer_envelope`, which supplies Forward's whole task-level floor, was measured in a different instrument, `aclrt_event_minus_internal_system_cycle`, and yields 5213.005 + 55.681 * active_aiv_workers ns, about 7.0 us at sixteen AIC groups. A single exploratory msprof capture of this probe's empty mix task at sixteen groups returned Task Duration 8.736 and 17.496 us. If an empty mix task really costs that much in the target instrument, then part of what this note has been attributing to the kernel's own AIC scalar work is task startup, and the two instruments are not interchangeable.

Timing boundary for this cell. The recorded quantity is the `Task Duration(us)` column of the newest `PROF_*/mindstudio_profiler_output/op_summary_*.csv` produced by `msprof --application=<binary> --ai-core=on --aic-metrics=PipeUtilization --duration=<n> --output=<dir>`, restricted to rows whose `Op Name` is `aic_scalar_control_floor`. This is the same column, same tool and same filter the canonical 910B kernel dataset uses for its timing target. Within one capture the **first** launch is discarded as warmup and the cell value is the **median** of the remaining launches; across processes the cell value is the median of per-capture medians. No other column, instrument or aggregation is used, and values from this instrument are never mixed with the in-kernel `GetSystemCycle` spans or the `aclrtEvent` spans recorded by the earlier cells.

Cell: `active_aic_groups` in `{1, 4, 8, 16, 22, 24}`, `iters = 0`, `gm_reads = 0`, twenty launches per capture, three fresh captures.

Declared interpretation, fixed before collection. Let `D(g)` be the empty mix-task duration and `E(g) = 5213.005 + 55.681 * 2g` ns be the published envelope. If `D(g)` stays within 25% of `E(g)` across the range, the two instruments agree and this candidate is rejected. If `D(g)` exceeds `E(g)` by more than 25% and rises with `g`, then Forward's task-level floor is under-modelled in the target instrument, and `D(g)` becomes a candidate lower bound — a launch and synchronization quantity, which is an allowed zero-fit source. Even then nothing reaches Forward from this cell until it is composed as a lower bound on the task, never as an additive residual, and shown against the full 334-row three-card comparison. Either outcome is reported as measured; no coefficient is chosen by evaluation error.

## Amendment: AIC-to-AIV handoff round trip (frozen before its own collection)

The three cells above are complete. This amendment adds the last structural candidate for the under-predicted body, using the same in-kernel `GetSystemCycle` boundary and aggregation as the first two cells.

Motivation and what is already excluded. The measured AIC scalar floor is 8.0 us at eight AIC groups rising to 48 us at twenty-four, inside the kernel. Rejected so far: scalar instruction issue (binary bound, at most 3302 instructions = 1.84 us); the slot-lifetime control skeleton (7.1875 ns per iteration, group independent); an under-measured mix-task envelope; serialized scalar GM reads (1.187x over one to twenty-four groups against a 1.25 threshold); task startup (empty mix task 6.13 us at sixteen groups where Forward already charges 6.99); and AIC instruction fetch, since `aic_icache_miss_rate` is populated in this dataset and can be non-zero elsewhere yet is exactly 0.0 across all 291 FA cases. Cross-core L2 port contention on the scalar read path is covered by the disjoint GM-read sweep at 1.187x.

What remains is the coupling itself. In a mix task both domains spend most of their time waiting on each other: production `aic_scalar_ratio` is 0.52 to 0.68 and `aiv_scalar_ratio` about 0.31, and the per-block layer shows AIV span tracking AIC span. The published `fia_cross_domain_three_handoff` measured the complete ring as a whole-program span, and Forward uses it only as a `max()` anchor which a counterfactual showed to be inert. The per-round-trip latency at production core counts has never been measured on its own.

Probe. Mix 1-AIC:2-AIV. Per round: the AIC issues `CrossCoreSetFlag<2, PIPE_FIX>` then `CrossCoreWaitFlag`, and the AIV mirrors it, using the source's own `FIA_SYNC_MODE2`. No DataCopy, Mmad, Fixpipe, Vector arithmetic or memory traffic. Cell: `handoffs` in `{0, 8, 32, 128}` crossed with `active_aic_groups` in `{1, 8, 16, 22, 24}`, `iters = 0`, `gm_reads = 0`, three fresh processes. Per-round-trip latency is the `handoffs = 128` slope against `handoffs = 0`.

Declared interpretation, fixed before collection. Let `h(g)` be the per-round-trip latency. FIA executes a bounded number of rounds per source task, so a candidate mechanism needs `h(g)` large enough that the source's round count reaches the several-microsecond floor, and it needs `h(g)` to grow with `g`. If `h(24) / h(8)` is below 1.25, the round trip does not scale with concurrency and cannot produce a floor that grows with block count; the candidate is rejected and the scoped G6 verdict stands with a sixth rejection. If it is at or above 1.25 and `h(16)` times a source-derived round count per task is within a factor of two of the 13.0 us floor at sixteen groups, the round trip becomes the candidate mechanism, still requiring an independent source-derived round count before anything reaches Forward. Either outcome is reported as measured; no coefficient is chosen by evaluation error.

## Amendment: shared-line scalar GM reads (frozen before its own collection)

The earlier `gm_reads` cell used worker-disjoint addresses and measured per-read cost rising only 1.187x from one group to twenty-four, on which the serialized-GM-read candidate was rejected. That tested the wrong regime. Differential decoding of the shipped kernel since then shows 229 scalar GM loads in the AIC function, 115 of them in the first quarter, and production has every AIC core reading the same 1560-byte tiling struct. That is a shared line under concurrency, not disjoint addresses; the cross-core L2 reuse trap recorded elsewhere concerns strided ND2NZ reuse, which is a different access. The candidate is therefore reopened and re-tested in the regime the production kernel actually uses.

Probe change. A `shared` flag makes every AIC core walk the same dependent chain over one small region rather than its own disjoint region. The region is 4 KiB, the order of the production tiling struct, so it is cache-resident as the production read is, and the chain stays dependent so reads remain serialized per core. Everything else, including the timing boundary and aggregation, is unchanged from the earlier `gm_reads` cell.

Cell: `gm_reads` in `{0, 32, 128, 512}` crossed with `active_aic_groups` in `{1, 8, 16, 22, 24}`, `shared = 1`, `iters = 0`, `handoffs = 0`, three fresh processes. Per-read cost is the `gm_reads = 512` slope, cells failing the 5% process-median stability gate are excluded from the ratio as before.

Declared interpretation, fixed before collection. The measured floor rises from 8.5 us at eight groups to 13.0 us at sixteen, a factor of 1.63. Let `c(g)` be the shared-line per-read cost. If `c(16)/c(8)` is at or above 1.63, shared-line contention can carry the floor's growth and the candidate survives as the mechanism, still requiring the read count per block to come from the binary rather than from any fit before it reaches Forward. If it is below 1.63 but at or above 1.25, the effect is real but insufficient alone and is reported as partial. If it is below 1.25, the candidate is rejected a second time, now in the correct regime, and the scoped G6 verdict is restored. Either outcome is reported as measured; no coefficient is chosen by evaluation error.

## Publication boundary

A passing campaign publishes, at most: an AIC scalar control service per source slot-lifetime iteration, in ns per iteration per AIC, over the exact measured `(active_aic_groups, iters)` domain; and, separately, a measured mix-task envelope versus active group count under a non-empty AIC program. It does not authorize replacing any existing per-stage service, does not by itself justify changing `fma_pipe`'s aggregate throughput, and licenses no extrapolation beyond the measured group counts. Forward must remain fail-closed outside the measured domain, and any Forward candidate must pass the same 334-row three-card comparison as the rest of this owner.
