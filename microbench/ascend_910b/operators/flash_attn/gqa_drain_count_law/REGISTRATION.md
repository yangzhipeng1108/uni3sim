# 910B FIA GQA drained ND2NZ descriptor-count law registration

- Owner/baseline: `wt-910b-fa-latency`, `1b6bf72064b2ba1011c3959aa7267943c19cde70`. Device 5 on `<capture-host-0118b599d5e7>`.
- Source scope: CANN 8.5.1 regular-FIA MM1/MM2 K/V ingress. With `qkvDSameFlag == true` — the case for every standard GQA/MHA shape, where Q, K and V share `D` — `pfa_service_matmul_kv_nd.h` skips the `mL0 == 2` next-tile prefetch block, so each `CopyKToL1`/`CopyVToL1` is followed immediately by its `WaitFlag<HardEvent::MTE2_MTE1>`. The descriptor program is therefore drained after every descriptor. This probe reproduces exactly that edge and nothing else.
- Hypothesis under test: the drained per-descriptor ND2NZ service at a fixed `(rows, columns, source_row_stride_elements, active_aic_groups)` is independent of the descriptor count. The already published `aic_mte2_fia_gqa_finite_drain` evidence is declared `exact_measured_geometry_and_descriptor_count_only`, so Forward may not interpolate it; production KV lengths almost never land on a measured count and fall back to a different extrapolation rule. If count-independence holds over a wide count range, a count-free per-descriptor service becomes publishable for the measured geometry/concurrency domain. This probe neither assumes a direction nor supplies a fitted correction.
- Probe: AIC-only. Per worker, `count` BF16 ND2NZ GM-to-L1 descriptors of the given geometry. Each descriptor runs the source's exact slot lifetime: `WaitFlag<HardEvent::MTE1_MTE2>` on its L1 slot, the ND2NZ `DataCopy`, `SetFlag`/`WaitFlag<HardEvent::MTE2_MTE1>`, one minimal `LoadData2D` L1-to-L0B consumer, then `SetFlag<HardEvent::MTE1_MTE2>` to release the slot. Two L1 slots, worker-disjoint source addresses, every descriptor at a distinct source offset so no address is revisited. It omits Q, P, MMAD, Fixpipe, AIV work, softmax, output composition, cross-domain handoff, production durations and evaluation counters.
- Construction note (recorded before any data was collected): a first draft omitted the MTE1 consumer. `WaitFlag<HardEvent::MTE2_MTE1>` stalls the MTE1 pipe, not the scalar pipe that reads the system counter, so without an MTE1 instruction the drain edge is not observable and the scalar loop races ahead. That draft produced only a smoke-test reading, which is discarded and published nowhere; the single `LoadData2D` restores the source's real MTE2-to-MTE1 dependency. The minimal consumer is a 16x16 block, three orders of magnitude smaller than the descriptor, so it does not make this an MTE1 measurement, but the reported service does include that one L1-to-L0 issue per descriptor and is declared as such.

## Frozen timing boundary

The timed span begins at the `GetSystemCycle()` read placed immediately before the first descriptor's `WaitFlag<HardEvent::MTE1_MTE2>` and ends at the `GetSystemCycle()` read placed immediately after the last descriptor's slot release, with one `PipeBarrier<PIPE_ALL>` between that release and the closing read so the scalar pipe observes the whole descriptor program. It excludes kernel entry, `TPipe`/`TBuf` initialization, the initial `SetFlag` arming, the final slot drain and kernel exit. The reported per-descriptor service is that span divided by `count`, in `ns_per_descriptor_per_aic`, using the 20.0 ns system-counter tick. The per-launch statistic is the median over the launch's workers; the cell value is the median over fresh processes. No other span definition, aggregation or estimator is used anywhere in this campaign, and coefficients from different aggregations are never mixed or selected between.

## Frozen cells

- Geometry A (D128 GQA K/V): `rows=256, columns=128, source_row_stride_elements=1024`.
- Geometry B (D64 GQA K/V): `rows=256, columns=64, source_row_stride_elements=512`.
- `active_aic_groups` in `{8, 16, 22, 24}`; `descriptor_count` in `{8, 16, 32, 64, 128}`.
- Three fresh processes per cell. One untimed full-size warmup launch per process precedes the timed launch.

## Acceptance (declared before collection)

1. Every worker validates, and stderr is empty, for every launch. Any failure rejects the campaign without resampling.
2. Process-median stability: for each cell, `(max - min) / median` over the three process medians must be `<= 5%`.
3. Count-independence gate: within each `(geometry, active_aic_groups)` cell, the per-descriptor service across `descriptor_count` 8..128 must span `<= 8%`. If any cell fails this gate, the count-free law is rejected for that cell and no count-free parameter is published for it; Forward keeps its existing fail-closed behaviour there.
4. Cross-check against already published evidence: at the overlapping points the new per-descriptor service must agree with `aic_mte2_fia_gqa_finite_drain*` within `10%`. Disagreement beyond that rejects publication and is reported as a measurement conflict rather than resolved by choosing the more favourable value.

No cell, count, group count or geometry in this registration was chosen by looking at evaluation APE, MAPE or any production duration. The geometries are the K/V source geometries the CANN launch selector derives for D64 and D128 GQA; the group counts are the `block_dim` values the canonical 910B FA dataset actually launches.

## Amendment: two-stream cell (frozen before its own collection)

The single-stream campaign above is complete and published. This amendment adds one further frozen question, using the identical timing boundary, aggregation and process count, and it changes nothing about the cells already collected.

Question: the FIA task issues K descriptors in `ComputeMm1` and V descriptors in `ComputeMm2`. The scalar pipe issues ahead, and each stream owns its own L1 slot pair, so MTE2 may hold a descriptor from one stream in flight while MTE1 consumes the other's. The single-stream probe cannot distinguish that from a strictly serial descriptor chain. Cell: `streams` in `{1, 2}` at geometry `rows=256, columns=128, source_row_stride_elements=1024`, `active_aic_groups` in `{8, 16, 24}`, `descriptor_count` 64, three fresh processes. With `streams=2` the probe uses four L1 slots and two disjoint source streams, alternating descriptors and draining each on its own `MTE2_MTE1` event; the total descriptor count and the total bytes are held equal to the `streams=1` cell.

Declared interpretation, fixed before collection: if the per-descriptor service at `streams=2` is within 5% of `streams=1`, the drained chain is serial and the production gap is not explained by MM1/MM2 stream overlap; that candidate is then rejected and recorded in the falsified list. If `streams=2` is faster by more than 5%, the overlap is real and its measured depth becomes the candidate mechanism. No Forward change is made on the basis of this cell alone, and no coefficient is chosen by evaluation error either way.

## Publication boundary

A passing campaign publishes only a count-free drained per-descriptor ND2NZ service over the exact measured `(geometry, active_aic_groups)` domain, with `dram_cold_disjoint` residency and a per-descriptor MTE2-to-MTE1 completion edge. It does not authorize interpolation across geometry or across group counts, does not replace any existing per-stage service, and does not by itself justify any change to the resident/hit path. Forward must remain fail-closed outside the measured domain.
