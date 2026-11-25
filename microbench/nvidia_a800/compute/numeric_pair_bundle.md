# A800 independent comparator/pair-select bundle

`numeric_pair_bundle.cu` contains an independent register mechanism, not an ArgMax benchmark. Inputs are synthetic FP32 values and two signed INT64 indices. It includes greater/less values, finite ties and signed zeros; index patterns cover equal high words with low-word ordering, unequal high words, signed boundaries and ordinary small indices. All GPU lanes/chains are compared to the scalar CPU reference. The recurrence repeatedly selects the maximum pair against a fixed second pair and is **idempotent after one application**; this is disclosed, not presented as a period-two stream or a proof of operand-independent timings. CPU-only reference also checks four NaN semantic cases; GPU smoke currently supplies finite synthetic inputs.

There are two explicit scopes:

- `numeric`: the decoded source's numeric boolean materialization plus FP32 FSEL and two INT64 word selects. It excludes NaN classification and source branch/reconvergence work. A bundle has 11 computational instructions. CUDA 12.2 complements the final `ISETP.EQ` plus positive selects to `ISETP.NE` plus inverted selects; this is an explicit equivalent predicate/consumer transformation, not a missing comparison. No rate is split into pure-predicate, select or INT64-word services.
- `classified` (default): the complete independent pair comparator retains `FSETP.GTU(abs(a),+INF)`, actual branch, numeric path, NaN path, `BSSY`/`BSYNC` and FP32/INT64 pair selection. The retained alternate NaN path checks `abs(b)>INF` and full-width signed ordering. Its captured inputs are finite synthetic states; this is neither user acceptance of a production non-NaN domain nor a proof of actual model logits. The unit includes classification/control rather than borrowing an unmeasured single predicate service.

Both scopes execute 1/4 independent chains, 31 statically unrolled bundles per outer iteration. The static gate checks the relevant FP32 predicates, two-word signed ordering, bool materialization, FSEL and double SEL counts, classified NaN/control paths, zero stack/shared/local allocation and absence of timed memory/reduction/shuffle. Limited loop/timer overhead and compiler register epilogue remain within the measured envelope. Source instruction-site counts are not dynamic counts; actual Torch thread/warp/block reduction recipes and branch applicability must be bound separately. In particular, compiled source sometimes shares a reconvergence scope across comparisons, so this primitive's BSSY/BSYNC ownership cannot be silently multiplied by mathematical N−1. It does not include loads, reduction, shuffle or the whole production operator.

The wrapper reuses the existing A800 CSV/clock helpers, records source/helper/binary identities and refuses changed frozen preparations. Preparation is CPU/static only. Raw output must use the registered `profiler_output/microbench/nvidia_a800/operator_support_typed_20261023/` campaign. The root executor handles GPU 7 idleness/ownership and freezes the files before execution.

```bash
python microbench/nvidia_a800/compute/numeric_pair_bundle_runner.py \
  --cpu-only --baseline-commit COMMIT40 --output /tmp/a800_pair_cpu
python microbench/nvidia_a800/compute/numeric_pair_bundle_runner.py \
  --prepare-only --device 7 --baseline-commit COMMIT40 \
  --output profiler_output/microbench/nvidia_a800/operator_support_typed_20261023/pair_prepare
python microbench/nvidia_a800/compute/numeric_pair_bundle_runner.py \
  --smoke --scope both --device 7 --baseline-commit COMMIT40 \
  --prepared profiler_output/microbench/nvidia_a800/operator_support_typed_20261023/pair_prepare \
  --output profiler_output/microbench/nvidia_a800/operator_support_typed_20261023/pair_smoke
```

Smoke uses chains1/4, one 32-thread block, outer1/2 and one sample (four cells per scope). Optional `--matrix --trial 1` uses chains1/4, geometries `(32,1),(128,108),(256,432)`, outer127/511 and three samples (12 cells per scope); trials2/3 repeat the same fixed matrix in distinct directories. All timestamps are grouped by actual SMID within a cell/sample; never subtract different-SM absolute clocks. GPU identity is checked against physical GPU 7's PCI ID. No GPU clock is changed. Before/after clock/UUID records are retained; an approved SI rate additionally requires adequate clock telemetry at the executor boundary. Neither preparation nor a successful smoke binds Forward or proves actual source merge counts, production-domain applicability, complete Serving, provider or capacity acceptance.
