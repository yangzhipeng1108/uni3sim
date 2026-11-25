# EngineCore host-round cadence probe (ascend_910b)

This independent mechanism microbenchmark uses real vLLM 0.18 components inside the 910B container to reproduce one EngineCore busy-loop host round. It measures admission-loop host segments not covered by existing Host profiles. Outputs supply host cadence parameters in seconds; device-stub duration is calibration evidence only and never enters Forward.

## Mechanism

`engine_core_host_round_runner.py` constructs a real `AsyncScheduler` (`async_scheduling=True`, matching 910B Serving), a two-deep `step_with_batch_queue` mirror, nonblocking `collective_rpc`, a single-thread `ThreadPoolExecutor` WorkerAsyncOutput mirror, the `synchronize_input_prep` event barrier, ACLGraphWrapper pre-replay `current_stream().synchronize()` barrier semantics, and the copy-stream D2H + tolist output-wakeup chain in container `<ascend-container>`. Device work is a fixed mm-chain stub: captured NPUGraph replay for the graph lane, individual mm operations for eager. Stub duration is request-independent, so dependence on request count R belongs to host processing by construction. `_RoundTimings` records individual segments in nanoseconds.

Graph submit = `prep_sync + h2d + pre_replay_sync + sample_submit + thread_submit + poll + queue_decision`; wake = `wake_block + d2h_event_sync + tolist + put`. Eager submit omits `pre_replay_sync`. Existing profiles already price four excluded segment classes: SchedulerHostProfile prices `plan` and `update`; host_runtime_prepare prices `state_host` (approximately model_runner_input_prepare); HostLaunchProfile prices `submit_lane` (per-operation enqueue approximately 9.8µs/op, consistent with eager 9.35µs/op). Exclusion prevents double counting.

## Reproduction

Run on `<capture-host-0118b599d5e7>`, in container `<ascend-container>`, from repository root `<repository-root>`, subject to the root hardware-access contract:

`/usr/local/python3.11.14/bin/python microbench/ascend_910b/runtime/engine_core_host_round/engine_core_host_round_runner.py --output-dir profiler_output/microbench/ascend_910b/engine_core_host_round/<run_dir> --device-id 1 --baseline-commit <full-commit-sha> --expected-source-sha256 <runner-sha256> --container-name <ascend-container> --model-path <model-path>`

Matrix: graph R∈{1,8,32,64,120,128} × stub {small (2-op), large (64-op)}; eager R∈{1,32,120,128} at ops700, plus ops∈{64,256} at R=32. Each cell uses 30 warmup and 200 measured rounds. The manifest verifies runner SHA256 and status, failing closed.

## Parameters (run4, sole engine-profile source)

`configs/host_timing/engine_core_host_round_profiles/ascend_910b.json` records the parameters. Function forms were fixed from mechanism and segment evidence: submit segments were flat in R, and slopes changing sign across runs indicated noise; wake slopes were positive in all three runs, consistent with per-request output put/tolist.

- Graph submit: constant 526.898µs; graph wake: 9.9499µs + 10.928ns/req.
- Eager submit: constant 438.20675µs; eager wake: 8.9607µs + 18.721ns/req.

Pure decode selects graph; prefill/mixed selects eager. At qps64, 1071/1098 observed steps were FULL and pure decode. Six pure-decode steps used NONE, bounding misclassification impact at approximately 6×80µs, considered negligible in the original analysis.

## Replication history

- run2 (2026-09-23, after correcting fsync artifacts; runner SHA256 `4e3b3cc…`, R≤120): engine-chain results and comparisons with run3/run4 are in `references/run2_*`.
- run3 (2026-09-24, same source/command): graph-submit median 551µs versus run2 529µs (+4%), wake 10.6 versus 9.8µs, eager submit 487 versus 446µs. Median segment differences were within host jitter, with consistent slope signs.
- run4 (2026-09-24, runner SHA256 `7b6ad67…`, matrix extended to R=128): parameter source; R≤120 medians agreed with run2 (graph submit 526.9 versus 525.9µs).

`references/run{2,3,4}_{summary,manifest,trials}` contains raw artifact copies from the three runs. Their manifests and the engine-profile JSON `artifacts` list bind SHA256 identities.
