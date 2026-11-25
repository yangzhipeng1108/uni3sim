# Ascend 910B helper dispatch and typed services

This extension separates two different evidence needs. The original source-0.18 Llama Serving case first refuses `vllm_embedding_id_widen`: its INT32-to-INT64 dispatch is unbound. The synthetic cast runner observes that library dispatch without deriving any rate from the Cast duration. Separately, opcodes 48–51 prepare independent FP32 Compare/Select services that may be useful after a matching sampler program is established. Neither experiment implements or times an entire argmax, and neither alone accepts a complete Serving request.

## INT32-to-INT64 dispatch

`integer_cast_dispatch_runner.py` uses independent contiguous INT32 inputs of 1, 127, 128, 129, 2048 and 32768 elements, including INT32_MIN/MAX and negative values. Each NPU result is copied back after synchronization and every element is checked against the CPU integer reference. No model, checkpoint, weights, routing, reduction or timing API is used. The declared runtime versions must exactly match the installed package metadata. Capturing the installed Torch-NPU runtime does not by itself prove that it is the intended source-0.18 production provider.

CPU-only preparation, without Torch/ACL imports or accelerator queries:

```bash
python microbench/ascend_910b/compute/vector_pipe/integer_cast_dispatch_runner.py --cpu-reference-only
```

The hardware executor first establishes exclusive ownership of an idle physical device, fixes the source snapshot and uses the registered remote task directory for all outputs and caches. In that directory, run the following through the prescribed container after sourcing the CANN environment. Replace the version strings with their exact `importlib.metadata.version` output; do not accept a mismatched environment by changing a production source identity.

```bash
msprof --output="$TASK_DIR/cast_profile" --application="python $TASK_DIR/source/microbench/ascend_910b/compute/vector_pipe/integer_cast_dispatch_runner.py --device-id 5 --baseline $BASELINE --expected-torch-version 2.9.0+cpu --expected-torch-npu-version 2.9.0.post1+gitee7ba04 --output $TASK_DIR/cast_validation.json"
python "$TASK_DIR/source/microbench/ascend_910b/compute/vector_pipe/integer_cast_dispatch_runner.py" --project-op-summary "$OP_SUMMARY_CSV" --output "$TASK_DIR/cast_discrete.json"
```

`cast_validation.json` records exact runtime identity and validation. `cast_discrete.json` retains only recognized operation names/types, block/mix dimensions, tiling key, shape/dtype/layout and task/stream identity fields, with the original CSV path/hash. A missing recognized operation name or block-dimension column fails. All raw msprof output remains intact, including its durations, under the registered raw campaign; those durations are never read by the projection and are ineligible as Forward rates. This is observation of synthetic helper dispatch, not the original Serving launch. Matching provider, shape applicability, tiler/compiled program, scratch and physical service mapping still need proof before production binding.

## Independent FP32 predicate and select streams

The CANN 8.5.1 API is independently visible in the version-matched plugin `apply_top_k_top_p_custom.h::GetKthResult`: `Compare` writes a `LocalTensor<uint8_t>` mask and `Select` consumes it with `VSEL_TENSOR_TENSOR_MODE`. The installed CANN declarations are under `aarch64-linux/asc/include/basic_api/kernel_operator_vec_cmpsel_intf.h`; implementation for this SoC is `dav_c220/kernel_operator_vec_cmpsel_impl.h`. This extension uses the explicit 64-FP32-lane repeat API with contiguous `BinaryRepeatParams{1,1,1,8,8,8}`. CANN compilation and selected instruction validation are required; a source API call is not assumed to be one ISA instruction.

| Opcode | Source invocation measured | Input/output domain |
| --- | --- | --- |
| 48 | FP32 `Compare(...,GT)` | FP32 pair to packed uint8 mask |
| 49 | FP32 `Compare(...,EQ)` | FP32 pair to packed uint8 mask |
| 50 | FP32 `Compare(...,NE)`, labeled `compare_fp32_ordered_ne` | Ordered inequality on the declared synthetic inputs; FP32 pair to packed uint8 mask |
| 51 | FP32 tensor/tensor `Select` | Precomputed GT mask, two FP32 tensors |

Each timed iteration is one vector invocation covering `elements/64` repeats. The loop alternates two destination buffers; interval setup, input construction, the Select-only mask producer, Compare-result decoding and GM validation stores are outside timing. The interval includes loop/dispatch overhead and completion of `PIPE_V`; it is not pure predicate latency or a lexicographic FP32/INT64 pair-comparator rate. The existing same-AIV `GetSystemCycle` domain uses the fixed 50 MHz counter, 20 ns/tick. Short/long intervals use N and 4N invocations and retain raw per-AIV counters. No cross-AIV timestamps are subtracted. Shape/concurrency observations remain separate; no extrapolated rate or fitted production offset is emitted.

All 16 synthetic lane patterns include ordinary order/ties, positive and negative zero, positive and negative infinity, and quiet NaNs on either/both sides. Every element of both destinations is checked after timing; the host additionally checks all 16 lane patterns independently. Incorrect NaN semantics or mask layout fails the smoke rather than silently changing the expected result. This stream does not test INT32 or INT64 comparison, index conversion, reduction topology, inter-core communication or actual sampler dispatch.

The initial opcode-50 failure remains retained under `experiment_logs/evaluation/operator_support/three_hardware_closure/ascend_910b/opcode52_3d32d5fc_failed`. Its original test incorrectly expected C++ unordered `!=`. The independent [per-lane device diagnostic](../../../../experiment_logs/evaluation/operator_support/three_hardware_closure/ascend_910b/fp32_ne_lane_diagnostic.json) records all 16 output values: the exact positive quiet-NaN bit pattern `0x7fc00001` on the left, right or both sides returns false, while unequal numeric/infinite pairs return true. Opcode 50 therefore validates ordered inequality for these declared inputs. The kernel, host and CPU reference now agree on that contract; the measured instruction stream and timer are unchanged. This evidence does not establish signaling-NaN behavior, other NaN payloads or a general NaN-detection service. In particular, its rate must not be used to price `isnan(x)` as `NE(x,x)`.

Build and CPU reference:

```bash
python microbench/ascend_910b/compute/vector_pipe/vector_opcode_runner.py --typed-reference-only
cmake -S "$TASK_DIR/source/microbench/ascend_910b/compute/vector_pipe" -B "$TASK_DIR/build_vector" -DASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1 -DSOC_VERSION=Ascend910B2
cmake --build "$TASK_DIR/build_vector" -j 4
```

First device smoke, only after root freezes the exact source snapshot and the executor rechecks card ownership:

```bash
ASCEND_RT_VISIBLE_DEVICES=5 "$TASK_DIR/build_vector/vector_opcode_bench" --batch 48 64 1 4096 49 64 1 4096 50 64 1 4096 51 64 1 4096
```

Only after smoke, the existing runner accepts `--campaign fp32_predicate_select`, with elements 64/512/4096 and block dimensions 1/48. It preserves the old campaigns, CLI and 16-column raw schema. The report accepts this campaign and summarizes repeated observations in ns/vector-invocation/AIV, keeping `production_argmax_bound=false`, `single_isa_instruction_rate_verified=false` and `mechanism_publishable=false`. Independent scalar/INT64 services and a verified production-program mapping remain separate gates. GPU execution and publication are owned by the root/hardware executor; this preparation does not change HWSpec or any production parameter.
