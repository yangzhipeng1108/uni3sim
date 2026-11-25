# A800 typed register capture

This wrapper reuses `microbench/nvidia_h200/compute/typed_register_primitives.cu` with `MIDLAYER_TYPED_A800`, targeting A800-SXM4-80GB SM80/108 SM. Without that macro, the existing H200 six primitives, guard and runner behavior remain unchanged. The A800 build includes the original FP32 gt/eq/isnan+select, signed64 lt+select, unsigned64 gt+select and b32 XOR shuffle, two PRMT streams, and one five-stage b32 SHFL.DOWN stream. These are independent register mechanisms, without model tensors, routing, argmax, reduction or production-duration input. No measured number is automatically written into HWSpec or Forward.

`prmt_rotate16` is exact `PRMT Rdst,Rsrc,0x1032,RZ`, swapping two 16-bit halves, with period two. `prmt_bf16_expand` is exact `PRMT Rdst,RZ,0x5410,Rsrc`, mapping the source's low 16 bits into the result's high 16 bits, as the SM80 BF16→FP32 data path. **The latter's dependent recurrence reaches zero after two operations**: the CPU expected-value function explicitly models this, and the receipt discloses it. Its repeated static PRMT instructions are still required; this stream does not prove a nonzero-data invariant, input-independent timing or a production-copy throughput. The rotation stream provides a separate noncollapsing dependency comparison. A pure predicate and a compare+select bundle are distinct units; neither is automatically a full lexicographic `(value,index)` comparator.

`shuffle_down32_1_2_4_8_16` is one dependent five-instruction bundle with a full-warp member mask, clamp lane 31 and offsets 1, 2, 4, 8 and 16. Each instruction consumes the preceding shuffle result. The CPU reference models valid source lanes and the clamp behavior at every stage: the five stages contain 31+30+28+24+16=129 source-active lane edges, while the hardware unit remains five full-warp instruction issues. A complete bundle maps every destination to its warp's original lane 31, so repeated bundles retain the cross-lane instruction dependency but have data collapsed after the first bundle; this limitation is part of the mechanism scope. This stream matches only the source SHFL.DOWN shape and is not an argmax, reduction, production duration or automatically publishable rate.

The A800-only integer inputs include lanes with equal high words and pivot±1 low words, and lanes with different high words. Runtime two-word pivots prevent sign-only specialization. All lane/chain output values are validated against the CPU reference after every launch. One/four independent chains execute 31 statically unrolled bundles per outer loop. Static gates require all 18 primitive/chain programs; the new down mode retains exactly five SHFL.DOWN instructions per bundle at offsets 1/2/4/8/16, while the original XOR and PRMT modes retain their previous exact counts. Full-width integer comparisons and no stack/shared/local allocation or timed memory/helper calls remain required. Substitution of SHF for PRMT is rejected. Failed preparation is not a usable capture artifact.

## Offline preparation

The root executor must first freeze the source/runner and record the exact commit. Preparation compiles/dumps code and runs the portable reference; it does not query or execute a GPU. Raw preparation/capture output must be under the worktree's registered `profiler_output/` campaign; CPU-only ephemeral checks may use `/tmp`.

```bash
python microbench/nvidia_a800/compute/typed_register_primitives_runner.py \
  --cpu-only --output /tmp/a800_typed_cpu --baseline-commit COMMIT40
python microbench/nvidia_a800/compute/typed_register_primitives_runner.py \
  --prepare-only --device 7 --baseline-commit COMMIT40 \
  --output profiler_output/CAMPAIGN/prepare
```

Preparation uses CUDA 12.2 at `/usr/local/cuda/bin/nvcc` and `cuobjdump`, overridable by explicit paths. It records compile commands, versions, shared-source/runner/binary hashes, CPU results, timed SASS inventory and register resources. Noncanonical preparation receipts may record the exact source snapshot; tests do not hardcode those identities.

## Owned-device smoke and matrix

Device idleness/ownership is a separate executor preflight. The wrapper requires `--device 7`, sets `CUDA_VISIBLE_DEVICES=7`, and the binary validates its internal ordinal zero as A800/SM80/108 SM. No GPU clocks are changed or assumed to be H200's clock. Capture verifies the preparation's hashes and exact baseline and refuses changed code/binaries.

```bash
python microbench/nvidia_a800/compute/typed_register_primitives_runner.py \
  --smoke --device 7 --baseline-commit COMMIT40 \
  --prepared profiler_output/CAMPAIGN/prepare \
  --output profiler_output/CAMPAIGN/smoke \
  --primitive shuffle_down32_1_2_4_8_16
python microbench/nvidia_a800/compute/typed_register_primitives_runner.py \
  --matrix --trial 1 --device 7 --baseline-commit COMMIT40 \
  --prepared profiler_output/CAMPAIGN/prepare \
  --output profiler_output/CAMPAIGN/trial1 \
  --primitive shuffle_down32_1_2_4_8_16
```

Repeat the same matrix with `--trial 2` and `--trial 3` into separate directories. With the shown `--primitive` selection, smoke has four cells and each matrix trial has 12 cells; the receipt records the exact captured subset. Omitting `--primitive` preserves the prior full behavior: smoke has 36 cells and each matrix trial has 108 cells across all nine mechanisms. Different trials repeat the same predeclared matrix rather than selecting rates by production error. Run serially on this single owned GPU; independent other-hardware tasks may run in parallel.

Each CSV row records a warp's same-SM begin/end `clock64`, actual SMID and SMID namespace, geometry, source bundle count, checksum and validation status. The receipt groups intervals only by actual SMID within the same cell/sample. It reports cycles per warp bundle and observed warp/lane bundles per SM cycle from that SM's envelope, with no cross-SM absolute timestamp subtraction. For the down mode, one recorded bundle means the five dependent SHFL.DOWN instructions and 129 source-active lane edges defined above. Clock telemetry records actual SM/memory clocks, UUID, temperature, power and utilization every 200 ms; no artificial frequency lock or assumed peak is applied. Envelope results retain scheduler and loop/timer overhead and are observations rather than a proof of peak instruction issue capacity. Source-program mapping and physical rate selection remain a separate reviewed Forward binding.

Preserve raw lifecycle, frozen commit, source/binary hashes, compile/static logs, device identities and telemetry. CPU-only or static success is not GPU correctness. Successful GPU checks are not normal-capacity, provider, complete sampler or Serving acceptance.
