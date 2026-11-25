#!/usr/bin/env bash
# Capture prefill-only windows of the real vLLM DSV3-L32 TP4 server on 4x H200.
# Same server and nsys options as nvidia_h200_nsys_pure_decode_capture.sh.  Four
# closed-loop clients send fixed 7373-token prompts with one output token, so
# every engine step prefills four fresh prompts (29492 tokens, one step at
# --max-num-batched-tokens 32768).  The window starts 3 s into the main run.
# Run from the operator-selected capture directory.  Evaluation only.
# Usage: VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random \
#            nvidia_h200_nsys_prefill_capture.sh <out tag>
set -euo pipefail
B=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}; M=$B/h200_microbench
RUNTIME_PYTHON=${MIDLAYER_H200_RUNTIME_PYTHON:?set MIDLAYER_H200_RUNTIME_PYTHON to the selected runtime executable}
OUT=$M/nsys/$1; mkdir -p $OUT
export HOME=$M/home XDG_CACHE_HOME=$B/cache/xdg HF_HOME=$B/cache/hf HF_HUB_OFFLINE=1
export PATH=$B/tools/nsys_2025.6.1/bin:$PATH
nsys profile -t cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=repeat-shutdown:1 \
  --trace-fork-before-exec=true -o $OUT/window_%n --force-overwrite true \
  $M/bin/serve_dsv3_l32.sh --profiler-config '{"profiler": "cuda"}' > $OUT/server.log 2>&1 &
until grep -q "Application startup complete" $OUT/server.log; do sleep 10; done
nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw --format=csv -lms 100 > $OUT/telemetry.csv &
TEL=$!
"$RUNTIME_PYTHON" -m vllm.entrypoints.cli.main bench serve --backend vllm \
  --model dsv3-l32 --tokenizer $B/models/DeepSeek-V3-L32 --trust-remote-code --port 8011 \
  --dataset-name random --random-input-len 7373 --random-output-len 1 --random-range-ratio 0 \
  --num-prompts 400 --max-concurrency 4 --ignore-eos > $OUT/load_c4.log 2>&1 &
LOAD=$!
# Each request lives one ~0.8 s prefill step, too short for a metrics gate;
# wait for the load generator to leave its single-prompt test run instead.
until grep -q "Starting main benchmark run" $OUT/load_c4.log; do sleep 0.5; done
sleep 3
echo "prefill start $(date +%T.%N)" >> $OUT/windows.txt
curl -s --max-time 120 -X POST http://localhost:8011/start_profile
sleep 1.5
curl -s --max-time 120 -X POST http://localhost:8011/stop_profile || true
echo "prefill stop $(date +%T.%N)" >> $OUT/windows.txt
kill $LOAD || true
kill $TEL
wait || true
