#!/usr/bin/env bash
# Capture pure-decode windows of the real vLLM DSV3-L32 TP4 server on 4x H200.
# Same server, nsys options and load generator as nsys_decode_capture.sh, but
# each concurrency sends exactly C requests (one wave), and the ~1.5 s window
# is taken after all C prefills have drained and before the shortest request
# (>= 819 output tokens) finishes, so every captured step is decode-only.
# Run from the operator-selected capture directory.  Evaluation only.
# Usage: VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random \
#            nvidia_h200_nsys_pure_decode_capture.sh <out tag> <C>...
set -euo pipefail
B=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}; M=$B/h200_microbench
RUNTIME_PYTHON=${MIDLAYER_H200_RUNTIME_PYTHON:?set MIDLAYER_H200_RUNTIME_PYTHON to the selected runtime executable}
OUT=$M/nsys/$1; shift; mkdir -p $OUT
export HOME=$M/home XDG_CACHE_HOME=$B/cache/xdg HF_HOME=$B/cache/hf HF_HUB_OFFLINE=1
export PATH=$B/tools/nsys_2025.6.1/bin:$PATH
nsys profile -t cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=repeat-shutdown:$# \
  --trace-fork-before-exec=true -o $OUT/window_%n --force-overwrite true \
  $M/bin/serve_dsv3_l32.sh --profiler-config '{"profiler": "cuda"}' > $OUT/server.log 2>&1 &
NSYS=$!
until grep -q "Application startup complete" $OUT/server.log; do sleep 10; done
nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw --format=csv -lms 100 > $OUT/telemetry.csv &
TEL=$!
for C in "$@"; do
  "$RUNTIME_PYTHON" -m vllm.entrypoints.cli.main bench serve --backend vllm \
    --model dsv3-l32 --tokenizer $B/models/DeepSeek-V3-L32 --trust-remote-code --port 8011 \
    --dataset-name random --random-input-len 7373 --random-output-len 922 --random-range-ratio 0.1111111111 \
    --num-prompts $C --max-concurrency $C --ignore-eos > $OUT/load_c$C.log 2>&1 &
  LOAD=$!
  # vllm bench serve first runs one full test prompt, so wait on the server
  # metrics until exactly C requests run with none waiting (the wave is
  # admitted and every prefill is scheduled), then 2 s for the last prefill
  # chunk; the shortest request still decodes >= 819 steps after that.
  until curl -s --max-time 5 http://localhost:8011/metrics | awk -v c=$C '
      /^vllm:num_requests_running/ {r += $NF} /^vllm:num_requests_waiting/ {w += $NF}
      END {exit !(r == c && w == 0)}'; do sleep 0.5; done
  sleep 2
  echo "C=$C start $(date +%T.%N)" >> $OUT/windows.txt
  curl -s --max-time 120 -X POST http://localhost:8011/start_profile
  sleep 1.5
  curl -s --max-time 120 -X POST http://localhost:8011/stop_profile || true
  echo "C=$C stop $(date +%T.%N)" >> $OUT/windows.txt
  wait $LOAD || true
  sleep 5
done
kill $TEL
kill -INT $NSYS || true
wait
