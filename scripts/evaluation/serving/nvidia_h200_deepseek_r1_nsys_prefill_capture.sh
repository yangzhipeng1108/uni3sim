#!/usr/bin/env bash
# Capture one prefill-only nsys window of the real vLLM DeepSeek-R1 server on
# H200 (nvidia_h200_serve_deepseek_r1.sh; MODEL/TP/MTP/PORT and CUDA_VISIBLE_DEVICES are
# passed through).  Same protocol as nvidia_h200_nsys_prefill_capture.sh: four
# closed-loop clients send fixed 7373-token prompts with one output token, so
# every engine step prefills four fresh prompts (29492 tokens, one step at
# --max-num-batched-tokens 32768); the 1.5 s window starts 3 s into the main
# run.  SM clock / power are logged at 100 ms.  Remote copy:
# the operator-selected capture directory.  Evaluation only.
# Usage: VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random MODEL=.. TP=.. MTP=0 PORT=.. \
#            nvidia_h200_deepseek_r1_nsys_prefill_capture.sh <out dir>
set -euo pipefail
B=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}; D=$B/h200_microbench/dsr1_mtpoff
RUNTIME_PYTHON=${MIDLAYER_H200_RUNTIME_PYTHON:?set MIDLAYER_H200_RUNTIME_PYTHON to the selected runtime executable}
OUT=$1; mkdir -p $OUT
export PATH=$B/tools/nsys_2025.6.1/bin:$PATH
echo "MODEL=$MODEL TP=$TP MTP=$MTP PORT=$PORT CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all} routing=${VLLM_MOE_ROUTING_SIMULATION_STRATEGY:-model}" > $OUT/config.txt
nsys profile -t cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=repeat-shutdown:1 \
  --trace-fork-before-exec=true -o $OUT/window_%n --force-overwrite true \
  $D/bin/nvidia_h200_serve_deepseek_r1.sh --profiler-config '{"profiler": "cuda"}' > $OUT/server.log 2>&1 &
NSYS=$!
until grep -q "Application startup complete" $OUT/server.log; do
  kill -0 $NSYS 2>/dev/null || { echo "server exited before startup" >&2; exit 1; }
  sleep 10
done
nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw --format=csv -lms 100 > $OUT/telemetry.csv &
TEL=$!
"$RUNTIME_PYTHON" -m vllm.entrypoints.cli.main bench serve --backend vllm \
  --model dsr1 --tokenizer $MODEL --trust-remote-code --port $PORT \
  --dataset-name random --random-input-len 7373 --random-output-len 1 --random-range-ratio 0 \
  --num-prompts 400 --max-concurrency 4 --ignore-eos > $OUT/load_c4.log 2>&1 &
LOAD=$!
until grep -q "Starting main benchmark run" $OUT/load_c4.log; do sleep 0.5; done
sleep 3
echo "prefill start $(date +%T.%N)" >> $OUT/windows.txt
curl -s --max-time 120 -X POST http://localhost:$PORT/start_profile
sleep 1.5
curl -s --max-time 120 -X POST http://localhost:$PORT/stop_profile || true
echo "prefill stop $(date +%T.%N)" >> $OUT/windows.txt
kill $LOAD 2>/dev/null || true
kill $TEL
for _ in $(seq 120); do kill -0 $NSYS 2>/dev/null || break; sleep 5; done
kill -INT $NSYS 2>/dev/null || true
wait $NSYS 2>/dev/null || true
pkill -P $$ 2>/dev/null || true
# The API server runs under the nsys launcher, not this shell: stop the vLLM
# server of this capture (served model dsr1 on this port) if it outlived nsys.
pkill -f -- "--served-model-name dsr1 .*--port $PORT" 2>/dev/null || true
