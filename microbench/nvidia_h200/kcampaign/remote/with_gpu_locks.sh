#!/bin/bash
# usage: with_gpu_locks.sh <gpu_csv> <wait_seconds> -- <command...>
# Take the whole-GPU process lock runs/gpu<N>.proc.lock (flock -x) for every
# listed GPU in ascending order, then run the command while holding all of
# them.  Ascending order is the campaign-wide rule so two multi-GPU jobs can
# never deadlock.  A lock that is not free within <wait_seconds> is a hard
# failure (a kernel chain or another instance still owns that GPU); nothing
# runs in that case.  Every GPU use outside launch.sh must go through this.
set -euo pipefail
K=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/h200_microbench/kcampaign
if [ "$#" -lt 4 ] || [ "$3" != "--" ]; then
  echo "usage: with_gpu_locks.sh <gpu_csv> <wait_seconds> -- <command...>" >&2
  exit 2
fi
GPUS_CSV=$1; WAIT=$2; shift 3
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
SORTED=$(printf '%s\n' "${GPUS[@]}" | sort -n | uniq | paste -sd, -)
if [ "$SORTED" != "$GPUS_CSV" ]; then
  echo "with_gpu_locks.sh: gpu list must be ascending and unique, got $GPUS_CSV (want $SORTED)" >&2
  exit 2
fi
fd=200
for g in "${GPUS[@]}"; do
  case "$g" in (''|*[!0-9]*) echo "with_gpu_locks.sh: bad gpu id '$g'" >&2; exit 2;; esac
  eval "exec ${fd}>>$K/runs/gpu${g}.proc.lock"
  echo "[with_gpu_locks] $(date '+%F %T') waiting gpu${g}.proc.lock (<= ${WAIT}s)" >&2
  if ! flock -x -w "$WAIT" "$fd"; then
    echo "[with_gpu_locks] $(date '+%F %T') FAILED to take gpu${g}.proc.lock within ${WAIT}s; aborting (nothing started)" >&2
    exit 3
  fi
  echo "[with_gpu_locks] $(date '+%F %T') holding gpu${g}.proc.lock" >&2
  fd=$((fd + 1))
done
# Lock fds are inherited by the command; the locks drop when it exits.
"$@"
