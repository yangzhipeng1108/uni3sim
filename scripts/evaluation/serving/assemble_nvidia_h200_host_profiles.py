#!/usr/bin/env python3
"Assemble the four H200 Host timing profiles for one vLLM version.\n\nOne command re-assembles every H200 Host profile of ``--vllm-version`` from the\nHost profile runs of ``microbench/nvidia_h200/runtime/host_profiles/\nrun_nvidia_h200_host_profiles.sh`` and the H200 ShareGPT candidate root of the same\nversion (Scheduler Host samples):\n\n* ``configs/host_timing/host_launch_profiles/nvidia_h200_<tag>.json``\n* ``configs/host_timing/host_request_path_profiles/nvidia_h200_<tag>.json``\n* ``configs/host_timing/host_runtime_prepare_profiles/nvidia_h200_<tag>_eager.json``\n* ``configs/host_timing/scheduler_host_profiles/nvidia_h200_<tag>.json``\n\nwith ``<tag>`` = ``vllm011`` / ``vllm030``, their committed references under\n``microbench/nvidia_h200/runtime/`` and an assembly record under\n``experiment_logs/h200_serving_eval/profiles/``.  ``--host-launch-run`` and\n``--host-mechanism-run`` may be the same directory.  A run measured under a\ndifferent vLLM version than ``--vllm-version`` is accepted only with\n``--substitution-reason``; the reason is written into every affected\nreference.  When a clean run of the target version exists, pass it as both\nruns and drop the reason.\n\nThe Scheduler Host profile is fitted by ``scheduler_host_profile.py`` (QPS 16\nheld out) on the candidate root's own Scheduler traces plus, when the mechanism\nrun was measured under the target version, its native prefill sweep\n(``rep1`` trains, ``rep2`` is held out at QPS 16)."
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.serving_dataset import nvidia_h200_host_profiles as h200
from evaluation.serving.experiments.nvidia_h200_host_profile_assembly import main

if __name__ == "__main__":
    raise SystemExit(main(profile_pipeline=h200))
