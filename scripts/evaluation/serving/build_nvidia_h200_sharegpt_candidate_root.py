#!/usr/bin/env python3
'Build one H200 ShareGPT candidate dataset root for one vLLM version.\n\nEvery case of the 15-case H200 matrix is converted by\n``build_nvidia_h200_sharegpt_candidate.py`` (unchanged converters, compact\nschema v5) and copied into ``<output-root>/vllm_<version>/nvidia_h200/<model>/\n<case>/``, the layout the Serving evaluators read as ``--dataset-root``.\nOutput never goes to ``serving_dataset/``.\n\nObserver-effect gate (the A800 canonical rule): a case whose own\ntrace-off/trace-on gate passes keeps it (``measured_for_this_case``); every\nother case inherits the gate of one representative case of the same hardware\nand runtime (``inherited_from_representative_case``).  A800 uses\n``qwen3-8b/tp1_qps4`` for all 14 other cases.  The representative is fixed per\nvLLM version in ``_REPRESENTATIVE``: 0.11.0 keeps the A800 case; under 0.30.0\nthat case fails its own gate, so the other TP1 qps4 case whose own gate passes\n(``llama-3.1-8b/tp1_qps4``) is used.  A representative that does not pass its\nown gate fails the build.\n\n``--alignment-stage-root`` reuses already validated stage-1 cases\n(``<root>/<model>/<case>/alignment_stage``) through hard links; otherwise the\nraw ``raw/ShareGPT-*`` tree of each case is converted again.'
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.serving_dataset.nvidia_h200_sharegpt_candidate_root import main

if __name__ == "__main__":
    raise SystemExit(main())
