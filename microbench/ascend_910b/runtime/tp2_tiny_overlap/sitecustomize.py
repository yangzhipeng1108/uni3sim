"""Load the explicit tiny TP2 timing hook in worker subprocesses."""

import os

if "MID_TP2_TINY_TRACE_DIR" in os.environ:
    import timing_hook  # noqa: F401
