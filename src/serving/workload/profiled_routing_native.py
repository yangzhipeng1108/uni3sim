import ctypes
import functools
import math
import os
from pathlib import Path
import subprocess
import tempfile


@functools.lru_cache(maxsize=1)
def _library():
    source = Path(__file__).with_name("profiled_routing.c")
    with tempfile.TemporaryDirectory(prefix="midlayer-profiled-routing-") as directory:
        output = Path(directory) / "profiled_routing.so"
        compiler = os.environ["CC"] if "CC" in os.environ else "cc"
        result = subprocess.run((compiler, "-O3", "-std=c99", "-shared", "-fPIC",
            str(source), "-o", str(output)), capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("native profiled routing build failed: " + result.stderr)
        library = ctypes.CDLL(str(output))
    function = library.midlayer_profiled_weighted_counts
    function.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint64))
    function.restype = ctypes.c_int
    return function


def sample(weights, ratios, topk):
    if type(topk) is not int or topk < 1:
        raise ValueError("native profiled routing requires positive integer top-k")
    if (not isinstance(weights, tuple) or not isinstance(ratios, tuple)
            or not weights or len(ratios) % topk or len(weights) < topk
            or any(type(weight) is not float or not math.isfinite(weight) or weight < 0
                for weight in weights)
            or any(type(ratio) is not float or not 0.0 <= ratio < 1.0 for ratio in ratios)):
        raise TypeError("native profiled routing requires validated float weights and ratios")
    function = _library()
    native_weights = (ctypes.c_double * len(weights))(*weights)
    native_ratios = (ctypes.c_double * len(ratios))(*ratios)
    counts = (ctypes.c_uint64 * len(weights))()
    status = function(native_weights, len(weights), len(ratios) // topk,
        topk, native_ratios, counts)
    if status == -2:
        raise ValueError("profiled MoE routing exhausted positive expert weights")
    if status != 0:
        raise RuntimeError("native profiled MoE routing could not allocate its index state")
    result = tuple(counts)
    if sum(result) != len(ratios):
        raise RuntimeError("native profiled MoE routing did not conserve its input assignments")
    return result
