"""Portable shim for the POSIX ``resource`` module.

The Serving simulation reports peak RSS via ``resource.getrusage``.  Windows
has no such module, so this shim exposes the same attribute surface
(``getrusage``, ``RUSAGE_SELF``, ``RUSAGE_CHILDREN``) and returns peak working
set bytes from ``GetProcessMemoryInfo`` instead of ru_maxrss.

On POSIX the real module is re-exported unchanged, so callers keep the
original semantics (``ru_maxrss`` in KiB on Linux).
"""

from __future__ import annotations

import sys

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    RUSAGE_SELF = 0
    RUSAGE_CHILDREN = -1

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    _COUNTERS_BYTES = ctypes.sizeof(_ProcessMemoryCounters)

    class _RUsage(ctypes.Structure):
        _fields_ = [
            ("ru_utime", ctypes.c_double),
            ("ru_stime", ctypes.c_double),
            ("ru_maxrss", ctypes.c_longlong),
        ]

    def _current_process_peak_rss_kib() -> int:
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            counters = _ProcessMemoryCounters()
            counters.cb = _COUNTERS_BYTES
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            ok = psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                _COUNTERS_BYTES,
            )
            if not ok:
                return 0
            return int(counters.PeakWorkingSetSize) // 1024
        except OSError:
            return 0

    def getrusage(who: int = RUSAGE_SELF) -> _RUsage:
        if who not in (RUSAGE_SELF, RUSAGE_CHILDREN):
            raise ValueError(f"invalid who value: {who}")
        usage = _RUsage()
        usage.ru_utime = 0.0
        usage.ru_stime = 0.0
        usage.ru_maxrss = (
            _current_process_peak_rss_kib() if who == RUSAGE_SELF else 0
        )
        return usage

else:
    from resource import (  # noqa: F401
        RUSAGE_CHILDREN,
        RUSAGE_SELF,
        getrusage,
    )
