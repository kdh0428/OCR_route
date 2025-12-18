"""Lightweight timing utilities for profiling pipeline latency."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator


@contextmanager
def track(name: str, timings: Dict[str, float]) -> Iterator[None]:
    """Context manager that measures elapsed milliseconds and stores them by name."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings[name] = elapsed_ms


def now_ms() -> int:
    """Return the current time in milliseconds since the epoch."""
    return int(time.time() * 1000)
