import functools
import logging
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

log = logging.getLogger("findex.timing")

P = ParamSpec("P")
R = TypeVar("R")


def timed(fn: Callable[P, R]) -> Callable[P, R]:
    """Decorator that logs execution time of a function while preserving metadata."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            log.info("[timed] %s took %.2f ms", fn.__qualname__, dt_ms)

    return wrapper
