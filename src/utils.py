__author__ = "Jason C. Klima"


import time

from functools import wraps
from typing import Any, Callable, TypeVar, cast


T = TypeVar("T", bound=Callable[..., Any])


def timeit(func: T) -> T:
    """
    Decorator that prints the runtime of a function after it finishes.

    Args:
        func: A required callable to be timed.

    Returns:
        A callable with the same function signature and return type as `func`.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper function that times `func` and prints its runtime."""
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        dt = t1 - t0
        print(f"The `{func.__name__}` function finished in {dt:.3f} seconds.")

        return result

    return cast(T, wrapper)
