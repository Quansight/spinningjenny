from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar

import pytest

from spinningjenny import ThreadPoolExecutor, thread_local_pool

TACH: ContextVar[int] = ContextVar("Tachyon readings")


@contextmanager
def set_tach(value: int) -> Generator[None, None, None]:
    """Nicer API for Python < 3.14."""
    token = TACH.set(value)
    try:
        yield
    finally:
        TACH.reset(token)


@pytest.mark.parametrize("executor_factory", [ThreadPoolExecutor, thread_local_pool])
def test_tasks_run_with_correct_contextvars(
    executor_factory: Callable[[int], ThreadPoolExecutor],
) -> None:
    """
    A function passed to the executor is run with the context in which the map
    happened.

    Python's built-in ``ThreadPoolExecutor`` caches the context on creation,
    rather than per-map, which is a problem.
    """

    def get(_):
        return TACH.get()

    # Check both reuse of same result of factory, and calling factory multiple
    # times for benefit of thread-local version.
    for value in [123, 456]:
        executor = executor_factory(2)

        with set_tach(value):
            assert get(None) == value
            assert list(executor.map(get, range(10))) == [value] * 10

        with set_tach(value + 1):
            assert get(None) == value + 1
            assert list(executor.map(get, range(10))) == [value + 1] * 10


@pytest.mark.parametrize("executor_factory", [ThreadPoolExecutor, thread_local_pool])
def test_contextvars_interleaved(executor_factory: Callable[[int], ThreadPoolExecutor],
) -> None:
    """Different map calls preserve their contextvar context."""
    def get(_):
        return TACH.get()

    # Hopefully do enough maps in parallel that they can potentially run
    # interleaved:
    result_iterators = []
    executor = executor_factory(4)
    for value in range(1000):
        with set_tach(value):
            result_iterators.append(executor.map(get, range(1000)))

    for value, it in enumerate(result_iterators):
        assert list(it) == [value] * 1000
