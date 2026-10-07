import threading
from concurrent.futures import ThreadPoolExecutor as OrigExecutor
from time import time_ns

import pytest
from joblib import Parallel, delayed
from sklearn.utils.parallel import Parallel as SkParallel
from sklearn.utils.parallel import delayed as SkDelayed

from spinningjenny import ThreadPoolExecutor as SpinExecutor
from spinningjenny import thread_local_pool
from spinningjenny._testing import run_for_usecs


def spin_100us(_x):
    run_for_usecs(100)


def spin_10us(_x):
    run_for_usecs(10)


def noop(_x):
    pass


def spin_nanos(nanos):
    start = time_ns()
    while time_ns() - start < nanos:
        pass


class OrigExecutor(OrigExecutor):
    def map(self, *args, buffersize=None, return_in_order=True):
        return super().map(*args, buffersize=buffersize)


class _LocalOrigExecutorStorage(threading.local):
    """Store and retrieve a cached thread-local ``OrigExecutor``."""

    pool = None
    n_threads = None

    def get(self, n_threads: int) -> OrigExecutor:
        """Get or create a cached pool, if the number of threads matches."""
        if n_threads == self.n_threads and self.pool is not None:
            return self.pool

        self.pool = OrigExecutor(n_threads)
        self.n_threads = n_threads
        return self.pool


_LOCAL_ORIG_EXECUTOR = _LocalOrigExecutorStorage()


def thread_local_orig_executor(n_threads: int) -> OrigExecutor:
    return _LOCAL_ORIG_EXECUTOR.get(n_threads)


class Sequential:
    def __init__(self, n_cpus):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def map(self, func, args, buffersize=None, return_in_order=True):
        return (func(arg) for arg in args)


class Joblib:
    def __init__(self, n_cpus):
        self.n_cpus = n_cpus

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def map(self, func, args, buffersize=None, return_in_order=True):
        func = delayed(func)
        return Parallel(self.n_cpus, backend="threading")(func(arg) for arg in args)


class Sklearn(Joblib):
    def map(self, func, args, buffersize=None, return_in_order=True):
        func = SkDelayed(func)
        return SkParallel(self.n_cpus, backend="threading")(func(arg) for arg in args)


@pytest.mark.parametrize("buffersize", [None, 100])
@pytest.mark.parametrize("function", [noop, spin_10us, spin_100us])
@pytest.mark.parametrize(
    "executor_factory",
    [
        OrigExecutor,
        thread_local_orig_executor,
        SpinExecutor,
        thread_local_pool,
        Sequential,
        Joblib,
        Sklearn,
    ],
)
@pytest.mark.parametrize("return_in_order", [True, False])
def test_one_thousand_calls(
    benchmark, buffersize, function, executor_factory, return_in_order
):
    def run():
        executor = executor_factory(8)
        result = list(
            executor.map(
                function,
                range(1000),
                buffersize=buffersize,
                return_in_order=return_in_order,
            )
        )
        del executor
        return list(result)

    result = benchmark(run)
    assert len(result) == 1000


# Keep function-based benchmark grouping available across the whole suite.
@pytest.mark.parametrize("function", [spin_nanos])
@pytest.mark.parametrize("return_in_order", [True, False])
def test_adversarial_delays(benchmark, function, return_in_order):
    """
    A message execution pattern that demonstrates when out-of-order execution
    is helpful.
    """
    sleep_nanos = ([1_000_000] + [1_000] * 99) * 100
    sleep_nanos.reverse()

    def run():
        with SpinExecutor(4) as executor:
            list(
                executor.map(
                    function,
                    sleep_nanos,
                    buffersize=100,
                    return_in_order=return_in_order,
                )
            )

    benchmark(run)
