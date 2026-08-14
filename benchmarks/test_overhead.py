from concurrent.futures import ThreadPoolExecutor as OrigExecutor

import pytest

from spinningjenny import ThreadPoolExecutor as SpinExecutor, thread_local_pool
from spinningjenny._testing import run_for_usecs


def spin_100us(_x):
    run_for_usecs(100)


def spin_10us(_x):
    run_for_usecs(10)


def noop(_x):
    pass


class OrigExecutor(OrigExecutor):
    def map(self, *args, buffersize=None, in_order=True):
        return super().map(*args, buffersize=None)


class Sequential:
    def __init__(self, n_cpus):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def map(self, func, args, buffersize=None, in_order=True):
        return (func(arg) for arg in args)


@pytest.mark.parametrize("buffersize", [None, 100])
@pytest.mark.parametrize("function", [noop, spin_10us, spin_100us])
@pytest.mark.parametrize(
    "executor_factory", [OrigExecutor, SpinExecutor, thread_local_pool, Sequential]
)
@pytest.mark.parametrize("in_order", [True, False])
def test_one_thousand_calls(benchmark, buffersize, function, executor_factory):
    def run():
        with executor_factory(8) as executor:
            result = executor.map(
                function, range(1000), buffersize=buffersize, in_order=in_order
            )
            return list(result)

    result = benchmark(run)
    assert len(result) == 1000
