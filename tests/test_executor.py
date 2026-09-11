from __future__ import annotations

from collections.abc import Callable, Iterable
from threading import Condition, Lock, RLock
from time import sleep, time_ns
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable, Iterable

from hypothesis import given, strategies as st
import pytest

from spinningjenny import ThreadPoolExecutor
from spinningjenny._testing import run_for_usecs


@pytest.mark.parametrize(
    "func,arguments",
    [
        (lambda x: x + 1, [range(1000)]),
        (lambda a, b: a + b, [range(1, 1001), range(2, 1002)]),
    ],
)
@pytest.mark.parametrize("n_threads", [1, 2, 4])
@pytest.mark.parametrize("in_order,to_list", [(True, list), (False, sorted)])
def test_map_results(
    func: Callable,
    arguments: list[Iterable],
    n_threads: int,
    in_order: bool,
    to_list: Callable[[Iterable], list],
) -> None:
    """
    ``ThreadPoolExecutor.map()`` gives the exact same results as ``map()`` by
    default, or if ``in_order=True`` is passed in.  If ``in_order=False``,
    results may be out of order.
    """
    expected = list(map(func, *arguments))
    with ThreadPoolExecutor(n_threads) as pool:
        actual = pool.map(func, *arguments, in_order=in_order)
        assert not isinstance(actual, list)
        assert expected == to_list(actual)

        # Omitting in_order= is the same as in_order=True:
        actual = pool.map(func, *arguments)
        assert not isinstance(actual, list)
        assert expected == list(actual)


class Resource:
    """
    A resource that can created concurrently, a stand-in for memory.
    """

    def __init__(self, factory: ResourceFactory):
        self.factory = factory

    def __del__(self):
        self.factory._destroy()


class ResourceFactory:
    """Factory for ``Resource`` instances."""

    def __init__(self):
        self.lock = RLock()
        self.max = 0
        self.resources = 0

    def create(self) -> Resource:
        with self.lock:
            self.resources += 1
            self.max = max(self.max, self.resources)
        return Resource(self)

    def _destroy(self) -> None:
        with self.lock:
            self.resources -= 1


@pytest.mark.parametrize("usecs", [0, 10, 100])
@pytest.mark.parametrize("num_threads", [2, 4, 6])
@pytest.mark.parametrize("buffersize", [None, 10, 100])
@pytest.mark.parametrize("in_order", [True, False])
def test_resource_usage(
    usecs: int, num_threads: int, buffersize: None | int, in_order: bool
) -> None:
    """
    The amounts of resources used by the executor should be constained.

    Resources would include memory (if _instantiating_ a task uses a lot of
    memory), or other places where instantiating many tasks in parallel is
    expensive.
    """
    factory = ResourceFactory()

    def task(resource):
        assert isinstance(resource, Resource)
        run_for_usecs(usecs)

    with ThreadPoolExecutor(num_threads) as executor:
        result = executor.map(
            task,
            (factory.create() for _ in range(1000)),
            buffersize=buffersize,
            in_order=in_order,
        )
        assert len(list(result)) == 1000
    # Give it some leeway in case it goes over:
    assert factory.max < 30 * num_threads


class TasksRun:
    """Track how many tasks ran."""

    def __init__(self):
        self.lock = Lock()
        self.ran = 0

    def run(self):
        with self.lock:
            self.ran += 1

    def get_ran(self):
        with self.lock:
            return self.ran


@pytest.mark.parametrize("in_order", [True, False])
@pytest.mark.parametrize("num_threads", [2, 4, 6])
def test_buffersize_limits_execution_when_no_iteration(
    num_threads: int, in_order: bool
) -> None:
    """
    If ``buffersize`` is set, at most ``buffersize + num_threads`` tasks can be
    executed before work stops so long as no iteration is happening.
    """
    tasks = TasksRun()
    with ThreadPoolExecutor(num_threads) as executor:
        result = executor.map(
            lambda _: tasks.run(), range(100), buffersize=20, in_order=in_order
        )
        # _is_full() is a private API specifically designed for testing:
        while not result._is_full():
            pass
        first_ran = tasks.get_ran()
        # At least half should've run:
        assert 10 <= first_ran <= 20 + num_threads
        # If we're full, sleeping should only be able to add tasks in the race
        # condition between hitting full and the rest of the threads finishing
        # a task and blocking on sending to the full queue:
        sleep(0.01)
        assert 10 <= tasks.get_ran() <= 20 + num_threads
        next(result)
        next(result)
        next(result)
        while not result._is_full():
            pass
        assert first_ran <= tasks.get_ran() <= 20 + num_threads + 3
        # Get the rest, ensure everything ran:
        list(result)
        assert tasks.get_ran() == 100


@pytest.mark.parametrize("in_order", [True, False])
@pytest.mark.parametrize("buffersize", [None, 5])
def test_drop_without_iterating_over_all_items(
    buffersize: None | int, in_order: bool
) -> None:
    """
    Dropping the results iterator doesn't stop execution.
    """
    counter = []
    results = Condition()

    def inc(x):
        run_for_usecs(10)
        with results:
            counter.append(x)
            if len(counter) == 1000:
                results.notify()
        return x

    with ThreadPoolExecutor(2) as executor:
        iterator = executor.map(
            inc, range(1000), buffersize=buffersize, in_order=in_order
        )
        next(iterator)
        del iterator

    with results:
        results.wait(10)
    assert sorted(counter) == list(range(1000))


@pytest.mark.parametrize("in_order", [True, False])
@pytest.mark.parametrize("buffersize", [None, 5])
def test_drop_does_not_panic(buffersize: None | int, in_order: bool) -> None:
    """Dropping the results iterator doesn't panic."""
    executor = ThreadPoolExecutor(2)
    it = executor.map(
        lambda x: x, range(1000), buffersize=buffersize, in_order=in_order
    )
    next(it)
    del it


def test_bad_buffersize() -> None:
    """`buffersize` must be > 0."""
    with ThreadPoolExecutor(2) as pool:
        with pytest.raises(ValueError, match="buffersize must be"):
            pool.map(lambda x: 1, range(2), buffersize=0)
        for i in [-100, -1]:
            with pytest.raises(OverflowError):
                pool.map(lambda x: 1, range(2), buffersize=i)


@given(
    st.lists(st.integers(0, 100_000), max_size=1000),
    st.integers(1, 100) | st.none(),
)
def test_in_order_delivery_property_test(
    sleep_nanos: list[int], buffersize: int | None
) -> None:
    """
    Messages are delivered in order.
    """

    # Messages are only delivered after a random delay, so order is not
    # guaranteed from execution time at least.
    def sleep_and_return(index, nanos):
        start = time_ns()
        while time_ns() - start < nanos:
            pass
        return index

    with ThreadPoolExecutor(4) as executor:
        result = list(
            executor.map(
                sleep_and_return,
                range(len(sleep_nanos)),
                sleep_nanos,
                buffersize=buffersize,
            )
        )

    assert result == list(range(len(sleep_nanos)))
