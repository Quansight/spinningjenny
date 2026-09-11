import threading

from ._spinningjenny import ThreadPoolExecutor

__all__ = ["ThreadPoolExecutor", "thread_local_pool"]


class _LocalPoolStorage(threading.local):
    """Store and retrieve a cached thread-local ``ThreadPoolExecutor``."""

    pool = None
    n_threads = None

    def get(self, n_threads: int) -> ThreadPoolExecutor:
        """Get or create a cached pool, if the number of threads matches."""
        if n_threads == self.n_threads and self.pool is not None:
            return self.pool

        self.pool = ThreadPoolExecutor(n_threads)
        self.n_threads = n_threads
        return self.pool


_LOCAL_POOL = _LocalPoolStorage()


def thread_local_pool(n_threads: int) -> ThreadPoolExecutor:
    """Return a (potentially) cached executor for this thread."""
    return _LOCAL_POOL.get(n_threads)
