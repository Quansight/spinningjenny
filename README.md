# An experiment at building a faster ThreadPoolExecutor

Compared to `concurrent.futures.ThreadPoolExecutor`:

Better performance and efficiency:

* Much less overhead per task.
  This means you can usefully parallelize more cases where there are many tasks and/or fast tasks.
* Limit how many tasks are iterated over, even when `buffersize` is not used, reducing memory usage when lazy task creation is used.

Correctness:

* `contextvars` context is propagated from the time a task is scheduled, rather than the thread pool creation.
  See [this discussion](https://discuss.python.org/t/make-threadpoolexecutor-propagate-context-at-submit-time/108274/4) for details.

More flexibility in API design:

* `map(func, *iterables, in_order=False)` allows for results to be returned when finished, rather than in order.
  This can speed up execution slightly.
