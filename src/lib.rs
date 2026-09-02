use pyo3::prelude::*;

mod ordered;

/// A faster ThreadPoolExecutor.
#[pymodule]
#[pyo3(name = "_spinningjenny")]
mod spinningjenny {
    use std::{
        cell::{Cell, RefCell},
        sync::{
            Mutex,
            atomic::{AtomicU64, Ordering},
        },
    };

    use crossbeam_channel::{Receiver, TrySendError, bounded, unbounded};
    use pyo3::{exceptions::PyValueError, intern, prelude::*, types::PyTuple};
    use rayon::{ThreadPool, ThreadPoolBuilder};

    use crate::ordered::OrderedResults;

    /// Result of calling a function.
    type PyOutcome = PyResult<Py<PyAny>>;

    #[pyclass]
    struct UnorderedResultIter {
        receiver: Mutex<Receiver<(usize, PyOutcome)>>,
    }

    impl UnorderedResultIter {
        fn new(receiver: Receiver<(usize, PyOutcome)>) -> Self {
            Self {
                receiver: Mutex::new(receiver),
            }
        }
    }

    #[pymethods]
    impl UnorderedResultIter {
        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__(&self, py: Python<'_>) -> Option<PyOutcome> {
            // Avoid blocking here, so we have consistent lock acquisition order
            // and don't deadlock. First, non-blocking fast pass:
            if let Some((_, result)) = self
                .receiver
                .try_lock()
                .ok()
                .and_then(|receiver| receiver.try_recv().ok())
            {
                return Some(result);
            }
            // If that fails, detach from Python and then block on recv():
            let receiver = &self.receiver;
            py.detach(|| receiver.lock().unwrap().recv().ok().map(|result| result.1))
        }

        /// Is the receiver buffer full? Intended for use by tests only.
        fn _is_full(&self, py: Python<'_>) -> bool {
            let receiver = &self.receiver;
            py.detach(|| receiver.lock().unwrap().is_full())
        }
    }

    #[pyclass]
    struct OrderedResultIter {
        results: Mutex<OrderedResults<PyOutcome>>,
    }

    impl OrderedResultIter {
        fn new(receiver: Receiver<(usize, PyOutcome)>, retries: u8) -> Self {
            Self {
                results: Mutex::new(OrderedResults::new(receiver, retries)),
            }
        }
    }

    #[pymethods]
    impl OrderedResultIter {
        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__(&self, py: Python<'_>) -> Option<PyOutcome> {
            // Avoid blocking here, so we have consistent lock acquisition order
            // and don't deadlock. First, non-blocking fast pass:
            if let Some(result) = self
                .results
                .try_lock()
                .ok()
                .and_then(|mut results| results.try_next().ok())
            {
                return Some(result);
            }

            // If that fails, detach from Python and then block on recv():
            let results = &self.results;
            py.detach(|| results.lock().unwrap().next())
        }

        /// Is the receiver buffer full? Intended for use by tests only.
        fn _is_full(&self, py: Python<'_>) -> bool {
            let results = &self.results;
            py.detach(|| results.lock().unwrap().is_full())
        }
    }

    thread_local! {
        // A cached copy of the `contextvars` Context to run functions in.
        pub static CONTEXTVARS_CONTEXT: RefCell<Option<Py<PyAny>>> = const { RefCell::new(None) };
        // The current generation of `contextvars` Context; if this changes, a
        // new copy of the parent context will need to be made.
        pub static GENERATION: Cell<u64> = const { Cell::new(0) };
    }

    /// Each `map()` call in the increments the global generation, so
    /// we can distinguish contexts between them.
    static GLOBAL_GENERATION: AtomicU64 = AtomicU64::new(0);

    /// Increment the generation.
    fn new_generation() -> u64 {
        GLOBAL_GENERATION.fetch_add(1, Ordering::AcqRel)
    }

    /// Return a copy of `parent_context`, which is a `contextvars.Context`. If
    /// possible this context is retrieved and cached on a thread local.
    fn thread_local_context(
        py: Python<'_>,
        parent_context: Py<PyAny>,
        generation: u64,
    ) -> PyResult<Py<PyAny>> {
        let context = if let Some(context) = CONTEXTVARS_CONTEXT.take()
            && GENERATION.get() == generation
        {
            context
        } else {
            GENERATION.set(generation);
            parent_context.call_method0(py, "copy")?
        };
        let result = context.clone_ref(py);
        CONTEXTVARS_CONTEXT.replace(Some(context));
        Ok(result)
    }

    #[pyclass]
    struct ThreadPoolExecutor {
        pool: ThreadPool,
        /// Cached `contextvars.copy_context`:
        copy_context: Py<PyAny>,
        /// Python built-in `zip()`:
        zip: Py<PyAny>,
        /// Python's `itertools.repeat`:
        repeat: Py<PyAny>,
    }

    #[pymethods]
    impl ThreadPoolExecutor {
        #[new]
        fn py_new(py: Python<'_>, n_threads: usize) -> PyResult<Self> {
            let copy_context = py.import("contextvars")?.getattr("copy_context")?.unbind();
            let zip = py.eval(c"zip", None, None)?.unbind();
            let repeat = py
                .eval(c"__import__('itertools').repeat", None, None)?
                .unbind();
            Ok(Self {
                copy_context,
                zip,
                repeat,
                pool: ThreadPoolBuilder::new()
                    .num_threads(n_threads)
                    .spawn_handler(|thread| {
                        // stay detached while idle so parked workers don't
                        // deadlock with the interpreter
                        std::thread::spawn(move || Python::attach(|py| py.detach(|| thread.run())));
                        Ok(())
                    })
                    .build()
                    .expect("TODO handle error"),
            })
        }

        #[pyo3(signature = (func, *iterables, buffersize = None, in_order = true))]
        fn map(
            &self,
            py: Python<'_>,
            func: Py<PyAny>,
            mut iterables: Vec<Py<PyAny>>,
            buffersize: Option<isize>,
            in_order: bool,
        ) -> PyResult<Py<PyAny>> {
            // Copy the current contextvars context:
            let context = self.copy_context.call0(py)?;
            // zip(*((itertools.repeat(func),) + iterables)), so we can get
            // tuples of matched values; this also checks if they actually are
            // iterables.
            let func_iterable = self.repeat.call1(py, (func,))?;
            iterables.insert(0, func_iterable);
            let iterables = PyTuple::new(py, iterables)?;
            let py_iterator = self.zip.bind(py).call1(iterables)?.try_iter()?.unbind();

            let (sender, receiver) = if let Some(buffersize) = buffersize {
                if buffersize < 1 {
                    return Err(PyValueError::new_err("buffersize must be >= 1"));
                }
                bounded(buffersize as usize)
            } else {
                unbounded()
            };

            // A unique id for the contextvars context associated with this
            // call.
            let generation = new_generation();

            let n_threads = self.pool.current_num_threads();
            // Iterate over the Python iterator in the thread pool, and spawn
            // tasks there.
            self.pool.spawn(move || {
                let orig_sender = sender.clone();
                let result = Python::attach(move |iterating_py| {
                    for (message_index, arguments) in
                        py_iterator.bind(iterating_py).into_iter().enumerate()
                    {
                        let arguments = || -> PyResult<Py<PyTuple>> {
                            Ok(arguments?.extract::<Py<PyTuple>>()?)
                        }()
                        .map_err(|err| (message_index, err))?;
                        let context = context.clone_ref(iterating_py);
                        let sender = sender.clone();
                        // This will spawn within the current pool.
                        rayon::spawn_fifo(move || {
                            Python::attach(move |thread_py| {
                                let result = thread_local_context(thread_py, context, generation)
                                    .and_then(|local_context| {
                                        local_context.call_method1(
                                            thread_py,
                                            intern!(thread_py, "run"),
                                            arguments,
                                        )
                                    });
                                // We don't want to block while attached, since that
                                // can block Python GC, resulting in deadlock when
                                // buffersize is set and these threads block.
                                if let Err(TrySendError::Full(to_resend)) =
                                    sender.try_send((message_index, result))
                                {
                                    // If we get an error sending, that
                                    // means the Receiver has been dropped.
                                    // So not much we can do.
                                    let _ = thread_py.detach(|| sender.send(to_resend));
                                };
                            });
                        });
                        // Occasionally take a break from iterating to run some
                        // tasks in this thread, so that we don't load too many
                        // tasks into memory. Other threads should steal from
                        // this one, so just because this one runs out of tasks
                        // doesn't mean no work is being done.
                        if message_index.is_multiple_of(4 * n_threads) {
                            while rayon::yield_local() != Some(rayon::Yield::Idle) {}
                        }
                    }
                    Result::<(), (usize, PyErr)>::Ok(())
                });
                if let Err((message_index, err)) = result {
                    // If we get an error, that means the Receiver has been
                    // dropped. So not much we can do.
                    let _ = orig_sender.send((message_index, Err(err)));
                }
            });
            Ok(if in_order {
                Py::new(
                    py,
                    OrderedResultIter::new(receiver, if buffersize.is_none() { 8 } else { 1 }),
                )?
                .into_any()
            } else {
                Py::new(py, UnorderedResultIter::new(receiver))?.into_any()
            })
        }

        fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __exit__(
            &self,
            _exc_type: Bound<'_, PyAny>,
            _exc_value: Bound<'_, PyAny>,
            _exc_traceback: Bound<'_, PyAny>,
        ) -> bool {
            false
        }
    }
}
