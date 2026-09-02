//! Low-level logic for ordered mapping.

use crossbeam_channel::Receiver;
use std::{
    collections::BTreeMap,
    sync::{Arc, Condvar, Mutex},
};

struct BatchState {
    batch_size: Option<usize>,
    num_processed: Mutex<usize>,
    batch_done: Condvar,
}

/// Allow thread that is processing potentially unordered messages and
/// reordering them to tell the producing thread that it has finished consuming
/// a batch in order.
pub fn batched_buffer_tracker(batch_size: Option<usize>) -> (BatchConsumer, BatchProducer) {
    let state = Arc::new(BatchState {
        batch_size,
        num_processed: Mutex::new(0),
        batch_done: Condvar::new(),
    });
    (
        BatchConsumer {
            state: state.clone(),
        },
        BatchProducer { state },
    )
}

pub struct BatchConsumer {
    state: Arc<BatchState>,
}

pub struct BatchProducer {
    state: Arc<BatchState>,
}

impl BatchConsumer {
    /// Called by the consumer, this notifies the tracker that a message has
    /// been reordered and has been completely processed. In practice this means
    /// it was passed on to the caller of the Python iterator.
    ///
    /// Once this is called `batch_size` times, the producing thread will be
    /// notified that it can produce another batch.
    pub fn processed_message(&self) {
        // If there is no batch size, no one is waiting on the other side for
        // notifications, so take a fast path without locking.
        let Some(batch_size) = self.state.batch_size else {
            return;
        };
        let mut num_processed_guard = self.state.num_processed.lock().unwrap();
        *num_processed_guard += 1;
        if *num_processed_guard == batch_size {
            self.state.batch_done.notify_one();
        }
    }
}

impl BatchProducer {
    /// Is the batch done?
    pub fn is_batch_done(&self) -> bool {
        // If batch_size is None, no batching was required.
        let Some(batch_size) = self.state.batch_size else {
            return true;
        };
        let num_processed_guard = self.state.num_processed.lock().unwrap();
        *num_processed_guard == batch_size
    }

    /// Called by the producer, wait until the consumer has consumed the whole
    /// batch.
    pub fn wait_until_batch_done(&self) {
        let Some(batch_size) = self.state.batch_size else {
            return;
        };
        let num_processed_guard = self.state.num_processed.lock().unwrap();
        let mut num_processed_guard = self
            .state
            .batch_done
            .wait_while(num_processed_guard, |num_processed| {
                *num_processed < batch_size
            })
            .unwrap();
        // Batch is over, consumer is waiting for us, we can now reset the
        // num_processed counter in preparation for the next batch.
        *num_processed_guard = 0;
    }
}

/// Convert a series of (message index, message) into an ordered series of
/// messages.
///
/// Generic in order to facilitate direct testing of the algorithm.
pub struct OrderedResults<M> {
    // Receives pairs of (message sequence id, message):
    receiver: Receiver<(usize, M)>,
    next_message_id: usize,
    // Map from message id to message, for ids higher than next_message_id:
    later_messages: BTreeMap<usize, M>,
    batch_consumer: BatchConsumer,
}

/// An error indicating an operation would block.
pub struct WouldBlock;

impl<M> OrderedResults<M> {
    /// Create a new instance.
    pub fn new(receiver: Receiver<(usize, M)>, batch_consumer: BatchConsumer) -> Self {
        Self {
            receiver,
            next_message_id: 0,
            later_messages: BTreeMap::new(),
            batch_consumer,
        }
    }

    /// Book keeping for a processed message we are about to return to Python.
    fn got_next_message(&mut self) {
        self.next_message_id += 1;
        self.batch_consumer.processed_message();
    }

    /// Return next message in a non-blocking manner.
    pub fn try_next(&mut self) -> Result<M, WouldBlock> {
        // First, check if we already received it.
        if let Some(entry) = self.later_messages.first_entry()
            && entry.key() == &self.next_message_id
        {
            let result = entry.remove();
            self.got_next_message();
            return Ok(result);
        }

        // Next, check if it's in the receiver queue.
        for _ in 0..8 {
            if let Some((id, message)) = self.receiver.try_recv().ok() {
                if id == self.next_message_id {
                    self.got_next_message();
                    return Ok(message);
                } else {
                    self.later_messages.insert(id, message);
                }
            } else {
                break;
            }
        }

        // Failed to get a result without blocking:
        Err(WouldBlock)
    }

    /// Return next message in blocking manner.
    ///
    /// `None` means no more messages.
    pub fn next(&mut self) -> Option<M> {
        // First, check if we already received it.
        if let Some(entry) = self.later_messages.first_entry()
            && entry.key() == &self.next_message_id
        {
            let result = entry.remove();
            self.got_next_message();
            return Some(result);
        }

        // Next, check if it's in the receiver queue.
        while let Some((id, message)) = self.receiver.recv().ok() {
            if id == self.next_message_id {
                self.got_next_message();
                return Some(message);
            } else {
                self.later_messages.insert(id, message);
            }
        }

        // Out of messages apparently (even if we've buffered some future
        // messages) since next expected message id was never seen.
        //
        // TODO maybe having something in later_messages merits a warning to
        // the user?
        None
    }

    /// Is the receiver full?
    pub fn is_full(&self) -> bool {
        self.receiver.is_full()
    }
}
