//! Low-level logic for ordered mapping.

use crossbeam_channel::Receiver;
use std::{
    collections::BTreeMap,
    sync::{Arc, Condvar, Mutex},
};

pub struct OrderedBufferState {
    max_buffer_size: usize,
    buffer_full: Mutex<bool>,
    still_has_space: Condvar,
}

impl OrderedBufferState {
    /// Is the buffer full?
    pub fn is_full(&self) -> bool {
        *self.buffer_full.lock().unwrap()
    }

    /// Called by the producer, wait until the consumer has consumed the whole
    /// batch.
    pub fn wait_for_buffer_space(&self) {
        let buffer_full_guard = self.buffer_full.lock().unwrap();
        let _guard = self
            .still_has_space
            .wait_while(buffer_full_guard, |buffer_full| *buffer_full)
            .unwrap();
    }
}

type OrderedBufferProducer = Arc<OrderedBufferState>;

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
    buffer_state: Option<Arc<OrderedBufferState>>,
}

/// An error indicating an operation would block.
pub struct WouldBlock;

impl<M> OrderedResults<M> {
    /// Create a new instance.
    pub fn new(
        receiver: Receiver<(usize, M)>,
        buffer_size: Option<usize>,
    ) -> (Self, Option<OrderedBufferProducer>) {
        let (buffer_state, producer) = if let Some(max_buffer_size) = buffer_size {
            let state = Arc::new(OrderedBufferState {
                max_buffer_size,
                buffer_full: Mutex::new(false),
                still_has_space: Condvar::new(),
            });
            (Some(state.clone()), Some(state))
        } else {
            (None, None)
        };
        (
            Self {
                receiver,
                next_message_id: 0,
                later_messages: BTreeMap::new(),
                buffer_state,
            },
            producer,
        )
    }

    fn buffer_size_changed(&self) {
        if let Some(buffer_state) = &self.buffer_state {
            let current_buffer_size = self.later_messages.len();
            let mut buffer_full = buffer_state.buffer_full.lock().unwrap();
            *buffer_full = current_buffer_size >= buffer_state.max_buffer_size;
            if !(*buffer_full) {
                buffer_state.still_has_space.notify_one();
            }
        }
    }

    /// Book keeping for a processed message we are about to return to Python.
    fn got_next_message(&mut self) {
        self.next_message_id += 1;
        self.buffer_size_changed();
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
                    self.buffer_size_changed();
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
                self.buffer_size_changed();
            }
        }

        // Out of messages apparently (even if we've buffered some future
        // messages) since next expected message id was never seen.
        //
        // TODO maybe having something in later_messages merits a warning to
        // the user?
        None
    }

    /// Is the buffer full? Used just for testing.
    pub fn _is_full(&self) -> bool {
        self.receiver.is_full()
            || self
                .buffer_state
                .as_ref()
                .map(|bs| bs.is_full())
                .unwrap_or(false)
    }
}

impl<M> Drop for OrderedResults<M> {
    fn drop(&mut self) {
        // Make sure any waiting threads get woken up:
        self.later_messages.clear();
        self.buffer_size_changed();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::unbounded;
    use proptest::option;
    use proptest::prelude::*;

    proptest! {
        /// OrderedResults::try_next returns results in order.
        #[test]
        fn ordered_results_delivery_try_next(
            buffer_size in option::of(1..100usize),
            shuffled_indexes in (0..1000usize)
                .prop_map(
                    |range| (0..range).collect::<Vec<usize>>())
                .prop_shuffle()
        ) {
            let expected : Vec<usize> = (0..(shuffled_indexes.len())).collect();
            let mut result = vec![];
            let (sender, receiver) = unbounded();
            let (mut ord_results, _) = OrderedResults::<usize>::new(receiver, buffer_size);
            for index in shuffled_indexes {
                sender.send((index, index)).unwrap();
                // Now try receiving in order:
                while let Ok(value) = ord_results.try_next() {
                    result.push(value);
                }
            }
            drop(sender);
            while let Ok(value) = ord_results.try_next() {
                result.push(value);
            }
            assert_eq!(result, expected);
        }

        /// OrderedResults::next returns results in order.
        #[test]
        fn ordered_results_delivery_next(
            buffer_size in option::of(1..100usize),
            shuffled_indexes in (0..1000usize)
                .prop_map(
                    |range| (0..range).collect::<Vec<usize>>())
                .prop_shuffle()
        ) {
            let expected : Vec<usize> = (0..(shuffled_indexes.len())).collect();
            let mut result = vec![];
            let (sender, receiver) = unbounded();
            let (mut ord_results, _) = OrderedResults::<usize>::new(receiver, buffer_size);

            let sending_thread = std::thread::spawn(move || {
                for index in shuffled_indexes {
                    sender.send((index, index)).unwrap();
                }
            });

            for value in std::iter::from_fn(|| ord_results.next()) {
                result.push(value);
            }
            sending_thread.join().unwrap();
            assert_eq!(result, expected);
        }
    }
}
