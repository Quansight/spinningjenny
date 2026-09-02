use crossbeam_channel::Receiver;
use std::{collections::BTreeMap, num::NonZeroU8};

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
    // How many times to try to load from the receiver if the latest message
    // isn't queued in later_messages:
    retries: NonZeroU8,
}

pub struct WouldBlock;

impl<M> OrderedResults<M> {
    pub fn new(receiver: Receiver<(usize, M)>, retries: u8) -> Self {
        Self {
            receiver,
            next_message_id: 0,
            later_messages: BTreeMap::new(),
            retries: NonZeroU8::new(retries).unwrap(),
        }
    }

    /// Return next message in a non-blocking manner.
    pub fn try_next(&mut self) -> Result<M, WouldBlock> {
        // First, check if we already received it.
        if let Some(entry) = self.later_messages.first_entry()
            && entry.key() == &self.next_message_id
        {
            self.next_message_id += 1;
            return Ok(entry.remove());
        }

        // Next, check if it's in the receiver queue.
        for _ in 0..self.retries.into() {
            if let Some((id, message)) = self.receiver.try_recv().ok() {
                if id == self.next_message_id {
                    self.next_message_id += 1;
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
            self.next_message_id += 1;
            return Some(entry.remove());
        }

        // Next, check if it's in the receiver queue.
        while let Some((id, message)) = self.receiver.recv().ok() {
            if id == self.next_message_id {
                self.next_message_id += 1;
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

    pub fn is_full(&self) -> bool {
        self.receiver.is_full()
    }
}
