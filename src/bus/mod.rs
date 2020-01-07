//! Message bus for inter-subsystem communication

#[derive(Default)]
pub struct MessageBus {
    queue: Vec<Message>,
}

impl MessageBus {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn submit(&mut self, m: Message) {
        self.queue.push(m);
    }

    pub fn drain_all(&mut self) -> std::vec::Drain<'_, Message> {
        self.queue.drain(..)
    }
}

pub enum Message {}
