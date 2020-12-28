//use serde::*;
use std::time::Instant;

const SEQUENCE_BUFFER_LEN: usize = 2048;

pub struct ReliablePeerState {
    pub packetdata_buffer: [PacketData; SEQUENCE_BUFFER_LEN],
    pub my_current_seq: u32,
}

#[derive(Clone)]
pub struct PacketData {
    pub seq: u32,
    pub acked: bool,
    pub send_time: Instant,
    pub first_recv_time: Option<Instant>,
}

impl Default for PacketData {
    fn default() -> Self {
        Self {
            seq: u32::MAX,
            acked: true,
            send_time: Instant::now(),
            first_recv_time: None,
        }
    }
}
