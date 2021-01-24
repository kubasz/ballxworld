use serde::*;
use std::cmp::Ordering;
use std::ops::{Add, AddAssign, Index, IndexMut};
use std::time::Instant;

const SEQUENCE_BUFFER_LEN: usize = 2048;

#[derive(Copy, Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Default, Hash)]
#[repr(transparent)]
#[serde(from = "u32", into = "u32")]
pub struct SeqNumber(pub u32);

#[derive(Clone, Default)]
pub struct ReliablePeerState {
    pub received: SeqBuffer,
    pub sent: SeqBuffer,
}

#[derive(Clone)]
pub struct SeqBuffer {
    pub packets: [PacketData; SEQUENCE_BUFFER_LEN],
    pub most_recent_seq: SeqNumber,
}

impl Default for SeqBuffer {
    fn default() -> Self {
        Self {
            packets: [PacketData::default(); SEQUENCE_BUFFER_LEN],
            most_recent_seq: SeqNumber::default(),
        }
    }
}

#[derive(Copy, Clone, Hash)]
pub struct PacketData {
    pub seq: SeqNumber,
    pub send_time: Instant,
    pub first_ack_time: Option<Instant>,
}

impl Default for PacketData {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            seq: SeqNumber(u32::MAX),
            send_time: now,
            first_ack_time: Some(now),
        }
    }
}

impl PacketData {
    pub fn acked(&self) -> bool {
        self.first_ack_time.is_some()
    }

    fn on_ack(&mut self, time: Instant) {
        self.first_ack_time.get_or_insert(time);
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct AckHeader {
    pub last_recv_seq: SeqNumber,
    pub ack_bitvec: u64,
}

impl ReliablePeerState {
    pub fn generate_ack_header(&self) -> AckHeader {
        let rsq = self.received.most_recent_seq.0;
        let mut ack_bitvec = 0u64;
        for seq_offset in 0..64u32 {
            let seq = rsq.wrapping_sub(seq_offset + 1);
            let pkt = &self.received[seq];
            if pkt.seq.0 == seq && pkt.acked() {
                ack_bitvec |= 1u64 << seq_offset;
            }
        }
        AckHeader {
            last_recv_seq: self.received.most_recent_seq,
            ack_bitvec,
        }
    }

    pub fn accept_ack_header(&mut self, header: &AckHeader, time: Instant) {
        let rsq = header.last_recv_seq.0;
        self.sent[rsq].on_ack(time);
        for seq_offset in 0..64u32 {
            let acked = (header.ack_bitvec & (1u64 << seq_offset)) != 0;
            if acked {
                let seq = rsq.wrapping_sub(seq_offset + 1);
                let pkt = &mut self.sent[seq];
                if pkt.seq.0 == seq {
                    self.sent[seq].on_ack(time);
                }
            }
        }
    }

    pub fn get_next_send_seq(&mut self, time: Instant) -> SeqNumber {
        self.sent.most_recent_seq += 1;
        let seq = self.sent.most_recent_seq;
        self.sent[seq] = PacketData {
            seq,
            send_time: time,
            first_ack_time: None,
        };
        seq
    }

    pub fn mark_received(&mut self, seq: SeqNumber, time: Instant) {
        self.received.most_recent_seq = self.received.most_recent_seq.max(seq);
        self.received[seq] = PacketData {
            seq,
            send_time: time,
            first_ack_time: None,
        };
    }
}

impl Index<u32> for SeqBuffer {
    type Output = PacketData;

    fn index(&self, index: u32) -> &PacketData {
        &self.packets[(index % SEQUENCE_BUFFER_LEN as u32) as usize]
    }
}

impl Index<SeqNumber> for SeqBuffer {
    type Output = PacketData;

    fn index(&self, index: SeqNumber) -> &PacketData {
        &self.packets[(index.0 % SEQUENCE_BUFFER_LEN as u32) as usize]
    }
}

impl IndexMut<u32> for SeqBuffer {
    fn index_mut(&mut self, index: u32) -> &mut PacketData {
        &mut self.packets[(index % SEQUENCE_BUFFER_LEN as u32) as usize]
    }
}

impl IndexMut<SeqNumber> for SeqBuffer {
    fn index_mut(&mut self, index: SeqNumber) -> &mut PacketData {
        &mut self.packets[(index.0 % SEQUENCE_BUFFER_LEN as u32) as usize]
    }
}

impl From<u32> for SeqNumber {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<SeqNumber> for u32 {
    fn from(v: SeqNumber) -> Self {
        v.0
    }
}

impl PartialOrd for SeqNumber {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SeqNumber {
    fn cmp(&self, other: &Self) -> Ordering {
        let ord = self.0.cmp(&other.0);
        let (max, min) = if self.0 > other.0 {
            (self.0, other.0)
        } else {
            (other.0, self.0)
        };
        if max - min > u32::MAX / 4 {
            ord.reverse()
        } else {
            ord
        }
    }
}

impl Add<u32> for SeqNumber {
    type Output = SeqNumber;

    fn add(self, rhs: u32) -> Self::Output {
        Self(self.0.wrapping_add(rhs))
    }
}

impl AddAssign<u32> for SeqNumber {
    fn add_assign(&mut self, rhs: u32) {
        self.0 = self.0.wrapping_add(rhs);
    }
}
