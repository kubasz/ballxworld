//! Authentication and packet encoding (as these are tied together by encryption)

use num_enum::*;
use serde::*;
use bxw_util::sodiumoxide::crypto::{box_, secretbox};

#[repr(u8)]
#[serde(try_from = "u8", into = "u8")]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, IntoPrimitive, TryFromPrimitive, Deserialize, Serialize)]
pub enum PacketFormatVersion {
    V1 = 1,
}

/**
 * Packet formats, sent as first byte of a packet
 * V1 = Version 1 of the network protocol (current)
 * E1 = Encrypted version 1 (current)
 */
#[repr(u8)]
#[serde(try_from = "u8", into = "u8")]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, IntoPrimitive, TryFromPrimitive, Deserialize, Serialize)]
pub enum PacketFormat {
    /// Unencrypted, first packet send client->server
    ConnectionHandshakeV1 = 0xB0,
    /// First server->client packet
    ConnectionHandshakeAckV1 = 0xB1,
    E1Keepalive = 0xB2,
}

impl PacketFormat {
    pub fn version(self) -> PacketFormatVersion {
        PacketFormatVersion::V1
    }
}

/**
 * Stream ID for the packet
 */
#[repr(u8)]
#[serde(try_from = "u8", into = "u8")]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, IntoPrimitive, TryFromPrimitive, Deserialize, Serialize)]
pub enum PacketStream {
    /// Keepalive, disconnect, handshake, etc. commands
    ConnectionControl = 0x00,
}

impl PacketStream {
    /// If the packets are ACKd back and required to arrive
    pub fn is_reliable(self) -> bool {
        match self {
            Self::ConnectionControl => true,
        }
    }

    /// If the packets within this stream must all be processed in order
    pub fn is_ordered(self) -> bool {
        match self {
            Self::ConnectionControl => true,
        }
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct PacketV1 {
    format: PacketFormat,
    stream: PacketStream,
    //
}

/// Checks if the packet (as received by the server) is a valid connection initiation packet
pub fn is_valid_connection_packet(data: &[u8]) -> bool {
    // UDP Amplification prevention
    if data.len() < 1024 {
        return false;
    }
    if data[0] != PacketFormat::ConnectionHandshakeV1 as u8 {
        return false;
    }
    if data[1] != PacketStream::ConnectionControl as u8 {
        return false;
    }
    
    true
}
