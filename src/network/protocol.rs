//! Authentication and packet encoding (as these are tied together by encryption)

use crate::network::packets::auth::PacketTypeHandshake;
use bxw_util::sodiumoxide::crypto::{box_, sealedbox, secretbox};
use bxw_util::sodiumoxide::hex::encode;
use bxw_util::sodiumoxide::padding;
use num_enum::*;
use serde::*;
use std::borrow::Cow;
use std::convert::{TryFrom, TryInto};

pub fn current_net_timestamp() -> u64 {
    use std::time;
    let dur = time::SystemTime::now()
        .duration_since(time::UNIX_EPOCH)
        .expect("Invalid system time set, check your date/time settings");
    dur.as_secs() * 1000 + dur.subsec_millis() as u64
}

// Allow up to 10s clock difference, then drop packets
const MAX_NET_TIMESTAMP_DELTA: u64 = 10_000;

pub fn net_timestamp_delta(a: u64, b: u64) -> u64 {
    if a > b {
        a - b
    } else {
        b - a
    }
}

#[repr(u8)]
#[serde(try_from = "u8", into = "u8")]
#[derive(
    Copy, Clone, Debug, Hash, Eq, PartialEq, IntoPrimitive, TryFromPrimitive, Deserialize, Serialize,
)]
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
#[derive(
    Copy, Clone, Debug, Hash, Eq, PartialEq, IntoPrimitive, TryFromPrimitive, Deserialize, Serialize,
)]
pub enum PacketFormat {
    /// Unencrypted, first packet send client->server
    ConnectionHandshakeV1 = 0xB0,
    /// First server->client packet, encrypted with the client's public key
    ConnectionHandshakeAckV1 = 0xB1,
    /// All other packets use this format, encrypted with the established session key
    EncryptedV1 = 0xB2,
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
#[derive(
    Copy, Clone, Debug, Hash, Eq, PartialEq, IntoPrimitive, TryFromPrimitive, Deserialize, Serialize,
)]
pub enum PacketStream {
    Handshake = 0x01,
    /// Keepalive, disconnect, etc. commands
    ConnectionControl = 0x02,
}

impl PacketStream {
    /// If the packets are ACKd back and required to arrive
    pub fn is_reliable(self) -> bool {
        match self {
            Self::Handshake => true,
            Self::ConnectionControl => true,
        }
    }

    /// If the packets within this stream must all be processed in order
    pub fn is_ordered(self) -> bool {
        match self {
            Self::Handshake => true,
            Self::ConnectionControl => true,
        }
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
#[repr(C)]
pub struct PacketV1<'d> {
    /// Unencrypted
    pub format: PacketFormat,
    /// Encrypted
    pub stream: PacketStream,
    /// Encrypted, value interpretation dependent on `stream`
    pub packet_id: u8,
    /// Encrypted, sequential number of the packet in the stream
    pub seq_id: u32,
    /// Encrypted, millisecond unix timestamp at the moment of sending - packets are rejected with too big of an offset from now
    pub sent_time: u64,
    /// Encrypted, the attached message
    pub message: Cow<'d, [u8]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PacketEncodeError {
    OutputBufferTooSmall,
    MessageTooLong,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PacketDecodeError {
    TooShort,
    InvalidPadding,
    UnexpectedFieldValue(&'static str, u64),
    TooBigTimeDelta,
    DecryptionError,
}

const HANDSHAKE_MSG_PADSIZE: usize = 1024;

impl<'d> PacketV1<'d> {
    fn to_owned_message(&self) -> PacketV1<'static> {
        let owned_msg: Vec<u8> = (*self.message).to_owned();
        PacketV1 {
            format: self.format,
            stream: self.stream,
            packet_id: self.packet_id,
            seq_id: self.seq_id,
            sent_time: self.sent_time,
            message: Cow::from(owned_msg),
        }
    }

    fn fields_from_decrypted_bytes(
        format: PacketFormat,
        raw_fields: &'d [u8],
    ) -> Result<Self, PacketDecodeError> {
        if raw_fields.len() < 14 {
            Err(PacketDecodeError::TooShort)
        } else {
            Ok(Self {
                format,
                stream: PacketStream::try_from(raw_fields[0]).map_err(|_| {
                    PacketDecodeError::UnexpectedFieldValue("stream", raw_fields[0] as u64)
                })?,
                packet_id: raw_fields[1],
                seq_id: u32::from_le_bytes(raw_fields[2..6].try_into().unwrap()),
                sent_time: u64::from_le_bytes(raw_fields[6..14].try_into().unwrap()),
                message: Cow::Borrowed(&raw_fields[14..]),
            })
        }
    }

    fn write_decrypted_fields<'o>(
        &self,
        output: &'o mut [u8],
    ) -> Result<&'o mut [u8], PacketEncodeError> {
        if output.len() < 14 + self.message.len() {
            Err(PacketEncodeError::OutputBufferTooSmall)
        } else {
            output[0] = self.stream.into();
            output[1] = self.packet_id;
            output[2..6].copy_from_slice(&self.seq_id.to_le_bytes());
            output[6..14].copy_from_slice(&self.sent_time.to_le_bytes());
            output[14..14 + self.message.len()].copy_from_slice(&self.message);
            Ok(&mut output[0..14 + self.message.len()])
        }
    }

    fn write_decrypted_fields_vec(&self) -> Vec<u8> {
        let mut v = vec![0u8; self.decrypted_fields_len()];
        self.write_decrypted_fields(&mut v).unwrap();
        v
    }

    fn decrypted_fields_len(&self) -> usize {
        14 + self.message.len()
    }

    pub fn try_decode_handshake(raw: &'d [u8]) -> Result<Self, PacketDecodeError> {
        // UDP amplification prevention - require the connection-starting packet to be large
        if raw.len() < HANDSHAKE_MSG_PADSIZE {
            return Err(PacketDecodeError::TooShort);
        }
        let format = PacketFormat::try_from(raw[0])
            .map_err(|_| PacketDecodeError::UnexpectedFieldValue("format", raw[0] as u64))?;
        if format != PacketFormat::ConnectionHandshakeV1 {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "format@handshake",
                raw[0] as u64,
            ));
        }
        let mut packet = Self::fields_from_decrypted_bytes(format, &raw[1..])?;
        if packet.stream != PacketStream::Handshake {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "stream@handshake",
                packet.stream as u64,
            ));
        }
        if packet.seq_id != 0 {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "seq_id@handshake",
                packet.seq_id as u64,
            ));
        }
        if PacketTypeHandshake::try_from(packet.packet_id) != Ok(PacketTypeHandshake::CSHandshake1)
        {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "packet_id@handshake",
                packet.packet_id as u64,
            ));
        }
        if net_timestamp_delta(packet.sent_time, current_net_timestamp()) > MAX_NET_TIMESTAMP_DELTA
        {
            return Err(PacketDecodeError::TooBigTimeDelta);
        }
        let msg_unpad_len =
            padding::unpad(&packet.message, packet.message.len(), HANDSHAKE_MSG_PADSIZE)
                .map_err(|_| PacketDecodeError::InvalidPadding)?;
        packet.message = match packet.message {
            Cow::Borrowed(original) => Cow::Borrowed(&original[0..msg_unpad_len]),
            Cow::Owned(mut original) => {
                original.resize(msg_unpad_len, 0u8);
                Cow::Owned(original)
            }
        };
        Ok(packet)
    }

    pub fn encode_handshake(message: &[u8]) -> Result<Vec<u8>, PacketEncodeError> {
        let mut packet: Vec<u8> = Vec::new();
        packet.push(PacketFormat::ConnectionHandshakeV1.into());
        if message.len() >= HANDSHAKE_MSG_PADSIZE - 4 {
            return Err(PacketEncodeError::MessageTooLong);
        }
        let mut pad_buffer = [0u8; HANDSHAKE_MSG_PADSIZE];
        pad_buffer[0..message.len()].copy_from_slice(message);
        let padded_message_len =
            padding::pad(&mut pad_buffer, message.len(), HANDSHAKE_MSG_PADSIZE)
                .map_err(|_| PacketEncodeError::MessageTooLong)?;
        let padded_message = &pad_buffer[0..padded_message_len];
        let pkt = PacketV1 {
            format: PacketFormat::ConnectionHandshakeV1,
            stream: PacketStream::Handshake,
            packet_id: PacketTypeHandshake::CSHandshake1.into(),
            seq_id: 0,
            sent_time: current_net_timestamp(),
            message: Cow::Borrowed(padded_message),
        };
        packet.resize(pkt.decrypted_fields_len() + 1, 0);
        pkt.write_decrypted_fields(&mut packet[1..])?;
        Ok(packet)
    }

    pub fn try_decode_handshake_ack(
        raw: &[u8],
        client_keypair: (&box_::PublicKey, &box_::SecretKey),
    ) -> Result<PacketV1<'static>, PacketDecodeError> {
        let format = PacketFormat::try_from(raw[0])
            .map_err(|_| PacketDecodeError::UnexpectedFieldValue("format", raw[0] as u64))?;
        if format != PacketFormat::ConnectionHandshakeAckV1 {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "format@handshakeAck",
                raw[0] as u64,
            ));
        }
        let encr = &raw[1..];
        let decr = sealedbox::open(encr, client_keypair.0, client_keypair.1)
            .map_err(|_| PacketDecodeError::DecryptionError)?;
        let packet = PacketV1::fields_from_decrypted_bytes(format, &decr)?;
        if packet.stream != PacketStream::Handshake {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "stream@handshakeAck",
                packet.stream as u64,
            ));
        }
        if packet.seq_id != 1 {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "seq_id@handshakeAck",
                packet.seq_id as u64,
            ));
        }
        if PacketTypeHandshake::try_from(packet.packet_id)
            != Ok(PacketTypeHandshake::SCHandshakeAck1)
        {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "packet_id@handshakeAck",
                packet.packet_id as u64,
            ));
        }
        if net_timestamp_delta(packet.sent_time, current_net_timestamp()) > MAX_NET_TIMESTAMP_DELTA
        {
            return Err(PacketDecodeError::TooBigTimeDelta);
        }
        let packet = packet.to_owned_message();
        Ok(packet)
    }

    pub fn encode_handshake_ack(
        message: &[u8],
        client_pubkey: &box_::PublicKey,
    ) -> Result<Vec<u8>, PacketEncodeError> {
        let mut packet: Vec<u8> = Vec::new();
        packet.push(PacketFormat::ConnectionHandshakeAckV1.into());
        let pkt = PacketV1 {
            format: PacketFormat::ConnectionHandshakeAckV1,
            stream: PacketStream::Handshake,
            packet_id: PacketTypeHandshake::SCHandshakeAck1.into(),
            seq_id: 1,
            sent_time: current_net_timestamp(),
            message: Cow::Borrowed(message),
        };
        let fields = pkt.write_decrypted_fields_vec();
        let mut encrypted_fields = sealedbox::seal(&fields, client_pubkey);
        packet.append(&mut encrypted_fields);
        Ok(packet)
    }
}

#[test]
fn test_packet_v1_decrypted_encoding() {
    let msg = b"Testing packet encoding";
    let pkt = PacketV1 {
        format: PacketFormat::ConnectionHandshakeV1,
        stream: PacketStream::ConnectionControl,
        packet_id: 8,
        seq_id: 123456,
        sent_time: current_net_timestamp(),
        message: Cow::Borrowed(msg),
    };
    let mut obuf = [255u8; 96];
    let oref: &[u8] = pkt
        .write_decrypted_fields(&mut obuf)
        .expect("Obuf too small?");
    let oref_len = oref.len();
    let dpkt =
        PacketV1::fields_from_decrypted_bytes(pkt.format, oref).expect("Couldn't re-decode packet");
    assert_eq!(dpkt, pkt);
    let unencoded: &[u8] = &obuf[oref_len..];
    assert!(unencoded.iter().all(|x| *x == 255u8));
    PacketV1::fields_from_decrypted_bytes(pkt.format, &[0, 1, 2])
        .expect_err("Short packet mustn't decode correctly");
}

#[test]
fn test_packet_v1_handshake_symmetry() {
    let hs_msg = b"Hello, world\0!";
    let encoded = PacketV1::encode_handshake(hs_msg).expect("Error creating handshake message");
    let decoded =
        PacketV1::try_decode_handshake(&encoded).expect("Error decoding handshake message");
    assert_eq!(&*decoded.message, hs_msg);
    assert!(decoded.message.ends_with(b"!"));
    assert_eq!(
        PacketV1::try_decode_handshake(&encoded[0..encoded.len() / 2]),
        Err(PacketDecodeError::TooShort)
    );
}

#[test]
fn test_packet_v1_handshake_ack_symmetry() {
    bxw_util::sodiumoxide::init().unwrap();
    let (client_pk, client_sk) = box_::gen_keypair();
    let hs_msg = b"Hello, world\0!";
    let mut encoded = PacketV1::encode_handshake_ack(hs_msg, &client_pk)
        .expect("Error creating handshake ack message");
    let decoded = PacketV1::try_decode_handshake_ack(&encoded, (&client_pk, &client_sk))
        .expect("Error decoding handshake ack message");
    assert_eq!(&*decoded.message, hs_msg);
    assert!(decoded.message.ends_with(b"!"));
    assert_eq!(
        PacketV1::try_decode_handshake_ack(
            &encoded[0..encoded.len() / 2],
            (&client_pk, &client_sk)
        ),
        Err(PacketDecodeError::DecryptionError)
    );
    encoded[8] = encoded[8].wrapping_add(1);
    assert_eq!(
        PacketV1::try_decode_handshake_ack(&encoded, (&client_pk, &client_sk)),
        Err(PacketDecodeError::DecryptionError)
    );
}
