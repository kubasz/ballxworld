use bxw_util::sodiumoxide::crypto::{box_, kx, secretbox};
use num_enum::*;
use serde::*;

#[repr(u8)]
#[serde(try_from = "u8", into = "u8")]
#[derive(
    Copy, Clone, Debug, Hash, Eq, PartialEq, IntoPrimitive, TryFromPrimitive, Deserialize, Serialize,
)]
pub enum ClientConnectionType {
    GameClient = 1,
}

#[repr(u8)]
#[serde(try_from = "u8", into = "u8")]
#[derive(
    Copy, Clone, Debug, Hash, Eq, PartialEq, IntoPrimitive, TryFromPrimitive, Deserialize, Serialize,
)]
pub enum PacketTypeHandshake {
    CSHandshake1 = 10,
    SCHandshakeAck1 = 11,
}

/// Client->Server first handshake packet
#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct PktCSHandshake1Payload {
    /// Version of the client, must match `super::PACKET_PROTOCOL_CURRENT_VERSION`
    pub c_version_id: u32,
    /// Client's key exchange public key for this session
    pub c_kx_public: kx::PublicKey,
    /// Client's permanent identifying public key
    pub c_player_id: box_::PublicKey,
    /// Client connection type
    pub c_type: ClientConnectionType,
    /// Random number identifying this specific request
    pub random_cookie: u32,
}

#[repr(u8)]
#[serde(try_from = "u8", into = "u8")]
#[derive(
    Copy, Clone, Debug, Hash, Eq, PartialEq, IntoPrimitive, TryFromPrimitive, Deserialize, Serialize,
)]
pub enum ConnectionResponse {
    Accepted = 1,
    BadVersion,
    Blocked,
    NoSlots,
    AlreadyPresent,
}

/// Server->Client first handshake ack packet
#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct PktSCHandshakeAck1Payload {
    /// Version of the server, must match `super::PACKET_PROTOCOL_CURRENT_VERSION`
    pub s_version_id: u32,
    /// Server's key exchange public key for this session
    pub s_kx_public: kx::PublicKey,
    /// Server's permanent identifying public key
    pub s_server_id: box_::PublicKey,
    /// Server name
    pub s_name: String,
    /// Server's response to the connection request
    pub s_response: ConnectionResponse,
    /// Random number identifying from the client request
    pub random_cookie: u32,
}
