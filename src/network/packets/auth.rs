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

/// Client->Server first handshake packet
#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
struct PktCSHandshake1Payload {
    /// Version of the client, must match `super::PACKET_PROTOCOL_CURRENT_VERSION`
    pub c_version_id: u32,
    /// Unix timestamp of the request
    pub c_time: u64,
    /// Client's key exchange public key for this session
    pub c_kx_public: kx::PublicKey,
    /// Client's permanent identifying public key
    pub c_mpid: box_::PublicKey,
}
