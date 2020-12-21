use num_enum::*;
use serde::*;

pub mod auth;

pub const PACKET_PROTOCOL_CURRENT_VERSION: u32 = 1;

#[repr(u8)]
#[serde(try_from = "u8", into = "u8")]
#[derive(
Copy, Clone, Debug, Hash, Eq, PartialEq, IntoPrimitive, TryFromPrimitive, Deserialize, Serialize,
)]
pub enum MessagePartType {
    AckInfo = 0x00,
    MultipartFragment = 0x01,
}
