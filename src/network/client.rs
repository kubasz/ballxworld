use crate::config::ConfigHandle;
use crate::network::get_tokio_runtime;
use crate::network::packets;
use crate::network::packets::auth::{ClientConnectionType, ConnectionResponse};
use crate::network::protocol::{
    authflow_client_handshake_packet, authflow_client_try_accept_handshake_ack,
};
use bxw_util::sodiumoxide::crypto::box_;
use std::net::SocketAddr;
use std::sync::Arc;
use std::thread;
use std::time;
use tokio::net;
use tokio::sync::broadcast;
use tokio::time::timeout_at;

#[derive(Debug)]
pub enum ClientCreationError {
    SocketConnectionError {
        error: std::io::Error,
    },
    Timeout {
        duration: time::Duration,
    },
    ConnectionRejected {
        reason: packets::auth::ConnectionResponse,
    },
}

#[derive(Clone, Debug, Hash)]
pub enum ClientControlMessage {
    Disconnect,
}

pub struct NetClient {
    client_thread: thread::JoinHandle<()>,
    client_control: broadcast::Sender<ClientControlMessage>,
    shared_state: Arc<NetClientSharedState>,
}

pub struct NetClientSharedState {
    client_id_keys: (box_::PublicKey, box_::SecretKey),
    server_address: SocketAddr,
}

impl NetClientSharedState {
    pub fn new(
        client_id_keys: (box_::PublicKey, box_::SecretKey),
        server_address: SocketAddr,
    ) -> Self {
        Self {
            client_id_keys,
            server_address,
        }
    }
}

const CLIENT_CONTROL_CHANNEL_BOUND: usize = 1024;

impl NetClient {
    pub fn new(cfg: ConfigHandle, address: &SocketAddr) -> Result<Self, ClientCreationError> {
        let tokrt = get_tokio_runtime(Some(cfg.clone()));
        let (ccon_tx, ccon_rx) = broadcast::channel(CLIENT_CONTROL_CHANNEL_BOUND);
        drop(ccon_rx);
        let ccon_tx2 = ccon_tx.clone();
        use std::net::{Ipv4Addr, Ipv6Addr, SocketAddrV4, SocketAddrV6};
        let bindaddr = match address {
            SocketAddr::V4(_) => SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0)),
            SocketAddr::V6(_) => SocketAddr::V6(SocketAddrV6::new(Ipv6Addr::UNSPECIFIED, 0, 0, 0)),
        };
        let socket = std::net::UdpSocket::bind(bindaddr)
            .map_err(|error| ClientCreationError::SocketConnectionError { error })?;
        socket
            .connect(address)
            .map_err(|error| ClientCreationError::SocketConnectionError { error })?;
        socket
            .set_nonblocking(true)
            .map_err(|error| ClientCreationError::SocketConnectionError { error })?;
        // TODO: Save/load identifying keys
        let shared_state = Arc::new(NetClientSharedState::new(box_::gen_keypair(), *address));
        let shared_state_copy = Arc::clone(&shared_state);
        let client_thread = thread::Builder::new()
            .name("bxw-client-netio-main".to_owned())
            .stack_size(2 * 1024 * 1024)
            .spawn(move || {
                tokrt.block_on(async move {
                    if let Err(e) = client_netmain(cfg, ccon_tx2, socket, shared_state_copy).await {
                        log::error!("Client netmain terminated with error: {:?}", e);
                    }
                });
            })
            .expect("Couldn't start main client network thread");
        Ok(Self {
            client_thread,
            client_control: ccon_tx,
            shared_state,
        })
    }

    pub fn send_control_message(&self, msg: ClientControlMessage) {
        match self.client_control.send(msg.clone()) {
            Ok(_nrecv) => {}
            Err(_) => {
                log::error!(
                    "NET CONTROL MSG failed to send - no listening interface handlers: {:?}",
                    msg
                );
            }
        }
    }

    pub fn wait_for_shutdown(self) {
        self.client_thread
            .join()
            .expect("Error when shutting down server network thread");
    }
}

const NET_CLIENT_CONNECTION_RETRIES: u32 = 5;
/// Effective timeout is this multiplied by `NET_CLIENT_CONNECTION_RETRIES`
const NET_CLIENT_CONNECTION_HANDSHAKE_TIMEOUT: time::Duration = time::Duration::from_millis(500);

async fn client_netmain(
    cfg: ConfigHandle,
    control: broadcast::Sender<ClientControlMessage>,
    socket: std::net::UdpSocket,
    shared_state: Arc<NetClientSharedState>,
) -> std::io::Result<()> {
    let socket = Arc::new(net::UdpSocket::from_std(socket).unwrap());
    let mtu = cfg.read().server_mtu;
    let mut _control_rx = control.subscribe();
    let mut msgbuf = vec![0u8; mtu as usize * 2];
    log::info!(
        "Attempting connection to {:?} from {:?}",
        shared_state.server_address,
        socket.local_addr()
    );
    let (hs0pkt, hs0state) = authflow_client_handshake_packet(
        &shared_state.client_id_keys.0,
        ClientConnectionType::GameClient,
    )
    .expect("Couldn't encode handshake packet");
    let mut hs0resp = None;
    'hsloop: for _retries in 0..NET_CLIENT_CONNECTION_RETRIES {
        socket.send(&hs0pkt).await?;
        let sendtime = time::Instant::now();
        let timeout_t = sendtime + NET_CLIENT_CONNECTION_HANDSHAKE_TIMEOUT;
        loop {
            match timeout_at(timeout_t.into(), socket.recv(&mut msgbuf)).await {
                Err(_) => {
                    break;
                }
                Ok(bytes_received) => {
                    let msg = &msgbuf[0..bytes_received?];
                    match authflow_client_try_accept_handshake_ack(
                        &hs0state,
                        &msg,
                        (
                            &shared_state.client_id_keys.0,
                            &shared_state.client_id_keys.1,
                        ),
                    ) {
                        Ok(resp) => {
                            hs0resp = Some(resp);
                            break 'hsloop;
                        }
                        Err(err) => {
                            // could be spoofed packets, so just ignore them (but log)
                            log::warn!("Potential spoofed/wrong packed during handshake -- error code: {:?}", err);
                            continue;
                        }
                    }
                }
            }
        }
    }
    drop(hs0pkt);
    drop(hs0state);
    let (connresp, cccstate) = hs0resp.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "Couldn't connect to server, check log for details",
        )
    })?;
    if connresp != ConnectionResponse::Accepted {
        log::error!("Server didn't accept our connection: {:?}", connresp);
        return Err(std::io::Error::new(
            std::io::ErrorKind::ConnectionRefused,
            "Server didn't accept connection, check log for details",
        ));
    }
    log::info!(
        "Connected to server at {:?}: Name(`{}`), Version({}), ID({})",
        shared_state.server_address,
        &cccstate.server_name,
        &cccstate.server_version_id,
        bxw_util::sodiumoxide::hex::encode(&cccstate.server_id)
    );
    /*'sockloop: loop {
        tokio::select! {
            ctrl_msg = control_rx.recv() => {
                use broadcast::error::RecvError;
                match ctrl_msg {
                    Ok(msg) => {
                        match msg {
                            ServerControlMessage::Stop => {
                                break 'sockloop;
                            }
                        }
                    }
                    Err(RecvError::Closed) => {
                        break 'sockloop;
                    }
                    Err(RecvError::Lagged(n)) => {
                        log::error!("Client socket handler lagged {} control messages!", n);
                    }
                }
            }
            recv_result = sock.recv_from(&mut msgbuf) => {
                let (pkt_len, pkt_src_addr) = recv_result?;
                if pkt_len > msgbuf.len() || pkt_len < 32 {
                    continue 'sockloop;
                }
                let pkt_data_ref: &[u8] = &msgbuf[0 .. pkt_len];
                let pkt_src: (usize, std::net::SocketAddr) = (sid, pkt_src_addr);
                let conn = conntable.entry(pkt_src.1);
                let mut processed = false;
                if let std::collections::hash_map::Entry::Occupied(conn) = conn {
                    match conn.get().try_send(RawPacket{source: pkt_src, data: Box::from(pkt_data_ref)}) {
                        Ok(()) => {processed = true;}
                        Err(mpsc::error::TrySendError::Closed(_)) => {
                            conn.remove_entry();
                        }
                        Err(mpsc::error::TrySendError::Full(_)) => {
                            log::warn!("Clientside packet handler lagging behind on packet processing (on incoming packet from {:?})", pkt_src);
                            processed = true;
                        }
                    }
                }
                if !processed {
                    let shs1 = match protocol::authflow_server_try_accept_handshake_packet(pkt_data_ref) {
                        Ok(x) => x,
                        Err(_) => continue 'sockloop
                    };
                    let (packets_tx, packets_rx) = mpsc::channel(SERVER_PACKET_CHANNEL_BOUND);
                    let shared_state_clone = shared_state.clone();
                    let sock_clone = sock.clone();
                    conntable.insert(pkt_src_addr, packets_tx);
                    shared_state.connection_raw_count.fetch_add(1, SeqCst);
                    tokio::spawn(async move {server_connection_handler(pkt_src, shs1, packets_rx, shared_state_clone, sock_clone).await});
                }
            }
        }
    }*/
    log::info!("Client socket handler terminating");
    Ok(())
}
