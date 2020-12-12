use crate::config::ConfigHandle;
use crate::network::new_tokio_runtime;
use crate::network::protocol;
use bxw_util::itertools::Itertools;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::thread;
use tokio::net;
use tokio::sync::{broadcast, mpsc};

#[derive(Debug)]
pub enum ServerCreationError {
    SocketBindError { addr: String, error: std::io::Error },
}

#[derive(Clone, Debug, Hash)]
pub enum ServerControlMessage {
    Stop,
}

pub struct NetServer {
    server_thread: thread::JoinHandle<()>,
    server_control: broadcast::Sender<ServerControlMessage>,
    shared_state: Arc<NetServerSharedState>,
}

#[derive(Default)]
pub struct NetServerSharedState {
    //
}

const SERVER_CONTROL_CHANNEL_BOUND: usize = 32;

impl NetServer {
    pub fn new(cfg: ConfigHandle) -> Result<Self, ServerCreationError> {
        let tokrt = new_tokio_runtime(cfg.clone());
        let cfg_clone = cfg.clone();
        let (scon_tx, scon_rx) = broadcast::channel(SERVER_CONTROL_CHANNEL_BOUND);
        drop(scon_rx);
        let scon_tx2 = scon_tx.clone();
        let sockets = {
            let mut v: Vec<std::net::UdpSocket> =
                Vec::with_capacity(cfg.read().server_listen_addresses.len());
            for addr in cfg.read().server_listen_addresses.iter() {
                match std::net::UdpSocket::bind(addr).and_then(|s| {
                    s.set_nonblocking(true)?;
                    Ok(s)
                }) {
                    Ok(sock) => {
                        v.push(sock);
                        log::info!("Bound socket #{} to address `{}`", v.len() - 1, addr);
                    }
                    Err(error) => {
                        return Err(ServerCreationError::SocketBindError {
                            addr: format!("{}", addr),
                            error,
                        });
                    }
                }
            }
            v
        };
        let shared_state = Arc::new(NetServerSharedState::default());
        let shared_state_copy = Arc::clone(&shared_state);
        let server_thread = thread::Builder::new()
            .name("bxw-netio-main".to_owned())
            .stack_size(2 * 1024 * 1024)
            .spawn(move || {
                tokrt.block_on(async move {
                    server_netmain(cfg_clone, scon_tx2, sockets, shared_state_copy).await
                });
            })
            .expect("Couldn't start main network thread");
        Ok(Self {
            server_thread,
            server_control: scon_tx,
            shared_state,
        })
    }

    pub fn send_control_message(&self, msg: ServerControlMessage) {
        match self.server_control.send(msg.clone()) {
            Ok(_nrecv) => {
                //
            }
            Err(_) => {
                log::error!(
                    "NET CONTROL MSG failed to send - no listening interface handlers: {:?}",
                    msg
                );
            }
        }
    }

    pub fn wait_for_shutdown(self) {
        self.server_thread
            .join()
            .expect("Error when shutting down server network thread");
    }
}

#[derive(Clone, Eq, PartialEq, Debug, Hash)]
struct RawPacket {
    source: (usize, SocketAddr),
    data: Box<[u8]>,
}

async fn server_connection_handler() {
    //
}

async fn server_netmain(
    cfg: ConfigHandle,
    control: broadcast::Sender<ServerControlMessage>,
    sockets: Vec<std::net::UdpSocket>,
    shared_state: Arc<NetServerSharedState>,
) {
    let sockets: Vec<net::UdpSocket> = sockets
        .into_iter()
        .map(|s| net::UdpSocket::from_std(s).unwrap())
        .collect_vec();
    let mtu = cfg.read().server_mtu;
    let tasks = sockets
        .into_iter()
        .enumerate()
        .map(|(sid, sock)| {
            let sock = Arc::new(sock);
            let mut control_rx = control.subscribe();
            let shared_state = Arc::clone(&shared_state);
            tokio::spawn(async move {
                let mut msgbuf = vec![0u8; mtu as usize * 2];
                let mut conntable: HashMap<SocketAddr, mpsc::Sender<RawPacket>> = HashMap::with_capacity(32);
                'sockloop: loop {
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
                                    log::error!("Socket handler #{} lagged {} control messages!", sid, n);
                                }
                            }
                        }
                        recv_result = sock.recv_from(&mut msgbuf) => {
                            let (pkt_len, pkt_src_addr) = recv_result?;
                            if pkt_len > msgbuf.len() || pkt_len < 32 {
                                continue;
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
                                        log::warn!("Client handler {:?} lagging behind on packet processing", pkt_src);
                                        processed = true;
                                    }
                                }
                            }
                            if !processed {
                                //
                            }
                        }
                    }
                }
                log::info!("Socket handler #{} terminating", sid);
                Ok(()) as std::io::Result<()>
            })
        })
        .collect_vec();
    for task in tasks.into_iter() {
        task.await
            .expect("Panic from a network socket handler task")
            .expect("Error in a network socket handler task");
    }
}
