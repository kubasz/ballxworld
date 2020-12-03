use crate::config::Config;
use crate::network::new_tokio_runtime;
use bxw_util::itertools::Itertools;
use std::thread;
use tokio::net;
use tokio::sync::broadcast;

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
}

const SERVER_CONTROL_CHANNEL_BOUND: usize = 32;

impl NetServer {
    pub fn new(cfg: &Config) -> Result<Self, ServerCreationError> {
        let cfg_clone = cfg.clone();
        let tokrt = new_tokio_runtime(&cfg_clone);
        let (scon_tx, scon_rx) = broadcast::channel(SERVER_CONTROL_CHANNEL_BOUND);
        drop(scon_rx);
        let scon_tx2 = scon_tx.clone();
        let sockets = {
            let mut v: Vec<std::net::UdpSocket> =
                Vec::with_capacity(cfg.server_listen_addresses.len());
            for addr in cfg.server_listen_addresses.iter() {
                match std::net::UdpSocket::bind(addr).and_then(|s| {
                    s.set_nonblocking(true)?;
                    Ok(s)
                }) {
                    Ok(sock) => {
                        v.push(sock);
                        eprintln!("Bound socket #{} to address `{}`", v.len() - 1, addr);
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
        let server_thread = thread::Builder::new()
            .name("bxw-netio-main".to_owned())
            .stack_size(2 * 1024 * 1024)
            .spawn(move || {
                tokrt.block_on(async move { server_netmain(cfg_clone, scon_tx2, sockets).await });
            })
            .expect("Couldn't start main network thread");
        Ok(Self {
            server_thread,
            server_control: scon_tx,
        })
    }

    pub fn send_control_message(&self, msg: ServerControlMessage) {
        match self.server_control.send(msg.clone()) {
            Ok(_nrecv) => {
                //
            }
            Err(_) => {
                eprintln!(
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

async fn server_netmain(
    cfg: Config,
    control: broadcast::Sender<ServerControlMessage>,
    sockets: Vec<std::net::UdpSocket>,
) {
    let sockets: Vec<net::UdpSocket> = sockets
        .into_iter()
        .map(|s| net::UdpSocket::from_std(s).unwrap())
        .collect_vec();
    let mtu = cfg.server_mtu;
    let tasks = sockets
        .into_iter()
        .enumerate()
        .map(|(sid, sock)| {
            let mut control_rx = control.subscribe();
            tokio::spawn(async move {
                let mut msgbuf = vec![0u8; mtu as usize * 4];
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
                                    eprintln!("WARNING: Socket handler #{} lagged {} control messages!", sid, n);
                                }
                            }
                        }
                        recv_result = sock.recv_from(&mut msgbuf) => {
                            let (pkt_len, pkt_src_addr) = recv_result?;
                            let pkt: &[u8] = &msgbuf[0 .. pkt_len];
                            let _pkt_src: (usize, std::net::SocketAddr) = (sid, pkt_src_addr);
                            sock.send_to(pkt, pkt_src_addr).await?;
                        }
                    }
                }
                eprintln!("Socket handler #{} terminating", sid);
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
