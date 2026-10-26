//! `glm keep`: one job after another, plus a relay that rides out the reloads.
//!
//! The short partition caps a job at an hour and allows one job at a time, so
//! a long workload sees the server vanish for the 20-40 minutes the next job
//! spends loading. The relay holds requests through that gap instead of
//! failing them, and stops sending new ones a few minutes before a job's
//! walltime so the ones in flight can finish.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::remote::{Remote, Status};
use crate::tunnel;

const DRAIN: Duration = Duration::from_secs(240);

#[derive(Default)]
struct Shared {
    /// a healthy server is behind the tunnel and not about to hit its walltime
    open: AtomicBool,
    inflight: AtomicUsize,
    held: AtomicUsize,
}

pub fn run(remote: &Remote, binds: &[String], hold: Duration, env: &[(&str, &str)]) -> Result<()> {
    let shared = Arc::new(Shared::default());
    let upstream = format!("127.0.0.1:{}", remote.cfg.local_port);
    for bind in binds {
        let listener = TcpListener::bind(bind).with_context(|| format!("binding relay on {bind}"))?;
        log(&format!("relay listening on {bind} -> {upstream}"));
        let (shared, upstream) = (shared.clone(), upstream.clone());
        std::thread::spawn(move || {
            for conn in listener.incoming().flatten() {
                let (shared, upstream) = (shared.clone(), upstream.clone());
                std::thread::spawn(move || {
                    if let Err(e) = relay(conn, &upstream, &shared, hold) {
                        log(&format!("relay: {e:#}"));
                    }
                });
            }
        });
    }

    let mut last = String::new();
    let mut tick = 0u64;
    loop {
        let line = match remote.status() {
            Ok(s) => control(remote, &s, &shared, env),
            Err(e) => {
                shared.open.store(false, Ordering::SeqCst);
                format!("status failed: {e:#}")
            }
        };
        // log state changes, plus a heartbeat every ~5 minutes
        if line != last || tick.is_multiple_of(15) {
            log(&format!(
                "{line} · {} in flight, {} held",
                shared.inflight.load(Ordering::SeqCst),
                shared.held.load(Ordering::SeqCst)
            ));
            last = line;
        }
        tick += 1;
        std::thread::sleep(Duration::from_secs(20));
    }
}

fn control(remote: &Remote, s: &Status, shared: &Shared, env: &[(&str, &str)]) -> String {
    let Some(job) = &s.job else {
        shared.open.store(false, Ordering::SeqCst);
        return match remote.run("up 1", env) {
            Ok(id) => format!("submitted job {id}"),
            Err(e) => format!("submit failed: {e:#}"),
        };
    };
    let left = slurm_secs(&job.limit).saturating_sub(slurm_secs(&job.elapsed));
    let ready = s.healthy && s.endpoint.is_some();
    if ready {
        if let Err(e) = tunnel::ensure(remote, s.endpoint.as_ref().unwrap()) {
            shared.open.store(false, Ordering::SeqCst);
            return format!("job {} up but tunnel failed: {e:#}", job.id);
        }
    }
    let open = ready && left > DRAIN.as_secs();
    shared.open.store(open, Ordering::SeqCst);
    match (ready, open) {
        (true, true) => format!("job {} serving, {}m left", job.id, left / 60),
        (true, false) => format!("job {} draining, {}s left", job.id, left),
        _ if job.state == "PENDING" => format!("job {} queued ({})", job.id, job.reason),
        _ => format!("job {} starting{}", job.id, s.load_pct.map(|p| format!(", weights {p}%")).unwrap_or_default()),
    }
}

/// "1:00:00", "28:00", "1-02:03:04" -> seconds
fn slurm_secs(t: &str) -> u64 {
    let (days, rest) = t.split_once('-').map_or((0, t), |(d, r)| (d.parse().unwrap_or(0), r));
    let secs = rest.split(':').fold(0u64, |acc, p| acc * 60 + p.parse::<u64>().unwrap_or(0));
    days * 86400 + secs
}

/// One request: buffer it, wait for an open server, send it, stream the answer back.
/// Retries only while no response byte has come back, so nothing is sent twice
/// to the client.
fn relay(mut client: TcpStream, upstream: &str, shared: &Shared, hold: Duration) -> Result<()> {
    let request = read_request(&client)?;
    let started = Instant::now();
    let mut held = false;
    loop {
        if !shared.open.load(Ordering::SeqCst) {
            if !held {
                held = true;
                shared.held.fetch_add(1, Ordering::SeqCst);
            }
            if started.elapsed() > hold {
                shared.held.fetch_sub(1, Ordering::SeqCst);
                client.write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")?;
                return Ok(());
            }
            std::thread::sleep(Duration::from_secs(2));
            continue;
        }
        if held {
            held = false;
            shared.held.fetch_sub(1, Ordering::SeqCst);
        }
        shared.inflight.fetch_add(1, Ordering::SeqCst);
        let sent = forward(&request, &mut client, upstream);
        shared.inflight.fetch_sub(1, Ordering::SeqCst);
        if sent {
            return Ok(());
        }
        // nothing came back: the tunnel or server went away under us
        std::thread::sleep(Duration::from_secs(3));
    }
}

/// True once any response byte reached the client; after that a broken stream
/// is the client's to retry, never ours.
fn forward(request: &[u8], client: &mut TcpStream, upstream: &str) -> bool {
    let Ok(mut up) = TcpStream::connect(upstream) else { return false };
    if up.write_all(request).is_err() {
        return false;
    }
    let mut buf = [0u8; 16384];
    let mut any = false;
    loop {
        let n = match up.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        // a 502 from a half-open tunnel would otherwise go straight to the agent
        if !any && buf[..n].starts_with(b"HTTP/1.1 502") {
            return false;
        }
        any = true;
        if client.write_all(&buf[..n]).is_err() {
            break;
        }
    }
    let _ = client.shutdown(Shutdown::Write);
    any
}

/// Request head plus body, with the connection forced to close after one exchange.
fn read_request(client: &TcpStream) -> Result<Vec<u8>> {
    let mut r = BufReader::new(client);
    let mut head = Vec::new();
    let mut length = 0usize;
    let mut chunked = false;
    loop {
        let mut line = String::new();
        if r.read_line(&mut line)? == 0 {
            anyhow::bail!("client closed before sending a request");
        }
        let lower = line.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            length = v.trim().parse().context("bad content-length")?;
        }
        if lower.starts_with("transfer-encoding:") && lower.contains("chunked") {
            chunked = true;
        }
        if lower.starts_with("connection:") || lower.starts_with("keep-alive:") {
            continue;
        }
        if line == "\r\n" || line == "\n" {
            head.extend_from_slice(b"Connection: close\r\n\r\n");
            break;
        }
        head.extend_from_slice(line.as_bytes());
    }
    if chunked {
        // keep the chunked framing as is; read through the terminating chunk
        loop {
            let mut size = String::new();
            r.read_line(&mut size)?;
            head.extend_from_slice(size.as_bytes());
            let n = usize::from_str_radix(size.trim().split(';').next().unwrap_or("0"), 16)?;
            let mut chunk = vec![0u8; n + 2];
            r.read_exact(&mut chunk)?;
            head.extend_from_slice(&chunk);
            if n == 0 {
                break;
            }
        }
    } else {
        let mut body = vec![0u8; length];
        r.read_exact(&mut body)?;
        head.extend_from_slice(&body);
    }
    Ok(head)
}

fn log(msg: &str) {
    let now = std::process::Command::new("date").arg("+%F %T").output();
    let ts = now.map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
    eprintln!("{ts} {msg}");
}
