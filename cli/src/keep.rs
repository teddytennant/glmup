//! `glm keep`: one job after another, plus a relay that rides out the reloads.
//!
//! The short partition caps a job at an hour and allows one job at a time, so
//! a long workload sees the server vanish for the 20-40 minutes the next job
//! spends loading. The relay holds requests through that gap instead of
//! failing them, and stops sending new ones a few minutes before a job's
//! walltime so the ones in flight can finish. Answers are buffered and only
//! handed on whole, so one cut off by a walltime is resent rather than lost.

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
    /// what `GET /v1/models` answers while there is no server
    models: Vec<u8>,
    /// a healthy server is behind the tunnel and not about to hit its walltime
    open: AtomicBool,
    inflight: AtomicUsize,
    held: AtomicUsize,
}

pub fn run(remote: &Remote, binds: &[String], hold: Duration, env: &[(&str, &str)]) -> Result<()> {
    let listing = serde_json::json!({
        "object": "list",
        "data": [{"id": remote.cfg.model, "object": "model", "owned_by": "glmup"}],
    })
    .to_string();
    let shared = Arc::new(Shared {
        models: format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{listing}",
            listing.len()
        )
        .into_bytes(),
        ..Default::default()
    });
    let upstream = format!("127.0.0.1:{}", remote.cfg.local_port);
    for bind in binds {
        let listener =
            TcpListener::bind(bind).with_context(|| format!("binding relay on {bind}"))?;
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
        _ => format!(
            "job {} starting{}",
            job.id,
            s.load_pct
                .map(|p| format!(", weights {p}%"))
                .unwrap_or_default()
        ),
    }
}

/// "1:00:00", "28:00", "1-02:03:04" -> seconds
fn slurm_secs(t: &str) -> u64 {
    let (days, rest) = t
        .split_once('-')
        .map_or((0, t), |(d, r)| (d.parse().unwrap_or(0), r));
    let secs = rest
        .split(':')
        .fold(0u64, |acc, p| acc * 60 + p.parse::<u64>().unwrap_or(0));
    days * 86400 + secs
}

/// One request: buffer it, wait for an open server, send it, stream the answer back.
/// Retries only while no response byte has come back, so nothing is sent twice
/// to the client.
fn relay(mut client: TcpStream, upstream: &str, shared: &Shared, hold: Duration) -> Result<()> {
    let request = read_request(&client)?;
    // harnesses health-check the model list with a short timeout before they
    // start; holding that through a reload would fail the trial outright
    if !shared.open.load(Ordering::SeqCst)
        && request.starts_with(b"GET ")
        && request
            .split(|&b| b == b' ')
            .nth(1)
            .is_some_and(|p| p.ends_with(b"/models"))
    {
        client.write_all(&shared.models)?;
        return Ok(());
    }
    let streaming = wants_stream(&request);
    if streaming {
        // commit to an event stream now, so the comment beats below can keep
        // the client's stall timer quiet through a reload or a long answer
        client.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
    }
    let started = Instant::now();
    let mut beat = Instant::now();
    let mut held = false;
    loop {
        if !shared.open.load(Ordering::SeqCst) {
            if !held {
                held = true;
                shared.held.fetch_add(1, Ordering::SeqCst);
            }
            // agents with short timeouts give up and retry; sending the abandoned
            // copy later would spend GPU time on an answer nobody reads
            let gone = client_gone(&client)
                || (streaming && beat.elapsed() > BEAT && !keepalive(&mut client, &mut beat));
            if gone || started.elapsed() > hold {
                shared.held.fetch_sub(1, Ordering::SeqCst);
                if !gone {
                    let _ = client.write_all(if streaming {
                        b"event: error\ndata: {\"error\":{\"message\":\"no GPU server came up in time\",\"type\":\"overloaded_error\"}}\n\n" as &[u8]
                    } else {
                        b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    });
                }
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
        let got = if streaming {
            fetch_with_beats(&request, upstream, &mut client, &mut beat)
        } else {
            Fetched::from(fetch(&request, upstream))
        };
        shared.inflight.fetch_sub(1, Ordering::SeqCst);
        match got {
            Fetched::Whole(response) => {
                deliver(&mut client, &response, streaming);
                return Ok(());
            }
            Fetched::ClientGone => return Ok(()),
            // nothing usable came back: the tunnel or server went away under us
            Fetched::Retry => std::thread::sleep(Duration::from_secs(3)),
        }
    }
}

const BEAT: Duration = Duration::from_secs(15);

enum Fetched {
    Whole(Vec<u8>),
    Retry,
    ClientGone,
}

impl From<Option<Vec<u8>>> for Fetched {
    fn from(r: Option<Vec<u8>>) -> Self {
        r.map_or(Fetched::Retry, Fetched::Whole)
    }
}

/// An SSE comment; every parser skips it. False when the client is gone.
fn keepalive(client: &mut TcpStream, beat: &mut Instant) -> bool {
    *beat = Instant::now();
    client.write_all(b": waiting on the GPU server\n\n").is_ok()
}

fn fetch_with_beats(
    request: &[u8],
    upstream: &str,
    client: &mut TcpStream,
    beat: &mut Instant,
) -> Fetched {
    let (tx, rx) = std::sync::mpsc::channel();
    let (request, upstream) = (request.to_vec(), upstream.to_string());
    std::thread::spawn(move || {
        let _ = tx.send(fetch(&request, &upstream));
    });
    loop {
        match rx.recv_timeout(BEAT) {
            Ok(r) => return r.into(),
            Err(_) if !keepalive(client, beat) => return Fetched::ClientGone,
            Err(_) => {}
        }
    }
}

/// Send one request and keep the answer only if it is complete.
///
/// A stream cut by a job's walltime would otherwise reach the agent half
/// written, and most harnesses treat that as fatal. Buffering makes every
/// request all or nothing, so a cut one is simply sent again to the next job.
/// The agents run headless, so token streaming buys them nothing.
fn fetch(request: &[u8], upstream: &str) -> Option<Vec<u8>> {
    let mut up = TcpStream::connect(upstream).ok()?;
    up.write_all(request).ok()?;
    let mut response = Vec::new();
    let _ = up.read_to_end(&mut response);
    complete(&response).then_some(response)
}

/// Hand a whole upstream response to the client. For a stream the headers
/// already went out, so only the events follow; an error status becomes an
/// error event, which the SDKs raise like any API error.
fn deliver(client: &mut TcpStream, response: &[u8], streaming: bool) {
    if streaming {
        let (status, chunked, body) = split(response);
        let body = if chunked {
            dechunk(body)
        } else {
            body.to_vec()
        };
        if status == "200" {
            let _ = client.write_all(&body);
        } else {
            let line = String::from_utf8_lossy(&body).replace('\n', " ");
            let _ = client.write_all(format!("event: error\ndata: {line}\n\n").as_bytes());
        }
    } else {
        let _ = client.write_all(response);
    }
    let _ = client.shutdown(Shutdown::Write);
}

fn split(response: &[u8]) -> (String, bool, &[u8]) {
    let at = response
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .unwrap_or(response.len());
    let head = String::from_utf8_lossy(&response[..at]).to_ascii_lowercase();
    let status = head.split_whitespace().nth(1).unwrap_or("").to_string();
    let chunked = head
        .lines()
        .any(|l| l.starts_with("transfer-encoding:") && l.contains("chunked"));
    (status, chunked, response.get(at + 4..).unwrap_or(&[]))
}

fn dechunk(mut body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    while let Some(eol) = body.windows(2).position(|w| w == b"\r\n") {
        let size = std::str::from_utf8(&body[..eol])
            .ok()
            .and_then(|l| usize::from_str_radix(l.split(';').next()?.trim(), 16).ok());
        let Some(size) = size.filter(|&n| n > 0 && body.len() >= eol + 2 + n) else {
            break;
        };
        out.extend_from_slice(&body[eol + 2..eol + 2 + size]);
        body = &body[(eol + 4 + size).min(body.len())..];
    }
    out
}

fn wants_stream(request: &[u8]) -> bool {
    let body = request
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map_or(&[][..], |at| &request[at + 4..]);
    serde_json::from_slice::<serde_json::Value>(body).is_ok_and(|v| v["stream"] == true)
}

fn client_gone(client: &TcpStream) -> bool {
    if client.set_nonblocking(true).is_err() {
        return true;
    }
    let mut b = [0u8; 1];
    let gone = match client.peek(&mut b) {
        Ok(0) => true,
        Ok(_) => false,
        Err(e) => e.kind() != std::io::ErrorKind::WouldBlock,
    };
    let _ = client.set_nonblocking(false);
    gone
}

/// Whether `response` is a whole HTTP/1.1 response worth handing to a client:
/// a body as long as its Content-Length, or chunked with the final chunk, and
/// not a 502/503 from a server or tunnel that is going away.
fn complete(response: &[u8]) -> bool {
    let Some(split) = response.windows(4).position(|w| w == b"\r\n\r\n") else {
        return false;
    };
    let head = String::from_utf8_lossy(&response[..split]).to_ascii_lowercase();
    let body = &response[split + 4..];
    let status = head.split_whitespace().nth(1).unwrap_or("");
    if status == "502" || status == "503" {
        return false;
    }
    for line in head.lines() {
        if let Some(v) = line.strip_prefix("content-length:") {
            return v.trim().parse::<usize>().is_ok_and(|n| body.len() >= n);
        }
        if line.starts_with("transfer-encoding:") && line.contains("chunked") {
            return body.ends_with(b"0\r\n\r\n");
        }
    }
    // no framing: the server's close marks the end
    true
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
    let ts = now
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    eprintln!("{ts} {msg}");
}

#[cfg(test)]
mod tests {
    use super::{complete, dechunk, wants_stream};

    #[test]
    fn dechunk_joins_the_chunks() {
        assert_eq!(dechunk(b"5\r\ndata:\r\n3\r\n ok\r\n0\r\n\r\n"), b"data: ok");
    }

    #[test]
    fn stream_flag_comes_from_the_body() {
        assert!(wants_stream(
            b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 15\r\n\r\n{\"stream\":true}"
        ));
        assert!(!wants_stream(
            b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"stream\":false}"
        ));
        assert!(!wants_stream(b"GET /v1/models HTTP/1.1\r\n\r\n"));
    }

    #[test]
    fn content_length_must_be_met() {
        assert!(complete(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"));
        assert!(!complete(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nok"));
    }

    #[test]
    fn chunked_needs_the_final_chunk() {
        let whole = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\ndata:\r\n0\r\n\r\n";
        assert!(complete(whole));
        assert!(!complete(&whole[..whole.len() - 5]));
    }

    #[test]
    fn gateway_errors_and_empty_replies_are_retried() {
        assert!(!complete(b""));
        assert!(!complete(b"HTTP/1.1 200 OK\r\nContent-Le"));
        assert!(!complete(
            b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n"
        ));
        assert!(complete(
            b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n"
        ));
    }
}
