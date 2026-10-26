use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{bail, Context, Result};

use crate::config::Config;
use crate::remote::{Endpoint, Remote};

/// A background `ssh -N -L` from 127.0.0.1:<local_port> to the compute node.
/// The pid file holds "<pid> <node>:<port>" so a moved job gets a new tunnel.
fn pid_file() -> PathBuf {
    Config::state_dir().join("tunnel.pid")
}

pub fn current() -> Option<(u32, String)> {
    let s = fs::read_to_string(pid_file()).ok()?;
    let (pid, target) = s.trim().split_once(' ')?;
    let pid: u32 = pid.parse().ok()?;
    alive(pid).then(|| (pid, target.to_string()))
}

fn alive(pid: u32) -> bool {
    fs::read_to_string(format!("/proc/{pid}/cmdline")).is_ok_and(|c| c.contains("ssh"))
}

pub fn ensure(remote: &Remote, ep: &Endpoint) -> Result<u32> {
    let target = format!("{}:{}", ep.node, ep.port);
    if let Some((pid, t)) = current() {
        if t == target {
            return Ok(pid);
        }
        stop();
    }
    let cfg = remote.cfg;
    let child = remote
        .ssh_base()
        .args(["-N", "-o", "ExitOnForwardFailure=yes", "-o", "ControlMaster=no"])
        .arg("-L")
        .arg(format!("127.0.0.1:{}:{target}", cfg.local_port))
        .arg(&cfg.host)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("starting ssh tunnel")?;
    let pid = child.id();
    fs::create_dir_all(Config::state_dir())?;
    fs::write(pid_file(), format!("{pid} {target}\n"))?;
    for _ in 0..40 {
        std::thread::sleep(Duration::from_millis(250));
        if !alive(pid) {
            bail!("tunnel exited; is 127.0.0.1:{} already in use?", cfg.local_port);
        }
        if std::net::TcpStream::connect(("127.0.0.1", cfg.local_port)).is_ok() {
            return Ok(pid);
        }
    }
    bail!("tunnel to {target} did not come up")
}

pub fn stop() -> bool {
    let Some((pid, _)) = current() else {
        let _ = fs::remove_file(pid_file());
        return false;
    };
    let _ = Command::new("kill").arg(pid.to_string()).status();
    let _ = fs::remove_file(pid_file());
    true
}
