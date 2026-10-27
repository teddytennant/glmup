use std::fs;
use std::os::unix::process::CommandExt;
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

/// The live tunnel on our port, found from the processes themselves rather
/// than trusted from the pid file: two `glm` processes opening a tunnel at the
/// same moment leave the file naming the one that lost the port and exited.
pub fn current() -> Option<(u32, String)> {
    let port = crate::config::Config::load().ok()?.local_port;
    let marker = format!("127.0.0.1:{port}:");
    for entry in fs::read_dir("/proc").ok()?.flatten() {
        let Ok(pid) = entry.file_name().to_string_lossy().parse::<u32>() else { continue };
        let Ok(cmdline) = fs::read(format!("/proc/{pid}/cmdline")) else { continue };
        let args: Vec<String> = cmdline.split(|&b| b == 0).map(|a| String::from_utf8_lossy(a).into_owned()).collect();
        if !args.first().is_some_and(|a| a.ends_with("ssh")) {
            continue;
        }
        if let Some(target) = args.iter().find_map(|a| a.strip_prefix(&marker)) {
            return Some((pid, target.to_string()));
        }
    }
    None
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
        .ssh_base(false)
        .args(["-N", "-o", "ExitOnForwardFailure=yes"])
        .arg("-L")
        .arg(format!("127.0.0.1:{}:{target}", cfg.local_port))
        .arg(&cfg.host)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        // own process group, so it outlives the shell that ran `glm up`
        .process_group(0)
        .spawn()
        .context("starting ssh tunnel")?;
    let pid = child.id();
    fs::create_dir_all(Config::state_dir())?;
    fs::write(pid_file(), format!("{pid} {target}\n"))?;
    for _ in 0..40 {
        std::thread::sleep(Duration::from_millis(250));
        if !alive(pid) {
            bail!(
                "tunnel exited; is 127.0.0.1:{} already in use?",
                cfg.local_port
            );
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
