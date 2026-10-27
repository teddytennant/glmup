use std::process::Command;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::config::Config;

/// Runs `glmup.sh` on the login node over one shared SSH connection.
pub struct Remote<'a> {
    pub cfg: &'a Config,
    pub verbose: bool,
}

#[derive(Debug, Deserialize)]
pub struct Status {
    pub job: Option<Job>,
    pub cluster: Cluster,
    #[serde(default)]
    pub api_key: String,
    pub log_last: Option<String>,
    pub load_pct: Option<u8>,
    #[serde(default)]
    pub gpus: Vec<Gpu>,
    pub endpoint: Option<Endpoint>,
    #[serde(default)]
    pub healthy: bool,
    pub health_error: Option<String>,
    pub metrics: Option<Metrics>,
}

#[derive(Debug, Deserialize)]
pub struct Job {
    pub id: String,
    pub state: String,
    pub reason: String,
    pub elapsed: String,
    pub limit: String,
    pub node: Option<String>,
    pub start: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Cluster {
    pub gpus_total: u32,
    pub gpus_free: u32,
    pub pending_jobs: u32,
}

#[derive(Debug, Deserialize)]
pub struct Gpu {
    pub index: u32,
    pub util: f64,
    pub mem_used_mib: f64,
    pub mem_total_mib: f64,
    pub power_w: f64,
}

#[derive(Debug, Deserialize)]
pub struct Endpoint {
    pub node: String,
    pub port: u16,
}

#[derive(Debug, Default, Deserialize)]
pub struct Metrics {
    pub running: Option<f64>,
    pub queued: Option<f64>,
    pub kv_usage: Option<f64>,
    pub gen_tok_s: Option<f64>,
    pub cache_hit_rate: Option<f64>,
    pub kv_capacity_tokens: Option<f64>,
}

impl Remote<'_> {
    /// `shared` reuses one control connection, which is what makes polling cheap.
    /// The tunnel must not share it: a muxed `ssh -N -L` hands the forward to the
    /// master and exits, leaving nothing to track or kill.
    pub fn ssh_base(&self, shared: bool) -> Command {
        let mut c = Command::new("ssh");
        c.args([
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=15",
        ]);
        if shared {
            let sock = Config::state_dir().join("ssh-%r@%h:%p");
            c.args(["-o", "ControlMaster=auto", "-o", "ControlPersist=10m"])
                .arg("-o")
                .arg(format!("ControlPath={}", sock.display()));
        } else {
            c.args(["-o", "ControlPath=none"]);
        }
        c
    }

    pub fn run(&self, args: &str, env: &[(&str, &str)]) -> Result<String> {
        std::fs::create_dir_all(Config::state_dir())?;
        let cfg = self.cfg;
        let mut line = format!(
            "GLMUP_ROOT={} GLMUP_ACCOUNT={} GLMUP_PARTITION={}",
            cfg.root, cfg.account, cfg.partition
        );
        for (k, v) in env {
            line.push_str(&format!(" {k}={v}"));
        }
        line.push_str(&format!(" bash {}/src/cluster/glmup.sh {args}", cfg.root));
        if self.verbose {
            eprintln!("{} ssh {} '{line}'", console::style("$").dim(), cfg.host);
        }
        // the login node drops connections now and then; one retry covers it
        let mut last = String::new();
        for _ in 0..2 {
            let out = self
                .ssh_base(true)
                .arg(&cfg.host)
                .arg(&line)
                .output()
                .context("running ssh")?;
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if self.verbose {
                    eprintln!("{}", console::style(&s).dim());
                }
                return Ok(s);
            }
            last = String::from_utf8_lossy(&out.stderr).trim().to_string();
            if out.status.code() != Some(255) {
                break;
            }
        }
        bail!("glmup.sh {args} failed on {}: {last}", cfg.host)
    }

    pub fn status(&self) -> Result<Status> {
        let raw = self.run("status", &[])?;
        serde_json::from_str(&raw).with_context(|| format!("bad status JSON: {raw}"))
    }
}
