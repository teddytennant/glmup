mod chat;
mod config;
mod keep;
mod remote;
mod tunnel;
mod wizard;

use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};

use config::Config;
use remote::{Remote, Status};

/// GLM-5.3-Flash (uncensored, IQ3_XXS GGUF) on 2 H200s with sglang.
///
/// `glmup` is the same binary and means `glm up`.
#[derive(Parser)]
#[command(
    name = "glm",
    version,
    after_help = "Config: ~/.config/glmup/config.toml (host, root, account, partition, local_port, model, wizard_provider)"
)]
struct Cli {
    /// Print the ssh commands and the raw status JSON
    #[arg(short, long, global = true)]
    verbose: bool,
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Submit the job (or attach to the live one), wait until it serves, open the tunnel, wire up wizard
    Up {
        /// Walltime in hours. 1h runs on the high-priority interactive partition; longer ones queue on `gpu`
        #[arg(long, default_value_t = 1)]
        hours: u32,
        /// Turn on MTP speculative decoding
        #[arg(long)]
        mtp: bool,
        /// Submit and return without waiting
        #[arg(long)]
        detach: bool,
    },
    /// Cancel the job and close the tunnel
    Off,
    /// Job, queue, GPUs, server load and tunnel
    Status,
    /// Base URL, API key and model name, plus a curl to try
    Creds {
        /// Print as JSON
        #[arg(long)]
        json: bool,
    },
    /// Tail the server log
    Logs {
        #[arg(short, default_value_t = 60)]
        n: u32,
        /// Keep following
        #[arg(short, long)]
        follow: bool,
    },
    /// Open (or with --stop, close) the local tunnel to a running job
    Tunnel {
        #[arg(long)]
        stop: bool,
    },
    /// Point wizard's provider at the tunnel with the current key
    Wizard,
    /// Keep a job running back to back and relay requests through the reloads
    Keep {
        /// Relay listen addresses (the docker bridge lets containers reach it)
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "127.0.0.1:30001,172.17.0.1:30001"
        )]
        bind: Vec<String>,
        /// Longest a request waits for a server before it gets a 503
        #[arg(long, default_value_t = 90)]
        hold_mins: u64,
    },
    /// Chat with the model in the terminal (waits for the server if it is still starting)
    Chat,
}

fn main() {
    // `glm -help` reads as -h -e -l -p to clap; take it to mean what it says
    let mut args: Vec<String> = std::env::args().collect();
    for a in args.iter_mut() {
        if a == "-help" {
            *a = "--help".into();
        }
    }
    let invoked_as_glmup = args[0].rsplit('/').next() == Some("glmup");
    if invoked_as_glmup
        && !args
            .iter()
            .skip(1)
            .any(|a| !a.starts_with('-') || a == "--help" || a == "-h")
    {
        args.insert(1, "up".into());
    }
    let cli = Cli::parse_from(args);
    if let Err(e) = run(cli) {
        eprintln!("{} {e:#}", style("error:").red().bold());
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    let cfg = Config::load()?;
    let remote = Remote {
        cfg: &cfg,
        verbose: cli.verbose,
    };
    match cli.cmd.unwrap_or(Cmd::Status) {
        Cmd::Up { hours, mtp, detach } => up(&remote, hours, mtp, detach),
        Cmd::Off => off(&remote),
        Cmd::Status => {
            let s = remote.status()?;
            print_status(&cfg, &s);
            Ok(())
        }
        Cmd::Creds { json } => creds(&remote, json),
        Cmd::Logs { n, follow } => logs(&remote, n, follow),
        Cmd::Tunnel { stop } => {
            if stop {
                println!(
                    "{}",
                    if tunnel::stop() {
                        "tunnel closed"
                    } else {
                        "no tunnel open"
                    }
                );
                return Ok(());
            }
            let s = remote.status()?;
            let Some(ep) = s.endpoint.as_ref() else {
                bail!("no running server to tunnel to")
            };
            tunnel::ensure(&remote, ep)?;
            println!(
                "tunnel open: {} -> {}:{}",
                cfg.local_url(),
                ep.node,
                ep.port
            );
            Ok(())
        }
        Cmd::Chat => chat::run(&remote),
        Cmd::Keep { bind, hold_mins } => keep::run(
            &remote,
            &bind,
            Duration::from_secs(hold_mins * 60),
            &[("GLMUP_CPUS_PER_GPU", "6"), ("GLMUP_MEM", "120G")],
        ),
        Cmd::Wizard => {
            let s = remote.status()?;
            let path = wizard::configure(&cfg, &s.api_key)?;
            println!(
                "wizard provider `{}` -> {} ({})",
                cfg.wizard_provider,
                cfg.local_url(),
                path.display()
            );
            Ok(())
        }
    }
}

pub(crate) fn spinner() -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {elapsed_precise:.dim} {wide_msg}").unwrap(),
    );
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

fn up(remote: &Remote, hours: u32, mtp: bool, detach: bool) -> Result<()> {
    let cfg = remote.cfg;
    let env: &[(&str, &str)] = if mtp { &[("GLMUP_MTP", "1")] } else { &[] };
    let id = remote.run(&format!("up {hours}"), env)?;
    println!(
        "{} job {id} ({hours}h walltime{})",
        style("submitted").green(),
        if mtp { ", MTP on" } else { "" }
    );
    if detach {
        println!("check on it with `glm status`");
        return Ok(());
    }
    println!(
        "{}",
        style("ctrl-c stops watching; the job keeps going (`glm off` cancels it)").dim()
    );

    let pb = spinner();
    let mut loading: Option<ProgressBar> = None;
    let started = Instant::now();
    let status = loop {
        let s = remote.status()?;
        let Some(job) = s.job.as_ref() else {
            pb.finish_and_clear();
            let tail = remote.run("logs 25", &[]).unwrap_or_default();
            bail!("job {id} is gone before serving. Last log lines:\n{tail}");
        };
        if s.healthy {
            break s;
        }
        match job.state.as_str() {
            "PENDING" => {
                let start = job
                    .start
                    .as_deref()
                    .map(|t| format!(" · est. start {t}"))
                    .unwrap_or_default();
                pb.set_message(format!(
                    "queued ({}) · {}/{} H200 free · {} jobs waiting{start}",
                    job.reason, s.cluster.gpus_free, s.cluster.gpus_total, s.cluster.pending_jobs
                ));
            }
            _ => {
                let node = job.node.clone().unwrap_or_default();
                if let Some(pct) = s.load_pct.filter(|p| *p < 100) {
                    let bar = loading.get_or_insert_with(|| {
                        pb.set_message(format!("loading weights on {node}"));
                        let b = ProgressBar::new(100);
                        b.set_style(
                            ProgressStyle::with_template(
                                "  {bar:40.cyan/blue} {pos:>3}% safetensors shards",
                            )
                            .unwrap(),
                        );
                        b
                    });
                    bar.set_position(pct as u64);
                } else {
                    if let Some(b) = loading.take() {
                        b.finish_and_clear();
                    }
                    let last = s.log_last.as_deref().unwrap_or("starting");
                    pb.set_message(format!(
                        "{} on {node}: {}",
                        job.state.to_lowercase(),
                        trim(last, 90)
                    ));
                }
            }
        }
        std::thread::sleep(Duration::from_secs(if job.state == "PENDING" {
            15
        } else {
            5
        }));
    };
    if let Some(b) = loading.take() {
        b.finish_and_clear();
    }
    pb.finish_and_clear();
    println!(
        "{} after {}",
        style("serving").green().bold(),
        human(started.elapsed())
    );

    let ep = status
        .endpoint
        .as_ref()
        .expect("healthy implies an endpoint");
    tunnel::ensure(remote, ep)?;
    check_local(cfg, &status.api_key)?;
    let path = wizard::configure(cfg, &status.api_key)?;
    println!(
        "{} provider `{}` updated ({})",
        style("wizard").green(),
        cfg.wizard_provider,
        path.display()
    );
    print_creds(cfg, &status.api_key);
    Ok(())
}

fn check_local(cfg: &Config, key: &str) -> Result<()> {
    let url = format!("http://127.0.0.1:{}/health", cfg.local_port);
    ureq::get(&url)
        .header("Authorization", &format!("Bearer {key}"))
        .call()?;
    Ok(())
}

fn off(remote: &Remote) -> Result<()> {
    let ids = remote.run("off", &[])?;
    let closed = tunnel::stop();
    println!(
        "{} {}{}",
        style("cancelled").yellow(),
        if ids == "none" {
            "nothing was running".into()
        } else {
            format!("job {ids}")
        },
        if closed { ", tunnel closed" } else { "" }
    );
    Ok(())
}

fn creds(remote: &Remote, json: bool) -> Result<()> {
    let cfg = remote.cfg;
    let s = remote.status()?;
    if json {
        let v = serde_json::json!({
            "base_url": cfg.local_url(),
            "api_key": s.api_key,
            "model": cfg.model,
            "remote_url": s.endpoint.as_ref().map(|e| format!("http://{}:{}/v1", e.node, e.port)),
        });
        println!("{}", serde_json::to_string_pretty(&v)?);
    } else {
        print_creds(cfg, &s.api_key);
        if s.endpoint.is_none() {
            println!(
                "{}",
                style("(no server running right now; `glm up` starts one)").dim()
            );
        }
    }
    Ok(())
}

fn print_creds(cfg: &Config, key: &str) {
    println!();
    println!("  {}  {}", style("base url").bold(), cfg.local_url());
    println!("  {}   {}", style("api key").bold(), key);
    println!("  {}     {}", style("model").bold(), cfg.model);
    println!();
    println!(
        "{}",
        style(format!(
            "  curl {}/chat/completions -H 'Authorization: Bearer {key}' -H 'Content-Type: application/json' \\\n    -d '{{\"model\": \"{}\", \"messages\": [{{\"role\": \"user\", \"content\": \"hi\"}}]}}'",
            cfg.local_url(),
            cfg.model
        ))
        .dim()
    );
}

fn logs(remote: &Remote, n: u32, follow: bool) -> Result<()> {
    if !follow {
        println!("{}", remote.run(&format!("logs {n}"), &[])?);
        return Ok(());
    }
    let mut seen = String::new();
    loop {
        let out = remote.run("logs 200", &[])?;
        // print only what came after the last line we already showed
        let fresh = match seen
            .lines()
            .last()
            .and_then(|l| out.rfind(l).map(|i| &out[i + l.len()..]))
        {
            Some(rest) => rest.trim_start_matches('\n').to_string(),
            None => out
                .lines()
                .rev()
                .take(n as usize)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n"),
        };
        if !fresh.is_empty() {
            println!("{fresh}");
        }
        seen = out;
        std::thread::sleep(Duration::from_secs(3));
    }
}

fn print_status(cfg: &Config, s: &Status) {
    let c = &s.cluster;
    println!(
        "{} {}/{} H200 free, {} jobs waiting",
        style("cluster").bold(),
        c.gpus_free,
        c.gpus_total,
        c.pending_jobs
    );
    let Some(job) = &s.job else {
        println!("{} none (start one with `glm up`)", style("job").bold());
        if tunnel::stop() {
            println!("{}", style("closed a stale tunnel").dim());
        }
        return;
    };
    let state = match job.state.as_str() {
        "RUNNING" => style(job.state.as_str()).green(),
        "PENDING" => style(job.state.as_str()).yellow(),
        _ => style(job.state.as_str()).red(),
    };
    println!(
        "{} {} {} · {} of {}{}",
        style("job").bold(),
        job.id,
        state,
        job.elapsed,
        job.limit,
        job.node
            .as_deref()
            .map(|n| format!(" · {n}"))
            .unwrap_or_default()
    );
    if job.state == "PENDING" {
        println!(
            "    reason {}{}",
            job.reason,
            job.start
                .as_deref()
                .map(|t| format!(", est. start {t}"))
                .unwrap_or_default()
        );
    }
    if let Some(pct) = s.load_pct.filter(|p| *p < 100 && !s.healthy) {
        println!("    loading weights {pct}%");
    }
    for g in &s.gpus {
        // a card held back by heat runs well under its max clock under load and
        // slows its tensor-parallel partner to match
        let throttled = g.sm_max_mhz > 0.0 && g.util > 50.0 && g.sm_mhz < 0.8 * g.sm_max_mhz;
        let clock = format!("{:>4.0}/{:.0} MHz {:>2.0}°C", g.sm_mhz, g.sm_max_mhz, g.temp_c);
        println!(
            "{} {} {:>3.0}% util  {:>6.1}/{:.0} GiB  {:>4.0} W  {}",
            style("gpu").bold(),
            g.index,
            g.util,
            g.mem_used_mib / 1024.0,
            g.mem_total_mib / 1024.0,
            g.power_w,
            if throttled { style(format!("{clock} throttled")).red().to_string() } else { clock }
        );
    }
    match (&s.endpoint, s.healthy) {
        (Some(ep), true) => {
            println!(
                "{} healthy at {}:{}",
                style("server").bold(),
                ep.node,
                ep.port
            );
            if let Some(m) = &s.metrics {
                let f = |v: Option<f64>| v.map(|x| format!("{x:.0}")).unwrap_or("-".into());
                println!(
                    "    {} running, {} queued, kv {}, {} tok/s generated, prefix cache hit {}",
                    f(m.running),
                    f(m.queued),
                    m.kv_usage
                        .map(|x| format!("{:.1}%", x * 100.0))
                        .unwrap_or("-".into()),
                    f(m.gen_tok_s),
                    m.cache_hit_rate
                        .map(|x| format!("{:.0}%", x * 100.0))
                        .unwrap_or("-".into())
                );
                if let Some(cap) = m.kv_capacity_tokens {
                    println!("    kv pool {cap:.0} tokens");
                }
            }
        }
        (Some(_), false) => println!(
            "{} not answering: {}",
            style("server").bold(),
            s.health_error.as_deref().unwrap_or("?")
        ),
        (None, _) => {
            if let Some(last) = &s.log_last {
                println!("{} starting: {}", style("server").bold(), trim(last, 100));
            }
        }
    }
    match tunnel::current() {
        Some((pid, target)) => println!(
            "{} {} -> {target} (pid {pid})",
            style("tunnel").bold(),
            cfg.local_url()
        ),
        None if s.healthy => println!("{} closed (`glm tunnel` opens it)", style("tunnel").bold()),
        None => {}
    }
}

fn trim(s: &str, n: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}

fn human(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 3600 {
        format!("{}h{:02}m", s / 3600, s % 3600 / 60)
    } else {
        format!("{}m{:02}s", s / 60, s % 60)
    }
}
