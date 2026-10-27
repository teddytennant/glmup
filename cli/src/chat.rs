use std::io::{BufRead, BufReader, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use console::style;
use rustyline::error::ReadlineError;
use serde_json::{json, Value};

use crate::config::Config;
use crate::remote::Remote;
use crate::tunnel;

const HELP: &str = "/new clears the conversation · /effort max|high|low sets how hard it thinks · /system <text> sets a system prompt · /exit";

pub fn run(remote: &Remote) -> Result<()> {
    let cfg = remote.cfg;
    let key = wait_for_server(remote)?;
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    // only fires while a reply streams; at the prompt rustyline owns ctrl-c
    ctrlc::set_handler(move || flag.store(true, Ordering::SeqCst))
        .context("installing ctrl-c handler")?;

    println!(
        "{} {} at {}",
        style("chat").green().bold(),
        cfg.model,
        cfg.local_url()
    );
    println!("{}", style(HELP).dim());
    let mut rl = rustyline::DefaultEditor::new()?;
    let history = Config::state_dir().join("chat_history");
    let _ = rl.load_history(&history);
    let mut messages: Vec<Value> = Vec::new();
    let mut system: Option<String> = None;
    // GLM-5.3 always reasons; its chat template takes an effort instead of an on/off switch
    let mut effort = "max".to_string();

    loop {
        println!();
        let line = match rl.readline(&format!("{} ", style("›").cyan().bold())) {
            Ok(l) => l,
            Err(ReadlineError::Interrupted) => continue,
            Err(ReadlineError::Eof) => break,
            Err(e) => return Err(e.into()),
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let _ = rl.add_history_entry(line);
        match line
            .split_once(' ')
            .map_or((line, ""), |(c, r)| (c, r.trim()))
        {
            ("/exit" | "/quit", _) => break,
            ("/help", _) => {
                println!("{}", style(HELP).dim());
                continue;
            }
            ("/new", _) => {
                messages.clear();
                println!("{}", style("new conversation").dim());
                continue;
            }
            ("/effort", level) => {
                if matches!(level, "max" | "high" | "low") {
                    effort = level.to_string();
                }
                println!("{}", style(format!("reasoning effort {effort}")).dim());
                continue;
            }
            ("/system", text) => {
                system = (!text.is_empty()).then(|| text.to_string());
                messages.clear();
                println!(
                    "{}",
                    style(if system.is_some() {
                        "system prompt set, new conversation"
                    } else {
                        "system prompt cleared"
                    })
                    .dim()
                );
                continue;
            }
            _ => {}
        }

        messages.push(json!({"role": "user", "content": line}));
        let mut body = Vec::with_capacity(messages.len() + 1);
        if let Some(s) = &system {
            body.push(json!({"role": "system", "content": s}));
        }
        body.extend(messages.iter().cloned());
        stop.store(false, Ordering::SeqCst);
        match stream(cfg, &key, body, &effort, &stop) {
            Ok(reply) => messages.push(json!({"role": "assistant", "content": reply})),
            Err(e) => {
                messages.pop();
                println!("\n{} {e:#}", style("error:").red().bold());
            }
        }
    }
    let _ = std::fs::create_dir_all(Config::state_dir());
    let _ = rl.save_history(&history);
    Ok(())
}

/// The key, once a healthy server is reachable through the tunnel.
fn wait_for_server(remote: &Remote) -> Result<String> {
    let pb = crate::spinner();
    loop {
        let s = remote.status()?;
        if let (true, Some(ep)) = (s.healthy, s.endpoint.as_ref()) {
            pb.set_message("opening tunnel");
            tunnel::ensure(remote, ep)?;
            pb.finish_and_clear();
            return Ok(s.api_key);
        }
        let Some(job) = s.job else {
            pb.finish_and_clear();
            bail!("no job running; start one with `glm up`");
        };
        let what = match job.state.as_str() {
            "PENDING" => format!("job {} queued ({})", job.id, job.reason),
            _ => match s.load_pct.filter(|p| *p < 100) {
                Some(p) => format!("loading weights {p}%"),
                None => format!(
                    "server starting: {}",
                    s.log_last
                        .as_deref()
                        .unwrap_or("")
                        .chars()
                        .take(80)
                        .collect::<String>()
                ),
            },
        };
        pb.set_message(format!("waiting for the server · {what}"));
        std::thread::sleep(Duration::from_secs(10));
    }
}

fn stream(
    cfg: &Config,
    key: &str,
    messages: Vec<Value>,
    effort: &str,
    stop: &AtomicBool,
) -> Result<String> {
    let req = json!({
        "model": cfg.model,
        "messages": messages,
        "stream": true,
        "stream_options": {"include_usage": true},
        "chat_template_kwargs": {"reasoning_effort": effort},
    });
    let started = Instant::now();
    let resp = ureq::post(&format!("{}/chat/completions", cfg.local_url()))
        .header("Authorization", &format!("Bearer {key}"))
        .header("Content-Type", "application/json")
        .send(req.to_string().as_str())?;
    let reader = BufReader::new(resp.into_body().into_reader());

    let mut out = std::io::stdout();
    let mut reply = String::new();
    let mut first: Option<Duration> = None;
    let mut in_reasoning = false;
    let mut tokens = 0u64;
    for line in reader.lines() {
        if stop.load(Ordering::SeqCst) {
            print!("{}", style(" [stopped]").yellow());
            break;
        }
        let line = line?;
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let v: Value = serde_json::from_str(data)?;
        if let Some(n) = v["usage"]["completion_tokens"].as_u64() {
            tokens = n;
        }
        let delta = &v["choices"][0]["delta"];
        if let Some(r) = delta["reasoning_content"]
            .as_str()
            .filter(|s| !s.is_empty())
        {
            first.get_or_insert(started.elapsed());
            if !in_reasoning {
                in_reasoning = true;
                print!("{}", style("thinking ").dim().bold());
            }
            print!("{}", style(r).dim().italic());
        }
        if let Some(c) = delta["content"].as_str().filter(|s| !s.is_empty()) {
            first.get_or_insert(started.elapsed());
            if in_reasoning {
                in_reasoning = false;
                println!("\n");
            }
            let c = if reply.is_empty() { c.trim_start() } else { c };
            print!("{c}");
            reply.push_str(c);
        }
        out.flush()?;
    }
    let total = started.elapsed();
    let ttft = first.unwrap_or(total);
    let gen = (total - ttft).as_secs_f64();
    let rate = if tokens > 0 && gen > 0.0 {
        format!(" · {:.0} tok/s", tokens as f64 / gen)
    } else {
        String::new()
    };
    println!(
        "\n{}",
        style(format!(
            "{tokens} tokens · first token {:.2}s{rate}",
            ttft.as_secs_f64()
        ))
        .dim()
    );
    Ok(reply)
}
