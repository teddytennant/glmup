"""glmup.sh status: everything the local CLI shows, as one JSON object."""

import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT, PARTITION = Path(sys.argv[1]), sys.argv[2]
JOB = "glmup-serve"


def sh(*cmd, timeout=20):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def job():
    line = sh("squeue", "-h", "--me", "-n", JOB, "-o", "%i|%T|%r|%M|%l|%N|%S")
    if not line:
        return None
    jid, state, reason, elapsed, limit, node, start = line.splitlines()[0].split("|")
    return {"id": jid, "state": state, "reason": reason, "elapsed": elapsed, "limit": limit,
            "node": node or None, "start": None if start in ("N/A", "") else start}


def cluster():
    total = free = 0
    for block in sh("scontrol", "show", "node", "-o").splitlines():
        parts = re.search(r"Partitions=(\S+)", block)
        if not parts or PARTITION not in parts.group(1).split(","):
            continue
        g =re.search(r"Gres=gpu:\w+:(\d+)", block)
        if not g:
            continue
        a = re.search(r"AllocTRES=\S*gres/gpu=(\d+)", block)
        n = int(g.group(1))
        state = re.search(r"State=(\S+)", block).group(1)
        total += n
        if not any(s in state for s in ("DOWN", "DRAIN", "FAIL")):
            free += n - (int(a.group(1)) if a else 0)
    pending = len(sh("squeue", "-h", "-p", PARTITION, "-t", "PD", "-o", "%i").splitlines())
    return {"gpus_total": total, "gpus_free": free, "pending_jobs": pending}


def log_tail(jid):
    p = ROOT / "logs" / f"serve-{jid}.log"
    if not p.exists():
        return None, None
    text = p.read_bytes()[-200_000:].decode(errors="replace")
    lines = [l for l in re.split(r"[\r\n]+", text) if l.strip()]
    pct = None
    for l in reversed(lines):
        m = re.search(r"(?:Loading safetensors checkpoint|Multi-thread loading) shards:\s+(\d+)%", l)
        if m:
            pct = int(m.group(1))
            break
    return (lines[-1][-300:] if lines else None), pct


def get(url, key, timeout=5):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def metrics(base, key):
    want = {
        "sglang:num_running_reqs": "running",
        "sglang:num_queue_reqs": "queued",
        "sglang:token_usage": "kv_usage",
        "sglang:gen_throughput": "gen_tok_s",
        "sglang:cache_hit_rate": "cache_hit_rate",
        "sglang:max_total_num_tokens": "kv_capacity_tokens",
    }
    out = {}
    for line in get(f"{base}/metrics", key).splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{")[0].split(" ")[0]
        if name in want:
            # every TP rank reports the same gauge; summing would double it
            out[want[name]] = max(out.get(want[name], 0.0), float(line.rsplit(" ", 1)[1]))
    return out


def gpus(jid):
    q = sh("srun", "--overlap", f"--jobid={jid}", "-t", "1", "nvidia-smi",
           "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw", "--format=csv,noheader,nounits", timeout=30)
    rows = []
    for line in q.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 5 and parts[0].isdigit():
            rows.append({"index": int(parts[0]), "util": float(parts[1]), "mem_used_mib": float(parts[2]),
                         "mem_total_mib": float(parts[3]), "power_w": float(parts[4])})
    return rows


key_file = ROOT / "state" / "api_key"
key = key_file.read_text().strip() if key_file.exists() else ""
out = {"root": str(ROOT), "partition": PARTITION, "job": job(), "cluster": cluster(), "api_key": key}
j = out["job"]
if j:
    out["log_last"], out["load_pct"] = log_tail(j["id"])
ep = ROOT / "state" / "endpoint"
if j and j["state"] == "RUNNING":
    out["gpus"] = gpus(j["id"])
    if ep.exists():
        node, port, jid = ep.read_text().split()
        if jid == j["id"]:
            base = f"http://{node}:{port}"
            out["endpoint"] = {"node": node, "port": int(port), "url": base}
            try:
                # /health runs a generation and stalls under load; this does not
                get(f"{base}/v1/models", key, timeout=10)
                out["healthy"] = True
                out["metrics"] = metrics(base, key)
            except Exception as e:
                out["healthy"] = False
                out["health_error"] = str(e)
print(json.dumps(out))
