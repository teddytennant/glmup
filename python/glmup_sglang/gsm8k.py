"""GSM8K through the OpenAI chat API, thinking on, so it works behind --api-key.

    python -m glmup_sglang.gsm8k --url URL --key KEY --model NAME --data test.jsonl
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen


def ask(url, key, model, question, max_tokens):
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": question + "\nGive the final answer as a number after '####'."}],
    }
    req = Request(f"{url}/v1/chat/completions", json.dumps(body).encode(), {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urlopen(req, timeout=1800) as r:
        d = json.load(r)
    return d["choices"][0]["message"].get("content") or "", d["usage"]["completion_tokens"]


def number(text):
    tail = text.split("####")[-1]
    found = re.findall(r"-?\d[\d,]*\.?\d*", tail)
    return found[-1].replace(",", "").rstrip(".") if found else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("-n", type=int, default=200)
    p.add_argument("--parallel", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=8192)
    a = p.parse_args()

    rows = [json.loads(line) for line in open(a.data)][: a.n]
    t0 = time.time()
    with ThreadPoolExecutor(a.parallel) as pool:
        outs = list(pool.map(lambda r: ask(a.url, a.key, a.model, r["question"], a.max_tokens), rows))
    correct = sum(number(o) == r["answer"].split("####")[-1].strip().replace(",", "") for (o, _), r in zip(outs, rows))
    empty = sum(not o.strip() for o, _ in outs)
    toks = sum(t for _, t in outs)
    print(f"gsm8k accuracy {correct}/{len(rows)} = {correct / len(rows):.3f}, empty {empty}, {toks} completion tokens in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
