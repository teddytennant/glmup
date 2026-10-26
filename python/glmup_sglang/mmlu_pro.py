"""MMLU-Pro through the OpenAI chat API, thinking on.

    python -m glmup_sglang.mmlu_pro --url URL --key KEY --model NAME --data mmlu-pro.jsonl [--deadline SECONDS]

Questions still running at the deadline are left out and reported as unanswered,
so a short GPU window still gives a score on what finished.
"""

import argparse
import collections
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from urllib.request import Request, urlopen

LETTERS = "ABCDEFGHIJ"


def ask(url, key, model, row, max_tokens):
    opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(row["options"]))
    prompt = f"{row['question']}\n\n{opts}\n\nFinish with 'The answer is (X)' where X is the letter of the correct option."
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    req = Request(f"{url}/v1/chat/completions", json.dumps(body).encode(), {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urlopen(req, timeout=3600) as r:
        d = json.load(r)
    return d["choices"][0]["message"].get("content") or "", d["usage"]["completion_tokens"]


def letter(text):
    found = re.findall(r"answer is \(?([A-J])\)?", text)
    if not found:
        found = re.findall(r"\b([A-J])\b", text[-40:])
    return found[-1] if found else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--parallel", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--deadline", type=float, default=None)
    a = p.parse_args()

    rows = [json.loads(line) for line in open(a.data)]
    t0 = time.time()
    done = {}
    pool = ThreadPoolExecutor(a.parallel)
    futures = {pool.submit(ask, a.url, a.key, a.model, r, a.max_tokens): i for i, r in enumerate(rows)}
    try:
        for f in as_completed(futures, timeout=a.deadline):
            try:
                done[futures[f]] = f.result()
            except Exception as e:  # one bad request should not sink the run
                done[futures[f]] = (f"error: {e}", 0)
    except TimeoutError:
        pass
    pool.shutdown(wait=False, cancel_futures=True)

    per = collections.defaultdict(lambda: [0, 0])
    for i, (text, _) in done.items():
        cat = rows[i]["category"]
        per[cat][0] += letter(text) == rows[i]["answer"]
        per[cat][1] += 1
    correct = sum(c for c, _ in per.values())
    toks = sum(t for _, t in done.values())
    for cat, (c, n) in sorted(per.items()):
        print(f"  {cat:<18} {c}/{n}")
    print(
        f"mmlu-pro accuracy {correct}/{len(done)} = {correct / max(len(done), 1):.3f}, "
        f"{len(rows) - len(done)} unanswered at the deadline, {toks} completion tokens in {time.time() - t0:.0f}s"
    )
    # requests still in flight would otherwise hold the interpreter open past the deadline
    os._exit(0)


if __name__ == "__main__":
    main()
