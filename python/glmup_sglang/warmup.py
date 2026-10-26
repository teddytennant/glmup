"""Push prompts of many lengths through a fresh server so the KDA Triton
kernels compile before real traffic does. Each new length bucket compiles
another specialization, 1-7 s each, and an unwarmed 8k prompt spent ~39 s
of its ~40 s here.

    python -m glmup_sglang.warmup URL KEY MODEL
"""

import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

LENGTHS = [16, 63, 130, 257, 511, 1000, 2047, 4100, 8191, 12000, 16400, 33000]


def complete(url, key, model, n, max_tokens=4):
    body = {"model": model, "prompt": "warm " * n, "max_tokens": max_tokens, "ignore_eos": True}
    req = urllib.request.Request(f"{url}/v1/completions", json.dumps(body).encode(), {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=1800).read()


def main():
    url, key, model = sys.argv[1:4]
    for n in LENGTHS:
        t = time.time()
        complete(url, key, model, n)
        print(f"warmup {n:6d} tokens: {time.time() - t:6.1f}s", flush=True)
    for batch in (8, 32):
        t = time.time()
        with ThreadPoolExecutor(batch) as pool:
            list(pool.map(lambda i: complete(url, key, model, 300 + 7 * i, 32), range(batch)))
        print(f"warmup batch of {batch}: {time.time() - t:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
