"""Long-context retrieval: hide a passcode in numbered filler lines and ask for it.

    python -m glmup_sglang.needle URL KEY MODEL TOKENIZER 131072:0.1,0.5,0.9 524288:0.5
"""

import json
import random
import sys
import time
import urllib.request

from transformers import AutoTokenizer


def ask(url, key, model, prompt):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0,
        # GLM-5.3 always thinks; low effort keeps it short
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        json.dumps(body).encode(),
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return d["choices"][0]["message"]["content"] or "", d["usage"]["prompt_tokens"], time.time() - t


def main():
    url, key, model, tok_path, *cases = sys.argv[1:]
    tok = AutoTokenizer.from_pretrained(tok_path)
    rng = random.Random(0)
    words = "river stone lamp orchard copper meadow signal harbor lantern violet thunder pebble".split()
    line = lambda i: f"Line {i}: the {rng.choice(words)} {rng.choice(words)} rests near the {rng.choice(words)}.\n"
    per_line = len(tok.encode(line(123456)))
    hits = total = 0
    for case in cases:
        size, depths = case.split(":")
        n = int(size) // per_line
        for depth in map(float, depths.split(",")):
            code = f"{rng.randrange(10**7, 10**8)}"
            lines = [line(i) for i in range(n)]
            lines.insert(int(n * depth), f"Line {int(n * depth)}b: the secret passcode is {code}.\n")
            prompt = "".join(lines) + "\nWhat is the secret passcode in the text above? Reply with the number only."
            answer, used, secs = ask(url, key, model, prompt)
            ok = code in answer
            hits += ok
            total += 1
            print(f"{used:>8} tokens, depth {depth:.0%}: {'found' if ok else 'MISSED'} in {secs:.1f}s ({answer.strip()[:40]!r})", flush=True)
    print(f"needle {hits}/{total}")


if __name__ == "__main__":
    main()
