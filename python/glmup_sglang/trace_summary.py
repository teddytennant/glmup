"""Sum GPU kernel time by name from torch profiler traces.

    python -m glmup_sglang.trace_summary TRACE_DIR [top]
"""

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    for path in sorted(Path(sys.argv[1]).rglob("*.json*")):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as f:
            events = json.load(f).get("traceEvents", [])
        kernels = defaultdict(lambda: [0.0, 0])
        for e in events:
            if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
                k = kernels[e["name"][:110]]
                k[0] += e.get("dur", 0)
                k[1] += 1
        total = sum(v[0] for v in kernels.values()) or 1.0
        span = [e["ts"] for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
        if span:
            wall = (max(span) - min(span)) / 1000
            print(f"   GPU busy {total / 1000:.1f} ms over a {wall:.1f} ms window ({100 * total / 1000 / max(wall, 1e-9):.0f}%)")
        print(f"== {path.name}: {total / 1000:.1f} ms of GPU work")
        for name, (dur, n) in sorted(kernels.items(), key=lambda kv: -kv[1][0])[:top]:
            print(f"{dur / 1000:9.2f} ms {100 * dur / total:5.1f}% x{n:<6d} {name}")


if __name__ == "__main__":
    main()
