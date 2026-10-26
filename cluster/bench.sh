#!/bin/bash
# Smoke test, accuracy check and throughput sweep against a running server.
# usage: bench.sh PORT KEY MODEL
set -uo pipefail
PORT=$1 KEY=$2 NAME=$3
URL=http://127.0.0.1:$PORT

echo "== memory after load"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
curl -s -H "Authorization: Bearer $KEY" "$URL/get_server_info" | python -c '
import json, sys
d = json.load(sys.stdin)
for k in ("max_total_num_tokens", "max_running_requests", "context_length", "max_mamba_cache_size", "mem_fraction_static"):
    v = d.get(k)
    if v is None:
        for s in d.get("internal_states", []):
            v = s.get(k, v)
    print(k, v)
'

if [ "${GLMUP_PROFILE:-0}" = 1 ]; then
  # profile pure decode: start 32 long requests, let their prefill finish, then trace
  dir=$GLMUP_ROOT/logs/trace-${SLURM_JOB_ID:-x}-decode32
  rm -rf "$dir"; mkdir -p "$dir"
  python - "$URL" "$KEY" "$NAME" "$dir" <<'PY'
import json, sys, threading, time, urllib.request
url, key, name, out = sys.argv[1:5]
hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
def post(path, body, timeout=900):
    req = urllib.request.Request(f"{url}{path}", json.dumps(body).encode(), hdr)
    return urllib.request.urlopen(req, timeout=timeout).read().decode()
def one(i):
    post("/v1/completions", {"model": name, "prompt": f"{i} " + "count " * 200, "max_tokens": 600, "ignore_eos": True})
threads = [threading.Thread(target=one, args=(i,)) for i in range(32)]
t0 = time.time()
for t in threads: t.start()
time.sleep(20)
print("start_profile:", post("/start_profile", {"output_dir": out, "num_steps": 10, "activities": ["GPU", "CPU"]})[:200])
for t in threads: t.join()
print(f"32 x 600 tokens done in {time.time() - t0:.1f}s")
PY
  sleep 5
  echo "== trace: decode at 32 streams"
  python -m glmup_sglang.trace_summary "$dir" 25
  grep "Decode batch" "$GLMUP_ROOT/logs/serve-${SLURM_JOB_ID:-x}.log" | tail -5
fi

echo "== chat smoke"
curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' "$URL/v1/chat/completions" -d "{
  \"model\": \"$NAME\", \"max_tokens\": 2048,
  \"messages\": [{\"role\": \"user\", \"content\": \"What is 17 * 23? Then name the capital of Australia. Be brief.\"}]
}" | python -c '
import json, sys
d = json.load(sys.stdin)
m = d["choices"][0]["message"]
print("reasoning chars:", len(m.get("reasoning_content") or ""))
print("content:", m.get("content"))
print("usage:", d.get("usage"))
'


run() {
  python -m sglang.benchmark.serving --backend sglang-oai --base-url "$URL" --model "$NAME" --tokenizer "$GLMUP_ROOT/models/sglang" \
    --dataset-name random-ids --random-range-ratio 1 --disable-tqdm "$@" 2>&1 \
    | grep -E "Max request concurrency|Successful requests|Input token throughput|Output token throughput|Mean TTFT|Median TTFT|Mean TPOT|Median ITL|Request throughput"
}
export OPENAI_API_KEY=$KEY

# cheapest and most important first: a job's hour can run out partway through
echo "== decode, 1 stream (1k in / 512 out)"
run --random-input-len 1024 --random-output-len 512 --num-prompts 3 --max-concurrency 1
for c in 8 32 64; do
  echo "== decode, $c streams (1k in / 512 out)"
  run --random-input-len 1024 --random-output-len 512 --num-prompts $c --max-concurrency $c
done

echo "== needle in a haystack (reasoning effort low)"
python -m glmup_sglang.needle "$URL" "$KEY" "$NAME" "$GLMUP_ROOT/models/sglang" 131072:0.1,0.5,0.9 524288:0.5

if [ -f "$GLMUP_ROOT/data/mmlu-pro-280.jsonl" ]; then
  echo "== mmlu-pro (280 questions, 20 per subject, thinking on, ${GLMUP_MMLU_DEADLINE:-480}s deadline)"
  python -m glmup_sglang.mmlu_pro --url "$URL" --key "$KEY" --model "$NAME" --data "$GLMUP_ROOT/data/mmlu-pro-280.jsonl" --deadline "${GLMUP_MMLU_DEADLINE:-480}"
fi

echo "== decode, 128 streams (1k in / 512 out)"
run --random-input-len 1024 --random-output-len 512 --num-prompts 128 --max-concurrency 128
echo "== prefill (8k in / 1 out, 8 at a time)"
run --random-input-len 8192 --random-output-len 1 --num-prompts 32 --max-concurrency 8
echo "== long prefill (64k in / 64 out, 1 at a time)"
run --random-input-len 65536 --random-output-len 64 --num-prompts 2 --max-concurrency 1

echo "== gsm8k (100 questions, chat, thinking on)"
python -m glmup_sglang.gsm8k --url "$URL" --key "$KEY" --model "$NAME" --data "$GLMUP_ROOT/data/gsm8k-test.jsonl" -n 100 2>&1 | tail -3

echo "== memory after bench"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
