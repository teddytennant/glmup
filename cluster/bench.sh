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

echo "== decode, 1 stream (1k in / 512 out)"
run --random-input-len 1024 --random-output-len 512 --num-prompts 3 --max-concurrency 1
for c in 8 32 128; do
  echo "== decode, $c streams (1k in / 512 out)"
  run --random-input-len 1024 --random-output-len 512 --num-prompts $c --max-concurrency $c
done
echo "== prefill (8k in / 1 out, 8 at a time)"
run --random-input-len 8192 --random-output-len 1 --num-prompts 32 --max-concurrency 8
echo "== long prefill (64k in / 64 out, 1 at a time)"
run --random-input-len 65536 --random-output-len 64 --num-prompts 2 --max-concurrency 1

echo "== gsm8k (100 questions, chat, thinking on)"
python -m glmup_sglang.gsm8k --url "$URL" --key "$KEY" --model "$NAME" --data "$GLMUP_ROOT/data/gsm8k-test.jsonl" -n 100 2>&1 | tail -3

echo "== memory after bench"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
