# glmup

Serves [BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF](https://huggingface.co/BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF)
(IQ3_XXS, plus the vision projector) with sglang on two H200s under Slurm, and
gives you a small `glm` CLI to start, stop and watch it from your own machine.

sglang 0.5.22 knows GLM-5.3-Flash but not its GGUF files. glmup converts the
GGUF once: routed experts are copied byte for byte and stay IQ3_XXS on the GPU,
everything else is dequantized to bf16 under its Hugging Face name. A small
sglang plugin (`glmup_gguf`) runs the GGUF experts: ggml's MMVQ kernel for
batches of up to 16 tokens, and for larger batches a Triton IQ3_XXS dequant
straight into sglang's own triton MoE runner.

## Install on the cluster

On a Slurm login node:

```
curl -fsSL https://raw.githubusercontent.com/teddytennant/glmup/main/install.sh | GLMUP_ACCOUNT=<account> bash
```

That makes a venv with sglang 0.5.22, downloads the IQ3_XXS quant (134 GB) and
the vision projector, converts them (a CPU job, about 20 minutes), writes an
API key and submits a one-hour 2xH200 job. Everything lands in `/work/$USER/glmup`
(or `~/glmup`). Slurm settings come from `GLMUP_ACCOUNT`, `GLMUP_PARTITION`
(`gpu`), `GLMUP_SHORT_PARTITION` (`interactive-gpu`, used for jobs of an hour
or less), `GLMUP_CPUS_PER_GPU` and `GLMUP_MEM`.

On the cluster itself `cluster/glmup.sh {setup,up,off,status,logs,key}` does
the same things the CLI does.

## The `glm` CLI

```
cd cli && cargo install --path .
ln -s ~/.cargo/bin/glm ~/.cargo/bin/glmup   # optional: `glmup` means `glm up`
```

```
glm up [--hours N] [--mtp]   submit, wait with a progress bar, open the tunnel, point wizard at it
glm off                      cancel the job, close the tunnel
glm                          status: queue, job, GPU load and memory, running/queued requests, KV use
glm creds [--json]           base URL, API key, model name, a curl to try
glm logs [-f]                server log
glm tunnel [--stop]          the local port forward on its own
glm wizard                   write the wizard provider and key
glm -v ...                   print the ssh commands and raw status
```

Settings live in `~/.config/glmup/config.toml` (`host`, `root`, `account`,
`partition`, `local_port`, `model`, `wizard_provider`); env vars of the same
name with a `GLMUP_` prefix win. The API is OpenAI-compatible at
`http://127.0.0.1:30000/v1`, model `glm-5.3-flash-uncensored`, reasoning in
`reasoning_content`, tool calls parsed.

## Measured on 2 x H200 (tp=2)

| | |
|---|---|
| weights per GPU | ~65 GB |
| KV pool | 2,602,752 tokens (10.8 KB/token, 28 GB per GPU) |
| max context | 1,048,576 (the model's own limit) |
| concurrent requests | 73, capped by the KDA state pool |
| decode, 1 stream | 75 tok/s (13.4 ms/token) |
| decode, 8 streams | 201 tok/s total |
| prefill, 64k prompt | 3,700 tok/s, 16.9 s to first token |
| GSM8K, 100 questions, thinking on | 98/100, no empty answers |
| cold start | ~25 min (weights over NFS, then decode CUDA graphs) |

Above 16 concurrent decodes throughput drops sharply (47 tok/s total at 32
streams) and is being worked on.

## Tests

```
cd python && pytest tests/   # name mapping needs the GGUF; kernel tests need a GPU
```

`cluster/bench.sh` is the benchmark above; `GLMUP_SELFTEST=1 glmup.sh up` runs
the tests and the benchmark inside the job.

## Differences from the GGUF on llama.cpp

Non-expert weights (attention, shared experts, dense layers, embeddings) are
`q8_0` in the file and run here as bf16 after dequantization, so they are not
bit-identical to llama.cpp's q8_0 matmuls. The MTP head (`blk.45`) is converted
but only used with `--mtp`.
