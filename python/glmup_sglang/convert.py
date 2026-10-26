"""Turn the GLM-5.3-Flash GGUF into a checkpoint sglang can load.

Routed experts are copied byte for byte, still in their GGUF block format
(IQ3_XXS, Q8_0 for the MTP layer), and run by `glmup_gguf` at serve time.
Everything else is dequantized to bf16 under its Hugging Face name, undoing
the two reshapes llama.cpp's converter applies (`kv_b_proj` split into
k_b/v_b, and `A_log` stored as -exp(A_log)).

    python -m glmup_sglang.convert --gguf DIR --meta DIR --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader
from gguf.quants import dequantize
from safetensors.torch import save_file

PREFIX = "model.language_model."
SHARD_BYTES = 4 << 30

GLOBAL = {
    "token_embd.weight": "model.language_model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.language_model.norm.weight",
}

LAYER = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "hc_attn_base.weight": "hc_attn_base",
    "hc_attn_fn.weight": "hc_attn_fn",
    "hc_attn_scale.weight": "hc_attn_scale",
    "hc_ffn_base.weight": "hc_ffn_base",
    "hc_ffn_fn.weight": "hc_ffn_fn",
    "hc_ffn_scale.weight": "hc_ffn_scale",
    # KDA linear attention
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "ssm_conv1d_q.weight": "self_attn.q_conv1d.weight",
    "ssm_conv1d_k.weight": "self_attn.k_conv1d.weight",
    "ssm_conv1d_v.weight": "self_attn.v_conv1d.weight",
    "ssm_beta.weight": "self_attn.b_proj.weight",
    "ssm_dt.bias": "self_attn.dt_bias",
    "ssm_f_a.weight": "self_attn.f_a_proj.weight",
    "ssm_f_b.weight": "self_attn.f_b_proj.weight",
    "ssm_g_a.weight": "self_attn.g_a_proj.weight",
    "ssm_g_b.weight": "self_attn.g_b_proj.weight",
    "ssm_norm.weight": "self_attn.o_norm.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    # MLA + DSA indexer
    "attn_q_a.weight": "self_attn.q_a_proj.weight",
    "attn_q_a_norm.weight": "self_attn.q_a_layernorm.weight",
    "attn_q_b.weight": "self_attn.q_b_proj.weight",
    "attn_kv_a_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
    "attn_kv_a_norm.weight": "self_attn.kv_a_layernorm.weight",
    "indexer.attn_k.weight": "self_attn.indexer.wk.weight",
    "indexer.attn_q_b.weight": "self_attn.indexer.wq_b.weight",
    "indexer.k_norm.weight": "self_attn.indexer.k_norm.weight",
    "indexer.k_norm.bias": "self_attn.indexer.k_norm.bias",
    "indexer.proj.weight": "self_attn.indexer.weights_proj.weight",
    "indexer_compressor_ape.weight": "self_attn.indexer.index_kpool_compress_ape",
    "indexer_compressor_gate.weight": "self_attn.indexer.index_kpool_compress_gate",
    # FFN
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "exp_probs_b.bias": "mlp.gate.e_score_correction_bias",
    "ffn_gate_shexp.weight": "mlp.shared_experts.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_experts.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_experts.down_proj.weight",
    # MTP head
    "nextn.eh_proj.weight": "eh_proj.weight",
    "nextn.enorm.weight": "enorm.weight",
    "nextn.hnorm.weight": "hnorm.weight",
    "nextn.shared_head_norm.weight": "shared_head.norm.weight",
}

EXPERTS = {"ffn_gate_exps.weight": "gate_proj", "ffn_up_exps.weight": "up_proj", "ffn_down_exps.weight": "down_proj"}

VISION_GLOBAL = {
    "v.patch_embd.bias": "patch_embed.proj.bias",
    "v.post_ln.weight": "post_layernorm.weight",
    "mm.patch_merger.weight": "downsample.weight",
    "mm.patch_merger.bias": "downsample.bias",
    "mm.model.fc.weight": "merger.proj.weight",
    "mm.post_norm.weight": "merger.post_projection_norm.weight",
    "mm.post_norm.bias": "merger.post_projection_norm.bias",
    "mm.gate.weight": "merger.gate_proj.weight",
    "mm.up.weight": "merger.up_proj.weight",
    "mm.down.weight": "merger.down_proj.weight",
}

VISION_BLOCK = {
    "ln1": "norm1",
    "ln2": "norm2",
    "attn_qkv": "attn.qkv",
    "attn_out": "attn.proj",
    "attn_q_norm": "attn.q_norm",
    "attn_k_norm": "attn.k_norm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}

UNQUANTIZED = {GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16}


def to_torch(t) -> torch.Tensor:
    """A GGUF tensor as float32 (F32) or bf16 (everything else), in torch dim order."""
    shape = tuple(reversed([int(x) for x in t.shape]))
    if t.tensor_type == GGMLQuantizationType.F32:
        return torch.from_numpy(np.array(t.data, dtype=np.float32)).reshape(shape)
    if t.tensor_type == GGMLQuantizationType.BF16:
        raw = np.ascontiguousarray(t.data).view(np.uint16)
        return torch.from_numpy(raw.copy()).view(torch.bfloat16).reshape(shape)
    if t.tensor_type == GGMLQuantizationType.F16:
        return torch.from_numpy(np.array(t.data, dtype=np.float16)).to(torch.bfloat16).reshape(shape)
    out = dequantize(np.asarray(t.data), t.tensor_type)
    return torch.from_numpy(out).to(torch.bfloat16).reshape(shape)


def expert_rows(t) -> torch.Tensor:
    """[n_experts, rows, row_bytes] uint8, the GGUF blocks untouched."""
    return torch.from_numpy(np.ascontiguousarray(t.data)).view(torch.uint8)


class ShardWriter:
    def __init__(self, out: Path):
        self.out = out
        self.buf: dict[str, torch.Tensor] = {}
        self.size = 0
        self.files: list[tuple[str, list[str]]] = []

    def add(self, name: str, tensor: torch.Tensor) -> None:
        self.buf[name] = tensor.contiguous()
        self.size += tensor.numel() * tensor.element_size()
        if self.size >= SHARD_BYTES:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        fname = f"model-{len(self.files) + 1:05d}.safetensors"
        save_file(self.buf, str(self.out / fname), metadata={"format": "pt"})
        self.files.append((fname, list(self.buf)))
        print(f"  wrote {fname} ({self.size / 2**30:.1f} GiB, {len(self.buf)} tensors)", flush=True)
        self.buf, self.size = {}, 0

    def index(self) -> dict:
        weight_map = {name: fname for fname, names in self.files for name in names}
        return {"metadata": {}, "weight_map": weight_map}


def hf_name(gguf_name: str) -> str | None:
    if gguf_name in GLOBAL:
        return GLOBAL[gguf_name]
    m = re.fullmatch(r"blk\.(\d+)\.(.+)", gguf_name)
    if m and m.group(2) in LAYER:
        return f"{PREFIX}layers.{m.group(1)}.{LAYER[m.group(2)]}"
    return None


def vision_name(gguf_name: str) -> str | None:
    if gguf_name in VISION_GLOBAL:
        return "model.visual." + VISION_GLOBAL[gguf_name]
    m = re.fullmatch(r"v\.blk\.(\d+)\.(\w+)\.(weight|bias)", gguf_name)
    if m and m.group(2) in VISION_BLOCK:
        return f"model.visual.blocks.{m.group(1)}.{VISION_BLOCK[m.group(2)]}.{m.group(3)}"
    return None


def convert_vision(mmproj: Path, writer: "ShardWriter") -> None:
    print(f"reading {mmproj.name}", flush=True)
    patch = {}
    for t in GGUFReader(str(mmproj)).tensors:
        if t.name in ("v.patch_embd.weight", "v.patch_embd.weight.1"):
            patch[t.name] = to_torch(t)
            continue
        name = vision_name(t.name)
        if name is None:
            sys.exit(f"unmapped mmproj tensor {t.name}")
        writer.add(name, to_torch(t))
    # llama.cpp splits the conv3d patch embedding into its two temporal slices
    w = torch.stack([patch["v.patch_embd.weight"], patch["v.patch_embd.weight.1"]], dim=2)
    writer.add("model.visual.patch_embed.proj.weight", w)


def mappable(gguf_name: str) -> bool:
    rest = gguf_name.split(".", 2)[-1] if gguf_name.startswith("blk.") else gguf_name
    return hf_name(gguf_name) is not None or rest in EXPERTS or rest in ("attn_k_b.weight", "attn_v_b.weight", "ssm_a")


def convert(gguf_dir: Path, meta: Path, out: Path, mmproj: Path | None = None) -> None:
    shards = sorted(gguf_dir.glob("*-0000?-of-0000?.gguf")) or sorted(gguf_dir.glob("*.gguf"))
    shards = [s for s in shards if "mmproj" not in s.name]
    if not shards:
        sys.exit(f"no .gguf shards in {gguf_dir}")
    config = json.loads((meta / "config.json").read_text())
    text = config["text_config"]
    n_head, nope, v_dim = text["num_attention_heads"], text["qk_nope_head_dim"], text["v_head_dim"]

    out.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(out)
    expert_qtypes: dict[str, dict[str, str]] = defaultdict(dict)
    pending_kv_b: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)

    readers = [GGUFReader(str(path)) for path in shards]
    unknown = [t.name for r in readers for t in r.tensors if not mappable(t.name)]
    if unknown:
        sys.exit(f"unmapped GGUF tensors: {unknown[:20]}")

    for path, reader in zip(shards, readers):
        print(f"reading {path.name}", flush=True)
        for t in reader.tensors:
            m = re.fullmatch(r"blk\.(\d+)\.(.+)", t.name)
            layer, rest = (m.group(1), m.group(2)) if m else (None, t.name)

            if rest in EXPERTS:
                role = EXPERTS[rest]
                data = expert_rows(t)
                expert_qtypes[layer][role] = t.tensor_type.name
                for e in range(data.shape[0]):
                    writer.add(f"{PREFIX}layers.{layer}.mlp.experts.{e}.{role}.weight", data[e].clone())
                continue

            if rest in ("attn_k_b.weight", "attn_v_b.weight"):
                pending_kv_b[layer][rest] = to_torch(t)
                parts = pending_kv_b[layer]
                if len(parts) == 2:
                    # llama.cpp stores k_b transposed per head; undo that and
                    # stack k/v back into the [n_head * (nope + v), kv_lora] layout.
                    k_b = parts["attn_k_b.weight"].transpose(1, 2)
                    v_b = parts["attn_v_b.weight"]
                    kv_b = torch.cat([k_b, v_b], dim=1).reshape(n_head * (nope + v_dim), -1)
                    writer.add(f"{PREFIX}layers.{layer}.self_attn.kv_b_proj.weight", kv_b)
                    del pending_kv_b[layer]
                continue

            if rest == "ssm_a":
                a = to_torch(t).float()
                writer.add(f"{PREFIX}layers.{layer}.self_attn.A_log", torch.log(-a))
                continue

            writer.add(hf_name(t.name), to_torch(t))
    if mmproj is not None:
        convert_vision(mmproj, writer)
    writer.flush()

    if pending_kv_b:
        sys.exit(f"incomplete kv_b pairs in layers {sorted(pending_kv_b)}")

    index = writer.index()
    check_against_reference(index["weight_map"], meta, vision=mmproj is not None)
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))

    config.pop("quantization_config", None)
    config["quantization_config"] = {"quant_method": "glmup_gguf", "expert_qtypes": dict(expert_qtypes)}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    for f in meta.iterdir():
        if f.is_file() and f.name not in ("config.json", "model.safetensors.index.json") and not f.name.startswith("."):
            shutil.copy2(f, out / f.name)
    (out / "GLMUP_CONVERTED").write_text(json.dumps({"shards": [s.name for s in shards]}) + "\n")
    print(f"done: {len(index['weight_map'])} tensors in {len(writer.files)} files -> {out}")


def check_against_reference(weight_map: dict, meta: Path, vision: bool) -> None:
    """Every tensor the reference checkpoint has must exist here, and nothing extra."""
    ref_index = meta / "model.safetensors.index.json"
    if not ref_index.exists():
        print("no reference index next to config.json, skipping name check")
        return
    ref = {
        n
        for n in json.loads(ref_index.read_text())["weight_map"]
        if not n.endswith("weight_scale_inv") and (vision or ".visual." not in n)
    }
    ours = set(weight_map)
    missing, extra = sorted(ref - ours), sorted(ours - ref)
    if missing or extra:
        sys.exit(f"name check failed: missing {missing[:10]} ({len(missing)}), extra {extra[:10]} ({len(extra)})")
    print(f"name check: all {len(ref)} reference tensors present")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gguf", type=Path, required=True, help="directory holding the GGUF shards")
    p.add_argument("--meta", type=Path, required=True, help="zai-org/GLM-5.3-Flash without weights (config, tokenizer)")
    p.add_argument("--mmproj", type=Path, help="the vision projector GGUF; without it the vision tower is left out")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    convert(a.gguf, a.meta, a.out, a.mmproj)


if __name__ == "__main__":
    main()
