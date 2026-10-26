import json
import os
import re
from pathlib import Path

import pytest
import torch

from glmup_sglang.convert import EXPERTS, hf_name, mappable

GGUF_DIR = Path(os.environ.get("GLMUP_GGUF", "/work/ttennant1/glmup/models/gguf/IQ3_XXS"))
META = Path(os.environ.get("GLMUP_META", "/work/ttennant1/glmup/models/hf-meta"))


def llama_cpp_split_kv_b(kv_b, n_head, nope, v_dim):
    """conversion/glm.py, Glm5NextModel.modify_tensors."""
    kv = kv_b.view(n_head, v_dim + nope, kv_b.shape[-1])
    k_b, v_b = torch.split(kv, [nope, v_dim], dim=1)
    return k_b.transpose(1, 2), v_b


def test_kv_b_round_trip():
    n_head, nope, v_dim, rank = 4, 6, 6, 5
    kv_b = torch.randn(n_head * (nope + v_dim), rank)
    k_b, v_b = llama_cpp_split_kv_b(kv_b, n_head, nope, v_dim)
    back = torch.cat([k_b.transpose(1, 2), v_b], dim=1).reshape(n_head * (nope + v_dim), -1)
    assert torch.equal(back, kv_b)


def test_a_log_round_trip():
    a_log = torch.randn(64)
    assert torch.allclose(torch.log(-(-torch.exp(a_log))), a_log, atol=1e-6)


@pytest.mark.skipif(not GGUF_DIR.exists(), reason="needs the GGUF")
def test_every_tensor_maps_to_the_reference_checkpoint():
    from gguf import GGUFReader

    names = [t.name for p in sorted(GGUF_DIR.glob("*.gguf")) for t in GGUFReader(str(p)).tensors]
    assert len(names) == 1412
    assert all(mappable(n) for n in names), [n for n in names if not mappable(n)]

    produced = set()
    for n in names:
        m = re.fullmatch(r"blk\.(\d+)\.(.+)", n)
        rest = m.group(2) if m else n
        if rest in EXPERTS:
            produced |= {f"model.language_model.layers.{m.group(1)}.mlp.experts.{e}.{EXPERTS[rest]}.weight" for e in range(288)}
        elif rest in ("attn_k_b.weight", "attn_v_b.weight"):
            produced.add(f"model.language_model.layers.{m.group(1)}.self_attn.kv_b_proj.weight")
        elif rest == "ssm_a":
            produced.add(f"model.language_model.layers.{m.group(1)}.self_attn.A_log")
        else:
            produced.add(hf_name(n))
    ref = {
        n
        for n in json.loads((META / "model.safetensors.index.json").read_text())["weight_map"]
        if not n.endswith("weight_scale_inv") and ".visual." not in n
    }
    assert produced == ref, (sorted(ref - produced)[:10], sorted(produced - ref)[:10])


MMPROJ = GGUF_DIR.parent / "mmproj-GLM-5.3-Flash-Uncensored-F16.gguf"


@pytest.mark.skipif(not MMPROJ.exists(), reason="needs the mmproj")
def test_vision_tensors_map_to_the_reference_checkpoint():
    from gguf import GGUFReader

    from glmup_sglang.convert import vision_name

    names = [t.name for t in GGUFReader(str(MMPROJ)).tensors]
    produced = {vision_name(n) for n in names if not n.startswith("v.patch_embd.weight")}
    assert None not in produced
    produced.add("model.visual.patch_embed.proj.weight")
    ref = {n for n in json.loads((META / "model.safetensors.index.json").read_text())["weight_map"] if ".visual." in n}
    assert produced == ref, (sorted(ref - produced)[:10], sorted(produced - ref)[:10])
