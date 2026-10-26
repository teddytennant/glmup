"""GPU checks of the expert kernels against gguf-py's reference dequantizer.

Uses one real MoE layer from the GGUF, so it needs a GPU and the files.
"""

import os
import types
from pathlib import Path

import numpy as np
import pytest
import torch

GGUF_DIR = Path(os.environ.get("GLMUP_GGUF", "/work/ttennant1/glmup/models/gguf/IQ3_XXS"))
LAYER = 3
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not GGUF_DIR.exists(), reason="needs a GPU and the GGUF")


@pytest.fixture(scope="module")
def experts():
    from gguf import GGUFReader

    found = {}
    for p in sorted(GGUF_DIR.glob("*.gguf")):
        for t in GGUFReader(str(p)).tensors:
            for role in ("gate", "up", "down"):
                if t.name == f"blk.{LAYER}.ffn_{role}_exps.weight":
                    found[role] = t
    assert set(found) == {"gate", "up", "down"}
    return found


def reference(t, expert):
    from gguf.quants import dequantize

    rows = np.ascontiguousarray(t.data[expert])
    return torch.from_numpy(dequantize(rows, t.tensor_type)).float()


def test_dequant_kernel_matches_gguf_py(experts):
    from glmup_sglang.gguf_experts import dequantize

    t = experts["down"]
    rows = torch.from_numpy(np.ascontiguousarray(t.data[:2])).cuda()
    got = dequantize(rows, t.tensor_type, 2048, torch.float32).cpu()
    for e in range(2):
        torch.testing.assert_close(got[e], reference(t, e), rtol=0, atol=1e-6)


def test_dequant_whole_tp_shard_of_a_layer(experts):
    """The serving shape: all 288 experts of one TP rank's down projection at once."""
    from glmup_sglang.gguf_experts import dequantize, row_bytes

    t = experts["down"]
    full = torch.from_numpy(np.ascontiguousarray(t.data))
    half = row_bytes(t.tensor_type, 1024)
    shard = full[:, :, half:].contiguous().cuda()  # rank 1 of tp=2
    got = dequantize(shard, t.tensor_type, 1024, torch.bfloat16)
    assert got.shape == (288, 4096, 1024)
    for e in (0, 143, 287):
        torch.testing.assert_close(got[e].float().cpu(), reference(t, e)[:, 1024:], rtol=1e-2, atol=1e-3)


def test_triton_iq3_xxs_matches_gguf_py(experts):
    from glmup_sglang.iq3_xxs import dequantize_iq3_xxs

    t = experts["gate"]
    assert t.tensor_type.name == "IQ3_XXS"
    rows = torch.from_numpy(np.ascontiguousarray(t.data[:3])).cuda()
    got = dequantize_iq3_xxs(rows, torch.empty(3, 2048, 4096, device="cuda")).cpu()
    for e in range(3):
        torch.testing.assert_close(got[e], reference(t, e), rtol=0, atol=0)


def test_triton_iq3_xxs_speed(experts):
    """A whole TP shard of one layer (gate+up+down) must dequantize in well under 10 ms."""
    from glmup_sglang.iq3_xxs import dequantize_iq3_xxs

    w13 = torch.randint(0, 255, (288, 2048, 1568), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 255, (288, 4096, 392), dtype=torch.uint8, device="cuda")
    w2[..., 0:2] = 0  # keep fp16 scales finite
    w13[..., 0:2] = 0
    o13 = torch.empty(288, 2048, 4096, dtype=torch.bfloat16, device="cuda")
    o2 = torch.empty(288, 4096, 1024, dtype=torch.bfloat16, device="cuda")
    for _ in range(3):
        dequantize_iq3_xxs(w13, o13), dequantize_iq3_xxs(w2, o2)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        dequantize_iq3_xxs(w13, o13), dequantize_iq3_xxs(w2, o2)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / 10
    print(f"IQ3_XXS dequant of one layer shard: {ms:.2f} ms")
    assert ms < 10


def test_vec_path_matches_reference(experts):
    from glmup_sglang.gguf_experts import GGUFExpertsMoEMethod

    torch.manual_seed(0)
    gate, up, down = experts["gate"], experts["up"], experts["down"]
    w13 = torch.cat(
        [torch.from_numpy(np.ascontiguousarray(gate.data)), torch.from_numpy(np.ascontiguousarray(up.data))], dim=1
    ).cuda()
    w2 = torch.from_numpy(np.ascontiguousarray(down.data)).cuda()

    method = GGUFExpertsMoEMethod(gate.tensor_type, down.tensor_type)
    method.hidden_size, method.inter = 4096, 2048
    method.moe_runner_config = types.SimpleNamespace(swiglu_limit=10.0, routed_scaling_factor=2.5)
    layer = types.SimpleNamespace(w13_weight=w13, w2_weight=w2)

    tokens, top_k = 4, 8
    x = (torch.randn(tokens, 4096, device="cuda") * 0.5).bfloat16()
    ids = torch.stack([torch.randperm(16)[:top_k] for _ in range(tokens)]).int().cuda()
    weights = torch.softmax(torch.randn(tokens, top_k, device="cuda"), -1)
    topk = types.SimpleNamespace(topk_weights=weights, topk_ids=ids)
    got = method._apply_vec(layer, x, topk).float().cpu()

    want = torch.zeros(tokens, 4096)
    cache = {}
    for i in range(tokens):
        for k in range(top_k):
            e = int(ids[i, k])
            if e not in cache:
                cache[e] = (reference(gate, e), reference(up, e), reference(down, e))
            g_w, u_w, d_w = cache[e]
            xi = x[i].float().cpu()
            g = (g_w @ xi).clamp(max=10.0)
            u = (u_w @ xi).clamp(min=-10.0, max=10.0)
            want[i] += float(weights[i, k]) * (d_w @ (torch.nn.functional.silu(g) * u))
    want *= 2.5

    rel = (got - want).norm() / want.norm()
    # the MMVQ kernel quantizes activations to q8_1, so this is not exact
    assert rel < 0.03, float(rel)
