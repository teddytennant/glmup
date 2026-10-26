"""Time each piece of one MoE layer (TP rank 0 of tp=2) on real GGUF experts.

    python -m glmup_sglang.profile_moe GGUF_DIR
"""

import sys
import types
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader

from .gguf_experts import GGUFExpertsMoEMethod, dequantize, row_bytes


def timed(fn, reps=5):
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def main():
    gguf_dir = Path(sys.argv[1])
    t = {}
    for p in sorted(gguf_dir.glob("*.gguf")):
        for x in GGUFReader(str(p)).tensors:
            for role in ("gate", "up", "down"):
                if x.name == f"blk.3.ffn_{role}_exps.weight":
                    t[role] = x
    inter = 1024
    qt = t["gate"].tensor_type
    g = torch.from_numpy(np.ascontiguousarray(t["gate"].data))[:, :inter]
    u = torch.from_numpy(np.ascontiguousarray(t["up"].data))[:, :inter]
    w13 = torch.cat([g, u], 1).contiguous().cuda()
    w2 = torch.from_numpy(np.ascontiguousarray(t["down"].data))[:, :, : row_bytes(qt, inter)].contiguous().cuda()
    o13 = torch.empty(288, 2 * inter, 4096, dtype=torch.bfloat16, device="cuda")
    o2 = torch.empty(288, 4096, inter, dtype=torch.bfloat16, device="cuda")

    from . import gguf_experts

    print(f"triton dequant w13+w2: {timed(lambda: (dequantize(w13, qt, 4096, torch.bfloat16, o13), dequantize(w2, qt, inter, torch.bfloat16, o2))):.2f} ms")
    w13_nc = w13[:, :, :]
    def ggml_path():
        from sgl_kernel.quantization import ggml_dequantize

        step = gguf_experts.DEQUANT_ROWS // w13.shape[1]
        for i in range(0, 288, step):
            part = w13_nc[i : i + step].reshape(-1, w13.shape[-1])
            o13[i : i + step].view(-1, 4096).copy_(ggml_dequantize(part, qt, part.shape[0], 4096, torch.bfloat16))
    print(f"ggml dequant w13 only: {timed(ggml_path):.2f} ms")

    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config as cfgmod
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts_impl

    # outside a server there is no runtime context; the config lookup only asks this
    cfgmod.get_exec = lambda: types.SimpleNamespace(deterministic=types.SimpleNamespace(enable_deterministic_inference=False))

    method = GGUFExpertsMoEMethod(qt, qt)
    method.hidden_size, method.inter = 4096, inter
    method.moe_runner_config = types.SimpleNamespace(swiglu_limit=10.0, routed_scaling_factor=2.5)
    layer = types.SimpleNamespace(w13_weight=w13, w2_weight=w2)
    for n in (1, 8, 32, 64, 128, 256, 1024, 8192):
        x = torch.randn(n, 4096, device="cuda", dtype=torch.bfloat16) * 0.5
        ids = torch.randint(0, 288, (n, 8), device="cuda", dtype=torch.int32)
        w = torch.softmax(torch.randn(n, 8, device="cuda"), -1)
        topk = types.SimpleNamespace(topk_weights=w, topk_ids=ids)
        row = f"T={n:5d}"
        if n <= 256:
            row += f"  vec {timed(lambda: method._apply_vec(layer, x, topk)):8.2f} ms"
        gemm = timed(lambda: fused_experts_impl(x, o13, o2, w, ids, swiglu_limit=10.0, routed_scaling_factor=2.5))
        row += f"  triton gemm (weights ready) {gemm:8.2f} ms"
        print(row, flush=True)


if __name__ == "__main__":
    main()
