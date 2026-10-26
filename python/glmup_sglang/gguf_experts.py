"""`glmup_gguf`: GLM-5.3-Flash with routed experts still in GGUF blocks.

Everything outside the routed experts is plain bf16 and goes through
sglang's unquantized paths. Each FusedMoE layer keeps its experts as raw
GGUF rows (uint8 [experts, rows, row_bytes]) and picks one of two paths:

* few tokens: ggml's MMVQ kernel reads the quantized rows directly, one
  matvec per (token, expert). This is the decode path.
* many tokens: the layer's experts are dequantized to bf16 and handed to
  sglang's own triton MoE runner, so prefill gets a real grouped GEMM and
  the same activation clamp and routed scaling as a bf16 checkpoint.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

import gguf
import torch
from gguf import GGMLQuantizationType
from torch.nn import Parameter

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe.moe_runner import MoeRunner, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import set_weight_attrs

NAME = "glmup_gguf"
VEC_MAX_TOKENS = int(os.environ.get("GLMUP_VEC_MAX_TOKENS", "16"))


def row_bytes(qtype: GGMLQuantizationType, cols: int) -> int:
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    if cols % block:
        raise ValueError(f"{cols} columns is not a whole number of {qtype.name} blocks ({block})")
    return cols // block * size


# ggml_dequantize fails with "invalid argument" on a whole layer at once
# (1.2M rows for the down projection), so it runs in slices of experts.
DEQUANT_ROWS = 1 << 17
_scratch: dict[tuple, torch.Tensor] = {}


def dequantize(rows: torch.Tensor, qtype: int, cols: int, dtype: torch.dtype, out: torch.Tensor | None = None) -> torch.Tensor:
    """[experts, n, row_bytes] uint8 -> [experts, n, cols]."""
    from sgl_kernel.quantization import ggml_dequantize

    experts, n = rows.shape[0], rows.shape[1]
    if out is None:
        out = torch.empty(experts, n, cols, dtype=dtype, device=rows.device)
    if qtype == GGMLQuantizationType.IQ3_XXS and rows.is_contiguous():
        from .iq3_xxs import dequantize_iq3_xxs

        return dequantize_iq3_xxs(rows, out)
    step = max(1, DEQUANT_ROWS // n)
    for i in range(0, experts, step):
        part = rows[i : i + step].reshape(-1, rows.shape[-1])
        out[i : i + step].view(-1, cols).copy_(ggml_dequantize(part, qtype, part.shape[0], cols, dtype))
    return out


def scratch(name: str, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """One bf16 buffer per role and shape, shared by every MoE layer on the device.

    Layers run one after another, so the dequantized experts of one layer are
    dead by the time the next layer fills the same buffer.
    """
    key = (name, shape, dtype, device)
    if key not in _scratch:
        _scratch[key] = torch.empty(shape, dtype=dtype, device=device)
    return _scratch[key]


class GlmupGGUFConfig(QuantizationConfig):
    def __init__(self, expert_qtypes: dict[str, dict[str, str]]) -> None:
        super().__init__()
        self.expert_qtypes = expert_qtypes

    def __repr__(self) -> str:
        return f"GlmupGGUFConfig(layers={len(self.expert_qtypes)})"

    def get_name(self) -> str:
        return NAME

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GlmupGGUFConfig":
        return cls(config["expert_qtypes"])

    def get_scaled_act_names(self) -> list[str]:
        return []

    def layer_qtypes(self, prefix: str) -> tuple[GGMLQuantizationType, GGMLQuantizationType]:
        if ".decoder." in prefix:
            # the MTP draft block lives past the last decoder layer
            layer = str(max(int(k) for k in self.expert_qtypes))
        else:
            m = re.search(r"layers\.(\d+)\.", prefix)
            if m is None:
                raise ValueError(f"cannot tell which layer {prefix} is")
            layer = m.group(1)
        q = self.expert_qtypes[layer]
        gate, up, down = (GGMLQuantizationType[q[r]] for r in ("gate_proj", "up_proj", "down_proj"))
        if gate != up:
            raise ValueError(f"layer {layer}: gate {gate.name} and up {up.name} must share a type to fuse")
        return gate, down

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        if isinstance(layer, FusedMoE):
            return GGUFExpertsMoEMethod(*self.layer_qtypes(prefix))
        return None


class GGUFExpertsMoEMethod(FusedMoEMethodBase):
    def __init__(self, w13_qtype: GGMLQuantizationType, w2_qtype: GGMLQuantizationType):
        super().__init__()
        self.w13_qtype = w13_qtype
        self.w2_qtype = w2_qtype

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if layer.moe_ep_size != 1:
            raise ValueError("glmup_gguf experts are tensor-parallel only, drop --ep-size")
        self.hidden_size = hidden_size
        self.inter = intermediate_size_per_partition
        self.params_dtype = params_dtype

        w13 = torch.empty(num_experts, 2 * self.inter, row_bytes(self.w13_qtype, hidden_size), dtype=torch.uint8)
        w2 = torch.empty(num_experts, hidden_size, row_bytes(self.w2_qtype, self.inter), dtype=torch.uint8)
        extra_weight_attrs.pop("weight_loader", None)
        for name, data in (("w13_weight", w13), ("w2_weight", w2)):
            param = Parameter(data, requires_grad=False)
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)
            set_weight_attrs(param, {"weight_loader": self._make_loader(layer)})

    def _make_loader(self, layer):
        def load(param, loaded_weight, weight_name, shard_id, expert_id):
            expert = layer._map_global_expert_id_to_local_expert_id(expert_id)
            if expert == -1:
                return
            rank, inter = layer.moe_tp_rank, self.inter
            dst = param.data[expert]
            if shard_id in ("w1", "w3"):
                rows = loaded_weight.narrow(0, rank * inter, inter)
                start = 0 if shard_id == "w1" else inter
                dst[start : start + inter].copy_(rows)
            elif shard_id == "w2":
                # GGUF blocks run along the input dim, so a TP shard of the
                # down projection is a contiguous byte range of every row.
                width = dst.shape[1]
                dst.copy_(loaded_weight.narrow(1, rank * width, width))
            else:
                raise ValueError(f"unexpected expert shard {shard_id}")

        return load

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return

    def create_moe_runner(self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def apply(self, layer: torch.nn.Module, dispatch_output):
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        if 0 < x.shape[0] <= VEC_MAX_TOKENS:
            return StandardCombineInput(hidden_states=self._apply_vec(layer, x, dispatch_output.topk_output))
        e = layer.w13_weight.shape[0]
        w13 = scratch("w13", (e, 2 * self.inter, self.hidden_size), x.dtype, x.device)
        w2 = scratch("w2", (e, self.hidden_size, self.inter), x.dtype, x.device)
        dequantize(layer.w13_weight, self.w13_qtype, self.hidden_size, x.dtype, out=w13)
        dequantize(layer.w2_weight, self.w2_qtype, self.inter, x.dtype, out=w2)
        return self.runner.run(dispatch_output, TritonMoeQuantInfo(w13_weight=w13, w2_weight=w2))

    def _apply_vec(self, layer, x: torch.Tensor, topk_output) -> torch.Tensor:
        from sgl_kernel import moe_sum
        from sgl_kernel.quantization import ggml_moe_a8_vec

        from sglang.kernels.ops.activation.activation import silu_and_mul

        cfg = self.moe_runner_config
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        tokens, top_k = topk_ids.shape

        gate_up = ggml_moe_a8_vec(x, layer.w13_weight, topk_ids, top_k, self.w13_qtype, 2 * self.inter, tokens)
        act = torch.empty(gate_up.shape[0], self.inter, dtype=x.dtype, device=x.device)
        if cfg.swiglu_limit is not None:
            from sglang.kernels.ops.moe.dsv4 import silu_and_mul_clamp

            silu_and_mul_clamp(gate_up, act, float(cfg.swiglu_limit))
        else:
            silu_and_mul(gate_up, act)
        down = ggml_moe_a8_vec(act, layer.w2_weight, topk_ids, 1, self.w2_qtype, self.hidden_size, tokens * top_k)
        down = down.view(tokens, top_k, self.hidden_size).mul_(topk_weights.view(tokens, top_k, 1).to(down.dtype))
        out = torch.empty_like(x)
        moe_sum(down, out)
        if cfg.routed_scaling_factor not in (None, 1.0):
            out.mul_(cfg.routed_scaling_factor)
        return out
