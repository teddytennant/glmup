"""IQ3_XXS -> bf16 in one Triton pass, straight into the destination buffer.

sgl_kernel's ggml_dequantize took ~40 ms per MoE layer here (it also needs a
copy, because it allocates its own output). The layer is ~1.4 GB of IQ3 in and
~7 GB of bf16 out per GPU, so the memory-bound floor is ~2 ms.

Block layout (256 values, 98 bytes): fp16 d, 64 grid indices (4 values each),
8 uint32 of scale (top 4 bits) and four 7-bit sign indices per 32 values.
"""

import torch
import triton
import triton.language as tl

BLOCK_BYTES = 98
_tables: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if device not in _tables:
        from gguf.quants import IQ2_XXS, IQ3_XXS

        IQ3_XXS.init_grid()
        grid = torch.from_numpy(IQ3_XXS.grid.reshape(256, 4).astype("float32").copy())
        ksigns = torch.frombuffer(bytearray(IQ2_XXS.ksigns), dtype=torch.uint8)
        _tables[device] = (grid.to(device).reshape(-1), ksigns.to(torch.int32).to(device))
    return _tables[device]


@triton.jit
def _dequant(src, dst, grid, ksigns, n_blocks, BLOCKS: tl.constexpr):
    b = tl.program_id(0) * BLOCKS + tl.arange(0, BLOCKS)[:, None]
    v = tl.arange(0, 256)[None, :]
    mask = b < n_blocks
    base = src + b.to(tl.int64) * 98

    d_lo = tl.load(base, mask=mask, other=0).to(tl.uint16)
    d_hi = tl.load(base + 1, mask=mask, other=0).to(tl.uint16)
    d = (d_lo | (d_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

    q = tl.load(base + 2 + v // 4, mask=mask, other=0).to(tl.int32)
    g = tl.load(grid + q * 4 + v % 4)

    s = base + 66 + (v // 32) * 4
    aux = (
        tl.load(s, mask=mask, other=0).to(tl.uint32)
        | (tl.load(s + 1, mask=mask, other=0).to(tl.uint32) << 8)
        | (tl.load(s + 2, mask=mask, other=0).to(tl.uint32) << 16)
        | (tl.load(s + 3, mask=mask, other=0).to(tl.uint32) << 24)
    )
    sidx = ((aux >> ((v % 32) // 8 * 7).to(tl.uint32)) & 127).to(tl.int32)
    neg = (tl.load(ksigns + sidx) >> (v % 8)) & 1
    db = d * (0.5 + (aux >> 28).to(tl.float32)) * 0.5
    y = db * g * (1 - 2 * neg).to(tl.float32)
    tl.store(dst + b.to(tl.int64) * 256 + v, y.to(dst.dtype.element_ty), mask=mask)


def dequantize_iq3_xxs(rows: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """rows: uint8 [..., row_bytes], out: [..., row_bytes // 98 * 256], both contiguous."""
    assert rows.is_contiguous() and out.is_contiguous()
    n_blocks = rows.numel() // BLOCK_BYTES
    assert out.numel() == n_blocks * 256
    grid, ksigns = tables(rows.device)
    blocks = 8
    _dequant[(triton.cdiv(n_blocks, blocks),)](rows, out, grid, ksigns, n_blocks, BLOCKS=blocks, num_warps=8)
    return out
