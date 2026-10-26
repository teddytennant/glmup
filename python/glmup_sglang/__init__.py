"""sglang plugin that serves GLM-5.3-Flash with its routed experts left in GGUF form."""


def register() -> None:
    """sglang.srt.plugins entry point: make `glmup_gguf` a known quant method."""
    from sglang.srt.arg_groups import choices
    from sglang.srt.layers import quantization

    from .gguf_experts import NAME, GlmupGGUFConfig

    quantization.QUANTIZATION_METHODS[NAME] = GlmupGGUFConfig
    quantization.BASE_QUANTIZATION_METHODS[NAME] = GlmupGGUFConfig
    if NAME not in choices.QUANTIZATION_CHOICES:
        choices.QUANTIZATION_CHOICES.append(NAME)
