from __future__ import annotations

from typing import Any, Literal, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

NormStyle = Literal["restormer", "instance"]
FfnActivation = Literal["gelu", "celu"]
MdtaTemperature = Literal["softplus", "unit"]
AttentionStyle = Literal["mdta", "conv"]


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, normalized_shape, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B C H W
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        return self.weight * x / torch.sqrt(var + self.eps)


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, normalized_shape, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, normalized_shape, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm over B,C,H,W."""

    def __init__(self, channels: int, with_bias: bool = False) -> None:
        super().__init__()
        self.body: nn.Module = (
            WithBiasLayerNorm(channels) if with_bias else BiasFreeLayerNorm(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def _make_spatial_norm(channels: int, style: NormStyle, layer_norm_with_bias: bool) -> nn.Module:
    """Normalization after channel projection (B,C,H,W).

    * ``restormer`` — original variance-based norm (ONNX tends to include Sqrt; riskier for MLA MXQ).
    * ``instance`` — ``InstanceNorm2d`` (typically a single ONNX ``InstanceNormalization``; PDF: NPU Yes).
    """

    if style == "restormer":
        return LayerNorm2d(channels, with_bias=layer_norm_with_bias)
    if style == "instance":
        return nn.InstanceNorm2d(channels, affine=True, track_running_stats=False)
    raise ValueError(f"Unknown norm style: {style!r}")


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch: int = 3, embed_dim: int = 48) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MDTA(nn.Module):
    """Multi-Dconv head transposed attention (Restormer-style).

    Learnable temperature is strictly positive via softplus (stable softmax scaling),
    or disabled (``temperature_mode="unit"``) to drop Softplus from the ONNX graph.
    """

    def __init__(
        self, dim: int, num_heads: int, bias: bool = False, *, temperature_mode: MdtaTemperature = "softplus"
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature_mode: MdtaTemperature = temperature_mode
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.head_dim = dim // num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)
        self._log_temperature: nn.Parameter | None
        if temperature_mode == "softplus":
            self._log_temperature = nn.Parameter(torch.zeros(num_heads, 1, 1))
        elif temperature_mode == "unit":
            self._log_temperature = None
        else:
            raise ValueError(f"Unknown mdta temperature mode: {temperature_mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1))
        if self.temperature_mode == "softplus":
            assert self._log_temperature is not None
            temperature = F.softplus(self._log_temperature) + 1e-6
            attn = attn * temperature
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.reshape(b, c, h, w)
        return self.project_out(out)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        old_key = prefix + "temperature"
        new_key = prefix + "_log_temperature"
        if old_key in state_dict and new_key not in state_dict:
            t = state_dict.pop(old_key)
            y = t.clamp_min(1e-6)
            state_dict[new_key] = torch.log(torch.expm1(y))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class GDFN(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool = False) -> None:
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float,
        bias: bool = False,
        layer_norm_with_bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim, with_bias=layer_norm_with_bias)
        self.attn = MDTA(dim, num_heads, bias=bias)
        self.norm2 = LayerNorm2d(dim, with_bias=layer_norm_with_bias)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


def _make_level(blocks: int, dim: int, heads: int, ffn: float, bias: bool, ln_bias: bool) -> nn.Sequential:
    return nn.Sequential(
        *[
            TransformerBlock(dim, heads, ffn, bias=bias, layer_norm_with_bias=ln_bias)
            for _ in range(blocks)
        ]
    )


class Downsample(nn.Module):
    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.body = nn.Conv2d(in_ch, in_ch * 2, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    """in_ch -> out_ch spatial x2 via PixelShuffle."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class RestormerTeacher(nn.Module):
    """
    Same-resolution Restormer-style encoder–decoder with global residual in RGB.
    Output: clamp(lr + head(features), 0, 1).
    """

    def __init__(
        self,
        dim: int = 48,
        num_blocks: tuple[int, int, int, int] = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads: tuple[int, int, int, int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_with_bias: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.patch_embed = OverlapPatchEmbed(3, dim)
        enc_block_counts = num_blocks
        self.encoder_level1 = _make_level(
            enc_block_counts[0], dim, heads[0], ffn_expansion_factor, bias, layer_norm_with_bias
        )
        self.down1_2 = Downsample(dim)
        c2 = dim * 2
        self.encoder_level2 = _make_level(
            enc_block_counts[1], c2, heads[1], ffn_expansion_factor, bias, layer_norm_with_bias
        )
        self.down2_3 = Downsample(c2)
        c3 = c2 * 2
        self.encoder_level3 = _make_level(
            enc_block_counts[2], c3, heads[2], ffn_expansion_factor, bias, layer_norm_with_bias
        )
        self.down3_4 = Downsample(c3)
        c4 = c3 * 2
        self.encoder_level4 = _make_level(
            enc_block_counts[3], c4, heads[3], ffn_expansion_factor, bias, layer_norm_with_bias
        )

        self.up4_3 = Upsample(c4, c3)
        self.reduce_chan_level3 = nn.Conv2d(c3 * 2, c3, kernel_size=1, bias=False)
        self.decoder_level3 = _make_level(
            enc_block_counts[2], c3, heads[2], ffn_expansion_factor, bias, layer_norm_with_bias
        )

        self.up3_2 = Upsample(c3, c2)
        self.reduce_chan_level2 = nn.Conv2d(c2 * 2, c2, kernel_size=1, bias=False)
        self.decoder_level2 = _make_level(
            enc_block_counts[1], c2, heads[1], ffn_expansion_factor, bias, layer_norm_with_bias
        )

        self.up2_1 = Upsample(c2, dim)
        self.reduce_chan_level1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.decoder_level1 = _make_level(
            enc_block_counts[0], dim, heads[0], ffn_expansion_factor, bias, layer_norm_with_bias
        )

        self.refinement = _make_level(
            num_refinement_blocks, dim, heads[0], ffn_expansion_factor, bias, layer_norm_with_bias
        )
        self.output = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, lr_rgb: torch.Tensor) -> torch.Tensor:
        inp = lr_rgb
        x = self.patch_embed(inp)
        enc1 = self.encoder_level1(x)
        x = self.down1_2(enc1)
        enc2 = self.encoder_level2(x)
        x = self.down2_3(enc2)
        enc3 = self.encoder_level3(x)
        x = self.down3_4(enc3)
        x = self.encoder_level4(x)
        x = self.up4_3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.reduce_chan_level3(x)
        x = self.decoder_level3(x)
        x = self.up3_2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.reduce_chan_level2(x)
        x = self.decoder_level2(x)
        x = self.up2_1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.reduce_chan_level1(x)
        x = self.decoder_level1(x)
        x = self.refinement(x)
        residual = self.output(x)
        return (inp + residual).clamp(0.0, 1.0)


def build_teacher_model(architecture: Any) -> RestormerTeacher:
    if isinstance(architecture, Mapping):
        dim = int(architecture["dim"])
        num_blocks = tuple(int(x) for x in architecture["num_blocks"])
        num_refinement_blocks = int(architecture["num_refinement_blocks"])
        heads = tuple(int(x) for x in architecture["heads"])
        ffn_expansion_factor = float(architecture["ffn_expansion_factor"])
    else:
        dim = int(architecture.dim)
        num_blocks = tuple(int(x) for x in architecture.num_blocks)
        num_refinement_blocks = int(architecture.num_refinement_blocks)
        heads = tuple(int(x) for x in architecture.heads)
        ffn_expansion_factor = float(architecture.ffn_expansion_factor)
    return RestormerTeacher(
        dim=dim,
        num_blocks=num_blocks,
        num_refinement_blocks=num_refinement_blocks,
        heads=heads,
        ffn_expansion_factor=ffn_expansion_factor,
    )
