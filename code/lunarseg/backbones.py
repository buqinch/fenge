"""Local-only DINOv3 ConvNeXt and MobileNetV3 feature encoders.

The ConvNeXt stage layout and tensor names follow the official Hugging Face
DINOv3 ConvNeXt implementation. Both branches receive the same full resolution
tensor; preprocessing belongs to the data loader, never to a backbone.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torchvision.models import mobilenet_v3_large


class GradientSafeGELU(nn.GELU):
    """Keep CPU backward gradients contiguous (PyTorch 2.1 compatibility).

    Query cross attention can return a channels-last gradient to spatial GELU.
    PyTorch 2.1's CPU GELU backward can produce invalid values for that stride
    pattern. The hook changes only storage layout, never values or CUDA code.
    """
    def forward(self, x: Tensor) -> Tensor:
        result = super().forward(x)
        if result.device.type == "cpu" and result.requires_grad:
            result.register_hook(lambda gradient: gradient.contiguous())
        return result


class ChannelLayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6, layer_scale: float = 1e-6):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.layer_norm = nn.LayerNorm(channels, eps=eps)
        self.pointwise_conv1 = nn.Linear(channels, 4 * channels)
        self.pointwise_conv2 = nn.Linear(4 * channels, channels)
        self.activation = GradientSafeGELU()
        self.gamma = nn.Parameter(torch.full((channels,), layer_scale))

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.depthwise_conv(x).permute(0, 2, 3, 1)
        x = self.pointwise_conv2(self.activation(self.pointwise_conv1(self.layer_norm(x))))
        return residual + (x * self.gamma).permute(0, 3, 1, 2)


class ConvNeXtStage(nn.Module):
    def __init__(self, in_channels: int, channels: int, depth: int, first: bool, eps: float, layer_scale: float):
        super().__init__()
        if first:
            self.downsample_layers = nn.ModuleList([
                nn.Conv2d(3, channels, 4, stride=4), ChannelLayerNorm(channels, eps=eps),
            ])
        else:
            self.downsample_layers = nn.ModuleList([
                ChannelLayerNorm(in_channels, eps=eps), nn.Conv2d(in_channels, channels, 2, stride=2),
            ])
        self.layers = nn.ModuleList([ConvNeXtBlock(channels, eps, layer_scale) for _ in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.downsample_layers:
            x = layer(x)
        for layer in self.layers:
            x = layer(x)
        return x


def _read_state(path: Path) -> dict[str, Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing pretrained checkpoint: {path}")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model", "teacher"):
        if isinstance(state, Mapping) and key in state and isinstance(state[key], Mapping):
            state = state[key]
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint does not contain a state dictionary: {path}")
    return dict(state)


def convert_timm_convnext_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Lossless 180-tensor name conversion, with no discarded classifier.

    timm's DINOv3 ConvNeXt checkpoint has num_classes=0. Its head.norm is
    the HF global-output layer_norm; all four spatial stages are identical.
    Unknown keys remain unknown so strict loading catches incompatible models.
    """
    converted = {}
    for key, value in state.items():
        if key.startswith("stem."):
            key = "stages.0.downsample_layers." + key[len("stem."):]
        elif key.startswith("head.norm."):
            key = "layer_norm." + key[len("head.norm."):]
        else:
            key = re.sub(r"^(stages\.\d+)\.downsample\.", r"\1.downsample_layers.", key)
            key = re.sub(r"^(stages\.\d+)\.blocks\.(\d+)\.", r"\1.layers.\2.", key)
            key = key.replace(".conv_dw.", ".depthwise_conv.")
            key = key.replace(".norm.", ".layer_norm.")
            key = key.replace(".mlp.fc1.", ".pointwise_conv1.")
            key = key.replace(".mlp.fc2.", ".pointwise_conv2.")
        if key in converted:
            raise ValueError(f"Duplicate ConvNeXt key after conversion: {key}")
        converted[key] = value
    return converted


class DINOv3ConvNeXtTiny(nn.Module):
    channels = (96, 192, 384, 768)
    strides = (4, 8, 16, 32)

    def __init__(self, checkpoint: Path | None = None, config_path: Path | None = None):
        super().__init__()
        config = json.loads(config_path.read_text()) if config_path and config_path.is_file() else {}
        if tuple(config.get("hidden_sizes", self.channels)) != self.channels or tuple(config.get("depths", (3, 3, 9, 3))) != (3, 3, 9, 3):
            raise ValueError("Expected the DINOv3 ConvNeXt-Tiny 96/192/384/768, 3/3/9/3 architecture")
        if config.get("hidden_act", "gelu") != "gelu" or config.get("drop_path_rate", 0.0) != 0.0:
            raise ValueError("This encoder implements the pretrained DINOv3 GELU, zero-drop-path configuration")
        eps = float(config.get("layer_norm_eps", 1e-6))
        scale = float(config.get("layer_scale_init_value", 1e-6))
        self.stages = nn.ModuleList([
            ConvNeXtStage(self.channels[i - 1] if i else 3, c, d, i == 0, eps, scale)
            for i, (c, d) in enumerate(zip(self.channels, (3, 3, 9, 3)))
        ])
        # Retained to load the complete official checkpoint strictly. Stage
        # features are intentionally taken before this classification-only norm.
        self.layer_norm = nn.LayerNorm(self.channels[-1], eps=eps)
        self.pretrained_report: dict = {"loaded": False}
        if checkpoint is not None:
            self.load_pretrained(checkpoint)

    def load_pretrained(self, checkpoint: Path) -> None:
        state = _read_state(checkpoint)
        source_format = "huggingface"
        for prefix in ("base.backbone.", "encoder.convnext.", "dinov3_convnext.", "module."):
            selected = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if selected and "stages.0.downsample_layers.0.weight" in selected:
                state = selected
                source_format = prefix
                break
        if "stem.0.weight" in state:
            state = convert_timm_convnext_state(state)
            source_format = "timm_dinov3"
        self.load_state_dict(state, strict=True)
        self.pretrained_report = {"loaded": True, "path": str(checkpoint), "tensor_count": len(state), "strict": True, "source_format": source_format}

    def forward(self, images: Tensor) -> list[Tensor]:
        outputs = []
        x = images
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


class MobileNetV3LargeFeatures(nn.Module):
    channels = (24, 40, 112, 960)
    strides = (4, 8, 16, 32)
    feature_indices = (3, 6, 12, 16)

    def __init__(self, checkpoint: Path | None = None):
        super().__init__()
        full_model = mobilenet_v3_large(weights=None)
        self.pretrained_report: dict = {"loaded": False}
        if checkpoint is not None:
            state = _read_state(checkpoint)
            full_model.load_state_dict(state, strict=True)
            self.pretrained_report = {"loaded": True, "path": str(checkpoint), "tensor_count": len(state), "strict": True}
        self.features = full_model.features

    def forward(self, images: Tensor) -> list[Tensor]:
        outputs = []
        x = images
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in self.feature_indices:
                outputs.append(x)
        return outputs


class DualCNNEncoder(nn.Module):
    def __init__(self, weight_dir: str | Path, pretrained: bool = True, backbone: str = "dual", convnext_checkpoint: str | None = None):
        super().__init__()
        root = Path(weight_dir)
        dino_root = root / "dinov3_convnext_tiny"
        dino_path = Path(convnext_checkpoint) if convnext_checkpoint else dino_root / "model.safetensors"
        self.convnext = DINOv3ConvNeXtTiny(dino_path if pretrained else None, dino_root / "config.json")
        self.mobile = MobileNetV3LargeFeatures(root / "mobilenet_v3_large-5c1a4163.pth" if pretrained else None) if backbone == "dual" else None
        self._frozen = False

    def freeze(self, frozen: bool = True) -> None:
        self._frozen = frozen
        for p in self.parameters():
            p.requires_grad_(not frozen)
        if frozen:
            self.eval()
        else:
            self.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._frozen:
            super().train(False)
        return self

    def forward(self, images: Tensor) -> tuple[list[Tensor], list[Tensor] | None]:
        convnext = self.convnext(images)
        mobile = self.mobile(images) if self.mobile is not None else None
        return convnext, mobile
