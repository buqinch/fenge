"""Single DINOv3 ViT encoder; native prefix tokens and RoPE are retained."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import timm
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


VIT_MODEL = "vit_small_patch16_dinov3.lvd1689m"
PRETRAINED_SHA256 = {
    "dinov3_vit_small": "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GradientSafeGELU(nn.GELU):
    """Keep PyTorch 2.1 CPU GELU backward gradients contiguous."""
    def forward(self, x: Tensor) -> Tensor:
        y = super().forward(x)
        if y.device.type == "cpu" and y.requires_grad:
            y.register_hook(lambda gradient: gradient.contiguous())
        return y


def _safe_gelu(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            setattr(module, name, GradientSafeGELU(approximate=child.approximate))
        else:
            _safe_gelu(child)


def load_pretrained(module: nn.Module, directory: Path, expected_architecture: str) -> dict:
    path = directory / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Missing local pretrained weights: {path}")
    actual = sha256(path)
    expected = PRETRAINED_SHA256[directory.name]
    if actual != expected:
        raise ValueError(f"Pretrained SHA256 mismatch: {path}: {actual} != {expected}")
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing pretrained configuration: {config_path}")
    config = json.loads(config_path.read_text())
    if config.get("architecture") != expected_architecture:
        raise ValueError(f"Wrong pretrained architecture in {config_path}")
    from safetensors.torch import load_file
    state = load_file(str(path), device="cpu")
    module.load_state_dict(state, strict=True)
    return {"loaded": True, "path": str(path.resolve()), "sha256": actual,
            "config_sha256": sha256(config_path), "architecture": expected_architecture,
            "tensor_count": len(state), "strict": True, "format": "timm_safetensors"}


class _FrozenBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self._frozen = False

    def freeze(self, frozen: bool = True) -> None:
        if type(frozen) is not bool:
            raise ValueError("frozen must be boolean")
        self._frozen = frozen
        for parameter in self.parameters():
            parameter.requires_grad_(not frozen)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(False if self._frozen else mode)
        return self


class DINOv3ViTSmall(_FrozenBackbone):
    channels, patch_size, num_prefix_tokens = 384, 16, 5
    feature_indices = (2, 5, 8, 11)

    def __init__(self, weight_dir: str | Path, input_size: int = 1024, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(VIT_MODEL, pretrained=False, img_size=input_size,
                                      dynamic_img_size=True, num_classes=0, drop_path_rate=0.0)
        if (self.model.num_features != 384 or len(self.model.blocks) != 12
                or self.model.num_prefix_tokens != 5 or self.model.reg_token.shape[1] != 4
                or self.model.rope is None or tuple(self.model.patch_embed.patch_size) != (16, 16)):
            raise ValueError("Expected non-plus DINOv3 ViT-S/16: 384 channels, 12 blocks, CLS + 4 registers + RoPE")
        _safe_gelu(self.model)
        self.pretrained_report = {"loaded": False, "architecture": VIT_MODEL}
        if pretrained:
            self.pretrained_report = load_pretrained(self.model, Path(weight_dir) / "dinov3_vit_small",
                                                    "vit_small_patch16_dinov3")

    def forward(self, images: Tensor, gradient_checkpointing: bool = False) -> dict:
        if images.ndim != 4 or images.shape[1] != 3 or any(s % 16 for s in images.shape[-2:]):
            raise ValueError("ViT input must be Bx3xHxW with dimensions divisible by 16")
        grid = (images.shape[-2] // 16, images.shape[-1] // 16)
        tokens = self.model.patch_embed(images)
        tokens, rope = self.model._pos_embed(tokens)
        tokens = self.model.norm_pre(tokens)
        if tokens.shape[1] != grid[0] * grid[1] + self.num_prefix_tokens:
            raise RuntimeError("DINOv3 patch grid/prefix count changed unexpectedly")
        features = []
        final_tokens = None
        for index, block in enumerate(self.model.blocks):
            position = rope[index] if getattr(self.model, "rope_mixed", False) else rope
            if gradient_checkpointing and torch.is_grad_enabled() and (
                    tokens.requires_grad or any(p.requires_grad for p in block.parameters())):
                tokens = checkpoint(block, tokens, rope=position, use_reentrant=False)
            else:
                tokens = block(tokens, rope=position)
            if index in self.feature_indices:
                # Normalize output copies only, never the tokens passed to the next block.
                normalized = self.model.norm(tokens)
                patches = normalized[:, self.num_prefix_tokens:]
                features.append(patches.reshape(len(images), *grid, self.channels).permute(0, 3, 1, 2).contiguous())
                final_tokens = normalized
        if len(features) != 4 or final_tokens is None:
            raise RuntimeError("Expected four normalized ViT feature maps")
        return {"tokens": final_tokens, "patch_features": features[-1],
                "intermediate_features": tuple(features), "grid_size": grid}
