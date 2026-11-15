"""Strict local DINOv3 encoders; native prefix tokens and RoPE are retained."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import timm
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


VIT_MODEL = "vit_small_patch16_dinov3.lvd1689m"
VIT_SPECS = {
    "small": ("vit_small_patch16_dinov3", 384, 12, 6),
    "base": ("vit_base_patch16_dinov3", 768, 12, 12),
    "large": ("vit_large_patch16_dinov3", 1024, 24, 16),
}
CNN_MODEL = "convnext_tiny.dinov3_lvd1689m"
PRETRAINED_SHA256 = {
    "dinov3_vit_small": "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040",
    "dinov3_vit_base": "1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332",
    "dinov3_vit_large": "45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941",
    "dinov3_convnext_tiny": "a7637753a2e85d709d1a5474e13382d39e7bc581a35868b477585c970f36847f",
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


class DINOv3ViT(_FrozenBackbone):
    patch_size, num_prefix_tokens = 16, 5

    def __init__(self, weight_dir: str | Path, input_size: int = 1024, pretrained: bool = True,
                 size: str = "small"):
        super().__init__()
        if size not in VIT_SPECS:
            raise ValueError(f"Unknown DINOv3 ViT size: {size!r}; expected small, base or large")
        architecture, self.channels, depth, heads = VIT_SPECS[size]
        self.size = size
        self.blocks_per_stage = depth // 4
        model_name = architecture + ".lvd1689m"
        self.model = timm.create_model(model_name, pretrained=False, img_size=input_size,
                                      dynamic_img_size=True, num_classes=0, drop_path_rate=0.0)
        if (self.model.num_features != self.channels or len(self.model.blocks) != depth
                or self.model.num_prefix_tokens != 5 or self.model.reg_token.shape[1] != 4
                or self.model.rope is None or tuple(self.model.patch_embed.patch_size) != (16, 16)
                or any(block.attn.num_heads != heads for block in self.model.blocks)):
            raise ValueError(f"Expected DINOv3 ViT-{size}/16: {self.channels} channels, {depth} blocks, "
                             f"{heads} heads, CLS + 4 registers + RoPE")
        _safe_gelu(self.model)
        self.pretrained_report = {"loaded": False, "architecture": model_name}
        if pretrained:
            self.pretrained_report = load_pretrained(self.model, Path(weight_dir) / f"dinov3_vit_{size}",
                                                    architecture)

    def forward(self, images: Tensor, before_stage=None, after_stage=None,
                gradient_checkpointing: bool = False) -> dict:
        grid = (images.shape[-2] // 16, images.shape[-1] // 16)
        tokens = self.model.patch_embed(images)
        tokens, rope = self.model._pos_embed(tokens)
        tokens = self.model.norm_pre(tokens)
        if tokens.shape[1] != grid[0] * grid[1] + self.num_prefix_tokens:
            raise RuntimeError("DINOv3 patch grid/prefix count changed unexpectedly")
        for stage in range(4):
            if before_stage is not None:
                tokens = before_stage(stage, tokens, grid)
            for index in range(stage * self.blocks_per_stage, (stage + 1) * self.blocks_per_stage):
                block = self.model.blocks[index]
                position = rope[index] if getattr(self.model, "rope_mixed", False) else rope
                if gradient_checkpointing and torch.is_grad_enabled() and (
                        tokens.requires_grad or any(p.requires_grad for p in block.parameters())):
                    tokens = checkpoint(block, tokens, rope=position, use_reentrant=False)
                else:
                    tokens = block(tokens, rope=position)
            if after_stage is not None:
                after_stage(stage, self.model.norm(tokens), grid)
        tokens = self.model.norm(tokens)
        patches = tokens[:, self.num_prefix_tokens:]
        return {"tokens": tokens, "patch_features": patches.transpose(1, 2).reshape(
            len(images), self.channels, *grid), "grid_size": grid}


class DINOv3ViTSmall(DINOv3ViT):
    """Backward-compatible small backbone entry point and state dictionary layout."""
    channels = 384

    def __init__(self, weight_dir: str | Path, input_size: int = 1024, pretrained: bool = True):
        super().__init__(weight_dir, input_size, pretrained, size="small")


class DINOv3ConvNeXtTiny(_FrozenBackbone):
    channels, strides = (96, 192, 384, 768), (4, 8, 16, 32)

    def __init__(self, weight_dir: str | Path, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(CNN_MODEL, pretrained=False, num_classes=0, drop_path_rate=0.0)
        _safe_gelu(self.model)
        self.pretrained_report = {"loaded": False, "architecture": CNN_MODEL}
        if pretrained:
            self.pretrained_report = load_pretrained(self.model, Path(weight_dir) / "dinov3_convnext_tiny",
                                                    "convnext_tiny")

    def forward(self, images: Tensor, gradient_checkpointing: bool = False) -> list[Tensor]:
        active = gradient_checkpointing and torch.is_grad_enabled()
        if active and (images.requires_grad or any(p.requires_grad for p in self.model.stem.parameters())):
            x = checkpoint(self.model.stem, images, use_reentrant=False)
        else:
            x = self.model.stem(images)
        features = []
        for stage in self.model.stages:
            if active and (x.requires_grad or any(p.requires_grad for p in stage.parameters())):
                x = checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
            features.append(x)
        return features


class DualDINOBackbones(nn.Module):
    def __init__(self, weight_dir: str | Path, input_size: int, pretrained: bool = True,
                 vit_size: str = "small"):
        super().__init__()
        self.vit = DINOv3ViT(weight_dir, input_size, pretrained, size=vit_size)
        self.convnext = DINOv3ConvNeXtTiny(weight_dir, pretrained)

    def freeze(self, frozen: bool = True) -> None:
        self.vit.freeze(frozen)
        self.convnext.freeze(frozen)

    @property
    def pretrained_report(self):
        return {"vit": self.vit.pretrained_report, "convnext": self.convnext.pretrained_report}
