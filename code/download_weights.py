#!/usr/bin/env python3
"""Download the two official pretrained CNN backbones into this project's model/.

No Hugging Face login, token, or global cache is read. Existing files are checked
before reuse. A TLS certificate workaround is opt-in and is only allowed when
the COMPLETE SHA256 of a public file is pinned; bytes must still match exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import ssl
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DINO_REPO = "https://huggingface.co/timm/convnext_tiny.dinov3_lvd1689m"
DINO_REVISION = "7acfec1"
# Full checksum published on the official timm Hugging Face file page.
DINO_SHA256 = "a7637753a2e85d709d1a5474e13382d39e7bc581a35868b477585c970f36847f"
# Official config.json content independently checked from its public file view.
DINO_CONFIG_SHA256 = "ca74f08faca70799f011184069ced874174e99cd9a83da966fbc8f176c04498b"
MOBILE_URL = "https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth"
# Downloaded over verified TLS and checked against torchvision's published hash
# prefix; pin the complete digest for all subsequent downloads and offline runs.
MOBILE_SHA256 = "5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _checked(path: Path, expected: str, minimum_size: int) -> str:
    if path.stat().st_size < minimum_size:
        raise ValueError(f"File is too small: {path}")
    digest = sha256(path)
    if not digest.startswith(expected):
        raise ValueError(f"SHA256 mismatch for {path.name}: expected {expected}, got {digest}")
    return digest


def _is_certificate_error(error: Exception) -> bool:
    reason = error.reason if isinstance(error, URLError) else error
    return isinstance(reason, ssl.SSLCertVerificationError)


def download(url: str, destination: Path, expected_hash: str, minimum_size: int,
             allow_pinned_tls_fallback: bool, ca_bundle: str | None,
             previous_record: dict | None = None) -> dict:
    expected = expected_hash
    if destination.is_file():
        digest = _checked(destination, expected, minimum_size)
        if previous_record and digest != previous_record.get("sha256"):
            raise ValueError(f"Existing file differs from its recorded full SHA256: {destination}")
        print(f"Verified existing {destination.name} ({destination.stat().st_size} bytes)", flush=True)
        return {"path": display_path(destination), "sha256": digest,
                "size_bytes": destination.stat().st_size, "source_url": url,
                "tls_certificate_fallback": (previous_record or {}).get("tls_certificate_fallback", False)}

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    context = ssl.create_default_context(cafile=ca_bundle)
    request = Request(url, headers={"User-Agent": "lunarseg-official-weight-downloader/1.0"})
    insecure = False
    try:
        response = urlopen(request, timeout=60, context=context)
    except Exception as error:
        if not (_is_certificate_error(error) and allow_pinned_tls_fallback and len(expected_hash) == 64):
            raise
        # The independently verified full content hash authenticates this public
        # artifact even when a local intercepting CA is missing from the host.
        print(f"Local TLS CA failure for {destination.name}; enforcing pinned full SHA256.", flush=True)
        response = urlopen(request, timeout=60, context=ssl._create_unverified_context())
        insecure = True

    print(f"Downloading {destination.name}", flush=True)
    last_report = time.monotonic()
    received = 0
    with response, temporary.open("wb") as handle:
        length = response.headers.get("Content-Length")
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            handle.write(chunk)
            received += len(chunk)
            if time.monotonic() - last_report >= 5:
                print(f"  {destination.name}: {received / 1048576:.1f} MiB", flush=True)
                last_report = time.monotonic()
        if length is not None and received != int(length):
            raise ValueError(f"Incomplete download: {received} of {length} bytes")
    digest = _checked(temporary, expected, minimum_size)
    os.replace(temporary, destination)
    print(f"Verified {destination.name}: {digest}", flush=True)
    return {"path": display_path(destination), "sha256": digest,
            "size_bytes": received, "source_url": url, "tls_certificate_fallback": insecure}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "model")
    parser.add_argument("--ca-bundle", help="Optional trusted CA bundle for the local network")
    parser.add_argument("--allow-pinned-tls-fallback", action="store_true",
                        help="On a local TLS CA error, allow ONLY files with a pinned complete SHA256")
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    previous_files = {record["path"]: record for record in previous.get("files", [])}
    requests = [
        (DINO_REPO + f"/resolve/{DINO_REVISION}/config.json", model_dir / "dinov3_convnext_tiny/config.json", DINO_CONFIG_SHA256, 100),
        (DINO_REPO + f"/resolve/{DINO_REVISION}/model.safetensors", model_dir / "dinov3_convnext_tiny/model.safetensors", DINO_SHA256, 100_000_000),
        (MOBILE_URL, model_dir / "mobilenet_v3_large-5c1a4163.pth", MOBILE_SHA256, 20_000_000),
    ]
    records = []
    for url, path, expected, minimum_size in requests:
        relative = display_path(path)
        records.append(download(url, path, expected, minimum_size,
                                args.allow_pinned_tls_fallback, args.ca_bundle,
                                previous_files.get(relative)))
    config = json.loads((model_dir / "dinov3_convnext_tiny/config.json").read_text())
    if config.get("architecture") != "convnext_tiny" or config.get("pretrained_cfg", {}).get("tag") != "dinov3_lvd1689m":
        raise ValueError("Downloaded ConvNeXt configuration is not DINOv3 ConvNeXt-Tiny")
    manifest = {
        "schema": "lunarseg_pretrained_backbones_v1", "state": "verified",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "convnext": {"source_repository": DINO_REPO, "revision": DINO_REVISION,
                     "format": "timm", "pretraining": "DINOv3 self-supervised LVD-1689M",
                     "lunar_finetuning": False, "license": "dinov3-license",
                     "hash_source": DINO_REPO + "/blob/main/model.safetensors"},
        "mobilenet": {"source": "torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2",
                      "format": "torchvision_state_dict", "pretraining": "supervised ImageNet-1K",
                      "lunar_finetuning": False,
                      "source_code": "https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py"},
        "files": records,
    }
    temporary = manifest_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, manifest_path)
    print(f"Both pretrained backbones ready: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
