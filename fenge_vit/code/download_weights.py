#!/usr/bin/env python3
"""Download pinned public DINOv3 weights and verify their SHA256 before use."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'dinov3_vit_small': {
        'repository': 'timm/vit_small_patch16_dinov3.lvd1689m',
        'revision': '3bf4720a82ec2066db88137180ff1f83a675cef0',
        'sha256': '2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040',
        'size_bytes': 86362376,
    },
    'dinov3_convnext_tiny': {
        'repository': 'timm/convnext_tiny.dinov3_lvd1689m',
        'revision': '7acfec1',
        'sha256': 'a7637753a2e85d709d1a5474e13382d39e7bc581a35868b477585c970f36847f',
        'size_bytes': 111298008,
    },
}

def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def download(url, path, opener, expected=None):
    if path.exists():
        if expected is not None and digest(path) != expected:
            raise ValueError(f'Existing weight hash differs: {path}; preserve and inspect it before replacing')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    try:
        with opener.open(url, timeout=120) as response, temporary.open('wb') as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        if expected is not None and digest(temporary) != expected:
            raise ValueError(f'Downloaded SHA256 differs from pinned source: {url}')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'model')
    parser.add_argument('--proxy', default=None, help='Optional HTTPS/HTTP proxy; TLS verification remains enabled')
    parser.add_argument('--reuse-convnext', type=Path, default=Path('/home/lzc/workspace/fenge/model/dinov3_convnext_tiny'))
    args = parser.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'http':args.proxy, 'https':args.proxy})) if args.proxy else urllib.request.build_opener()
    rows = []
    for name, spec in SOURCES.items():
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=True)
        candidate = args.reuse_convnext / 'model.safetensors' if name == 'dinov3_convnext_tiny' else None
        if candidate is not None and candidate.is_file() and not (folder/'model.safetensors').exists():
            if digest(candidate) != spec['sha256']:
                raise ValueError('Local ConvNeXt source hash mismatch')
            shutil.copy2(candidate, folder/'model.safetensors')
            if (args.reuse_convnext/'config.json').is_file():
                shutil.copy2(args.reuse_convnext/'config.json', folder/'config.json')
        for filename in ['config.json', 'model.safetensors', 'LICENSE.md']:
            url = f"https://huggingface.co/{spec['repository']}/resolve/{spec['revision']}/{filename}"
            if filename == 'LICENSE.md':
                # The DINOv3 license is shared; the ConvNeXt mirror at its
                # pinned revision does not contain a separate license file.
                license_source = SOURCES['dinov3_vit_small']
                url = f"https://huggingface.co/{license_source['repository']}/resolve/{license_source['revision']}/LICENSE.md"
            path = folder / filename
            download(url, path, opener, spec['sha256'] if filename=='model.safetensors' else None)
            if filename=='model.safetensors' and path.stat().st_size != spec['size_bytes']:
                raise ValueError('Weight size differs from pinned source')
            rows.append({'path':str(path.relative_to(args.output)), 'sha256':digest(path), 'bytes':path.stat().st_size, 'source_url':url})
        print(name + ': verified', flush=True)
    report = {'schema':'lusnar_vit_pretrained_v1','created_at':datetime.now(timezone.utc).isoformat(),
              'sources':SOURCES, 'files':rows, 'pretraining_only':True, 'license':'dinov3-license',
              'tls_verification':True,'training_does_not_download_weights':True}
    (args.output/'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')

if __name__=='__main__':
    main()
