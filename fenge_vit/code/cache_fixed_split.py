#!/usr/bin/env python3
"""Copy an immutable original-only train/val split into a verified local NAS mirror.

The training manifest, canonical paths and training implementation stay unchanged.
Mount the resulting nas/ directory read-only over /home/nas only in the trainer's
private mount namespace. This never changes the machine-wide NAS mount.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.manifest.read_text())
    rows = [row for split in ('train', 'val') for row in document['splits'][split]]
    assert len(document['splits']['train']) == 6401 and len(document['splits']['val']) == 1601
    assert all(row['source_kind'] == 'original' for row in rows)
    assert not {row['scene'] for row in rows} & {'Moon_2', 'Moon_4', 'Moon_9'}
    assert len({row['source_id'] for row in rows}) == 8002
    paths = {Path(row[key]): None for row in rows for key in ('image', 'mask')}
    for prefix in ('source', 'augmented'):
        paths[Path(document[prefix + '_manifest'])] = document[prefix + '_manifest_sha256']
    source_root = Path('/home/nas')
    assert all(path.is_relative_to(source_root) for path in paths)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'complete.json').exists():
        raise FileExistsError('Complete cache already exists; verify it instead of overwriting')
    start = time.monotonic()

    def copy(path, expected):
        destination = args.output / 'nas' / path.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(6):
            try:
                content = path.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                if expected is not None and digest != expected:
                    raise ValueError(f'Manifest SHA differs: {path}')
                if path.suffix == '.png':
                    with Image.open(io.BytesIO(content)) as im:
                        assert im.size == (1024, 1024), (str(path), im.size)
                        im.verify()
                temporary = destination.with_name(destination.name + '.partial')
                temporary.write_bytes(content)
                if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
                    raise OSError('Local copy checksum mismatch')
                temporary.replace(destination)
                return {'source': str(path), 'relative': str(path.relative_to(source_root)),
                        'bytes': len(content), 'sha256': digest, 'attempts': attempt + 1}
            except OSError:
                if attempt == 5:
                    raise
                time.sleep(min(1 + attempt, 5))

    copied = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(copy, path, digest) for path, digest in paths.items()]
        for job in as_completed(jobs):
            copied.append(job.result())
            if len(copied) % 500 == 0 or len(copied) == len(jobs):
                print(json.dumps({'copied': len(copied), 'total': len(jobs),
                    'bytes': sum(row['bytes'] for row in copied), 'seconds': time.monotonic() - start}), flush=True)
    report = {'complete': True, 'created_at': datetime.now(timezone.utc).isoformat(),
              'manifest': str(args.manifest.resolve()), 'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'train_images': 6401, 'val_images': 1601, 'files_count': len(copied),
              'bytes': sum(row['bytes'] for row in copied), 'seconds': time.monotonic() - start,
              'verification': 'SHA256 of source bytes equals re-read local copy; PNG dimensions and integrity verified',
              'files': sorted(copied, key=lambda row: row['source'])}
    temporary = args.output / 'complete.json.tmp'
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(args.output / 'complete.json')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}), flush=True)


if __name__ == '__main__':
    main()
