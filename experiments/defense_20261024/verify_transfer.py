"""Verify streamed copies against the hashes computed while reading the source account."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    if root.stat().st_uid != os.getuid():
        raise SystemExit('Destination root is not owned by this user')
    manifest = json.loads((root / 'ops/transfer_manifest.json').read_text())
    failed = []
    for row in manifest['files']:
        f = root / row['path']
        if not f.resolve().is_relative_to(root) or not f.is_file() or f.is_symlink():
            failed.append([row['path'], 'missing or unsafe destination'])
            continue
        sha = hashlib.sha256()
        with f.open('rb') as h:
            for b in iter(lambda: h.read(2**20), b''):
                sha.update(b)
        if f.stat().st_size != row['bytes'] or sha.hexdigest() != row['sha256']:
            failed.append([row['path'], 'hash mismatch'])
    result = dict(passed=not failed, checked=len(manifest['files']), bytes=manifest['bytes'],
                  source=manifest['source_root'], destination=str(root), failures=failed,
                  time=datetime.now().astimezone().isoformat())
    (root / 'ops/transfer_verified.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    if failed:
        raise SystemExit('Transfer verification failed')


if __name__ == '__main__':
    main()
