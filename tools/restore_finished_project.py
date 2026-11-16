"""Verify published model/metadata hashes and optionally restore training records."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    entries = json.loads((root / 'archive/catalog.json').read_text())['files']
    for entry in entries:
        archived = root / entry['archived']
        if digest(archived) != entry['sha256']:
            raise ValueError(f'Checksum mismatch: {archived}. Run git lfs pull.')
    print(f'Verified {len(entries)} archived files.')
    if args.verify_only:
        return
    for entry in entries:
        if not entry['archived'].startswith('archive/training/'):
            continue
        target = (root / entry['source']).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f'Unsafe target: {target}')
        archived = root / entry['archived']
        if archived.suffix == '.gz':
            with gzip.open(archived, 'rb') as stream:
                data = stream.read()
        else:
            data = archived.read_bytes()
        if target.exists():
            if target.read_bytes() != data:
                raise FileExistsError(f'Refusing to replace different file: {target}')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print('Training metadata restored; raw images must be downloaded separately.')


if __name__ == '__main__':
    main()
