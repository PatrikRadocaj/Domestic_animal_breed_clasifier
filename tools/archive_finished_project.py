"""Preserve final weights and training provenance without copying raw images."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil


def main():
    root = Path(__file__).resolve().parents[1]
    entries = []

    def save(source, destination, compress=False):
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if compress:
            with source.open('rb') as src, gzip.open(destination, 'wb') as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.copy2(source, destination)
        entries.append({
            'source': source.relative_to(root).as_posix(),
            'archived': destination.relative_to(root).as_posix(),
            'bytes': destination.stat().st_size,
            'sha256': hashlib.file_digest(destination.open('rb'), 'sha256').hexdigest(),
        })

    stage1 = root / 'runs/detect/artifacts/hybrid_shared_detector/hybrid_shared_fresh_qa_full_20260706_b12w12_yolo40'
    breed_run = root / 'artifacts/two-stage-breed-models-20260706T103441Z-744ad391'
    direct = root / 'runs/detect/artifacts/detector-20260706T125408Z-c7aed550/checkpoints/yolo11_breed_retrain_qa_full_20260706_b12w12_yolo40_resume'
    save(stage1 / 'weights/best.pt', root / 'models/final/species/best.pt')
    save(direct / 'weights/best.pt', root / 'models/final/direct_breed/best.pt')
    for p in sorted((breed_run / 'checkpoints/breed_models').glob('*/best.pt')):
        save(p, root / 'models/final/breed' / p.parent.name / 'best.pt')

    # Keep all experiment metrics, resolved configuration, QA and snapshot hashes.
    for base in [root / 'artifacts', root / 'runs/detect/artifacts']:
        for here, dirs, files in __import__('os').walk(base):
            dirs[:] = [d for d in dirs if d not in {'weights', 'checkpoints', '__pycache__'}]
            for name in files:
                p = Path(here) / name
                if p.suffix in {'.json', '.csv', '.yaml', '.yml'}:
                    dest = root / 'archive/training' / p.relative_to(root)
                    save(p, dest.with_name(dest.name + '.gz'), True)

    # Include detector results beneath checkpoint directories explicitly.
    for base in [stage1, direct]:
        for name in ['args.yaml', 'results.csv']:
            p = base / name
            save(p, root / 'archive/training' / p.relative_to(root))

    # Exact final labels, boxes, split assignments and source references.
    datasets = [
        root / 'data/processed/species_aware/hybrid_shared_qa_full_20260706_b12w12_yolo40',
        root / 'data/processed/breed_runs/temporary_breed_run_retrain_qa_full_20260706_b12w12_yolo40',
    ]
    for dataset in datasets:
        for p in sorted(dataset.iterdir()):
            if p.is_file():
                dest = root / 'archive/training' / p.relative_to(root)
                save(p, dest.with_name(dest.name + '.gz'), True)
    for p in sorted((root / 'data/processed').glob('*.yaml')):
        save(p, root / 'archive/training' / p.relative_to(root))
    # Human review decisions are irreplaceable even when source images are downloadable.
    for p in (root / 'results').rglob('*reviews*.csv'):
        dest = root / 'archive/training' / p.relative_to(root)
        save(p, dest.with_name(dest.name + '.gz'), True)

    index = root / 'archive/catalog.json'
    index.write_text(json.dumps({'files': entries}, indent=2), encoding='utf-8')
    print(f'Archived {len(entries)} files, {sum(e["bytes"] for e in entries)/1e6:.1f} MB')


if __name__ == '__main__':
    main()
