from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import re

from PIL import Image, ImageOps

from animal_breed.detection.yolo_source_index import discover_yolo_sources, resolve_image_for_label


SUSPECT_DATASETS = [
    "alpaca-4lxyr-roboflow-v41",
    "alpaca-xqfiw-roboflow-v2",
    "arabian-camel-classification2-roboflow-v1",
    "camel-of42k-rf",
    "cat-breeds-vjndm-rf",
    "cats-n9b87-rf",
    "cattle-pig-sheep-cow-horse-amiteshpatra07",
    "cow-buffalo-rf-v1",
    "detection-of-stray-dogs-v1-rf",
    "donkey-auto-dataset-roboflow-v4",
    "donkey-auto-dataset-roboflow-v5",
    "donkey-bdn3l-roboflow-v1",
    "farm-animals-v1i-yolo26-roboflow",
    "geese-counting-project-2-fipqg-roboflow-v4",
    "goat-detector-v2-rf",
    "llamas-2tdzp-roboflow-v1",
    "llamas-anvhx-roboflow-v1",
    "llamas-hardee-roboflow-v1",
    "peacock-fibzw-v2-rf",
    "pig-breed-identification-v3",
    "pig-detection-kaimq-a8ret-abpd",
    "pigeon-1ww2b-v8-rf",
    "pigeon-detection-6ompk-v5-rf",
    "project-goat-v6-rf",
    "pure-chicken-breeds-identification-roboflow",
    "rabbit-alemt-v2-rf",
    "rabbit-bthcx-rf",
    "sheep-detection-vg9sh-v27-rf",
    "turkey-v1i-yolo26-roboflow",
]

MAX_SAMPLES_PER_LABEL = 12
PADDING_FRACTION = 0.1
MAX_SIDE = 512


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-") or "label"


def parse_yolo_row(row: str) -> tuple[int, float, float, float, float] | None:
    parts = row.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        xc = float(parts[1])
        yc = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except ValueError:
        return None
    return cls, xc, yc, w, h


def yolo_to_xyxy(width: int, height: int, xc: float, yc: float, w: float, h: float) -> tuple[int, int, int, int]:
    x1 = int(round((xc - w / 2.0) * width))
    y1 = int(round((yc - h / 2.0) * height))
    x2 = int(round((xc + w / 2.0) * width))
    y2 = int(round((yc + h / 2.0) * height))
    return x1, y1, x2, y2


def expand_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int, padding_fraction: float) -> tuple[int, int, int, int]:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(round(bw * padding_fraction))
    py = int(round(bh * padding_fraction))
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(width, x2 + px)
    y2 = min(height, y2 + py)
    return x1, y1, x2, y2


def save_contact_sheet(image_paths: list[Path], out_path: Path, thumb_size: int = 180, columns: int = 4) -> None:
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((thumb_size, thumb_size))
            canvas = Image.new("RGB", (thumb_size, thumb_size), (245, 240, 228))
            x = (thumb_size - img.width) // 2
            y = (thumb_size - img.height) // 2
            canvas.paste(img, (x, y))
            thumbs.append(canvas)

    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_size, rows * thumb_size), (250, 246, 238))
    for idx, thumb in enumerate(thumbs):
        x = (idx % columns) * thumb_size
        y = (idx // columns) * thumb_size
        sheet.paste(thumb, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=95)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    yolo_root = repo_root / "data" / "raw" / "yolo_sources"
    out_root = repo_root / "outputs" / "suspect_yolo_label_crops"
    crops_root = out_root / "crops"
    out_root.mkdir(parents=True, exist_ok=True)
    crops_root.mkdir(parents=True, exist_ok=True)

    datasets = {ds.dataset_id: ds for ds in discover_yolo_sources(yolo_root)}
    rows: list[dict[str, str | int | float]] = []
    saved_by_label: dict[tuple[str, str], list[Path]] = defaultdict(list)

    for dataset_id in SUSPECT_DATASETS:
        ds = datasets.get(dataset_id)
        if ds is None:
            continue

        label_counts: dict[int, int] = defaultdict(int)
        for split, label_dir in ds.split_label_dirs.items():
            image_dir = ds.split_image_dirs.get(split)
            if image_dir is None or not label_dir.exists():
                continue
            for label_path in sorted(label_dir.rglob("*.txt")):
                image_path = resolve_image_for_label(label_path, image_dir)
                if image_path is None:
                    continue
                try:
                    with Image.open(image_path) as image:
                        image = ImageOps.exif_transpose(image).convert("RGB")
                        width, height = image.size
                        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                        for det_idx, line in enumerate(lines):
                            parsed = parse_yolo_row(line)
                            if parsed is None:
                                continue
                            class_idx, xc, yc, w, h = parsed
                            if class_idx < 0 or class_idx >= len(ds.names):
                                continue
                            label_name = ds.names[class_idx]
                            key = (dataset_id, label_name)
                            if label_counts[class_idx] >= MAX_SAMPLES_PER_LABEL:
                                continue
                            x1, y1, x2, y2 = yolo_to_xyxy(width, height, xc, yc, w, h)
                            x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, width, height, PADDING_FRACTION)
                            if x2 <= x1 or y2 <= y1:
                                continue
                            crop = image.crop((x1, y1, x2, y2))
                            if max(crop.size) > MAX_SIDE:
                                crop.thumbnail((MAX_SIDE, MAX_SIDE))

                            label_slug = slugify(label_name)
                            crop_dir = crops_root / dataset_id / label_slug
                            crop_dir.mkdir(parents=True, exist_ok=True)
                            crop_name = f"{label_path.stem}__{split}__{det_idx:02d}.jpg"
                            crop_path = crop_dir / crop_name
                            crop.save(crop_path, quality=95)

                            rows.append(
                                {
                                    "dataset_id": dataset_id,
                                    "label_name": label_name,
                                    "label_slug": label_slug,
                                    "split": split,
                                    "source_image": str(image_path),
                                    "source_label": str(label_path),
                                    "crop_path": str(crop_path),
                                    "class_idx": class_idx,
                                    "det_idx": det_idx,
                                    "x1": x1,
                                    "y1": y1,
                                    "x2": x2,
                                    "y2": y2,
                                }
                            )
                            saved_by_label[key].append(crop_path)
                            label_counts[class_idx] += 1
                except Exception:
                    continue

    manifest_path = out_root / "index.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_id",
                "label_name",
                "label_slug",
                "split",
                "source_image",
                "source_label",
                "crop_path",
                "class_idx",
                "det_idx",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    for (dataset_id, label_name), crop_paths in saved_by_label.items():
        label_slug = slugify(label_name)
        sheet_path = out_root / "contact_sheets" / dataset_id / f"{label_slug}.jpg"
        save_contact_sheet(crop_paths, sheet_path)

    html_lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Suspect YOLO Label Crops</title>",
        "<style>body{font-family:Georgia,serif;background:#f6f1e6;color:#241b12;padding:24px}h1,h2,h3{margin:0 0 12px}section{margin:0 0 28px}img{max-width:100%;border:1px solid #6a5740}code{background:#efe3cf;padding:2px 6px}li{margin:4px 0}</style>",
        "</head><body>",
        "<h1>Suspect YOLO Label Crops</h1>",
        f"<p>Datasets audited: <code>{', '.join(SUSPECT_DATASETS)}</code></p>",
        f"<p>Index CSV: <code>{manifest_path}</code></p>",
    ]

    grouped: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for (dataset_id, label_name), crop_paths in saved_by_label.items():
        grouped[dataset_id][label_name] = crop_paths

    for dataset_id in sorted(grouped):
        html_lines.append(f"<section><h2>{dataset_id}</h2>")
        html_lines.append("<ul>")
        for label_name in sorted(grouped[dataset_id]):
            label_slug = slugify(label_name)
            rel_sheet = (out_root / "contact_sheets" / dataset_id / f"{label_slug}.jpg").relative_to(out_root)
            html_lines.append(f"<li><strong>{label_name}</strong> ({len(grouped[dataset_id][label_name])} crops)</li>")
            html_lines.append(f"<p><img src='{rel_sheet.as_posix()}' alt='{dataset_id} {label_name}'></p>")
        html_lines.append("</ul></section>")

    html_lines.append("</body></html>")
    (out_root / "index.html").write_text("\n".join(html_lines), encoding="utf-8")

    print(f"Wrote {len(rows)} crops to {crops_root}")
    print(f"Manifest: {manifest_path}")
    print(f"HTML:     {out_root / 'index.html'}")


if __name__ == "__main__":
    main()
