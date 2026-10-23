"""Day 1: verified download, image audit, and reproducible split manifests.

Run: python flower_data.py --root .
Only Pillow is needed. No training or image modification happens here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import tarfile
import tempfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
ARCHIVE_SHA256 = "4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c"
ARCHIVE_BYTES = 228813984
CHECKSUM_SOURCE = "https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/tf_flowers/checksums.tsv"
CLASS_NAMES = ("daisy", "dandelion", "roses", "sunflowers", "tulips")
SPLITS = ("train", "val", "test")
MANIFEST_FIELDS = ("relative_path", "class_name", "label", "width", "height", "file_sha256", "pixel_sha256", "split")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_dataset(raw_dir: Path) -> Path:
    """Verify the official archive before extracting regular files only."""
    raw_dir = Path(raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = raw_dir / "flower_photos"
    marker = dataset_dir / ".verified_archive_sha256"
    if dataset_dir.exists():
        if marker.is_file() and marker.read_text().strip() == ARCHIVE_SHA256:
            return dataset_dir
        raise ValueError(f"Unverified existing directory: {dataset_dir}. Use a new --root.")

    archive = raw_dir / "flower_photos.tgz"
    if not archive.exists():
        partial = archive.with_suffix(".tgz.part")
        received, next_notice = 0, 20 * 1024 * 1024
        print(f"Downloading {ARCHIVE_BYTES / 1024**2:.1f} MiB of flower images...", flush=True)
        request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "flower-cnn-day1/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                out.write(block)
                received += len(block)
                if received >= next_notice:
                    print(f"  {received / 1024**2:.0f} MiB downloaded", flush=True)
                    next_notice += 20 * 1024 * 1024
        if partial.stat().st_size != ARCHIVE_BYTES or sha256_file(partial) != ARCHIVE_SHA256:
            partial.unlink(missing_ok=True)
            raise ValueError("Dataset download failed its size/SHA-256 check; rerun to retry.")
        partial.replace(archive)
    if archive.stat().st_size != ARCHIVE_BYTES or sha256_file(archive) != ARCHIVE_SHA256:
        raise ValueError(f"Cached archive checksum mismatch: {archive}. Rename it and retry.")

    # Staging keeps a partially extracted dataset from being mistaken for a complete one.
    with tempfile.TemporaryDirectory(prefix="extract-", dir=raw_dir) as temp:
        staging = Path(temp).resolve()
        with tarfile.open(archive, "r:gz") as source:
            for member in source:
                parts = PurePosixPath(member.name).parts
                if not parts or parts[0] != "flower_photos" or ".." in parts or "\\" in member.name:
                    raise ValueError(f"Unexpected archive path: {member.name}")
                target = staging.joinpath(*parts).resolve()
                if not target.is_relative_to(staging):
                    raise ValueError(f"Archive path escapes staging: {member.name}")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as incoming, target.open("wb") as out:
                        shutil.copyfileobj(incoming, out)
                else:
                    raise ValueError(f"Unsupported archive member: {member.name}")
        extracted = staging / "flower_photos"
        if not all((extracted / name).is_dir() for name in CLASS_NAMES):
            raise ValueError("Archive does not contain all five flower categories.")
        (extracted / ".verified_archive_sha256").write_text(ARCHIVE_SHA256 + "\n")
        extracted.rename(dataset_dir)
    return dataset_dir


def audit_images(dataset_dir: Path) -> tuple[list[dict], dict]:
    """Decode every image; keep one copy per exact decoded RGB image.

    Identical images with conflicting class labels are all excluded for review.
    This detects exact duplicates, not crops or other near duplicates.
    """
    candidates, corrupt = [], []
    dataset_dir = Path(dataset_dir)
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            raise ValueError(f"Missing class directory: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            relative = path.relative_to(dataset_dir).as_posix()
            try:
                with Image.open(path) as image:
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                    rgb.load()
                    width, height = rgb.size
                    pixels = hashlib.sha256(f"{width}x{height}:RGB:".encode() + rgb.tobytes()).hexdigest()
                candidates.append({"relative_path": relative, "class_name": class_name, "label": label,
                                   "width": width, "height": height, "file_sha256": sha256_file(path),
                                   "pixel_sha256": pixels})
            except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
                corrupt.append({"relative_path": relative, "error": str(error)})

    groups = defaultdict(list)
    for row in candidates:
        groups[row["pixel_sha256"]].append(row)
    clean, duplicates, conflicts = [], [], []
    for group in groups.values():
        if len({row["label"] for row in group}) > 1:
            conflicts.append([row["relative_path"] for row in group])
            continue
        clean.append(group[0])
        duplicates.extend({"kept": group[0]["relative_path"], "excluded": row["relative_path"]} for row in group[1:])
    clean.sort(key=lambda row: row["relative_path"])
    if any(not any(row["class_name"] == name for row in clean) for name in CLASS_NAMES):
        raise ValueError("Every class must contain readable, non-conflicting images.")
    audit = {"images_scanned": len(candidates) + len(corrupt), "readable_images": len(candidates),
             "usable_images": len(clean), "corrupt_images": corrupt, "excluded_duplicates": duplicates,
             "conflicting_label_groups": conflicts, "duplicate_detection": "Exact decoded RGB pixels; near duplicates are not detected."}
    return clean, audit


def stratified_split(rows: list[dict], seed: int = 42) -> list[dict]:
    """Assign approximately 70/15/15 percent within each class, deterministically."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["class_name"]].append(dict(row))
    result = []
    for class_name in sorted(grouped):
        group = sorted(grouped[class_name], key=lambda row: row["relative_path"])
        if len(group) < 7:
            raise ValueError(f"Need at least 7 usable images in {class_name} for three splits.")
        random.Random(f"{seed}:{class_name}").shuffle(group)
        n_val = int(len(group) * 0.15 + 0.5)
        n_test = int(len(group) * 0.15 + 0.5)
        n_train = len(group) - n_val - n_test
        for index, row in enumerate(group):
            row["split"] = "train" if index < n_train else "val" if index < n_train + n_val else "test"
            result.append(row)
    return sorted(result, key=lambda row: row["relative_path"])


def validate_splits(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Split manifest cannot be empty.")
    if len({row["relative_path"] for row in rows}) != len(rows):
        raise ValueError("A file appears more than once in the split manifest.")
    if len({row["pixel_sha256"] for row in rows}) != len(rows):
        raise ValueError("Duplicate decoded images remain in the split manifest.")
    for row in rows:
        if row["split"] not in SPLITS:
            raise ValueError(f"Unknown split: {row['split']}")
        if row["class_name"] not in CLASS_NAMES or int(row["label"]) != CLASS_NAMES.index(row["class_name"]):
            raise ValueError("Class name and numeric label do not match.")
    for name in CLASS_NAMES:
        if {row["split"] for row in rows if row["class_name"] == name} != set(SPLITS):
            raise ValueError(f"Class {name} is missing from a split.")


def save_sample_grid(rows: list[dict], dataset_dir: Path, destination: Path, seed: int = 42) -> None:
    """Only training images are shown; labels are folder labels, not predictions."""
    tile_width, tile_height = 190, 166
    canvas = Image.new("RGB", (5 * tile_width, 5 * tile_height + 40), "#f5f7fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), "Training samples: five images per class", fill="#17253c")
    for class_index, class_name in enumerate(CLASS_NAMES):
        group = [row for row in rows if row["class_name"] == class_name and row["split"] == "train"]
        selection = random.Random(f"{seed}:{class_name}:preview").sample(group, min(5, len(group)))
        for column, row in enumerate(selection):
            x, y = column * tile_width + 8, class_index * tile_height + 40
            with Image.open(dataset_dir / row["relative_path"]) as image:
                thumbnail = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), (174, 134))
                canvas.paste(thumbnail, (x, y))
            draw.text((x, y + 138), class_name, fill="#17253c")
    canvas.save(destination)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def record_environment(destination: Path) -> None:
    packages = {}
    for name in ("Pillow", "numpy", "matplotlib", "torch", "torchvision"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    write_json(destination, {"python": platform.python_version(), "platform": platform.platform(), "packages": packages})


def prepare_project(root: Path, seed: int = 42, dataset_dir: Path | None = None) -> dict:
    root = Path(root).resolve()
    dataset_dir = Path(dataset_dir).resolve() if dataset_dir else download_dataset(root / "data" / "raw")
    print("Auditing images and checking exact duplicates...", flush=True)
    rows, audit = audit_images(dataset_dir)
    rows = stratified_split(rows, seed)
    validate_splits(rows)
    processed, reports = root / "data" / "processed", root / "outputs" / "day1"
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    manifest = processed / "splits.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    class_counts = {name: {split: sum(row["class_name"] == name and row["split"] == split for row in rows)
                           for split in SPLITS} for name in CLASS_NAMES}
    summary = {"dataset": "Flower Photos", "dataset_url": DATASET_URL, "source_archive_sha256": ARCHIVE_SHA256 if dataset_dir == root / "data" / "raw" / "flower_photos" else None,
               "dataset_dir": str(dataset_dir), "seed": seed, "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
               "class_to_idx": {name: label for label, name in enumerate(CLASS_NAMES)},
               "class_counts": class_counts, "split_counts": dict(Counter(row["split"] for row in rows)),
               "width_range": [min(row["width"] for row in rows), max(row["width"] for row in rows)],
               "height_range": [min(row["height"] for row in rows), max(row["height"] for row in rows)],
               "manifest_sha256": sha256_file(manifest), "audit": audit}
    write_json(reports / "data_summary.json", summary)
    write_json(reports / "audit.json", audit)
    write_json(processed / "class_to_idx.json", summary["class_to_idx"])
    record_environment(reports / "environment.json")
    save_sample_grid(rows, dataset_dir, reports / "sample_grid.jpg", seed)
    lines = ["# Day 1 data audit", "", f"Usable images: {len(rows)}. Seed: {seed}.", "",
             "| Class | Train | Validation | Test |", "|---|---:|---:|---:|"]
    lines += [f"| {name} | {counts['train']} | {counts['val']} | {counts['test']} |" for name, counts in class_counts.items()]
    lines += ["", f"Unreadable files: {len(audit['corrupt_images'])}.",
              f"Excluded exact duplicates: {len(audit['excluded_duplicates'])}.",
              f"Conflicting-label groups excluded: {len(audit['conflicting_label_groups'])}.", "",
              "Manual label review and near-duplicate review are still needed. The class folders provide the labels.",
              "Only training images appear in sample_grid.jpg. Reuse splits.csv for every experiment.",
              "No model has been trained; there are no accuracy results yet."]
    (reports / "data_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"usable_images": len(rows), "split_counts": summary["split_counts"], "report": str(reports / "data_report.md")}, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-dir", type=Path, help="Use an already downloaded folder containing the five class directories.")
    args = parser.parse_args()
    prepare_project(args.root, args.seed, args.dataset_dir)
