"""PyTorch datasets and separate transforms for the saved Day 1 splits."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from flower_data import CLASS_NAMES, sha256_file, validate_splits

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FlowerDataset(Dataset):
    def __init__(self, dataset_dir: Path, rows: list[dict], transform):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.rows = rows
        self.transform = transform
        for row in rows:
            image_path = (self.dataset_dir / row["relative_path"]).resolve()
            if not image_path.is_relative_to(self.dataset_dir) or not image_path.is_file():
                raise ValueError(f"Missing or invalid image path: {image_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.dataset_dir / row["relative_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        return self.transform(image), int(row["label"])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loaders(root: Path, batch_size=32, num_workers=0, augment=False, seed=42, dataset_dir=None,
                 crop_scale=(0.75, 1.0)):
    """Baseline training is unaugmented; val/test are always deterministic.

    Start a fresh set of loaders with the same seed for each experiment.
    Fixed seeds improve repeatability; identical GPU results across machines
    and library versions are not guaranteed.
    """
    root = Path(root).resolve()
    if len(crop_scale) != 2 or not 0 < crop_scale[0] <= crop_scale[1] <= 1:
        raise ValueError("Crop scale must contain increasing area fractions in (0, 1].")
    manifest = root / "data" / "processed" / "splits.csv"
    summary = json.loads((root / "outputs" / "day1" / "data_summary.json").read_text())
    if sha256_file(manifest) != summary["manifest_sha256"]:
        raise ValueError("Split manifest changed since preparation. Restore it or rerun data preparation.")
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    validate_splits(rows)
    dataset_dir = Path(dataset_dir) if dataset_dir else root / "data" / "raw" / "flower_photos"
    normalise = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    evaluation_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), *normalise])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=tuple(crop_scale)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        *normalise,
    ]) if augment else evaluation_transform
    seed_everything(seed)
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = FlowerDataset(dataset_dir, [row for row in rows if row["split"] == split],
                                train_transform if split == "train" else evaluation_transform)
        generator = torch.Generator().manual_seed(seed)
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=split == "train",
                                    num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                                    worker_init_fn=seed_worker, generator=generator, drop_last=False)
    return loaders, list(CLASS_NAMES)


def save_eda_charts(root: Path):
    """Standard matplotlib charts of counts and original image dimensions."""
    root = Path(root)
    config = root / "outputs" / ".matplotlib"
    config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config.resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (root / "data" / "processed" / "splits.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    bottom = np.zeros(len(CLASS_NAMES))
    for split, colour in zip(("train", "val", "test"), ("#2962d9", "#e3a11a", "#249977")):
        counts = [sum(row["class_name"] == name and row["split"] == split for row in rows) for name in CLASS_NAMES]
        axes[0].bar(CLASS_NAMES, counts, bottom=bottom, color=colour, label=split)
        bottom += np.array(counts)
    axes[0].set(title="Images per class and split", ylabel="Number of images")
    axes[0].legend()
    axes[1].hist([int(row["width"]) for row in rows], bins=25, alpha=0.65, label="Width", color="#2962d9")
    axes[1].hist([int(row["height"]) for row in rows], bins=25, alpha=0.6, label="Height", color="#249977")
    axes[1].set(title="Original image dimensions", xlabel="Pixels", ylabel="Number of images")
    axes[1].legend()
    destination = root / "outputs" / "day1" / "eda_charts.png"
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination
