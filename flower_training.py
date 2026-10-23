"""Day 2: custom CNN and paired training/augmentation experiments.

Run: python flower_training.py --root . --epochs 12
Full training requires CUDA unless --allow-cpu is explicitly supplied.
Bounded CPU checks use --smoke and are saved separately from full experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from itertools import islice

import torch
from torch import nn

from flower_data import CLASS_NAMES, download_dataset, record_environment, sha256_file, validate_splits, write_json
from flower_pipeline import make_loaders, seed_everything

EXPECTED_DAY1_MANIFEST_SHA256 = "4fc79e62cb179ff75a18772ba16f72e091184263e6fb8f4731819bbdc07c2dbb"
DAY1_REQUIRED = ("data/processed/splits.csv", "data/processed/class_to_idx.json", "outputs/day1/data_summary.json")


def restore_day1_backup(backup: Path, root: Path) -> dict:
    """Restore known data artifacts only; never execute Python from an archive."""
    root = Path(root).resolve()
    with zipfile.ZipFile(backup) as archive:
        names = archive.namelist()
        for name in DAY1_REQUIRED:
            if names.count(name) != 1:
                raise ValueError(f"Backup must contain exactly one {name}.")
        selected = [info for info in archive.infolist()
                    if not info.is_dir() and info.filename.startswith(("data/processed/", "outputs/day1/"))]
        if any(info.file_size > 20 * 1024**2 for info in selected) or sum(info.file_size for info in selected) > 40 * 1024**2:
            raise ValueError("Day 1 backup artifacts are unexpectedly large.")
        contents = {info.filename: archive.read(info) for info in selected}
        for name in contents:
            if ".." in PurePosixPath(name).parts or "\\" in name:
                raise ValueError(f"Invalid backup path: {name}")
        summary = json.loads(contents["outputs/day1/data_summary.json"])
        manifest_bytes = contents["data/processed/splits.csv"]
        actual = hashlib.sha256(manifest_bytes).hexdigest()
        if actual != EXPECTED_DAY1_MANIFEST_SHA256 or actual != summary["manifest_sha256"]:
            raise ValueError("Backup splits differ from the verified Day 1 splits. Keep the original manifest.")
        validate_splits(list(csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8")))))
        expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
        if json.loads(contents["data/processed/class_to_idx.json"]) != expected_mapping:
            raise ValueError("Backup class mapping does not match Day 1.")
        for name, data in contents.items():
            destination = (root / name).resolve()
            if not destination.is_relative_to(root):
                raise ValueError(f"Invalid backup path: {name}")
            if destination.suffix.lower() not in {".csv", ".json", ".md", ".txt", ".jpg", ".png"}:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    return validate_day1(root)


def validate_day1(root: Path) -> dict:
    root = Path(root)
    manifest = root / "data/processed/splits.csv"
    actual = sha256_file(manifest)
    summary = json.loads((root / "outputs/day1/data_summary.json").read_text(encoding="utf-8"))
    if actual != EXPECTED_DAY1_MANIFEST_SHA256 or actual != summary["manifest_sha256"]:
        raise ValueError("Day 1 manifest changed. Restore the verified Day 1 backup.")
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    validate_splits(rows)
    mapping = json.loads((root / "data/processed/class_to_idx.json").read_text(encoding="utf-8"))
    if mapping != {name: index for index, name in enumerate(CLASS_NAMES)}:
        raise ValueError("Class mapping changed since Day 1.")
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")}
    if counts != summary["split_counts"]:
        raise ValueError("Recorded split counts do not match the manifest.")
    return {"manifest_sha256": actual, "class_to_idx": mapping, "split_counts": counts}


class SmallCNN(nn.Module):
    """Three conv/BatchNorm/ReLU/pool blocks, followed by a small classifier."""
    def __init__(self, num_classes=5):
        super().__init__()
        blocks = []
        for incoming, outgoing in ((3, 16), (16, 32), (32, 64)):
            blocks.extend([nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                           nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True), nn.MaxPool2d(2)])
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, 128),
                                        nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, images):
        return self.classifier(self.pool(self.features(images)))


def metrics_from_confusion(confusion: torch.Tensor) -> dict:
    counts = confusion.to(dtype=torch.float64, device="cpu")
    true_positive = counts.diag()
    predicted = counts.sum(0)
    actual = counts.sum(1)
    precision = torch.where(predicted > 0, true_positive / predicted.clamp_min(1), 0.0)
    recall = torch.where(actual > 0, true_positive / actual.clamp_min(1), 0.0)
    denominator = precision + recall
    f1 = torch.where(denominator > 0, 2 * precision * recall / denominator.clamp_min(1e-15), 0.0)
    total = counts.sum().item()
    return {"accuracy": float(true_positive.sum().item() / total) if total else 0.0,
            "macro_f1": float(f1.mean().item()), "per_class_f1": f1.tolist(), "samples": int(total)}


def run_epoch(model, loader, device, optimizer=None, max_batches=None, log_every=20):
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64)
    total_loss, seen = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    started = time.perf_counter()
    batches = islice(loader, max_batches) if max_batches is not None else loader
    for index, (images, labels) in enumerate(batches):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training/validation loss is not finite.")
            if training:
                loss.backward()
                optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        seen += batch_size
        codes = labels.detach().cpu() * len(CLASS_NAMES) + logits.detach().argmax(1).cpu()
        confusion += torch.bincount(codes, minlength=len(CLASS_NAMES)**2).reshape(len(CLASS_NAMES), len(CLASS_NAMES))
        if training and log_every and (index + 1) % log_every == 0:
            print(f"    train batch {index + 1}/{len(loader)} | loss {total_loss / seen:.4f}", flush=True)
    if seen == 0:
        raise ValueError("Epoch received no images.")
    metrics = metrics_from_confusion(confusion)
    metrics.update(loss=total_loss / seen, seconds=time.perf_counter() - started)
    return metrics


def state_fingerprint(state) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_one_run(root, directory, run_name, initial_state, config, data_info, augment, model_factory=None):
    from torch.utils.tensorboard import SummaryWriter

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    device = torch.device(config["device"])
    loaders, _ = make_loaders(root, batch_size=config["batch_size"], num_workers=config["num_workers"],
                              augment=augment, seed=config["seed"], crop_scale=config.get("crop_scale", (0.75, 1.0)))
    seed_everything(config["seed"])
    model = model_factory() if model_factory is not None else SmallCNN()
    model.load_state_dict(initial_state)
    model.to(device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                  lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    initial_hash = state_fingerprint(model.state_dict())
    run_config = {**config, "run": run_name, "augmentation": augment,
                  "manifest_sha256": data_info["manifest_sha256"], "initial_state_sha256": initial_hash,
                  "class_to_idx": data_info["class_to_idx"], "selection_metric": "validation_macro_f1",
                  "selection_tiebreak": "lower_validation_loss", "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    write_json(directory / "config.json", run_config)
    record_environment(directory / "environment.json")
    history = []
    best_score, best_loss, best_epoch = -1.0, math.inf, 0
    started = time.perf_counter()
    fields = ("epoch", "learning_rate", "train_loss", "train_accuracy", "train_macro_f1",
              "val_loss", "val_accuracy", "val_macro_f1", "train_samples", "val_samples", "epoch_seconds")
    writer = SummaryWriter(log_dir=str(directory / "tensorboard"))
    try:
        with (directory / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            history_writer = csv.DictWriter(stream, fieldnames=fields)
            history_writer.writeheader()
            for epoch in range(1, config["epochs"] + 1):
                learning_rate = optimizer.param_groups[0]["lr"]
                train = run_epoch(model, loaders["train"], device, optimizer,
                                  config["max_train_batches"], log_every=config["log_every"])
                validation = run_epoch(model, loaders["val"], device,
                                       max_batches=config["max_val_batches"], log_every=0)
                row = {"epoch": epoch, "learning_rate": learning_rate,
                       "train_loss": train["loss"], "train_accuracy": train["accuracy"], "train_macro_f1": train["macro_f1"],
                       "val_loss": validation["loss"], "val_accuracy": validation["accuracy"], "val_macro_f1": validation["macro_f1"],
                       "train_samples": train["samples"], "val_samples": validation["samples"],
                       "epoch_seconds": train["seconds"] + validation["seconds"]}
                history.append(row)
                history_writer.writerow(row)
                stream.flush()
                for stage, metrics in (("train", train), ("val", validation)):
                    for metric in ("loss", "accuracy", "macro_f1"):
                        writer.add_scalar(f"{metric}/{stage}", metrics[metric], epoch)
                writer.add_scalar("learning_rate", learning_rate, epoch)
                is_best = validation["macro_f1"] > best_score or (validation["macro_f1"] == best_score and validation["loss"] < best_loss)
                if is_best:
                    best_score, best_loss, best_epoch = validation["macro_f1"], validation["loss"], epoch
                    checkpoint = {"model_state": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
                                  "model_name": config.get("model_name", "SmallCNN"), "num_classes": len(CLASS_NAMES), "epoch": epoch,
                                  "val_macro_f1": best_score, "val_loss": best_loss, "config": run_config}
                    pending = directory / "best.pt.tmp"
                    torch.save(checkpoint, pending)
                    pending.replace(directory / "best.pt")
                print(f"{run_name} | epoch {epoch:02}/{config['epochs']} | train loss {train['loss']:.4f} "
                      f"| val loss {validation['loss']:.4f} | val accuracy {validation['accuracy']:.1%} "
                      f"| val macro-F1 {validation['macro_f1']:.4f}{' * best' if is_best else ''}", flush=True)
                writer.flush()
                scheduler.step()
    finally:
        writer.close()
    best_row = history[best_epoch - 1]
    result = {"run": run_name, "mode": config["mode"], "augmentation": augment, "epochs": config["epochs"],
              "best_epoch": best_epoch, "val_accuracy": best_row["val_accuracy"], "val_macro_f1": best_score,
              "val_loss": best_loss, "training_seconds": time.perf_counter() - started,
              "checkpoint": str((directory / "best.pt").resolve()), "manifest_sha256": data_info["manifest_sha256"],
              "initial_state_sha256": initial_hash, "test_evaluated": False}
    write_json(directory / "summary.json", result)
    return result


def plot_comparison(experiment):
    experiment = Path(experiment)
    config = experiment / ".matplotlib"
    config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config.resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metadata = json.loads((experiment / "experiment.json").read_text())
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    for run, color in (("cnn_baseline", "#2962d9"), ("cnn_augmented", "#249977")):
        with (experiment / run / "history.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        epochs = [int(row["epoch"]) for row in rows]
        label = "Baseline" if run == "cnn_baseline" else "Augmented"
        axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], "--", color=color, label=f"{label}: train")
        axes[0].plot(epochs, [float(row["val_loss"]) for row in rows], color=color, label=f"{label}: val")
        axes[1].plot(epochs, [float(row["val_accuracy"]) for row in rows], color=color, label=label)
        axes[2].plot(epochs, [float(row["val_macro_f1"]) for row in rows], color=color, label=label)
    for axis, title, ylabel in zip(axes, ("Training and validation loss", "Validation accuracy", "Validation macro-F1"),
                                   ("Cross-entropy", "Accuracy", "Macro-F1")):
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    axes[1].set_ylim(0, 1)
    axes[2].set_ylim(0, 1)
    figure.suptitle("CPU SMOKE CHECK - partial data; not project results" if metadata["mode"] == "smoke" else "Day 2 custom CNN comparison (validation)")
    destination = experiment / "learning_curves.png"
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def run_comparison(root: Path, epochs=12, batch_size=32, seed=42, num_workers=0, allow_cpu=False,
                   smoke=False, output_dir=None, max_train_batches=None, max_val_batches=None):
    root = Path(root).resolve()
    if epochs < 1 or batch_size < 1 or num_workers < 0:
        raise ValueError("Epochs/batch size must be positive and workers must be nonnegative.")
    if not smoke and (max_train_batches is not None or max_val_batches is not None):
        raise ValueError("Partial-data limits are only allowed with smoke=True.")
    if smoke:
        max_train_batches = 2 if max_train_batches is None else max_train_batches
        max_val_batches = 2 if max_val_batches is None else max_val_batches
        if min(max_train_batches, max_val_batches) < 1:
            raise ValueError("Smoke batch limits must be positive.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not (allow_cpu or smoke):
        raise RuntimeError("Enable a Colab GPU runtime first and rerun setup. For deliberate CPU training, set ALLOW_CPU=True.")
    if device == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    data_info = validate_day1(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    experiment = Path(output_dir) if output_dir else root / "outputs" / ("day2_smoke" if smoke else "day2") / stamp
    experiment = experiment.resolve()
    experiment.mkdir(parents=True, exist_ok=False)
    config = {"mode": "smoke" if smoke else "full", "epochs": epochs, "batch_size": batch_size, "seed": seed,
              "num_workers": num_workers, "learning_rate": 0.001, "weight_decay": 0.0001,
              "optimizer": "AdamW",
              "device": device, "max_train_batches": max_train_batches, "max_val_batches": max_val_batches,
              "log_every": 0 if smoke else 20, "scheduler": "CosineAnnealingLR", "image_size": 224,
              "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None, "python": platform.python_version()}
    write_json(experiment / "experiment.json", {**config, **data_info})
    print(f"Experiment: {experiment}\nMode: {config['mode']} | device: {device} | epochs per run: {epochs}", flush=True)
    seed_everything(seed)
    initial_state = {name: tensor.detach().cpu().clone() for name, tensor in SmallCNN().state_dict().items()}
    initial_hash = state_fingerprint(initial_state)
    results = []
    for name, augment in (("cnn_baseline", False), ("cnn_augmented", True)):
        result = train_one_run(root, experiment / name, name, initial_state, config, data_info, augment)
        if result["initial_state_sha256"] != initial_hash:
            raise AssertionError("Paired runs did not start from identical weights.")
        results.append(result)
    with (experiment / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    plot_comparison(experiment)
    winner = max(results, key=lambda row: (row["val_macro_f1"], -row["val_loss"]))
    write_json(experiment / "comparison.json", {"mode": config["mode"], "runs": results,
               "selected_by_validation": winner["run"], "test_evaluated": False,
               "caveat": "One seed and one training budget. Augmentation is not guaranteed to improve validation results."})
    print(f"Finished. Validation selection: {winner['run']}. Test set was not evaluated.", flush=True)
    return experiment


def package_day2(root: Path, experiment: Path) -> Path:
    root, experiment = Path(root).resolve(), Path(experiment).resolve()
    mode = json.loads((experiment / "experiment.json").read_text())["mode"]
    destination = root / ("day2_smoke_backup.zip" if mode == "smoke" else "day2_backup.zip")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("flower_data.py", "flower_pipeline.py", "flower_training.py"):
            archive.write(root / name, name)
        for folder in (root / "data/processed", root / "outputs/day1", experiment):
            for path in sorted(folder.rglob("*")):
                if path.is_file() and ".matplotlib" not in path.parts:
                    run_prefix = "outputs/day2_smoke" if mode == "smoke" else "outputs/day2"
                    archive.write(path, path.relative_to(root) if folder != experiment else Path(run_prefix) / experiment.name / path.relative_to(experiment))
        credits = root / "data/raw/flower_photos/LICENSE.txt"
        if credits.is_file():
            archive.write(credits, "docs/DATASET_IMAGE_CREDITS.txt")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_comparison(args.root, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
                   num_workers=args.num_workers, allow_cpu=args.allow_cpu, smoke=args.smoke)
