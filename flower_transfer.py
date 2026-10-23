"""Day 3: pretrained ResNet18, a frozen-head stage, then full fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from flower_data import CLASS_NAMES, record_environment, sha256_file, write_json
from flower_pipeline import seed_everything
from flower_training import (EXPECTED_DAY1_MANIFEST_SHA256, restore_day1_backup,
                             state_fingerprint, train_one_run, validate_day1)

WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
DAY2_RUNS = {"cnn_baseline", "cnn_augmented"}


class FlowerResNet18(nn.Module):
    """Frozen-head training keeps every backbone parameter and BN buffer fixed."""
    def __init__(self, num_classes=5, pretrained=False, stage="head"):
        super().__init__()
        self.network = resnet18(weights=WEIGHTS if pretrained else None)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)
        self.set_stage(stage)

    def set_stage(self, stage):
        if stage not in {"head", "finetune"}:
            raise ValueError("Stage must be head or finetune.")
        self.stage = stage
        for parameter in self.network.parameters():
            parameter.requires_grad_(stage == "finetune")
        for parameter in self.network.fc.parameters():
            parameter.requires_grad_(True)
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "stage", None) == "head":
            self.network.eval()
            self.network.fc.train(mode)
        return self

    def forward(self, images):
        return self.network(images)


def backbone_fingerprint(state):
    return state_fingerprint({name: value for name, value in state.items() if not name.startswith("network.fc.")})


def load_pretrained_state(root, seed=42):
    """Use a project-local cache and torchvision's verified official weights."""
    cache = Path(root).resolve() / "data/pretrained"
    torch.hub.set_dir(str(cache))
    seed_everything(seed)
    model = FlowerResNet18(pretrained=True, stage="head")
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    path = cache / "checkpoints" / WEIGHTS.url.rsplit("/", 1)[1]
    metadata = {"pretrained_weights": "ResNet18_Weights.IMAGENET1K_V1", "weights_url": WEIGHTS.url,
                "pretrained_file_sha256": sha256_file(path), "initial_backbone_sha256": backbone_fingerprint(state)}
    return state, metadata


def _validate_day2_comparison(comparison):
    rows = comparison.get("runs", [])
    if comparison.get("mode") != "full" or {row.get("run") for row in rows} != DAY2_RUNS or len(rows) != 2:
        raise ValueError("Use the completed full Day 2 comparison backup.")
    if comparison.get("test_evaluated") is not False:
        raise ValueError("Day 2 comparison must leave the test set untouched.")
    if len({row.get("initial_state_sha256") for row in rows}) != 1:
        raise ValueError("Day 2 runs do not record the same initial weights.")
    for row in rows:
        if row.get("manifest_sha256") != EXPECTED_DAY1_MANIFEST_SHA256 or row.get("test_evaluated") is not False:
            raise ValueError("Day 2 manifest/test-evaluation record is inconsistent.")
        if row.get("epochs") != 12 or row.get("mode") != "full":
            raise ValueError("Expected the verified 12-epoch Day 2 experiments.")
    return rows


def restore_day2_backup(backup: Path, root: Path):
    """Restore data and result artifacts only; archive Python is never executed."""
    root = Path(root).resolve()
    with zipfile.ZipFile(backup) as archive:
        names = archive.namelist()
        matches = [name for name in names if name.startswith("outputs/day2/") and name.endswith("/comparison.json")]
        if len(matches) != 1:
            raise ValueError("Expected exactly one full Day 2 comparison in the backup.")
        comparison_name = matches[0]
        prefix = comparison_name.rsplit("/", 1)[0] + "/"
        comparison = json.loads(archive.read(comparison_name))
        rows = _validate_day2_comparison(comparison)
        selected = [info for info in archive.infolist() if not info.is_dir() and info.filename.startswith(prefix)]
        if len({info.filename for info in selected}) != len(selected):
            raise ValueError("Duplicate Day 2 result paths in archive.")
        if any(info.file_size > 50 * 1024**2 for info in selected) or sum(info.file_size for info in selected) > 200 * 1024**2:
            raise ValueError("Day 2 result artifacts are unexpectedly large.")
        for info in selected:
            parts = PurePosixPath(info.filename).parts
            if ".." in parts or "\\" in info.filename or not (root / info.filename).resolve().is_relative_to(root / "outputs/day2"):
                raise ValueError(f"Invalid result path: {info.filename}")
        for row in rows:
            for filename in ("best.pt", "history.csv", "config.json"):
                if prefix + row["run"] + "/" + filename not in names:
                    raise ValueError(f"Missing Day 2 artifact: {row['run']}/{filename}")
        # This also verifies and restores the unchanged original split files.
        restore_day1_backup(backup, root)
        for info in selected:
            if Path(info.filename).suffix == ".py":
                continue
            target = root / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
        pointer = {"experiment": prefix.rstrip("/"), "comparison_sha256": hashlib.sha256(archive.read(comparison_name)).hexdigest(),
                   "backup_sha256": sha256_file(Path(backup))}
        write_json(root / "outputs/day2_import.json", pointer)
    return load_day2_results(root)


def load_day2_results(root: Path):
    root = Path(root).resolve()
    pointer = json.loads((root / "outputs/day2_import.json").read_text(encoding="utf-8"))
    experiment = (root / pointer["experiment"]).resolve()
    if not experiment.is_relative_to(root / "outputs/day2"):
        raise ValueError("Invalid restored Day 2 directory.")
    comparison_path = experiment / "comparison.json"
    if sha256_file(comparison_path) != pointer["comparison_sha256"]:
        raise ValueError("The restored Day 2 comparison changed.")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    _validate_day2_comparison(comparison)
    validate_day1(root)
    return experiment, comparison


def _comparison_row(row, model_name, stage_epochs, cumulative_epochs, cumulative_seconds):
    return {"run": row["run"], "model": model_name, "best_epoch_in_stage": row["best_epoch"],
            "stage_epoch_budget": stage_epochs, "cumulative_epoch_budget": cumulative_epochs,
            "val_accuracy": row["val_accuracy"], "val_macro_f1": row["val_macro_f1"], "val_loss": row["val_loss"],
            "stage_seconds": row["training_seconds"], "cumulative_seconds": cumulative_seconds,
            "manifest_sha256": row["manifest_sha256"], "test_evaluated": False}


def plot_transfer_comparison(experiment):
    import csv
    experiment = Path(experiment)
    os.environ.setdefault("MPLCONFIGDIR", str(experiment / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metadata = json.loads((experiment / "experiment.json").read_text())
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    for name, label, color in (("resnet18_head", "Frozen head", "#2962d9"),
                               ("resnet18_finetuned", "Fine-tuned", "#249977")):
        with (experiment / name / "history.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        epochs = [int(row["epoch"]) for row in rows]
        axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], "--", color=color, label=f"{label}: train")
        axes[0].plot(epochs, [float(row["val_loss"]) for row in rows], color=color, label=f"{label}: val")
        axes[1].plot(epochs, [float(row["val_accuracy"]) for row in rows], color=color, label=label)
        axes[2].plot(epochs, [float(row["val_macro_f1"]) for row in rows], color=color, label=label)
    for axis, title, ylabel in zip(axes, ("Training/validation loss", "Validation accuracy", "Validation macro-F1"),
                                   ("Cross-entropy", "Accuracy", "Macro-F1")):
        axis.set(title=title, xlabel="Epoch within stage", ylabel=ylabel)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[1].set_ylim(0, 1)
    axes[2].set_ylim(0, 1)
    figure.suptitle("CPU SMOKE CHECK - partial data; not project results" if metadata["mode"] == "smoke"
                    else "Day 3 ResNet18 stages (fine-tuning starts from the best head checkpoint)")
    destination = experiment / "learning_curves.png"
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def run_transfer_comparison(root, head_epochs=6, fine_epochs=6, batch_size=32, seed=42,
                            num_workers=0, allow_cpu=False, smoke=False, output_dir=None):
    import csv
    root = Path(root).resolve()
    if min(head_epochs, fine_epochs, batch_size) < 1 or num_workers < 0:
        raise ValueError("Epochs/batch size must be positive and workers nonnegative.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not (allow_cpu or smoke):
        raise RuntimeError("Select a Colab GPU runtime and rerun setup, or deliberately set ALLOW_CPU=True.")
    if device == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    source_experiment, day2 = load_day2_results(root)
    data_info = validate_day1(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    experiment = Path(output_dir) if output_dir else root / "outputs" / ("day3_smoke" if smoke else "day3") / stamp
    experiment = experiment.resolve()
    experiment.mkdir(parents=True, exist_ok=False)
    print(f"Experiment: {experiment}\nLoading official ImageNet-pretrained ResNet18 weights...", flush=True)
    initial_state, provenance = load_pretrained_state(root, seed)
    common = {"mode": "smoke" if smoke else "full", "batch_size": batch_size, "seed": seed,
              "num_workers": num_workers, "weight_decay": 0.0001, "optimizer": "AdamW", "device": device,
              "max_train_batches": 1 if smoke else None, "max_val_batches": 1 if smoke else None,
              "log_every": 0 if smoke else 20, "scheduler": "CosineAnnealingLR", "image_size": 224,
              "model_name": "FlowerResNet18", "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
              **provenance}
    write_json(experiment / "experiment.json", {**common, **data_info, "head_epochs": head_epochs, "fine_epochs": fine_epochs,
               "day2_source": str(source_experiment.relative_to(root)), "comparison_budget_note":
               "Frozen head uses its stage budget. Fine-tuning adds to that budget and starts from the best head checkpoint."})
    head_config = {**common, "epochs": head_epochs, "learning_rate": 0.001, "stage": "head", "backbone_frozen": True}
    head = train_one_run(root, experiment / "resnet18_head", "resnet18_head", initial_state,
                         head_config, data_info, augment=True, model_factory=lambda: FlowerResNet18(stage="head"))
    head_checkpoint = torch.load(experiment / "resnet18_head/best.pt", map_location="cpu", weights_only=True)
    head_state = head_checkpoint["model_state"]
    if backbone_fingerprint(head_state) != provenance["initial_backbone_sha256"]:
        raise AssertionError("Frozen-head training changed backbone weights or BatchNorm buffers.")
    parent_hash = state_fingerprint(head_state)
    fine_config = {**common, "epochs": fine_epochs, "learning_rate": 0.0001, "stage": "finetune", "backbone_frozen": False,
                   "parent_run": "resnet18_head", "parent_best_epoch": head_checkpoint["epoch"],
                   "parent_model_sha256": parent_hash, "parent_checkpoint_sha256": sha256_file(experiment / "resnet18_head/best.pt")}
    fine = train_one_run(root, experiment / "resnet18_finetuned", "resnet18_finetuned", head_state,
                         fine_config, data_info, augment=True, model_factory=lambda: FlowerResNet18(stage="finetune"))
    if fine["initial_state_sha256"] != parent_hash:
        raise AssertionError("Fine-tuning did not start from the best frozen-head checkpoint.")
    rows = [_comparison_row(row, "SmallCNN", row["epochs"], row["epochs"], row["training_seconds"]) for row in day2["runs"]]
    rows.extend([_comparison_row(head, "ResNet18", head_epochs, head_epochs, head["training_seconds"]),
                 _comparison_row(fine, "ResNet18", fine_epochs, head_epochs + fine_epochs,
                                 head["training_seconds"] + fine["training_seconds"])])
    if smoke:
        for row in rows:
            row["verification_only"] = True
    with (experiment / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    winner = max(rows, key=lambda row: (row["val_macro_f1"], -row["val_loss"]))
    write_json(experiment / "comparison.json", {"mode": common["mode"], "runs": rows,
               "selected_by_validation": None if smoke else winner["run"], "test_evaluated": False,
               "head_backbone_unchanged": True, "fine_tune_parent_verified": True,
               "day2_source": str(source_experiment.relative_to(root)),
               "caveats": ["One seed; different model capacities and pretraining histories.",
                           "Fine-tuned cumulative time includes the frozen-head stage; head-only has fewer epochs."]})
    plot_transfer_comparison(experiment)
    print("Both stages finished. Test set was not evaluated.", flush=True)
    return experiment


def package_day3(root, experiment):
    root, experiment = Path(root).resolve(), Path(experiment).resolve()
    mode = json.loads((experiment / "experiment.json").read_text())["mode"]
    day2_experiment, _ = load_day2_results(root)
    destination = root / ("day3_smoke_backup.zip" if mode == "smoke" else "day3_backup.zip")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("flower_data.py", "flower_pipeline.py", "flower_training.py", "flower_transfer.py", "outputs/day2_import.json"):
            archive.write(root / name, name)
        for directory in (root / "data/processed", root / "outputs/day1", day2_experiment, experiment):
            for path in sorted(directory.rglob("*")):
                if path.is_file() and ".matplotlib" not in path.parts:
                    archive.write(path, path.relative_to(root))
        credits = root / "data/raw/flower_photos/LICENSE.txt"
        if credits.is_file():
            archive.write(credits, "docs/DATASET_IMAGE_CREDITS.txt")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--head-epochs", type=int, default=6)
    parser.add_argument("--fine-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_transfer_comparison(args.root, head_epochs=args.head_epochs, fine_epochs=args.fine_epochs,
                            batch_size=args.batch_size, allow_cpu=args.allow_cpu, smoke=args.smoke)
