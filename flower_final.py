"""Day 5 controlled crop experiment, validation selection, and frozen final testing."""

from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from flower_data import CLASS_NAMES, sha256_file, write_json
from flower_pipeline import make_loaders
from flower_training import EXPECTED_DAY1_MANIFEST_SHA256, state_fingerprint, train_one_run, validate_day1
from flower_transfer import FlowerResNet18
from flower_evaluation import (MODEL_NAMES, classification_report_from_matrix, configure_matplotlib,
    load_day3_results, load_model, predict_split, restore_day3_backup, save_confusion_figure, write_csv)

DAY5_RUNS = ("resnet18_control", "resnet18_crop")
PREPROCESSING = "RGB/EXIF; Resize(256); CenterCrop(224); ImageNet mean/std"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _inside(root, relative, prefix):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to((Path(root) / prefix).resolve()):
        raise ValueError("Artifact path is outside its expected folder.")
    return path


def restore_day4_backup(backup, root):
    root = Path(root).resolve()
    with zipfile.ZipFile(backup) as archive:
        names = archive.namelist()
        matches = [name for name in names if name.startswith("outputs/day4/") and name.endswith("/summary.json")]
        if len(matches) != 1 or len(names) != len(set(names)):
            raise ValueError("Choose one completed Day 4 backup with unique archive entries.")
        prefix = matches[0].rsplit("/", 1)[0]
        items = [item for item in archive.infolist() if item.filename.startswith(prefix + "/") and not item.is_dir()]
        if any(item.file_size > 128 * 1024**2 for item in items) or sum(item.file_size for item in items) > 512 * 1024**2:
            raise ValueError("Day 4 artifacts exceed the expected size.")
        for item in items:
            if ".." in PurePosixPath(item.filename).parts or "\\" in item.filename:
                raise ValueError("Invalid Day 4 archive path.")
            _inside(root, item.filename, "outputs/day4")
        restore_day3_backup(backup, root)
        for item in items:
            if Path(item.filename).suffix in {".py", ".exe", ".bat", ".ps1"}:
                continue
            target = root / item.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(item))
        write_json(root / "outputs/day4_import.json", {"experiment": prefix,
            "summary_sha256": hashlib.sha256(archive.read(matches[0])).hexdigest(),
            "backup_sha256": sha256_file(Path(backup))})
    return load_day4_results(root)


def load_day4_results(root):
    root = Path(root).resolve()
    pointer = _json(root / "outputs/day4_import.json")
    directory = _inside(root, pointer["experiment"], "outputs/day4")
    if sha256_file(directory / "summary.json") != pointer["summary_sha256"]:
        raise ValueError("Day 4 summary changed since import.")
    summary, notes, cases = (_json(directory / name) for name in ("summary.json", "day4_observations.json", "review_cases.json"))
    if summary.get("mode") != "full" or summary.get("test_evaluated") is not False or summary.get("split") != "val":
        raise ValueError("Day 5 requires the completed validation-only Day 4 results.")
    if summary["manifest_sha256"] != EXPECTED_DAY1_MANIFEST_SHA256 or "TODO" in json.dumps(notes):
        raise ValueError("Restore the accepted split and complete the Day 4 observation answers.")
    reviewed = [case for case in cases if case.get("reviewed") and case.get("reviewer") in {"human", "codex_visual"}
                and case.get("attention") and case.get("pattern")]
    if len(reviewed) != len(cases) or sum(case["correct"] for case in reviewed) < 20 or sum(not case["correct"] for case in reviewed) < 20:
        raise ValueError("The 20 correct and 20 incorrect visual case reviews are incomplete.")
    _, day3, checkpoints = load_day3_results(root)
    if summary["selected_model"] != day3["selected_by_validation"]:
        raise ValueError("The selected Day 3/4 model changed.")
    for row in summary["model_metrics"]:
        if sha256_file(checkpoints[row["run"]]) != row["checkpoint_sha256"]:
            raise ValueError("An earlier checkpoint changed.")
    return directory, summary, notes, checkpoints


def select_validation_winner(rows):
    """Stable ties keep the earlier candidate; test metrics never enter this decision."""
    return max(rows, key=lambda row: (row["val_macro_f1"], -row["val_loss"]))["run"]


def run_day5_experiment(root, epochs=6, batch_size=32, seed=42, num_workers=0, allow_cpu=False, smoke=False):
    root = Path(root).resolve()
    if min(epochs, batch_size) < 1 or num_workers < 0:
        raise ValueError("Epochs/batch size must be positive and workers nonnegative.")
    day4_dir, day4, notes, checkpoints = load_day4_results(root)
    mode = "smoke" if smoke else "full"
    lock_path = root / "outputs" / ("smoke_final_test_lock.json" if smoke else "final_test_lock.json")
    active_path = root / "outputs" / ("day5_smoke_active.json" if smoke else "day5_active.json")
    protocol = {"mode": mode, "epochs": epochs, "batch_size": batch_size, "seed": seed, "num_workers": num_workers,
        "learning_rate": 1e-5, "weight_decay": 1e-4, "control_crop_scale": [0.75, 1.0], "treatment_crop_scale": [0.5, 1.0],
        "parent_checkpoint_sha256": sha256_file(checkpoints["resnet18_finetuned"]),
        "manifest_sha256": EXPECTED_DAY1_MANIFEST_SHA256, "source_day4": str(day4_dir.relative_to(root))}
    protocol_hash = _digest(protocol)
    if active_path.exists():
        active = _json(active_path)
        directory = _inside(root, active["experiment"], "outputs/day5_smoke" if smoke else "outputs/day5")
        if active["protocol_sha256"] == protocol_hash and (directory / "comparison.json").is_file():
            print("Using the completed Day 5 experiment; no retraining.", flush=True)
            return directory
        if active["protocol_sha256"] == protocol_hash:
            raise RuntimeError("A training run was interrupted. Preserve that folder and restart in a fresh runtime using the Day 4 backup; no test results should have been opened.")
    if lock_path.exists():
        raise RuntimeError("Final test evaluation is already locked. Use the saved results; do not train new candidates after seeing test scores.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not (allow_cpu or smoke):
        raise RuntimeError("Select Runtime > Change runtime type > T4 GPU and rerun setup. CPU training requires deliberate ALLOW_CPU=True.")
    if device == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = root / "outputs" / ("day5_smoke" if smoke else "day5") / stamp
    directory.mkdir(parents=True)
    write_json(directory / "experiment.json", {**protocol, "protocol_sha256": protocol_hash,
        "proposed_fix": notes["one_proposed_fix_and_reason"], "device": device,
        "budget_note": "Each Day 5 arm has six additional epochs by default, after the prior 12-epoch allocated transfer-learning budget. Same parent and settings; only crop scale differs."})
    write_json(active_path, {"experiment": str(directory.relative_to(root)), "protocol_sha256": protocol_hash})
    parent = torch.load(checkpoints["resnet18_finetuned"], map_location="cpu", weights_only=True)
    initial_state = parent["model_state"]
    initial_hash = state_fingerprint(initial_state)
    data_info = validate_day1(root)
    common = {"mode": mode, "epochs": epochs, "batch_size": batch_size, "num_workers": num_workers,
        "seed": seed, "learning_rate": 1e-5, "weight_decay": 1e-4, "device": device,
        "model_name": "FlowerResNet18", "stage": "finetune", "optimizer": "AdamW", "scheduler": "CosineAnnealingLR",
        "image_size": 224, "max_train_batches": 1 if smoke else None, "max_val_batches": 1 if smoke else None,
        "log_every": 0 if smoke else 20, "parent_run": "resnet18_finetuned", "parent_best_epoch": parent["epoch"],
        "parent_checkpoint_sha256": protocol["parent_checkpoint_sha256"], "parent_model_sha256": initial_hash}
    _, original, _ = load_day3_results(root)
    rows = [{"run": row["run"], "val_accuracy": row["val_accuracy"], "val_macro_f1": row["val_macro_f1"],
             "val_loss": row["val_loss"], "stage_epochs": row["stage_epoch_budget"],
             "allocated_total_epochs": row["cumulative_epoch_budget"], "cumulative_seconds": row["cumulative_seconds"],
             "checkpoint": str(checkpoints[row["run"]].relative_to(root)), "validation_samples": 551} for row in original["runs"]]
    parent_time = next(row["cumulative_seconds"] for row in rows if row["run"] == "resnet18_finetuned")
    for name, scale in zip(DAY5_RUNS, ([0.75, 1.0], [0.5, 1.0])):
        print(f"Training {name}: crop area {scale}, same parent and seed...", flush=True)
        result = train_one_run(root, directory / name, name, initial_state, {**common, "crop_scale": scale}, data_info,
                               augment=True, model_factory=lambda: FlowerResNet18(stage="finetune"))
        if result["initial_state_sha256"] != initial_hash:
            raise AssertionError("The controlled runs must start from identical model weights.")
        rows.append({"run": name, "val_accuracy": result["val_accuracy"], "val_macro_f1": result["val_macro_f1"],
            "val_loss": result["val_loss"], "stage_epochs": epochs, "allocated_total_epochs": 12 + epochs,
            "cumulative_seconds": parent_time + result["training_seconds"],
            "checkpoint": str((directory / name / "best.pt").relative_to(root)),
            "validation_samples": batch_size if smoke else 551})
    selected = day4["selected_model"] if smoke else select_validation_winner(rows)
    control, treatment = rows[-2:]
    comparison = {"mode": mode, "runs": rows, "selected_by_validation": selected, "selection_is_final": not smoke,
        "test_evaluated": False, "selection_metric": "validation_macro_f1", "tiebreak": "lower_validation_loss_then_existing_candidate",
        "crop_macro_f1_change_vs_control": treatment["val_macro_f1"] - control["val_macro_f1"],
        "parent_initial_states_equal": True, "protocol_sha256": protocol_hash,
        "warning": "SMOKE CHECK ONLY: partial training/validation, prior model used for demo; no new project selection." if smoke else "One seed; crop change may help or hurt. Selection uses validation only."}
    write_json(directory / "comparison.json", comparison)
    write_csv(directory / "validation_comparison.csv", rows)
    plot_day5_curves(directory)
    print(f"Validation selection frozen: {selected}. Test results have not been opened.", flush=True)
    return directory


def plot_day5_curves(directory):
    import csv
    directory = Path(directory)
    configure_matplotlib(directory)
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for name in DAY5_RUNS:
        with (directory / name / "history.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        epochs = [int(row["epoch"]) for row in rows]
        axes[0].plot(epochs, [float(row["val_macro_f1"]) for row in rows], marker="o", label=name)
        axes[1].plot(epochs, [float(row["val_loss"]) for row in rows], marker="o", label=name)
    for axis, title in zip(axes, ("Validation macro-F1", "Validation loss")):
        axis.set(title=title, xlabel="Additional epoch")
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    figure.savefig(directory / "learning_curves.png", dpi=150)
    plt.close(figure)


def freeze_final_selection(root, directory):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    comparison = _json(directory / "comparison.json")
    candidates = [{"run": row["run"], "checkpoint": row["checkpoint"],
        "checkpoint_sha256": sha256_file(root / row["checkpoint"]), "val_macro_f1": row["val_macro_f1"],
        "val_loss": row["val_loss"]} for row in comparison["runs"]]
    if comparison["mode"] == "full" and comparison["selected_by_validation"] != select_validation_winner(comparison["runs"]):
        raise ValueError("Selection must follow the validation-only rule.")
    plan = {"mode": comparison["mode"], "selected_by_validation": comparison["selected_by_validation"],
        "selection_is_final": comparison["selection_is_final"], "candidates": candidates,
        "manifest_sha256": validate_day1(root)["manifest_sha256"], "preprocessing": PREPROCESSING}
    plan["plan_sha256"] = _digest(plan)
    path = directory / "selection.json"
    if path.exists() and _json(path) != plan:
        raise ValueError("Frozen selection/checkpoints changed. Preserve the original selection and results.")
    write_json(path, plan)
    return plan


def evaluate_final_test(root, directory, batch_size=32, num_workers=0):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    plan = freeze_final_selection(root, directory)
    smoke = plan["mode"] == "smoke"
    lock_path = root / "outputs" / ("smoke_final_test_lock.json" if smoke else "final_test_lock.json")
    if lock_path.exists():
        lock = _json(lock_path)
        if lock["plan_sha256"] != plan["plan_sha256"] or lock["experiment"] != str(directory.relative_to(root)):
            raise ValueError("Final test results belong to a different frozen experiment. Do not retune using test feedback.")
    else:
        lock = {"plan_sha256": plan["plan_sha256"], "experiment": str(directory.relative_to(root)),
                "status": "running", "completed": {}, "started_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(lock_path, lock)
    final_path = directory / "final_results.json"
    if final_path.exists() and lock["status"] == "complete":
        results = _json(final_path)
        if results["plan_sha256"] != plan["plan_sha256"] or sha256_file(final_path) != lock.get("results_sha256"):
            raise ValueError("Cached results do not match the frozen plan.")
        for name, digest in lock["completed"].items():
            if sha256_file(directory / "test" / name / "summary.json") != digest:
                raise ValueError("A cached test report changed.")
        print("Using cached final test results; no test inference repeated.", flush=True)
        return results
    configure_matplotlib(directory)
    loaders, _ = make_loaders(root, batch_size=batch_size, num_workers=num_workers, augment=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    summaries = []
    for candidate in plan["candidates"]:
        name = candidate["run"]
        result_dir = directory / "test" / name
        report_path = result_dir / "summary.json"
        if name in lock["completed"]:
            if sha256_file(report_path) != lock["completed"][name]:
                raise ValueError("A completed test report changed.")
            summaries.append(_json(report_path))
            continue
        print(f"Final {'FIXTURE' if smoke else 'held-out test'} evaluation: {name}", flush=True)
        model, _ = load_model(root / candidate["checkpoint"], device)
        before = state_fingerprint(model.state_dict())
        started = time.perf_counter()
        records, loss = predict_split(model, loaders["test"], device, split="test", max_batches=1 if smoke else None)
        if not smoke and len(records) != 551:
            raise ValueError("Full final evaluation must include all 551 test images.")
        matrix = np.zeros((5, 5), dtype=np.int64)
        for record in records:
            matrix[record["true_label"], record["predicted_label"]] += 1
        report = classification_report_from_matrix(matrix, CLASS_NAMES)
        if before != state_fingerprint(model.state_dict()):
            raise AssertionError("Final inference changed model weights or buffers.")
        result_dir.mkdir(parents=True, exist_ok=True)
        write_json(result_dir / "classification_report.json", report)
        write_json(result_dir / "confusion_matrix.json", {"class_names": list(CLASS_NAMES), "counts": matrix.tolist()})
        write_csv(result_dir / "predictions.csv", records)
        save_confusion_figure(matrix, result_dir / "confusion_matrix.png", f"{name}: {'SMOKE FIXTURE' if smoke else 'final test'}")
        summary = {"run": name, "test_accuracy": report["accuracy"], "test_macro_f1": report["macro avg"]["f1-score"],
            "test_weighted_f1": report["weighted avg"]["f1-score"], "test_loss": loss, "samples": len(records),
            "inference_seconds": time.perf_counter() - started, "checkpoint_sha256": candidate["checkpoint_sha256"],
            "selected_before_test": name == plan["selected_by_validation"]}
        write_json(report_path, summary)
        lock["completed"][name] = sha256_file(report_path)
        write_json(lock_path, lock)
        summaries.append(summary)
    results = {"mode": plan["mode"], "plan_sha256": plan["plan_sha256"], "selected_by_validation": plan["selected_by_validation"],
        "test_evaluated": not smoke, "fixture_only": smoke, "runs": summaries,
        "selection_changed_after_test": False, "preprocessing": PREPROCESSING}
    write_json(final_path, results)
    write_csv(directory / "final_test_comparison.csv", summaries)
    lock.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat(), results_sha256=sha256_file(final_path))
    write_json(lock_path, lock)
    return results


def write_day5_observations(root, directory):
    directory = Path(directory)
    comparison, final = _json(directory / "comparison.json"), _json(directory / "final_results.json")
    _, day4, _, _ = load_day4_results(root)
    selected = next(row for row in final["runs"] if row["run"] == final["selected_by_validation"])
    delta = comparison["crop_macro_f1_change_vs_control"]
    notes = {"mode": comparison["mode"], "controlled_change": "Training crop scale (0.75, 1.0) to (0.5, 1.0).",
        "crop_validation_macro_f1_change_vs_control": delta,
        "crop_result": "Higher validation macro-F1 than matched continued training." if delta > 0 else
                       "Equal validation macro-F1 to matched continued training." if delta == 0 else
                       "Lower validation macro-F1 than matched continued training; the proposed fix did not improve this metric.",
        "selected_before_test": final["selected_by_validation"], "selected_test_accuracy": selected["test_accuracy"],
        "selected_test_macro_f1": selected["test_macro_f1"], "selection_changed_after_test": False,
        "test_evaluated": final["test_evaluated"], "day4_review_attribution": day4.get("reviewer_counts", {}),
        "limitations": "One seed; diagnostic case selection; mixed user/AI visual review; different pretraining and allocated budgets; "
                       "Grad-CAM is approximate; scores are uncalibrated; smaller crops can exclude flowers.",
        "demo_check": _json(directory / "demo_check.json") if (directory / "demo_check.json").exists() else None}
    write_json(directory / "day5_observations.json", notes)
    return notes


def package_day5(root, directory):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    comparison, final = _json(directory / "comparison.json"), _json(directory / "final_results.json")
    if comparison["mode"] == "full" and final.get("test_evaluated") is not True:
        raise ValueError("Complete the frozen final test comparison before the full backup.")
    if final["plan_sha256"] != freeze_final_selection(root, directory)["plan_sha256"]:
        raise ValueError("Final results do not match the frozen selection.")
    notes = _json(directory / "day5_observations.json")
    if "TODO" in json.dumps(notes):
        raise ValueError("Complete the Day 5 observation answers before the final backup.")
    required = ("final_report.pdf", "final_report.md", "presentation_notes.md", "demo_config.json", "submission/app.py")
    if any(not (directory / name).is_file() for name in required):
        raise ValueError("Generate the report, presentation notes, and demo files before backing up.")
    smoke = comparison["mode"] == "smoke"
    destination = root / ("day5_smoke_backup.zip" if smoke else "day5_backup.zip")
    day4_dir, _, _, _ = load_day4_results(root)
    day3_dir, day3, _ = load_day3_results(root)
    day2_dir = root / day3["day2_source"]
    folders = (root / "data/processed", root / "outputs/day1", day2_dir, day3_dir, day4_dir, directory)
    helper_names = ("flower_data.py", "flower_pipeline.py", "flower_training.py", "flower_transfer.py", "flower_evaluation.py",
                    "flower_review.py", "flower_final.py", "flower_demo.py", "flower_deliverables.py", "requirements.txt")
    lock_name = "smoke_final_test_lock.json" if smoke else "final_test_lock.json"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in helper_names:
            archive.write(root / name, name)
        for name in ("day2_import.json", "day3_import.json", "day4_import.json", lock_name,
                     "day5_smoke_active.json" if smoke else "day5_active.json"):
            archive.write(root / "outputs" / name, "outputs/" + name)
        for folder in folders:
            for path in sorted(folder.rglob("*")):
                if path.is_file() and ".matplotlib" not in path.parts and "report_preview" not in path.parts:
                    archive.write(path, path.relative_to(root))
        credits = root / "data/raw/flower_photos/LICENSE.txt"
        if credits.is_file():
            archive.write(credits, "docs/DATASET_IMAGE_CREDITS.txt")
    return destination
