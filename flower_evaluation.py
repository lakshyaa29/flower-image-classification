"""Day 4: validation metrics, confusion matrices, and recorded Grad-CAM review."""

from __future__ import annotations

import csv
import hashlib
import html
import base64
import json
import os
import random
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.nn import functional as F

from flower_data import CLASS_NAMES, record_environment, sha256_file, write_json
from flower_pipeline import IMAGENET_MEAN, IMAGENET_STD, make_loaders
from flower_training import EXPECTED_DAY1_MANIFEST_SHA256, SmallCNN, state_fingerprint, validate_day1
from flower_transfer import FlowerResNet18, load_day2_results, restore_day2_backup

MODEL_NAMES = ("cnn_baseline", "cnn_augmented", "resnet18_head", "resnet18_finetuned")


def restore_day3_backup(backup, root):
    root = Path(root).resolve()
    with zipfile.ZipFile(backup) as archive:
        matches = [name for name in archive.namelist() if name.startswith("outputs/day3/") and name.endswith("/comparison.json")]
        if len(matches) != 1:
            raise ValueError("Select the completed full Day 3 backup ZIP.")
        comparison_name = matches[0]
        prefix = comparison_name.rsplit("/", 1)[0] + "/"
        comparison = json.loads(archive.read(comparison_name))
        if comparison.get("mode") != "full" or {row["run"] for row in comparison["runs"]} != set(MODEL_NAMES):
            raise ValueError("Day 3 comparison must contain all four full model results.")
        if comparison.get("test_evaluated") is not False or comparison.get("selected_by_validation") not in MODEL_NAMES:
            raise ValueError("Invalid Day 3 selection or test-evaluation record.")
        if any(row["manifest_sha256"] != EXPECTED_DAY1_MANIFEST_SHA256 for row in comparison["runs"]):
            raise ValueError("Day 3 splits do not match the accepted original splits.")
        selected = [info for info in archive.infolist() if not info.is_dir() and info.filename.startswith(prefix)]
        if len({info.filename for info in selected}) != len(selected):
            raise ValueError("Duplicate Day 3 archive paths.")
        if any(info.file_size > 128 * 1024**2 for info in selected) or sum(info.file_size for info in selected) > 512 * 1024**2:
            raise ValueError("Unexpectedly large Day 3 artifacts.")
        for info in selected:
            target = (root / info.filename).resolve()
            if ".." in PurePosixPath(info.filename).parts or "\\" in info.filename or not target.is_relative_to(root / "outputs/day3"):
                raise ValueError(f"Invalid archive path: {info.filename}")
        for run in ("resnet18_head", "resnet18_finetuned"):
            if prefix + run + "/best.pt" not in archive.namelist():
                raise ValueError(f"Missing checkpoint for {run}.")
        restore_day2_backup(backup, root)
        for info in selected:
            if Path(info.filename).suffix == ".py":
                continue
            target = root / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
        write_json(root / "outputs/day3_import.json", {
            "experiment": prefix.rstrip("/"),
            "comparison_sha256": hashlib.sha256(archive.read(comparison_name)).hexdigest(),
            "backup_sha256": sha256_file(Path(backup)),
        })
    return load_day3_results(root)


def load_day3_results(root):
    root = Path(root).resolve()
    pointer = json.loads((root / "outputs/day3_import.json").read_text(encoding="utf-8"))
    directory = (root / pointer["experiment"]).resolve()
    if not directory.is_relative_to(root / "outputs/day3"):
        raise ValueError("Invalid Day 3 result directory.")
    comparison_file = directory / "comparison.json"
    if sha256_file(comparison_file) != pointer["comparison_sha256"]:
        raise ValueError("The restored Day 3 comparison changed.")
    comparison = json.loads(comparison_file.read_text(encoding="utf-8"))
    if comparison.get("mode") != "full" or comparison.get("test_evaluated") is not False:
        raise ValueError("Use full Day 3 results with the test set untouched.")
    day2_directory, _ = load_day2_results(root)
    if (root / comparison["day2_source"]).resolve() != day2_directory:
        raise ValueError("Day 2 and Day 3 source comparisons are inconsistent.")
    validate_day1(root)
    checkpoints = {name: (day2_directory if name.startswith("cnn_") else directory) / name / "best.pt" for name in MODEL_NAMES}
    return directory, comparison, checkpoints


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config["manifest_sha256"] != EXPECTED_DAY1_MANIFEST_SHA256 or config["class_to_idx"] != dict(zip(CLASS_NAMES, range(5))):
        raise ValueError("Checkpoint split hash or label mapping changed.")
    if checkpoint["model_name"] == "SmallCNN":
        model = SmallCNN()
    elif checkpoint["model_name"] == "FlowerResNet18":
        model = FlowerResNet18(stage=config["stage"])
    else:
        raise ValueError("Unsupported model architecture.")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def classification_report_from_matrix(matrix, names):
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (len(names), len(names)) or np.any(matrix < 0):
        raise ValueError("Confusion matrix has invalid shape or counts.")
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    diagonal = np.diag(matrix).astype(float)
    precision = np.divide(diagonal, predicted, out=np.zeros(len(names)), where=predicted > 0)
    recall = np.divide(diagonal, support, out=np.zeros(len(names)), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(names)), where=(precision + recall) > 0)
    total = int(support.sum())
    report = {name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1-score": float(f1[i]),
                     "support": int(support[i])} for i, name in enumerate(names)}
    report["accuracy"] = float(diagonal.sum() / total) if total else 0.0
    report["macro avg"] = {"precision": float(precision.mean()), "recall": float(recall.mean()),
                           "f1-score": float(f1.mean()), "support": total}
    report["weighted avg"] = {metric: float(np.dot(values, support) / total) if total else 0.0
                              for metric, values in (("precision", precision), ("recall", recall), ("f1-score", f1))}
    report["weighted avg"]["support"] = total
    return report


def predict_split(model, loader, device, split="val", max_batches=None):
    if split not in {"val", "test"}:
        raise ValueError("Prediction reports support validation or test data only.")
    records, loss_sum = [], 0.0
    model.eval()
    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            logits = model(images.to(device, non_blocking=True))
            loss_sum += F.cross_entropy(logits, labels.to(device), reduction="sum").item()
            probabilities = logits.softmax(1).cpu()
            predictions = probabilities.argmax(1)
            for offset in range(len(labels)):
                index = len(records)
                source = loader.dataset.rows[index]
                truth, prediction = int(labels[offset]), int(predictions[offset])
                if int(source["label"]) != truth or source["split"] != split:
                    raise ValueError("Prediction record order or split is inconsistent.")
                top = probabilities[offset].topk(2).values
                record = {"index": index, "split": split, "relative_path": source["relative_path"],
                          "true_label": truth, "true_class": CLASS_NAMES[truth], "predicted_label": prediction,
                          "predicted_class": CLASS_NAMES[prediction], "correct": truth == prediction,
                          "confidence": float(top[0]), "top2_margin": float(top[0] - top[1])}
                record.update({f"prob_{name}": float(probabilities[offset, i]) for i, name in enumerate(CLASS_NAMES)})
                records.append(record)
            if (batch_index + 1) % 5 == 0:
                print(f"  {split} batch {batch_index + 1}/{len(loader)}", flush=True)
    if not records:
        raise ValueError("No predictions were produced.")
    return records, loss_sum / len(records)


def predict_validation(model, loader, device, max_batches=None):
    return predict_split(model, loader, device, split="val", max_batches=max_batches)


def configure_matplotlib(directory):
    config = Path(directory) / ".matplotlib"
    config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config.resolve()))
    import matplotlib
    matplotlib.use("Agg")


def save_confusion_figure(matrix, destination, title):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    support = matrix.sum(1, keepdims=True)
    normalized = np.divide(matrix, support, out=np.zeros_like(matrix, dtype=float), where=support > 0)
    for axis, values, label in zip(axes, (matrix, normalized), ("Counts", "Fraction within true class")):
        image = axis.imshow(values, cmap="Blues", vmin=0, vmax=None if label == "Counts" else 1)
        figure.colorbar(image, ax=axis, fraction=0.046)
        axis.set(xticks=range(5), yticks=range(5), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                 xlabel="Predicted class", ylabel="True class", title=label)
        axis.tick_params(axis="x", labelrotation=30)
        threshold = float(values.max()) / 2
        for i in range(5):
            for j in range(5):
                text = str(int(values[i, j])) if label == "Counts" else f"{values[i, j]:.2f}"
                axis.text(j, i, text, ha="center", va="center", color="white" if values[i, j] > threshold else "black")
    figure.suptitle(title)
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def select_review_cases(records, per_outcome=20, seed=42):
    if per_outcome < 1:
        raise ValueError("Review count must be positive.")
    selected = []
    for outcome in (True, False):
        groups = defaultdict(list)
        for record in records:
            if bool(record["correct"]) == outcome:
                groups[int(record["true_label"])].append(record)
        for label, group in groups.items():
            if outcome:
                group.sort(key=lambda row: row["relative_path"])
                random.Random(f"{seed}:{label}:correct").shuffle(group)
            else:
                group.sort(key=lambda row: (-row["confidence"], row["relative_path"]))
        chosen = []
        while len(chosen) < per_outcome and any(groups.values()):
            for label in sorted(groups):
                if groups[label] and len(chosen) < per_outcome:
                    chosen.append(dict(groups[label].pop(0)))
        selected.extend(chosen)
    return selected


def save_per_class_f1_figure(reports, destination):
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(10, 5), layout="constrained")
    positions = np.arange(5)
    for index, name in enumerate(MODEL_NAMES):
        axis.bar(positions + (index - 1.5) * 0.18, [reports[name][label]["f1-score"] for label in CLASS_NAMES],
                 width=0.18, label=name)
    axis.set(xticks=positions, xticklabels=CLASS_NAMES, ylim=(0, 1.05), ylabel="F1 score", title="Validation per-class F1")
    axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def compute_gradcam(model, image_tensor, target_layer, target_classes):
    """Class-logit Grad-CAM. Input gradients support a completely frozen backbone."""
    if image_tensor.ndim != 4 or image_tensor.size(0) != 1:
        raise ValueError("Grad-CAM expects one image at a time.")
    targets = list(dict.fromkeys(int(label) for label in target_classes))
    if not targets:
        raise ValueError("At least one target class is required.")
    captured = {}
    def capture(module, inputs, output):
        captured["activation"] = output
    handle = target_layer.register_forward_hook(capture)
    was_training = model.training
    model.eval()
    maps, flat = {}, {}
    try:
        with torch.inference_mode(False), torch.enable_grad():
            image = image_tensor.detach().clone().requires_grad_(True)
            logits = model(image)
            activation = captured["activation"]
            for position, label in enumerate(targets):
                if label < 0 or label >= logits.size(1):
                    raise ValueError("Grad-CAM target class is out of range.")
                gradient = torch.autograd.grad(logits[0, label], activation, retain_graph=position < len(targets) - 1)[0]
                weights = gradient.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * activation).sum(1, keepdim=True))
                cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
                spread = cam.max() - cam.min()
                flat[label] = bool(spread.detach().item() < 1e-12)
                cam = torch.zeros_like(cam) if flat[label] else (cam - cam.min()) / spread
                maps[label] = cam.detach().cpu().numpy()
            scores = logits.detach().cpu()
    finally:
        handle.remove()
        captured.clear()
        model.train(was_training)
    return maps, flat, scores


def target_layer_for(model):
    if isinstance(model, FlowerResNet18):
        return model.network.layer4[-1]
    if isinstance(model, SmallCNN):
        return next(module for module in reversed(list(model.features)) if isinstance(module, nn.Conv2d))
    raise ValueError("Unsupported Grad-CAM model.")


def save_case_panel(tensor, maps, record, destination):
    import matplotlib
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    rgb = (tensor.detach().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    truth, prediction = record["true_label"], record["predicted_label"]
    source = Image.fromarray(np.uint8(np.round(rgb * 255)))
    overlays = []
    for label in (prediction, truth):
        color = matplotlib.colormaps["inferno"](maps[label])[..., :3]
        overlays.append(Image.fromarray(np.uint8(np.round((0.6 * rgb + 0.4 * color) * 255))))
    canvas = Image.new("RGB", (3 * 224, 292), "#f5f7fb")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=13)
    headings = ("Model input crop", f"Predicted: {CLASS_NAMES[prediction]}", f"True: {CLASS_NAMES[truth]}")
    for index, (image, heading) in enumerate(zip((source, *overlays), headings)):
        draw.text((index * 224 + 6, 7), heading, fill="#17253c", font=font)
        canvas.paste(image, (index * 224, 30))
    draw.text((6, 259), f"{'CORRECT' if record['correct'] else 'INCORRECT'} | softmax score {record['confidence']:.1%}", fill="#17253c", font=font)
    draw.text((6, 277), record["relative_path"], fill="#17253c", font=ImageFont.load_default(size=10))
    canvas.save(destination)


def write_csv(path, records):
    if not records:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def review_progress(directory):
    cases = json.loads((Path(directory) / "review_cases.json").read_text(encoding="utf-8"))
    human = [case for case in cases if case.get("reviewed") and case.get("reviewer") == "human"]
    return {"total": len(cases), "reviewed": len(human), "remaining": len(cases) - len(human),
            "correct_reviewed": sum(bool(case["correct"]) for case in human),
            "incorrect_reviewed": sum(not case["correct"] for case in human)}


def save_case_review(directory, case_id, attention, pattern, comment="", reviewer="human"):
    if attention not in {"flower", "background", "mixed", "unclear"}:
        raise ValueError("Choose a valid attention category.")
    if not pattern or not pattern.strip():
        raise ValueError("Choose an observation/pattern.")
    directory = Path(directory)
    cases = json.loads((directory / "review_cases.json").read_text(encoding="utf-8"))
    case = next((case for case in cases if case["case_id"] == case_id), None)
    if case is None:
        raise ValueError("Unknown review case.")
    case.update(reviewed=True, reviewer=reviewer, attention=attention, pattern=pattern, comment=comment.strip(),
                reviewed_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(directory / "review_cases.json", cases)
    write_csv(directory / "review_cases.csv", cases)
    return review_progress(directory)


def run_day4_analysis(root, batch_size=32, num_workers=0, per_outcome=20, seed=42, smoke=False):
    root = Path(root).resolve()
    data_info = validate_day1(root)
    source_directory, day3, checkpoints = load_day3_results(root)
    selected_name = day3["selected_by_validation"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = root / "outputs" / ("day4_smoke" if smoke else "day4") / stamp
    directory.mkdir(parents=True, exist_ok=False)
    configure_matplotlib(directory)
    config = {"mode": "smoke" if smoke else "full", "split": "val", "test_evaluated": False, "seed": seed,
              "batch_size": batch_size, "num_workers": num_workers, "selected_model": selected_name,
              "requested_per_outcome": per_outcome, "manifest_sha256": data_info["manifest_sha256"],
              "source_day3": str(source_directory.relative_to(root)), "device": str(device),
              "review_selection": "Round-robin true classes; seeded correct samples; highest-confidence errors within class."}
    write_json(directory / "analysis.json", config)
    record_environment(directory / "environment.json")
    loaders, _ = make_loaders(root, batch_size=batch_size, num_workers=num_workers, augment=False, seed=seed)
    loader = loaders["val"]
    summaries, all_reports = [], {}
    selected_records = None
    for name in MODEL_NAMES:
        print(f"Evaluating {name} on validation data...", flush=True)
        model, checkpoint = load_model(checkpoints[name], device)
        started = time.perf_counter()
        records, loss = predict_validation(model, loader, device, max_batches=2 if smoke else None)
        seconds = time.perf_counter() - started
        matrix = np.zeros((5, 5), dtype=np.int64)
        for record in records:
            matrix[record["true_label"], record["predicted_label"]] += 1
        report = classification_report_from_matrix(matrix, CLASS_NAMES)
        result_dir = directory / name
        result_dir.mkdir()
        write_json(result_dir / "classification_report.json", report)
        write_json(result_dir / "confusion_matrix.json", {"class_names": list(CLASS_NAMES), "counts": matrix.tolist()})
        write_csv(result_dir / "predictions.csv", records)
        per_class = [{"class": label, **report[label]} for label in CLASS_NAMES]
        write_csv(result_dir / "per_class_metrics.csv", per_class)
        save_confusion_figure(matrix, result_dir / "confusion_matrix.png", f"{name}: validation")
        summary = {"run": name, "accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"],
                   "weighted_f1": report["weighted avg"]["f1-score"], "loss": loss, "samples": len(records),
                   "correct": int(np.trace(matrix)), "incorrect": int(matrix.sum() - np.trace(matrix)),
                   "inference_seconds": seconds, "checkpoint_sha256": sha256_file(checkpoints[name])}
        summaries.append(summary)
        all_reports[name] = report
        if name == selected_name:
            selected_records = records
            selected_matrix = matrix
        print(f"  accuracy {summary['accuracy']:.2%} | macro-F1 {summary['macro_f1']:.4f} | errors {summary['incorrect']}", flush=True)
        del model, checkpoint
    write_csv(directory / "model_metrics.csv", summaries)
    write_json(directory / "classification_reports.json", all_reports)
    save_per_class_f1_figure(all_reports, directory / "per_class_f1.png")
    selected_report = all_reports[selected_name]
    pairs = sorted([{"true_class": CLASS_NAMES[i], "predicted_class": CLASS_NAMES[j], "count": int(selected_matrix[i, j])}
                    for i in range(5) for j in range(5) if i != j and selected_matrix[i, j] > 0],
                   key=lambda row: (-row["count"], row["true_class"], row["predicted_class"]))
    write_json(directory / "failure_patterns.json", {"selected_model": selected_name, "confusion_pairs": pairs,
               "lowest_f1_class": min(CLASS_NAMES, key=lambda label: selected_report[label]["f1-score"]),
               "note": "Confusion counts are measured; background bias and label problems require visual review."})
    cases = select_review_cases(selected_records, per_outcome, seed)
    model, _ = load_model(checkpoints[selected_name], device)
    before = state_fingerprint(model.state_dict())
    case_dir = directory / "gradcam"
    case_dir.mkdir()
    gallery = ["<!doctype html><meta charset='utf-8'><title>Day 4 validation Grad-CAM review</title>",
               "<h1>Validation Grad-CAM review</h1><p>Each panel uses the exact model input crop. Heatmaps are normalized independently. They are class-sensitivity maps, not segmentation or proof of causation. Softmax scores are uncalibrated.</p>"]
    for position, case in enumerate(cases, 1):
        tensor, label = loader.dataset[case["index"]]
        maps, flat, logits = compute_gradcam(model, tensor.unsqueeze(0).to(device), target_layer_for(model),
                                            [case["predicted_label"], case["true_label"]])
        if int(logits.argmax(1)) != case["predicted_label"]:
            raise ValueError("Grad-CAM prediction differs from the saved validation prediction.")
        case_id = f"val_{case['index']:04d}"
        image_path = case_dir / f"{case_id}.png"
        save_case_panel(tensor, maps, case, image_path)
        np.savez_compressed(case_dir / f"{case_id}.npz", predicted=maps[case["predicted_label"]], truth=maps[case["true_label"]])
        case.update(case_id=case_id, panel=str(image_path.relative_to(directory)),
                    predicted_heatmap_flat=flat[case["predicted_label"]], true_heatmap_flat=flat[case["true_label"]],
                    reviewed=False, reviewer="", attention="", pattern="", comment="", reviewed_at_utc="")
        encoded = base64.b64encode(image_path.read_bytes()).decode()
        gallery.append(f"<h2>{position}. {html.escape(case_id)}</h2><img style='max-width:100%' src='data:image/png;base64,{encoded}'><p>{html.escape(case['relative_path'])}</p>")
        print(f"Grad-CAM {position}/{len(cases)}: {case_id} ({'correct' if case['correct'] else 'incorrect'})", flush=True)
    if state_fingerprint(model.state_dict()) != before:
        raise AssertionError("Evaluation/Grad-CAM changed model parameters or buffers.")
    write_json(directory / "review_cases.json", cases)
    write_csv(directory / "review_cases.csv", cases)
    (directory / "gradcam_gallery.html").write_text("\n".join(gallery), encoding="utf-8")
    correct_cases = sum(case["correct"] for case in cases)
    wrong_cases = len(cases) - correct_cases
    result = {**config, "model_metrics": summaries, "correct_cases": correct_cases, "incorrect_cases": wrong_cases,
              "review_target_met": correct_cases >= 20 and wrong_cases >= 20, "manual_review_complete": False,
              "model_state_unchanged": True, "gradcam_target": "predicted class plus true class for errors"}
    write_json(directory / "summary.json", result)
    lines = ["# Day 4 validation evaluation", "", f"Selected model: {selected_name}. Test set was not evaluated.", "",
             "| Model | Accuracy | Macro-F1 | Errors |", "|---|---:|---:|---:|"]
    lines += [f"| {row['run']} | {row['accuracy']:.2%} | {row['macro_f1']:.4f} | {row['incorrect']} |" for row in summaries]
    lines += ["", f"## Per-class report: {selected_name}", "",
              "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {label} | {selected_report[label]['precision']:.4f} | {selected_report[label]['recall']:.4f} | "
              f"{selected_report[label]['f1-score']:.4f} | {selected_report[label]['support']} |" for label in CLASS_NAMES]
    lines += ["", f"Grad-CAM cases generated: {correct_cases} correct and {wrong_cases} incorrect.",
              "Complete the saved review form for every case before finalizing the Day 4 backup.",
              "Use validation evidence to propose one fix. Final held-out test comparisons are reserved for Day 5."]
    (directory / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Analysis ready: {directory}\nReview {correct_cases} correct and {wrong_cases} incorrect cases.", flush=True)
    return directory


def package_day4(root, directory):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    summary = json.loads((directory / "summary.json").read_text())
    progress = review_progress(directory)
    if summary["mode"] == "full" and (progress["remaining"] or progress["correct_reviewed"] < 20 or progress["incorrect_reviewed"] < 20):
        raise ValueError("Complete the 20 correct and 20 incorrect cases in the review form before making the final backup.")
    notes = json.loads((directory / "day4_observations.json").read_text())
    if summary["mode"] == "full" and any("TODO" in str(value) for value in notes.values()):
        raise ValueError("Fill the observation answers before making the final backup.")
    summary["manual_review_complete"] = summary["mode"] == "full" and progress["remaining"] == 0
    summary["review_progress"] = progress
    write_json(directory / "summary.json", summary)
    source_day3, _, _ = load_day3_results(root)
    source_day2, _ = load_day2_results(root)
    destination = root / ("day4_smoke_backup.zip" if summary["mode"] == "smoke" else "day4_backup.zip")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("flower_data.py", "flower_pipeline.py", "flower_training.py", "flower_transfer.py", "flower_evaluation.py", "flower_review.py",
                     "outputs/day2_import.json", "outputs/day3_import.json"):
            archive.write(root / name, name)
        for folder in (root / "data/processed", root / "outputs/day1", source_day2, source_day3, directory):
            for path in sorted(folder.rglob("*")):
                if path.is_file() and ".matplotlib" not in path.parts:
                    archive.write(path, path.relative_to(root))
        credits = root / "data/raw/flower_photos/LICENSE.txt"
        if credits.is_file():
            archive.write(credits, "docs/DATASET_IMAGE_CREDITS.txt")
    return destination
