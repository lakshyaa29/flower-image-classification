"""Check recorded evidence without loading weights or reading test photographs."""

import ast
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("daisy", "dandelion", "roses", "sunflowers", "tulips")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, description):
    require(math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12), description)


def main():
    registry = read_json(ROOT / "results/provenance/artifacts.json")
    manifest = ROOT / "data/processed/splits.csv"
    require(digest_file(manifest) == registry["manifest_sha256"], "The split manifest hash changed.")
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(Counter(row["split"] for row in rows) == {"train": 2564, "val": 551, "test": 551}, "Unexpected split counts.")
    require(len({row["relative_path"] for row in rows}) == len(rows), "Repeated manifest paths.")
    require(len({row["pixel_sha256"] for row in rows}) == len(rows), "Exact pixel duplicates remain in the manifest.")
    test_rows = {row["relative_path"]: row for row in rows if row["split"] == "test"}
    results = read_json(ROOT / "results/day5/final_results.json")
    plan = read_json(ROOT / "results/day5/selection.json")
    comparison = read_json(ROOT / "results/day5/comparison.json")
    lock = read_json(ROOT / "results/provenance/final_test_lock.json")
    require(results["mode"] == "full" and results["test_evaluated"] and not results["fixture_only"], "Final results are incomplete or development-only.")
    require(len(results["runs"]) == len(plan["candidates"]) == 6, "Expected six candidates.")
    require(plan["selection_is_final"] and not results["selection_changed_after_test"], "Selection was not frozen.")
    frozen = dict(plan)
    frozen.pop("plan_sha256")
    require(hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest() == plan["plan_sha256"], "Frozen plan hash mismatch.")
    winner = max(comparison["runs"], key=lambda row: (row["val_macro_f1"], -row["val_loss"]))["run"]
    require(winner == plan["selected_by_validation"] == results["selected_by_validation"], "Selection is inconsistent with validation.")
    require(lock["status"] == "complete" and lock["plan_sha256"] == plan["plan_sha256"] == results["plan_sha256"], "Final lock/plan mismatch.")
    require(digest_file(ROOT / "results/day5/final_results.json") == lock["results_sha256"], "Final results hash changed.")
    for name in ("selection.json", "final_results.json", "demo_config.json"):
        require((ROOT / "results/day5" / name).read_bytes() == (ROOT / "results/provenance" / name).read_bytes(), f"Provenance copy mismatch: {name}")
    checkpoints = {row["run"]: row["checkpoint_sha256"] for row in plan["candidates"]}
    for run in results["runs"]:
        name = run["run"]
        folder = ROOT / "results/day5/test" / name
        require(run["samples"] == len(test_rows), f"Wrong test sample count: {name}")
        require(run["checkpoint_sha256"] == checkpoints[name], f"Checkpoint selection mismatch: {name}")
        require(digest_file(folder / "summary.json") == lock["completed"][name], f"Locked summary hash mismatch: {name}")
        with (folder / "predictions.csv").open(newline="", encoding="utf-8") as stream:
            predictions = list(csv.DictReader(stream))
        require(len(predictions) == len(test_rows) and {row["relative_path"] for row in predictions} == set(test_rows), f"Predictions do not match the test manifest: {name}")
        counts = [[0] * 5 for _ in CLASSES]
        for row in predictions:
            truth, predicted = int(row["true_label"]), int(row["predicted_label"])
            require(row["split"] == "test" and truth == int(test_rows[row["relative_path"]]["label"]), f"Prediction split/label mismatch: {name}")
            require(0 <= predicted < 5 and row["predicted_class"] == CLASSES[predicted], f"Prediction class mismatch: {name}")
            require(row["correct"] == str(truth == predicted), f"Incorrect correctness flag: {name}")
            probabilities = [float(row["prob_" + label]) for label in CLASSES]
            require(all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities), f"Invalid scores: {name}")
            require(abs(sum(probabilities) - 1) < 1e-5, f"Scores do not sum to one: {name}")
            counts[truth][predicted] += 1
        require(counts == read_json(folder / "confusion_matrix.json")["counts"], f"Confusion counts mismatch: {name}")
        report = read_json(folder / "classification_report.json")
        f1s, supports = [], []
        for index, label in enumerate(CLASSES):
            support, predicted_total, tp = sum(counts[index]), sum(row[index] for row in counts), counts[index][index]
            precision = tp / predicted_total if predicted_total else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * tp / (support + predicted_total) if support + predicted_total else 0.0
            require(report[label]["support"] == support, f"Support mismatch: {name}/{label}")
            for key, value in (("precision", precision), ("recall", recall), ("f1-score", f1)):
                close(report[label][key], value, f"Per-class {key} mismatch: {name}/{label}")
            f1s.append(f1)
            supports.append(support)
        accuracy = sum(counts[i][i] for i in range(5)) / len(test_rows)
        close(run["test_accuracy"], accuracy, f"Accuracy mismatch: {name}")
        close(run["test_macro_f1"], sum(f1s) / 5, f"Macro-F1 mismatch: {name}")
        close(run["test_weighted_f1"], sum(f * s for f, s in zip(f1s, supports)) / len(test_rows), f"Weighted-F1 mismatch: {name}")
    cases = read_json(ROOT / "results/day4/review_cases.json")
    require(len(cases) == 40 and all(case["reviewed"] for case in cases), "Incomplete visual review.")
    require(Counter(case["correct"] for case in cases) == {True: 20, False: 20}, "Expected 20 correct/20 incorrect reviews.")
    require(Counter(case["reviewer"] for case in cases) == {"human": 13, "codex_visual": 27}, "Review attribution changed.")
    require(all((ROOT / "results/day4" / case["panel"]).is_file() for case in cases), "Missing Grad-CAM panel.")
    for notebook in ROOT.glob("notebooks/**/*.ipynb"):
        document = read_json(notebook)
        require(document["nbformat"] == 4, f"Unsupported notebook: {notebook.name}")
        for cell in document["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                require(not any(output.get("output_type") == "error" for output in cell.get("outputs", [])), f"Saved notebook error: {notebook.name}")
    for record in registry["notebooks"]:
        require(digest_file(ROOT / record["path"]) == record["published_sha256"], f"Completed notebook changed: {record['path']}")
    print(json.dumps({"status": "passed", "accepted_images": len(rows), "test_predictions_checked": len(test_rows) * len(results["runs"]), "models": 6, "visual_reviews": 40, "reviewer_counts": {"human": 13, "codex_visual": 27}, "selected_model": winner, "real_test_images_loaded": 0}, indent=2))


if __name__ == "__main__":
    main()
