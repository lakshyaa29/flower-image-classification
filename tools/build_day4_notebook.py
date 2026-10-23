"""Build the self-contained Day 4 validation and Grad-CAM Colab notebook."""

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def cell(kind, source):
    item = {"cell_type": kind, "id": f"day4-{len(cells):02}", "metadata": {},
            "source": textwrap.dedent(source).strip().splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    cells.append(item)


cell("markdown", """
    # Day 4 - Evaluation and Grad-CAM

    Today you will evaluate your **four saved models**, inspect confusion matrices
    and per-class precision/recall/F1, and review **20 correct + 20 incorrect**
    predictions from the selected fine-tuned ResNet18. No training is needed today.

    We use **validation** errors to choose a proposed improvement. The held-out test
    set remains reserved for the final Day 5 comparison.

    1. Keep your latest **`day3_backup (1).zip`** ready.
    2. Run cells in order. Upload that ZIP when prompted. A T4 GPU helps; CPU also works.
    3. **Pause at the review form:** inspect each image, choose two categories, and
       click **Save and next** until the counter reaches **40/40**.
    4. Fill the two observation answers, then create and download your backup.
    5. Download the executed notebook from the top **File -> Download -> Download .ipynb** menu.

    Avoid Run all: the review form needs your input before the final cells can finish.
""")
cell("markdown", """
    ## 1. Check packages

    Only missing packages are installed, preserving Colab's supplied torch/torchvision
    pair. There is no need to download ImageNet weights again: your checkpoints contain them.
""")
cell("code", """
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path

    modules = {"Pillow": "PIL", "numpy": "numpy", "matplotlib": "matplotlib",
               "torch": "torch", "torchvision": "torchvision", "tensorboard": "tensorboard",
               "ipywidgets": "ipywidgets"}
    missing = [package for package, module in modules.items() if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    import torch
    IN_COLAB = importlib.util.find_spec("google.colab") is not None if importlib.util.find_spec("google") else False
    current = Path.cwd()
    ROOT = Path("/content/flower-cnn") if IN_COLAB else (current.parent if current.name == "notebooks" else current)
    ROOT.mkdir(parents=True, exist_ok=True)
    print("PyTorch:", torch.__version__)
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (supported; slower)")
""")
cell("markdown", """
    ## 2. Load the helpers included in this notebook

    This cell includes the model definitions, evaluation code, and review form.
    Your backup supplies the saved data splits and trained model checkpoints.
""")
names = ("flower_data.py", "flower_pipeline.py", "flower_training.py", "flower_transfer.py",
         "flower_evaluation.py", "flower_review.py")
sources = {name: (ROOT / name).read_text(encoding="utf-8") for name in names}
cell("code", "# @title Load the reusable project files\nHELPER_SOURCES = " + repr(sources) + "\n"
     + "for name, source in HELPER_SOURCES.items():\n"
       "    destination = ROOT / name\n"
       "    if IN_COLAB or not destination.exists():\n"
       "        destination.write_text(source, encoding='utf-8')\n"
       "if str(ROOT) not in sys.path:\n"
       "    sys.path.insert(0, str(ROOT))\n"
       "import importlib\n"
       "for module_name in ('flower_data', 'flower_pipeline', 'flower_training', 'flower_transfer', 'flower_evaluation', 'flower_review'):\n"
       "    importlib.reload(importlib.import_module(module_name))\n"
       "from flower_data import CLASS_NAMES, download_dataset, write_json\n"
       "from flower_evaluation import (restore_day3_backup, load_day3_results, run_day4_analysis, review_progress, package_day4)\n"
       "from flower_review import launch_review_form\n"
       "import json\n"
       "print('Day 4 helpers ready:', ROOT)\n")
cells[-1]["metadata"]["cellView"] = "form"
cell("markdown", """
    ## 3. Restore Day 3

    If asked, upload **`day3_backup (1).zip`**, or your latest completed Day 3 backup.
    It restores all four checkpoints and the accepted Day 1 split manifest.
    Flower photographs are downloaded again only when missing.
""")
cell("code", """
    if not (ROOT / "outputs/day3_import.json").is_file():
        if IN_COLAB:
            from google.colab import files
            print("Upload your latest completed Day 3 backup ZIP now.")
            uploaded = files.upload()
            backups = [name for name in uploaded if name.lower().endswith(".zip")]
            if len(backups) != 1:
                raise ValueError("Select exactly one Day 3 backup ZIP and rerun this cell.")
            restore_day3_backup(Path(backups[0]), ROOT)
        else:
            raise FileNotFoundError("First use restore_day3_backup(backup_path, ROOT) to restore your completed Day 3 backup.")
    day3_directory, day3_comparison, checkpoint_paths = load_day3_results(ROOT)
    download_dataset(ROOT / "data/raw")
    print("Verified Day 3 results:", day3_directory)
    print("Selected model:", day3_comparison["selected_by_validation"])
    print("Saved checkpoints:", list(checkpoint_paths))
""")
cell("markdown", """
    ## 4. Evaluate the models and create the heatmaps

    This evaluates all **551 validation images** with each saved model. Then it
    creates 40 Grad-CAM panels for the selected model. Wait until **Analysis ready** appears.

    Correct cases are sampled with seed 42 across true classes. Incorrect cases
    are selected across true classes, taking higher-confidence errors first.
    The gallery is a diagnostic sample; the metrics above cover the full validation set.

    Running this cell again starts a new analysis folder. Once it finishes, continue
    below; later edits to notes only require rerunning the notes and backup cells.
""")
cell("code", """
    BATCH_SIZE = 32
    NUM_WORKERS = 2 if IN_COLAB else 0
    CASES_PER_OUTCOME = 20
    SEED = 42

    analysis = run_day4_analysis(ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                per_outcome=CASES_PER_OUTCOME, seed=SEED)
    summary = json.loads((analysis / "summary.json").read_text())
    if not summary["review_target_met"]:
        raise RuntimeError("Fewer than 20 correct or 20 incorrect predictions are available. Keep these results and discuss the shortfall before finalizing.")
""")
cell("markdown", """
    ## 5. Read the metrics

    **Precision:** among images predicted as a class, how many were correct?
    **Recall:** among actual images of a class, how many were found?
    **F1:** balances precision and recall. **Macro-F1** gives each class equal weight.
    Confusion matrix rows are true labels; columns are predicted labels.

    All four models' full per-class reports and prediction tables are saved in the backup.
""")
cell("code", """
    from IPython.display import display, Markdown, Image as DisplayImage

    rows = summary["model_metrics"]
    table = ["| Model | Val accuracy | Macro-F1 | Errors |", "|---|---:|---:|---:|"]
    table += [f"| {row['run']} | {row['accuracy']:.2%} | {row['macro_f1']:.4f} | {row['incorrect']} |" for row in rows]
    display(Markdown("\\n".join(table)))
    selected_name = summary["selected_model"]
    report = json.loads((analysis / selected_name / "classification_report.json").read_text())
    table = [f"### Per-class report: {selected_name}", "", "| Class | Precision | Recall | F1 | Support |",
             "|---|---:|---:|---:|---:|"]
    table += [f"| {name} | {report[name]['precision']:.3f} | {report[name]['recall']:.3f} | {report[name]['f1-score']:.3f} | {report[name]['support']} |" for name in CLASS_NAMES]
    display(Markdown("\\n".join(table)))
    display(DisplayImage(filename=str(analysis / "per_class_f1.png")))
    for row in rows:
        display(DisplayImage(filename=str(analysis / row["run"] / "confusion_matrix.png")))
    failures = json.loads((analysis / "failure_patterns.json").read_text())
    print("Lowest F1 class:", failures["lowest_f1_class"])
    print("Largest confusion pairs:")
    for pair in failures["confusion_pairs"][:5]:
        print(f"  True {pair['true_class']} -> predicted {pair['predicted_class']}: {pair['count']} images")
""")
cell("markdown", """
    ## 6. Review the 40 Grad-CAM cases - pause here

    Every panel shows the **actual input crop**, the **predicted-class** heatmap,
    and the **true-class** heatmap. For a correct prediction, the two heatmaps match.
    Brighter colours show relatively stronger positive sensitivity to the selected
    class logit. Each map is normalized separately; colour strength cannot be compared
    numerically between maps. Flat maps contain no positive spatial contrast.

    Inspect where the heatmap falls: flower, background, both, or unclear. Choose an
    observed pattern, optionally add a note, then click **Save and next**.
    Continue until **Reviewed 40/40 (20 correct, 20 incorrect)**.

    Heatmaps are approximate explanations, not object boundaries or proof of a cause.
    A background hotspot is a candidate concern; it does not by itself prove background
    bias. Softmax scores are uncalibrated. Avoid inventing an explanation when unclear.

    Review progress is saved after each click. Rerunning this form cell resumes the
    first unfinished case. Do not rerun the analysis cell to resume reviewing.
""")
cell("code", """
    if IN_COLAB:
        from google.colab import output
        output.enable_custom_widget_manager()
    review_form = launch_review_form(analysis)
""")
cell("markdown", """
    ## 7. Check your review and summarize the patterns

    Run this after the form reaches 40/40. The counts below summarize your recorded
    observations. They describe this deliberately selected sample, not population rates.
""")
cell("code", """
    from collections import Counter

    progress = review_progress(analysis)
    print(progress)
    if progress["remaining"] or progress["correct_reviewed"] < 20 or progress["incorrect_reviewed"] < 20:
        raise ValueError("Return to the form and finish all 20 correct + 20 incorrect cases, then rerun this cell.")
    cases = json.loads((analysis / "review_cases.json").read_text())
    wrong_cases = [case for case in cases if not case["correct"]]
    attention_counts = dict(Counter(case["attention"] for case in wrong_cases))
    pattern_counts = dict(Counter(case["pattern"] for case in wrong_cases))
    print("Incorrect-case heatmap categories:", attention_counts)
    print("Incorrect-case observed patterns:", pattern_counts)
""")
cell("markdown", """
    ## 8. Write two observation answers

    Replace both TODO strings with your own evidence, then run this cell.
    Mention recurring confusions and what you saw in the incorrect cases. Propose
    **one** feasible fix for Day 5 and explain which validation evidence supports it.
    For example, a crop/scale augmentation change may help small-subject errors;
    such a change is a hypothesis to test, not a guaranteed improvement.

    Keep labels and splits fixed. Do not choose a fix using test-set errors.
    If you want help interpreting your results, share `day4_observations.json` and
    your metrics/review outputs. You can download this JSON before making the final backup.
""")
cell("code", """
    selected_metrics = next(row for row in rows if row["run"] == selected_name)
    my_observations = {
        "analysis_split": "val",
        "selected_model": selected_name,
        "validation_accuracy": selected_metrics["accuracy"],
        "validation_macro_f1": selected_metrics["macro_f1"],
        "lowest_f1_class": failures["lowest_f1_class"],
        "top_confusion_pairs": failures["confusion_pairs"][:5],
        "review_progress": progress,
        "incorrect_attention_counts": attention_counts,
        "incorrect_pattern_counts": pattern_counts,
        "failure_patterns_with_evidence": "TODO: describe recurring errors and visible heatmap patterns, with case IDs or counts; distinguish observations from possible causes",
        "one_proposed_fix_and_reason": "TODO: choose one specific change for Day 5 and explain which validation evidence supports testing it",
        "limitations": "One split and seed; review errors selected partly by confidence; Grad-CAM is approximate and independently normalized; softmax scores are uncalibrated; no test metrics yet.",
    }
    write_json(analysis / "day4_observations.json", my_observations)
    print("Saved:", analysis / "day4_observations.json")
    if any("TODO" in str(value) for value in my_observations.values()):
        print("Replace both TODO answers and rerun this cell before making your final backup.")
""")
cell("markdown", """
    ## 9. Download your completed Day 4 backup

    This cell checks that all 40 cases have been reviewed and both observation answers
    are filled. It includes the four checkpoints, original splits, earlier results,
    new metrics/plots, all heatmaps, the standalone HTML gallery, and your review notes.

    Keep **`day4_backup.zip`**. Also download this executed notebook using the top
    **File -> Download -> Download .ipynb** menu. The notebook download is separate
    from the left Files panel and separate from the ZIP download.
""")
cell("code", """
    exact_versions = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (analysis / "requirements-runtime.txt").write_text(exact_versions, encoding="utf-8")
    backup = package_day4(ROOT, analysis)
    print("Backup:", backup)
    if IN_COLAB:
        from google.colab import files
        files.download(str(backup))
""")
cell("markdown", """
    ## Next: Day 5

    Test one proposed fix using validation, then run the final held-out test comparison,
    create an upload demo, and prepare the report/presentation.

    Sources:
    - [Grad-CAM paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf)
    - [PyTorch autograd](https://docs.pytorch.org/docs/stable/autograd.html)
    - [Classification report definitions](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

    The helper computes these report metrics directly from confusion counts with zero
    division set to zero; scikit-learn is not required. Image credits are retained in the backup.
""")
notebook = {"cells": cells, "metadata": {"colab": {"name": "04_evaluation_gradcam.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
destination = ROOT / "notebooks/04_evaluation_gradcam.ipynb"
destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Created {destination} ({len(cells)} cells)")
