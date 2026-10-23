"""Build the final-day Colab notebook with all project helper sources embedded."""

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def cell(kind, source):
    item = {"cell_type": kind, "id": f"day5-{len(cells):02}", "metadata": {},
            "source": textwrap.dedent(source).strip().splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    cells.append(item)


cell("markdown", """
    # Day 5 - Controlled improvement, final testing, and upload demo

    Keep **`day4_backup.zip`** ready. Select **Runtime -> Change runtime type -> T4 GPU**.
    Run the cells in order. Today has two six-epoch continuations from the same saved
    ResNet18 checkpoint:

    - **Control:** original training crop area `(0.75, 1.0)`.
    - **Crop experiment:** proposed training crop area `(0.5, 1.0)`.

    Both use the same parent weights, seed, optimizer, lower learning rate, and epoch
    budget. Only crop scale changes. This tests whether the proposed fix helps beyond
    extra training. It may help or hurt; smaller crops can also exclude flowers.

    The notebook selects from all six saved candidates using validation, freezes the
    selection, and then evaluates all six on the held-out test set. Test results never
    change the selected model. Completed training and final test results are cached
    for safe reruns in the same runtime. Keep your final ZIP to preserve those records.

    Outputs include the six-model comparison, three-page PDF report, presentation notes,
    private upload demo, runnable Gradio app, and final backup. Observations are filled
    from actual results; there are no required TODO answers today.
""")
cell("markdown", """
    ## 1. Packages and GPU

    Keep Colab's supplied torch/torchvision pair. Only missing packages are installed.
    Training expects a GPU. CPU is possible with `ALLOW_CPU=True`, but takes longer.
""")
cell("code", """
    import importlib.util
    import subprocess
    import sys
    import os
    import json
    from pathlib import Path

    ALLOW_CPU = False
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    modules = {"Pillow": "PIL", "numpy": "numpy", "matplotlib": "matplotlib", "torch": "torch",
               "torchvision": "torchvision", "tensorboard": "tensorboard", "ipywidgets": "ipywidgets",
               "gradio>=5.49,<7": "gradio", "reportlab>=4,<5": "reportlab", "pypdf>=5,<7": "pypdf"}
    missing = [package for package, module in modules.items() if importlib.util.find_spec(module) is None]
    if importlib.util.find_spec("gradio") is not None:
        from importlib.metadata import version
        from packaging.version import Version
        if not Version("5.49") <= Version(version("gradio")) < Version("7"):
            missing.append("gradio>=5.49,<7")
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    import torch
    IN_COLAB = importlib.util.find_spec("google.colab") is not None if importlib.util.find_spec("google") else False
    current = Path.cwd()
    ROOT = Path("/content/flower-cnn") if IN_COLAB else (current.parent if current.name == "notebooks" else current)
    ROOT.mkdir(parents=True, exist_ok=True)
    print("PyTorch:", torch.__version__)
    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    elif not ALLOW_CPU:
        raise RuntimeError("Select a T4 GPU runtime and rerun setup. If this still shows CPU, start a fresh GPU runtime. For deliberate CPU training set ALLOW_CPU=True.")
""")
cell("markdown", """
    ## 2. Load the included project helpers

    The helper sources are embedded here. Restoring your backup will load data/results,
    rather than executing Python from the ZIP. Completed earlier notebooks remain separate.
""")
names = ("flower_data.py", "flower_pipeline.py", "flower_training.py", "flower_transfer.py", "flower_evaluation.py",
         "flower_review.py", "flower_final.py", "flower_demo.py", "flower_deliverables.py", "requirements.txt")
sources = {name: (ROOT / name).read_text(encoding="utf-8") for name in names}
cell("code", "# @title Load the reusable project files\nHELPER_SOURCES = " + repr(sources) + "\n"
     + "for name, source in HELPER_SOURCES.items():\n"
       "    destination = ROOT / name\n"
       "    if IN_COLAB or not destination.exists():\n"
       "        destination.write_text(source, encoding='utf-8')\n"
       "if str(ROOT) not in sys.path:\n"
       "    sys.path.insert(0, str(ROOT))\n"
       "import importlib\n"
       "for module_name in ('flower_data', 'flower_pipeline', 'flower_training', 'flower_transfer', 'flower_evaluation', 'flower_review', 'flower_final', 'flower_demo', 'flower_deliverables'):\n"
       "    importlib.reload(importlib.import_module(module_name))\n"
       "from flower_data import download_dataset, write_json\n"
       "from flower_final import (restore_day4_backup, load_day4_results, run_day5_experiment, freeze_final_selection, evaluate_final_test, write_day5_observations, package_day5)\n"
       "from flower_demo import FlowerPredictor, export_demo_files, launch_private_upload, create_gradio_demo\n"
       "from flower_deliverables import build_deliverables\n"
       "print('Day 5 helpers ready:', ROOT)\n")
cells[-1]["metadata"]["cellView"] = "form"
cell("markdown", """
    ## 3. Restore the completed Day 4 backup

    Upload **`day4_backup.zip`** if prompted. It includes your original splits, four
    models, completed observations, and 40 review records. The restore preserves the
    distinction between your 13 reviews and 27 authorized AI-assisted reviews.
    Photographs are downloaded only if missing; pretrained weights need no new download.
""")
cell("code", """
    if not (ROOT / "outputs/day4_import.json").is_file():
        if IN_COLAB:
            from google.colab import files
            print("Upload your completed day4_backup.zip now.")
            uploaded = files.upload()
            backups = [name for name in uploaded if name.lower().endswith(".zip")]
            if len(backups) != 1:
                raise ValueError("Select exactly one completed Day 4 ZIP and rerun this cell.")
            restore_day4_backup(Path(backups[0]), ROOT)
        else:
            raise FileNotFoundError("Restore your Day 4 backup with restore_day4_backup(backup_path, ROOT) first.")
    day4_directory, day4_summary, day4_notes, earlier_checkpoints = load_day4_results(ROOT)
    download_dataset(ROOT / "data/raw")
    print("Verified Day 4 results:", day4_directory)
    print("Completed review:", day4_summary["review_progress"])
    print("Review attribution:", day4_summary.get("reviewer_counts", {}))
""")
cell("markdown", """
    ## 4. Run the matched control and crop experiment

    Wait for **six control epochs, then six crop epochs**. Each run starts from the
    best original fine-tuned ResNet18, using learning rate `1e-5` and weight decay `1e-4`.
    Validation/test preprocessing stays fixed. Best checkpoints use validation macro-F1.

    Once this cell finishes, rerunning it reuses the same completed experiment. After
    final testing, preserve the selected checkpoint and results rather than retuning.
""")
cell("code", """
    EPOCHS = 6
    BATCH_SIZE = 32
    SEED = 42
    NUM_WORKERS = 2 if IN_COLAB else 0

    experiment = run_day5_experiment(ROOT, epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED,
                                     num_workers=NUM_WORKERS, allow_cpu=ALLOW_CPU, smoke=False)
    comparison = json.loads((experiment / "comparison.json").read_text())
""")
cell("markdown", """
    ## 5. Compare validation and freeze the selection

    The direct crop comparison is **crop versus control**. The original four models
    remain candidates, so a failed improvement attempt does not replace a stronger
    earlier checkpoint. Day 5 arms have 18 allocated epochs including prior stages;
    the original CNNs/fine-tuned ResNet18 have 12, and the head-only run has six.
""")
cell("code", """
    from IPython.display import display, Markdown, Image as DisplayImage

    table = ["| Model | Val accuracy | Val macro-F1 | Allocated epochs |", "|---|---:|---:|---:|"]
    table += [f"| {row['run']} | {row['val_accuracy']:.2%} | {row['val_macro_f1']:.4f} | {row['allocated_total_epochs']} |" for row in comparison["runs"]]
    display(Markdown("\\n".join(table)))
    display(DisplayImage(filename=str(experiment / "learning_curves.png")))
    print("Crop macro-F1 change versus matched control:", comparison["crop_macro_f1_change_vs_control"])
    plan = freeze_final_selection(ROOT, experiment)
    print("Selected BEFORE test:", plan["selected_by_validation"])
    print("Frozen plan SHA-256:", plan["plan_sha256"])
""")
cell("markdown", """
    ## 6. Final held-out test evaluation

    The frozen candidate list now receives one final comparison on all **551 test
    images**. This produces reports for all six models. The validation-selected model
    stays selected regardless of test ranking. Reruns reuse cached test results.
    An interrupted test run can resume the same frozen plan without retuning.
""")
cell("code", """
    final_results = evaluate_final_test(ROOT, experiment, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    table = ["| Model | Test accuracy | Test macro-F1 | Selected before test |", "|---|---:|---:|---|"]
    table += [f"| {row['run']} | {row['test_accuracy']:.2%} | {row['test_macro_f1']:.4f} | {'Yes' if row['selected_before_test'] else ''} |" for row in final_results["runs"]]
    display(Markdown("\\n".join(table)))
    selected_name = final_results["selected_by_validation"]
    display(DisplayImage(filename=str(experiment / "test" / selected_name / "confusion_matrix.png")))
    print("Selected model unchanged after test:", not final_results["selection_changed_after_test"])
""")
cell("markdown", """
    ## 7. Try the private upload demo

    Click **Upload image** below and choose a flower photograph. You will see all five
    scores and the predicted-class heatmap. This form runs inside the notebook without
    a public Gradio link. Scores are uncalibrated; unfamiliar images still receive one
    of the five known labels. The automatic check uses one training image.
""")
cell("code", """
    import csv
    import numpy as np
    from PIL import Image as PILImage

    demo_config = export_demo_files(ROOT, experiment)
    predictor = FlowerPredictor(ROOT, demo_config)
    with (ROOT / "data/processed/splits.csv").open(newline="") as stream:
        demo_example = next(row for row in csv.DictReader(stream) if row["split"] == "train")
    with PILImage.open(ROOT / "data/raw/flower_photos" / demo_example["relative_path"]) as image:
        demo_scores, demo_heatmap = predictor.predict(image)
    assert len(demo_scores) == 5 and abs(sum(demo_scores.values()) - 1) < 1e-5
    assert demo_heatmap.shape == (224, 224, 3) and np.isfinite(demo_heatmap).all()
    write_json(experiment / "demo_check.json", {"method": "Automatic inference check on one training image",
               "example": demo_example["relative_path"], "selected_model": selected_name,
               "five_scores_sum_to_one": True, "heatmap_shape": list(demo_heatmap.shape)})
    if IN_COLAB:
        from google.colab import output
        output.enable_custom_widget_manager()
    private_demo = launch_private_upload(predictor)
""")
cell("markdown", """
    ## 8. Optional Gradio demonstration

    A runnable Gradio app is saved in the backup for localhost use. In Colab, Gradio
    requires a temporary public share link. To launch that link for your presentation,
    deliberately set `LAUNCH_PUBLIC_DEMO=True` below. Anyone with the link can use the
    classifier while it runs. The private form above works without this step.
""")
cell("code", """
    LAUNCH_PUBLIC_DEMO = False
    if LAUNCH_PUBLIC_DEMO:
        demo = create_gradio_demo(predictor)
        demo.launch(share=True, inline=True, prevent_thread_lock=True, max_file_size="10mb")
    else:
        print("Use the private upload form above. The Gradio app is also saved for localhost use.")
""")
cell("markdown", """
    ## 9. Save observations, the three-page report, and presentation notes

    The factual observations use your measured results. A losing crop experiment is
    reported honestly. The report includes the frozen selection, all six final test
    scores, error analysis, review attribution, and limitations. Read it before sharing.

    `presentation_notes.md` provides a 15-minute walkthrough with a live demo and
    preparation for ten minutes of questions. There are no required TODO fields.
""")
cell("code", """
    observations = write_day5_observations(ROOT, experiment)
    print(json.dumps(observations, indent=2))
    report_path = build_deliverables(ROOT, experiment)
    from pypdf import PdfReader
    print("Report pages:", len(PdfReader(str(report_path)).pages))
    print("Report:", report_path)
    print("Presentation notes:", experiment / "presentation_notes.md")
""")
cell("markdown", """
    ## 10. Download and preserve the final submission

    Download **day5_backup.zip**, the **three-page PDF**, and the **presentation notes**.
    The ZIP includes all six checkpoints, original splits and earlier results, case
    reviews, logs, final test reports, frozen selection, observations, and runnable demo.
    Raw photographs are excluded. It is larger than the Day 4 ZIP.

    Download the executed notebook separately using the top
    **File -> Download -> Download .ipynb** menu. Keep these files before disconnecting.
""")
cell("code", """
    exact_versions = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (experiment / "requirements-runtime.txt").write_text(exact_versions, encoding="utf-8")
    backup = package_day5(ROOT, experiment)
    print("Final backup:", backup)
    if IN_COLAB:
        from google.colab import files
        files.download(str(backup))
        files.download(str(experiment / "final_report.pdf"))
        files.download(str(experiment / "presentation_notes.md"))
""")
cell("markdown", """
    ## Submission check

    Keep the executed notebook, final ZIP, PDF report, and presentation notes. Try the
    demo and rehearse its upload step. Share the notebook and ZIP for the final check.
    Further experiments after viewing test results require a new evaluation protocol.

    Sources:
    - [Torchvision crop scale uses an area fraction](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomResizedCrop.html)
    - [PyTorch transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
    - [Gradio sharing behaviour](https://www.gradio.app/guides/sharing-your-app)
    - [Grad-CAM paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf)
""")
notebook = {"cells": cells, "metadata": {"colab": {"name": "05_final_experiment_demo.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
destination = ROOT / "notebooks/05_final_experiment_demo.ipynb"
destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Created {destination} ({len(cells)} cells)")
