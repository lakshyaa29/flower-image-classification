"""Build the standalone Colab notebook for Day 3 ResNet18 transfer learning."""

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def cell(kind, source):
    item = {"cell_type": kind, "id": f"day3-{len(cells):02}", "metadata": {},
            "source": textwrap.dedent(source).strip().splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    cells.append(item)


cell("markdown", """
    # Day 3 - Transfer learning with ResNet18

    **Question:** how do ImageNet-pretrained features compare with your custom CNN?

    You will run two stages:
    - **Frozen head:** keep the pretrained backbone and BatchNorm statistics fixed; train a new five-class head for six epochs.
    - **Full fine-tuning:** start from the best frozen-head checkpoint, unfreeze every layer, and train for six more epochs at a lower learning rate.

    The fine-tuned model receives a cumulative 12-epoch budget. The head-only result
    receives six epochs. Your Day 2 CNNs each received 12 epochs. These budgets and
    cumulative training times are documented in the comparison rather than hidden.
    ResNet18 also benefits from prior ImageNet training.

    ## Before running

    1. Select **Runtime -> Change runtime type -> T4 GPU**, if available.
    2. Keep your latest `day2_backup (1).zip` available on your computer.
    3. Run the cells in order. Upload that ZIP when the restoration cell asks for it.
    4. Watch six head-training epochs followed by six fine-tuning epochs.
    5. Complete your observations, download `day3_backup.zip`, and download this notebook with outputs using Colab's top **File** menu.

    Today's scores use validation data. The test set stays reserved for final evaluation.
""")
cell("markdown", """
    ## 1. Check packages and GPU

    Only missing packages are installed. Keep Colab's supplied GPU-compatible
    torch/torchvision pair. Setup should print `GPU available: True`.
    Deliberate CPU training is supported with `ALLOW_CPU=True`, but is slower.
""")
cell("code", """
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path

    ALLOW_CPU = False
    modules = {"Pillow": "PIL", "numpy": "numpy", "matplotlib": "matplotlib",
               "torch": "torch", "torchvision": "torchvision", "tensorboard": "tensorboard"}
    missing = [package for package, module in modules.items() if importlib.util.find_spec(module) is None]
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
        raise RuntimeError("Select a T4 GPU runtime and rerun setup. If PyTorch still ends in +cpu, use a fresh GPU runtime. For deliberate CPU training, set ALLOW_CPU=True.")
""")
cell("markdown", """
    ## 2. Load the project helpers

    The helper files are included in this notebook. This loads the data pipeline,
    the shared training/logging loop, and the new ResNet18 model and stages.
""")
sources = {name: (ROOT / name).read_text(encoding="utf-8") for name in
           ("flower_data.py", "flower_pipeline.py", "flower_training.py", "flower_transfer.py")}
cell("code", "# @title Load the reusable project files\nHELPER_SOURCES = " + repr(sources) + "\n"
     + "for name, source in HELPER_SOURCES.items():\n"
       "    destination = ROOT / name\n"
       "    if IN_COLAB or not destination.exists():\n"
       "        destination.write_text(source, encoding='utf-8')\n"
       "if str(ROOT) not in sys.path:\n"
       "    sys.path.insert(0, str(ROOT))\n"
       "import importlib\n"
       "for module_name in ('flower_data', 'flower_pipeline', 'flower_training', 'flower_transfer'):\n"
       "    importlib.reload(importlib.import_module(module_name))\n"
       "from flower_data import download_dataset, write_json\n"
       "from flower_transfer import (FlowerResNet18, restore_day2_backup, load_day2_results,\n"
       "                             run_transfer_comparison, package_day3)\n"
       "print('Day 3 helpers ready:', ROOT)\n")
cells[-1]["metadata"]["cellView"] = "form"
cell("markdown", """
    ## 3. Restore your completed Day 2 results

    If prompted, upload **`day2_backup (1).zip`**, or your latest completed Day 2 backup.
    It contains the original splits, your two CNN results, and their checkpoints.
    The restore checks the accepted split hash. No new split is generated.

    The flower photographs are downloaded again only when missing. The official
    ResNet18 weight download happens at training time and is cached for future runs.
""")
cell("code", """
    if not (ROOT / "outputs/day2_import.json").is_file():
        if IN_COLAB:
            from google.colab import files
            print("Upload your latest completed Day 2 backup ZIP now.")
            uploaded = files.upload()
            backups = [name for name in uploaded if name.lower().endswith(".zip")]
            if len(backups) != 1:
                raise ValueError("Select exactly one Day 2 backup ZIP, then rerun this cell.")
            restore_day2_backup(Path(backups[0]), ROOT)
        else:
            raise FileNotFoundError("Restore the Day 2 backup with restore_day2_backup(backup_path, ROOT) first.")
    day2_directory, day2_comparison = load_day2_results(ROOT)
    download_dataset(ROOT / "data/raw")
    print("Verified Day 2 results:", day2_directory)
    for row in day2_comparison["runs"]:
        print(f"{row['run']}: val accuracy {row['val_accuracy']:.2%}, macro-F1 {row['val_macro_f1']:.4f}")
""")
cell("markdown", """
    ## 4. Understand frozen features and fine-tuning

    ResNet18 has learned useful image features from ImageNet. We replace its final
    1,000-class layer with a five-class layer for your flower labels.

    During frozen-head training, only the new layer receives gradient updates.
    The backbone is in evaluation mode so its BatchNorm running statistics stay
    fixed as well. During full fine-tuning, all layers can learn, including BatchNorm.
    The learning rate is reduced from 0.001 to 0.0001.

    Preprocessing remains the same 256-pixel resize, 224-pixel center crop, and
    ImageNet normalization for validation that you used in Days 1 and 2.
""")
cell("code", """
    preview = FlowerResNet18(stage="head")
    print("New classifier:", preview.network.fc)
    print("Head-only trainable parameters:", sum(p.numel() for p in preview.parameters() if p.requires_grad))
    preview.set_stage("finetune")
    print("Fine-tuning trainable parameters:", sum(p.numel() for p in preview.parameters() if p.requires_grad))
    del preview
""")
cell("markdown", """
    ## 5. Set stage budgets

    Both stages use the same training augmentation as the augmented CNN, batch size,
    saved splits, and validation transforms. A cosine schedule is used within each
    stage. The best checkpoint uses validation macro-F1 with validation loss as a
    tie-breaker. Fine-tuning starts from the best head checkpoint, not the last epoch.
""")
cell("code", """
    HEAD_EPOCHS = 6
    FINE_TUNE_EPOCHS = 6
    BATCH_SIZE = 32
    SEED = 42
    NUM_WORKERS = 2 if IN_COLAB else 0
    print(f"Frozen head: {HEAD_EPOCHS} epochs; full fine-tuning: {FINE_TUNE_EPOCHS} additional epochs.")
""")
cell("markdown", """
    ## 6. Train both stages

    The first run downloads the official ImageNet-pretrained ResNet18 weights.
    Watch progress every 20 training batches and after every epoch.

    The frozen backbone is checked for changes after head training. The fine-tuning
    start is checked against the best head checkpoint. Each execution of this cell
    creates a new experiment folder, so keep a successful run and rerun only the
    notes and backup cells when editing observations.
""")
cell("code", """
    experiment = run_transfer_comparison(
        ROOT, head_epochs=HEAD_EPOCHS, fine_epochs=FINE_TUNE_EPOCHS,
        batch_size=BATCH_SIZE, seed=SEED, num_workers=NUM_WORKERS,
        allow_cpu=ALLOW_CPU, smoke=False
    )
    print("Saved Day 3 experiment:", experiment)
""")
cell("markdown", """
    ## 7. Compare all four results

    The table uses best-validation checkpoints. Cumulative seconds for the fine-tuned
    model include head training plus fine-tuning; the weight download/setup time is
    excluded. The head-only model has a smaller epoch budget. Model capacities and
    pretraining histories differ, so use these recorded budgets when interpreting results.

    The curves plot epoch within each stage. Fine-tuning begins from a trained head;
    its first point is not an independent from-scratch starting point.
""")
cell("code", r"""
    import json
    from IPython.display import Image as DisplayImage, Markdown, display

    comparison = json.loads((experiment / "comparison.json").read_text())
    rows = comparison["runs"]
    table = ["| Model/run | Val accuracy | Val macro-F1 | Cumulative epochs | Cumulative minutes |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        table.append(f"| {row['run']} | {row['val_accuracy']:.2%} | {row['val_macro_f1']:.4f} | "
                     f"{row['cumulative_epoch_budget']} | {row['cumulative_seconds']/60:.1f} |")
    display(Markdown("\n".join(table)))
    display(DisplayImage(filename=str(experiment / "learning_curves.png")))
    print("Selected using validation:", comparison["selected_by_validation"])
    print("Test set evaluated:", comparison["test_evaluated"])
""")
cell("markdown", """
    ## 8. Inspect TensorBoard

    Inspect training/validation loss, accuracy, and macro-F1 for both stages. Look
    for fluctuations and gaps between training and validation performance. Scalar
    logs and CSV histories are included in your backup.
""")
cell("code", """
    from IPython import get_ipython
    shell = get_ipython()
    if shell is not None:
        shell.run_line_magic("load_ext", "tensorboard")
        shell.run_line_magic("tensorboard", f"--logdir {experiment}")
    else:
        print("TensorBoard logs are saved for viewing in Colab/Jupyter.")
""")
cell("markdown", """
    ## 9. Write your observations

    Metrics are filled from your actual runs. Replace the two TODO explanations after
    reviewing the table and curves, then rerun this cell. Report the observed result;
    fine-tuning is not guaranteed to win every run.
""")
cell("code", """
    head = next(row for row in rows if row["run"] == "resnet18_head")
    fine = next(row for row in rows if row["run"] == "resnet18_finetuned")
    best_cnn = max((row for row in rows if row["model"] == "SmallCNN"), key=lambda row: (row["val_macro_f1"], -row["val_loss"]))
    my_observations = {
        "best_custom_cnn": best_cnn["run"],
        "best_custom_cnn_val_macro_f1": best_cnn["val_macro_f1"],
        "head_val_macro_f1": head["val_macro_f1"],
        "fine_tuned_val_macro_f1": fine["val_macro_f1"],
        "fine_tuning_macro_f1_change": fine["val_macro_f1"] - head["val_macro_f1"],
        "validation_selection": comparison["selected_by_validation"],
        "what_improved_and_why": "TODO: compare pretrained models with the CNN and interpret the observed head/fine-tuning results",
        "stability_and_limitations": "TODO: discuss curve fluctuations, one seed, pretraining, and the recorded stage budgets",
    }
    write_json(experiment / "day3_observations.json", my_observations)
    print("Day 3 observations saved.")
""")
cell("markdown", """
    ## 10. Back up the models and results

    After filling your observations, run this cell. The ZIP includes both ResNet18
    checkpoints, both original CNN checkpoints, original splits, histories, plots,
    TensorBoard logs, observations, environment records, and helper scripts.
    It is larger than the Day 2 ZIP because it includes two ResNet18 models.

    Keep **`day3_backup.zip`**. Also download this executed notebook from Colab's
    top **File -> Download -> Download .ipynb** menu with outputs included.
""")
cell("code", """
    exact_versions = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (experiment / "requirements-runtime.txt").write_text(exact_versions, encoding="utf-8")
    backup = package_day3(ROOT, experiment)
    print("Backup:", backup)
    if IN_COLAB:
        from google.colab import files
        files.download(str(backup))
""")
cell("markdown", """
    ## Next: evaluation and Grad-CAM

    Day 4 will examine per-class metrics, confusion patterns, and Grad-CAM explanations.
    Use validation errors to choose improvements; keep final test comparisons separate.

    Sources:
    - [PyTorch transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
    - [ResNet18 weights and preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
    - [Saving/loading weights](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)
""")
notebook = {"cells": cells, "metadata": {"colab": {"name": "03_resnet18_transfer.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
destination = ROOT / "notebooks/03_resnet18_transfer.ipynb"
destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Created {destination} ({len(cells)} cells)")
