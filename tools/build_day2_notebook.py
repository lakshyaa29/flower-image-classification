"""Build the standalone, guided Colab notebook for the custom CNN comparison."""

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def cell(kind, source):
    result = {"cell_type": kind, "id": f"day2-{len(cells):02}", "metadata": {},
              "source": textwrap.dedent(source).strip().splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    cells.append(result)


cell("markdown", """
    # Day 2 - Build and train your custom CNN

    **Today's question:** how does training with data augmentation affect the same small CNN?

    You will train two fresh models: **baseline** (no augmentation) and **augmented**
    (random crop, flip, rotation, and color jitter on training images only).
    Both start with identical weights, the same seed, saved Day 1 splits, optimizer,
    batch size, and number of epochs. The test set stays untouched.

    ## Before running

    1. In Colab choose **Runtime -> Change runtime type -> T4 GPU**, if available.
    2. Keep your latest `day1_backup (2).zip` available on your computer.
    3. Run cells in order. If prompted, upload that ZIP in the restoration cell.
    4. Expect 12 epochs for each model. Watch the printed progress; runtime depends on the assigned GPU.
    5. At the end, fill your observations, download `day2_backup.zip`, and download this executed notebook from Colab's top **File** menu.

    **Day 2 is finished when:** both full runs finish, curves and comparison are saved,
    two best checkpoints exist, and your observations and backup are saved.
""")
cell("markdown", """
    ## 1. Check packages and GPU

    Preserve Colab's installed torch/torchvision pair. Only missing packages are installed.
    GPU availability should print `True`. If setup stops on a CPU runtime, switch to a
    GPU runtime and rerun this cell. `ALLOW_CPU=True` is available for deliberate CPU
    training, which can take substantially longer.
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
        raise RuntimeError(
            "Switch to Runtime > Change runtime type > T4 GPU and rerun setup. "
            "If PyTorch still reports +cpu, use a fresh GPU runtime and rerun this notebook. "
            "For deliberate CPU training, change ALLOW_CPU to True."
        )
""")
cell("markdown", """
    ## 2. Set up project helpers

    This notebook includes the Python helper files, so it works as a standalone upload.
    The data-preparation helpers are the same as Day 1. `flower_training.py` adds the
    model, training loops, logging, saved checkpoints, plots, and backup packaging.
""")
sources = {name: (ROOT / name).read_text(encoding="utf-8")
           for name in ("flower_data.py", "flower_pipeline.py", "flower_training.py")}
cell("code", "# @title Set up the reusable project files\nHELPER_SOURCES = " + repr(sources) + "\n"
     + "for name, source in HELPER_SOURCES.items():\n"
       "    destination = ROOT / name\n"
       "    if IN_COLAB or not destination.exists():\n"
       "        destination.write_text(source, encoding='utf-8')\n"
       "if str(ROOT) not in sys.path:\n"
       "    sys.path.insert(0, str(ROOT))\n"
       "from flower_data import download_dataset, write_json\n"
       "from flower_training import (SmallCNN, restore_day1_backup, validate_day1,\n"
       "                             run_comparison, package_day2)\n"
       "print('Project helpers ready:', ROOT)\n")
cells[-1]["metadata"]["cellView"] = "form"
cell("markdown", """
    ## 3. Restore the Day 1 splits and photographs

    If Day 1 files are already in this runtime, they are verified and reused. Otherwise
    this cell asks you to upload your latest Day 1 backup ZIP. Select only that ZIP.

    The original split manifest is restored unchanged and verified against your accepted
    Day 1 hash. The photo archive is downloaded again only when needed, with its official
    checksum verified. **No new split is generated.**
""")
cell("code", """
    required = [ROOT / name for name in (
        "data/processed/splits.csv", "data/processed/class_to_idx.json", "outputs/day1/data_summary.json"
    )]
    if not all(path.is_file() for path in required):
        if IN_COLAB:
            from google.colab import files
            print("Upload your latest day1_backup ZIP now.")
            uploaded = files.upload()
            backups = [name for name in uploaded if name.lower().endswith(".zip")]
            if len(backups) != 1:
                raise ValueError("Select exactly one Day 1 backup ZIP, then rerun this cell.")
            restore_day1_backup(Path(backups[0]), ROOT)
        else:
            raise FileNotFoundError("Restore your Day 1 backup with restore_day1_backup(backup_path, ROOT) first.")
    day1 = validate_day1(ROOT)
    dataset_dir = download_dataset(ROOT / "data/raw")
    print("Day 1 splits:", day1["split_counts"])
    print("Manifest SHA-256:", day1["manifest_sha256"])
    print("Photographs:", dataset_dir)
""")
cell("markdown", """
    ## 4. Understand the model

    Input: **3 x 224 x 224** pixels, after the same preprocessing used on Day 1.

    - Block 1: convolution with 16 filters -> batch normalization -> ReLU -> max pooling.
    - Block 2: the same pattern with 32 filters.
    - Block 3: the same pattern with 64 filters.
    - Adaptive pooling -> a 128-unit layer -> dropout -> five output scores (logits).

    Convolutions learn visual patterns. Pooling reduces image dimensions. Batch
    normalization stabilizes intermediate activations. Dropout provides regularization.
    Cross-entropy loss receives the raw logits; do not add a softmax before that loss.
""")
cell("code", """
    model_preview = SmallCNN()
    print(model_preview)
    print("Trainable parameters:", sum(parameter.numel() for parameter in model_preview.parameters()))
    model_preview.eval()
    with torch.no_grad():
        print("Output shape:", tuple(model_preview(torch.zeros(2, 3, 224, 224)).shape))
    del model_preview
""")
cell("markdown", """
    ## 5. Set the shared training budget

    Use the same values for both runs. The optimizer is AdamW (learning rate 0.001,
    weight decay 0.0001) with a cosine learning-rate schedule. Training minimizes
    cross-entropy. Each epoch logs training/validation loss, accuracy, and macro-F1.

    Macro-F1 weights all five classes equally. The best checkpoint is selected using
    **validation macro-F1**, with lower validation loss as a tie-breaker.
""")
cell("code", """
    EPOCHS = 12
    BATCH_SIZE = 32
    SEED = 42
    NUM_WORKERS = 2 if IN_COLAB else 0
    print(f"Two runs, {EPOCHS} epochs each, batch size {BATCH_SIZE}, seed {SEED}.")
""")
cell("markdown", """
    ## 6. Train both models

    First the baseline runs for all epochs, then the augmented model starts from the
    same initial weights. This cell prints progress every 20 training batches and at
    the end of every epoch. An asterisk marks a new best checkpoint.

    Each execution creates a new experiment folder. Avoid rerunning this training
    cell after a successful full run unless you want a new experiment.
    The augmentation result may improve or worsen under this budget; report what happens.
""")
cell("code", """
    experiment = run_comparison(
        ROOT, epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED,
        num_workers=NUM_WORKERS, allow_cpu=ALLOW_CPU, smoke=False
    )
    print("Saved experiment:", experiment)
""")
cell("markdown", """
    ## 7. Compare validation results and learning curves

    These scores come from the best checkpoint of each run. Training time covers that
    run's training, validation, checkpoint writing, and logging; it is not a pure GPU
    benchmark. Training losses are measured on differently transformed inputs, so
    compare validation scores for the model-quality comparison.

    This is a single-seed experiment. Do not generalize a small difference as proof
    that augmentation always helps or hurts. No test metrics are computed today.
""")
cell("code", r"""
    import json
    from IPython.display import Image as DisplayImage, Markdown, display

    comparison = json.loads((experiment / "comparison.json").read_text())
    rows = comparison["runs"]
    table = ["| Run | Best epoch | Val accuracy | Val macro-F1 | Time (min) |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        table.append(f"| {row['run']} | {row['best_epoch']} | {row['val_accuracy']:.2%} | "
                     f"{row['val_macro_f1']:.4f} | {row['training_seconds']/60:.1f} |")
    display(Markdown("\n".join(table)))
    display(DisplayImage(filename=str(experiment / "learning_curves.png")))
    print("Selected using validation:", comparison["selected_by_validation"])
    print("Test set evaluated:", comparison["test_evaluated"])
""")
cell("markdown", """
    ## 8. Inspect experiment tracking

    TensorBoard shows the saved scalar logs for both runs. Expand the loss, accuracy,
    and macro-F1 plots and compare their training and validation curves. The event
    files are included in the backup, along with the exported CSV histories.
""")
cell("code", """
    from IPython import get_ipython

    shell = get_ipython()
    if shell is not None:
        shell.run_line_magic("load_ext", "tensorboard")
        shell.run_line_magic("tensorboard", f"--logdir {experiment}")
    else:
        print("TensorBoard logs saved. To view them, open this notebook in Colab/Jupyter.")
""")
cell("markdown", """
    ## 9. Write your observations

    Read the curves and replace the three TODO answers below, then rerun this cell.
    Your metrics are filled in automatically from the actual runs. Describe the
    outcome you observed rather than assuming the augmented model will win.
""")
cell("code", """
    baseline = next(row for row in rows if row["run"] == "cnn_baseline")
    augmented = next(row for row in rows if row["run"] == "cnn_augmented")
    my_observations = {
        "baseline_val_macro_f1": baseline["val_macro_f1"],
        "augmented_val_macro_f1": augmented["val_macro_f1"],
        "augmentation_macro_f1_difference": augmented["val_macro_f1"] - baseline["val_macro_f1"],
        "validation_selection": comparison["selected_by_validation"],
        "what_the_curves_show": "TODO: explain the training and validation loss trends",
        "augmentation_effect": "TODO: explain the observed difference under this training budget",
        "one_limitation": "TODO: describe a limitation, such as one seed, limited epochs, or background bias",
    }
    write_json(experiment / "day2_observations.json", my_observations)
    print("Observations saved:", experiment / "day2_observations.json")
""")
cell("markdown", """
    ## 10. Back up your work

    Run this cell after filling your observations. It saves **both best model checkpoints**,
    run configurations, histories, TensorBoard events, plots, observations, package
    versions, helper scripts, and the original Day 1 split files. Raw photographs are
    excluded because they can be downloaded again.

    Keep the downloaded `day2_backup.zip`. Also save this notebook with its outputs
    using Colab's top **File -> Download -> Download .ipynb** menu. The notebook is
    saved separately from the runtime files.
""")
cell("code", """
    exact_versions = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (experiment / "requirements-runtime.txt").write_text(exact_versions, encoding="utf-8")
    backup = package_day2(ROOT, experiment)
    print("Backup:", backup)
    print("Best checkpoints:")
    for row in rows:
        print(" ", row["run"], "->", row["checkpoint"])
    if IN_COLAB:
        from google.colab import files
        files.download(str(backup))
""")
cell("markdown", """
    ## What comes next

    Day 3 adds a pretrained ResNet18, first training only its new classification head,
    then fine-tuning all layers with a lower learning rate. Keep today's splits fixed.

    Sources:
    - [PyTorch TensorBoard logging](https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)
    - [Saving and loading model weights](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)
    - [Colab GPU setup](https://docs.pytorch.org/tutorials/beginner/colab)

    Checkpoint loading uses a state dictionary and `torch.load(..., weights_only=True)`.
    Image attribution remains in the dataset's `LICENSE.txt`, also included in the backup when available.
""")
notebook = {"cells": cells, "metadata": {"colab": {"name": "02_custom_cnn.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
destination = ROOT / "notebooks/02_custom_cnn.ipynb"
destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Created {destination} ({len(cells)} cells)")
