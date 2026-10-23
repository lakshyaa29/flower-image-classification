"""Regenerate the standalone Colab notebook from the checked-in Python helpers."""

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def cell(kind, text):
    item = {"cell_type": kind, "id": f"day1-{len(cells):02}", "metadata": {},
            "source": textwrap.dedent(text).strip().splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    cells.append(item)


cell("markdown", """
    # Flower Classification - Day 1: understand and prepare the data

    **Goal:** turn the flower photos into audited, reproducible datasets ready for a CNN.
    No model is trained in this notebook.

    1. In Colab, select a GPU runtime if one is available. Today's data checks also work on CPU.
    2. Run the cells in order. The first run downloads a 218 MiB archive.
    3. Check the figures and answer the short review questions.
    4. Download the Day 1 backup at the end, before closing Colab.

    We use five folder labels: daisy, dandelion, roses, sunflowers, and tulips.
    The project brief asks you to get your mentor's approval for the dataset.

    **Later:** custom CNN -> augmentation experiment -> frozen ResNet18 -> full fine-tuning
    -> evaluation and Grad-CAM -> demo and report.
""")
cell("markdown", """
    ## 1. Check the environment

    Colab usually already has PyTorch. Preserve its installed torch/torchvision pair.
    Install only missing packages and record the actual versions.
""")
cell("code", """
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path

    package_modules = {"Pillow": "PIL", "numpy": "numpy", "matplotlib": "matplotlib",
                       "torch": "torch", "torchvision": "torchvision"}
    missing = [package for package, module in package_modules.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

    IN_COLAB = importlib.util.find_spec("google.colab") is not None if importlib.util.find_spec("google") else False
    current = Path.cwd()
    ROOT = Path("/content/flower-cnn") if IN_COLAB else (current.parent if current.name == "notebooks" else current)
    ROOT.mkdir(parents=True, exist_ok=True)
    print("Project directory:", ROOT)
""")
helper_sources = {name: (ROOT / name).read_text(encoding="utf-8") for name in ("flower_data.py", "flower_pipeline.py")}
cell("markdown", """
    ## 2. Load the project helpers

    The next cell contains the same helper code as the project scripts, so this notebook
    works as a standalone upload. Run the cell without editing the embedded source.
    Existing local helper files are preserved; Colab receives its own copies.
""")
cell("code", "# @title Set up the reusable project helpers\n"
     + "HELPER_SOURCES = " + repr(helper_sources) + "\n"
     + "for name, source in HELPER_SOURCES.items():\n"
       "    destination = ROOT / name\n"
       "    if IN_COLAB or not destination.exists():\n"
       "        destination.write_text(source, encoding='utf-8')\n"
       "if str(ROOT) not in sys.path:\n"
       "    sys.path.insert(0, str(ROOT))\n"
       "import torch\n"
       "from flower_data import prepare_project, record_environment\n"
       "from flower_pipeline import make_loaders, save_eda_charts, seed_everything\n"
       "SEED = 42\n"
       "seed_everything(SEED)\n"
       "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
       "print('PyTorch:', torch.__version__)\n"
       "print('Device:', device)\n"
       "if torch.cuda.is_available():\n"
       "    print('GPU:', torch.cuda.get_device_name(0))\n")
cells[-1]["metadata"]["cellView"] = "form"
cell("markdown", """
    ## 3. Download, audit, and split

    The helper verifies the official archive's SHA-256 checksum, decodes every image,
    excludes corrupt files, removes exact duplicates, and flags conflicting duplicate labels.
    It leaves the original photos unchanged.

    Each flower class is split approximately **70% training / 15% validation / 15% test**.
    Stratification keeps each class represented in every split. A fixed seed makes the
    split repeatable. All later experiments must reuse `data/processed/splits.csv`.

    Exact duplicate checks do not catch all crops, alternate views, or near duplicates.
    Folder labels still need visual spot checks.
""")
cell("code", r"""
    summary = prepare_project(ROOT, seed=SEED)
    print("\nClass          Train   Val  Test")
    for name, counts in summary["class_counts"].items():
        print(f"{name:13} {counts['train']:5} {counts['val']:5} {counts['test']:5}")
    audit = summary["audit"]
    print("\nUnreadable files:", len(audit["corrupt_images"]))
    print("Exact duplicate copies excluded:", len(audit["excluded_duplicates"]))
    print("Conflicting-label groups excluded:", len(audit["conflicting_label_groups"]))
    print("Width range:", summary["width_range"])
    print("Height range:", summary["height_range"])
""")
cell("markdown", """
    ## 4. Explore the images

    The grid shows **training images only**. Look for incorrect labels, distracting
    backgrounds, multiple flowers, very small flowers, and unusual lighting.
    The charts show class balance and why resizing is necessary.
""")
cell("code", """
    from IPython.display import Image as DisplayImage, display

    display(DisplayImage(filename=str(ROOT / "outputs/day1/sample_grid.jpg")))
    chart_path = save_eda_charts(ROOT)
    display(DisplayImage(filename=str(chart_path)))
""")
cell("markdown", """
    ## 5. Build PyTorch image loaders and check one batch

    A Dataset reads one photo and its label; a DataLoader assembles photos into batches.
    Images become tensors shaped `[batch, 3, 224, 224]`.

    The baseline uses deterministic resizing and cropping. Later, a separate experiment
    enables random augmentation **only for training**. Validation and test preprocessing
    stay deterministic. ImageNet normalization prepares the pipeline for ResNet18.
""")
cell("code", """
    loaders, class_names = make_loaders(ROOT, batch_size=32, num_workers=0, augment=False, seed=SEED)
    images, labels = next(iter(loaders["train"]))
    assert images.ndim == 4 and tuple(images.shape[1:]) == (3, 224, 224)
    assert torch.isfinite(images).all()
    assert labels.min().item() >= 0 and labels.max().item() < len(class_names)
    print("Image tensor:", tuple(images.shape))
    print("Label tensor:", tuple(labels.shape))
    print("First five labels:", [class_names[index] for index in labels[:5].tolist()])
    print("Batches per split:", {name: len(loader) for name, loader in loaders.items()})
    record_environment(ROOT / "outputs/day1/environment.json")
    print("Day 1 pipeline is ready. No training has run yet.")
""")
cell("markdown", """
    ## 6. Your notes

    Write your answers in the dictionary in the next cell:

    - Which class has the most images? Which has the fewest?
    - Do any training examples look mislabeled or ambiguous?
    - What backgrounds might the model learn instead of the flower?
    - Why do we need separate validation and test sets?
    - Why must augmentation happen after the split, only on training images?

    Use validation data to choose settings and investigate fixes. Keep the test set
    for final comparisons. Do not select improvements using test performance.
""")
cell("code", """
    import json

    my_observations = {
        "largest_and_smallest_classes": "TODO: read the class counts above",
        "ambiguous_training_examples": "TODO: inspect the sample grid",
        "possible_background_bias": "TODO: describe what you noticed",
        "why_validation_and_test_are_separate": "TODO: explain in your own words",
        "why_augmentation_is_training_only": "TODO: explain in your own words",
    }
    (ROOT / "outputs/day1/my_observations.json").write_text(
        json.dumps(my_observations, indent=2) + "\\n", encoding="utf-8"
    )
    print("Observations saved. Edit the TODO answers and rerun this cell before backing up.")
""")
cell("markdown", """
    ## 7. Save your work

    Colab's runtime storage is temporary. This cell creates a small backup containing
    helper scripts, the split manifests, audit reports, figures, exact package versions,
    and your notes. It excludes the image archive and photographs; those can be downloaded again.

    Save a copy of this notebook to Drive as well. Re-running it recreates the dataset
    and the same split when the source data and seed are unchanged.
""")
cell("code", r"""
    import zipfile

    exact_versions = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (ROOT / "outputs/day1/requirements-runtime.txt").write_text(exact_versions, encoding="utf-8")
    notes = ROOT / "outputs/day1/my_notes.md"
    if not notes.exists():
        notes.write_text("# My Day 1 observations\n\nWrite your observations here.\n", encoding="utf-8")
    backup = ROOT / "day1_backup.zip"
    with zipfile.ZipFile(backup, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in ("data/processed", "outputs/day1"):
            for path in sorted((ROOT / relative).rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(ROOT))
        for name in HELPER_SOURCES:
            archive.write(ROOT / name, name)
    print("Backup created:", backup)
    if IN_COLAB:
        from google.colab import files
        files.download(str(backup))
""")
cell("markdown", """
    ## Next: the custom CNN

    Day 2 will add three convolution blocks with batch normalization and pooling,
    a training/validation loop, best-checkpoint saving, and a comparison of training
    with and without augmentation. Keep today's split fixed.

    Sources:
    - [Official Flower Photos dataset](https://www.tensorflow.org/tutorials/load_data/images)
    - [Dataset checksum](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/tf_flowers/checksums.tsv)
    - [PyTorch Dataset and DataLoader tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)

    Image credits and individual CC-BY attribution are in the downloaded `flower_photos/LICENSE.txt`.
""")
notebook = {"cells": cells, "metadata": {"colab": {"name": "01_data_preparation.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
destination = ROOT / "notebooks" / "01_data_preparation.ipynb"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Created {destination} ({len(cells)} cells)")
