# Reproducing the five-day workflow

## Environment and artifacts

Use Python 3.12 locally or the current Colab Python runtime with its compatible PyTorch/torchvision pair. The completed training used Python 3.13.16, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, Pillow 11.3.0, NumPy 2.1.3, and matplotlib 3.10.0. Each recorded run includes its environment and configuration. Full Colab package exports are historical records, not portable installation files; install the root requirements locally.

The code repository contains small artifacts and completed notebooks. The separate completed Day 5 backup contains all six checkpoints, original manifests, full output folders, TensorBoard events, and the frozen selection/test lock. Its exact SHA-256 is recorded in `results/provenance/artifacts.json`.

Restore the accepted artifacts without retraining:

```powershell
python tools/restore_artifacts.py --backup "path\to\day5_backup.zip"
python app.py --check
python app.py
```

Restoration accepts the recorded backup, checks paths/checkpoint hashes, reuses identical files, and rejects conflicts. It restores outputs/manifests and skips archived source files. The committed Python helpers are the same code as the final backup, apart from repository-specific launch/restore/check tools.

## Run from the beginning in a fresh directory/runtime

1. **Day 1:** open `notebooks/01_data_preparation.ipynb`. Download and audit Flower Photos, inspect EDA, and create seed-42 stratified splits. Keep the backup and executed notebook. Raw data are downloaded from the official source and checked against the archive checksum.
2. **Day 2:** open `02_custom_cnn.ipynb`, restore Day 1, and train the baseline and augmented CNN for 12 epochs each from the same initial weights. Keep histories, TensorBoard events, best checkpoints, and observations.
3. **Day 3:** open `03_resnet18_transfer.ipynb`, restore Day 2, train the replacement head for six epochs with frozen backbone/BatchNorm statistics, then fine-tune all layers for six epochs at a lower rate. Official ImageNet weights are verified.
4. **Day 4:** open `04_evaluation_gradcam.ipynb`, restore Day 3, evaluate on validation, and review 20 correct/20 incorrect Grad-CAM cases. The clean template requests human review. The accepted historical submission records 13 user reviews and 27 authorized AI-assisted visual reviews; it does not claim 40 independent human reviews.
5. **Day 5:** open `05_final_experiment_demo.ipynb`, restore the completed Day 4 backup, and run the matched continuations. Both begin with the same best original fine-tuned weights, seed 42, AdamW at 1e-5, weight decay 1e-4, cosine scheduling, six epochs, and batch size 32. Only crop scale differs: `(0.75, 1.0)` control versus `(0.5, 1.0)` treatment. Generate the frozen plan, final test comparison, demo, report, and backup.

Every notebook embeds the helper sources. Download executed notebooks separately using Colab's top **File > Download > Download .ipynb** menu. The `completed/` snapshots retain execution counts and metric/figure outputs, while Colab authorship metadata and interactive widget views/state have been removed for publication. Original and published hashes are recorded in the provenance registry.

## Selection and held-out evaluation

The split manifest SHA-256 is:

```text
4fc79e62cb179ff75a18772ba16f72e091184263e6fb8f4731819bbdc07c2dbb
```

`.gitattributes` preserves the manifest and frozen artifact bytes across operating systems. Do not alter line endings in these files before validation.

The selected model maximizes validation macro-F1, with lower validation loss and stable earlier-candidate ties. `selection.json` freezes all candidate checkpoint hashes and preprocessing before inference on the 551 test images. `final_test_lock.json` records completion and result hashes. Rerunning a completed Day 5 notebook reuses the recorded experiment/test outputs; modified plans or cached results are rejected.

The accepted test set has now been evaluated. Its scores are reported without selecting a different checkpoint. Further research informed by these results needs a new evaluation protocol. No local fixture test or repository check reopens real test images.

## Tracking and notebook builders

Recorded run tables and histories are in `results/day2/`, `results/day3/`, and `results/day5/`. TensorBoard events can be inspected using:

```powershell
tensorboard --logdir results
```

After editing shared helper code, regenerate clean notebook templates with the five `tools/build*notebook.py` scripts. These builders write `notebooks/*.ipynb`; the `completed/` snapshots remain historical evidence. New results should clearly identify their environment, seed, manifest, and checkpoint selection policy.
