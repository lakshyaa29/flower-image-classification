# Project files

| Module | Responsibility |
|---|---|
| `flower_data.py` | Official dataset download/checksum, safe extraction, image audit, duplicate checks, stratified split, EDA metadata |
| `flower_pipeline.py` | PyTorch datasets/loaders, training-only augmentation, fixed evaluation transforms, plotting |
| `flower_training.py` | Three-block CNN, shared training/evaluation loop, metrics, checkpoint selection, TensorBoard |
| `flower_transfer.py` | Pretrained ResNet18, frozen-backbone/head training, full fine-tuning, prior-run restoration |
| `flower_evaluation.py` | Split-aware predictions, per-class metrics/confusion matrices, Grad-CAM, diagnostic case selection |
| `flower_review.py` | Interactive notebook case review and saved progress |
| `flower_final.py` | Matched crop experiment, validation selection, frozen test plan, cached final comparison, packaging |
| `flower_demo.py` | Checkpoint-verified prediction, Gradio interface, private notebook uploader |
| `flower_deliverables.py` | Generate the three-page report and presentation notes from measured results |
| `app.py` | Repository demo entry point and optional sample inference check |
| `tools/restore_artifacts.py` | Restore the recorded Day 5 artifact ZIP, preserving code and existing matching files |
| `tools/check_repository.py` | Independently check manifest, predictions/metrics, frozen result hashes, and review records |
| `tools/build*notebook.py` | Rebuild the five standalone notebook templates from helper code |

`notebooks/` contains clean workflow templates; `notebooks/completed/` contains executed snapshots. `data/processed/` contains the exact shared split manifest and label mapping. `results/` holds small original artifacts with their timestamp directory removed for readability; recorded paths inside JSON remain historical Colab paths. See `results/README.md` for the mapping.

`reports/` contains the already-generated final report and presentation notes. `assets/` contains one credited training sample. Runtime weights, downloads, and full outputs are restored or generated under ignored directories. The GitHub Actions workflow runs fixture tests and evidence checks without downloading photographs or using the held-out test loader.
