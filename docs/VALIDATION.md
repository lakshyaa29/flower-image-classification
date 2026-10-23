# Repository verification

Checked on 24 October 2026 using Python 3.12 on Windows with PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, and Gradio 6.30.0. The completed GPU experiment used the earlier PyTorch 2.11.0/torchvision 0.26.0 pair recorded in its environment files; root requirements pin that experiment pair. The local fixture check demonstrates compatibility with the verification runtime and is not a second full training run.

## Automated checks

`python -m unittest discover -s tests -v`: **23 tests passed**. Coverage includes:

- Corrupt images, duplicate/conflicting labels, split reproducibility, and portable manifests.
- Image-weighted losses, metric calculations, and preservation of evaluation BatchNorm/model state.
- Frozen-head/full-fine-tuning behavior.
- Analytically known Grad-CAM maps, frozen-model gradients, and hook cleanup.
- Balanced, repeatable review-case selection and reviewer attribution.
- Training-only crop intervention, split mismatch rejection, validation selection, and immutable plans.
- Backup hash/path validation, repeatable restoration, skipping archived code, and preserving conflicting local artifacts.

`python tools/check_repository.py`: **passed**. Independently checked the 3,666-row manifest, all **3,306 test prediction rows** across six models, per-class metrics/confusion counts, validation-only selection, frozen plan/result/summary hashes, 40 reviewed Grad-CAM panels, and the 13 user / 27 AI-assisted attribution. No real test photographs or model weights are loaded by this check.

All ten notebooks were checked for valid notebook structure. Code cells parsed, completed snapshots contained no saved exception outputs, and source/published notebook hashes were recorded. All six checkpoint hashes were verified directly against the completed backup. Local Markdown links and credential-shaped strings were also checked before packaging.

## Demo inference

The repository command `python app.py --backup <completed-backup-path> --check` restored the recorded artifacts, reused the exact committed manifests, and loaded the selected `resnet18_finetuned` checkpoint. It returned five finite scores summing approximately to one and saved a 224 x 224 RGB Grad-CAM overlay. The credited daisy training sample was classified as daisy with a softmax score of approximately 0.99985. This is a functional check, not a generalization result.

The interface factory was verified during Day 5 development using one image input and class-score/heatmap outputs. The repository packaging check did not launch a public demo or retrain/evaluate the real test set. The included GitHub Actions workflow will run after publication; no remote CI success is claimed before that run.

## Packaging

The repository includes code, manifests, clean/executed notebooks, small recorded results, report files, and image attribution. Full backups, raw data, model weights, virtual environments, caches, and restored runtime outputs are excluded from the prepared Git upload. The existing daily work folders remain separate and unchanged.
