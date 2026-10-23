# Model card: five-class flower classifier

## Model and intended use

The selected model is ImageNet-pretrained ResNet18 with a five-output replacement head, trained on Flower Photos. It supports educational image classification and comparison of transfer learning with a small CNN. It accepts a photograph and returns scores for daisy, dandelion, roses, sunflowers, and tulips, plus a predicted-class Grad-CAM overlay.

The checkpoint is `resnet18_finetuned`, selected by validation before the held-out test comparison. Its SHA-256 is `f7367786fc3828752b77105cc8c4b4db8ea79e3fc98df5413ecec58305dc6dcc`. Restore it from the completed Day 5 backup; it is excluded from Git.

## Data and training

Flower Photos originally contains 3,670 images. Exact-duplicate/conflicting-label auditing excluded four images. The accepted 3,666 images use seed-42 stratified splits: 2,564 training, 551 validation, and 551 test. Class counts and exact paths are committed in the split manifest.

The head was trained for six epochs with frozen backbone and BatchNorm statistics. Full fine-tuning used six further epochs with a lower learning rate. Training uses random crop, horizontal flip, rotation, and mild color jitter. Evaluation and inference apply EXIF correction, RGB conversion, Resize(256), CenterCrop(224), tensor conversion, and ImageNet normalization. Background content outside the center crop is not visible to the model.

## Evaluation

| Split | Accuracy | Macro-F1 | Samples |
|---|---:|---:|---:|
| Validation | 94.01% | 0.9393 | 551 |
| Held-out test | 95.10% | 0.9494 | 551 |

Test per-class F1: daisy 0.9637, dandelion 0.9740, roses 0.9180, sunflowers 0.9537, and tulips 0.9378. Full precision, recall, F1, support, predictions, and confusion counts are in `results/day5/test/resnet18_finetuned/`.

The Day 5 crop intervention lost validation macro-F1 against its matched control by 0.0022. Its later test ranking did not replace the selected model. Training budgets and pretraining differ across model families; this is not an equal-compute architecture benchmark.

## Limitations

- The model always assigns one of five classes, including for unrelated photographs and flower species outside the training classes.
- Softmax scores are uncalibrated. They should not be treated as verified confidence or botanical identification certainty.
- One seed/split and an internet-photo dataset limit claims about new locations, devices, lighting, and backgrounds.
- Exact duplicates were checked; near duplicates and folder-label errors may remain.
- Grad-CAM gives approximate sensitivity at the final feature layer. It is independently normalized and does not establish causation or precise segmentation.
- The 40-case validation review is diagnostic and favors confident errors. Review attribution is 13 user and 27 AI-assisted visual reviews.

Future work includes near-duplicate review, multiple seeds, calibration, and robustness/unfamiliar-input evaluation with a newly reserved hold-out protocol. The supplied demo sample is a training image used only for a functional check.

## Provenance and licensing

The frozen plan, exact checkpoint hashes, backup hash, test lock, and notebook provenance are under `results/provenance/`. Individual image attribution is retained in `docs/DATASET_IMAGE_CREDITS.txt`. Image licenses and dependency licenses remain separate from the repository's code license.
