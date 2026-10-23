# Dataset files

`processed/splits.csv` is the exact accepted seed-42 manifest: 2,564 training, 551 validation, and 551 held-out test images. `processed/class_to_idx.json` fixes label order. These small files are committed so every comparison uses the same accepted images.

Raw Flower Photos images and the download cache are excluded from Git. Run `python flower_data.py --root . --seed 42` to download the official archive, check its checksum, audit images, and reproduce the split. The code rejects exact duplicate leakage; near duplicates may remain.

Manifest SHA-256: `4fc79e62cb179ff75a18772ba16f72e091184263e6fb8f4731819bbdc07c2dbb`.

The audit excludes two redundant sunflower copies and both copies of one rose/tulip conflicting-label photo, leaving 3,666 accepted images from 3,670 originals. Individual photo credits are in `../docs/DATASET_IMAGE_CREDITS.txt`. The demo sample in `../assets/` is from training, not the held-out test set.
