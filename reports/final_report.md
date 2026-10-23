# Flower Image Classification

Final report - frozen selection and held-out test evaluation

## Dataset and methods

Flower Photos contains five folder labels: daisy, dandelion, roses, sunflowers, and tulips. From 3,670 original images, the audit excluded two redundant sunflower copies and both copies of one conflicting-label image. The 3,666 accepted images use fixed, stratified seed-42 splits: 2,564 train, 551 validation, and 551 test. All model selection uses validation macro-F1, with lower validation loss as the tie-breaker. The frozen model list and hashes precede test inference.

The custom CNN has three convolution/BatchNorm/ReLU/pooling blocks and a five-class classifier. Its baseline and augmented runs each used 12 epochs from the same initial weights. ResNet18 first trained a five-class head with its backbone and BatchNorm statistics frozen for six epochs, then fine-tuned all layers for six epochs. Day 5 compares two matched continuations from the same best original fine-tuned weights. Both use AdamW, learning rate 0.00001, weight decay 0.0001, cosine scheduling, and the same seed/budget. Only training crop area changes: (0.75, 1.0) versus (0.5, 1.0).

Inputs use RGB/EXIF correction and 224-pixel model inputs. Training augmentation includes random crop, horizontal flip, 15-degree rotation, and mild color jitter. Validation, test, and demo inputs use Resize(256), CenterCrop(224), and ImageNet normalization. Test scores do not change the selected model. Per-image predictions, per-class reports, confusion matrices, histories, and runtime versions accompany this report.

## Results

| Model | Val acc. | Val F1 | Test acc. | Test F1 | Epoch budget |
|---|---:|---:|---:|---:|---:|
| CNN baseline | 68.06% | 0.6740 | 73.32% | 0.7293 | 12 |
| CNN augmented | 67.51% | 0.6667 | 72.78% | 0.7207 | 12 |
| ResNet head | 84.75% | 0.8460 | 89.47% | 0.8928 | 6 |
| ResNet finetuned | 94.01% | 0.9393 | 95.10% | 0.9494 | 12 |
| ResNet control | 93.83% | 0.9379 | 95.46% | 0.9530 | 18 |
| ResNet crop | 93.65% | 0.9357 | 96.01% | 0.9577 | 18 |

The crop arm's validation macro-F1 was 0.9357, versus 0.9379 for matched continued training. It was lower by 0.0022. The final validation-selected model is resnet18_finetuned.

Selected model: resnet18_finetuned. Validation accuracy 94.01%, macro-F1 0.9393. Final test accuracy 95.10%, macro-F1 0.9494, on 551 test images. The selected model remained fixed after testing.

## Error analysis and limitations

Day 4's largest validation confusions were dandelion to daisy (7), roses to tulips (7), and tulips to roses (5). Roses had the lowest per-class F1. The 40-case diagnostic review covered 20 correct and 20 incorrect predictions. Review attribution: 13 user and 27 authorized AI-assisted visual reviews. Error tags included six similar-class cases, five clutter cases, and three small/distant subjects. Heatmaps sometimes highlighted surrounding objects. These observations motivate a hypothesis, not a causal conclusion.

One seed and one split limit generalization claims. ResNet18 benefits from ImageNet pretraining and greater capacity. The head-only run had six epochs, original fine-tuning had 12 allocated epochs including head training, and each Day 5 arm adds 6 epochs from the same selected fine-tuned checkpoint. Budgets differ across the six models. The review sample favors high-confidence errors and is not representative of all failures. Exact pixel duplicate checks do not eliminate near duplicates or every label problem. Grad-CAM is approximate. Scores are uncalibrated. Smaller training crops can exclude flowers.

Sources: TensorFlow Flower Photos image-loading tutorial (tensorflow.org/tutorials/load_data/images); PyTorch transfer-learning tutorial (docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html); Selvaraju et al., Grad-CAM, ICCV 2017. Individual image credits are preserved in docs/DATASET_IMAGE_CREDITS.txt in the backup.
