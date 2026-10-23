# Day 4 validation evaluation

Selected model: resnet18_finetuned. Test set was not evaluated.

| Model | Accuracy | Macro-F1 | Errors |
|---|---:|---:|---:|
| cnn_baseline | 68.06% | 0.6740 | 176 |
| cnn_augmented | 67.51% | 0.6667 | 179 |
| resnet18_head | 84.75% | 0.8460 | 84 |
| resnet18_finetuned | 94.01% | 0.9393 | 33 |

## Per-class report: resnet18_finetuned

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| daisy | 0.9091 | 0.9474 | 0.9278 | 95 |
| dandelion | 0.9618 | 0.9333 | 0.9474 | 135 |
| roses | 0.9167 | 0.9167 | 0.9167 | 96 |
| sunflowers | 0.9714 | 0.9714 | 0.9714 | 105 |
| tulips | 0.9333 | 0.9333 | 0.9333 | 120 |

Grad-CAM cases generated: 20 correct and 20 incorrect.
Complete the saved review form for every case before finalizing the Day 4 backup.
Use validation evidence to propose one fix. Final held-out test comparisons are reserved for Day 5.
