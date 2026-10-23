# 15-minute presentation and live demo

Final report - frozen selection and held-out test evaluation

## 0:00-1:00 - Project question
Explain the five flower classes and the goal: compare a CNN trained from scratch with pretrained ResNet18. State that checkpoint/model selection used validation, then test results were opened after freezing the decision.

## 1:00-3:00 - Dataset and audit
Show the Day 1 image grid and class counts. Explain 3,670 original images, four exclusions, and 2,564/551/551 fixed splits. Explain exact duplicate and conflicting-label exclusions, and the remaining near-duplicate/label limitations.

## 3:00-5:00 - Custom CNN experiments
Describe the three convolution blocks. Compare baseline and augmentation under the same 12-epoch budget and starting weights. Show the saved curves rather than claiming augmentation must always help.

## 5:00-7:00 - Transfer learning
Explain ImageNet features, the five-class replacement head, fixed backbone BatchNorm in the head stage, and lower-rate full fine-tuning. Show the four original validation results. Acknowledge capacity, pretraining, and budget differences.

## 7:00-9:00 - Errors and Grad-CAM
Day 4's largest validation confusions were dandelion to daisy (7), roses to tulips (7), and tulips to roses (5). Roses had the lowest per-class F1. The 40-case diagnostic review covered 20 correct and 20 incorrect predictions. Review attribution: 13 user and 27 authorized AI-assisted visual reviews. Error tags included six similar-class cases, five clutter cases, and three small/distant subjects. Heatmaps sometimes highlighted surrounding objects. These observations motivate a hypothesis, not a causal conclusion.
Show the validation example in the report and explain that a hotspot is an approximate sensitivity signal. Disclose the user/AI review attribution.

## 9:00-11:00 - One controlled improvement attempt
Both continuations start from the same checkpoint, use the same six-epoch default budget and learning rate, and differ only in crop scale. The crop arm's validation macro-F1 was 0.9357, versus 0.9379 for matched continued training. It was lower by 0.0022. The final validation-selected model is resnet18_finetuned.
State the observed result even if the crop arm loses. One seed does not establish a reliable effect. Smaller crops can remove the subject.

## 11:00-13:00 - Final test results
Selected model: resnet18_finetuned. Validation accuracy 94.01%, macro-F1 0.9393. Final test accuracy 95.10%, macro-F1 0.9494, on 551 test images. The selected model remained fixed after testing.
Show final_test_comparison.csv and the selected model confusion matrix. Explain that the test ranking did not change model selection.

## 13:00-15:00 - Live upload demo
Open the private notebook upload form or the localhost Gradio app. Upload one flower photo, explain all five scores, and show the predicted-class heatmap. Try another viewpoint. Explain that scores are uncalibrated and unfamiliar inputs still receive a known label. Finish with the key limitation and one future improvement.

## 10-minute Q&A preparation
- Why macro-F1? It weights the five classes equally and shows weaknesses hidden by accuracy.
- Why a matched control? Extra training could improve the model even without a crop change.
- Why keep the original model if it wins? Validation decides; an attempted fix need not improve results.
- Why freeze before test? Test feedback should not guide checkpoint or hyperparameter selection.
- Does Grad-CAM prove background bias? No. It suggests a region to investigate with controlled tests.
- What remains uncertain? One seed and one split limit generalization claims. ResNet18 benefits from ImageNet pretraining and greater capacity. The head-only run had six epochs, original fine-tuning had 12 allocated epochs including head training, and each Day 5 arm adds 6 epochs from the same selected fine-tuned checkpoint. Budgets differ across the six models. The review sample favors high-confidence errors and is not representative of all failures. Exact pixel duplicate checks do not eliminate near duplicates or every label problem. Grad-CAM is approximate. Scores are uncalibrated. Smaller training crops can exclude flowers.

## Files to show
Use the three-page final_report.pdf, saved learning curves, selected confusion matrix, Day 4 gallery, and upload demo. Keep the full backup and image credits available.
