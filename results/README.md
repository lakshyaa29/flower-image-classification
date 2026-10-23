# Recorded experiment evidence

These are original outputs from the completed full project. Checkpoints and raw data are stored separately in the completed Day 5 backup. Partial development/smoke experiments are not included.

| Folder | Original artifact directory | Contents |
|---|---|---|
| `day1/` | `outputs/day1/` | Audit, EDA, class/split counts, observations, environment |
| `day2/` | `outputs/day2/20261023T180719_403099Z/` | CNN configurations, histories, tracking events, comparison |
| `day3/` | `outputs/day3/20261023T190527_723814Z/` | Transfer-learning configurations, histories, tracking events, comparison |
| `day4/` | `outputs/day4/20261023T195709_381690Z/` | Validation reports, confusion matrices, 40 Grad-CAM panels, saved reviews |
| `day5/` | `outputs/day5/20261023T213253_024550Z/` | Controlled continuations, frozen plan, six-model final test reports and predictions |
| `provenance/` | Frozen project metadata | Backup/notebook hashes, selected model/configuration, final result hash lock |

Artifact bytes are preserved. Existing timestamped/absolute checkpoint paths inside historical JSON describe the original Colab run; use the restoration tool to recover the original runtime structure. The repository launcher reads the root-relative checkpoint reference in the frozen demo configuration.

The standalone Grad-CAM HTML gallery and raw heatmap arrays are retained in the separate backup; all 40 readable panel PNGs and the complete review log are included here. The report/notes are indexed under `../reports/`.

Selected model: **resnet18_finetuned**, chosen on validation. Final held-out test accuracy: **95.10%**; macro-F1: **0.9494**; 551 images. The crop arm lost against its matched control on validation, so its later test ranking did not change the selection. Diagnostic review attribution is **13 user / 27 AI-assisted**, covering 20 correct and 20 incorrect cases.

To reconcile predictions, reports, splits, hashes, and reviews without loading model weights or test photographs:

```text
python tools/check_repository.py
```
