# Notebooks

The five numbered notebooks in this folder are clean templates for reproducing the workflow in Colab. They embed trusted helper sources, use `/content/flower-cnn` as their Colab working directory, and restore the previous day's artifact when needed. This internal runtime path does not need to match the GitHub repository name.

| Notebook | Stage |
|---|---|
| `01_data_preparation.ipynb` | Audit, EDA, accepted manifest, and loaders |
| `02_custom_cnn.ipynb` | Matched CNN baseline and augmentation runs |
| `03_resnet18_transfer.ipynb` | Frozen backbone and full fine-tuning |
| `04_evaluation_gradcam.ipynb` | Validation metrics and 40-case Grad-CAM review |
| `05_final_experiment_demo.ipynb` | Crop-control experiment, frozen test comparison, demo, report |

`completed/` contains the corresponding executed user notebooks with their recorded outputs and observations. Only publication metadata and interactive widget state/views were cleaned. Completed notebooks are evidence of historical runs; use the clean templates for a fresh run. The provenance registry records both original and published notebook hashes.

The historical Day 4 review used 13 user and 27 authorized AI-assisted visual reviews. Its saved notebook/log retains that attribution. The clean Day 4 template requests human observations and should be completed honestly according to the new run's actual reviewers.
