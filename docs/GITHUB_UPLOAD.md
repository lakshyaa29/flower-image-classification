# Upload this repository to GitHub

Use the repository name **flower-image-classification** and description:

> Five-class flower classification with PyTorch: custom CNN vs ResNet18, Grad-CAM, controlled augmentation experiments, and a Gradio demo.

Suggested topics: `pytorch`, `image-classification`, `cnn`, `resnet18`, `transfer-learning`, `grad-cam`, `gradio`.

## Publish using GitHub Desktop

1. Open GitHub Desktop and choose **File > Add local repository**. Select this `flower-image-classification` folder. If prompted, create a repository at this location without adding another README, license, or ignore file.
2. Review the included files and make the first commit with the message `Add completed flower classification project`.
3. Choose **Publish repository**, keep the name above, and choose the visibility you want.
4. Open the published repository and confirm the README, notebooks, images, and workflow appear.

The local repository is prepared without a GitHub remote or fabricated commit history. If it already has a first commit, proceed to publishing.

## Publish using Git

From a terminal in this folder, initialize Git if it has not already been initialized. Create an empty repository on GitHub, then use the exact remote URL GitHub provides:

```text
git init -b main
git add .
git commit -m "Add completed flower classification project"
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Set your own Git author name/email if Git asks. This project does not invent an author identity. A single honest initial commit is appropriate; the five-day work is evidenced by the executed notebooks and timestamped experiment records.

## Files and artifacts

Upload this project folder's contents, with `README.md` at the repository root. Keep your existing daily backup folders separate. The `.gitignore` excludes weights, raw data, environments, ZIPs, and generated outputs when publishing through Git/GitHub Desktop.

The completed `day5_backup.zip` is the separate model artifact required for inference. Provide it with your internship submission or attach it as a release asset later, then add its real download URL to the README. No download URL has been invented. A reader can also reproduce training from the clean notebooks.

GitHub's browser uploader does not apply `.gitignore` to files you manually select. If using browser upload, select only the prepared code/result files and keep any subsequently restored `outputs/` folder, model files, raw data, and virtual environment out of the upload. Git/GitHub Desktop handles the existing ignore rules automatically.
