"""Generate a three-page factual report and a timed presentation guide from saved results."""

import json
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

from flower_final import load_day4_results


def build_deliverables(root, directory):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from pypdf import PdfReader

    root, directory = Path(root).resolve(), Path(directory).resolve()
    read = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))
    comparison, final = read("comparison.json"), read("final_results.json")
    selection = read("selection.json")
    if final["plan_sha256"] != selection["plan_sha256"]:
        raise ValueError("Report inputs do not match the frozen model selection.")
    smoke = comparison["mode"] == "smoke"
    day4_dir, day4, observations, _ = load_day4_results(root)
    cases = json.loads((day4_dir / "review_cases.json").read_text())
    selected = comparison["selected_by_validation"]
    selected_test = next(row for row in final["runs"] if row["run"] == selected)
    selected_val = next(row for row in comparison["runs"] if row["run"] == selected)
    test_by_name = {row["run"]: row for row in final["runs"]}
    control, crop = (next(row for row in comparison["runs"] if row["run"] == name) for name in ("resnet18_control", "resnet18_crop"))
    delta = comparison["crop_macro_f1_change_vs_control"]
    outcome = "higher" if delta > 0 else "lower" if delta < 0 else "equal"
    fix_text = (f"The crop arm's validation macro-F1 was {crop['val_macro_f1']:.4f}, versus {control['val_macro_f1']:.4f} "
                f"for matched continued training. It was {outcome} by {abs(delta):.4f}. The final validation-selected model is {selected}.")
    scope = "SMOKE CHECK ONLY - fixture scores are not project results" if smoke else "Final report - frozen selection and held-out test evaluation"
    reviewers = dict(Counter(case["reviewer"] for case in cases))
    error_text = ("Day 4's largest validation confusions were dandelion to daisy (7), roses to tulips (7), and tulips to roses (5). "
        "Roses had the lowest per-class F1. The 40-case diagnostic review covered 20 correct and 20 incorrect predictions. "
        f"Review attribution: {reviewers.get('human', 0)} user and {reviewers.get('codex_visual', 0)} authorized AI-assisted visual reviews. "
        "Error tags included six similar-class cases, five clutter cases, and three small/distant subjects. "
        "Heatmaps sometimes highlighted surrounding objects. These observations motivate a hypothesis, not a causal conclusion.")
    limits = ("One seed and one split limit generalization claims. ResNet18 benefits from ImageNet pretraining and greater capacity. "
        "The head-only run had six epochs, original fine-tuning had 12 allocated epochs including head training, and each Day 5 arm "
        f"adds {control['stage_epochs']} epochs from the same selected fine-tuned checkpoint. Budgets differ across the six models. "
        "The review sample favors high-confidence errors and is not representative of all failures. Exact pixel duplicate checks do not "
        "eliminate near duplicates or every label problem. Grad-CAM is approximate. Scores are uncalibrated. Smaller training crops can exclude flowers.")
    methods = ("Flower Photos contains five folder labels: daisy, dandelion, roses, sunflowers, and tulips. "
        "From 3,670 original images, the audit excluded two redundant sunflower copies and both copies of one conflicting-label image. "
        "The 3,666 accepted images use fixed, stratified seed-42 splits: 2,564 train, 551 validation, and 551 test. "
        "All model selection uses validation macro-F1, with lower validation loss as the tie-breaker. The frozen model list and hashes precede test inference.")
    model_methods = ("The custom CNN has three convolution/BatchNorm/ReLU/pooling blocks and a five-class classifier. "
        "Its baseline and augmented runs each used 12 epochs from the same initial weights. ResNet18 first trained a five-class head "
        "with its backbone and BatchNorm statistics frozen for six epochs, then fine-tuned all layers for six epochs. "
        "Day 5 compares two matched continuations from the same best original fine-tuned weights. Both use AdamW, learning rate 0.00001, "
        "weight decay 0.0001, cosine scheduling, and the same seed/budget. Only training crop area changes: (0.75, 1.0) versus (0.5, 1.0).")
    transforms = ("Inputs use RGB/EXIF correction and 224-pixel model inputs. Training augmentation includes random crop, horizontal flip, "
        "15-degree rotation, and mild color jitter. Validation, test, and demo inputs use Resize(256), CenterCrop(224), and ImageNet normalization. "
        "Test scores do not change the selected model. Per-image predictions, per-class reports, confusion matrices, histories, and runtime versions accompany this report.")
    result_table = [["Model", "Val acc.", "Val F1", "Test acc.", "Test F1", "Epoch budget"]]
    for row in comparison["runs"]:
        test = test_by_name[row["run"]]
        result_table.append([row["run"].replace("resnet18_", "ResNet ").replace("cnn_", "CNN "),
            f"{row['val_accuracy']:.2%}", f"{row['val_macro_f1']:.4f}", f"{test['test_accuracy']:.2%}",
            f"{test['test_macro_f1']:.4f}", str(row["allocated_total_epochs"])])
    test_text = (f"Selected model: {selected}. Validation accuracy {selected_val['val_accuracy']:.2%}, macro-F1 {selected_val['val_macro_f1']:.4f}. "
        f"Final {'fixture' if smoke else 'test'} accuracy {selected_test['test_accuracy']:.2%}, macro-F1 {selected_test['test_macro_f1']:.4f}, "
        f"on {selected_test['samples']} {'fixture' if smoke else 'test'} images. The selected model remained fixed after testing.")
    example = next((case for case in cases if not case["correct"] and case["pattern"] == "small_subject"), next(case for case in cases if not case["correct"]))
    sources = ("Sources: TensorFlow Flower Photos image-loading tutorial (tensorflow.org/tutorials/load_data/images); "
        "PyTorch transfer-learning tutorial (docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html); "
        "Selvaraju et al., Grad-CAM, ICCV 2017. Individual image credits are preserved in docs/DATASET_IMAGE_CREDITS.txt in the backup.")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyReport", fontName="Helvetica", fontSize=10, leading=13.5, spaceAfter=8))
    styles.add(ParagraphStyle(name="CaptionReport", fontName="Helvetica", fontSize=8, leading=10, spaceAfter=7))
    styles["Title"].fontName = "Helvetica-Bold"
    styles["Title"].fontSize = 23
    styles["Title"].leading = 28
    styles["Title"].alignment = TA_LEFT
    styles["Heading2"].fontSize = 14
    styles["Heading2"].spaceBefore = 10
    story = []
    def paragraph(text, style="BodyReport"):
        story.append(Paragraph(escape(text), styles[style]))
    def heading(text):
        paragraph(text, "Heading2")
    def table(data, widths):
        wrapped = [[Paragraph(escape(str(value)), styles["CaptionReport"]) for value in row] for row in data]
        item = Table(wrapped, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0),colors.HexColor("#e7eef8")),
            ("VALIGN", (0,0),(-1,-1),"TOP"), ("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("TOPPADDING",(0,0),(-1,-1),5), ("LINEBELOW",(0,0),(-1,0),.6,colors.HexColor("#7890aa"))]))
        story.append(item)
        story.append(Spacer(1,8))
    paragraph("Flower Image Classification", "Title")
    paragraph(scope, "CaptionReport")
    heading("1. Dataset and experimental methods")
    paragraph(methods)
    table([["Class", "Train", "Validation", "Test"], ["daisy",443,95,95], ["dandelion",628,135,135],
           ["roses",448,96,96], ["sunflowers",487,105,105], ["tulips",558,120,120], ["Total",2564,551,551]], [210,85,95,85])
    paragraph(model_methods)
    paragraph(transforms)
    paragraph("Split manifest SHA-256: " + selection["manifest_sha256"], "CaptionReport")
    story.append(PageBreak())
    heading("2. Results and the controlled crop experiment")
    if smoke:
        paragraph(scope, "CaptionReport")
    table(result_table, [139,66,66,70,66,68])
    paragraph(fix_text)
    paragraph(test_text)
    story.append(Image(str(directory / "test" / selected / "confusion_matrix.png"), width=475, height=198))
    paragraph("Selected model confusion matrix. Rows are true classes; columns are predictions. The right panel normalizes within each true class.", "CaptionReport")
    paragraph("The four earlier models retain their original validation-selected checkpoints. Day 5's matched control isolates additional training from the crop intervention. Negative or unchanged results are reported without claiming improvement.")
    story.append(PageBreak())
    heading("3. Error analysis, demo, and limitations")
    paragraph(error_text)
    story.append(Image(str(day4_dir / example["panel"]), width=475, height=206.4))
    paragraph(f"Validation example {example['case_id']}: a reviewed small/distant-subject error. Panels show the exact input crop, predicted-class Grad-CAM, and true-class Grad-CAM. Heatmaps normalize independently.", "CaptionReport")
    paragraph(limits)
    paragraph("The upload demo uses the frozen validation-selected checkpoint and the same evaluation preprocessing. It returns five class scores and a predicted-class heatmap. A private notebook form and a localhost Gradio app are included. The classifier always assigns one of the five labels, even to unfamiliar images.")
    paragraph(sources, "CaptionReport")
    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#526177"))
        canvas.drawString(20*mm, 12*mm, "SMOKE VERIFICATION - NOT FINAL RESULTS" if smoke else "Flower CNN internship project")
        canvas.drawRightString(A4[0]-20*mm,12*mm,f"{document.page} / 3")
        canvas.restoreState()
    path = directory / "final_report.pdf"
    SimpleDocTemplate(str(path), pagesize=A4, rightMargin=20*mm, leftMargin=20*mm,
                      topMargin=17*mm, bottomMargin=20*mm, title="Flower classification final report").build(story, onFirstPage=footer, onLaterPages=footer)
    if len(PdfReader(str(path)).pages) != 3:
        raise ValueError("The report must fit exactly three pages; inspect the rendered layout before final submission.")
    md_table = "\n".join(["| " + " | ".join(map(str,row)) + " |" for row in result_table[:1]] +
        ["|---|---:|---:|---:|---:|---:|"] + ["| " + " | ".join(map(str,row)) + " |" for row in result_table[1:]])
    (directory / "final_report.md").write_text(f"# Flower Image Classification\n\n{scope}\n\n## Dataset and methods\n\n{methods}\n\n{model_methods}\n\n{transforms}\n\n## Results\n\n{md_table}\n\n{fix_text}\n\n{test_text}\n\n## Error analysis and limitations\n\n{error_text}\n\n{limits}\n\n{sources}\n", encoding="utf-8")
    notes = f"""# 15-minute presentation and live demo

{scope}

## 0:00-1:00 - Project question
Explain the five flower classes and the goal: compare a CNN trained from scratch with pretrained ResNet18. State that checkpoint/model selection used validation, then test results were opened after freezing the decision.

## 1:00-3:00 - Dataset and audit
Show the Day 1 image grid and class counts. Explain 3,670 original images, four exclusions, and 2,564/551/551 fixed splits. Explain exact duplicate and conflicting-label exclusions, and the remaining near-duplicate/label limitations.

## 3:00-5:00 - Custom CNN experiments
Describe the three convolution blocks. Compare baseline and augmentation under the same 12-epoch budget and starting weights. Show the saved curves rather than claiming augmentation must always help.

## 5:00-7:00 - Transfer learning
Explain ImageNet features, the five-class replacement head, fixed backbone BatchNorm in the head stage, and lower-rate full fine-tuning. Show the four original validation results. Acknowledge capacity, pretraining, and budget differences.

## 7:00-9:00 - Errors and Grad-CAM
{error_text}
Show the validation example in the report and explain that a hotspot is an approximate sensitivity signal. Disclose the user/AI review attribution.

## 9:00-11:00 - One controlled improvement attempt
Both continuations start from the same checkpoint, use the same six-epoch default budget and learning rate, and differ only in crop scale. {fix_text}
State the observed result even if the crop arm loses. One seed does not establish a reliable effect. Smaller crops can remove the subject.

## 11:00-13:00 - Final test results
{test_text}
Show final_test_comparison.csv and the selected model confusion matrix. Explain that the test ranking did not change model selection.

## 13:00-15:00 - Live upload demo
Open the private notebook upload form or the localhost Gradio app. Upload one flower photo, explain all five scores, and show the predicted-class heatmap. Try another viewpoint. Explain that scores are uncalibrated and unfamiliar inputs still receive a known label. Finish with the key limitation and one future improvement.

## 10-minute Q&A preparation
- Why macro-F1? It weights the five classes equally and shows weaknesses hidden by accuracy.
- Why a matched control? Extra training could improve the model even without a crop change.
- Why keep the original model if it wins? Validation decides; an attempted fix need not improve results.
- Why freeze before test? Test feedback should not guide checkpoint or hyperparameter selection.
- Does Grad-CAM prove background bias? No. It suggests a region to investigate with controlled tests.
- What remains uncertain? {limits}

## Files to show
Use the three-page final_report.pdf, saved learning curves, selected confusion matrix, Day 4 gallery, and upload demo. Keep the full backup and image credits available.
"""
    (directory / "presentation_notes.md").write_text(notes, encoding="utf-8")
    return path
