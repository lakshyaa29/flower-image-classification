"""A small notebook form for recording the actual Grad-CAM case review."""

import html
import json
from pathlib import Path

from flower_evaluation import review_progress, save_case_review


def launch_review_form(directory):
    import ipywidgets as widgets
    from IPython.display import display

    directory = Path(directory)
    cases = json.loads((directory / "review_cases.json").read_text(encoding="utf-8"))
    if not cases:
        raise ValueError("No review cases were generated.")
    incomplete = [i for i, case in enumerate(cases) if not (case.get("reviewed") and case.get("reviewer") == "human")]
    state = {"index": incomplete[0] if incomplete else 0}
    progress_label = widgets.HTML()
    caption = widgets.HTML()
    panel = widgets.Image(format="png", layout=widgets.Layout(width="100%", max_width="900px"))
    attention = widgets.Dropdown(description="Heatmap:", options=[("Choose...", None), ("Mainly flower", "flower"),
                                 ("Mainly background", "background"), ("Flower and background", "mixed"), ("Unclear / flat", "unclear")])
    pattern = widgets.Dropdown(description="Pattern:", options=[("Choose...", None)], layout=widgets.Layout(width="95%"))
    comment = widgets.Textarea(description="Note:", placeholder="Optional: describe the visible evidence or uncertainty.",
                               layout=widgets.Layout(width="95%", height="80px"))
    previous = widgets.Button(description="Previous")
    save_next = widgets.Button(description="Save and next", button_style="primary")
    message = widgets.HTML()

    def render():
        current = json.loads((directory / "review_cases.json").read_text(encoding="utf-8"))
        case = current[state["index"]]
        progress = review_progress(directory)
        progress_label.value = (f"<b>Reviewed {progress['reviewed']}/{progress['total']}</b> "
                                f"({progress['correct_reviewed']} correct, {progress['incorrect_reviewed']} incorrect). "
                                f"Showing case {state['index']+1}/{len(current)}.")
        caption.value = (f"<b>{html.escape(case['case_id'])}</b> — "
                         f"{'CORRECT' if case['correct'] else 'INCORRECT'} prediction. "
                         f"<br>{html.escape(case['relative_path'])}")
        panel.value = (directory / case["panel"]).read_bytes()
        attention.value = case.get("attention") or None
        choices = [("Choose...", None)]
        if case["correct"]:
            choices.append(("Correct prediction; no obvious issue", "correct_prediction"))
        choices.extend([("Similar-looking classes", "class_confusion"), ("Small or distant flower", "small_subject"),
                        ("Blur, lighting, or edited colour", "image_quality"), ("Multiple flowers or clutter", "clutter"),
                        ("Possible label ambiguity", "label_ambiguity"), ("Possible background reliance", "background_candidate"),
                        ("Unclear / other", "unclear_other")])
        pattern.options = choices
        valid_patterns = {value for _, value in choices}
        pattern.value = case.get("pattern") if case.get("pattern") in valid_patterns else None
        comment.value = case.get("comment", "")
        previous.disabled = state["index"] == 0

    def save_clicked(button):
        if attention.value is None or pattern.value is None:
            message.value = "<span style='color:#b42318'>Choose a heatmap category and an observed pattern before saving.</span>"
            return
        current = json.loads((directory / "review_cases.json").read_text(encoding="utf-8"))
        progress = save_case_review(directory, current[state["index"]]["case_id"], attention.value, pattern.value, comment.value)
        if progress["remaining"]:
            current = json.loads((directory / "review_cases.json").read_text(encoding="utf-8"))
            later = [i for i in range(state["index"] + 1, len(current)) if not (current[i].get("reviewed") and current[i].get("reviewer") == "human")]
            remaining = [i for i, case in enumerate(current) if not (case.get("reviewed") and case.get("reviewer") == "human")]
            state["index"] = later[0] if later else remaining[0]
            message.value = "Saved. Review the next case."
        else:
            message.value = "<b>All cases are reviewed. Continue to the observations cell, then make your backup.</b>"
        render()

    def previous_clicked(button):
        state["index"] = max(0, state["index"] - 1)
        message.value = "Previous case shown. Unsaved form changes were not recorded."
        render()

    save_next.on_click(save_clicked)
    previous.on_click(previous_clicked)
    render()
    explanation = widgets.HTML("Look at the input crop, predicted-class heatmap, and true-class heatmap. "
                               "Heatmaps suggest class sensitivity; they do not prove background bias or identify object boundaries. "
                               "Softmax scores are uncalibrated. Choose the categories that match what you can see.")
    form = widgets.VBox([progress_label, caption, panel, explanation, attention, pattern, comment,
                         widgets.HBox([previous, save_next]), message])
    display(form)
    return form
