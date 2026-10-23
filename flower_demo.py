"""Upload inference with the validation-selected checkpoint and optional Gradio UI."""

import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from flower_data import CLASS_NAMES, sha256_file, write_json
from flower_pipeline import IMAGENET_MEAN, IMAGENET_STD
from flower_evaluation import compute_gradcam, load_model, target_layer_for


class FlowerPredictor:
    def __init__(self, root, config_path):
        self.root = Path(root).resolve()
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        checkpoint = (self.root / self.config["checkpoint"]).resolve()
        if not checkpoint.is_relative_to(self.root) or sha256_file(checkpoint) != self.config["checkpoint_sha256"]:
            raise ValueError("The demo checkpoint path/hash does not match the frozen selection.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = load_model(checkpoint, self.device)
        self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def predict(self, image):
        if not isinstance(image, Image.Image):
            raise ValueError("Upload a valid image first.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = self.transform(image)
        with torch.inference_mode():
            probabilities = self.model(tensor.unsqueeze(0).to(self.device)).softmax(1)[0].cpu()
        chosen = int(probabilities.argmax())
        maps, _, _ = compute_gradcam(self.model, tensor.unsqueeze(0).to(self.device), target_layer_for(self.model), [chosen])
        rgb = (tensor * torch.tensor(IMAGENET_STD).view(3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(3, 1, 1)).clamp(0, 1).permute(1, 2, 0).numpy()
        import matplotlib
        color = matplotlib.colormaps["inferno"](maps[chosen])[..., :3]
        overlay = np.uint8(np.round((.6 * rgb + .4 * color) * 255))
        return {name: float(probabilities[index]) for index, name in enumerate(CLASS_NAMES)}, overlay


def create_gradio_demo(predictor):
    import gradio as gr
    return gr.Interface(fn=predictor.predict, inputs=gr.Image(type="pil", label="Upload a flower image"),
        outputs=[gr.Label(num_top_classes=5, label="Class scores"), gr.Image(label="Predicted-class Grad-CAM on the model input crop")],
        title="Flower classifier", description="Five classes: daisy, dandelion, roses, sunflowers, tulips. "
        "Scores are uncalibrated. The model always chooses a known class, including for unfamiliar images. "
        "Grad-CAM is approximate and does not identify object boundaries.",
        flagging_mode="never", analytics_enabled=False)


def launch_private_upload(predictor):
    import ipywidgets as widgets
    from IPython.display import display, Markdown
    uploader = widgets.FileUpload(accept="image/*", multiple=False, description="Upload image")
    output = widgets.Output()
    def uploaded(change):
        if not uploader.value:
            return
        entry = next(iter(uploader.value.values())) if isinstance(uploader.value, dict) else uploader.value[0]
        with output:
            output.clear_output(wait=True)
            try:
                with Image.open(io.BytesIO(bytes(entry["content"]))) as source:
                    scores, overlay = predictor.predict(source)
                ordered = sorted(scores.items(), key=lambda item: -item[1])
                display(Markdown("| Class | Score |\n|---|---:|\n" + "\n".join(f"| {name} | {score:.2%} |" for name, score in ordered)))
                display(Image.fromarray(overlay))
                print("Uncalibrated scores. Heatmap uses the exact model input crop.")
            except Exception as error:
                print(f"Could not read/predict this image: {error}")
    uploader.observe(uploaded, names="value")
    form = widgets.VBox([widgets.HTML("<b>Private notebook upload demo</b><br>Choose one image to see all five scores and a heatmap."), uploader, output])
    display(form)
    return form


def export_demo_files(root, directory):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    plan = json.loads((directory / "selection.json").read_text())
    selected = next(row for row in plan["candidates"] if row["run"] == plan["selected_by_validation"])
    write_json(directory / "demo_config.json", {**selected, "class_names": list(CLASS_NAMES),
        "preprocessing": plan["preprocessing"], "selected_before_test": True, "mode": plan["mode"]})
    submission = directory / "submission"
    submission.mkdir(exist_ok=True)
    source = '''"""Run from the extracted project: python outputs/day5/<run>/submission/app.py"""
import os
import sys
from pathlib import Path
directory = Path(__file__).resolve().parent.parent
root = next(parent for parent in directory.parents if (parent / "flower_demo.py").is_file())
sys.path.insert(0, str(root))
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
from flower_demo import FlowerPredictor, create_gradio_demo
if __name__ == "__main__":
    predictor = FlowerPredictor(root, directory / "demo_config.json")
    demo = create_gradio_demo(predictor)
    demo.launch(share=False, server_name="127.0.0.1", inbrowser=True, max_file_size="10mb")
'''
    (submission / "app.py").write_text(source, encoding="utf-8")
    (submission / "README.md").write_text(
        "# Flower upload demo\n\nExtract the entire Day 5 ZIP, keep its folder structure, install the root requirements, "
        "and run this app.py with Python. The app reads demo_config.json and the original selected checkpoint in the full archive. "
        "It starts on localhost without public sharing. The notebook also has a private ipywidgets upload form. "
        "Gradio in Colab requires a temporary public share link; the notebook makes that opt-in.\n", encoding="utf-8")
    return directory / "demo_config.json"
