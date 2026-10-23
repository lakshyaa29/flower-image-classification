"""Launch the flower classifier locally, using the frozen validation selection."""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, help="Restore the completed day5_backup.zip before starting.")
    parser.add_argument("--config", type=Path, help="Use an existing demo_config.json.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true", help="Predict the included training sample and exit.")
    args = parser.parse_args()
    config = args.config or ROOT / "results/provenance/demo_config.json"
    if args.backup:
        from tools.restore_artifacts import restore_backup

        restored = restore_backup(args.backup, ROOT)
        config = restored["config_path"] if args.config is None else args.config
        print(f"Restored {restored['restored']} artifact files; reused {restored['reused']} unchanged files.")
    if not config.is_file():
        parser.error("The demo configuration is missing. Supply --backup with the completed Day 5 ZIP.")
    metadata = json.loads(config.read_text(encoding="utf-8"))
    if not (ROOT / metadata["checkpoint"]).is_file():
        parser.error("Trained weights are stored in the separate Day 5 backup. Run with --backup path/to/day5_backup.zip.")
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    from flower_demo import FlowerPredictor

    predictor = FlowerPredictor(ROOT, config)
    if args.check:
        from PIL import Image

        with Image.open(ROOT / "assets/sample_daisy.jpg") as image:
            scores, overlay = predictor.predict(image)
        output = ROOT / "outputs/demo_check_heatmap.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(output)
        print(json.dumps({"model": metadata["run"], "scores": scores, "heatmap": str(output)}, indent=2))
        return
    from flower_demo import create_gradio_demo

    demo = create_gradio_demo(predictor)
    demo.launch(share=False, server_name="127.0.0.1", server_port=args.port,
                inbrowser=not args.no_browser, max_file_size="10mb")


if __name__ == "__main__":
    main()
