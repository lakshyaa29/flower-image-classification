"""Guard the crop intervention, selection rule, split checks, and immutable plans."""

import copy
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flower_data import CLASS_NAMES, MANIFEST_FIELDS, write_json
from flower_pipeline import make_loaders
from flower_evaluation import predict_split
from flower_final import freeze_final_selection, select_validation_winner


class FinalDayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_selection_uses_validation_and_stable_loss_ties(self):
        rows = [{"run":"old", "val_macro_f1":.9, "val_loss":.2, "test_accuracy":.1},
                {"run":"crop", "val_macro_f1":.8, "val_loss":.1, "test_accuracy":1.0}]
        self.assertEqual(select_validation_winner(rows), "old")
        rows[1]["val_macro_f1"] = .9
        self.assertEqual(select_validation_winner(rows), "crop")
        rows[1]["val_loss"] = .2
        self.assertEqual(select_validation_winner(rows), "old")

    def test_crop_scale_changes_only_training_transform(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            raw = root / "data/raw/flower_photos"
            rows = []
            for label, name in enumerate(CLASS_NAMES):
                (raw/name).mkdir(parents=True)
                for split in ("train", "val", "test"):
                    relative = f"{name}/{split}.png"
                    Image.new("RGB", (260,280), (label*40,70,90)).save(raw/relative)
                    digest = hashlib.sha256(relative.encode()).hexdigest()
                    rows.append({"relative_path":relative,"class_name":name,"label":label,"width":260,"height":280,
                                 "file_sha256":digest,"pixel_sha256":digest,"split":split})
            output = io.StringIO(newline="")
            writer = csv.DictWriter(output,fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            manifest = output.getvalue().encode()
            (root/"data/processed").mkdir(parents=True)
            (root/"data/processed/splits.csv").write_bytes(manifest)
            write_json(root/"outputs/day1/data_summary.json",{"manifest_sha256":hashlib.sha256(manifest).hexdigest()})
            control,_ = make_loaders(root,augment=True,crop_scale=(.75,1.0))
            crop,_ = make_loaders(root,augment=True,crop_scale=(.5,1.0))
            self.assertEqual(control["train"].dataset.transform.transforms[0].scale,(.75,1.0))
            self.assertEqual(crop["train"].dataset.transform.transforms[0].scale,(.5,1.0))
            for split in ("val","test"):
                self.assertEqual(repr(control[split].dataset.transform),repr(crop[split].dataset.transform))
                self.assertTrue(torch.equal(control[split].dataset[0][0],crop[split].dataset[0][0]))

    def test_split_mismatch_rejected_before_recording_predictions(self):
        data = TensorDataset(torch.tensor([[4.,0.,0.,0.,0.]]),torch.tensor([0]))
        data.rows = [{"label":0,"split":"val","relative_path":"daisy/example.png"}]
        with self.assertRaisesRegex(ValueError,"split is inconsistent"):
            predict_split(nn.Identity(),DataLoader(data,batch_size=1),torch.device("cpu"),split="test")

    def test_frozen_plan_rejects_changed_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root=Path(temp)
            directory=root/"outputs/day5/fixture"
            directory.mkdir(parents=True)
            checkpoint=root/"outputs/weights.pt"
            checkpoint.write_bytes(b"fixture weights")
            comparison={"mode":"full","selected_by_validation":"a","selection_is_final":True,
                        "runs":[{"run":"a","checkpoint":"outputs/weights.pt","val_macro_f1":.9,"val_loss":.2}]}
            write_json(directory/"comparison.json",comparison)
            with patch("flower_final.validate_day1",return_value={"manifest_sha256":"fixture"}):
                plan=freeze_final_selection(root,directory)
                self.assertEqual(plan,freeze_final_selection(root,directory))
                checkpoint.write_bytes(b"changed fixture weights")
                with self.assertRaisesRegex(ValueError,"changed"):
                    freeze_final_selection(root,directory)


if __name__ == "__main__":
    unittest.main()
