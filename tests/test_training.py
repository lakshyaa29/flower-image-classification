"""Checks for metrics, evaluation behavior, and safe Day 1 restoration."""

import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flower_data import CLASS_NAMES, MANIFEST_FIELDS
from flower_training import SmallCNN, metrics_from_confusion, restore_day1_backup, run_epoch


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_metrics_match_hand_calculated_example_and_missing_class(self):
        confusion = torch.tensor([[2, 1, 0], [0, 1, 1], [1, 0, 0]])
        metrics = metrics_from_confusion(confusion)
        self.assertEqual(metrics["samples"], 6)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], 7 / 18)
        self.assertEqual(metrics["per_class_f1"][2], 0.0)

    def test_loss_is_weighted_by_images_in_an_uneven_final_batch(self):
        logits = torch.tensor([[4., 0, 0, 0, 0], [0., 4, 0, 0, 0], [0., 0, 4, 0, 0],
                               [0., 0, 0, 4, 0], [4., 0, 0, 0, 0]])
        labels = torch.tensor([0, 1, 2, 3, 4])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        result = run_epoch(nn.Identity(), loader, torch.device("cpu"), log_every=0)
        self.assertAlmostEqual(result["loss"], nn.CrossEntropyLoss()(logits, labels).item(), places=6)
        self.assertEqual(result["samples"], 5)

    def test_evaluation_preserves_batchnorm_state_and_parameters(self):
        torch.manual_seed(42)
        model = SmallCNN()
        loader = DataLoader(TensorDataset(torch.randn(5, 3, 32, 32), torch.tensor([0, 1, 2, 3, 4])), batch_size=2)
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        result = run_epoch(model, loader, torch.device("cpu"), log_every=0)
        self.assertEqual(result["samples"], 5)
        self.assertFalse(model.training)
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[name]), name)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def make_backup(self, path, extra=None):
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for label, name in enumerate(CLASS_NAMES):
            for split in ("train", "val", "test"):
                identifier = f"{name}/{split}.jpg"
                digest = hashlib.sha256(identifier.encode()).hexdigest()
                writer.writerow({"relative_path": identifier, "class_name": name, "label": label,
                                 "width": 32, "height": 32, "file_sha256": digest, "pixel_sha256": digest, "split": split})
        manifest = buffer.getvalue().encode()
        digest = hashlib.sha256(manifest).hexdigest()
        summary = {"manifest_sha256": digest, "split_counts": {"train": 5, "val": 5, "test": 5}}
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("data/processed/splits.csv", manifest)
            archive.writestr("data/processed/class_to_idx.json", json.dumps(dict(zip(CLASS_NAMES, range(5)))))
            archive.writestr("outputs/day1/data_summary.json", json.dumps(summary))
            archive.writestr("flower_training.py", "raise RuntimeError('archive code must never execute')")
            if extra:
                archive.writestr(*extra)
        return digest

    def test_restore_verifies_manifest_and_does_not_restore_executable_code(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            folder = Path(temp)
            backup = folder / "backup.zip"
            digest = self.make_backup(backup)
            with patch("flower_training.EXPECTED_DAY1_MANIFEST_SHA256", digest):
                result = restore_day1_backup(backup, folder / "project")
            self.assertEqual(result["split_counts"], {"train": 5, "val": 5, "test": 5})
            self.assertFalse((folder / "project/flower_training.py").exists())
            with self.assertRaises(ValueError):
                restore_day1_backup(backup, folder / "wrong_hash")

    def test_restore_rejects_path_traversal_before_writing(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            folder = Path(temp)
            backup = folder / "backup.zip"
            digest = self.make_backup(backup, ("outputs/day1/../../bad.md", "unexpected write"))
            with patch("flower_training.EXPECTED_DAY1_MANIFEST_SHA256", digest):
                with self.assertRaises(ValueError):
                    restore_day1_backup(backup, folder / "project")
            self.assertFalse((folder / "project").exists())


if __name__ == "__main__":
    unittest.main()
