"""Independent checks for reports, Grad-CAM, case sampling, and review records."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flower_data import write_json
from flower_evaluation import (classification_report_from_matrix, compute_gradcam, review_progress,
                               save_case_review, select_review_cases)


class TinyCAMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        self.fc = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1)
            self.fc.weight.copy_(torch.tensor([[1.], [-1.]]))
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, image):
        return self.fc(self.conv(image).mean(dim=(2, 3)))


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_classification_report_matches_hand_calculated_values(self):
        report = classification_report_from_matrix([[3, 1], [2, 4]], ["a", "b"])
        self.assertAlmostEqual(report["accuracy"], 0.7)
        self.assertAlmostEqual(report["a"]["precision"], 0.6)
        self.assertAlmostEqual(report["a"]["recall"], 0.75)
        self.assertAlmostEqual(report["macro avg"]["f1-score"], 23 / 33)
        self.assertAlmostEqual(report["weighted avg"]["f1-score"], 232 / 330)
        empty = classification_report_from_matrix([[0, 0], [0, 0]], ["a", "b"])
        self.assertEqual(empty["macro avg"]["f1-score"], 0)

    def test_gradcam_known_positive_map_and_zero_negative_map_with_frozen_model(self):
        model = TinyCAMModel().train()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        image = torch.tensor([[[[0., 1.], [2., 3.]]]])
        with torch.no_grad():
            maps, flat, logits = compute_gradcam(model, image, model.conv, [0, 1])
        np.testing.assert_allclose(maps[0], [[0, 1/3], [2/3, 1]], atol=1e-6)
        np.testing.assert_array_equal(maps[1], np.zeros((2, 2)))
        self.assertFalse(flat[0])
        self.assertTrue(flat[1])
        self.assertTrue(model.training)
        self.assertEqual(len(model.conv._forward_hooks), 0)
        self.assertTrue(all(torch.equal(before[name], value) for name, value in model.state_dict().items()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_gradcam_removes_hook_after_invalid_target(self):
        model = TinyCAMModel()
        with self.assertRaises(ValueError):
            compute_gradcam(model, torch.ones(1, 1, 2, 2), model.conv, [3])
        self.assertEqual(len(model.conv._forward_hooks), 0)

    def test_review_sampling_is_reproducible_balanced_and_has_no_duplicate_cases(self):
        records = []
        for label in range(5):
            for correct in (True, False):
                for i in range(6):
                    records.append({"index": len(records), "true_label": label, "correct": correct,
                                    "relative_path": f"{label}/{correct}_{i}.jpg", "confidence": 0.5 + i / 20})
        first = select_review_cases(records, 20, seed=42)
        self.assertEqual(first, select_review_cases(list(reversed(records)), 20, seed=42))
        self.assertEqual(sum(row["correct"] for row in first), 20)
        self.assertEqual(len(first), 40)
        self.assertEqual(len({row["index"] for row in first}), 40)
        self.assertEqual({row["true_label"] for row in first if not row["correct"]}, set(range(5)))

    def test_automated_entries_do_not_count_as_human_reviews(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            folder = Path(temp)
            write_json(folder / "review_cases.json", [{"case_id": "val_0001", "correct": False, "reviewed": False, "reviewer": ""}])
            save_case_review(folder, "val_0001", "unclear", "unclear_other", reviewer="automated_verification")
            self.assertEqual(review_progress(folder)["reviewed"], 0)
            save_case_review(folder, "val_0001", "mixed", "class_confusion", "Fixture review", reviewer="human")
            self.assertEqual(review_progress(folder)["incorrect_reviewed"], 1)


if __name__ == "__main__":
    unittest.main()
