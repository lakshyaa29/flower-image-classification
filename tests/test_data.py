"""Small fixtures test leakage protection, corruption checks, and repeatability."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from flower_data import CLASS_NAMES, audit_images, prepare_project, stratified_split, validate_splits


class DataPreparationTests(unittest.TestCase):
    def make_dataset(self, root):
        for label, name in enumerate(CLASS_NAMES):
            folder = root / name
            folder.mkdir(parents=True)
            for index in range(20):
                Image.new("RGB", (32 + index, 30), (label * 40, index * 10, 90)).save(folder / f"{index:02}.png")

    def test_audit_excludes_duplicates_corruption_and_conflicting_labels(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            self.make_dataset(root)
            original = root / "daisy" / "00.png"
            (root / "daisy" / "copy.png").write_bytes(original.read_bytes())
            (root / "roses" / "conflict.png").write_bytes((root / "tulips" / "01.png").read_bytes())
            (root / "daisy" / "broken.jpg").write_bytes(b"not an image")
            rows, audit = audit_images(root)
            self.assertEqual(len(audit["excluded_duplicates"]), 1)
            self.assertEqual(len(audit["corrupt_images"]), 1)
            self.assertEqual(len(audit["conflicting_label_groups"]), 1)
            self.assertEqual(len(rows), 99)

    def test_splits_are_reproducible_stratified_and_disjoint(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            self.make_dataset(root)
            rows, _ = audit_images(root)
            splits = stratified_split(rows, seed=42)
            self.assertEqual(splits, stratified_split(list(reversed(rows)), seed=42))
            self.assertNotEqual(splits, stratified_split(rows, seed=43))
            validate_splits(splits)
            for name in CLASS_NAMES:
                group = [row for row in splits if row["class_name"] == name]
                self.assertEqual([sum(row["split"] == split for row in group) for split in ("train", "val", "test")], [14, 3, 3])
            duplicate = dict(splits[0], relative_path="daisy/different_name.png", split="test")
            with self.assertRaises(ValueError):
                validate_splits(splits + [duplicate])

    def test_prepare_project_writes_portable_manifest_and_real_reports(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            self.make_dataset(root / "fixture")
            summary = prepare_project(root / "project", seed=42, dataset_dir=root / "fixture")
            self.assertEqual(summary["split_counts"], {"train": 70, "val": 15, "test": 15})
            self.assertIsNone(summary["source_archive_sha256"])
            self.assertTrue((root / "project" / "outputs" / "day1" / "sample_grid.jpg").is_file())
            manifest = (root / "project" / "data" / "processed" / "splits.csv").read_text()
            self.assertNotIn(str(root), manifest)


if __name__ == "__main__":
    unittest.main()
