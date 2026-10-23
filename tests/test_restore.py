"""Check artifact restoration boundaries, integrity, and repeatability."""

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.restore_artifacts import restore_backup, sha256_file


class RestoreTests(unittest.TestCase):
    def make_backup(self, path, extra=None):
        checkpoint = b"fixture checkpoint bytes"
        config = {"checkpoint": "outputs/day3/run/model.pt", "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest()}
        entries = {
            "outputs/day5_active.json": json.dumps({"experiment": "outputs/day5/run"}),
            "outputs/day5/run/demo_config.json": json.dumps(config),
            config["checkpoint"]: checkpoint,
            "data/processed/splits.csv": b"fixture manifest\n",
            "flower_demo.py": b"raise RuntimeError('must not run')",
            "outputs/day5/run/submission/app.py": b"raise RuntimeError('must not run')",
        }
        entries.update(extra or {})
        with zipfile.ZipFile(path, "w") as archive:
            for name, value in entries.items():
                archive.writestr(name, value)

    def test_restore_is_repeatable_and_skips_archived_code(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            base = Path(temporary)
            backup, root = base / "backup.zip", base / "project"
            root.mkdir()
            self.make_backup(backup)
            expected = sha256_file(backup)
            first = restore_backup(backup, root, expected)
            self.assertEqual(first["restored"], 4)
            self.assertTrue(first["config_path"].is_file())
            self.assertFalse((root / "flower_demo.py").exists())
            self.assertFalse((root / "outputs/day5/run/submission/app.py").exists())
            second = restore_backup(backup, root, expected)
            self.assertEqual(second["restored"], 0)
            self.assertEqual(second["reused"], 4)

    def test_traversal_is_rejected_before_any_write(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            base = Path(temporary)
            backup, root = base / "backup.zip", base / "project"
            root.mkdir()
            self.make_backup(backup, {"outputs/../../escape.txt": "unsafe"})
            with self.assertRaisesRegex(ValueError, "Unsafe archive path"):
                restore_backup(backup, root, sha256_file(backup))
            self.assertEqual(list(root.iterdir()), [])
            self.assertFalse((base / "escape.txt").exists())

    def test_wrong_archive_hash_rejected_before_any_write(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            base = Path(temporary)
            backup, root = base / "backup.zip", base / "project"
            root.mkdir()
            self.make_backup(backup)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                restore_backup(backup, root, "0" * 64)
            self.assertEqual(list(root.iterdir()), [])

    def test_existing_conflict_is_preserved_without_partial_restore(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            base = Path(temporary)
            backup, root = base / "backup.zip", base / "project"
            existing = root / "data/processed/splits.csv"
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"different manifest")
            self.make_backup(backup)
            with self.assertRaisesRegex(ValueError, "Existing artifact differs"):
                restore_backup(backup, root, sha256_file(backup))
            self.assertEqual(existing.read_bytes(), b"different manifest")
            self.assertFalse((root / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
