"""Restore verified model artifacts without importing code from a backup archive."""

import argparse
import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_FILE_BYTES = 128 * 1024 * 1024


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member(root, name):
    relative = PurePosixPath(name)
    if not name or "\\" in name or ":" in name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive path: {name}")
    destination = (root / Path(*relative.parts)).resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"Archive destination escapes the project: {name}")
    return destination


def restore_backup(backup, root=ROOT, expected_sha256=None):
    """Restore data manifests and outputs; preserve existing files or reject conflicts."""
    backup, root = Path(backup).resolve(), Path(root).resolve()
    if expected_sha256 is None:
        registry = json.loads((root / "results/provenance/artifacts.json").read_text(encoding="utf-8"))
        expected_sha256 = registry["backup_sha256"]
    if sha256_file(backup) != expected_sha256:
        raise ValueError("Backup SHA-256 does not match the completed project artifact. No files were restored.")
    restored = reused = 0
    with zipfile.ZipFile(backup) as archive:
        entries = archive.infolist()
        if len(entries) > 10000 or sum(entry.file_size for entry in entries) > MAX_TOTAL_BYTES:
            raise ValueError("Backup exceeds the supported artifact size/count.")
        planned, names = [], set()
        for entry in entries:
            destination = safe_member(root, entry.filename)
            if entry.filename in names:
                raise ValueError(f"Duplicate archive entry: {entry.filename}")
            names.add(entry.filename)
            if stat.S_ISLNK(entry.external_attr >> 16) or entry.file_size > MAX_FILE_BYTES:
                raise ValueError(f"Unsupported archive entry: {entry.filename}")
            if entry.is_dir():
                continue
            allowed = entry.filename.startswith(("outputs/", "data/processed/"))
            if not allowed or Path(entry.filename).suffix.lower() in {".py", ".pyc", ".exe", ".dll", ".bat", ".ps1", ".sh"}:
                continue
            if destination.exists():
                if not destination.is_file():
                    raise ValueError(f"Destination is not a regular file: {destination}")
                with archive.open(entry) as stream:
                    incoming = hashlib.file_digest(stream, "sha256").hexdigest()
                if sha256_file(destination) != incoming:
                    raise ValueError(f"Existing artifact differs; preserve it and use a fresh project copy: {destination}")
                reused += 1
            else:
                planned.append((entry, destination))
        active = json.loads(archive.read("outputs/day5_active.json"))
        experiment = safe_member(root, active["experiment"])
        if not experiment.is_relative_to(root / "outputs/day5"):
            raise ValueError("The active experiment must be inside outputs/day5.")
        config_name = (PurePosixPath(active["experiment"]) / "demo_config.json").as_posix()
        config = json.loads(archive.read(config_name))
        checkpoint = safe_member(root, config["checkpoint"])
        if not checkpoint.is_relative_to(root / "outputs"):
            raise ValueError("The selected checkpoint must be inside outputs.")
        with archive.open(config["checkpoint"]) as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != config["checkpoint_sha256"]:
            raise ValueError("The selected checkpoint does not match its recorded hash.")
        # All paths, hashes, and overwrite conflicts are checked before the first write.
        for entry, destination in planned:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".restoring")
            if temporary.exists():
                raise ValueError(f"A partial restore already exists: {temporary}")
            try:
                with archive.open(entry) as source, temporary.open("xb") as target:
                    shutil.copyfileobj(source, target)
                temporary.replace(destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
            restored += 1
    return {"config_path": experiment / "demo_config.json", "restored": restored, "reused": reused}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", required=True, type=Path)
    args = parser.parse_args()
    result = restore_backup(args.backup)
    print(f"Restored {result['restored']} files; reused {result['reused']} unchanged files.")
    print(f"Demo configuration: {result['config_path']}")


if __name__ == "__main__":
    main()
