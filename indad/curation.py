"""Reviewable dataset edits with preserved captures and persistent undo records."""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

from indad.workspace import IMAGE_EXTENSIONS, _name, inspect_dataset


class DatasetChanged(ValueError):
    """A client attempted to edit a dataset using an outdated inventory."""


def _safe_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.parts[0] not in {"train", "test", "ground_truth", ".indad"}
        or (path.parts[0] == ".indad" and path.parts[1:2] != ("archive",))
        or path.suffix.lower() not in IMAGE_EXTENSIONS
    ):
        raise ValueError("Choose an image path within this dataset.")
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()) or target.is_symlink():
        raise ValueError(
            "Image paths must stay inside the dataset without file symlinks."
        )
    return target


def _metadata(root):
    folder = root / ".indad"
    if not folder.resolve().is_relative_to(root.resolve()):
        raise ValueError("Dataset metadata must stay inside the dataset.")
    folder.mkdir(exist_ok=True)
    return folder


def _history(root):
    folder = _metadata(root) / "history"
    if not folder.resolve().is_relative_to(root.resolve()):
        raise ValueError("Dataset history must stay inside the dataset.")
    folder.mkdir(exist_ok=True)
    return folder


def list_changes(root: str | Path) -> list[dict]:
    root = Path(root)
    if not (root / ".indad/history").exists():
        return []
    records = [json.loads(path.read_text()) for path in _history(root).glob("*.json")]
    undone = {record["undo_of"] for record in records if "undo_of" in record}
    return [
        dict(record, undone=record["id"] in undone)
        for record in sorted(
            records, key=lambda r: (r["created_at"], r["id"]), reverse=True
        )
    ]


def _checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prune(root, folder):
    protected = {
        "train/good",
        "test/good",
        "train",
        "test",
        "ground_truth",
        ".indad",
        ".indad/history",
    }
    while folder != root and folder.relative_to(root).as_posix() not in protected:
        try:
            folder.rmdir()
        except OSError:
            break
        folder = folder.parent


def _apply(root, moves, record):
    """Publish without overwrites; roll back completed moves on handled errors."""
    checked = []
    for move in moves:
        source = _safe_path(root, move["from"])
        target = _safe_path(root, move["to"])
        if not source.is_file():
            raise ValueError(
                f"Sample moved or is missing: {move['from']}. Undo newer changes first."
            )
        if target.exists():
            raise FileExistsError(f"Cannot replace existing file: {move['to']}")
        if _checksum(source) != move["sha256"]:
            raise DatasetChanged(
                f"File contents changed: {move['from']}. Refresh before editing."
            )
        checked.append((source, target))
    log = _history(root) / f"{record['id']}.json"
    created, removed = [], []
    log_created = False
    try:
        # Copy every file before removing any source. Original bytes are retained.
        for source, target in checked:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                created.append(target)
                with source.open("rb") as original:
                    shutil.copyfileobj(original, output)
        for move, (source, target) in zip(moves, checked):
            if (
                _checksum(source) != move["sha256"]
                or _checksum(target) != move["sha256"]
            ):
                raise DatasetChanged(
                    "File contents changed while saving. Refresh and try again."
                )
        for source, target in checked:
            source.unlink()
            removed.append((source, target))
        with log.open("x", encoding="utf-8") as output:
            log_created = True
            json.dump(dict(record, files=moves), output, indent=2)
    except Exception:
        # Preserve the destination if restoring its source is unexpectedly blocked.
        for source, target in reversed(removed):
            with source.open("xb") as output, target.open("rb") as original:
                shutil.copyfileobj(original, output)
        for target in created:
            target.unlink()
        for target in created:
            _prune(root, target.parent)
        if log_created:
            log.unlink()
        raise
    # Empty defect folders upset torchvision.ImageFolder, so prune them.
    for source, _ in checked:
        _prune(root, source.parent)
    return dict(record, files=moves)


def revise_sample(
    root: str | Path,
    path: str,
    *,
    fingerprint: str,
    reason: str,
    split: str | None = None,
    label: str = "good",
    exclude: bool = False,
) -> dict:
    """Relabel/move a sample, or archive it outside the active dataset."""
    root = Path(root)
    if not reason.strip() or len(reason) > 500:
        raise ValueError("Provide a reason of 1–500 characters for this change.")
    source = _safe_path(root, path)
    if Path(path).parts[0] not in {"train", "test"}:
        raise ValueError("Only active training or inspection samples can be edited.")
    with FileLock(str(_metadata(root) / "curation.lock"), timeout=10):
        report = inspect_dataset(root)
        if report["fingerprint"] != fingerprint:
            raise DatasetChanged(
                "Dataset changed since you opened it. Refresh and review the sample again."
            )
        sample = next((s for s in report["samples"] if s["path"] == path), None)
        if sample is None:
            raise ValueError("Sample not found in the active dataset.")
        if (
            sample["mask"]
            and sum(s["mask"] == sample["mask"] for s in report["samples"]) > 1
        ):
            raise ValueError(
                "This mask is shared by multiple images. Give those images unique stems before editing."
            )
        change_id = uuid4().hex
        archive = Path(".indad/archive") / change_id
        moves = []

        def move(origin, destination):
            moves.append(
                {
                    "from": origin,
                    "to": destination.as_posix(),
                    "sha256": _checksum(_safe_path(root, origin)),
                }
            )

        if exclude:
            move(path, archive / path)
            if sample["mask"]:
                move(sample["mask"], archive / sample["mask"])
        else:
            if split not in {"train", "test"}:
                raise ValueError("Choose training or inspection for the sample.")
            _name(label)
            if split == "train" and label != "good":
                raise ValueError("Only healthy samples can be used for training.")
            tail = (
                Path(*Path(path).parts[2:])
                if len(Path(path).parts) > 2
                else Path(source.name)
            )
            destination = Path(split) / label / tail
            if destination.as_posix() == path:
                raise ValueError("Choose a different collection or label.")
            move(path, destination)
            if split == "test" and label != "good":
                mask_target = (
                    Path("ground_truth") / label / tail.parent / f"{tail.stem}_mask.png"
                )
                if _safe_path(root, mask_target.as_posix()).exists():
                    raise FileExistsError(
                        f"A mask already exists for the new label: {mask_target}"
                    )
            else:
                mask_target = archive / (sample["mask"] or "unused.png")
            if sample["mask"]:
                move(sample["mask"], mask_target)
        record = dict(
            id=change_id,
            action="exclude" if exclude else "relabel",
            path=path,
            reason=reason.strip(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        return _apply(root, moves, record)


def undo_change(root: str | Path, change_id: str, *, fingerprint: str) -> dict:
    """Undo an edit only when its files still match and destinations are free."""
    root = Path(root)
    with FileLock(str(_metadata(root) / "curation.lock"), timeout=10):
        if inspect_dataset(root)["fingerprint"] != fingerprint:
            raise DatasetChanged("Dataset changed. Refresh before undoing this edit.")
        record = next((r for r in list_changes(root) if r["id"] == change_id), None)
        if record is None or record["action"] == "undo" or record["undone"]:
            raise ValueError("This change is unavailable or has already been undone.")
        reverse = [
            {"from": move["to"], "to": move["from"], "sha256": move["sha256"]}
            for move in record["files"]
        ]
        event = dict(
            id=uuid4().hex,
            action="undo",
            undo_of=change_id,
            path=record["path"],
            reason=f"Undo {record['action']}: {record['reason']}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        return _apply(root, reverse, event)
