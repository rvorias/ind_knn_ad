"""Persistent MVTec-compatible datasets shared by operators and automation."""

import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("Use letters, numbers, hyphens or underscores for names.")
    return value


def create_dataset(parent: str | Path, name: str) -> Path:
    """Create a new dataset without replacing an existing directory."""
    root = Path(parent) / _name(name)
    root.mkdir(parents=True, exist_ok=False)
    for folder in ("train/good", "test/good", "ground_truth"):
        (root / folder).mkdir(parents=True)
    return root


def import_images(
    root: str | Path,
    files: list[tuple[str, bytes]],
    split: str = "train",
    label: str = "good",
) -> list[str]:
    """Validate a batch before publishing it; never overwrite existing files.

    Ground-truth uploads are PNG masks named <image-stem>_mask.png.
    Images retain their original bytes and names for traceability.
    """
    root = Path(root)
    if not (root / "train" / "good").is_dir():
        raise ValueError("Create or select a dataset with a train/good folder first.")
    _name(label)
    if split not in {"train", "test", "ground_truth"}:
        raise ValueError("Split must be train, test or ground_truth.")
    if split == "train" and label != "good":
        raise ValueError("Training images must be healthy (good).")
    if split == "ground_truth" and label == "good":
        raise ValueError("Masks belong to a defect type, not good images.")
    destination = root / split / label
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("Destination must stay inside the dataset.")
    names = set()
    with TemporaryDirectory(prefix=".import-", dir=root) as temp:
        for name, data in files:
            if Path(name).name != name or "\\" in name:
                raise ValueError(f"Use a filename without directories: {name}")
            if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image format: {name}")
            if split == "ground_truth" and not name.endswith("_mask.png"):
                raise ValueError(f"Name masks <image-stem>_mask.png: {name}")
            if name in names or (destination / name).exists():
                raise ValueError(f"File already exists or is repeated: {name}")
            try:
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
            except (OSError, ValueError) as exc:
                raise ValueError(f"Cannot read image: {name}") from exc
            (Path(temp) / name).write_bytes(data)
            names.add(name)
        destination.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also prevents concurrent imports from overwriting data.
        created = []
        try:
            for name in sorted(names):
                target = destination / name
                with target.open("xb") as output:
                    created.append(target)
                    output.write((Path(temp) / name).read_bytes())
        except OSError:
            for target in created:
                target.unlink()
            raise
    return [(destination / name).relative_to(root).as_posix() for name in sorted(names)]


def inspect_dataset(root: str | Path) -> dict:
    """Return a deterministic inventory and actionable readiness checks.

    Missing masks block pixel evaluation only. Duplicate decoded images across
    splits block evaluation, while still allowing an exploratory baseline.
    No files are changed and no model weights are loaded or downloaded.
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    samples, issues = [], []
    seen = {}
    used_masks = set()

    def issue(code, message, path=None, severity="warning"):
        issues.append(dict(code=code, severity=severity, path=path, message=message))

    def read_image(path):
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("File points outside the dataset")
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            size = image.size
            rgb = image.convert("RGB")
            pixels = hashlib.sha256(str(size).encode() + rgb.tobytes()).hexdigest()
        return size, hashlib.sha256(data).hexdigest(), pixels

    for split in ("train", "test"):
        for path in sorted((root / split).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = path.relative_to(root).as_posix()
            parts = path.relative_to(root / split).parts
            label = parts[0] if len(parts) >= 2 else None
            record = dict(path=relative, split=split, label=label, mask=None)
            samples.append(record)
            if label is None or (split == "train" and label != "good"):
                issue(
                    "invalid_label",
                    "Put healthy training images in train/good and inspection images in test/<label>.",
                    relative,
                    "error",
                )
            try:
                size, digest, pixels = read_image(path)
                record.update(width=size[0], height=size[1], sha256=digest)
            except (OSError, ValueError) as exc:
                issue(
                    "unreadable_image",
                    f"Replace this unreadable image: {exc}",
                    relative,
                    "error",
                )
                continue
            if pixels in seen:
                original = seen[pixels]
                cross_split = original["split"] != split
                if original["label"] != label:
                    issue(
                        "conflicting_labels",
                        f"Same decoded image as {original['path']} has a different label. Review both labels.",
                        relative,
                    )
                issue(
                    "split_leakage" if cross_split else "duplicate_image",
                    f"Same decoded image as {original['path']}. Keep related captures in one split.",
                    relative,
                )
            else:
                seen[pixels] = record
            if split != "test" or label in (None, "good"):
                continue
            mask = (
                root
                / "ground_truth"
                / path.relative_to(root / "test").parent
                / f"{path.stem}_mask.png"
            )
            if not mask.is_file():
                issue(
                    "missing_mask",
                    "Add a matching ground-truth mask to enable pixel evaluation.",
                    relative,
                )
                continue
            mask_relative = mask.relative_to(root).as_posix()
            used_masks.add(mask_relative)
            record["mask"] = mask_relative
            try:
                mask_size, mask_digest, _ = read_image(mask)
                record["mask_sha256"] = mask_digest
                if mask_size != size:
                    issue(
                        "mask_size",
                        "Mask dimensions must match the original image.",
                        mask_relative,
                        "error",
                    )
                with Image.open(mask) as image:
                    values = set(image.convert("L").getdata())
                if not values <= {0, 255}:
                    issue(
                        "mask_values",
                        "Use a binary mask: 0 for healthy pixels, 255 for defects.",
                        mask_relative,
                        "error",
                    )
                elif not (values - {0}):
                    issue(
                        "empty_mask",
                        "This defective image has an empty mask. Review its label or mask.",
                        mask_relative,
                        "error",
                    )
            except (OSError, ValueError) as exc:
                issue(
                    "unreadable_mask",
                    f"Replace this unreadable mask: {exc}",
                    mask_relative,
                    "error",
                )
    for mask in sorted((root / "ground_truth").rglob("*")):
        if mask.is_file() and mask.suffix.lower() in IMAGE_EXTENSIONS:
            relative = mask.relative_to(root).as_posix()
            if relative not in used_masks:
                issue(
                    "orphan_mask",
                    "No inspection image uses this mask. Check its name and defect folder.",
                    relative,
                )
    counts = Counter(f"{s['split']}/{s['label']}" for s in samples)
    healthy_train = counts["train/good"]
    test_samples = [s for s in samples if s["split"] == "test"]
    if healthy_train < 2:
        issue(
            "few_training_images",
            "Add at least two healthy training images for the baseline workflow.",
            severity="error",
        )
    if not test_samples:
        issue("no_test_images", "Add inspection images to review predictions.")
    labels = {s["label"] == "good" for s in test_samples if s["label"]}
    if len(labels) < 2:
        issue(
            "single_test_class",
            "Include both healthy and defective inspection images to measure image ROC AUC.",
        )
    sizes = {(s.get("width"), s.get("height")) for s in samples if "width" in s}
    if len(sizes) > 1:
        issue(
            "mixed_dimensions",
            "Image sizes differ. Review framing; the baseline resizes and center-crops images.",
        )
    training_errors = any(
        i["severity"] == "error"
        and (i["path"] is None or i["path"].startswith("train/"))
        for i in issues
    )
    image_errors = any(
        i["code"]
        in {"invalid_label", "unreadable_image", "split_leakage", "conflicting_labels"}
        for i in issues
    )
    ready_train = healthy_train >= 2 and not training_errors
    ready_inspect = (
        ready_train
        and bool(test_samples)
        and not any(i["code"] in {"invalid_label", "unreadable_image"} for i in issues)
    )
    ready_image = ready_inspect and len(labels) == 2 and not image_errors
    ready_pixel = ready_image and not any(
        i["code"]
        in {"missing_mask", "mask_size", "mask_values", "empty_mask", "unreadable_mask"}
        for i in issues
    )
    fingerprint = hashlib.sha256(
        json.dumps(samples, sort_keys=True).encode()
    ).hexdigest()
    return dict(
        schema_version=1,
        name=root.name,
        fingerprint=fingerprint,
        counts=dict(sorted(counts.items())),
        readiness=dict(
            training=ready_train,
            inspection=ready_inspect,
            image_evaluation=ready_image,
            pixel_evaluation=ready_pixel,
        ),
        issues=issues,
        samples=samples,
    )
