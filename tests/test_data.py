import io
import tarfile

import pytest
import torch
from PIL import Image

from indad.data import MVTecTestDataset, _safe_extract


def _save_image(path, value: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color=(value, value, value)).save(path)


def test_dataset_uses_path_components_for_labels_and_masks(tmp_path):
    class_name = "good_test_parts"
    dataset_root = tmp_path / class_name
    _save_image(dataset_root / "test" / "good" / "normal.png", 0)
    _save_image(dataset_root / "test" / "crack" / "broken.png", 255)
    _save_image(dataset_root / "ground_truth" / "crack" / "broken_mask.png", 255)

    dataset = MVTecTestDataset(class_name, size=16, root=tmp_path)
    samples = [dataset[index] for index in range(len(dataset))]
    by_label = {label: mask for _, mask, label in samples}

    assert set(by_label) == {0, 1}
    assert by_label[0].shape == (1, 16, 16)
    assert torch.count_nonzero(by_label[0]) == 0
    assert torch.all(by_label[1] == 1)


def test_safe_extract_rejects_parent_directory_members(tmp_path):
    archive_path = tmp_path / "unsafe.tar"
    payload = b"unsafe"
    with tarfile.open(archive_path, "w") as archive:
        member = tarfile.TarInfo("../escape.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    destination = tmp_path / "destination"
    destination.mkdir()
    with tarfile.open(archive_path) as archive:
        with pytest.raises(ValueError, match="Unsafe archive member"):
            _safe_extract(archive, destination)
    assert not (tmp_path / "escape.txt").exists()
