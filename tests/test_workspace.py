import io
import json

import pytest
from click.testing import CliRunner
from PIL import Image

from indad.dataset_cli import cli
from indad.workspace import create_dataset, import_images, inspect_dataset


def image_bytes(value=0, size=(24, 20), format="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", size, (value, value, value)).save(buffer, format=format)
    return buffer.getvalue()


def populated_dataset(tmp_path):
    root = create_dataset(tmp_path, "parts")
    import_images(
        root, [("healthy1.png", image_bytes(10)), ("healthy2.png", image_bytes(20))]
    )
    import_images(root, [("healthy3.png", image_bytes(30))], "test")
    import_images(root, [("scratch.png", image_bytes(200))], "test", "scratch")
    return root


def test_create_and_import_never_overwrite_or_publish_invalid_batch(tmp_path):
    root = create_dataset(tmp_path, "parts")
    with pytest.raises(FileExistsError):
        create_dataset(tmp_path, "parts")
    with pytest.raises(ValueError, match="Cannot read"):
        import_images(root, [("valid.png", image_bytes()), ("broken.png", b"bad")])
    assert list((root / "train/good").iterdir()) == []
    import_images(root, [("valid.png", image_bytes())])
    with pytest.raises(ValueError, match="already exists"):
        import_images(root, [("valid.png", image_bytes(255))])
    assert (root / "train/good/valid.png").read_bytes() == image_bytes()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "bad/name", ""])
def test_create_rejects_invalid_names(tmp_path, name):
    with pytest.raises(ValueError):
        create_dataset(tmp_path, name)


def test_import_rejects_traversal_and_defective_training(tmp_path):
    root = create_dataset(tmp_path, "parts")
    with pytest.raises(ValueError, match="directories"):
        import_images(root, [("../escape.png", image_bytes())])
    with pytest.raises(ValueError, match="healthy"):
        import_images(root, [("bad.png", image_bytes())], "train", "scratch")
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "test/scratch").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="inside"):
        import_images(root, [("bad.png", image_bytes())], "test", "scratch")
    assert not list(outside.iterdir())


def test_missing_masks_do_not_block_inspection_or_image_evaluation(tmp_path):
    root = populated_dataset(tmp_path)
    report = inspect_dataset(root)
    assert report["readiness"] == dict(
        training=True, inspection=True, image_evaluation=True, pixel_evaluation=False
    )
    assert "missing_mask" in {issue["code"] for issue in report["issues"]}
    assert inspect_dataset(root) == report
    import_images(
        root, [("scratch_mask.png", image_bytes(255))], "ground_truth", "scratch"
    )
    with_mask = inspect_dataset(root)
    assert all(with_mask["readiness"].values())
    assert with_mask["fingerprint"] != report["fingerprint"]
    assert with_mask["samples"][-1]["sha256"]


@pytest.mark.parametrize(
    "mask,code",
    [
        (image_bytes(255, (10, 10)), "mask_size"),
        (image_bytes(0), "empty_mask"),
        (image_bytes(128), "mask_values"),
    ],
)
def test_invalid_masks_block_only_pixel_evaluation(tmp_path, mask, code):
    root = populated_dataset(tmp_path)
    import_images(root, [("scratch_mask.png", mask)], "ground_truth", "scratch")
    report = inspect_dataset(root)
    assert code in {i["code"] for i in report["issues"]}
    assert report["readiness"]["image_evaluation"]
    assert not report["readiness"]["pixel_evaluation"]


def test_cross_split_duplicates_detected_even_with_different_encoding(tmp_path):
    root = populated_dataset(tmp_path)
    import_images(root, [("copy.bmp", image_bytes(10, format="BMP"))], "test")
    report = inspect_dataset(root)
    assert "split_leakage" in {i["code"] for i in report["issues"]}
    assert report["readiness"]["inspection"]
    assert not report["readiness"]["image_evaluation"]


def test_unreadable_training_image_blocks_baseline(tmp_path):
    root = populated_dataset(tmp_path)
    (root / "train/good/healthy1.png").write_bytes(b"broken")
    report = inspect_dataset(root)
    assert not report["readiness"]["training"]
    assert not report["readiness"]["inspection"]


def test_cli_create_import_and_inspect(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["create", "parts", "--root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    root = json.loads(result.output)["dataset"]
    source = tmp_path / "source.png"
    source.write_bytes(image_bytes())
    result = runner.invoke(cli, ["import", root, str(source)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["imported"] == ["train/good/source.png"]
    result = runner.invoke(cli, ["inspect", root, "--require", "inspection"])
    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report["schema_version"] == 1
    assert not report["readiness"]["inspection"]
    assert runner.invoke(cli, ["inspect", root]).exit_code == 0


def test_inventory_loads_with_existing_mvtec_adapter(tmp_path):
    from indad.data import MVTecDataset

    root = populated_dataset(tmp_path)
    import_images(
        root, [("scratch_mask.png", image_bytes(255))], "ground_truth", "scratch"
    )
    train, test = MVTecDataset("parts", root=tmp_path, size=16).get_datasets()
    assert len(train) == 2
    assert {int(test[i][2]) for i in range(len(test))} == {0, 1}


def test_conflicting_test_labels_block_evaluation(tmp_path):
    root = populated_dataset(tmp_path)
    import_images(root, [("conflict.png", image_bytes(30))], "test", "scratch")
    report = inspect_dataset(root)
    assert "conflicting_labels" in {i["code"] for i in report["issues"]}
    assert report["readiness"]["inspection"]
    assert not report["readiness"]["image_evaluation"]


def test_dataset_cli_does_not_import_model_dependencies():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import indad.dataset_cli, sys; assert 'torch' not in sys.modules; assert 'timm' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
