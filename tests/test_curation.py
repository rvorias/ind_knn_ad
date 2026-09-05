import io
from pathlib import Path

import pytest
from PIL import Image

from indad.curation import DatasetChanged, list_changes, revise_sample, undo_change
from indad.workspace import create_dataset, import_images, inspect_dataset


def png(value):
    output = io.BytesIO()
    Image.new("RGB", (24, 24), (value, value, value)).save(output, "PNG")
    return output.getvalue()


@pytest.fixture
def collection(tmp_path):
    root = create_dataset(tmp_path, "parts")
    import_images(root, [("a.png", png(10)), ("b.png", png(20))])
    import_images(root, [("c.png", png(30))], "test")
    import_images(root, [("d.png", png(40))], "test", "scratch")
    import_images(root, [("d_mask.png", png(255))], "ground_truth", "scratch")
    return root


def fingerprint(root):
    return inspect_dataset(root)["fingerprint"]


def edit(root, path="test/scratch/d.png", **kwargs):
    return revise_sample(
        root, path, fingerprint=fingerprint(root), reason="Reviewed capture", **kwargs
    )


def test_relabel_moves_image_and_mask_and_undo_restores_inventory(collection):
    before = inspect_dataset(collection)
    record = edit(collection, split="test", label="dent")
    assert (collection / "test/dent/d.png").read_bytes() == png(40)
    assert (collection / "ground_truth/dent/d_mask.png").read_bytes() == png(255)
    assert not (collection / "test/scratch").exists()
    assert fingerprint(collection) != before["fingerprint"]
    undo_change(collection, record["id"], fingerprint=fingerprint(collection))
    assert inspect_dataset(collection) == before
    assert list_changes(collection)[0]["action"] == "undo"
    assert list_changes(collection)[1]["undone"]


def test_marking_healthy_archives_mask_until_undo(collection):
    record = edit(collection, split="test", label="good")
    assert (collection / "test/good/d.png").exists()
    mask_move = record["files"][1]
    assert mask_move["to"].startswith(".indad/archive/")
    assert (collection / mask_move["to"]).read_bytes() == png(255)
    assert not any(
        i["code"] == "orphan_mask" for i in inspect_dataset(collection)["issues"]
    )
    undo_change(collection, record["id"], fingerprint=fingerprint(collection))
    assert (collection / "ground_truth/scratch/d_mask.png").read_bytes() == png(255)


def test_exclusion_preserves_files_and_is_reversible(collection):
    before = fingerprint(collection)
    record = edit(collection, exclude=True)
    assert all((collection / move["to"]).is_file() for move in record["files"])
    assert not (collection / "test/scratch/d.png").exists()
    assert inspect_dataset(collection)["counts"].get("test/scratch", 0) == 0
    undo_change(collection, record["id"], fingerprint=fingerprint(collection))
    assert fingerprint(collection) == before


def test_corrupt_capture_can_be_excluded(collection):
    path = collection / "train/good/a.png"
    path.write_bytes(b"broken capture")
    record = edit(collection, "train/good/a.png", exclude=True)
    assert (collection / record["files"][0]["to"]).read_bytes() == b"broken capture"
    assert "unreadable_image" not in {
        i["code"] for i in inspect_dataset(collection)["issues"]
    }


def test_stale_edit_is_rejected_without_moving_files(collection):
    before = fingerprint(collection)
    import_images(collection, [("new.png", png(50))])
    with pytest.raises(DatasetChanged):
        revise_sample(
            collection,
            "test/scratch/d.png",
            fingerprint=before,
            reason="Review",
            exclude=True,
        )
    assert (collection / "test/scratch/d.png").is_file()
    assert list_changes(collection) == []


@pytest.mark.parametrize("target", ["test/dent/d.png", "ground_truth/dent/d_mask.png"])
def test_relabel_never_overwrites_image_or_orphan_mask(collection, target):
    path = collection / target
    path.parent.mkdir(parents=True)
    path.write_bytes(png(200))
    with pytest.raises(FileExistsError):
        edit(collection, split="test", label="dent")
    assert path.read_bytes() == png(200)
    assert (collection / "test/scratch/d.png").read_bytes() == png(40)


def test_cannot_label_defect_as_training_or_submit_empty_reason(collection):
    with pytest.raises(ValueError, match="healthy"):
        edit(collection, split="train", label="scratch")
    with pytest.raises(ValueError, match="reason"):
        revise_sample(
            collection,
            "test/scratch/d.png",
            fingerprint=fingerprint(collection),
            reason=" ",
            exclude=True,
        )


def test_undo_rejects_changed_archived_files(collection):
    record = edit(collection, exclude=True)
    (collection / record["files"][0]["to"]).write_bytes(png(99))
    with pytest.raises(DatasetChanged):
        undo_change(collection, record["id"], fingerprint=fingerprint(collection))
    assert not (collection / "test/scratch/d.png").exists()


def test_undo_does_not_replace_a_new_capture_at_the_old_path(collection):
    record = edit(collection, exclude=True)
    import_images(collection, [("d.png", png(99))], "test", "scratch")
    with pytest.raises(FileExistsError):
        undo_change(collection, record["id"], fingerprint=fingerprint(collection))
    assert (collection / "test/scratch/d.png").read_bytes() == png(99)
    assert (collection / record["files"][0]["to"]).read_bytes() == png(40)


def test_history_write_failure_rolls_back_files(collection, monkeypatch):
    before = inspect_dataset(collection)
    original = Path.open

    def fail_log(path, mode="r", *args, **kwargs):
        if path.parent.name == "history" and mode == "x":
            raise OSError("Disk full")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_log)
    with pytest.raises(OSError, match="Disk full"):
        edit(collection, split="test", label="dent")
    assert inspect_dataset(collection) == before
    assert not (collection / "test/dent").exists()
    assert list_changes(collection) == []


def test_files_can_be_undone_in_reverse_edit_order(collection):
    first = edit(collection, split="test", label="dent")
    second = edit(collection, "test/dent/d.png", split="train")
    with pytest.raises(ValueError, match="newer"):
        undo_change(collection, first["id"], fingerprint=fingerprint(collection))
    undo_change(collection, second["id"], fingerprint=fingerprint(collection))
    undo_change(collection, first["id"], fingerprint=fingerprint(collection))
    assert (collection / "test/scratch/d.png").exists()


def test_external_file_and_archive_symlinks_are_rejected(collection, tmp_path):
    outside = tmp_path / "outside.png"
    outside.write_bytes(png(77))
    (collection / "test/good/link.png").symlink_to(outside)
    with pytest.raises(ValueError, match="inside"):
        edit(collection, "test/good/link.png", exclude=True)
    (collection / ".indad").mkdir(exist_ok=True)
    (collection / ".indad/archive").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="inside"):
        edit(collection, exclude=True)
    assert outside.read_bytes() == png(77)


def test_agent_cli_revises_and_undoes_sample(collection):
    import json

    from click.testing import CliRunner

    from indad.dataset_cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "revise",
            str(collection),
            "test/scratch/d.png",
            "--exclude",
            "--reason",
            "Blurred capture",
            "--fingerprint",
            fingerprint(collection),
        ],
    )
    assert result.exit_code == 0, result.output
    change_id = json.loads(result.output)["id"]
    assert (
        json.loads(runner.invoke(cli, ["changes", str(collection)]).output)["changes"][
            0
        ]["id"]
        == change_id
    )
    result = runner.invoke(
        cli,
        ["undo", str(collection), change_id, "--fingerprint", fingerprint(collection)],
    )
    assert result.exit_code == 0, result.output
    assert (collection / "test/scratch/d.png").exists()


def test_shared_mask_cannot_be_moved_away_from_another_sample(collection):
    import_images(collection, [("d.jpg", png(80))], "test", "scratch")
    with pytest.raises(ValueError, match="shared"):
        edit(collection, exclude=True)
    assert (collection / "ground_truth/scratch/d_mask.png").exists()
    assert (collection / "test/scratch/d.png").exists()
