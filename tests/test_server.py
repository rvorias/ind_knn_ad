import base64
import io
import threading
import time
from importlib import import_module

import pytest
from PIL import Image

from indad.workspace import create_dataset, import_images

TestClient = pytest.importorskip("fastapi.testclient").TestClient
create_app = import_module("indad.server").create_app


def image_bytes(value=0):
    output = io.BytesIO()
    Image.new("RGB", (32, 32), (value, value, value)).save(output, "PNG")
    return output.getvalue()


@pytest.fixture
def workspace(tmp_path):
    root = create_dataset(tmp_path, "parts")
    import_images(root, [("a.png", image_bytes(10)), ("b.png", image_bytes(20))])
    import_images(root, [("c.png", image_bytes(30))], "test")
    return root


@pytest.fixture
def client(workspace):
    with TestClient(create_app(workspace.parent)) as client:
        yield client


def wait_for_run(client, run_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] != "training":
            return run
        time.sleep(0.02)
    pytest.fail("Training did not complete")


def test_console_and_api_are_served_together(client):
    assert client.get("/").status_code == 200
    assert "Operator console" in client.get("/").text
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/assets/style.css").status_code == 200
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/datasets").json() == {"datasets": ["parts"]}
    assert "/api/datasets/{name}/import" in client.get("/openapi.json").json()["paths"]


def test_create_upload_report_and_duplicate_errors(client):
    assert client.post("/api/datasets", json={"name": "new_parts"}).status_code == 201
    assert client.post("/api/datasets", json={"name": "new_parts"}).status_code == 409
    response = client.post(
        "/api/datasets/new_parts/import",
        files=[("files", ("capture.png", image_bytes(), "image/png"))],
        data={"split": "train", "label": "good"},
    )
    assert response.status_code == 201
    assert response.json()["imported"] == ["train/good/capture.png"]
    report = client.get("/api/datasets/new_parts/report").json()
    assert report["counts"] == {"train/good": 1}
    assert not report["readiness"]["inspection"]
    response = client.post(
        "/api/datasets/new_parts/import",
        files=[("files", ("capture.png", image_bytes(10), "image/png"))],
    )
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]


def test_import_rejects_invalid_batch_and_training_labels(client, workspace):
    response = client.post(
        "/api/datasets/parts/import",
        files=[
            ("files", ("valid.png", image_bytes())),
            ("files", ("broken.png", b"bad")),
        ],
    )
    assert response.status_code == 400
    assert not (workspace / "train/good/valid.png").exists()
    response = client.post(
        "/api/datasets/parts/import",
        files=[("files", ("defect.png", image_bytes()))],
        data={"split": "train", "label": "scratch"},
    )
    assert response.status_code == 400


@pytest.mark.parametrize(
    "path",
    ["../outside.png", "/etc/passwd", "train/good/../../../outside.png", "notes.txt"],
)
def test_image_route_rejects_paths_outside_dataset(client, path):
    assert (
        client.get("/api/datasets/parts/image", params={"path": path}).status_code
        == 404
    )


def test_symlink_image_cannot_escape_dataset(client, workspace):
    outside = workspace.parent / "outside.png"
    outside.write_bytes(image_bytes())
    (workspace / "test/good/linked.png").symlink_to(outside)
    assert (
        client.get(
            "/api/datasets/parts/image", params={"path": "test/good/linked.png"}
        ).status_code
        == 404
    )


def test_image_thumbnails_and_report_checksums(client):
    response = client.get(
        "/api/datasets/parts/image",
        params={"path": "train/good/a.png", "thumbnail": True},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    with Image.open(io.BytesIO(response.content)) as image:
        assert image.size == (32, 32)
    assert client.get("/api/datasets/parts/report").json()["samples"][0]["sha256"]


def test_baseline_predicts_and_rejects_changed_dataset(client, workspace):
    response = client.post("/api/runs", json={"dataset": "parts", "method": "spade"})
    assert response.status_code == 202
    run_id = response.json()["id"]
    assert wait_for_run(client, run_id)["status"] == "ready"
    assert client.get("/api/runs/current").json()["id"] == run_id
    invalid = client.post(
        f"/api/runs/{run_id}/predict", json={"path": "train/good/a.png"}
    )
    assert invalid.status_code == 400
    result = client.post(
        f"/api/runs/{run_id}/predict", json={"path": "test/good/c.png"}
    )
    assert result.status_code == 200, result.text
    data = result.json()
    assert data["score"] >= 0
    for key in ("original", "heatmap", "overlay"):
        image = Image.open(io.BytesIO(base64.b64decode(data[key].split(",")[1])))
        assert image.size == (224, 224)
    import_images(workspace, [("new.png", image_bytes(50))])
    result = client.post(
        f"/api/runs/{run_id}/predict", json={"path": "test/good/c.png"}
    )
    assert result.status_code == 409
    assert client.get(f"/api/runs/{run_id}").json()["status"] == "stale"


def test_training_failure_is_visible_to_operator(client, monkeypatch):
    import indad.cli

    def fail(*args):
        raise RuntimeError("Weights unavailable")

    monkeypatch.setattr(indad.cli, "_build_model", fail)
    response = client.post("/api/runs", json={"dataset": "parts"})
    run = wait_for_run(client, response.json()["id"])
    assert run["status"] == "failed"
    assert run["message"] == "Weights unavailable"


def test_training_is_background_and_rejects_overlapping_jobs(client, monkeypatch):
    import indad.cli

    started, release = threading.Event(), threading.Event()

    def wait_then_fail(*args):
        started.set()
        release.wait(5)
        raise RuntimeError("test finished")

    monkeypatch.setattr(indad.cli, "_build_model", wait_then_fail)
    response = client.post("/api/runs", json={"dataset": "parts"})
    try:
        assert response.status_code == 202
        assert started.wait(3)
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/datasets/parts/report").status_code == 200
        assert client.post("/api/runs", json={"dataset": "parts"}).status_code == 409
    finally:
        release.set()


def test_empty_dataset_cannot_train(client):
    client.post("/api/datasets", json={"name": "empty"})
    assert client.post("/api/runs", json={"dataset": "empty"}).status_code == 409
    assert (
        client.post(
            "/api/runs", json={"dataset": "parts", "method": "unknown"}
        ).status_code
        == 422
    )
    assert client.get("/api/runs/missing").status_code == 404
