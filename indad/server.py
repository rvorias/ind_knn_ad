"""Local operator console and documented HTTP API for dataset automation."""

import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4

import click
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from indad._version import __version__
from indad.curation import DatasetChanged, list_changes, revise_sample, undo_change
from indad.workspace import (
    IMAGE_EXTENSIONS,
    _name,
    create_dataset,
    import_images,
    inspect_dataset,
)

WEB_ROOT = Path(__file__).parent / "web"
MAX_IMPORT_BYTES = 100 * 1024 * 1024


class DatasetRequest(BaseModel):
    name: str


class RunRequest(BaseModel):
    dataset: str
    method: Literal["patchcore", "padim", "spade"] = "patchcore"
    device: Literal["cpu", "cuda"] = "cpu"


class SampleEditRequest(BaseModel):
    path: str
    fingerprint: str
    reason: str
    split: Literal["train", "test"] | None = None
    label: str = "good"
    exclude: bool = False


class UndoRequest(BaseModel):
    fingerprint: str


class PredictionRequest(BaseModel):
    path: str


def _png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _image_url(image: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(_png(image)).decode("ascii")


class InspectionRuns:
    """One active baseline per local server; keep its identity explicit."""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.RLock()
        self.current = None
        self.model = None

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def status(self, run_id=None):
        with self.lock:
            if self.current is None or (run_id and self.current["id"] != run_id):
                raise HTTPException(
                    404, "Baseline run not found. Train a new baseline."
                )
            return dict(self.current)

    def start(self, root: Path, options: RunRequest):
        report = inspect_dataset(root)
        if not report["readiness"]["inspection"]:
            raise HTTPException(
                409,
                "Resolve the dataset's inspection readiness findings before training.",
            )
        with self.lock:
            if self.current and self.current["status"] in {"training", "predicting"}:
                raise HTTPException(
                    409, "An inspection task is already running. Wait for it to finish."
                )
            self.model = None
            self.current = dict(
                id=uuid4().hex,
                dataset=root.name,
                method=options.method,
                device=options.device,
                fingerprint=report["fingerprint"],
                status="training",
                message="Loading baseline and training on healthy images.",
            )
            result = dict(self.current)
            self.executor.submit(self._train, root, options, report)
            return result

    def _train(self, root, options, report):
        try:
            from torch.utils.data import DataLoader

            from indad.cli import _build_model
            from indad.data import StreamingDataset

            training = StreamingDataset()
            for record in report["samples"]:
                if record["split"] == "train":
                    with Image.open(root / record["path"]) as image:
                        training.add_pil_image(image)
            model = _build_model(options.method, "resnet18", 224, options.device, 0)
            model.fit(DataLoader(training))
            if inspect_dataset(root)["fingerprint"] != report["fingerprint"]:
                raise ValueError(
                    "Dataset changed during training. Train again with the updated collection."
                )
            with self.lock:
                self.model = model
                self.current.update(
                    status="ready",
                    message="Baseline ready. Select an inspection image.",
                )
        except Exception as exc:
            with self.lock:
                self.model = None
                self.current.update(status="failed", message=str(exc))

    def predict(self, run_id, root, relative):
        with self.lock:
            run = self.status(run_id)
            if run["status"] != "ready":
                raise HTTPException(409, "Baseline is not ready for inspection.")
            self.current["status"] = "predicting"
            model = self.model
        try:
            report = inspect_dataset(root)
            if report["fingerprint"] != run["fingerprint"]:
                with self.lock:
                    self.model = None
                    self.current.update(
                        status="stale", message="Dataset changed. Train a new baseline."
                    )
                raise HTTPException(409, "Dataset changed. Train a new baseline.")
            if relative not in {
                s["path"] for s in report["samples"] if s["split"] == "test"
            }:
                raise HTTPException(
                    400, "Choose an inspection image from this dataset."
                )
            from indad.data import IMAGENET_MEAN, IMAGENET_STD, StreamingDataset

            dataset = StreamingDataset()
            with Image.open(root / relative) as image:
                dataset.add_pil_image(image)
            sample, _ = dataset[0]
            score, score_map = model.predict(sample.unsqueeze(0))
            import numpy as np

            pixels = sample * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
            original = Image.fromarray(
                (pixels.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype("uint8")
            )
            values = score_map.detach().cpu().squeeze().numpy()
            low, high = float(values.min()), float(values.max())
            normalized = (
                (values - low) / (high - low) if high > low else np.zeros_like(values)
            )
            # A fixed blue → cyan → amber palette; normalization is per image.
            stops = np.array(
                [[19, 41, 63], [27, 158, 157], [254, 201, 85]], dtype=float
            )
            channels = [
                np.interp(normalized, [0, 0.5, 1], stops[:, c]) for c in range(3)
            ]
            heatmap = Image.fromarray(np.stack(channels, axis=-1).astype("uint8"))
            result = dict(
                run_id=run_id,
                path=relative,
                score=float(score),
                minimum=low,
                maximum=high,
                original=_image_url(original),
                heatmap=_image_url(heatmap),
                overlay=_image_url(Image.blend(original, heatmap, 0.45)),
            )
            if inspect_dataset(root)["fingerprint"] != run["fingerprint"]:
                with self.lock:
                    self.model = None
                    self.current.update(
                        status="stale",
                        message="Dataset changed during inspection. Train again.",
                    )
                raise HTTPException(
                    409, "Dataset changed during inspection. Train again."
                )
            return result
        finally:
            with self.lock:
                if self.current["status"] == "predicting":
                    self.current["status"] = "ready"


def create_app(dataset_root: str | Path = "datasets") -> FastAPI:
    storage = Path(dataset_root).resolve()
    runs = InspectionRuns()
    # Coordinate imports with report reads, including requests from other tabs.
    dataset_lock = threading.RLock()

    @asynccontextmanager
    async def lifespan(app):
        yield
        runs.close()

    app = FastAPI(title="Indad Operator API", version=__version__, lifespan=lifespan)
    app.state.runs = runs

    @app.exception_handler(DatasetChanged)
    async def stale_dataset(request, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def invalid_value(request, exc):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(OSError)
    async def file_error(request, exc):
        status = (
            409
            if isinstance(exc, FileExistsError)
            else 404
            if isinstance(exc, FileNotFoundError)
            else 400
        )
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    def dataset_path(name):
        _name(name)
        path = (storage / name).resolve()
        if not path.is_relative_to(storage) or not path.is_dir():
            raise HTTPException(
                404, "Dataset not found in the configured storage folder."
            )
        return path

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/api/datasets")
    def list_datasets():
        if not storage.is_dir():
            return {"datasets": []}
        return {
            "datasets": sorted(
                p.name
                for p in storage.iterdir()
                if p.is_dir()
                and p.resolve().is_relative_to(storage)
                and (p / "train/good").is_dir()
            )
        }

    @app.post("/api/datasets", status_code=201)
    def new_dataset(body: DatasetRequest):
        with dataset_lock:
            path = create_dataset(storage, body.name)
        return {"name": path.name}

    @app.get("/api/datasets/{name}/report")
    def report(name: str):
        with dataset_lock:
            return inspect_dataset(dataset_path(name))

    @app.post("/api/datasets/{name}/import", status_code=201)
    def upload(
        name: str,
        files: list[UploadFile] = File(...),
        split: str = Form("train"),
        label: str = Form("good"),
    ):
        if len(files) > 500:
            raise HTTPException(413, "Import at most 500 files at a time.")
        batch, size = [], 0
        for upload in files:
            data = upload.file.read(MAX_IMPORT_BYTES - size + 1)
            size += len(data)
            if size > MAX_IMPORT_BYTES:
                raise HTTPException(413, "Keep each import below 100 MB.")
            batch.append((upload.filename or "", data))
        with dataset_lock:
            paths = import_images(dataset_path(name), batch, split, label)
        return {"imported": paths}

    @app.get("/api/datasets/{name}/changes")
    def changes(name: str):
        with dataset_lock:
            return {"changes": list_changes(dataset_path(name))}

    @app.post("/api/datasets/{name}/samples/revise")
    def edit_sample(name: str, body: SampleEditRequest):
        with dataset_lock:
            return revise_sample(
                dataset_path(name),
                body.path,
                fingerprint=body.fingerprint,
                reason=body.reason,
                split=body.split,
                label=body.label,
                exclude=body.exclude,
            )

    @app.post("/api/datasets/{name}/changes/{change_id}/undo")
    def undo(name: str, change_id: str, body: UndoRequest):
        with dataset_lock:
            return undo_change(
                dataset_path(name), change_id, fingerprint=body.fingerprint
            )

    @app.get("/api/datasets/{name}/image")
    def image(name: str, path: str, thumbnail: bool = False):
        root = dataset_path(name)
        relative = Path(path)
        target = (root / relative).resolve()
        if (
            relative.is_absolute()
            or not relative.parts
            or relative.parts[0] not in {"train", "test", "ground_truth"}
            or not target.is_relative_to(root)
            or target.suffix.lower() not in IMAGE_EXTENSIONS
            or not target.is_file()
        ):
            raise HTTPException(404, "Image not found in this dataset.")
        with Image.open(target) as source:
            source = source.convert("RGB")
            if thumbnail:
                source.thumbnail((360, 280))
            return Response(
                _png(source),
                media_type="image/png",
                headers={"Cache-Control": "no-cache"},
            )

    @app.post("/api/runs", status_code=202)
    def train(body: RunRequest):
        with dataset_lock:
            return runs.start(dataset_path(body.dataset), body)

    @app.get("/api/runs/current")
    def current_run():
        return runs.status() if runs.current else None

    @app.get("/api/runs/{run_id}")
    def run_status(run_id: str):
        return runs.status(run_id)

    @app.post("/api/runs/{run_id}/predict")
    def predict(run_id: str, body: PredictionRequest):
        run = runs.status(run_id)
        return runs.predict(run_id, dataset_path(run["dataset"]), body.path)

    app.mount("/assets", StaticFiles(directory=WEB_ROOT), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_ROOT / "index.html")

    return app


@click.command()
@click.option(
    "--datasets",
    default="datasets",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Dataset storage folder.",
)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=click.IntRange(1, 65535))
def main(datasets, host, port):
    """Serve the operator console and its HTTP API."""
    import uvicorn

    uvicorn.run(create_app(datasets), host=host, port=port)


if __name__ == "__main__":
    main()
