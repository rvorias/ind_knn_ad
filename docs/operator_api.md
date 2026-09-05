# Operator console and HTTP API

Install and start the local application:

```shell
python -m pip install -e ".[web]"
indad-web --datasets ./datasets --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. No Node installation, build step, external fonts, or
frontend service is required. HTML, CSS and JavaScript ship inside the Python
package. `python -m indad.server` is equivalent to the console entry point.

The console is a local workspace with no accounts or authentication. The default
bind address is loopback. Files are scoped to the storage folder configured at
startup. Training and prediction use the server's compute device.

## Operator workflow

1. Create or select a collection from the sidebar.
2. Import images as healthy training, healthy inspection, defective inspection, or
   defect masks. Choose a defect type when relevant. Imports preserve original
   filenames and bytes; existing files are never overwritten. Each batch is limited
   to 500 files and 100 MB.
3. Browse samples, search filenames, filter by collection, and review the selected
   image and its mask. Arrow buttons move between samples. The gallery loads 24
   samples at a time and uses thumbnails.
4. Use **Correct label or collection** or **Exclude sample** in the sample panel,
   record the reason, and save. Review or undo edits in **Change history**. Excluded
   images and unused masks are preserved in the dataset's `.indad` folder.
5. Open **Quality checks** to see operation-specific readiness and actionable
   findings. Clicking a finding's sample path opens the relevant image.
6. Open **Inspection baseline**, choose a core model, and train. The task runs in the
   background; you can continue reviewing samples. Select an inspection image to
   see the model input, anomaly map, overlay, and numerical score.

Refresh rescans changes made on disk or through another client. Export manifest
saves the same deterministic JSON inventory used by `indad-data inspect`.
Annotation coverage counts linked mask files; mask validity is checked separately.

One baseline is held per server. It survives page reloads but is replaced when a
new baseline starts and is lost on restart. The interface matches dataset contents
and model settings to the run; the API rejects predictions if dataset contents have
changed. Concurrent training/prediction requests receive a conflict response. A
failed training run includes its error in the run status.

## Agent workflow

The API documentation is at `/docs`; its machine-readable schema is at
`/openapi.json`. Dataset operations call the same Python functions as the CLI.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Server status and version |
| GET | `/api/datasets` | List local dataset names |
| POST | `/api/datasets` | Create a dataset with `{"name":"parts"}` |
| GET | `/api/datasets/{name}/report` | Inventory, checksums, findings and readiness |
| POST | `/api/datasets/{name}/import` | Multipart `files`, `split`, `label` |
| GET | `/api/datasets/{name}/image?path=...&thumbnail=true` | PNG preview of a dataset image or mask |
| GET | `/api/datasets/{name}/changes` | Edit and undo history, newest first |
| POST | `/api/datasets/{name}/samples/revise` | Relabel, move, or exclude a sample |
| POST | `/api/datasets/{name}/changes/{id}/undo` | Reverse a recorded edit |
| POST | `/api/runs` | Start background training |
| GET | `/api/runs/current` | Current run, or null before training |
| GET | `/api/runs/{id}` | Run status and dataset fingerprint |
| POST | `/api/runs/{id}/predict` | Inspect a test image with `{"path":"test/good/001.png"}` |

For example:

```shell
curl -X POST http://127.0.0.1:8000/api/datasets \
  -H 'Content-Type: application/json' -d '{"name":"parts"}'

curl -X POST http://127.0.0.1:8000/api/datasets/parts/import \
  -F 'split=train' -F 'label=good' \
  -F 'files=@captures/healthy_001.png' -F 'files=@captures/healthy_002.png'

curl http://127.0.0.1:8000/api/datasets/parts/report

# Add inspection images before starting a baseline.
curl -X POST http://127.0.0.1:8000/api/runs \
  -H 'Content-Type: application/json' \
  -d '{"dataset":"parts","method":"patchcore","device":"cpu"}'
```

Training returns HTTP 202 with a run ID. Poll the run until `status` is `ready` or
`failed`. Other states are `training`, `predicting`, and `stale`. Run options are
`patchcore`, `padim`, or `spade`, with `cpu` or `cuda`. The standard baseline uses
ResNet18, a 224 × 224 crop, and seed 0. The first run may download pretrained weights.

Prediction returns the sample path, run ID, image anomaly score, score-map minimum
and maximum, and PNG data URLs for `original` (the model input), `heatmap` and
`overlay`. Heatmap normalization is per image. Scores are not pass/fail decisions.

File and validation errors return a non-2xx response with a JSON `detail`. Unready
or stale baselines and busy inspection tasks return HTTP 409; oversized imports
return 413. Invalid typed request fields return 422. Inspect the report's readiness
flags before requesting a run.

## Sample curation requests

`POST /api/datasets/{name}/samples/revise` accepts:

```json
{
  "path": "test/scratch/001.png",
  "fingerprint": "fingerprint from the latest report",
  "reason": "Reviewed defect type",
  "split": "test",
  "label": "dent",
  "exclude": false
}
```

Set `exclude` to `true` to archive the sample and its mask. For exclusion, `split`
and `label` are ignored. Every edit requires a nonempty reason of up to 500
characters. The result includes a change ID, timestamp, reason and before/after
file paths with checksums.

Undo takes `{"fingerprint":"latest report fingerprint"}`. Restore is refused if
archived files changed or their original paths now contain other captures. Undo
successive edits to the same image in reverse order. Stale fingerprints return
HTTP 409. No operation overwrites existing images or masks.

The history endpoint marks previously undone edits with `undone: true`. The
archive/history directory must accompany dataset backups to preserve undo.

## Development

```shell
python -m pip install -e ".[web,dev]"
make lint
python -m pytest -m 'not export'
```

API tests use deterministic model backbones without downloading weights. Optional
browser tests require `playwright` and `python -m playwright install chromium`.
Static files live in `indad/web/`; dataset operations remain in `indad/workspace.py`.
