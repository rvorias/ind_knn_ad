# Operator dataset workflow

Start with a collection of parts, not a model configuration. The browser operator
console follows **add images → check and review → train a baseline → inspect**.
SPADE, PaDiM and PatchCore remain the three core baselines.

```shell
python -m pip install -e ".[web]"
indad-web
# Open http://127.0.0.1:8000
```

Create a named dataset and upload images into one of four groups:

- **Healthy training images:** examples of acceptable parts. Start with at least
  two; meaningful coverage needs representative lighting, positioning and variation.
- **Healthy inspection images:** separate captures to check false alarms.
- **Defective inspection images:** assign a defect type such as `scratch` or `dent`.
- **Defect masks:** optional binary PNGs, white (255) for defects and black (0)
  elsewhere. Match the original image dimensions and name them
  `<image-stem>_mask.png` under the same defect type.

Imports persist in the selected storage folder. Existing filenames are rejected,
not replaced. Every file in a batch is decoded before any is published. Original
image bytes are preserved. Use separate filenames for separate captures.

Review the health findings and browse images and masks by filename. Readiness is
specific to the next operation:

| Operation | Requirements |
| --- | --- |
| Train baseline | At least two readable healthy training images; no invalid training labels |
| Visual inspection | Training ready and readable inspection images |
| Image ROC AUC | Inspection ready, both healthy and defective labels, no detected split leakage or conflicting labels |
| Pixel ROC AUC | Image evaluation ready and matching, nonempty binary defect masks |

Exact decoded-image duplicates across training and inspection block evaluation
readiness. Duplicate images within one split and mixed dimensions produce warnings.
This does not detect near-duplicate frames or shared production batches: keep
related captures in the same split yourself. Readiness checks are structural checks,
not evidence of adequate sample size or production accuracy.

Train the default PatchCore baseline or choose one of the other two core models in
the Inspection baseline tab. The first run may download pretrained ResNet18 weights.
Dataset contents are checked against the baseline fingerprint before and after
inspection. Changed data requires retraining. The console also checks that the
selected model settings match the run. Images are resized and center-cropped to 224 × 224;
review whether the crop retains your defects. Heatmaps are scaled per image and
anomaly scores are not calibrated pass/fail decisions.

The baseline stays in server memory and survives page reloads. One baseline is held
at a time; training a new one replaces it. Restarting the server requires retraining.
Training runs in the background, and operators can continue reviewing samples.
Datasets stay on disk. Use `indad-web --datasets /path/to/datasets` to select storage.

The workspace reviews predictions; numerical benchmark evaluation remains available
through `indad METHOD --dataset NAME` for datasets with pixel masks under `./datasets`.
The browser UI replaces the previous Streamlit demo. Its standard HTTP API also
supports agent clients; see the [API guide](operator_api.md).

## Automation uses the same dataset operations

```shell
indad-data create line_1_bottles --root datasets
indad-data import datasets/line_1_bottles captures/healthy/*.png --split train
indad-data import datasets/line_1_bottles captures/check_good/*.png --split test
indad-data import datasets/line_1_bottles captures/scratched/*.png --split test --label scratch
indad-data import datasets/line_1_bottles captures/masks/*_mask.png --split ground_truth --label scratch
indad-data inspect datasets/line_1_bottles --require inspection > manifest.json
```

Without installing the console entry point, use `python -m indad.dataset_cli`.
Commands are non-interactive. Successful create/import commands return JSON;
inspection always returns its JSON report for an existing readable dataset.
`--require` accepts `training`, `inspection`, `image_evaluation` or
`pixel_evaluation` and returns exit code 1 when that workflow is not ready. Without
`--require`, inspection returns 0 even when findings need attention. Invalid command
arguments and file-operation errors are reported on stderr with a nonzero exit code.

The version-1 JSON report contains split/label counts, individual findings with
stable codes and relative paths, readiness flags, and a sorted sample inventory.
Each readable sample includes dimensions, a SHA-256 file checksum and its mask path
and checksum when available. The fingerprint identifies that inventory, including
referenced mask contents, without depending on the absolute dataset location.
Exporting a report does not lock or copy the data; archive the dataset alongside it
when you need a reproducible snapshot. Inspection does not write files or download
weights.

Python callers use the same API:

```python
from indad.workspace import create_dataset, import_images, inspect_dataset

root = create_dataset("datasets", "parts")
import_images(root, [("part_001.png", image_bytes)], split="train", label="good")
report = inspect_dataset(root)
```

The on-disk layout remains compatible with `MVTecDataset`:

```text
parts/
  train/good/*.png
  test/good/*.png
  test/scratch/*.png
  ground_truth/scratch/*_mask.png
```

## Where this project should invest next

Prioritize correcting labels and excluding bad captures from the UI, capture-group
splits, and versioned dataset snapshots with provenance. Then add comparable saved
inspection runs and threshold review with false-positive and missed-defect examples.
Keep synthetic data explicitly traceable if generation is introduced. Expand the
model set only when a demonstrated operator workflow needs it.

Keep dataset operations independent of the browser UI and model implementations. UI and
agent commands should share validation and file operations; changes must preserve
original captures and make decisions reviewable.
