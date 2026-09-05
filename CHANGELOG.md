# Changelog

All notable changes to this project are documented here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- In-app sample relabeling, train/inspection reassignment, and exclusion with
  required reasons, mask preservation, persistent history, and undo/restore.
- Shared curation functions, HTTP endpoints, and `indad-data revise/changes/undo`
  commands with stale-inventory and overwrite protection.

### Fixed

- ONNX export now uses the supported `torch.onnx.export` API on current PyTorch,
  while retaining the older exporter for PyTorch 2.2–2.4.
- Pin the formatter version so local and CI formatting checks agree.

### Changed

- Run Chromium browser workflow tests in the Python 3.12 CI job and let both
  Python quality jobs finish independently when one fails.

## [0.4.0] - 2026-09-05

This release makes dataset preparation and operator workflows the focus of the
project, keeping SPADE, PaDiM and PatchCore as the three core baselines.

### Added

- Responsive browser operator console with dataset creation, persistent imports,
  searchable sample galleries, image/mask review and inspection readiness checks.
- `indad-web` server with a documented HTTP API and background baseline training.
- Shared dataset API and `indad-data create/import/inspect` commands with JSON
  inventories, SHA-256 checksums, content fingerprints and readiness exit codes.
- Checks for unreadable images, duplicate captures, split leakage, conflicting
  labels, missing or invalid masks, and mixed image dimensions.
- Operator workflow and HTTP API documentation, plus a console screenshot.
- Python packaging, runtime version access, and the `indad` benchmark CLI.
- Deterministic model, dataset, API, browser, persistence and export tests.
- CI coverage for Python 3.10 and 3.12, linting, ONNX export and package builds.

### Changed

- Replaced Streamlit with HTML/CSS/JavaScript served by FastAPI. The `web` extra
  replaces the previous demo dependencies; no frontend build step is required.
- The app and README now lead with dataset creation and review.
- Model exports load on demand so dataset commands do not import Torch or TIMM.
- Baselines run in server memory, survive page reloads, and reject predictions
  when their dataset fingerprint is stale.
- Model score maps follow input image dimensions, fitted tensors are persistent
  state-dict buffers, and PatchCore computes distances in bounded chunks.
- The benchmark CLI validates methods and exposes image size, device and seed.

### Fixed

- PaDiM handling of requested feature dimensions at or above the available count.
- SPADE neighbor counts exceeding the training gallery size.
- Repeated fitting accumulating previous model state.
- PatchCore reweighting stability and tests requesting unsupported ONNX export.
- Dataset labels and mask paths inferred through unsafe substring replacement.

### Data protection

- Imports validate the complete batch and reject existing filenames rather than
  overwriting original captures.
- HTTP image access is constrained to the configured dataset storage.
- Dataset archives are validated before extraction to reject path traversal and
  unsupported archive members.

[Unreleased]: https://github.com/rvorias/ind_knn_ad/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/rvorias/ind_knn_ad/releases/tag/v0.4.0
