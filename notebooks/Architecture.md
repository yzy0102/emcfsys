# EMCFsys Plugin Architecture

This document summarizes the current plugin architecture around
`src/emcfsys/_widget.py`, clarifies responsibility boundaries, and proposes
a minimal refactor plan that preserves behavior while making later feature
development easier.

## 1. Current Loading Path

Napari loads this plugin through the manifest entry in
`pyproject.toml`, then resolves commands and widgets from
`src/emcfsys/napari.yaml`.

```text
napari
  -> pyproject.toml [napari.manifest]
  -> src/emcfsys/napari.yaml
  -> src/emcfsys/__init__.py or src/emcfsys/_widget.py
  -> widget class/function
  -> algorithm module
  -> napari viewer layer update / file output
```

## 2. Current Module Map

```text
src/emcfsys/
|
+- __init__.py
|  Exports plugin-facing objects
|
+- napari.yaml
|  Declares reader/writer/sample-data/widgets for napari
|
+- _reader.py / _writer.py / _sample_data.py
|  Template-style basic plugin capabilities
|
+- _widget.py
|  Main GUI entry point. Currently mixes:
|  - widget layout
|  - UI state handling
|  - background task startup
|  - direct file I/O
|  - model invocation
|  - result rendering back to napari
|
+- EMCellFound/
|  Segmentation training and inference backend
|  - inference.py
|  - train.py
|  - models/
|  - datasets/
|  - transforms/
|  - metrics/
|  - utils/
|
+- EMCellFiner/
|  Image restoration / super-resolution backend
|
+- PhenotypeAnalysis/
|  Morphology/statistics backend
|
+- dataset_transform/
   Dataset conversion utility code
```

## 3. Responsibility Split Inside `_widget.py`

### 3.1 `ImageResize`

Role:
- Small utility widget for resizing napari image layers.

Current responsibilities:
- Build UI controls
- Read selected image layer
- Compute resize parameters
- Execute resize logic directly
- Add/update output layer in napari

Assessment:
- This class is reasonably self-contained.
- It is acceptable to keep this one in `_widget.py` for now.

### 3.2 `DLInferenceContainer`

Role:
- Main segmentation inference GUI for EMCellFound.

Current responsibilities:
- Build inference UI
- Manage run/stop state
- Decide single-image vs folder mode
- Load model
- Launch worker thread
- Perform image folder iteration
- Save masks to disk
- Add predicted labels back to napari

Assessment:
- This widget mixes UI, orchestration, and batch file processing.
- It is one of the main pressure points in the current architecture.

### 3.3 `DLTrainingContainer`

Role:
- Main segmentation training GUI for EMCellFound.

Current responsibilities:
- Build training UI
- Manage stop state
- Launch training worker
- Pass callback into training loop
- Append log output
- Maintain matplotlib loss-curve widget
- Dock extra canvas into napari window

Assessment:
- The widget correctly acts as a training console, but currently owns both
  view logic and training-session orchestration logic.

### 3.4 `EMCellFinerSingleInferWidget`

Role:
- Single-image EMCellFiner inference from a napari image layer.

Current responsibilities:
- Build UI
- Read layer/model/device options
- Start worker
- Create HAT model
- Run restoration
- Add result image to viewer

Assessment:
- Similar pattern to segmentation inference, but narrower in scope.

### 3.5 `EMCellFinerBatchInferWidget`

Role:
- Folder-based EMCellFiner batch inference.

Current responsibilities:
- Build UI
- Discover input files
- Manage stop flag
- Start worker
- Load model
- Process folder in loop
- Save outputs to disk
- Update progress

Assessment:
- Another mixed UI + orchestration + batch I/O widget.

### 3.6 `LabelMe2Seg`

Role:
- Convert LabelMe JSON annotations into segmentation masks.

Current responsibilities:
- Build UI
- Parse label mapping file
- Iterate JSON files
- Decode image data
- Build semantic masks
- Save image/mask/visualization outputs

Assessment:
- Useful as a tool widget, but its conversion logic should live outside the
  widget class so it can be tested independently.

### 3.7 `PhenotypeAnalysis`

Role:
- Run phenotype feature extraction from image + label layers.

Current responsibilities:
- Build feature-selection UI
- Trigger analysis function
- Add/update instance layer
- Render result table
- Bind results to layer features
- Handle click-to-focus interaction
- Export CSV

Assessment:
- This widget is conceptually clean.
- The analysis math is already separated into
  `PhenotypeAnalysis/functions.py`, which is a good direction.

## 4. Cross-Cutting Patterns Already Present

The current codebase already has the beginnings of a healthy architecture:

- Napari manifest layer:
  plugin registration and widget exposure
- Widget layer:
  parameter collection and user interaction
- Backend layer:
  model inference/training and phenotype analysis functions
- Worker layer:
  long-running tasks use `thread_worker`

This means we do not need a rewrite. We mainly need to move shared logic out
of `_widget.py` and make widget classes thinner.

## 5. Main Architectural Problems

### 5.1 `_widget.py` is overloaded

It currently acts as:
- widget registry
- UI implementation
- task orchestration layer
- file-processing layer
- partial service layer

This makes navigation and later extension harder.

### 5.2 Similar patterns are duplicated

The following patterns recur in several widgets:
- device selection
- stop-flag management
- worker creation
- folder image discovery
- output naming
- viewer layer update

These should become reusable helpers/services.

### 5.3 Batch file processing lives in the UI layer

Folder iteration and save-path generation are not UI concerns. They should be
moved to dedicated service modules.

### 5.4 Testability is weak

Many operations are only reachable through widget methods.
This makes unit testing difficult and contributes to the current mismatch
between old template tests and the actual plugin behavior.

## 6. Minimal Refactor Goal

The first refactor should not change user-facing behavior.

The goal is only:
- make `_widget.py` smaller
- isolate reusable logic
- create clean extension points for new features
- improve testability

## 7. Minimal Refactor Plan

### Step 1. Keep widget class names and manifest entries unchanged

Do not rename the exported plugin widgets yet.

Reason:
- avoids breaking `napari.yaml`
- avoids changing plugin-facing API
- keeps this refactor low risk

### Step 2. Split non-UI logic out of `_widget.py`

Create small helper modules under `src/emcfsys/`:

```text
src/emcfsys/
|
+- _widget.py                  # only widget classes and signal wiring
+- _viewer_ops.py             # add/update napari layers
+- _io_utils.py               # folder scan, output path helpers
+- _labelme_service.py        # labelme conversion logic
+- _inference_tasks.py        # orchestration wrappers for EMCellFound
+- _training_tasks.py         # orchestration wrappers for training
+- _emcellfiner_tasks.py      # orchestration wrappers for EMCellFiner
```

These modules should not be “framework heavy”.
They should mostly hold plain Python functions.

### Step 3. Keep backend math/model code where it already lives

Do not move:
- `EMCellFound/inference.py`
- `EMCellFound/train.py`
- `PhenotypeAnalysis/functions.py`
- EMCellFiner model internals

Reason:
- these modules already represent backend responsibilities
- moving them now adds risk without much benefit

### Step 4. Introduce a thin orchestration layer

Each heavy widget should call a small task function instead of directly doing
everything itself.

Example shape:

```text
Widget
  -> collect parameters
  -> start worker
  -> call task/service function
  -> receive result
  -> update viewer
```

Instead of:

```text
Widget
  -> collect parameters
  -> do file scan
  -> load model
  -> run inference
  -> save files
  -> update viewer
  -> manage progress
```

### Step 5. Standardize shared helper responsibilities

Recommended helper ownership:

- `_viewer_ops.py`
  - add/update image layer
  - add/update labels layer
  - output naming helpers for layer names

- `_io_utils.py`
  - collect image files from folder
  - ensure output directories exist
  - normalize optional file/folder inputs

- `_inference_tasks.py`
  - single-image segmentation inference
  - folder segmentation inference
  - full-image vs sliding-window task wrappers

- `_training_tasks.py`
  - wrap train-loop invocation
  - package callback payloads for UI

- `_labelme_service.py`
  - parse label mapping text
  - convert one json
  - convert folder

### Step 6. Add tests for services before deep UI tests

Priority should be:
- service/helper tests
- backend function tests
- a few smoke tests for widget wiring

Do not start by writing many GUI-heavy tests.

## 8. Suggested Target Dependency Flow

```text
napari.yaml
  -> _widget.py
      -> _viewer_ops.py
      -> _io_utils.py
      -> _inference_tasks.py
      -> _training_tasks.py
      -> _labelme_service.py
      -> EMCellFound/*
      -> EMCellFiner/*
      -> PhenotypeAnalysis/functions.py
```

Rules:
- widgets may depend on services
- services may depend on backend modules
- backend modules should not depend on napari widgets

## 9. Recommended Refactor Order

Lowest-risk sequence:

1. Extract common folder/file helpers
2. Extract viewer add/update helpers
3. Extract LabelMe conversion service
4. Extract EMCellFound inference orchestration
5. Extract EMCellFiner batch/single orchestration
6. Extract training-session orchestration
7. Update tests to target the new helpers/services

## 10. What Should Stay in `_widget.py`

After the first refactor, `_widget.py` should mostly contain:

- widget class definitions
- widget layout declarations
- Qt or magicgui signal connections
- conversion from widget values to task parameters
- result callbacks that update the viewer

It should no longer be the main place for:

- folder walking
- output file naming logic
- labelme conversion loops
- long procedural inference/training logic

## 11. Immediate Next Coding Task

The best first code change is:

1. add `_io_utils.py`
2. add `_viewer_ops.py`
3. move the repeated folder-scan and napari layer update logic there
4. update `_widget.py` to call those helpers

This is the smallest refactor that:
- reduces duplication
- does not change plugin registration
- creates a reusable base for later features

