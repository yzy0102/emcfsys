# EMCFsys Development Guide

## 1. Project Overview

`EMCFsys` is a `napari` plugin for EM image analysis.  
At the UI level, it is organized as a group of dock widgets registered through the plugin manifest.  
At the implementation level, it now follows a clearer split:

- `src/emcfsys/_widget.py`
  UI widgets, parameter collection, napari layer updates
- `src/emcfsys/utils/*.py`
  reusable task logic, IO helpers, image processing helpers, viewer helpers
- `src/emcfsys/EMCellFound/*`
  segmentation model inference and training backend
- `src/emcfsys/EMCellFiner/*`
  super-resolution backend
- `src/emcfsys/PhenotypeAnalysis/*`
  phenotype feature calculation


## 2. Plugin Entry

Plugin registration is defined in [src/emcfsys/napari.yaml](/d:/napari_EMCF/EMCFsys/emcfsys/src/emcfsys/napari.yaml).

Main exported objects are collected in [src/emcfsys/__init__.py](/d:/napari_EMCF/EMCFsys/emcfsys/src/emcfsys/__init__.py).

Current widget entry points:

- `Image Resize`
- `EMCellFound(Segment model) - Inference`
- `EMCellFound(Segment model) - Training`
- `EMCellFiner(Super Resolution) - Single Image Inference`
- `EMCellFiner(Super Resolution) - Batch Inference`
- `DatasetConverter - Convert labelme 2 semantic segmentation masks`
- `PhenotypeAnalysis`


## 3. Current Feature Map

### 3.1 Image Resize

Widget:
[src/emcfsys/_widget.py](/d:/napari_EMCF/EMCFsys/emcfsys/src/emcfsys/_widget.py#L121)

Core helper:
[src/emcfsys/utils/image_resize_ops.py](/d:/napari_EMCF/EMCFsys/emcfsys/src/emcfsys/utils/image_resize_ops.py)

Current capability:

- select a napari image layer
- resize by absolute size or scale factor
- optional aspect-ratio maintenance
- algorithms:
  `Nearest Neighbor`, `Bilinear`, `Bicubic`, `Lanczos`
- write resized result back to napari as a new image layer

Development note:

- if you want to add new resize methods, modify `resize_array`
- if you want to add new UI controls, keep pure array logic inside `utils/image_resize_ops.py`


### 3.2 EMCellFound Inference

Widget:
[src/emcfsys/_widget.py](/d:/napari_EMCF/EMCFsys/emcfsys/src/emcfsys/_widget.py#L305)

Task module:
[src/emcfsys/utils/inference_tasks.py](/d:/napari_EMCF/EMCFsys/emcfsys/src/emcfsys/utils/inference_tasks.py)

Backend:
[src/emcfsys/EMCellFound/inference.py](/d:/napari_EMCF/EMCFsys/emcfsys/src/emcfsys/EMCellFound/inference.py)

Current capability:

- single-image full inference
- single-image sliding-window inference
- batch folder inference
- stop flag support in worker thread
- output normalization for napari labels
- folder output separation:
  raw label mask
  palette visualization mask
  blended original-image plus visualization result

Current output behavior for folder mode:

- `*_mask.png`
  raw class-index mask, suitable for further label workflows
- `*_mask_viz.png`
  colorized segmentation visualization
- `*_mask_stack.png`
  blended original image and visualization result

UI options currently supported:

- `Inference from folder`
- `Label mask output folder`
- `Save visualization mask`
- `Visualization output folder`
- `Save stacked image + visualization`
- `Stacked visualization output folder`

Development note:

- do not save raw label masks as RGB overlays
- keep label-safe saving in `save_label_mask`
- keep visualization-only saving in `save_palette_mask` and `save_stacked_visualization`
- if new export styles are needed, add them to `utils/inference_tasks.py` first


### 3.3 EMCellFound Training

Widget:
[src/emcfsys/_widget.py](/d:/napari_EMCF/EMCFsys/emcfsys/src/emcfsys/_widget.py#L496)

Task module:
[src/emcfsys/utils/training_tasks.py](/d:/napari_EMCF/EMCFsys/emcfsys/src/emcfsys/utils/training_tasks.py)

Backend:
[src/emcfsys/EMCellFound/train.py](/d:/napari_EMCF/EMCFsys/emcfsys/src/emcfsys/EMCellFound/train.py)

Current capability:

- train segmentation model from images and masks
- optional pretrained checkpoint input
- training log panel
- loss curve dock widget
- stop flag support with interrupted-model save path

Development note:

- widget now mainly handles form values, plot updates, and log display
- training loop integration should stay inside `run_training_task`


### 3.4 EMCellFiner Single Image Inference

Widget:
[src/emcfsys/_widget.py](/d:/napari_EMCF/EMCFsys/emcfsys/src/emcfsys/_widget.py#L658)

Task module:
[src/emcfsys/utils/emcellfiner_tasks.py](/d:/napari_EMCF/EMCFsys/emcfsys/src/emcfsys/utils/emcellfiner_tasks.py)

Backend:
[src/emcfsys/EMCellFiner/hat/models/inference_hat.py](/d:/napari_EMCF/EMCFsys/emcfsys/src/emcfsys/EMCellFiner/hat/models/inference_hat.py)

Current capability:

- select a napari image layer
- run super-resolution inference
- support manual model path or downloaded model
- add SR result back into napari


### 3.5 EMCellFiner Batch Inference

Widget:
[src/emcfsys/_widget.py](/d:/napari_EMCF/EMCFsys/emcfsys/src/emcfsys/_widget.py#L726)

Task module:
[src/emcfsys/utils/emcellfiner_tasks.py](/d:/napari_EMCF/EMCFsys/emcfsys/src/emcfsys/utils/emcellfiner_tasks.py)

Current capability:

- folder batch inference
- stop support
- optional resize-before-inference preprocessing
- resize controls:
  enable or disable
  resize factor
  resize algorithm

Resize behavior:

- default factor is `0.25`, which means downscale by 4
- default algorithm is `Bilinear`
- available algorithms match `Image Resize`
- minimum resized spatial size is clamped to `16`
  this avoids EMCellFiner window-padding failures on tiny inputs

Development note:

- preprocessing is handled by `_maybe_resize_input`
- if model input constraints change, update `MIN_EMCELLFINER_INPUT_SIZE`


### 3.6 LabelMe to Semantic Segmentation

Widget:
[src/emcfsys/_widget.py](/d:/napari_EMCF/EMCFsys/emcfsys/src/emcfsys/_widget.py#L847)

Current capability:

- convert LabelMe JSON annotations to segmentation masks
- auto-parse label mapping from multiple text styles
- save image, label, and visualization outputs

Development note:

- this logic is still inside `_widget.py`
- a future cleanup target is extracting it into `utils/labelme_service.py`


### 3.7 Phenotype Analysis

Widget:
[src/emcfsys/_widget.py](/d:/napari_EMCF/EMCFsys/emcfsys/src/emcfsys/_widget.py#L1012)

Backend:
[src/emcfsys/PhenotypeAnalysis/functions.py](/d:/napari_EMCF/EMCFsys/emcfsys/src/emcfsys/PhenotypeAnalysis/functions.py)

Current capability:

- compute selected phenotype features from image and label layers
- display results in table
- export CSV
- click row to locate object in napari
- create instance label layer for phenotype results


## 4. Utility Modules

Current reusable modules live in:
[src/emcfsys/utils](/d:/napari_EMCF/EMCFsys/emcfsys/src/emcfsys/utils)

Key files:

- `io_utils.py`
  file collection, path normalization, directory creation
- `viewer_ops.py`
  upsert image and labels layers in napari
- `image_resize_ops.py`
  pure resize array operations
- `inference_tasks.py`
  EMCellFound inference tasks and output saving
- `training_tasks.py`
  EMCellFound training orchestration
- `emcellfiner_tasks.py`
  EMCellFiner single and batch task logic

Compatibility note:

- root-level helper modules such as `_inference_tasks.py` still exist as thin compatibility wrappers
- new development should prefer imports from `emcfsys.utils.*`


## 5. UI Design Conventions Used So Far

These conventions are now used in `_widget.py` and should be kept consistent:

- widgets collect parameters, but heavy computation lives in `utils`
- long-running tasks run inside `thread_worker`
- napari layer writes use helper functions such as `upsert_image_layer` and `upsert_labels_layer`
- descriptive `Label` widgets are configured for auto-wrap and responsive resizing
- optional controls appear only when the corresponding checkbox is enabled


## 6. Test Layout

Relevant tests currently include:

- [tests/test_io_utils.py](/d:/napari_EMCF/EMCFsys/emcfsys/tests/test_io_utils.py)
- [tests/test_image_resize_ops.py](/d:/napari_EMCF/EMCFsys/emcfsys/tests/test_image_resize_ops.py)
- [tests/test_inference_tasks.py](/d:/napari_EMCF/EMCFsys/emcfsys/tests/test_inference_tasks.py)
- [tests/test_training_tasks.py](/d:/napari_EMCF/EMCFsys/emcfsys/tests/test_training_tasks.py)
- [tests/test_emcellfiner_tasks.py](/d:/napari_EMCF/EMCFsys/emcfsys/tests/test_emcellfiner_tasks.py)

Current limitation:

- the environment used during recent refactoring did not have `pytest` available, so validation was done mostly with syntax checks and targeted logic tests in code form


## 7. Recommended Next Refactor Targets

The main remaining large block inside `_widget.py` is:

- `LabelMe2Seg`
- `PhenotypeAnalysis`

Recommended next steps:

1. extract LabelMe conversion logic into `utils/labelme_service.py`
2. extract phenotype table and export support into a smaller helper or service module
3. keep `_widget.py` focused on UI assembly, signal wiring, and napari viewer updates


## 8. Practical Development Rules

When extending EMCFsys, prefer this pattern:

1. add or update UI controls in `_widget.py`
2. pass values into a request object
3. execute logic in `utils/*.py`
4. return plain arrays or simple result structures
5. update napari layers only in widget code

This keeps the plugin easier to test, easier to refactor, and safer to extend.
