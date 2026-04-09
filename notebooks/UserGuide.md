# EMCFsys User Guide

## 1. What EMCFsys Can Do

`EMCFsys` is a `napari` plugin for electron microscopy image analysis.

Current main functions:

- image resize
- EMCellFound segmentation inference
- EMCellFound segmentation training
- EMCellFiner super-resolution inference
- LabelMe annotation conversion
- phenotype analysis

## 2. How To Open The Plugin In napari

After installing and enabling the plugin in `napari`, open widgets from the plugin menu.

Available widget names:

- `Image Resize`
- `EMCellFound(Segment model) - Inference`
- `EMCellFound(Segment model) - Training`
- `EMCellFiner(Super Resolution) - Single Image Inference`
- `EMCellFiner(Super Resolution) - Batch Inference`
- `DatasetConverter - Convert labelme 2 semantic segmentation masks`
- `PhenotypeAnalysis`

## 3. Image Resize

Widget:
`Image Resize`

### What it does

Resize a napari image layer and add the resized result back into the viewer.

### Main options

- `Image`
  choose the image layer to resize
- `Resize Mode`
  `Absolute Size` or `Scale Factor`
- `Width`, `Height`
  used in absolute size mode
- `Scale X`, `Scale Y`
  used in scale factor mode
- `Maintain Aspect Ratio`
  automatically keep width-height ratio
- `Algorithm`
  `Nearest Neighbor`, `Bilinear`, `Bicubic`, `Lanczos`

### Typical workflow

1. Open an image in napari.
2. Open `Image Resize`.
3. Select the image layer.
4. Choose resize mode.
5. Enter target size or scale.
6. Click `Apply Resize!`

### Result

A new image layer named like `original_name_resized` will be added.

## 4. EMCellFound Segmentation Inference

Widget:
`EMCellFound(Segment model) - Inference`

### What it does

Run semantic segmentation inference with EMCellFound.

### Two inference modes

- `Run Full Inference`
  use the whole image directly
- `Run Slide Inference`
  use sliding-window inference for large images

### Main options

- `Model (.pt/.pth/.ptscript)`
  segmentation model weights
- `Image`
  selected napari image layer for single-image inference
- `Backbone`
- `Model`
- `num classes`
- `Device`
  `auto`, `cpu`, `cuda`
- `Image size to model`
- `Slide window size`

### Folder inference options

When `Inference from folder` is checked, additional controls appear:

- `Image folder`
- `Label mask output folder`
- `Save visualization mask`
- `Visualization output folder`
- `Save stacked image + visualization`
- `Stacked visualization output folder`

### Output file meanings

- `*_mask.png`
  raw label mask, suitable for later label operations
- `*_mask_viz.png`
  colorized segmentation visualization
- `*_mask_stack.png`
  blended original image and segmentation visualization

### Typical workflow for single image

1. Open the image in napari.
2. Open the inference widget.
3. Select model path.
4. Select image layer.
5. Set model parameters.
6. Click `Run Full Inference` or `Run Slide Inference`.

### Result for single image

A new labels layer is added to napari:

- full inference:
  `image_name_dl_mask`
- sliding-window inference:
  `image_name_slide_mask`

### Typical workflow for folder inference

1. Check `Inference from folder`.
2. Select `Image folder`.
3. Select `Label mask output folder`.
4. Optionally check `Save visualization mask`.
5. Optionally check `Save stacked image + visualization`.
6. Click `Run Full Inference` or `Run Slide Inference`.

### Notes

- if you want to use the results as labels later, use `*_mask.png`
- do not use visualization files as raw label masks

## 5. EMCellFound Segmentation Training

Widget:
`EMCellFound(Segment model) - Training`

### What it does

Train a segmentation model from image-mask pairs.

### Main options

- `Images folder`
- `Masks folder`
- `Save model as (.pth)`
- `Pretrained model (.pth)`
- `Backbone`
- `Model`
- `Learning rate`
- `Batch size`
- `Epochs`
- `Device`
- `Classes num`
- `Target size`
- `Ignore the index in mask`

### Typical workflow

1. Prepare training images folder.
2. Prepare corresponding masks folder.
3. Open training widget.
4. Fill in training parameters.
5. Click `Start Training`.

### During training

- log messages appear in the text panel
- loss curve is shown in a dock widget

### Stop training

Click `Stop Training`.

If interrupted during supported stages, the plugin attempts to save an interrupted model checkpoint.

## 6. EMCellFiner Single Image Inference

Widget:
`EMCellFiner(Super Resolution) - Single Image Inference`

### What it does

Run super-resolution inference on one image layer.

### Main options

- `Model (.pth)`
- `Image`
- `Scale`
- `Tile Size`
- `Device`

### Typical workflow

1. Open an image in napari.
2. Open the single-image EMCellFiner widget.
3. Select model path or leave blank for automatic model download if supported.
4. Select image layer.
5. Set scale and tile size.
6. Click `Run Inference`.

### Result

A new image layer named like `image_name_EMCFiner_SR` is added.

## 7. EMCellFiner Batch Inference

Widget:
`EMCellFiner(Super Resolution) - Batch Inference`

### What it does

Run super-resolution inference on all images in a folder.

### Main options

- `Model (.pth)`
- `Input Folder`
- `Save Folder`
- `Scale`
- `Tile Size`
- `Device`

### Optional resize-before-inference

When `Resize input before inference` is checked, additional controls appear:

- `Resize factor`
  default is `0.25`, which means downscale by 4
- `Resize algorithm`
  `Nearest Neighbor`, `Bilinear`, `Bicubic`, `Lanczos`

### Important note

For EMCellFiner model safety, resized input is automatically prevented from becoming too small.

### Typical workflow

1. Open the batch inference widget.
2. Select input folder.
3. Select save folder.
4. Set scale, tile size, and device.
5. Optionally enable resize-before-inference.
6. Click `Run Batch Inference`.

### Stop inference

Click `Stop Inference`.

## 8. Convert LabelMe To Segmentation Masks

Widget:
`DatasetConverter - Convert labelme 2 semantic segmentation masks`

### What it does

Convert LabelMe JSON annotations into semantic segmentation masks.

### Input requirements

- a folder of LabelMe `.json` files
- a label mapping text file

### Main options

- `Jsons labels folder`
- `Save mask Folder`
- `Label Name to Value (txt)`

### Label mapping examples

Supported styles include simple text mappings such as:

```txt
_background_ : 0
mito : 1
nu : 2
```

### Output folders

The tool creates subfolders such as:

- `image`
- `label`
- `vis`

## 9. Phenotype Analysis

Widget:
`PhenotypeAnalysis`

### What it does

Calculate selected phenotype features from an image and label layer.

### Main options

- `Image`
- `Label`
- feature checkboxes such as:
  `Area`
  `Perimeter`
  `Elongation`
  `Roundness`
  `Shape Complexity`
  `Electron Density`

### Typical workflow

1. Open image and label layers in napari.
2. Open `PhenotypeAnalysis`.
3. Select image layer and label layer.
4. Choose the features to calculate.
5. Click `Run Analysis`.

### Results

- a result table is shown in the widget
- an instance labels layer is added to napari
- clicking a row in the table moves the camera to that object

### Export

Click `Export to CSV` after analysis is complete.

## 10. Common Tips

### If inference result looks wrong in napari

- make sure the correct model and class count are selected
- for large images, try sliding-window inference
- for label workflows, use raw label masks rather than visualization images

### If folder inputs do not appear

- check whether the corresponding checkbox is enabled
- several UI sections are intentionally hidden until needed

### If EMCellFiner fails on very small images

- reduce aggressive downscaling
- keep resize-before-inference disabled if not needed
- the plugin already applies a minimum safe input size, but extremely unusual images may still need manual review

## 11. Recommended Usage Order

If you are new to the plugin, this is a good order to explore:

1. `Image Resize`
2. `EMCellFound(Segment model) - Inference`
3. `EMCellFiner(Super Resolution) - Single Image Inference`
4. `EMCellFiner(Super Resolution) - Batch Inference`
5. `DatasetConverter - Convert labelme 2 semantic segmentation masks`
6. `PhenotypeAnalysis`
7. `EMCellFound(Segment model) - Training`

## 12. Related Documents

For developers, see:
[notebooks/DevelopmentGuide.md](/d:/napari_EMCF/EMCFsys/emcfsys/notebooks/DevelopmentGuide.md)
