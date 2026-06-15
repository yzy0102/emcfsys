import json
import numpy as np
from pathlib import Path
from PIL import Image
from qtpy.QtCore import QEventLoop, QTimer
from qtpy.QtWidgets import QTextEdit

from emcfsys._widget import (
    ClassificationInferenceContainer,
    ClassificationTrainingContainer,
    DLInferenceContainer,
    DLTrainingContainer,
    EMCellFinerBatchInferWidget,
    EMCellFinerSingleInferWidget,
    ExampleQWidget,
    ImageResize,
    ImageThreshold,
    InstanceSegmentationInferenceContainer,
    InstanceSegmentationTrainingContainer,
    LabelMe2COCOInstance,
    LabelMe2Seg,
    ModelManagerContainer,
    DatasetValidatorContainer,
    threshold_autogenerate_widget,
    threshold_magic_widget,
)
from tests.test_instance_segmentation import _write_tiny_coco_dataset


class FakeWindow:
    def __init__(self):
        self.docks = []

    def add_dock_widget(self, widget, name=None, area=None):
        self.docks.append({"widget": widget, "name": name, "area": area})
        return widget


class FakeViewer:
    def __init__(self):
        self.window = FakeWindow()


def _wait_for_condition(condition, timeout_ms=5000):
    loop = QEventLoop()
    deadline = QTimer()
    deadline.setSingleShot(True)
    poller = QTimer()

    def _poll():
        if condition():
            poller.stop()
            deadline.stop()
            loop.quit()

    poller.timeout.connect(_poll)
    deadline.timeout.connect(loop.quit)
    poller.start(20)
    deadline.start(timeout_ms)
    _poll()
    if not condition():
        loop.exec_()
    poller.stop()
    deadline.stop()
    assert condition()


def _write_labelme_instance_widget_sample(root, name="img"):
    root.mkdir(parents=True, exist_ok=True)
    image_path = root / f"{name}.png"
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_path)
    payload = {
        "version": "5.0.0",
        "imagePath": image_path.name,
        "imageHeight": 16,
        "imageWidth": 16,
        "shapes": [
            {
                "label": "Mito",
                "points": [[2, 3], [10, 3], [10, 9], [2, 9]],
                "shape_type": "polygon",
            }
        ],
    }
    (root / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_threshold_autogenerate_widget():
    # because our "widget" is a pure function, we can call it and
    # test it independently of napari
    im_data = np.random.random((100, 100))
    thresholded = threshold_autogenerate_widget(im_data, 0.5)
    assert thresholded.shape == im_data.shape
    # etc.


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# you don't need to import it, as long as napari is installed
# in your testing environment
def test_threshold_magic_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))

    # our widget will be a MagicFactory or FunctionGui instance
    my_widget = threshold_magic_widget()

    # if we "call" this object, it'll execute our function
    thresholded = my_widget(viewer.layers[0], 0.5)
    assert thresholded.shape == layer.data.shape
    # etc.


def test_image_threshold_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))
    my_widget = ImageThreshold(viewer)

    # because we saved our widgets as attributes of the container
    # we can set their values without having to "interact" with the viewer
    my_widget._image_layer_combo.value = layer
    my_widget._threshold_slider.value = 0.5

    # this allows us to run our functions directly and ensure
    # correct results
    my_widget._threshold_im()
    assert len(viewer.layers) == 2


# capsys is a pytest fixture that captures stdout and stderr output streams
def test_example_q_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    my_widget = ExampleQWidget(viewer)

    # call our widget method
    my_widget._on_click()

    # read captured output and check that it's as we expected
    captured = capsys.readouterr()
    assert captured.out == "napari has 1 layers\n"


def test_image_resize_widget(make_napari_viewer):
    """Test the ImageResize widget."""
    viewer = make_napari_viewer()
    # Create a test image
    test_image = np.random.random((100, 100))
    layer = viewer.add_image(test_image)

    # Create the resize widget
    resize_widget = ImageResize(viewer)

    # Set the image layer
    resize_widget._image_layer_combo.value = layer

    # Test absolute size mode
    resize_widget._mode_combo.value = "Absolute Size"
    resize_widget._width_spinbox.value = 50
    resize_widget._height_spinbox.value = 50
    resize_widget._algorithm_combo.value = "Bilinear"

    # Apply resize
    resize_widget._resize_image()

    # Check that a new layer was created
    assert len(viewer.layers) == 2
    assert "test_image_resized" in viewer.layers

    # Check the output shape
    resized_layer = viewer.layers["test_image_resized"]
    assert resized_layer.data.shape == (50, 50)


def test_image_resize_scale_mode(make_napari_viewer):
    """Test the ImageResize widget with scale factor mode."""
    viewer = make_napari_viewer()
    # Create a test image
    test_image = np.random.random((100, 100))
    layer = viewer.add_image(test_image)

    # Create the resize widget
    resize_widget = ImageResize(viewer)

    # Set the image layer
    resize_widget._image_layer_combo.value = layer

    # Test scale factor mode
    resize_widget._mode_combo.value = "Scale Factor"
    resize_widget._scale_x_spinbox.value = 0.5
    resize_widget._scale_y_spinbox.value = 0.5
    resize_widget._algorithm_combo.value = "Nearest Neighbor"

    # Apply resize
    resize_widget._resize_image()

    # Check that a new layer was created
    assert len(viewer.layers) == 2

    # Check the output shape (should be 50x50)
    resized_layer = viewer.layers["test_image_resized"]
    assert resized_layer.data.shape == (50, 50)


def test_image_resize_algorithms(make_napari_viewer):
    """Test different resize algorithms."""
    viewer = make_napari_viewer()
    test_image = np.random.random((100, 100))
    layer = viewer.add_image(test_image)

    algorithms = ["Nearest Neighbor", "Bilinear", "Bicubic", "Lanczos"]

    for algorithm in algorithms:
        # Clear previous layers except the original
        while len(viewer.layers) > 1:
            viewer.layers.pop()

        resize_widget = ImageResize(viewer)
        resize_widget._image_layer_combo.value = layer
        resize_widget._mode_combo.value = "Absolute Size"
        resize_widget._width_spinbox.value = 50
        resize_widget._height_spinbox.value = 50
        resize_widget._algorithm_combo.value = algorithm

        # Apply resize
        resize_widget._resize_image()

        # Check that resize was successful
        assert len(viewer.layers) == 2
        resized_layer = viewer.layers["test_image_resized"]
        assert resized_layer.data.shape == (50, 50)


def test_instance_segmentation_widgets_construct(make_napari_viewer):
    viewer = make_napari_viewer()

    training_widget = InstanceSegmentationTrainingContainer(viewer)
    inference_widget = InstanceSegmentationInferenceContainer(viewer)
    converter_widget = LabelMe2COCOInstance(viewer)

    assert training_widget.model_name.value == "rtm_instance"
    assert "yolact_instance" in training_widget.model_name.choices
    assert "mask_rcnn_instance" in training_widget.model_name.choices
    assert "condinst_instance" in training_widget.model_name.choices
    assert "solov2_instance" in training_widget.model_name.choices
    assert "mask2former_instance" in training_widget.model_name.choices
    assert training_widget.use_separate_eval_sets.value is False
    assert training_widget.val_image_dir.native.isHidden()
    assert training_widget.val_annotation_path.native.isHidden()
    assert training_widget.test_image_dir.native.isHidden()
    assert training_widget.test_annotation_path.native.isHidden()
    assert training_widget.boundary_loss_weight.native.isHidden()
    assert training_widget.focal_mask_loss_weight.native.isHidden()
    assert training_widget.tversky_loss_weight.native.isHidden()
    assert training_widget.use_data_augmentation.value is True
    assert training_widget.aug_horizontal_flip_prob.value == 0.5
    assert training_widget.aug_vertical_flip_prob.value == 0.5
    assert training_widget.aug_rotate90_prob.value == 0.5
    assert training_widget.aug_brightness.value == 0.15
    assert training_widget.aug_contrast.value == 0.15
    assert training_widget.aug_random_crop_prob.value == 0.3
    assert not training_widget.aug_horizontal_flip_prob.native.isHidden()
    assert not training_widget.aug_random_crop_prob.native.isHidden()
    assert not hasattr(training_widget, "_log_widget")
    assert training_widget._log_text is not None
    assert inference_widget.model_name.value == "rtm_instance"
    assert "rtm_instance_tiny" in inference_widget.model_name.choices
    assert "mask_rcnn_instance" in inference_widget.model_name.choices
    assert "mask2former_instance" in inference_widget.model_name.choices
    assert inference_widget.mask_threshold.value == 0.5
    assert converter_widget.copy_images.value is True
    assert converter_widget.split_dataset.value is False
    assert converter_widget.train_ratio.value == 0.8
    assert converter_widget.val_ratio.value == 0.1
    assert converter_widget.test_ratio.value == 0.1
    assert converter_widget._cancel_button.enabled is False


def test_labelme_instance_converter_widget_runs_in_worker_and_can_cancel(make_napari_viewer, tmp_path):
    labelme_dir = tmp_path / "labelme"
    _write_labelme_instance_widget_sample(labelme_dir)

    viewer = make_napari_viewer()
    widget = LabelMe2COCOInstance(viewer)
    widget.json_dir.value = labelme_dir
    widget.output_json.value = tmp_path / "coco" / "instances.json"
    widget.copy_images.value = True
    widget.image_output_dir.value = tmp_path / "coco" / "images"

    widget._convert_labelme_to_coco()
    assert widget._worker is not None
    assert widget._run_button.enabled is False
    assert widget._cancel_button.enabled is True
    _wait_for_condition(lambda: widget._worker is None)

    assert "converted:" in widget.info.value
    assert "Images: 1" in widget.info.value
    assert widget._run_button.enabled is True
    assert widget._cancel_button.enabled is False
    assert (tmp_path / "coco" / "instances.json").exists()

    widget._worker = object()
    widget._cancel_button.enabled = True
    widget._cancel_conversion()

    assert widget._stop_conversion is True
    assert widget._cancel_button.enabled is False
    assert "Cancel requested" in widget.info.value
    widget._worker = None


def test_labelme_semantic_converter_widget_checks_previews_and_converts(make_napari_viewer, tmp_path):
    labelme_dir = tmp_path / "labelme"
    labelme_dir.mkdir()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(image).save(labelme_dir / "img.png")
    payload = {
        "version": "5.0.0",
        "flags": {},
        "shapes": [
            {
                "label": "mito",
                "points": [[1, 1], [6, 1], [6, 6], [1, 6]],
                "shape_type": "polygon",
                "flags": {},
            }
        ],
        "imagePath": "img.png",
        "imageData": None,
        "imageHeight": 8,
        "imageWidth": 8,
    }
    (labelme_dir / "img.json").write_text(json.dumps(payload), encoding="utf-8")

    viewer = make_napari_viewer()
    widget = LabelMe2Seg(viewer)
    assert widget._cancel_button.enabled is False
    widget.json_path.value = labelme_dir
    widget.output_dir.value = tmp_path / "converted"
    widget.label_map_path.value = tmp_path / "label_map.json"
    widget.split_dataset.value = True
    widget.save_rgb_mask.value = True
    widget.save_overlay.value = True

    widget._infer_label_map()
    assert widget.label_map_table.native.rowCount() >= 3

    widget._save_label_map()
    assert (tmp_path / "label_map.json").exists()

    widget._check_labelme_folder()
    assert "LabelMe semantic folder check: OK" in widget.info.value

    widget._preview_labelme_folder()
    assert "LabelMe semantic preview image" in viewer.layers
    assert "LabelMe semantic preview mask" in viewer.layers
    assert "LabelMe semantic preview overlay" in viewer.layers

    widget._convert_labelme_json_to_mask()
    assert widget._worker is not None
    assert widget._run_button.enabled is False
    assert widget._cancel_button.enabled is True
    _wait_for_condition(lambda: widget._worker is None)

    assert "success=1" in widget.info.value
    assert "converted:" in widget.info.value
    assert widget._run_button.enabled is True
    assert (tmp_path / "converted" / "split.json").exists()
    assert len(list((tmp_path / "converted" / "images").rglob("*.tif"))) == 1
    assert len(list((tmp_path / "converted" / "masks").rglob("*.png"))) == 1
    assert len(list((tmp_path / "converted" / "rgb_masks").rglob("*.png"))) == 1
    assert len(list((tmp_path / "converted" / "overlay").rglob("*.png"))) == 1


def test_labelme_semantic_converter_cancel_button_sets_stop_flag(make_napari_viewer, tmp_path):
    labelme_dir = tmp_path / "labelme"
    labelme_dir.mkdir()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(image).save(labelme_dir / "img.png")
    payload = {
        "version": "5.0.0",
        "flags": {},
        "shapes": [
            {
                "label": "mito",
                "points": [[1, 1], [6, 1], [6, 6], [1, 6]],
                "shape_type": "polygon",
                "flags": {},
            }
        ],
        "imagePath": "img.png",
        "imageData": None,
        "imageHeight": 8,
        "imageWidth": 8,
    }
    (labelme_dir / "img.json").write_text(json.dumps(payload), encoding="utf-8")

    viewer = make_napari_viewer()
    widget = LabelMe2Seg(viewer)
    widget.json_path.value = labelme_dir
    widget.output_dir.value = tmp_path / "converted"
    widget._infer_label_map()
    widget._convert_labelme_json_to_mask()

    assert widget._worker is not None
    widget._cancel_conversion()

    assert widget._stop_conversion is True
    assert widget._cancel_button.enabled is False
    assert "Cancel requested" in widget.info.value
    _wait_for_condition(lambda: widget._worker is None)


def test_semantic_segmentation_training_widget_builds_advanced_loss_request(tmp_path):
    training_widget = DLTrainingContainer(None)

    training_widget.images_dir.value = tmp_path / "images"
    training_widget.masks_dir.value = tmp_path / "masks"
    training_widget.save_path.value = tmp_path / "save"
    assert training_widget.dice_loss_weight.native.isHidden()
    assert training_widget.focal_loss_weight.native.isHidden()
    assert training_widget.tversky_loss_weight.native.isHidden()
    assert training_widget.boundary_loss_weight.native.isHidden()
    assert training_widget.lovasz_loss_weight.native.isHidden()
    assert training_widget.ohem_ce_loss_weight.native.isHidden()

    training_widget.use_advanced_losses.value = True
    training_widget.dice_loss_weight.value = 0.5
    training_widget.focal_loss_weight.value = 0.6
    training_widget.tversky_loss_weight.value = 0.7
    training_widget.boundary_loss_weight.value = 0.8
    training_widget.lovasz_loss_weight.value = 0.9
    training_widget.ohem_ce_loss_weight.value = 1.0
    request = training_widget._build_training_request()

    assert not training_widget.dice_loss_weight.native.isHidden()
    assert request.use_advanced_losses is True
    assert request.dice_loss_weight == 0.5
    assert request.focal_loss_weight == 0.6
    assert request.tversky_loss_weight == 0.7
    assert request.boundary_loss_weight == 0.8
    assert request.lovasz_loss_weight == 0.9
    assert request.ohem_ce_loss_weight == 1.0


def test_semantic_segmentation_training_widget_applies_preset():
    training_widget = DLTrainingContainer(None)

    training_widget.training_preset.value = "Small Organelle"

    assert training_widget.use_advanced_losses.value is True
    assert training_widget.target_size.value == 768
    assert training_widget.batch_size.value == 4
    assert training_widget.epochs.value == 150
    assert training_widget.focal_loss_weight.value == 1.0
    assert training_widget.tversky_loss_weight.value == 1.0
    assert training_widget.boundary_loss_weight.value == 0.3
    assert not training_widget.focal_loss_weight.native.isHidden()


def test_semantic_segmentation_training_widget_saves_and_loads_config(tmp_path):
    training_widget = DLTrainingContainer(None)
    config_path = tmp_path / "semantic_config.json"

    training_widget.images_dir.value = tmp_path / "images"
    training_widget.masks_dir.value = tmp_path / "masks"
    training_widget.save_path.value = tmp_path / "save"
    training_widget.config_path.value = config_path
    training_widget.backbone_name.value = "resnet34"
    training_widget.model_name.value = "unet"
    training_widget.use_advanced_losses.value = True
    training_widget.focal_loss_weight.value = 0.6
    training_widget._save_config()

    training_widget.model_name.value = "deeplabv3plus"
    training_widget.use_advanced_losses.value = False
    training_widget.focal_loss_weight.value = 0.0
    training_widget._load_config()

    assert config_path.exists()
    assert training_widget.model_name.value == "unet"
    assert training_widget.use_advanced_losses.value is True
    assert training_widget.focal_loss_weight.value == 0.6
    assert not training_widget.focal_loss_weight.native.isHidden()


def test_semantic_segmentation_inference_widget_loads_training_config(tmp_path):
    training_widget = DLTrainingContainer(None)
    config_path = tmp_path / "semantic_config.json"
    training_widget.config_path.value = config_path
    training_widget.images_dir.value = tmp_path / "images"
    training_widget.masks_dir.value = tmp_path / "masks"
    training_widget.save_path.value = tmp_path / "save"
    training_widget.backbone_name.value = "resnet34"
    training_widget.model_name.value = "unet"
    training_widget.classes_num.value = 4
    training_widget.target_size.value = 256
    training_widget.device.value = "cpu"
    training_widget._save_config()

    inference_widget = DLInferenceContainer(None)
    inference_widget.config_path.value = config_path
    inference_widget.backbone_name.value = "emcellfound_vit_base"
    inference_widget.model_name.value = "deeplabv3plus"
    inference_widget.num_classes.value = 2
    inference_widget.img_size.value = 512
    inference_widget.device.value = "auto"
    inference_widget._load_config()

    assert inference_widget.backbone_name.value == "resnet34"
    assert inference_widget.model_name.value == "unet"
    assert inference_widget.num_classes.value == 4
    assert inference_widget.img_size.value == 256
    assert inference_widget.device.value == "cpu"


def test_instance_segmentation_training_widget_builds_advanced_loss_request(tmp_path):
    training_widget = InstanceSegmentationTrainingContainer(None)

    training_widget.image_dir.value = tmp_path / "train_images"
    training_widget.annotation_path.value = tmp_path / "train.json"
    training_widget.save_path.value = tmp_path / "save"
    training_widget.use_advanced_mask_losses.value = True
    training_widget.boundary_loss_weight.value = 0.2
    training_widget.focal_mask_loss_weight.value = 0.3
    training_widget.tversky_loss_weight.value = 0.4
    request = training_widget._build_training_request()

    assert not training_widget.boundary_loss_weight.native.isHidden()
    assert request.use_advanced_mask_losses is True
    assert request.boundary_loss_weight == 0.2
    assert request.focal_mask_loss_weight == 0.3
    assert request.tversky_loss_weight == 0.4


def test_instance_segmentation_training_widget_applies_preset():
    training_widget = InstanceSegmentationTrainingContainer(None)

    training_widget.training_preset.value = "Boundary Sensitive"

    assert training_widget.use_advanced_mask_losses.value is True
    assert training_widget.img_size.value == 768
    assert training_widget.batch_size.value == 1
    assert training_widget.epochs.value == 100
    assert training_widget.boundary_loss_weight.value == 1.0
    assert training_widget.focal_mask_loss_weight.value == 0.3
    assert training_widget.tversky_loss_weight.value == 0.7
    assert not training_widget.boundary_loss_weight.native.isHidden()


def test_instance_segmentation_training_widget_saves_and_loads_config(tmp_path):
    training_widget = InstanceSegmentationTrainingContainer(None)
    config_path = tmp_path / "instance_config.json"

    training_widget.image_dir.value = tmp_path / "train_images"
    training_widget.annotation_path.value = tmp_path / "train.json"
    training_widget.save_path.value = tmp_path / "save"
    training_widget.config_path.value = config_path
    training_widget.model_name.value = "mask_rcnn_instance"
    training_widget.use_advanced_mask_losses.value = True
    training_widget.boundary_loss_weight.value = 0.2
    training_widget.use_data_augmentation.value = True
    training_widget.aug_horizontal_flip_prob.value = 0.8
    training_widget.aug_random_crop_prob.value = 0.4
    training_widget.use_separate_eval_sets.value = True
    training_widget.val_annotation_path.value = tmp_path / "val.json"
    training_widget._save_config()

    training_widget.model_name.value = "rtm_instance"
    training_widget.use_advanced_mask_losses.value = False
    training_widget.boundary_loss_weight.value = 0.0
    training_widget.use_data_augmentation.value = False
    training_widget.aug_horizontal_flip_prob.value = 0.0
    training_widget.aug_random_crop_prob.value = 0.0
    training_widget.use_separate_eval_sets.value = False
    training_widget.val_annotation_path.value = None
    training_widget._load_config()

    assert config_path.exists()
    assert training_widget.model_name.value == "mask_rcnn_instance"
    assert training_widget.use_advanced_mask_losses.value is True
    assert training_widget.boundary_loss_weight.value == 0.2
    assert training_widget.use_data_augmentation.value is True
    assert training_widget.aug_horizontal_flip_prob.value == 0.8
    assert training_widget.aug_random_crop_prob.value == 0.4
    assert not training_widget.aug_horizontal_flip_prob.native.isHidden()
    assert training_widget.use_separate_eval_sets.value is True
    assert not training_widget.val_annotation_path.native.isHidden()


def test_instance_segmentation_training_widget_builds_eval_dataset_request(tmp_path):
    training_widget = InstanceSegmentationTrainingContainer(None)

    training_widget.image_dir.value = tmp_path / "train_images"
    training_widget.annotation_path.value = tmp_path / "train.json"
    training_widget.save_path.value = tmp_path / "save"
    training_widget.use_separate_eval_sets.value = True
    training_widget.val_image_dir.value = tmp_path / "val_images"
    training_widget.val_annotation_path.value = tmp_path / "val.json"
    training_widget.test_image_dir.value = tmp_path / "test_images"
    training_widget.test_annotation_path.value = tmp_path / "test.json"
    request = training_widget._build_training_request()

    assert training_widget.val_split.native.isHidden()
    assert not training_widget.val_image_dir.native.isHidden()
    assert request.val_split == 0.0
    assert request.val_image_dir == str(tmp_path / "val_images")
    assert request.val_annotation_path == str(tmp_path / "val.json")
    assert request.test_image_dir == str(tmp_path / "test_images")
    assert request.test_annotation_path == str(tmp_path / "test.json")


def test_instance_segmentation_training_widget_keeps_val_split_for_single_coco(tmp_path):
    training_widget = InstanceSegmentationTrainingContainer(None)

    training_widget.image_dir.value = tmp_path / "images"
    training_widget.annotation_path.value = tmp_path / "instances.json"
    training_widget.save_path.value = tmp_path / "save"
    training_widget.use_separate_eval_sets.value = False
    training_widget.val_split.value = 0.25
    training_widget.use_data_augmentation.value = True
    training_widget.aug_rotate90_prob.value = 0.6
    training_widget.aug_contrast.value = 0.2
    request = training_widget._build_training_request()

    assert request.val_split == 0.25
    assert request.val_image_dir is None
    assert request.use_data_augmentation is True
    assert request.aug_rotate90_prob == 0.6
    assert request.aug_contrast == 0.2
    assert request.val_annotation_path is None
    assert not training_widget.val_split.native.isHidden()


def test_instance_segmentation_training_widget_allows_json_only_eval_sets(tmp_path):
    training_widget = InstanceSegmentationTrainingContainer(None)

    training_widget.image_dir.value = tmp_path / "train_images"
    training_widget.annotation_path.value = tmp_path / "train.json"
    training_widget.save_path.value = tmp_path / "save"
    training_widget.use_separate_eval_sets.value = True
    training_widget.val_annotation_path.value = tmp_path / "val.json"
    training_widget.test_annotation_path.value = tmp_path / "test.json"
    request = training_widget._build_training_request()

    assert request.val_image_dir is None
    assert request.val_annotation_path == str(tmp_path / "val.json")
    assert request.test_image_dir is None
    assert request.test_annotation_path == str(tmp_path / "test.json")


def test_instance_segmentation_training_widget_creates_log_dock():
    viewer = FakeViewer()
    training_widget = InstanceSegmentationTrainingContainer(viewer)

    log_docks = [
        dock
        for dock in viewer.window.docks
        if dock["name"] == "Instance Segmentation Training Log"
    ]

    assert len(log_docks) == 1
    assert log_docks[0]["area"] == "bottom"
    assert isinstance(log_docks[0]["widget"], QTextEdit)

    training_widget._log("hello instance log")
    assert "hello instance log" in training_widget._log_text.toPlainText()


def test_core_task_widgets_create_log_docks():
    viewer = FakeViewer()
    widgets = [
        (DLTrainingContainer(viewer), "Semantic Segmentation Training Log"),
        (DLInferenceContainer(viewer), "Semantic Segmentation Inference Log"),
        (ClassificationTrainingContainer(viewer), "Classification Training Log"),
        (ClassificationInferenceContainer(viewer), "Classification Inference Log"),
        (EMCellFinerSingleInferWidget(viewer), "Super Resolution Single Inference Log"),
        (EMCellFinerBatchInferWidget(viewer), "Super Resolution Batch Inference Log"),
    ]

    for widget, dock_name in widgets:
        log_docks = [dock for dock in viewer.window.docks if dock["name"] == dock_name]
        assert len(log_docks) == 1
        assert log_docks[0]["area"] == "bottom"
        assert isinstance(log_docks[0]["widget"], QTextEdit)

        message = f"hello {dock_name}"
        widget._log(message)
        assert message in widget._log_text.toPlainText()


def test_instance_segmentation_widgets_are_in_manifest():
    manifest = Path("src/emcfsys/napari.yaml").read_text(encoding="utf-8")

    assert "emcfsys.instance_segmentation.training" in manifest
    assert "InstanceSegmentationTrainingContainer" in manifest
    assert "emcfsys.instance_segmentation.inference" in manifest
    assert "InstanceSegmentationInferenceContainer" in manifest
    assert "emcfsys.instance_segmentation.dataset_inspector" not in manifest
    assert "COCOInstanceDatasetInspector" not in manifest
    assert "emcfsys.labelme_2_coco_instance" in manifest
    assert "LabelMe2COCOInstance" in manifest
    assert "emcfsys.labelme_2_semseg" in manifest
    assert "LabelMe2Seg" in manifest


def test_dataset_validator_widget_checks_exports_and_previews_semantic(tmp_path):
    viewer = FakeViewer()
    viewer.layers = {}
    viewer.add_shapes = lambda *args, **kwargs: None
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(images / "a.tif")
    Image.fromarray(np.ones((8, 8), dtype=np.uint8)).save(masks / "a.png")
    widget = DatasetValidatorContainer(viewer)
    widget.images_dir.value = images
    widget.masks_dir.value = masks
    widget.report_path.value = tmp_path / "report.json"

    widget._check_dataset()
    widget._export_report()

    assert "Dataset validation (semantic_segmentation): OK" in widget.report.value
    assert "recommended_preset" in widget.report.value
    assert (tmp_path / "report.json").exists()


def test_dataset_validator_widget_previews_classification(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    dataset_dir = tmp_path / "classification"
    (dataset_dir / "class_a").mkdir(parents=True)
    (dataset_dir / "class_b").mkdir()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(dataset_dir / "class_a" / "a.png")
    Image.fromarray(np.ones((8, 8, 3), dtype=np.uint8)).save(dataset_dir / "class_b" / "b.png")
    widget = DatasetValidatorContainer(viewer)
    widget.task_type.value = "classification"
    widget.classification_dir.value = dataset_dir
    widget.preview_count.value = 2

    widget._check_dataset()
    widget._preview_dataset()

    assert "Dataset validation (classification): OK" in widget.report.value
    assert "Classification validator grid" in viewer.layers
    assert len(viewer.layers["Classification validator grid"].metadata["labels"]) == 2


def test_dataset_validator_widget_previews_instance(make_napari_viewer, tmp_path):
    image_dir, annotation_path = _write_tiny_coco_dataset(tmp_path)
    viewer = make_napari_viewer()
    widget = DatasetValidatorContainer(viewer)
    widget.task_type.value = "instance_segmentation"
    widget.images_dir.value = image_dir
    widget.coco_json.value = annotation_path
    widget.preview_index.value = 0

    widget._check_dataset()
    widget._preview_dataset()

    assert "Dataset validation (instance_segmentation): OK" in widget.report.value
    assert "Instance validator image" in viewer.layers
    assert "Instance validator labels" in viewer.layers


def test_model_manager_widget_scans_and_registers_training_run(tmp_path):
    run_dir = tmp_path / "semantic_run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        """
        {
          "task": "semantic_segmentation",
          "version": 1,
          "parameters": {
            "model_name": "unet",
            "backbone_name": "resnet34",
            "target_size": 256,
            "classes_num": 2
          }
        }
        """,
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        '{"final": {"val_IoU": 0.8}}',
        encoding="utf-8",
    )
    (run_dir / "training_log.csv").write_text("epoch,loss\n1,0.1\n", encoding="utf-8")
    (run_dir / "best_model_epoch1_IoU=0.8000.pth").write_bytes(b"checkpoint")

    widget = ModelManagerContainer(None)
    widget.registry_path.value = tmp_path / "registry.json"
    widget._registry = {"version": 1, "models": []}
    widget.scan_root.value = tmp_path
    widget._scan_folder()
    widget._show_selected_model()

    assert len(widget._registry["models"]) == 1
    assert widget.model_table.native.rowCount() == 1
    assert "unet" in widget.model_list.value
    assert "best_model_epoch1_IoU=0.8000.pth" in widget.model_details.value


def test_model_manager_manual_registration_is_collapsible():
    widget = ModelManagerContainer(None)

    assert widget.manual_registration.value is False
    assert widget.manual_name.native.isHidden()
    assert widget.manual_checkpoint.native.isHidden()
    assert widget._add_manual_button.native.isHidden()

    widget.manual_registration.value = True

    assert not widget.manual_name.native.isHidden()
    assert not widget.manual_checkpoint.native.isHidden()
    assert not widget._add_manual_button.native.isHidden()


def test_model_manager_fills_semantic_inference_widget(tmp_path):
    run_dir = tmp_path / "semantic_run"
    run_dir.mkdir()
    checkpoint = run_dir / "final_model.pth"
    checkpoint.write_bytes(b"checkpoint")
    config = run_dir / "config.json"
    config.write_text(
        """
        {
          "task": "semantic_segmentation",
          "version": 1,
          "parameters": {
            "model_name": "unet",
            "backbone_name": "resnet34",
            "target_size": 384,
            "classes_num": 4
          }
        }
        """,
        encoding="utf-8",
    )

    manager = ModelManagerContainer(None)
    manager._registry = {
        "version": 1,
        "models": [
            {
                "name": "semantic",
                "task": "semantic_segmentation",
                "checkpoint_path": str(checkpoint),
                "config_path": str(config),
                "summary": {},
                "status": "available",
            }
        ],
    }
    manager._refresh_model_list()

    inference = manager._fill_selected_inference_widget()

    assert isinstance(inference, DLInferenceContainer)
    assert inference.model_path.value == checkpoint
    assert inference.config_path.value == config
    assert inference.model_name.value == "unet"
    assert inference.backbone_name.value == "resnet34"
    assert inference.img_size.value == 384
    assert inference.num_classes.value == 4


def test_model_manager_fills_classification_inference_widget(tmp_path):
    checkpoint = tmp_path / "classification_knn.pth"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.json"
    config.write_text('{"task": "classification", "parameters": {}}', encoding="utf-8")

    manager = ModelManagerContainer(None)
    manager._registry = {
        "version": 1,
        "models": [
            {
                "name": "classification",
                "task": "classification",
                "checkpoint_path": str(checkpoint),
                "config_path": str(config),
                "summary": {},
                "status": "available",
            }
        ],
    }
    manager._refresh_model_list()

    inference = manager._fill_selected_inference_widget()

    assert isinstance(inference, ClassificationInferenceContainer)
    assert inference.checkpoint_path.value == checkpoint


def test_model_manager_fills_instance_inference_widget(tmp_path):
    checkpoint = tmp_path / "final_instance_segmentation.pth"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.json"
    config.write_text(
        """
        {
          "task": "instance_segmentation",
          "version": 1,
          "parameters": {
            "model_name": "mask_rcnn_instance",
            "backbone_name": "resnet50",
            "img_size": 640,
            "num_classes": 3
          }
        }
        """,
        encoding="utf-8",
    )

    manager = ModelManagerContainer(None)
    manager._registry = {
        "version": 1,
        "models": [
            {
                "name": "instance",
                "task": "instance_segmentation",
                "checkpoint_path": str(checkpoint),
                "config_path": str(config),
                "summary": {},
                "status": "available",
            }
        ],
    }
    manager._refresh_model_list()

    inference = manager._fill_selected_inference_widget()

    assert isinstance(inference, InstanceSegmentationInferenceContainer)
    assert inference.checkpoint_path.value == checkpoint
    assert inference.model_name.value == "mask_rcnn_instance"
    assert inference.backbone_name.value == "resnet50"
    assert inference.img_size.value == 640
    assert inference.num_classes.value == 3


def test_model_manager_edits_deletes_imports_exports_and_checks_model(tmp_path):
    run_dir = tmp_path / "semantic_run"
    run_dir.mkdir()
    checkpoint = run_dir / "final_model.pth"
    checkpoint.write_bytes(b"checkpoint")
    config = run_dir / "config.json"
    config.write_text(
        """
        {
          "task": "semantic_segmentation",
          "version": 1,
          "parameters": {
            "model_name": "unet",
            "backbone_name": "resnet34",
            "target_size": 256,
            "classes_num": 2
          }
        }
        """,
        encoding="utf-8",
    )

    manager = ModelManagerContainer(None)
    manager.registry_path.value = tmp_path / "registry.json"
    manager._registry = {"version": 1, "models": []}
    manager.scan_root.value = tmp_path
    manager._scan_folder()
    manager._on_model_table_cell_clicked(0, 1)

    assert manager.selected_index.value == 0
    assert manager.edit_name.value == "final_model"

    manager.edit_name.value = "renamed model"
    manager.edit_notes.value = "useful note"
    manager._save_selected_metadata()

    assert manager._registry["models"][0]["name"] == "renamed model"
    assert manager._registry["models"][0]["notes"] == "useful note"

    export_path = tmp_path / "exported_registry.json"
    manager.export_registry_path.value = export_path
    manager._export_registry()
    assert export_path.exists()

    manager._check_selected_model()
    assert "Model health check" in manager.model_details.value
    assert "config_task_matches" in manager.model_details.value

    manager._delete_selected_model()
    assert manager._registry["models"] == []

    manager.import_registry_path.value = export_path
    manager._import_registry()
    assert len(manager._registry["models"]) == 1
    assert manager._registry["models"][0]["name"] == "renamed model"


def test_model_manager_fills_training_widgets(tmp_path):
    checkpoint = tmp_path / "final_instance_segmentation.pth"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.json"
    config.write_text(
        """
        {
          "task": "instance_segmentation",
          "version": 1,
          "parameters": {
            "model_name": "mask_rcnn_instance",
            "backbone_name": "resnet50",
            "img_size": 640,
            "num_classes": 3
          }
        }
        """,
        encoding="utf-8",
    )
    manager = ModelManagerContainer(None)
    manager._registry = {
        "version": 1,
        "models": [
            {
                "name": "instance",
                "task": "instance_segmentation",
                "checkpoint_path": str(checkpoint),
                "config_path": str(config),
                "summary": {},
                "status": "available",
            }
        ],
    }
    manager._refresh_model_list()

    training = manager._fill_selected_training_widget()

    assert isinstance(training, InstanceSegmentationTrainingContainer)
    assert training.checkpoint_path.value == checkpoint
    assert training.model_name.value == "mask_rcnn_instance"
    assert training.backbone_name.value == "resnet50"
    assert training.img_size.value == 640
    assert training.num_classes.value == 3


def test_model_manager_is_in_manifest():
    manifest = Path("src/emcfsys/napari.yaml").read_text(encoding="utf-8")

    assert "emcfsys.model_manager" in manifest
    assert "ModelManagerContainer" in manifest
    assert "Model Manager | Registry" in manifest
    assert "emcfsys.dataset_validator" in manifest
    assert "DatasetValidatorContainer" in manifest
    assert "Dataset Tools | Dataset Validator" in manifest
