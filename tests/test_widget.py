import numpy as np
from pathlib import Path
from qtpy.QtWidgets import QTextEdit

from emcfsys._widget import (
    COCOInstanceDatasetInspector,
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
    inspector_widget = COCOInstanceDatasetInspector(viewer)

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
    assert inspector_widget.include_crowd.value is False
    assert inspector_widget.preview_index.value == 0


def test_coco_instance_dataset_inspector_checks_and_previews(
    make_napari_viewer,
    tmp_path,
):
    image_dir, annotation_path = _write_tiny_coco_dataset(tmp_path)
    viewer = make_napari_viewer()
    widget = COCOInstanceDatasetInspector(viewer)

    widget.image_dir.value = image_dir
    widget.annotation_path.value = annotation_path
    widget._check_dataset()
    widget._preview_dataset_item()

    assert widget._last_report["ok"] is True
    assert "COCO instance dataset check: OK" in widget.report.value
    assert "COCO preview image 1" in viewer.layers
    assert "COCO preview instances 1" in viewer.layers
    assert "COCO preview boxes 1" in viewer.layers
    assert viewer.layers["COCO preview instances 1"].data.max() == 1


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
    training_widget.use_separate_eval_sets.value = True
    training_widget.val_annotation_path.value = tmp_path / "val.json"
    training_widget._save_config()

    training_widget.model_name.value = "rtm_instance"
    training_widget.use_advanced_mask_losses.value = False
    training_widget.boundary_loss_weight.value = 0.0
    training_widget.use_separate_eval_sets.value = False
    training_widget.val_annotation_path.value = None
    training_widget._load_config()

    assert config_path.exists()
    assert training_widget.model_name.value == "mask_rcnn_instance"
    assert training_widget.use_advanced_mask_losses.value is True
    assert training_widget.boundary_loss_weight.value == 0.2
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
    assert "emcfsys.instance_segmentation.dataset_inspector" in manifest
    assert "COCOInstanceDatasetInspector" in manifest
    assert "emcfsys.labelme_2_coco_instance" in manifest
    assert "LabelMe2COCOInstance" in manifest
