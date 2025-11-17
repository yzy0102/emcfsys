import numpy as np

from emcfsys._widget import (
    ExampleQWidget,
    ImageResize,
    ImageThreshold,
    threshold_autogenerate_widget,
    threshold_magic_widget,
)


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
