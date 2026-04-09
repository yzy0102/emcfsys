import numpy as np

from emcfsys.utils.image_resize_ops import (
    output_shape_for_image,
    resize_array,
    spatial_shape,
    target_size_from_mode,
)


def test_spatial_shape_returns_last_two_dims():
    image = np.zeros((5, 8, 9), dtype=np.uint8)
    assert spatial_shape(image) == (8, 9)


def test_target_size_from_mode_absolute_size():
    image = np.zeros((4, 6), dtype=np.uint8)
    assert target_size_from_mode(
        image=image,
        mode="Absolute Size",
        width=11,
        height=7,
        scale_x=2.0,
        scale_y=3.0,
    ) == (7, 11)


def test_target_size_from_mode_scale_factor():
    image = np.zeros((4, 6), dtype=np.uint8)
    assert target_size_from_mode(
        image=image,
        mode="Scale Factor",
        width=0,
        height=0,
        scale_x=2.0,
        scale_y=1.5,
    ) == (6, 12)


def test_output_shape_for_rgb_image():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    assert output_shape_for_image(image, 10, 12) == (10, 12, 3)


def test_resize_array_nearest_neighbor_2d():
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    resized = resize_array(image, 8, 6, "Nearest Neighbor")
    assert resized.shape == (8, 6)


def test_resize_array_bilinear_rgb():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    resized = resize_array(image, 10, 12, "Bilinear")
    assert resized.shape == (10, 12, 3)


def test_resize_array_lanczos_preserves_shape_contract():
    image = np.zeros((2, 4, 6), dtype=np.float32)
    resized = resize_array(image, 8, 10, "Lanczos")
    assert resized.shape == (4, 8, 10)
