import numpy as np
from scipy import ndimage
from skimage.transform import resize


ALGORITHM_MAP = {
    "Nearest Neighbor": 0,
    "Bilinear": 1,
    "Bicubic": 3,
    "Lanczos": None,
}


def spatial_shape(image):
    if image is None or image.ndim < 2:
        return None
    return image.shape[-2], image.shape[-1]


def target_size_from_mode(*, image, mode, width, height, scale_x, scale_y):
    if mode == "Absolute Size":
        return height, width
    return int(image.shape[-2] * scale_y), int(image.shape[-1] * scale_x)


def output_shape_for_image(image, output_height, output_width):
    if image.ndim == 2:
        return (output_height, output_width)
    if image.ndim == 3 and image.shape[-1] in [3, 4]:
        return (output_height, output_width, image.shape[-1])
    if image.ndim == 3:
        return (
            int(image.shape[0] * (output_height / image.shape[-2])),
            output_height,
            output_width,
        )

    scale_factors = [1.0] * image.ndim
    scale_factors[-2] = output_height / image.shape[-2]
    scale_factors[-1] = output_width / image.shape[-1]
    return tuple(int(size * factor) for size, factor in zip(image.shape, scale_factors))


def resize_with_ndimage(image, output_shape, order):
    if image.ndim == 2:
        return ndimage.zoom(
            image,
            (output_shape[0] / image.shape[0], output_shape[1] / image.shape[1]),
            order=order,
        )

    if image.ndim == 3 and image.shape[-1] in [3, 4]:
        resized = np.zeros(output_shape, dtype=image.dtype)
        zoom_shape = (
            output_shape[0] / image.shape[0],
            output_shape[1] / image.shape[1],
        )
        for channel_index in range(image.shape[-1]):
            resized[..., channel_index] = ndimage.zoom(
                image[..., channel_index],
                zoom_shape,
                order=order,
            )
        return resized

    zoom_factors = [out_size / in_size for out_size, in_size in zip(output_shape, image.shape)]
    return ndimage.zoom(image, zoom_factors, order=order)


def resize_array(image, output_height, output_width, algorithm):
    output_shape = output_shape_for_image(image, output_height, output_width)
    if algorithm == "Lanczos":
        return resize(
            image,
            output_shape,
            order=3,
            mode="reflect",
            anti_aliasing=True,
        )

    return resize_with_ndimage(image, output_shape, ALGORITHM_MAP[algorithm])
