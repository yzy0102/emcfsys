"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/building_a_plugin/guides.html#sample-data

Replace code below according to your needs.
"""

from __future__ import annotations

import numpy


def make_sample_data():
    """Generates an image"""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image
    
    # download the test image from github
    import requests
    import io
    from PIL import Image
    url = "github:yzy0102/emcfsys/raw/main/test_image.png"
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))
    img = numpy.array(img)
    return [(img, {})]
