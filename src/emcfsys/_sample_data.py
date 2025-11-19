"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/building_a_plugin/guides.html#sample-data

Replace code below according to your needs.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

import os
import numpy as np
from PIL import Image

def make_sample_data():
    folder = os.path.dirname(__file__)
    path = os.path.join(folder, "test_imgs", "test_img.tif")

    img = np.array(Image.open(path))
    return [(img, {"name": "local_test_image"})]
