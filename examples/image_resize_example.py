"""
Example script demonstrating the Image Resize widget in napari.

This example shows how to:
1. Load an image into napari
2. Use the Image Resize widget to resize images
3. Try different interpolation algorithms
"""

import napari
import numpy as np

# Create a sample image
sample_image = np.random.random((200, 200))

# Add some structure to make resizing effects more visible
x, y = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
sample_image = np.sin(np.sqrt(x**2 + y**2))

# Create napari viewer
viewer = napari.Viewer()

# Add the sample image
viewer.add_image(sample_image, name="Original Image")

# The Image Resize widget can be accessed from:
# Plugins > EMCFsys > Image Resize

print("Image Resize Widget Example")
print("=" * 50)
print("\nTo use the Image Resize widget:")
print("1. Go to: Plugins > EMCFsys > Image Resize")
print("2. Select the image layer you want to resize")
print("3. Choose resize mode:")
print("   - Absolute Size: Specify exact width and height")
print("   - Scale Factor: Specify scaling factors for X and Y")
print("4. Select interpolation algorithm:")
print("   - Nearest Neighbor: Fastest, blocky results")
print("   - Bilinear: Good balance of speed and quality")
print("   - Bicubic: Higher quality, slower")
print("   - Lanczos: Best quality, slowest")
print("5. Optionally enable 'Maintain Aspect Ratio'")
print("6. Click 'Apply Resize' to create a resized copy")
print("\nThe resized image will be added as a new layer.")
print("=" * 50)

# Start the napari event loop
napari.run()

