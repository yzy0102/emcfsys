# frc

`frc` is a Python library for calculating Fourier Ring Correlation (FRC) curves and their associated resolution.

This is particularly useful for determining the resolution of images taken with super-resolution microscopy techniques.

Its most computationally intensive functions are all implemented either in NumPy or Rust, making this library quite performant.

### Fourier Ring Correlation

Introduced in 1982, Fourier Ring Correlation compares two 2D images, which are assumed to be noise-independent. In Fourier space, mage structure dominates the lower spatial frequencies (in the Fourier domain), while noise dominates at the higher frequencies.

In Fourier space, there are rings of constant spatial frequency around the origin. By calculating the correlation between the rings in the two images (varying the spatial frequency), you can compute an FRC curve (y-axis: correlation, x-axis: spatial frequency).

At some spatial frequency, the signal cannot be separated from the noise. What spatial frequency exactly depends on what threshold function is used. The standard 0.143 and 1/2-bit thresholds are both available, as well as an algorithm to compute the intersection and resulting resolution.

For additional information about Fourier Ring Correlation, see [R.P.J. Nieuwenhuizen et al. (2013)](https://doi.org/10.1038/nmeth.2448).

#### 1FRC

FRC requires two noise-independent images to work. However, modern cameras are often shot noise-limited (Poisson noise). This library includes a new method due to Bernd Rieger and Sjoerd Stallinga (Department of Imaging Physics, TU Delft), called 1FRC, which uses a technique called binomial splitting to derive two images from a single one, where the pixel counts of the derived images are independently Poisson noise distributed.

This library was produced for my bachelor thesis in 2022 (which focused on 1FRC) and was supervised by Bernd Rieger. [The bachelor thesis can be found here](http://resolver.tudelft.nl/uuid:abdc31f6-6ecd-4c1b-a0c4-53711829467a). Bernd Rieger and Sjoerd Stallinga continued research, leading to a peer reviewed paper published in 2024 in Optics Express, [which can be found here](https://doi.org/10.1364/OE.524683). Their paper includes many details on the limitations of 1FRC and when exactly it is appropriate. I repeat the abstract below:

> We address resolution assessment for (light super-resolution) microscopy imaging. In modalities where imaging is not diffraction limited, correlation between two noise independent images is the standard way to infer the resolution. Here we take away the need for two noise independent images by computationally splitting one image acquisition into two noise independent realizations. This procedure generates two Poisson noise distributed images if the input is Poissonian distributed. As most modern cameras are shot-noise limited this procedure is directly applicable. However, also in the presence of readout noise we can compute the resolution faithfully via a correction factor. We evaluate our method on simulations and experimental data of widefield microscopy, STED microscopy, rescan confocal microscopy, image scanning microscopy, conventional confocal microscopy, and transmission electron microscopy. In all situations we find that using one image instead of two results in the same computed image resolution.

### Installation

You can download this library from PyPI:

```shell
pip install frc
```

This library depends on [tmtenbrink/rustfrc](https://www.github.com/tmtenbrink/rustfrc) (a Rust extension developed specifically for this library), [DIPlib](https://github.com/DIPlib/diplib), [scipy](https://scipy.org/) and of course, [numpy](https://numpy.org/).

### Usage

The code snippet below (which for illustration purposes assumes you have a numpy array or DIP image representing your input, its associated scale and also matplotlib to plot the result) will calculate the FRC curve and the associated resolution using the standard 1/7 threshold.

```python
import frc
import numpy as np
import matplotlib.pyplot as plt

... # get an image and scale

img = np.array(img)
# Input can be a numpy array or DIP image
img = frc.util.square_image(img, add_padding=False)
img = frc.util.apply_tukey(img)
# Apply 1FRC technique
frc_curve = frc.one_frc(img)

img_size = img.shape[0]
xs_pix = np.arange(len(frc_curve)) / img_size
# scale has units [pixels <length unit>^-1] corresponding to original image
xs_nm_freq = xs_pix * scale
frc_res, res_y, thres = frc.frc_res(xs_nm_freq, frc_curve, img_size)
plt.plot(xs_nm_freq, thres(xs_nm_freq))
plt.plot(xs_nm_freq, frc_curve)
plt.show()
```

#### Troubleshooting

If you cannot find an intersection (`NoIntersectionException`), be sure to plot the curve without using the `frc_res` method and see if there even is an intersection. For example:

```python
import frc
import numpy as np
import matplotlib.pyplot as plt

img = ... # get an image and scale

# Apply 1FRC technique
frc_curve = frc.one_frc(img)
img_size = img.shape[0]
xs_pix = np.arange(len(frc_curve)) / img_size
# scale has units [pixels <length unit>^-1] corresponding to original image
xs_nm_freq = xs_pix * scale
plt.plot(xs_nm_freq, frc_curve)
plt.show()
```

For images with a significant amount of non-Poisson noise, the 1FRC method has been shown to fail (adjustments are possible).

### Batch processing

The package also includes an optional batch runner that follows a numbered-image
folder convention while continuing to use the library's existing `two_frc` and
`frc_res` calculation functions.

Each sample folder can contain:

- `1.tif` / `1.png` / `1.jpg`: reference (ground truth)
- `2.*`: Simple
- `3.*`: EMDiffuse-r
- `4.*`: Our

Only `1` plus at least one of `2`, `3`, or `4` are required. Missing comparison
images are logged and the remaining comparisons continue to run. If a comparison
image has a different size from `1`, it is resized to the reference size in the
batch layer before both images are made square and passed to `frc.two_frc`.

Run from an installed package:

```shell
frc-batch /path/to/root
```

Or directly as a module:

```shell
python -m frc.batch /path/to/root
```

Useful options include `--recursive`, `--output-dir-name`, `--smooth-window`,
`--square-mode {trim,pad}`, `--subtract-mean`, `--no-window`, and `--no-plots`.
The defaults keep smoothing and mean subtraction off, and use the documented
1/8 Tukey window (`alpha=0.125`) before the unchanged `two_frc` calculation.

For every successful sample, the runner writes curve data, threshold results,
and a record of the exact input files used. The root output folder additionally
contains processing logs, long-form threshold summaries, GraphPad-friendly
threshold tables, all interpolated curves, and mean/SD curves. Plot files are
also produced when `matplotlib` is available.
