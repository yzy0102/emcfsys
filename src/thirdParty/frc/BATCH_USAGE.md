# Batch FRC usage

This repository uses a `src/` package layout. From a freshly extracted source
folder, choose one of the following ways to run the batch processor.

## Direct source-tree launcher (no package installation required)

```bash
python run_batch.py /Users/jason/Desktop/EMCF/solo
```

## Run as a Python module without installing

```bash
PYTHONPATH="$PWD/src" python -m frc.batch /Users/jason/Desktop/EMCF/solo
```

## Editable installation (recommended when developing this package)

```bash
python -m pip install -e .
python -m frc.batch /Users/jason/Desktop/EMCF/solo
```

After editable installation, the console command is also available:

```bash
frc-batch /Users/jason/Desktop/EMCF/solo
```

The launcher and installation setup do not modify the library's existing FRC
calculation functions. Batch processing delegates to the existing
`frc.two_frc` / `frc.frc_res` implementation.

## PNG/TIFF/JPEG compatibility

The batch reader keeps DIPlib as the first choice. If DIPlib rejects a valid
PNG/TIFF/JPEG encoding, it automatically falls back to OpenCV using
`np.fromfile + cv2.imdecode(..., cv2.IMREAD_UNCHANGED)`, which preserves
8-bit/16-bit grayscale data and supports RGB/RGBA images. Pillow is used as a
final optional fallback.

This compatibility layer only changes image decoding. The FRC calculation
still uses the package's unchanged `frc.two_frc()` / `frc.frc_res()` engine.

If both compatibility readers are missing, install one of them in the active
environment, for example:

```bash
python -m pip install opencv-python
```
