"""Batch processing helpers for the :mod:`frc` package.

This module intentionally keeps the library's existing FRC implementation as
its calculation engine.  Batch processing only adds image discovery,
pre-processing, reporting and plotting around :func:`frc.two_frc` and
:func:`frc.frc_res`.
"""

from __future__ import annotations

import argparse
import csv
import math
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from frc.deps_types import NoIntersectionException, dip, np
from frc.frc_functions import frc_res, two_frc
import frc.utility as frc_util

SUPPORTED_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
EXTENSION_PRIORITY = {".tif": 0, ".tiff": 1, ".png": 2, ".jpg": 3, ".jpeg": 4}
METHOD_CONFIG = {
    "2": {"label": "Simple", "color": "#7E6148", "linewidth": 1.35},
    "3": {"label": "EMDiffuse-r", "color": "#4DBBD5", "linewidth": 1.45},
    "4": {"label": "Our", "color": "#E64B35", "linewidth": 1.85},
}

DEFAULT_OUTPUT_DIR_NAME = "FRC_batch_output"
DEFAULT_RECURSIVE = False
DEFAULT_SMOOTH_WINDOW = 1
DEFAULT_SUBTRACT_MEAN = False
DEFAULT_WINDOW_ALPHA = 0.125
DEFAULT_SQUARE_MODE = "trim"

# The original package documents 1/7 as the standard FRC threshold.  The batch
# report also includes 0.5 because it is useful for side-by-side comparisons.
THRESHOLD_SPECS = (
    ("0.5", 0.5, "constant"),
    ("0.143", 1.0 / 7.0, "one_seventh"),
)
COMMON_FREQ_POINTS = 501
COMMON_FREQ_MAX = 1.0
PLOT_X_MAX = 1.0
PLOT_Y_MIN = -0.05
PLOT_Y_MAX = 1.05
FIGSIZE_INCH = (4.2, 3.2)
FIG_DPI = 600
SAVE_TIFF = True
PLOT_RAW_CURVES = False


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch Fourier Ring Correlation analysis using frc.two_frc/frc.frc_res "
            "without replacing the package's normal calculation functions."
        )
    )
    parser.add_argument("root_dir", type=Path, help="Root directory containing sample folders")
    parser.add_argument(
        "--output-dir-name",
        default=DEFAULT_OUTPUT_DIR_NAME,
        help=f"Output folder created under root_dir (default: {DEFAULT_OUTPUT_DIR_NAME})",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=DEFAULT_RECURSIVE,
        help="Search sample folders recursively (default: off)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help="Odd moving-average window used before threshold analysis (default: 1 = no smoothing)",
    )
    parser.add_argument(
        "--square-mode",
        choices=("trim", "pad"),
        default=DEFAULT_SQUARE_MODE,
        help=(
            "How to make equal-sized image pairs square before frc.two_frc. "
            "trim matches the README example (default: trim)."
        ),
    )
    parser.add_argument(
        "--window-alpha",
        type=float,
        default=DEFAULT_WINDOW_ALPHA,
        help="Tukey window alpha passed to frc.util.apply_tukey (default: 0.125)",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Skip Tukey apodization in the batch pre-processing layer",
    )
    parser.add_argument(
        "--subtract-mean",
        action="store_true",
        default=DEFAULT_SUBTRACT_MEAN,
        help="Subtract each image mean before square/window pre-processing (default: off)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write CSV outputs only; do not create PNG/TIFF figures",
    )
    return parser.parse_args(argv)


def find_numbered_image(folder: Path, number: str) -> Tuple[Optional[Path], List[Path]]:
    """Find an image whose stem is exactly ``number``.

    If multiple supported extensions are present, the same deterministic
    extension priority as the user's batch script is used.
    """
    matches = [
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and p.stem.strip() == number
    ]
    if not matches:
        return None, []
    matches.sort(
        key=lambda p: (
            EXTENSION_PRIORITY.get(p.suffix.lower(), 999),
            p.name.lower(),
        )
    )
    return matches[0], matches[1:]


def _to_grayscale(arr: np.ndarray, *, channel_order: str = "rgb") -> np.ndarray:
    """Convert a decoded image array to 2D grayscale without rescaling values."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3:
        # Accept both HxWxC and CxHxW layouts.
        if arr.shape[-1] in (1, 3, 4):
            channels = arr
        elif arr.shape[0] in (1, 3, 4):
            channels = np.moveaxis(arr, 0, -1)
        else:
            raise ValueError(f"Unsupported image dimensions/channels: {arr.shape}")

        if channels.shape[-1] == 1:
            gray = channels[..., 0]
        else:
            values = channels[..., :3].astype(np.float64, copy=False)
            if channel_order.lower() == "bgr":
                # OpenCV decodes color images as BGR/BGRA.
                gray = 0.0722 * values[..., 0] + 0.7152 * values[..., 1] + 0.2126 * values[..., 2]
            else:
                # DIPlib/Pillow arrays are treated as RGB/RGBA.
                gray = 0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]
    else:
        raise ValueError(f"Unsupported image dimensions: {arr.shape}")

    gray = np.asarray(gray, dtype=np.float64)
    if not np.all(np.isfinite(gray)):
        finite = gray[np.isfinite(gray)]
        fill_value = float(np.median(finite)) if finite.size else 0.0
        gray = np.nan_to_num(gray, nan=fill_value, posinf=fill_value, neginf=fill_value)
    if min(gray.shape) < 4:
        raise ValueError(f"Image is too small for FRC: {gray.shape}")
    return gray


def _imread_opencv(path: Path) -> np.ndarray:
    """Decode an image with OpenCV while preserving PNG/TIFF bit depth.

    Reading bytes with ``np.fromfile`` before ``cv2.imdecode`` also avoids
    path-encoding issues on platforms where ``cv2.imread`` can be fragile.
    OpenCV is imported lazily so it remains an optional compatibility reader
    and does not change the FRC package's calculation dependencies.
    """
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"Empty image file: {path}")
    arr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"OpenCV cannot decode image: {path}")
    return _to_grayscale(arr, channel_order="bgr")


def _imread_pillow(path: Path) -> np.ndarray:
    """Last-resort reader for formats DIPlib/OpenCV cannot decode."""
    from PIL import Image

    with Image.open(path) as image:
        # Keep native grayscale/16-bit data unchanged.  Palette/CMYK and other
        # color modes are converted to RGB only when needed.
        if image.mode in ("1", "L", "I", "I;16", "F"):
            arr = np.asarray(image)
        else:
            arr = np.asarray(image.convert("RGB"))
    return _to_grayscale(arr, channel_order="rgb")


def imread_image(path: Path) -> np.ndarray:
    """Read an image with compatibility fallbacks, without changing FRC math.

    DIPlib remains the first reader.  If it rejects a valid PNG/TIFF/JPEG
    encoding, OpenCV is tried next (matching the user's original batch script),
    followed by Pillow when available.  All readers return the same 2D
    ``float64`` array interface consumed by the unchanged ``two_frc`` engine.
    """
    errors: List[str] = []

    try:
        arr = np.asarray(dip.ImageRead(str(path)))
        return _to_grayscale(arr, channel_order="rgb")
    except Exception as exc:
        errors.append(f"DIPlib: {exc}")

    try:
        return _imread_opencv(path)
    except (ImportError, ModuleNotFoundError) as exc:
        errors.append(f"OpenCV unavailable: {exc}")
    except Exception as exc:
        errors.append(f"OpenCV: {exc}")

    try:
        return _imread_pillow(path)
    except (ImportError, ModuleNotFoundError) as exc:
        errors.append(f"Pillow unavailable: {exc}")
    except Exception as exc:
        errors.append(f"Pillow: {exc}")

    details = " | ".join(errors)
    raise ValueError(f"Cannot read image: {path}. Reader errors: {details}")


def resize_to_reference(image: np.ndarray, reference_shape: Tuple[int, int]) -> np.ndarray:
    """Resize a comparison image to the reference shape in the batch layer."""
    if image.shape == reference_shape:
        return image.astype(np.float64, copy=False)

    # scipy is already a runtime dependency of frc.  order=3 provides cubic
    # interpolation without adding OpenCV/Pillow as new package dependencies.
    from scipy.ndimage import zoom

    factors = (
        float(reference_shape[0]) / float(image.shape[0]),
        float(reference_shape[1]) / float(image.shape[1]),
    )
    resized = zoom(image, factors, order=3, mode="nearest", prefilter=True)

    # Older scipy versions can differ by one pixel due to rounding.  Enforce
    # the exact requested shape without changing the FRC engine itself.
    target_h, target_w = reference_shape
    resized = resized[:target_h, :target_w]
    if resized.shape != reference_shape:
        pad_h = max(0, target_h - resized.shape[0])
        pad_w = max(0, target_w - resized.shape[1])
        resized = np.pad(resized, ((0, pad_h), (0, pad_w)), mode="edge")
        resized = resized[:target_h, :target_w]
    return resized.astype(np.float64, copy=False)


def smooth_curve(curve: np.ndarray, window: int = 1) -> np.ndarray:
    values = np.asarray(curve, dtype=np.float64)
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    if window > values.size:
        window = values.size if values.size % 2 == 1 else values.size - 1
    if window <= 1:
        return values.copy()
    finite = np.isfinite(values)
    if not np.all(finite):
        if not np.any(finite):
            return values.copy()
        indices = np.arange(values.size)
        values = np.interp(indices, indices[finite], values[finite])
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def prepare_pair(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    square_mode: str = DEFAULT_SQUARE_MODE,
    subtract_mean: bool = DEFAULT_SUBTRACT_MEAN,
    apply_window: bool = True,
    window_alpha: float = DEFAULT_WINDOW_ALPHA,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare a pair for the unchanged :func:`two_frc` engine.

    The comparison image must already have the same shape as the reference.
    Both images undergo exactly the same batch-only pre-processing.
    """
    if reference.shape != prediction.shape:
        raise ValueError(
            f"Batch pair must have equal shapes before FRC: {reference.shape} vs {prediction.shape}"
        )
    if square_mode not in ("trim", "pad"):
        raise ValueError("square_mode must be 'trim' or 'pad'")

    ref = reference.astype(np.float64, copy=False)
    pred = prediction.astype(np.float64, copy=False)
    if subtract_mean:
        ref = ref - np.mean(ref)
        pred = pred - np.mean(pred)

    add_padding = square_mode == "pad"
    ref = np.asarray(frc_util.square_image(ref, add_padding=add_padding), dtype=np.float64)
    pred = np.asarray(frc_util.square_image(pred, add_padding=add_padding), dtype=np.float64)

    if apply_window:
        ref = np.asarray(frc_util.apply_tukey(ref, alpha=window_alpha), dtype=np.float64)
        pred = np.asarray(frc_util.apply_tukey(pred, alpha=window_alpha), dtype=np.float64)
    return ref, pred


def normalized_frequency_axis(curve_size: int, image_size: int) -> np.ndarray:
    """Return frequency with Nyquist normalized to 1, for plotting/reporting."""
    return 2.0 * np.arange(curve_size, dtype=np.float64) / float(image_size)


def threshold_result(
    curve: np.ndarray,
    image_size: int,
    threshold_kind: str,
    threshold_value: float,
) -> Tuple[float, float, bool]:
    """Use the package's existing ``frc_res`` intersection method.

    Returns ``(normalized_nyquist_frequency, resolution_pixels, crossed)``.
    """
    xs_cycles_per_pixel = np.arange(len(curve), dtype=np.float64) / float(image_size)
    if threshold_kind == "one_seventh":
        threshold_arg = "1/7"
    elif threshold_kind == "constant":
        value = float(threshold_value)

        def threshold_arg(x):
            return np.asarray(x) * 0.0 + value
    else:
        raise ValueError(f"Unknown threshold kind: {threshold_kind}")

    try:
        # frc.utility.intersection can adjust the first value in one special
        # case, so pass a copy and keep the recorded FRC curve untouched.
        resolution_pixels, _, _ = frc_res(
            xs_cycles_per_pixel,
            np.asarray(curve, dtype=np.float64).copy(),
            image_size,
            threshold=threshold_arg,
        )
    except NoIntersectionException:
        return math.nan, math.nan, False

    if not np.isfinite(resolution_pixels) or resolution_pixels <= 0:
        return math.nan, math.nan, False
    normalized_frequency = 2.0 / float(resolution_pixels)
    return normalized_frequency, float(resolution_pixels), True


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(list(header))
        writer.writerows(rows)


def csv_value(value: object) -> object:
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    return value


def write_folder_curve_graphpad_csv(
    path: Path,
    frequencies: np.ndarray,
    raw_results: Dict[str, np.ndarray],
    smooth_results: Dict[str, np.ndarray],
    method_labels: Sequence[str],
) -> None:
    header = ["Spatial Frequency (Nyquist=1)"]
    header.extend(f"{label} Raw FRC" for label in method_labels)
    header.extend(f"{label} Smoothed FRC" for label in method_labels)
    header.extend(f"FRC Threshold {tag}" for tag, _, _ in THRESHOLD_SPECS)
    rows: List[List[object]] = []
    for i, frequency in enumerate(frequencies):
        row: List[object] = [float(frequency)]
        row.extend(csv_value(float(raw_results[label][i])) for label in method_labels)
        row.extend(csv_value(float(smooth_results[label][i])) for label in method_labels)
        row.extend(float(value) for _, value, _ in THRESHOLD_SPECS)
        rows.append(row)
    write_csv(path, header, rows)


def write_threshold_summary_graphpad(
    output_root: Path,
    summary_rows: Sequence[Dict[str, object]],
    folder_order: Sequence[str],
) -> None:
    all_labels = [str(cfg["label"]) for cfg in METHOD_CONFIG.values()]
    for threshold_tag, threshold_value, _ in THRESHOLD_SPECS:
        value_map: Dict[Tuple[str, str], object] = {}
        for row in summary_rows:
            if math.isclose(float(row["Threshold Value"]), threshold_value, rel_tol=0, abs_tol=1e-12):
                value_map[(str(row["Folder"]), str(row["Method"]))] = row[
                    "Crossing Frequency (Nyquist=1)"
                ]
        rows: List[List[object]] = []
        for folder in folder_order:
            row_values: List[object] = [folder]
            for label in all_labels:
                row_values.append(csv_value(value_map.get((folder, label), "")))
            rows.append(row_values)
        write_csv(
            output_root / f"FRC_threshold_{threshold_tag.replace('.', 'p')}_graphpad.csv",
            ["Folder", *all_labels],
            rows,
        )


def interpolate_curve(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
) -> np.ndarray:
    source_x = np.asarray(source_x, dtype=np.float64)
    source_y = np.asarray(source_y, dtype=np.float64)
    finite = np.isfinite(source_x) & np.isfinite(source_y)
    source_x = source_x[finite]
    source_y = source_y[finite]
    if source_x.size < 2:
        return np.full_like(target_x, np.nan, dtype=np.float64)
    result = np.interp(target_x, source_x, source_y, left=np.nan, right=np.nan)
    result[(target_x < source_x[0]) | (target_x > source_x[-1])] = np.nan
    return result


def write_aggregate_graphpad_csvs(
    output_root: Path,
    all_curves: Dict[str, List[Dict[str, object]]],
) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    common_frequency = np.linspace(0.0, COMMON_FREQ_MAX, COMMON_FREQ_POINTS, dtype=np.float64)

    all_header: List[str] = ["Spatial Frequency (Nyquist=1)"]
    all_columns: List[np.ndarray] = []
    for cfg in METHOD_CONFIG.values():
        label = str(cfg["label"])
        for record in all_curves.get(label, []):
            folder = str(record["folder"])
            interpolated = interpolate_curve(
                np.asarray(record["frequencies"]),
                np.asarray(record["smooth"]),
                common_frequency,
            )
            all_header.append(f"{folder} | {label} Smoothed FRC")
            all_columns.append(interpolated)
    all_header.extend(f"FRC Threshold {tag}" for tag, _, _ in THRESHOLD_SPECS)
    all_rows: List[List[object]] = []
    for idx, frequency in enumerate(common_frequency):
        row: List[object] = [float(frequency)]
        row.extend(csv_value(float(column[idx])) for column in all_columns)
        row.extend(float(value) for _, value, _ in THRESHOLD_SPECS)
        all_rows.append(row)
    write_csv(output_root / "FRC_all_curves_graphpad.csv", all_header, all_rows)

    aggregate_stats: Dict[str, Dict[str, np.ndarray]] = {}
    mean_header: List[str] = ["Spatial Frequency (Nyquist=1)"]
    mean_columns: List[np.ndarray] = []
    for cfg in METHOD_CONFIG.values():
        label = str(cfg["label"])
        records = all_curves.get(label, [])
        if not records:
            continue
        matrix = np.vstack(
            [
                interpolate_curve(
                    np.asarray(record["frequencies"]),
                    np.asarray(record["smooth"]),
                    common_frequency,
                )
                for record in records
            ]
        )
        finite_mask = np.isfinite(matrix)
        valid_n = np.sum(finite_mask, axis=0).astype(np.float64)
        finite_sum = np.nansum(matrix, axis=0)
        mean = np.divide(
            finite_sum,
            valid_n,
            out=np.full_like(finite_sum, np.nan, dtype=np.float64),
            where=valid_n > 0,
        )
        centered = np.where(finite_mask, matrix - mean, 0.0)
        squared_sum = np.sum(centered**2, axis=0)
        sd = np.sqrt(
            np.divide(
                squared_sum,
                valid_n - 1.0,
                out=np.full_like(squared_sum, np.nan, dtype=np.float64),
                where=valid_n > 1,
            )
        )
        sd[valid_n < 2] = np.nan
        mean[valid_n < 1] = np.nan
        aggregate_stats[label] = {"mean": mean, "sd": sd, "n": valid_n}
        mean_header.extend([f"{label} Mean FRC", f"{label} SD", f"{label} N"])
        mean_columns.extend([mean, sd, valid_n])

    mean_header.extend(f"FRC Threshold {tag}" for tag, _, _ in THRESHOLD_SPECS)
    mean_rows: List[List[object]] = []
    for idx, frequency in enumerate(common_frequency):
        row = [float(frequency)]
        row.extend(csv_value(float(column[idx])) for column in mean_columns)
        row.extend(float(value) for _, value, _ in THRESHOLD_SPECS)
        mean_rows.append(row)
    write_csv(output_root / "FRC_mean_curves_graphpad.csv", mean_header, mean_rows)
    return common_frequency, aggregate_stats


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["font.size"] = 10
    return plt


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", direction="out", which="major", labelsize=10, width=0.8, length=3.5)
    ax.set_xlim(0.0, PLOT_X_MAX)
    ax.set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)
    ax.set_xlabel("Spatial Frequency (Nyquist = 1)", fontsize=11)
    ax.set_ylabel("Fourier Ring Correlation", fontsize=11)
    ax.set_facecolor("white")


def _save_figure(fig, png_path: Path, tif_path: Optional[Path] = None) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    if SAVE_TIFF and tif_path is not None:
        try:
            fig.savefig(
                tif_path,
                dpi=FIG_DPI,
                bbox_inches="tight",
                pad_inches=0.03,
                facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"},
            )
        except (TypeError, ValueError):
            fig.savefig(tif_path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.03, facecolor="white")


def plot_folder_curve(
    output_folder: Path,
    frequencies: np.ndarray,
    raw_results: Dict[str, np.ndarray],
    smooth_results: Dict[str, np.ndarray],
    method_labels: Sequence[str],
) -> bool:
    plt = _load_matplotlib()
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=FIGSIZE_INCH)
    for tag, threshold, _ in THRESHOLD_SPECS:
        linestyle = "--" if tag == "0.5" else ":"
        ax.axhline(threshold, color="#8A8A8A", linestyle=linestyle, linewidth=0.75, zorder=1)
    config_by_label = {str(cfg["label"]): cfg for cfg in METHOD_CONFIG.values()}
    for label in method_labels:
        cfg = config_by_label[label]
        if PLOT_RAW_CURVES:
            ax.plot(
                frequencies,
                raw_results[label],
                color=str(cfg["color"]),
                linewidth=0.7,
                alpha=0.28,
                zorder=2,
            )
        ax.plot(
            frequencies,
            smooth_results[label],
            color=str(cfg["color"]),
            linewidth=float(cfg["linewidth"]),
            label=label,
            antialiased=True,
            zorder=3,
        )
    _style_axes(ax)
    ax.legend(loc="best", frameon=False, fontsize=9.5, handlelength=1.5, handletextpad=0.5)
    fig.tight_layout()
    _save_figure(fig, output_folder / "FRC_curve.png", output_folder / "FRC_curve.tif")
    plt.close(fig)
    return True


def plot_batch_mean_curve(
    output_root: Path,
    common_frequency: np.ndarray,
    aggregate_stats: Dict[str, Dict[str, np.ndarray]],
) -> bool:
    if not aggregate_stats:
        return False
    plt = _load_matplotlib()
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=FIGSIZE_INCH)
    for tag, threshold, _ in THRESHOLD_SPECS:
        linestyle = "--" if tag == "0.5" else ":"
        ax.axhline(threshold, color="#8A8A8A", linestyle=linestyle, linewidth=0.75, zorder=1)
    for cfg in METHOD_CONFIG.values():
        label = str(cfg["label"])
        if label not in aggregate_stats:
            continue
        mean = aggregate_stats[label]["mean"]
        sd = aggregate_stats[label]["sd"]
        valid_mean = np.isfinite(mean)
        ax.plot(
            common_frequency[valid_mean],
            mean[valid_mean],
            color=str(cfg["color"]),
            linewidth=float(cfg["linewidth"]),
            label=label,
            zorder=3,
        )
        valid_band = np.isfinite(mean) & np.isfinite(sd)
        if np.any(valid_band):
            ax.fill_between(
                common_frequency[valid_band],
                mean[valid_band] - sd[valid_band],
                mean[valid_band] + sd[valid_band],
                color=str(cfg["color"]),
                alpha=0.16,
                linewidth=0,
                zorder=2,
            )
    _style_axes(ax)
    ax.legend(loc="best", frameon=False, fontsize=9.5, handlelength=1.5, handletextpad=0.5)
    fig.tight_layout()
    _save_figure(fig, output_root / "FRC_batch_mean_curve.png", output_root / "FRC_batch_mean_curve.tif")
    plt.close(fig)
    return True


def plot_legend_only(output_root: Path) -> bool:
    plt = _load_matplotlib()
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(4.4, 0.55))
    for cfg in METHOD_CONFIG.values():
        ax.plot([], [], color=str(cfg["color"]), linewidth=float(cfg["linewidth"]), label=str(cfg["label"]))
    ax.axis("off")
    ax.legend(
        loc="center",
        ncol=len(METHOD_CONFIG),
        frameon=False,
        fontsize=10.5,
        handlelength=1.5,
        handletextpad=0.45,
        columnspacing=1.0,
        borderaxespad=0,
    )
    _save_figure(fig, output_root / "FRC_legend_only.png", output_root / "FRC_legend_only.tif")
    plt.close(fig)
    return True


def process_one_folder(
    folder: Path,
    root: Path,
    output_root: Path,
    *,
    smooth_window: int,
    square_mode: str,
    subtract_mean: bool,
    apply_window: bool,
    window_alpha: float,
    make_plots: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]], List[str]]:
    warnings: List[str] = []
    reference_path, reference_duplicates = find_numbered_image(folder, "1")
    if reference_path is None:
        raise FileNotFoundError("Missing reference image 1")
    if reference_duplicates:
        warnings.append(
            f"Multiple files found for 1; using {reference_path.name}; ignored: "
            + ", ".join(p.name for p in reference_duplicates)
        )

    method_paths: Dict[str, Path] = {}
    for number in METHOD_CONFIG:
        chosen, duplicates = find_numbered_image(folder, number)
        if chosen is None:
            warnings.append(f"Image {number} not found")
            continue
        method_paths[number] = chosen
        if duplicates:
            warnings.append(
                f"Multiple files found for {number}; using {chosen.name}; ignored: "
                + ", ".join(p.name for p in duplicates)
            )
    if not method_paths:
        raise FileNotFoundError("No comparison images found (expected 2, 3 and/or 4)")

    reference = imread_image(reference_path)
    reference_shape = reference.shape
    folder_rel = folder.relative_to(root)
    folder_id = folder_rel.as_posix()
    sample_output = output_root / folder_rel
    sample_output.mkdir(parents=True, exist_ok=True)

    raw_results: Dict[str, np.ndarray] = {}
    smooth_results: Dict[str, np.ndarray] = {}
    curve_records: Dict[str, Dict[str, object]] = {}
    threshold_rows: List[Dict[str, object]] = []
    frequencies: Optional[np.ndarray] = None
    method_labels: List[str] = []
    used_file_rows: List[List[object]] = [
        ["1", "Ground truth", reference_path.name, reference_shape[1], reference_shape[0], "No"]
    ]

    for number, cfg in METHOD_CONFIG.items():
        if number not in method_paths:
            continue
        label = str(cfg["label"])
        image_path = method_paths[number]
        prediction_original = imread_image(image_path)
        was_resized = prediction_original.shape != reference_shape
        prediction = resize_to_reference(prediction_original, reference_shape)
        ref_prepared, pred_prepared = prepare_pair(
            reference,
            prediction,
            square_mode=square_mode,
            subtract_mean=subtract_mean,
            apply_window=apply_window,
            window_alpha=window_alpha,
        )

        # Core calculation is the unchanged package function.
        raw_curve = np.asarray(two_frc(ref_prepared, pred_prepared), dtype=np.float64)
        smooth = smooth_curve(raw_curve, smooth_window)
        current_frequencies = normalized_frequency_axis(len(raw_curve), ref_prepared.shape[0])

        if frequencies is None:
            frequencies = current_frequencies
        elif current_frequencies.shape != frequencies.shape or not np.allclose(current_frequencies, frequencies):
            raise RuntimeError("Inconsistent frequency grids within one sample folder")

        raw_results[label] = raw_curve
        smooth_results[label] = smooth
        method_labels.append(label)
        curve_records[label] = {
            "folder": folder_id,
            "frequencies": current_frequencies,
            "raw": raw_curve,
            "smooth": smooth,
        }
        used_file_rows.append(
            [
                number,
                label,
                image_path.name,
                prediction_original.shape[1],
                prediction_original.shape[0],
                "Yes" if was_resized else "No",
            ]
        )

        for threshold_tag, threshold_value, threshold_kind in THRESHOLD_SPECS:
            crossing_frequency, resolution_pixels, crossed = threshold_result(
                smooth,
                ref_prepared.shape[0],
                threshold_kind,
                threshold_value,
            )
            threshold_rows.append(
                {
                    "Folder": folder_id,
                    "Method": label,
                    "Threshold": threshold_tag,
                    "Threshold Value": float(threshold_value),
                    "Crossing Frequency (Nyquist=1)": float(crossing_frequency),
                    "Resolution (pixels)": float(resolution_pixels),
                    "Threshold Crossed": "Yes" if crossed else "No",
                    "Reference File": reference_path.name,
                    "Compared File": image_path.name,
                    "Original Width": int(prediction_original.shape[1]),
                    "Original Height": int(prediction_original.shape[0]),
                    "Reference Width": int(reference_shape[1]),
                    "Reference Height": int(reference_shape[0]),
                    "FRC Input Size": int(ref_prepared.shape[0]),
                    "Resized": "Yes" if was_resized else "No",
                }
            )

    if frequencies is None:
        raise RuntimeError("No FRC curves generated")

    write_folder_curve_graphpad_csv(
        sample_output / "FRC_curve_graphpad.csv",
        frequencies,
        raw_results,
        smooth_results,
        method_labels,
    )
    threshold_header = [
        "Folder",
        "Method",
        "Threshold",
        "Threshold Value",
        "Crossing Frequency (Nyquist=1)",
        "Resolution (pixels)",
        "Threshold Crossed",
        "Reference File",
        "Compared File",
        "Original Width",
        "Original Height",
        "Reference Width",
        "Reference Height",
        "FRC Input Size",
        "Resized",
    ]
    write_csv(
        sample_output / "FRC_thresholds.csv",
        threshold_header,
        [[csv_value(row[column]) for column in threshold_header] for row in threshold_rows],
    )
    write_csv(
        sample_output / "input_files_used.csv",
        ["Number", "Role/Method", "Filename", "Original Width", "Original Height", "Resized"],
        used_file_rows,
    )

    if make_plots and not plot_folder_curve(
        sample_output,
        frequencies,
        raw_results,
        smooth_results,
        method_labels,
    ):
        warnings.append("matplotlib is not installed; plot files were skipped")

    return threshold_rows, curve_records, warnings


def collect_candidate_folders(root: Path, output_root: Path, recursive: bool) -> List[Path]:
    if recursive:
        candidates = [
            p
            for p in root.rglob("*")
            if p.is_dir() and p != output_root and output_root not in p.parents
        ]
        candidates.sort(key=lambda p: p.relative_to(root).as_posix().lower())
    else:
        candidates = [p for p in root.iterdir() if p.is_dir() and p != output_root]
        candidates.sort(key=lambda p: p.name.lower())
    return candidates


def run_batch(args: argparse.Namespace) -> Path:
    root = args.root_dir.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")
    if args.window_alpha < 0.0 or args.window_alpha > 1.0:
        raise ValueError("--window-alpha must be between 0 and 1")

    output_root = root / args.output_dir_name
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_folders = collect_candidate_folders(root, output_root, args.recursive)
    if not candidate_folders:
        raise RuntimeError(f"No candidate folders found under: {root}")

    all_threshold_rows: List[Dict[str, object]] = []
    all_curves: Dict[str, List[Dict[str, object]]] = {
        str(cfg["label"]): [] for cfg in METHOD_CONFIG.values()
    }
    processing_log: List[List[object]] = []
    successful_folders: List[str] = []

    print("=" * 78)
    print("FRC batch processing started")
    print(f"Input root: {root}")
    print(f"Output root: {output_root}")
    print(f"Candidate folders: {len(candidate_folders)}")
    print("Calculation engine: frc.two_frc + frc.frc_res (unchanged core)")
    print("=" * 78)

    for index, folder in enumerate(candidate_folders, start=1):
        folder_id = folder.relative_to(root).as_posix()
        print(f"[{index}/{len(candidate_folders)}] {folder_id}")
        try:
            threshold_rows, curve_records, warnings = process_one_folder(
                folder,
                root,
                output_root,
                smooth_window=args.smooth_window,
                square_mode=args.square_mode,
                subtract_mean=args.subtract_mean,
                apply_window=not args.no_window,
                window_alpha=args.window_alpha,
                make_plots=not args.no_plots,
            )
            all_threshold_rows.extend(threshold_rows)
            for label, record in curve_records.items():
                all_curves[label].append(record)
            successful_folders.append(folder_id)
            warning_text = " | ".join(warnings)
            processing_log.append([folder_id, "Success", len(curve_records), warning_text, ""])
            print(f"Success: {len(curve_records)} curves")
            for warning in warnings:
                print(f"Warning: {warning}")
        except Exception as exc:
            error_text = str(exc)
            processing_log.append([folder_id, "Failed", 0, "", error_text])
            print(f"Failed: {error_text}")
            print(traceback.format_exc())

    write_csv(
        output_root / "FRC_processing_log.csv",
        ["Folder", "Status", "Method Count", "Warnings", "Error"],
        processing_log,
    )
    if not successful_folders:
        raise RuntimeError("No folders were processed successfully")

    summary_header = [
        "Folder",
        "Method",
        "Threshold",
        "Threshold Value",
        "Crossing Frequency (Nyquist=1)",
        "Resolution (pixels)",
        "Threshold Crossed",
        "Reference File",
        "Compared File",
        "Original Width",
        "Original Height",
        "Reference Width",
        "Reference Height",
        "FRC Input Size",
        "Resized",
    ]
    write_csv(
        output_root / "FRC_threshold_summary_long.csv",
        summary_header,
        [[csv_value(row[column]) for column in summary_header] for row in all_threshold_rows],
    )
    write_threshold_summary_graphpad(output_root, all_threshold_rows, successful_folders)
    common_frequency, aggregate_stats = write_aggregate_graphpad_csvs(output_root, all_curves)

    if not args.no_plots:
        plot_batch_mean_curve(output_root, common_frequency, aggregate_stats)
        plot_legend_only(output_root)

    success_count = len(successful_folders)
    failed_count = len(candidate_folders) - success_count
    print("=" * 78)
    print("FRC batch processing completed")
    print(f"Successful folders: {success_count}")
    print(f"Failed folders: {failed_count}")
    print(f"Results: {output_root}")
    print("=" * 78)
    return output_root


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_batch(parse_args(argv))


if __name__ == "__main__":
    main()
