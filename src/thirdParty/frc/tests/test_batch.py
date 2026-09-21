from pathlib import Path

import numpy as np

import frc.batch as batch


def test_find_numbered_image_uses_extension_priority(tmp_path: Path):
    (tmp_path / "1.jpg").write_bytes(b"jpg")
    (tmp_path / "1.tif").write_bytes(b"tif")
    (tmp_path / "10.tif").write_bytes(b"not image 1")

    chosen, duplicates = batch.find_numbered_image(tmp_path, "1")

    assert chosen is not None
    assert chosen.name == "1.tif"
    assert [path.name for path in duplicates] == ["1.jpg"]


def test_smooth_curve_window_one_is_identity_copy():
    values = np.array([1.0, 0.5, 0.25])
    smoothed = batch.smooth_curve(values, 1)
    np.testing.assert_array_equal(smoothed, values)
    assert smoothed is not values


def test_normalized_frequency_axis_reaches_near_nyquist():
    axis = batch.normalized_frequency_axis(curve_size=4, image_size=8)
    np.testing.assert_allclose(axis, [0.0, 0.25, 0.5, 0.75])


def test_collect_candidate_folders_excludes_output(tmp_path: Path):
    (tmp_path / "A").mkdir()
    (tmp_path / "B").mkdir()
    output = tmp_path / batch.DEFAULT_OUTPUT_DIR_NAME
    output.mkdir()

    folders = batch.collect_candidate_folders(tmp_path, output, recursive=False)

    assert [folder.name for folder in folders] == ["A", "B"]
