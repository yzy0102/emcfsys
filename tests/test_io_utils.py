from emcfsys._io_utils import collect_image_files, ensure_directory, normalize_optional_path


def test_normalize_optional_path():
    assert normalize_optional_path(None) is None
    assert normalize_optional_path("") is None
    assert normalize_optional_path(".") is None
    assert normalize_optional_path("None") is None
    assert normalize_optional_path("abc") == "abc"


def test_ensure_directory(tmp_path):
    target = tmp_path / "nested" / "output"
    created = ensure_directory(target)
    assert target.is_dir()
    assert created == str(target)


def test_collect_image_files(tmp_path):
    (tmp_path / "a.png").write_text("x", encoding="utf-8")
    (tmp_path / "b.tif").write_text("x", encoding="utf-8")
    (tmp_path / "c.txt").write_text("x", encoding="utf-8")

    files = collect_image_files(tmp_path)

    assert str(tmp_path / "a.png") in files
    assert str(tmp_path / "b.tif") in files
    assert str(tmp_path / "c.txt") not in files
