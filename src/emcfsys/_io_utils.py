from pathlib import Path


IMAGE_FILE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")


def is_missing_path(value) -> bool:
    return value in (None, "", ".", "None")


def normalize_optional_path(value):
    if is_missing_path(value):
        return None
    return str(value)


def ensure_directory(path) -> str:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def collect_image_files(directory) -> list[str]:
    directory_path = Path(directory)
    image_files = []
    for pattern in IMAGE_FILE_PATTERNS:
        image_files.extend(str(path) for path in directory_path.glob(pattern))
    return sorted(image_files)
