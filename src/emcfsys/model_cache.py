from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Callable, Iterator
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import torch


_download_progress_callback: ContextVar[Callable[[str], None] | None] = ContextVar(
    "emcfsys_model_download_progress_callback",
    default=None,
)


def project_model_cache_dir() -> Path:
    """Return the project-local directory used for downloaded model weights."""

    cache_dir = Path(__file__).resolve().parents[2] / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def resolve_pretrained_weight_path(
    filename: str,
    explicit_path: str | Path | None = None,
) -> Path | None:
    """Resolve project cache first, then an explicitly supplied weight path."""

    cached_path = project_model_cache_dir() / filename
    if cached_path.is_file():
        return cached_path

    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained weight file not found: {path}")
        return path
    return None


@contextmanager
def model_download_progress(
    callback: Callable[[str], None] | None,
) -> Iterator[None]:
    """Temporarily route model-download progress messages to ``callback``."""

    token = _download_progress_callback.set(callback)
    try:
        yield
    finally:
        _download_progress_callback.reset(token)


def _format_bytes(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def _emit_download_progress(message: str) -> None:
    callback = _download_progress_callback.get()
    if callback is not None:
        callback(message)


def _download_to_project_cache(url: str, destination: Path) -> None:
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    temporary_path.unlink(missing_ok=True)
    request = Request(url, headers={"User-Agent": "EMCFsys"})

    try:
        with urlopen(request) as response, temporary_path.open("wb") as output:
            content_length = response.headers.get("Content-Length")
            total_bytes = int(content_length) if content_length else None
            downloaded_bytes = 0
            last_percent = -5
            last_reported_bytes = 0
            _emit_download_progress(
                f"Downloading model {destination.name}: "
                "[--------------------] 0.0%"
            )

            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                downloaded_bytes += len(chunk)

                if total_bytes:
                    percent = min(100, int(downloaded_bytes * 100 / total_bytes))
                    if percent >= last_percent + 5 or percent == 100:
                        filled = min(20, percent // 5)
                        bar = "#" * filled + "-" * (20 - filled)
                        _emit_download_progress(
                            f"Downloading model {destination.name}: [{bar}] "
                            f"{percent:.1f}% ({_format_bytes(downloaded_bytes)} / "
                            f"{_format_bytes(total_bytes)})"
                        )
                        last_percent = percent
                elif downloaded_bytes >= last_reported_bytes + 16 * 1024 * 1024:
                    _emit_download_progress(
                        f"Downloading model {destination.name}: "
                        f"{_format_bytes(downloaded_bytes)} downloaded"
                    )
                    last_reported_bytes = downloaded_bytes

        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    _emit_download_progress(f"Model download complete: {destination}")


def _load_cached_state_dict(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_state_dict_from_project_url(
    url: str,
    *,
    map_location="cpu",
    progress: bool = True,
    check_hash: bool = False,
    file_name: str | None = None,
):
    """Load a cloud checkpoint from the project cache, downloading with UI progress."""

    del progress, check_hash
    filename = file_name or Path(urlparse(url).path).name
    if not filename:
        raise ValueError(f"Could not determine a checkpoint filename from URL: {url}")

    cached_path = project_model_cache_dir() / filename
    if not cached_path.is_file():
        _download_to_project_cache(url, cached_path)
    return _load_cached_state_dict(cached_path, map_location)
