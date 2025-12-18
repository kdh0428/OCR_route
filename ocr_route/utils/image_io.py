"""Image loading helpers supporting local paths, URLs, and PIL images."""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from PIL import Image

LOGGER = logging.getLogger(__name__)


def is_url(path: str) -> bool:
    """Return True if the provided string is an HTTP(S) URL."""
    try:
        parsed = urlparse(path)
        return parsed.scheme in {"http", "https"}
    except Exception:
        return False


def load_image(source: Any) -> Image.Image:
    """Load an image from a path, URL, or PIL Image.

    Raises:
        ValueError: If the source cannot be resolved to an image.
    """
    if isinstance(source, Image.Image):
        return source.convert("RGB")

    if isinstance(source, (str, Path)):
        path_str = str(source)
        if is_url(path_str):
            try:
                with urlopen(path_str) as response:
                    data = response.read()
            except Exception as exc:  # pragma: no cover - network errors
                raise ValueError(f"Failed to download image from URL: {path_str}") from exc
            return Image.open(io.BytesIO(data)).convert("RGB")

        file_path = Path(path_str)
        if not file_path.exists():
            raise ValueError(f"Image path does not exist: {file_path}")
        try:
            return Image.open(file_path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Failed to open image: {file_path}") from exc

    raise ValueError(f"Unsupported image source: {type(source)!r}")
