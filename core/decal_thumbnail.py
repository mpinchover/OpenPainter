"""Small, persistent previews for the decal shelf."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


THUMBNAIL_EDGE = 256
_CACHE_VERSION = 1


@dataclass(frozen=True)
class DecalThumbnail:
    path: str
    size: tuple[int, int]
    rgba: bytes


def thumbnail_cache_key(path: str | Path) -> str:
    """Identify a source revision without reading its image payload."""
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    identity = (
        f"{_CACHE_VERSION}\0{source}\0{stat.st_mtime_ns}\0{stat.st_size}"
    ).encode("utf-8")
    return hashlib.sha256(identity).hexdigest()


def load_thumbnail(path: str | Path, cache_dir: str | Path) -> DecalThumbnail:
    """Load a cached shelf preview, generating it when the source changed."""
    source = Path(path).expanduser()
    cache_dir = Path(cache_dir)
    cached = cache_dir / f"{thumbnail_cache_key(source)}.png"

    try:
        with Image.open(cached) as opened:
            preview = opened.convert("RGBA")
            preview.load()
    except (OSError, FileNotFoundError):
        with Image.open(source) as opened:
            preview = ImageOps.exif_transpose(opened).convert("RGBA")
            preview.thumbnail((THUMBNAIL_EDGE, THUMBNAIL_EDGE), Image.Resampling.LANCZOS)
            preview.load()
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = cached.with_suffix(f".{os.getpid()}.tmp")
            preview.save(temporary, format="PNG", optimize=True)
            temporary.replace(cached)
        except OSError:
            # A read-only or full cache must not make the asset unusable.
            pass

    return DecalThumbnail(str(source), preview.size, preview.tobytes())
