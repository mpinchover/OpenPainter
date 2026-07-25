"""GLSL sources, loaded from disk so they stay editable without a rebuild."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

SHADER_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load_shader(name: str) -> str:
    """Read a shader file out of ``render/shaders`` by filename."""
    path = SHADER_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Shader not found: {path}")
    return path.read_text(encoding="utf-8")
