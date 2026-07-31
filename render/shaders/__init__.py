"""GLSL sources, loaded from disk so they stay editable without a rebuild.

GLSL has no include of its own, so :func:`load_shader` provides one: a line
reading ``#include "noise.glsl"`` is replaced by that file's text. It exists for
exactly one reason -- the ArmorPaint noise is sampled by both the wear pass and
the mask tree, and two copies of a ported algorithm is two things to keep in
step and one to eventually get wrong.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

SHADER_DIR = Path(__file__).resolve().parent

_INCLUDE = re.compile(r'^[ \t]*#include[ \t]+"([^"]+)"[ \t]*$', re.MULTILINE)

#: Guards a cycle, and an include chain deep enough to be a mistake.
_MAX_INCLUDE_DEPTH = 8


@lru_cache(maxsize=None)
def load_shader(name: str, _depth: int = 0) -> str:
    """Read a shader file out of ``render/shaders`` by filename, includes and all."""
    path = SHADER_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Shader not found: {path}")
    source = path.read_text(encoding="utf-8")

    if _depth >= _MAX_INCLUDE_DEPTH:
        raise RecursionError(f"#include nested past {_MAX_INCLUDE_DEPTH} in {name}")

    return _INCLUDE.sub(
        lambda match: load_shader(match.group(1), _depth + 1).rstrip("\n"), source
    )
