"""What the application calls itself, read from ``metadata.json``.

The name in the title bar and the icon in the dock are branding, not behaviour,
and they are the two things most likely to be changed by someone who is not
otherwise editing the source. So they live in a file at the project root:

.. code-block:: json

    {
        "title": "OpenPainter",
        "application_icon": "./assets/application_icon.png",
        "decals": "./assets/decals"
    }

Every field is optional and nothing here can stop the app opening. A file that
is missing, unreadable, malformed or full of the wrong types falls back to the
defaults below, because a typo in a branding file is not a reason to refuse to
start.

Paths are resolved against the metadata file's own directory, so the relative
form above means what it looks like it means wherever the app is launched from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

#: Used when the file says nothing. The name the project had before it had a
#: metadata file at all.
DEFAULT_TITLE = "MeshMap - Edge Wear"

#: Where the file lives: the project root, beside main.py.
METADATA_FILE = Path(__file__).resolve().parent.parent / "metadata.json"

#: Spellings accepted for the icon, the first being the name. The second is what
#: the field was called to begin with, kept because a key this file does not
#: recognise is an icon that silently does not appear -- a bad way to find out
#: about a rename. Anything else is a typo and should read as one.
_ICON_KEYS = ("application_icon", "ApplicationIcon")


@dataclass(frozen=True)
class Metadata:
    """The application's own name and face."""

    title: str = DEFAULT_TITLE
    icon: Optional[Path] = None
    """An icon file that exists, or None. Checked when it is read rather than
    when it is used, so a path pointing at nothing is simply no icon instead of
    an error at the moment the window opens."""

    decals: Optional[Path] = None
    """A directory of decal images to offer in the Decal tab, or None. Checked
    the same way and for the same reason: a library that is not there is an
    empty shelf, not a crash."""


def load_metadata(path: str | Path = METADATA_FILE) -> Metadata:
    """Read ``metadata.json``, falling back to the defaults for anything absent."""
    path = Path(path)
    try:
        stored = json.loads(path.read_text())
    except (OSError, ValueError):
        return Metadata()
    if not isinstance(stored, dict):
        return Metadata()

    title = stored.get("title")
    if not isinstance(title, str) or not title.strip():
        title = DEFAULT_TITLE

    return Metadata(
        title=title.strip(),
        icon=_resolve(stored, _ICON_KEYS, path.parent, Path.is_file),
        decals=_resolve(stored, ("decals",), path.parent, Path.is_dir),
    )


def _resolve(stored: dict, keys, root: Path, exists) -> Optional[Path]:
    """The first of ``keys`` naming something that is there, or None.

    Relative paths are resolved against ``root`` -- the metadata file's own
    directory -- so they mean the same thing wherever the app is launched from.
    """
    for key in keys:
        value = stored.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value.strip()).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        if exists(candidate):
            return candidate
    return None


#: Read once, at import. The title is a class attribute on the window config,
#: which moderngl-window reads before there is an instance to ask.
METADATA = load_metadata()
