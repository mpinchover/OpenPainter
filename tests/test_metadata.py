"""What the application calls itself, and the icon it wears.

Branding is the thing most likely to be edited by someone who is not otherwise
touching the source, so it lives in a file at the project root -- and nothing in
that file can stop the app opening. These tests are mostly about that: every bad
input falls back rather than raising, because a typo in a name is not a reason
to refuse to start.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.metadata import (  # noqa: E402
    DEFAULT_TITLE,
    METADATA,
    METADATA_FILE,
    Metadata,
    load_metadata,
)


def write(tmp_path: Path, contents, name: str = "metadata.json") -> Path:
    path = tmp_path / name
    path.write_text(contents if isinstance(contents, str) else json.dumps(contents))
    return path


# --------------------------------------------------------------------------
# reading it
# --------------------------------------------------------------------------

def test_the_title_and_icon_are_read(tmp_path):
    icon = tmp_path / "logo.png"
    icon.write_bytes(b"not really a png, but it is a file")
    path = write(tmp_path, {"title": "OpenPainter", "ApplicationIcon": "./logo.png"})

    data = load_metadata(path)
    assert data.title == "OpenPainter"
    assert data.icon == icon.resolve()


def test_a_relative_icon_is_found_beside_the_metadata_not_the_shell(tmp_path):
    """Launched from anywhere, "./assets/x.png" has to mean the same file."""
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "icon.png").write_bytes(b"file")
    path = write(tmp_path, {"ApplicationIcon": "./assets/icon.png"})

    assert load_metadata(path).icon == (assets / "icon.png").resolve()


def test_an_absolute_icon_path_is_taken_as_it_is(tmp_path):
    icon = tmp_path / "elsewhere.png"
    icon.write_bytes(b"file")
    path = write(tmp_path, {"ApplicationIcon": str(icon)}, name="meta.json")

    assert load_metadata(path).icon == icon.resolve()


def test_the_icon_key_is_application_icon(tmp_path):
    icon = tmp_path / "logo.png"
    icon.write_bytes(b"file")
    path = write(tmp_path, {"application_icon": "logo.png"})
    assert load_metadata(path).icon == icon.resolve()


def test_the_name_the_field_had_first_is_still_read(tmp_path):
    """A key this file does not recognise is an icon that silently does not
    appear, which is a bad way to find out the field has been renamed."""
    icon = tmp_path / "logo.png"
    icon.write_bytes(b"file")
    path = write(tmp_path, {"ApplicationIcon": "logo.png"}, name="old.json")
    assert load_metadata(path).icon == icon.resolve()


def test_an_unrecognised_key_is_not_guessed_at(tmp_path):
    (tmp_path / "logo.png").write_bytes(b"file")
    for key in ("appIcon", "icon", "Application_Icon"):
        path = write(tmp_path, {key: "logo.png"}, name=f"{key}.json")
        assert load_metadata(path).icon is None, key


def test_the_shipped_file_uses_the_name_the_tests_document():
    """If the two ever disagree, the icon disappears and nothing says why."""
    stored = json.loads(METADATA_FILE.read_text())
    assert "application_icon" in stored, sorted(stored)


# --------------------------------------------------------------------------
# and surviving it
# --------------------------------------------------------------------------

def test_no_file_at_all_is_the_defaults(tmp_path):
    data = load_metadata(tmp_path / "absent.json")
    assert data == Metadata()
    assert data.title == DEFAULT_TITLE


def test_a_malformed_file_is_the_defaults(tmp_path):
    for contents in ("{not json", "", "[]", '"a string"', "null"):
        path = write(tmp_path, contents, name="broken.json")
        assert load_metadata(path) == Metadata(), repr(contents)


def test_a_title_that_is_not_a_name_falls_back(tmp_path):
    for title in (None, 42, "", "   ", [], {}):
        path = write(tmp_path, {"title": title}, name="odd.json")
        assert load_metadata(path).title == DEFAULT_TITLE, repr(title)


def test_surrounding_space_is_not_part_of_the_name(tmp_path):
    path = write(tmp_path, {"title": "  OpenPainter \n"})
    assert load_metadata(path).title == "OpenPainter"


def test_an_icon_that_is_not_there_is_no_icon(tmp_path):
    """Checked when it is read rather than when the window opens, so a path
    pointing at nothing is simply no icon instead of an error at launch."""
    path = write(tmp_path, {"ApplicationIcon": "./missing.png"})
    assert load_metadata(path).icon is None

    directory = tmp_path / "adirectory.png"
    directory.mkdir()
    path = write(tmp_path, {"ApplicationIcon": "adirectory.png"}, name="dir.json")
    assert load_metadata(path).icon is None


def test_one_bad_field_does_not_take_the_other_with_it(tmp_path):
    icon = tmp_path / "logo.png"
    icon.write_bytes(b"file")
    path = write(tmp_path, {"title": 7, "ApplicationIcon": "logo.png"})

    data = load_metadata(path)
    assert data.title == DEFAULT_TITLE
    assert data.icon == icon.resolve()


# --------------------------------------------------------------------------
# what the app does with it
# --------------------------------------------------------------------------

def test_the_project_ships_a_readable_metadata_file():
    assert METADATA_FILE.is_file(), METADATA_FILE
    assert METADATA.title, "the window would have no name"


def test_the_window_takes_its_title_from_the_file():
    """moderngl-window reads the class attribute before there is an instance to
    ask, so this is where the name has to arrive."""
    from render.viewport import MeshMapApp

    assert MeshMapApp.title == METADATA.title


def test_the_shipped_icon_is_where_the_file_says(tmp_path):
    shipped = load_metadata(METADATA_FILE)
    if shipped.icon is None:
        pytest.skip("this checkout ships no application icon")
    assert shipped.icon.is_file()
    assert shipped.icon.is_absolute(), "resolved, so the finders are bypassed"
