"""Shared fixtures.

The important one keeps the suite out of the real user's settings. Several
tests build a whole :class:`~render.viewport.MeshMapApp`, and the app saves
preferences the moment anything touches a setting -- UI scale, navigation
speeds -- into the platform's config directory. A test run has no business
rewriting the scale someone picked, and a suite that reads it back is not
deterministic either.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Point the app's settings directory at a throwaway one."""
    import render.viewport as viewport

    settings = tmp_path / "settings"
    settings.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(viewport, "_settings_dir", lambda: settings)
    return settings
