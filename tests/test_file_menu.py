"""The File menu: bringing a mesh in, and writing every map back out.

Both ends of the app used to live inside the Bake tab, next to the curvature
sliders -- which made an export look like part of baking, when it takes what the
Texture and Decal tabs produced too. They are one menu now, and an export asks
only where to put things: there is nothing to pick, because the maps are one
material split across several files.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import trimesh
from imgui_bundle import imgui

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ui import panel  # noqa: E402


class FakeDialog:
    """A native dialog, answered one frame after it is opened.

    The real ones are asynchronous -- opened on one frame, polled on later ones
    while the user is still looking at them -- so what the panel depends on is
    this pair of methods rather than the platform's window. Answering on the
    second poll rather than the first is what keeps that distinction honest.
    """

    def __init__(self, answer):
        self.answer = answer
        self.polls = 0

    def ready(self) -> bool:
        self.polls += 1
        return self.polls > 1

    def result(self):
        return self.answer


@pytest.fixture
def app(tmp_path):
    """The real app, headless, with a bake behind it."""
    import moderngl_window as mglw

    from render.viewport import MeshMapApp

    mesh = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        process=False,
        visual=trimesh.visual.TextureVisuals(
            uv=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        ),
    )
    mesh.export(tmp_path / "quad.obj")

    MeshMapApp.initial_mesh = str(tmp_path / "quad.obj")
    MeshMapApp.initial_resolution = 256
    try:
        instance = mglw.create_window_config_instance(MeshMapApp, args=["-wnd", "headless"])
    except Exception as exc:  # pragma: no cover - depends on the host
        pytest.skip(f"no headless window available: {exc}")

    instance.request_bake()
    deadline = time.monotonic() + 30.0
    while instance.controller.running and time.monotonic() < deadline:
        instance.controller.pump()
        time.sleep(0.002)
    instance.on_render(0.0, 1 / 60.0)

    yield instance
    instance.controller.release()
    imgui.destroy_context()
    MeshMapApp.initial_mesh = None


def drawn(app) -> None:
    """One panel frame, which is what polls the dialogs."""
    imgui.new_frame()
    panel.draw_panel(app)
    imgui.end_frame()


# --------------------------------------------------------------------------
# what an export would write
# --------------------------------------------------------------------------

def test_every_map_is_listed_once_there_is_something_to_write(app):
    """A standard PBR set. No subset to choose from: they describe one surface
    between them."""
    maps = panel.exportable_maps(app)
    assert maps == ["color", "metallic", "roughness", "ao"]
    assert "normal" not in maps, "no decal has been imported"


def test_the_listed_maps_are_the_ones_written(app, tmp_path):
    """The Settings tab and the File menu both promise this list; an export
    that wrote something else would make both of them lie."""
    app.export_dir = str(tmp_path / "out")
    app.export()

    written = sorted(p.name for p in (tmp_path / "out" / "quad").glob("*.png"))
    assert written == sorted(f"{name}.png" for name in panel.exportable_maps(app))


def test_a_decal_adds_the_normal_map(app, tmp_path):
    from PIL import Image

    decal = tmp_path / "vent.png"
    Image.fromarray(
        np.tile(np.array([230, 128, 255], np.uint8), (16, 16, 1))
    ).save(decal)
    app.open_decal(decal)
    app.on_render(0.0, 1 / 60.0)

    assert "normal" in panel.exportable_maps(app)


def test_nothing_is_listed_before_a_bake(app):
    app.controller.curvature_map = None
    app.controller.occlusion_map = None
    assert panel.exportable_maps(app) == []


# --------------------------------------------------------------------------
# exporting: where, then everything
# --------------------------------------------------------------------------

def test_exporting_asks_where_and_then_writes_there(app, tmp_path, monkeypatch):
    """The whole gesture, in the order the user performs it."""
    chosen = tmp_path / "chosen"
    opened = []

    def fake_select_folder(title, start):
        opened.append((title, start))
        return FakeDialog(str(chosen))

    monkeypatch.setattr(panel.pfd, "select_folder", fake_select_folder)

    app.request_export()
    drawn(app)  # asks where, and waits on the answer

    assert opened, "an export has to ask before it writes anything"
    assert opened[0][1] == app.export_dir, "opening where the last one went"
    assert not chosen.exists(), "and writes nothing until it is answered"

    drawn(app)  # the answer arrives, and the maps follow

    written = sorted(p.name for p in (chosen / "quad").glob("*.png"))
    assert written == ["ao.png", "color.png", "metallic.png", "roughness.png"]
    assert app.export_dir == str(chosen), "the next export starts from here"


def test_a_cancelled_export_writes_nothing(app, tmp_path, monkeypatch):
    before = app.export_dir
    monkeypatch.setattr(
        panel.pfd, "select_folder", lambda *_: FakeDialog("")  # dismissed
    )

    app.request_export()
    for _ in range(3):
        drawn(app)

    assert app.export_dir == before
    assert not (tmp_path / "quad").exists(), "no folder, no maps"


def test_the_e_key_asks_the_same_question_as_the_menu(app, monkeypatch):
    """It used to write straight into a folder the user had never chosen, which
    is a surprising thing for a single keystroke to do."""
    keys = app.wnd.keys
    if keys.E == keys.B:
        # The headless backend leaves every letter key undefined, so they all
        # compare equal and there is no way to press one in particular.
        pytest.skip("this window backend has no letter keys")

    opened = []
    monkeypatch.setattr(
        panel.pfd, "select_folder", lambda *args: opened.append(args) or FakeDialog("")
    )

    app.on_key_event(keys.E, keys.ACTION_PRESS, app.wnd.modifiers)
    assert app.export_pending, "the key asks; the panel does the asking"

    drawn(app)
    assert opened, "and the folder chooser is what it asks with"


def test_the_menu_and_the_key_go_through_one_door(app):
    """Both set the same flag, so neither can drift into writing files without
    asking. Checked here because the backend cannot press the key itself."""
    from render.viewport import MeshMapApp

    assert not app.export_pending
    MeshMapApp.request_export(app)
    assert app.export_pending


def test_exporting_with_nothing_to_write_says_so(app, monkeypatch):
    """Rather than opening a chooser for an export that would write no files."""
    app.controller.curvature_map = None
    app.controller.occlusion_map = None
    app.remove_texture()
    opened = []
    monkeypatch.setattr(
        panel.pfd, "select_folder", lambda *args: opened.append(args) or FakeDialog("")
    )

    app.request_export()
    drawn(app)

    assert not opened
    assert app.status_is_error and "Nothing to export" in app.status


# --------------------------------------------------------------------------
# importing
# --------------------------------------------------------------------------

def test_a_chosen_mesh_is_opened(app, tmp_path):
    """The same dialog the Bake tab's button used to open, from the File menu."""
    other = tmp_path / "second.obj"
    trimesh.creation.box(extents=(1, 1, 1)).export(other)

    app.file_dialog = FakeDialog([str(other)])
    drawn(app)
    drawn(app)

    assert app.file_dialog is None
    assert Path(app.mesh_info.path).name == "second.obj"


def test_the_panel_draws_with_the_menu_in_it(app):
    """The navbar, the tabs and the status bar, all in one frame -- the menu is
    drawn inside the navbar window, and getting that wrong unbalances ImGui's
    window stack rather than raising anything obvious."""
    for _ in range(2):
        drawn(app)  # must not raise


# --------------------------------------------------------------------------
# choosing which maps to write
# --------------------------------------------------------------------------

def test_a_map_switched_off_is_not_written(app, tmp_path):
    app.set_map_enabled("metallic", False)
    app.set_map_enabled("ao", False)

    app.export_dir = str(tmp_path / "out")
    app.export()

    written = sorted(p.name for p in (tmp_path / "out" / "quad").glob("*.png"))
    assert written == ["color.png", "roughness.png"]


def test_metallic_and_roughness_can_be_asked_for_separately(app, tmp_path):
    """They arrive together in one array, because they travel together through
    the compositor -- which is not a reason to have to take both."""
    for name in ("color", "ao", "metallic"):
        app.set_map_enabled(name, False)

    app.export_dir = str(tmp_path / "out")
    app.export()

    written = [p.name for p in (tmp_path / "out" / "quad").glob("*.png")]
    assert written == ["roughness.png"]


def test_the_listing_follows_the_switches(app):
    assert "ao" in panel.exportable_maps(app)
    app.set_map_enabled("ao", False)
    assert "ao" not in panel.exportable_maps(app)
    assert panel.exportable_maps(app), "the others are still there"


def test_switching_everything_off_leaves_nothing_to_export(app, monkeypatch):
    """The File menu greys out, and the export refuses, on the same answer --
    so neither can promise what the other will not do."""
    for name, _ in panel._MAP_LABELS:
        app.set_map_enabled(name, False)
    assert panel.exportable_maps(app) == []

    opened = []
    monkeypatch.setattr(
        panel.pfd, "select_folder", lambda *args: opened.append(args) or FakeDialog("")
    )
    app.request_export()
    drawn(app)

    assert not opened
    assert app.status_is_error and "Nothing to export" in app.status


def test_turning_occlusion_off_takes_it_out_of_the_bake_too(app):
    """The other four cost nothing to produce, so switching them off only stops
    a file being written. This one is most of what a bake costs."""
    assert app.controller.bake_occlusion

    app.set_map_enabled("ao", False)
    assert not app.controller.bake_occlusion
    assert "occlusion" in app.controller.pending_stages()

    app.set_map_enabled("ao", True)
    assert app.controller.bake_occlusion


def test_the_choice_is_remembered_between_launches(app):
    from render.viewport import _load_prefs

    app.set_map_enabled("normal", False)
    app.set_map_enabled("ao", False)

    stored = set(_load_prefs()["maps"])
    assert stored == {"color", "metallic", "roughness"}


def test_a_malformed_stored_choice_falls_back_to_everything():
    """A hand-edited typo should cost a tick box, not every map the session
    would otherwise have written."""
    from core.export import MAP_NAMES
    from render.viewport import _stored_maps

    assert _stored_maps(None) == set(MAP_NAMES)
    assert _stored_maps("ao") == set(MAP_NAMES)
    assert _stored_maps([]) == set(MAP_NAMES)
    assert _stored_maps(["nonsense"]) == set(MAP_NAMES)
    assert _stored_maps(["ao", "nonsense"]) == {"ao"}
