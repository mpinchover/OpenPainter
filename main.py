"""Curvature map baking tool.

    python main.py [mesh] [--z-up] [--resolution N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import moderngl_window as mglw  # noqa: E402

from core.params import RESOLUTIONS, BevelParams  # noqa: E402
from render.viewport import MeshMapApp  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mesh",
        nargs="?",
        help="Mesh to load on startup (.fbx, .obj, .glb, ...). Without one the "
             "app opens on a starter cube; files can also be dropped onto the "
             "window.",
    )
    parser.add_argument(
        "--z-up",
        action="store_true",
        help="The source file is already Z-up, so import it as-is. Off by "
             "default, which rotates a Y-up source into this app's Z-up world "
             "-- what Blender's FBX and glTF exporters write.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=RESOLUTIONS,
        default=1024,
        help="Initial bake resolution (default: 1024).",
    )
    parser.add_argument(
        "--bevel",
        nargs="?",
        const="0.001,3,30",
        metavar="AMOUNT,SEGMENTS,ANGLE",
        help="Bevel sharp edges before baking, so the curvature has geometry to "
             "land on. Defaults to 0.001,3,30 -- 1 mm, 3 segments, edges sharper "
             "than 30 degrees. Mirrors Blender's Bevel modifier under Offset "
             "width and the Angle limit method.",
    )
    parser.add_argument(
        "--auto-unwrap",
        action="store_true",
        help="Let xatlas generate a fresh atlas instead of baking into the "
             "mesh's own UV map. Only for a mesh with no UVs -- the resulting "
             "texture fits the exported OBJ, not your original mesh.",
    )
    parser.add_argument(
        "--ui-scale",
        type=float,
        default=None,
        help="Size of the control panel and its text (default: 1.35, or whatever "
             "was last set in the app). Also adjustable live under Interface.",
    )
    parser.add_argument(
        "--selftest",
        metavar="DIR",
        help="Run the whole pipeline offscreen: bake, export the maps, and "
             "save a viewport render to DIR, then exit. Needs a mesh argument.",
    )
    args = parser.parse_args(argv)
    args.bevel_params = _parse_bevel(parser, args.bevel)
    return args


def _parse_bevel(parser: argparse.ArgumentParser, spec: str | None) -> BevelParams:
    """Turn ``AMOUNT,SEGMENTS,ANGLE`` into a :class:`BevelParams`."""
    if spec is None:
        return BevelParams(enabled=False)
    fields = [part.strip() for part in spec.split(",")]
    if len(fields) != 3:
        parser.error(f"--bevel wants AMOUNT,SEGMENTS,ANGLE, got {spec!r}")
    try:
        amount, segments, angle = float(fields[0]), int(fields[1]), float(fields[2])
    except ValueError:
        parser.error(f"--bevel values must be numbers, got {spec!r}")
    if amount <= 0 or segments < 1 or not 0 < angle <= 180:
        parser.error(f"--bevel needs amount > 0, segments >= 1, 0 < angle <= 180: {spec!r}")
    return BevelParams(enabled=True, amount=amount, segments=segments, angle=angle)


def run_selftest(args: argparse.Namespace) -> int:
    """Drive the real app through a headless context and dump the results.

    This exercises the same code path the window does -- bake controller,
    shaping pass, preview shader, ImGui -- without needing a display. With no
    mesh argument it runs on the same starter cube the window opens on.
    """
    from PIL import Image

    output = Path(args.selftest).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    MeshMapApp.initial_mesh = args.mesh
    MeshMapApp.initial_z_up = args.z_up
    MeshMapApp.initial_resolution = args.resolution
    MeshMapApp.initial_auto_unwrap = args.auto_unwrap
    MeshMapApp.initial_bevel = args.bevel_params
    if args.ui_scale is not None:
        MeshMapApp.initial_ui_scale = args.ui_scale
        MeshMapApp.initial_ui_scale_explicit = True

    app = mglw.create_window_config_instance(MeshMapApp, args=["-wnd", "headless"])
    if app.mesh is None:
        print(f"error: {app.status}", file=sys.stderr)
        return 1

    app.request_bake()
    if not app.controller.running:
        print(f"error: nothing to bake ({app.status})", file=sys.stderr)
        return 1

    while app.controller.running:
        app.controller.pump()
        # The AO stage runs on a worker thread. Busy-spinning here would fight
        # it for the GIL; the real render loop pumps once a frame instead.
        time.sleep(0.005)
    if app.controller.error:
        print(f"error: bake failed\n{app.controller.error}", file=sys.stderr)
        return 1

    app.export_dir = str(output)
    from render.viewport import PREVIEW_MODES

    for index, mode in enumerate(PREVIEW_MODES):
        app.preview_index = index
        mode = mode.label.lower()
        # ImGui needs a couple of frames to size and place its windows before
        # they show up in a capture.
        for _ in range(3):
            app.on_render(0.0, 0.016)
        pixels = app.wnd.fbo.read(components=3)
        image = Image.frombytes("RGB", app.wnd.buffer_size, pixels).transpose(
            Image.FLIP_TOP_BOTTOM
        )
        image.save(output / f"viewport_{mode}.png")

    app.export()
    print(app.status)
    for stage, seconds in app.controller.timings.items():
        print(f"  {stage:<10} {seconds:6.2f}s")
    print(f"viewport renders and maps written to {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.mesh and not Path(args.mesh).expanduser().exists():
        print(f"error: no such file: {args.mesh}", file=sys.stderr)
        return 2

    if args.selftest:
        return run_selftest(args)

    MeshMapApp.initial_mesh = args.mesh
    MeshMapApp.initial_z_up = args.z_up
    MeshMapApp.initial_resolution = args.resolution
    MeshMapApp.initial_auto_unwrap = args.auto_unwrap
    MeshMapApp.initial_bevel = args.bevel_params
    if args.ui_scale is not None:
        MeshMapApp.initial_ui_scale = args.ui_scale
        MeshMapApp.initial_ui_scale_explicit = True

    # moderngl-window runs its own argparse over sys.argv for window flags, and
    # its override is `args or sys.argv[1:]` -- so an empty list falls straight
    # back through. Clearing sys.argv is what actually keeps it off our
    # already-consumed arguments.
    sys.argv = sys.argv[:1]
    mglw.run_window_config(MeshMapApp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
