"""Tests for the bake orchestration.

The controller is what makes the tool feel responsive: it must re-run only the
stages whose inputs actually changed, keep GL work on the GL thread, and survive
being cancelled halfway through.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import moderngl  # noqa: E402

from core.mesh_io import load_mesh  # noqa: E402
from core.pipeline import STAGES, BakeController  # noqa: E402

ASSETS = ROOT / "assets"
ALL_STAGES = list(STAGES)


@pytest.fixture(scope="module")
def ctx():
    try:
        context = moderngl.create_standalone_context(require=330)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"no GL context available: {exc}")
    yield context
    context.release()


@pytest.fixture
def controller(ctx):
    # The sample assets carry no UVs, so unwrap once here to stand in for a mesh
    # exported out of Blender with its own UV map -- that is what the controller
    # bakes into by default.
    mesh, _ = load_mesh(ASSETS / "sample.obj")
    controller = BakeController(ctx)
    controller.bake_params.resolution = 128
    controller.set_mesh(mesh.unwrap())
    yield controller
    controller.release()


@pytest.fixture
def uvless_controller(ctx):
    mesh, _ = load_mesh(ASSETS / "sample.obj")
    controller = BakeController(ctx)
    controller.bake_params.resolution = 128
    controller.set_mesh(mesh)
    yield controller
    controller.release()


def run_to_completion(controller: BakeController, timeout: float = 120.0) -> None:
    """Pump the controller the way the render loop does, until it settles."""
    deadline = time.time() + timeout
    while controller.running:
        controller.pump()
        if time.time() > deadline:
            raise TimeoutError(f"bake stuck in stage {controller.stage!r}")
        time.sleep(0.001)
    assert controller.error is None, controller.error


def test_fresh_mesh_needs_every_stage(controller):
    assert controller.pending_stages() == ALL_STAGES
    assert controller.is_dirty()


def test_full_bake_produces_the_maps(controller):
    assert controller.request_bake()
    run_to_completion(controller)

    assert controller.curvature_map is not None
    assert controller.position_map is not None
    assert controller.curvature_map.shape == (128, 128)
    assert controller.position_map.shape == (128, 128, 3)
    assert controller.maps_version == 1
    assert controller.progress == pytest.approx(1.0)

    assert 0.0 <= controller.curvature_map.min()
    assert controller.curvature_map.max() <= 1.0
    assert controller.curvature_map.max() > 0.0, "the bracket's edges should register"

    # bposition is normalised over the bounding box, so it stays in 0..1.
    covered = controller.gbuffer.mask
    inside = controller.position_map[covered]
    assert inside.min() >= -1e-4 and inside.max() <= 1.0 + 1e-4

    assert set(controller.timings) == set(ALL_STAGES)


def test_the_default_bake_uses_the_meshs_own_uvs(controller):
    """No reindexing: the atlas is the source layout, vertex for vertex."""
    assert controller.unwrap_params.use_source_uvs
    controller.request_bake()
    run_to_completion(controller)

    result = controller.unwrap_result
    assert result.source == "source"
    assert len(result.vertices) == len(controller.mesh.vertices)
    assert np.array_equal(result.faces, controller.mesh.faces)
    assert np.allclose(result.uvs, controller.mesh.visual.uv, atol=1e-6)
    assert np.array_equal(result.vmapping, np.arange(len(controller.mesh.vertices)))


def test_source_uvs_bake_normals_stay_smooth_across_seams(controller):
    """A UV seam on a smooth surface must not read as a crease.

    Vertices split for UVs sit at one position, so their normals have to match;
    if they did not, the bake would draw a wear line down every seam.
    """
    controller.request_bake()
    run_to_completion(controller)

    mesh = controller.mesh
    normals = controller.unwrap_result.normals
    _, inverse = trimesh.grouping.unique_rows(mesh.vertices)
    for group in range(inverse.max() + 1):
        members = normals[inverse == group]
        if len(members) > 1:
            assert np.allclose(members, members[0], atol=1e-5)


def test_source_uvs_are_required_by_default(uvless_controller):
    """A mesh with no UVs fails loudly rather than silently inventing an atlas."""
    uvless_controller.request_bake()
    while uvless_controller.running:
        uvless_controller.pump()
        time.sleep(0.001)

    assert uvless_controller.error is not None
    assert "no UV map" in uvless_controller.error
    assert uvless_controller.curvature_map is None


def test_xatlas_fallback_still_bakes_a_uvless_mesh(uvless_controller):
    uvless_controller.unwrap_params.use_source_uvs = False
    uvless_controller.request_bake()
    run_to_completion(uvless_controller)

    assert uvless_controller.unwrap_result.source == "xatlas"
    assert uvless_controller.curvature_map is not None
    assert uvless_controller.curvature_map.max() > 0.0


def test_toggling_the_uv_source_reruns_everything(controller):
    controller.request_bake()
    run_to_completion(controller)
    assert controller.pending_stages() == []

    controller.unwrap_params.use_source_uvs = False
    assert controller.pending_stages() == ALL_STAGES


def test_nothing_to_do_after_a_clean_bake(controller):
    controller.request_bake()
    run_to_completion(controller)

    assert controller.pending_stages() == []
    assert not controller.is_dirty()
    assert controller.request_bake() is False


def test_wear_params_never_dirty_the_bake(controller):
    """This is the split that keeps slider tweaking interactive."""
    controller.request_bake()
    run_to_completion(controller)

    # EdgeWearParams live on the app, not the controller, precisely so that
    # touching them cannot invalidate a bake stage.
    assert not hasattr(controller, "wear_params")
    assert controller.pending_stages() == []


@pytest.mark.parametrize(
    "field, value",
    [("strength", 1.5), ("radius", 3.0), ("offset", 0.5), ("smooth", 3), ("axis", 2)],
)
def test_curvature_params_rerun_the_bake_and_the_padding(controller, field, value):
    controller.request_bake()
    run_to_completion(controller)

    setattr(controller.bake_params, field, value)
    assert controller.pending_stages() == ["curvature", "post"], f"{field} mis-keyed"


def test_dilation_change_reruns_only_padding(controller):
    controller.request_bake()
    run_to_completion(controller)

    controller.bake_params.dilation = 8
    assert controller.pending_stages() == ["post"]


def test_repadding_starts_from_the_gbuffer(controller):
    """Re-running the padding must not pad an already-padded map again."""
    controller.bake_params.dilation = 2
    controller.request_bake()
    run_to_completion(controller)
    outside = ~controller.gbuffer.mask
    once = controller.curvature_map[outside].copy()

    # Re-run padding at the same width. Same input, so it must be a no-op.
    controller._completed.pop("post")
    controller.request_bake()
    run_to_completion(controller)
    assert np.array_equal(controller.curvature_map[outside], once)


def test_resolution_change_reruns_everything(controller):
    controller.request_bake()
    run_to_completion(controller)

    controller.bake_params.resolution = 64
    assert controller.pending_stages() == ALL_STAGES

    controller.request_bake()
    run_to_completion(controller)
    assert controller.curvature_map.shape == (64, 64)


def test_new_mesh_invalidates_previous_results(controller):
    controller.request_bake()
    run_to_completion(controller)
    assert controller.curvature_map is not None

    other, _ = load_mesh(ASSETS / "beveled_cube.obj")
    controller.set_mesh(other)

    assert controller.curvature_map is None
    assert controller.position_map is None
    assert controller.unwrap_result is None
    assert controller.pending_stages() == ALL_STAGES


def test_cancel_stops_the_bake(controller):
    """A cancelled stage must not be recorded as done."""
    controller.bake_params.resolution = 512
    assert controller.request_bake()

    controller.cancel()
    deadline = time.time() + 60.0
    while controller.running:
        controller.pump()
        assert time.time() < deadline, "cancel never took effect"

    assert controller.error == "Bake cancelled"
    assert controller.pending_stages(), "everything was marked complete anyway"

    # And re-baking must actually finish the job.
    controller.bake_params.resolution = 128
    controller.request_bake()
    run_to_completion(controller)
    assert controller.pending_stages() == []


def test_gl_stage_runs_on_the_calling_thread(controller):
    """The rasteriser touches GL, so pump() must run it inline, never threaded."""
    import threading

    controller.request_bake()
    main_thread = threading.get_ident()
    observed: list[int] = []

    original = controller._run_curvature

    def spy():
        observed.append(threading.get_ident())
        original()

    controller._run_curvature = spy
    controller._plan = controller._build_plan()
    run_to_completion(controller)

    assert observed, "the curvature stage never ran"
    assert all(ident == main_thread for ident in observed)


def test_mesh_scale_is_reported(controller):
    assert controller.mesh_scale > 0.0
    assert controller.mesh_scale == pytest.approx(
        float(np.linalg.norm(controller.mesh.extents))
    )
