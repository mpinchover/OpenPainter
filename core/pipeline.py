"""Orchestration for the expensive half of the pipeline.

Three stages: unwrap, the UV-space curvature bake, and seam padding. Each is
keyed on its own parameters *plus* the key of the stage before it, so nudging
the curvature strength re-rasterises but leaves the unwrap alone, and nudging
seam padding re-runs only the padding.

Stages are tagged CPU or GL. GL stages must run on the thread that owns the
context, so :meth:`BakeController.pump` executes those inline during the render
loop and hands CPU stages to a worker thread so the window keeps drawing.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Optional

import moderngl
import numpy as np
import trimesh

from .baking import GBuffer, UVSpaceBaker, dilate
from .bevel import bevel
from .edge_wear import normalize_position
from .params import BakeParams, BevelParams, UnwrapParams
from .uv_unwrap import UnwrapResult, source_uv_layout, unwrap

STAGES = ("bevel", "unwrap", "curvature", "post")


@dataclass
class _Step:
    stage: str
    label: str
    needs_gl: bool
    run: Callable[[], None]
    key: tuple


class BakeController:
    """Runs the bake chain incrementally, one step per frame."""

    def __init__(self, ctx: moderngl.Context):
        self.baker = UVSpaceBaker(ctx)

        self.mesh: Optional[trimesh.Trimesh] = None
        self.mesh_token: int = 0
        self.bevel_params = BevelParams()
        self.unwrap_params = UnwrapParams()
        self.bake_params = BakeParams()

        # Stage outputs.
        self.beveled_mesh: Optional[trimesh.Trimesh] = None
        self.unwrap_result: Optional[UnwrapResult] = None
        self.gbuffer: Optional[GBuffer] = None
        self.curvature_map: Optional[np.ndarray] = None
        self.position_map: Optional[np.ndarray] = None

        # Bumped whenever the maps change, so the viewport knows to re-upload
        # without polling the arrays themselves.
        self.maps_version: int = 0

        self.running: bool = False
        self.stage: str = ""
        self.progress: float = 0.0
        self.error: Optional[str] = None
        self.timings: dict[str, float] = {}

        self._completed: dict[str, tuple] = {}
        self._plan: list[_Step] = []
        self._index: int = 0
        self._thread: Optional[threading.Thread] = None
        self._thread_error: Optional[str] = None
        self._step_progress: float = 0.0
        self._step_started: float = 0.0
        self.cancel_event = threading.Event()

    # -- inputs -----------------------------------------------------------

    def set_mesh(self, mesh: trimesh.Trimesh) -> None:
        """Swap in new geometry and invalidate every stage."""
        self.mesh = mesh
        self.mesh_token += 1
        self._completed.clear()
        self.beveled_mesh = None
        self.unwrap_result = None
        self.gbuffer = None
        self.curvature_map = None
        self.position_map = None
        self.error = None

    @property
    def mesh_scale(self) -> float:
        if self.mesh is None:
            return 1.0
        return max(float(np.linalg.norm(self.mesh.extents)), 1e-9)

    # -- planning ---------------------------------------------------------

    def _keys(self) -> dict[str, tuple]:
        bevel_key = (self.mesh_token, self.bevel_params.key())
        unwrap_key = bevel_key + (
            self.unwrap_params.key(),
            self.bake_params.resolution,
        )
        curvature_key = unwrap_key + self.bake_params.curvature_key()
        post_key = curvature_key + (self.bake_params.dilation,)
        return {
            "bevel": bevel_key,
            "unwrap": unwrap_key,
            "curvature": curvature_key,
            "post": post_key,
        }

    def pending_stages(self) -> list[str]:
        """Which stages a bake would actually re-run right now."""
        if self.mesh is None:
            return []
        keys = self._keys()
        return [stage for stage in STAGES if self._completed.get(stage) != keys[stage]]

    def is_dirty(self) -> bool:
        return bool(self.pending_stages())

    def _build_plan(self) -> list[_Step]:
        keys = self._keys()
        stale = set(self.pending_stages())

        unwrap_label = (
            "Reading source UVs" if self.unwrap_params.use_source_uvs else "Unwrapping UVs"
        )
        definitions = {
            "bevel": ("Bevelling edges", False, self._run_bevel),
            "unwrap": (unwrap_label, False, self._run_unwrap),
            "curvature": ("Baking curvature", True, self._run_curvature),
            "post": ("Padding seams", False, self._run_post),
        }
        return [
            _Step(stage=stage, label=definitions[stage][0], needs_gl=definitions[stage][1],
                  run=definitions[stage][2], key=keys[stage])
            for stage in STAGES
            if stage in stale
        ]

    # -- stage bodies -----------------------------------------------------

    def _run_bevel(self) -> None:
        """Give the bake geometry to see at every sharp edge.

        The bevel exists only for the bake. It inherits the source UV layout,
        so the texture still addresses the *original* unbeveled mesh -- the wear
        lands in the outer rim of each UV island, which on that mesh is the
        texture right against the edge.
        """
        assert self.mesh is not None
        self.beveled_mesh = bevel(self.mesh, self.bevel_params)

    def _run_unwrap(self) -> None:
        assert self.beveled_mesh is not None
        resolution = self.bake_params.resolution
        if self.unwrap_params.use_source_uvs:
            # No charting and no reindexing -- bake into the layout the artist
            # authored, so the PNG applies to their mesh and not to ours.
            self.unwrap_result = source_uv_layout(self.beveled_mesh, resolution)
        else:
            self.unwrap_result = unwrap(
                self.beveled_mesh, self.unwrap_params, resolution
            )

    def _run_curvature(self) -> None:
        assert self.unwrap_result is not None
        result = self.unwrap_result
        self.baker.upload(result.vertices, result.normals, result.uvs, result.faces)
        self.gbuffer = self.baker.rasterize(self.bake_params.resolution, self.bake_params)

    def _run_post(self) -> None:
        assert self.gbuffer is not None and self.beveled_mesh is not None
        mask = self.gbuffer.mask
        padding = self.bake_params.dilation

        # Always derive from the G-buffer, never from a previous padded result,
        # so re-running with a different width cannot pad an already-padded map.
        self.curvature_map = dilate(self.gbuffer.curvature, mask, padding)

        # Normalise against the geometry that was actually rasterised, so the
        # 0..1 range still spans the bounding box the G-buffer positions live in.
        lower = np.asarray(self.beveled_mesh.bounds[0], dtype=np.float32)
        extents = np.asarray(self.beveled_mesh.extents, dtype=np.float32)
        bposition = normalize_position(self.gbuffer.position, lower, extents)
        self.position_map = dilate(bposition, mask, padding)

        self.maps_version += 1

    # -- execution --------------------------------------------------------

    def _report(self, fraction: float) -> None:
        self._step_progress = float(fraction)

    def request_bake(self) -> bool:
        """Queue the stale stages. Returns False if there was nothing to do."""
        if self.mesh is None or self.running:
            return False
        self._plan = self._build_plan()
        if not self._plan:
            return False

        self._index = 0
        self._thread = None
        self._thread_error = None
        self._step_progress = 0.0
        self.cancel_event.clear()
        self.error = None
        self.running = True
        self.stage = self._plan[0].label
        self.progress = 0.0
        return True

    def cancel(self) -> None:
        """Ask the running bake to stop at the next stage boundary."""
        if self.running:
            self.cancel_event.set()

    def pump(self) -> None:
        """Advance the bake. Call once per frame from the GL thread."""
        if not self.running:
            return

        if self._index >= len(self._plan):
            self._finish()
            return

        step = self._plan[self._index]
        self.stage = step.label

        if self._thread is not None:
            if self._thread.is_alive():
                self._update_progress()
                return
            self._thread = None
            if self._thread_error is not None:
                self.error = self._thread_error
                self._abort()
                return
            if self.cancel_event.is_set():
                # A cancelled stage returns whatever it had finished, which is
                # not a valid result -- never record its key, or the next bake
                # would skip it as already done.
                self.error = "Bake cancelled"
                self._abort()
                return
            self._complete_step(step)
            return

        if self.cancel_event.is_set():
            self.error = "Bake cancelled"
            self._abort()
            return

        self._step_started = time.perf_counter()
        self._step_progress = 0.0

        if step.needs_gl:
            try:
                step.run()
            except Exception:
                self.error = traceback.format_exc(limit=6)
                self._abort()
                return
            self._complete_step(step)
        else:
            self._thread = threading.Thread(
                target=self._thread_body, args=(step,), daemon=True
            )
            self._thread.start()

        self._update_progress()

    def _thread_body(self, step: _Step) -> None:
        try:
            step.run()
        except Exception:
            self._thread_error = traceback.format_exc(limit=6)

    def _complete_step(self, step: _Step) -> None:
        self.timings[step.stage] = time.perf_counter() - self._step_started
        self._completed[step.stage] = step.key
        self._index += 1
        self._step_progress = 0.0
        if self._index >= len(self._plan):
            self._finish()
        else:
            self.stage = self._plan[self._index].label
        self._update_progress()

    def _update_progress(self) -> None:
        if not self._plan:
            self.progress = 1.0
            return
        self.progress = (self._index + self._step_progress) / len(self._plan)

    def _finish(self) -> None:
        self.running = False
        self.stage = ""
        self.progress = 1.0
        self._plan = []
        self._index = 0

    def _abort(self) -> None:
        self.running = False
        self.stage = ""
        self.progress = 0.0
        self._plan = []
        self._index = 0
        self._thread = None

    def release(self) -> None:
        self.cancel()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self.baker.release()
