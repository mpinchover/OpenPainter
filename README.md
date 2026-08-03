# MeshMap

A standalone desktop tool for **baking maps into a mesh's own UV layout and
authoring a material on top of them**. Import a UV-mapped FBX (or OBJ, glTF, …),
bake curvature and ambient occlusion into its atlas, build a material out of
masks and procedural generators, stamp normal-map decals onto the surface by
pointing at it, and export a standard PBR set of PNGs that drop straight back
onto the original mesh in Blender.

Everything is tuned live on the model. The expensive half of the work — anything
that has to rasterise or ray-cast the mesh — is cached and re-run only when its
own inputs change; the cheap half — anything that is arithmetic on already-baked
textures — is a full-screen GPU pass that re-runs the frame you move a slider.

```bash
cd meshmap && source myenv/bin/activate
python main.py                          # opens on a chamfered cube
python main.py chair.fbx                # load on startup
python main.py chair.fbx --resolution 2048 --z-up
python main.py chair.fbx --bevel 0.01,3,30
python main.py chair.fbx --auto-unwrap  # only for a mesh with no UVs at all
python main.py --ui-scale 1.8
python main.py chair.fbx --selftest out/   # headless bake + export + renders
```

Press **B** to bake, build a material in the sidebar, press **E** to export. A
mesh dropped on the window replaces whatever is loaded.

---

## Contents

- [Install](#install)
- [What it produces](#what-it-produces)
- [Conventions the code relies on](#conventions-the-code-relies-on)
- [The window](#the-window)
- [The bake pipeline](#the-bake-pipeline)
- [Materials](#materials)
- [Viewport shading](#viewport-shading)
- [Decals](#decals)
- [Input, navigation and state](#input-navigation-and-state)
- [Algorithms at a glance](#algorithms-at-a-glance)
- [Code layout](#code-layout)
- [Tests](#tests)
- [Known limitations](#known-limitations)

---

## Install

```bash
python -m venv myenv && source myenv/bin/activate
pip install -r requirements.txt
brew install assimp          # macOS; FBX parsing goes through it
```

The stack is numpy/scipy/Pillow for the CPU work, trimesh for mesh handling,
xatlas for the fallback unwrap, moderngl + moderngl-window (pyglet backend) for
the GL context and window, imgui-bundle for the interface, and embreex for
accelerated ray casting.

FBX is parsed by Assimp. `core/mesh_io.py` tries the `pyassimp` ctypes wrapper
first and falls back to converting through the `assimp` command-line tool —
pyassimp's last release imports `distutils`, removed in Python 3.12, so on any
current Python the CLI path is what actually runs. Formats trimesh reads
natively (`.obj`, `.glb`, `.gltf`, `.stl`, `.ply`, `.dae`, `.off`, `.3mf`, …)
skip Assimp entirely.

`embreex` is optional but worth having: it is what makes the occlusion bake a
few seconds rather than a coffee break. trimesh picks it up automatically.

---

## What it produces

`File > Export textures` (or **E**) writes a standard PBR set into
`<export folder>/<mesh name>/`:

| File | Source | Notes |
| --- | --- | --- |
| `color.png` | the material tree, resolved to colour | Base Color |
| `normal.png` | the decals, composited | tangent space, OpenGL (+Y) |
| `metallic.png` | the material tree | grayscale |
| `roughness.png` | the material tree | grayscale |
| `ao.png` | the ray-cast bake × the material's own AO channel | grayscale |

Each map has a switch under **Settings → Export**, and the depth is 8- or 16-bit
PNG. The five come from independent halves of the app, so any subset can be
written: a decal needs no bake behind it, and a material needs no decals.

Opacity and emission are authored per material and shown in the viewport, but
they are not exported — the set stops at what a standard shader graph wants, one
file per socket rather than a packed ORM, because packing is a step to undo at
the other end.

**Preview and export are the same pixels.** The exporter reads the compositor's
own framebuffers back off the GPU rather than recomputing anything on the CPU,
so the PNG is by construction what the viewport was showing.

In Blender, plug `color.png` into Base Color and the rest into the sockets they
are named after, and set everything except `color.png` to **Color Space:
Non-Color** — they are data, not colour.

### 16-bit RGB PNGs

Pillow's 16-bit support is the single-channel `I;16` mode; handing it a
three-channel uint16 array raises. `core/export.py` therefore emits 16-bit RGB
files directly — signature, `IHDR`, one zlib-compressed `IDAT` of
filter-byte-prefixed big-endian rows, `IEND`, each chunk with its CRC. It is the
depth a normal map wants: 8 bits puts visible steps in a shallow bevel or a
gentle vent lip.

---

## Conventions the code relies on

**UV your mesh first.** MeshMap bakes into *the UV map already on the mesh*.
Unwrap in Blender before exporting (UV Editing workspace, or `U` → Smart UV
Project); the FBX exporter always includes UV layers. That is what lets the
exported maps apply to your original mesh. `--auto-unwrap` lets xatlas invent an
atlas instead, but that atlas fits only MeshMap's internal triangulated copy, so
reach for it only when the mesh has no UVs at all.

**Z-up, like Blender.** A Y-up source — what Blender's FBX and glTF exporters
write — is rotated in on import with the same +90° about X that Blender's
importers apply. `--z-up` (or *Source is Z-up*) skips that. `MeshInfo.to_world`
records the transform, so anything needing the source file's own coordinates can
invert it. The conversion affects presentation and the bake's Axis option; the
geometry is otherwise untouched.

**FBX units.** FBX records `UnitScaleFactor` as centimetres per coordinate unit.
`core/mesh_io.py` reads it out of both the ASCII and the binary container and
bakes `UnitScaleFactor / 100` into the vertices, so the model arrives in metres
regardless of what it was authored in.

**Row 0 is v = 0.** Every array in the app stores the bottom row first. PNG
stores the top row first, so `core/decal.py` flips on read and `core/export.py`
flips on write — the two cancel, and a decal comes out of the exporter the way
round it went in.

**Normals are OpenGL convention** (+Y up), which is what Blender expects. A map
baked DirectX-style has a per-decal *Flip green* switch.

---

## The window

Three pieces of permanent chrome, and a 3D view laid out in what is left:

- a **navigation bar** across the top — the preview mode, view toggles, the File
  menu;
- a **sidebar** down the left, an icon rail selecting one of seven views:
  **Bake**, **Material**, **Decal**, **Mesh**, **Library**, **Console**,
  **Settings**. Drag its right edge to resize, up to half the window;
- a **status bar** along the bottom, carrying bake progress and the last message.

Nothing floats. The 3D view gets `MeshMapApp.viewport_rect`, and the projection
matrix and the cursor rays are both built from that rect rather than from the
whole window — so nothing is ever hidden behind a panel, and pointing at the
model lands where you point.

Two preview modes, because there are two products: **Shaded** (`1`) is the
material tree, lit; **Normals** (`2`) is the decal normal map. Where a map does
not exist yet the mesh is drawn flat grey rather than black.

Maps are shown where they are made rather than in a window of their own: the
Material tab's tree carries a swatch or a live thumbnail per row, and the
library shows a cached preview of every image on the decal shelf.

### Keyboard

| Key | Action |
| --- | --- |
| `1` / `2` | Preview: shaded / normals |
| `B` / `E` | Bake / export maps |
| `F` | Frame the mesh (also resets the view) |
| `W` / `L` | Wireframe / lighting |
| `G` / `S` / `R` | Move / scale / rotate the selected decal with the pointer |
| `X` / `Y` during `G`/`S`/`R` | Constrain to one decal axis |
| Click or `Enter` during a transform | Confirm |
| `Shift+D` | Duplicate the selected decal |
| `X` | Delete the selection (asks first) |
| `Cmd/Ctrl+Z`, `+Shift` | Undo / redo |
| `+` / `-` | Bigger / smaller interface |
| `Esc` | Cancel a placement or a transform |

---

## The bake pipeline

`core/pipeline.py` runs five stages. Each is keyed on its own parameters **plus
the key of the stage before it**, so invalidation cascades forward and never
backward:

```
mesh ──▶ bevel ──▶ unwrap ──▶ curvature ──▶ occlusion ──▶ post
                                  │             │           │
                             (GL thread)    (worker)    (worker)
```

```python
bevel_key     = (mesh_token, bevel_params.key())
unwrap_key    = bevel_key + (unwrap_params.key(), resolution)
curvature_key = unwrap_key + bake_params.curvature_key()
occlusion_key = unwrap_key + (bake_occlusion, samples, distance)
post_key      = curvature_key + occlusion_key + (dilation,)
```

Occlusion hangs off the *unwrap*, not the curvature: it is a ray cast against
geometry, and no curvature slider can change what a ray hits. So nudging
Strength re-rasterises and re-pads without paying for a re-cast.

| You change | Re-runs |
| --- | --- |
| Seam padding | `post` |
| Strength / Radius / Offset / Smooth / Axis | `curvature`, `post` |
| Resolution, atlas settings, bevel, or the mesh | everything |
| Anything on a material layer | nothing — it is not a bake stage |

The Bake button reports how many stages are stale before you commit.

**Threading.** Stages are tagged CPU or GL. GL work must happen on the thread
that owns the context, so `BakeController.pump()` — called once per frame from
the render loop — runs GL stages inline and hands CPU stages to a worker thread.
The window keeps drawing, the progress bar keeps moving, and Cancel works. A
cancelled stage returns whatever it had finished, which is not a valid result,
so its key is deliberately never recorded; otherwise the next bake would skip it
and keep the garbage.

Occlusion can be switched off entirely. The stage still *runs* — it clears its
map and records its key — because a stage that never records a key is a stage
that is permanently pending, and the bake would always look stale.

### 1. Import — `core/mesh_io.py`

Load, triangulate, weld (never across a UV seam — trimesh's `merge_vertices`
defaults to `merge_tex=False`), repair winding, apply the unit scale and the
axis fix. The importer collects human-readable notes (missing UVs, UVs outside
0..1, unit conversions) that the Mesh tab surfaces, and measures **UV density** —
UV units per metre of surface — which times the resolution gives texels per
metre, the number that decides whether a feature is wide enough to bake at all.

### 2. Bevel — `core/bevel.py`

The curvature bake differentiates the *interpolated* vertex normal per texel, so
a perfectly sharp edge — zero width, no geometry — has nowhere to put a
gradient. Welding averages the corner normals of the faces meeting there
instead, and that swing smears across the whole face: entire faces bake grey and
the edge does not stand out at all.

The Bevel panel replaces sharp edges with a narrow strip of geometry before the
bake, which confines the gradient to the strip. It approximates Blender's Bevel
modifier under Affect = Edges, Width Type = Offset, Limit Method = Angle, Miter
Sharp, Clamp Overlap off, Loop Slide on, and exposes its three real knobs:
amount (metres), segments, and the dihedral angle threshold.

How it works:

1. **Edge selection.** Weld by position, build an edge → (face, corner) table,
   and tag every *manifold* edge whose two face normals diverge by more than the
   threshold (`dot(n₁, n₂) < cos θ`).
2. **Corner solve, once per sector.** Around a beveled vertex, the incident
   faces are ordered into a ring and split into *sectors* — runs of faces between
   two beveled edges. Each sector solves for the single point that sits `amount`
   inside *both* of its bounding edges, in the sector's own area-weighted plane;
   the faces inside inherit it. Solving per sector rather than per face is what
   keeps the result watertight, and it is where Loop Slide falls out: a corner
   bounded by one beveled and one unbeveled edge slides along the unbeveled one
   instead of lifting away from it.
3. **The strip.** Along each beveled edge, a circular arc is built between the
   two inset corners, centred at the point equidistant along both face normals so
   the strip meets both faces tangentially. The arc is slerped and its radius
   lerped, so an asymmetric pair still closes. `segments = 1` degenerates to the
   straight chord — Blender's flat chamfer.
4. **Corners.** The hole each beveled vertex leaves is bounded by the sector
   points and the arcs between them. A three-sided hole is emitted as one planar
   triangle; anything larger is fanned from an apex lifted onto the corner
   sphere, so the patch stays round instead of collapsing flat.
5. **UVs.** Every new vertex lies within the band between a face's original and
   inset outlines, so its UV is interpolated barycentrically from that face's
   corner UVs, and each new triangle is assigned to a single owning face. The
   strip therefore lands in the **outer rim of the UV island** — which, on your
   original unbeveled mesh, is the texture right up against the edge. That is why
   the bevel is never exported and never needs to be: it is geometry for the bake
   only, and the texture still addresses the original mesh through its own UVs.

Divergence from Blender: corners where three or more beveled edges meet are
fanned rather than Grid Filled, and miters are whatever the corner solve
produces. Since nothing here reaches the export, the cost is a little accuracy
in the wear falloff at corners, not correctness anywhere downstream.

**A bevel has to be at least a texel or two wide to bake anything** — the Bake
tab reports the texels per metre implied by the UV density and the resolution.

### 3. Unwrap — `core/uv_unwrap.py`

Two sources, both producing an `UnwrapResult` (vertices, normals, UVs, faces,
and a `vmapping` back to the original vertex indices):

- **The mesh's own UV map** (default). No charting, no packing, no reindexing —
  `vmapping` is the identity and the baked texture lines up with the source mesh
  exactly as authored.
- **xatlas** (`--auto-unwrap`), which charts, packs and splits vertices along the
  seams it cuts, handing back a real `vmapping`.

One subtlety matters more than it looks. The normals the bake rasterises are
**welded, area-and-angle-weighted vertex normals**, not trimesh's:

- *Welded*, because vertices split for UV reasons still sit at one position. If
  the normals were split too, every seam would read as a crease and bake a wear
  line down the middle of a flat face.
- *Area-weighted as well as angle-weighted*, because a vertex where a large flat
  face meets a 1 mm bevel quad has a right angle in both. Angle weighting alone
  lets the sliver drag the big face's normal ~20° off flat, and since the bake
  differentiates that normal, the whole face lights up. Folding in area lets the
  large face dominate by its area ratio, confining the gradient to the bevel.
  This is Blender's Weighted Normal modifier in "Face Area & Angle" mode.

### 4. The curvature bake — `core/baking.py`, `render/shaders/gbuffer.*`

An edge is *where the surface normal changes fastest*. Not where two faces meet
at an angle, not how a vertex compares to its neighbours — the rate of change of
the normal across the surface.

What makes that measurable is rasterising the mesh **into UV space**. The vertex
shader feeds the UV in as clip-space position instead of projecting the model:

```glsl
gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
```

Every fragment the rasteriser produces is therefore a *texel of the atlas*, and
`dFdx`/`dFdy` in the fragment shader are the change in the interpolated normal
**between neighbouring texels of the atlas** rather than between neighbouring
pixels of the screen. Where the normal swings fast (a bevel) the derivative is
large; across a flat face it is zero.

```glsl
vec3 dx = dFdx(n), dy = dFdy(n);
float curvature = max(dot(dx, dx), dot(dy, dy));
float exponent  = (1.0 / max(u_radius, 1e-4)) * 0.25;
curvature = clamp(pow(curvature, exponent) * u_strength * 2.0 + u_offset / 10.0, 0.0, 1.0);
if (dot(u_axis, u_axis) > 0.0) curvature *= dot(n, u_axis);
```

**The `pow` is the whole trick.** A squared derivative across a 1 mm bevel is a
minuscule number; raising it to a small fractional exponent is what lifts it into
a visible band. `Radius` *is* that exponent, not a distance — larger Radius means
a smaller exponent, which lifts harder and spreads the band wider. `Offset` is
added afterwards as `offset / 10`, so the default −2.0 crushes near-flat areas to
black. `Axis` masks the result by facing direction, which is how gravity-driven
scuffing (`-Z`) is expressed.

The pass writes three attachments at once — world position (its alpha doubling as
the coverage mask), world normal, and curvature — with depth test, culling and
blending all off, because a valid atlas has no overlaps and each texel keeps
exactly the surface point it covers.

**Smoothing is a resolution round-trip, not a kernel.** `Smooth` copies the
curvature into a target at 95% size and straight back, N times, letting the
hardware's bilinear filter do the work. Each round-trip is a gentle, slightly
anisotropic smear that redistributes rather than brightens.

Defaults: 1024², Strength 0.5, Radius 2.0, Offset −2.0, Smooth 1, Axis XYZ, seam
padding 4. Resolutions run 512 → 8192.

### 5. Ambient occlusion — `core/occlusion.py`

The one map here that is not arithmetic on the G-buffer. Curvature is a
derivative of the normals at a texel — it says what the surface does *there*.
Occlusion depends on geometry that may be nowhere near it, on the surface or in
the atlas, so it is a real ray cast: **32 rays per covered texel**, into the
hemisphere it faces, against the same beveled and unwrapped geometry the bake
rasterised, through trimesh's Embree backend where `embreex` is installed.

Rays go out in batches of 200 000 across a thread pool (capped at 8 workers,
since each batch carries large origin and direction arrays), with completed
chunks accumulated in batch order so the map is bit-for-bit deterministic.
Progress is reported per sample round and Cancel is honoured between them.

Three things make a modest ray count usable:

- **Cosine weighting**, so an unoccluded surface comes back at exactly 1 without
  a per-sample `dot(n, l)` term — the sampling density carries it.
- **Stratification in both directions.** Elevation walks outwards one ring per
  sample (`r = √((i + jitter)/N)`), azimuth turns by the **golden angle** each
  step — the least-clumping rotation there is. Each texel's spiral starts at its
  own random phase, so what noise is left looks like grain rather than a pattern.
  That is most of the difference between 32 rays that look smooth and 32 that
  speckle.
- **A 3×3 average over covered texels only**, far cheaper than the extra rays
  that would smooth the residue. Uncovered texels are excluded from both the sum
  and the count, so a texel at a chart border is not dragged towards the empty
  space beside it, and the map does not wrap at its edges — an atlas is not a
  tiling texture, and a chart against the left border has nothing to do with one
  against the right.

Rays stop at **20% of the model's bounding-box diagonal**, and a hit's
contribution falls off linearly with distance. Without a limit, occlusion means
"is there anything at all in this direction", which on a closed model darkens
every concave surface no matter how far away the far side is; with one, it means
"is there anything *nearby*", which is what reads as contact. A fraction rather
than a distance, so the same setting suits a bolt and a bridge.

Uncovered texels come back as 1.0 — no surface, nothing to shade, and white is
what the seam padding should spread outwards from.

### 6. Seam padding — `dilate()` in `core/baking.py`

Single-sample rasterisation leaves a texel-wide hole around every chart border.
Left alone, bilinear filtering in a renderer pulls that empty gutter in and draws
a dark fringe along every seam. `dilate()` runs N flood iterations, each
averaging the eight neighbours of every uncovered texel that has at least one
covered one, and growing the coverage mask as it goes.

It always re-derives from the G-buffer rather than from a previous padded result
— otherwise re-running with a different width would pad an already-padded map,
compounding the dilation.

The same stage normalises the position attachment into **`bposition`** —
object-space position over the bounding box, in 0..1 — which is the domain every
procedural generator is evaluated in.

---

## Materials

A material is a **binary tree of masks** (`core/layers.py`). A mask produces
black and white across the surface; what each of those *means* is the tree. White
can be a colour, or another mask that in turn decides between two more things,
and so can black. Edge wear picking between rust and bare metal is one node; edge
wear picking between rust and a noise that itself picks between two greys is
three.

A new material starts as **one flat colour**. Changing its type to a mask grows a
white and a black side underneath it, so every branch that exists is one that was
asked for, and a fresh mask reads as plain black and white on the model — the
mask itself, before any colour is put under it. Nesting stops at 8 levels.

Every material you make stays around: the dropdown at the top of the tab lists
them and switches between them, and past five entries it grows a case-insensitive
search box. A material can be assigned to the mesh in the Mesh tab, or used to
tint a decal.

### A leaf is a material, not just a colour

`ColorSlot` carries the full set of surface channels, and the mask blends **all
of them** the same way it blends colour — one pass, three attachments:

| Channel | Meaning |
| --- | --- |
| Colour | base colour |
| Metallic | 0 dielectric, 1 bare metal |
| Roughness | 0 mirror, 1 chalk |
| Opacity | below 1 the surface lets what is behind it through (viewport only) |
| Emission | multiples of the surface's own colour (viewport only) |
| Ambient occlusion | material-authored occlusion, multiplied into the baked AO on export |

Emission is stored in the material texture as a fraction of `MAX_EMISSION = 16`,
because binding a float framebuffer turns fragment-colour clamping on and
moderngl exposes no way to turn it back off. Nothing outside the compositor sees
the packing — `read_material()` hands back the real values.

### Where the boundary falls

A mask is a **continuous field, not a stencil**: edge wear ramps up as the
surface curves, noise wanders through every value between. Mixing the two sides
by that directly gives a blend everywhere the field sits mid-way, which reads as
the colours bleeding into each other rather than as a boundary. So each mask
carries:

- **Threshold** — the level that divides the two sides;
- **Softness** — zero by default, so every texel belongs to one side or the
  other, and the boundary is a boundary. Raise it to feather;
- **Range low / high** — the slice of the field stretched across the whole 0..1
  *before* the split. Few masks use all of their range (noise clusters around its
  middle), which leaves the threshold living in a few pixels of slider travel.
  Stretching the part in use gives the slider back its resolution. The remap is
  monotonic, so with a hard threshold it moves the boundary rather than reshaping
  it.

```glsl
mask = clamp((mask - u_rangeLow) / max(u_rangeHigh - u_rangeLow, 1e-3), 0.0, 1.0);
mask = u_softness <= 0.0 ? step(u_threshold, mask)
                         : smoothstep(u_threshold - u_softness, u_threshold + u_softness, mask);
```

### The mask kinds

Twelve, all evaluated in `render/shaders/layer.frag` at the texel's
**object-space `bposition`** rather than at its UV — which is why every pattern
runs continuously across atlas seams instead of breaking at every chart border.
Each has a rotation and a seed, and each ends with the same bias/contrast remap:
`clamp((value − bias) · contrast + 0.5, 0, 1)`.

| Kind | How it is computed |
| --- | --- |
| **Edge wear** | `clamp((curvature − noise · wearAmount) · contrast, 0, 1)` — see below |
| **Noise** | value-noise fBm directly |
| **Grunge** | a broad noise minus a finer, harsher one, so patches get eaten into by detail |
| **Scratches** | lanes along one axis, each hashed to an offset; a thin line mask across the lane, gated by a broken-up length mask |
| **Brushed metal** | high-frequency sine bands along one axis, warped by a low-frequency noise and mixed towards a second noise for variation |
| **Cells** | Worley/cellular: distance to the nearest jittered point per lattice cell, then a smoothstep |
| **Clouds** | a low-frequency, high-detail fBm pushed through a smoothstep |
| **Directional streaks** | fBm sampled with one axis stretched by `streakLength` and the others squeezed by `streakWidth` |
| **Gradient** | a linear ramp along one axis |
| **Brick / tile** | a scaled 2D lattice with alternate rows offset half a cell; mortar from the distance to the nearest cell edge |
| **Wood grain** | concentric rings — `abs(sin((radius + wobble) · 10))`, the radius perturbed by a noise |
| **Marble** | 3D veins: `abs(sin(directional ramp + turbulence))`, continuous through the volume |

Four of them (scratches, brushed metal, streaks, brick) are inherently
directional, and a 2D pattern evaluated in a 3D volume collapses on faces
perpendicular to its plane. Those sample a **face-projected position**: the
UV-space derivatives of `bposition` recover the face's normal, the dominant axis
picks which coordinate plane to project into, and the pattern lies flat on every
face. A dominant-axis choice rather than a triplanar blend keeps hard-surface
corners crisp.

**Edge wear** is the one kind that reads the curvature bake. The noise is
*subtracted* from the curvature rather than multiplied into it, and that is the
whole reason it looks weathered: a multiply would dim the wear evenly, while a
subtract erodes it — wherever the noise runs high the edge drops out completely,
so the wear is patchy and interrupted instead of a uniform stripe tracing every
bevel. Its `Value` is not the noise scale directly; the scale is `value × 10`.
Defaults: Value 1.0, Wear amount 0.6, Contrast 3.0, and noise Detail 5.0 /
Roughness 0.7 / Lacunarity 5.0 / Distortion 0.15.

The noise itself (`render/shaders/noise.glsl`, mirrored in `core/edge_wear.py`)
is a sine-hashed 3D value noise on a lattice, fBm'd over `detail` octaves with
`roughness` amplitude falloff and `lacunarity` frequency step, including a
fractional-octave blend for non-integer detail, with the sample point optionally
pre-warped by the noise itself (`distortion`).

### Evaluating the tree — `render/composite.py`

**One full-screen pass per node, bottom up.** Everything here is already in UV
space, so by the time a node runs its children are just textures — no recursion
in GLSL, no shader generated per tree, no depth limit in the pass itself. Depth
costs *passes*, not shader complexity.

Each pass writes three attachments: colour, the packed surface channels
(metallic, roughness, opacity, emission), and material AO. Targets are half
float — 8 bits per channel would band across a deep tree, and full floats cost
four times the memory for detail no monitor resolves — and are recycled through a
pool. A post-order walk holds at most one result per level in flight, so a tree
of depth *d* needs *d + 1* targets however wide it is.

Every node therefore ends up holding a real texture, which is exactly what the
panel wants for its **thumbnails**: each is a 96 px blit of the node's own
output, so drilling into a subtree is a navigation choice rather than the only
way to find out what is in it.

The compositor re-runs when `texture_key()` — a structural hash of everything
that changes what the tree renders — differs from what it last drew, rather than
when something remembered to set a flag. A missed flag is an edit that silently
does nothing; comparing the data itself cannot drift. Names are excluded on
purpose: they are for reading, not rendering.

### The sidebar's two halves

The tree is binary and its depth is unbounded, so indenting the whole thing runs
out of panel before it runs out of tree. **Inspector** on top edits whatever is
selected, at a fixed size no matter how deep it sits; **Material tree** underneath
is one line per slot, with a swatch or a thumbnail, for selecting with. Depth
costs a line rather than a column. The divider between them drags, and the split
is remembered between runs.

Any row can be renamed by double-clicking it, the material itself included. A
name describes the part of the material rather than the kind of thing it
currently is, so it survives converting a colour into a mask and back; clear it
and the automatic label (the mask's kind, or the colour's hex code) takes over
again.

---

## Viewport shading

`render/shaders/preview.frag` shades the model the way Blender's Principled BSDF
does: **Cook-Torrance** specular with a GGX distribution, Smith geometry and
Schlick Fresnel, roughness squared into the GGX alpha so the middle of the slider
looks like the middle of Blender's. A metal has no diffuse of its own — its
colour is what it reflects.

The lamp is given a **size floor** (`roughness ≥ 0.07` for the direct term). A
light of no size puts a mirror's highlight inside a single pixel, so polishing a
surface would read as *removing* the highlight rather than sharpening it.

**World lighting** (Settings) is both lights. *Rotation* walks a key light around
the model — anchored to the model, not the camera, at a fixed 38° elevation, so
orbiting shows you a differently lit side rather than the same one from further
round. A line of text names where it stands and which way it therefore shines,
and an arrow gizmo appears in the viewport while you turn it, because *which side
of what I am looking at is this* is answered fastest by pointing at it.
*Strength* and *Colour* are the world it stands in — a two-tone gradient,
brighter above than below. Without it a metal has nothing to reflect and a rough
surface has nothing to scatter, and both render black however their own sliders
are set. Roughness **blurs the direction the reflection gathers from** rather
than dimming what it gathers, so a rough metal is still a metal; a
roughness-aware Schlick (Karis' IBL approximation) keeps a rough surface from
going to full reflection at grazing angles.

**Emission is a light, not a bright colour.** Past white a surface cannot get any
brighter on screen, so the scene is drawn into a target with 32× headroom, a
bright pass keeps only what is over white (subtracting the threshold, so a
surface creeping past 1 starts to glow rather than switching on), a separable
5-tap Gaussian blurs it at quarter resolution, and a tonemap adds it back and
rolls the whole colour off together rather than clipping channel by channel.
Clipping per channel is what turns a bright red emission white; this keeps the
hue and lets the glow carry the brightness.

Opacity is drawn with ordinary alpha blending and depth writes still on — a
proper transparent pass would sort by depth, which is more machinery than a
preview of an alpha channel is worth.

The tangent frame for the normal map is derived **per pixel** from the
derivatives of position and UV (Mikkelsen's cotangent frame). The mesh carries no
tangent attribute, and deriving one in the shader means a decal lights correctly
on anything the app can load, however it was unwrapped.

---

## Decals

The Decal tab stamps imported normal maps into the mesh — a sci-fi vent, a hatch,
a panel seam — compositing them into an atlas-wide `normal.png`. Everywhere
outside a decal the output is flat `(0.5, 0.5, 1.0)`, so the map can be plugged
into a Normal Map node and change nothing except where the decals are.

**None of it involves the bake.** A decal is placed in UV space and needs no
geometry pass behind it: import, place, export, with no mesh baked at all if the
normal map is all you want.

### Loading an image — `core/decal.py`

An RGB image is taken as a tangent-space normal map. A **grayscale** one is taken
as a height map and converted, because that is the only reading of it that
produces a surface: central differences in texels give the slope, and
`(−du, −dv, 1)` normalised and encoded gives the normals. (The test for grayscale
is a channel spread under 2/255 — decoding a grey image as a normal map would
point every normal along the diagonal.)

Images are downsampled to a 2048 px maximum edge, flipped into the app's
row-0-at-v=0 convention on read, and **transparent texels are forced to a flat
normal** before upload. Nothing samples one directly — the alpha multiplies it
out — but every bilinear tap and every mipmap level around the edge of the
artwork averages it in, and left as the white the canvas happened to be, that
averaging invents a tilt that was never drawn.

### Decals accumulate as slopes, not normals

This is the central idea of the composite. Each decal pass writes the **surface
gradient** `xy/z` into a float target with additive blending, and one final pass
turns the accumulated total back into a normal map:

```glsl
// decal.frag, per decal
slope = normal.xy / max(normal.z, 1e-4) * (u_intensity * texel.a * fade);
// normal_encode.frag, once
out_color = vec4(normalize(vec3(slope, 1.0)) * 0.5 + 0.5, 1.0);
```

Slopes add the way the surfaces do: a vent stamped across a panel line should
read as both. Adding the encoded normals instead would average them towards flat,
and the second decal would rub out the first as much as it added itself. Outside
a decal's rectangle the contribution is zero, which adds nothing.

It is also why **Height intensity scales the slope rather than the stored
vector** — that is what makes 2.0 genuinely twice as steep, where scaling the
vector runs the normal flat against the surface and then past it, which reads as
the bump inverting rather than deepening.

### Three ways a decal can be projected

| Path | Shader | When |
| --- | --- | --- |
| UV rectangle | `decal.frag` | the placement stated in atlas coordinates |
| World projector | `decal_project.frag` | placed by pointing at the model; projects along a view-aligned frame and converts into tangent space per texel |
| Surface wrap | `decal_wrap.*` | faces whose authored UVs break away from the continuous surface chart |

Interactive placement uses the world projector — a centre, a right/up/forward
frame and a size, exactly like painting through a projector in a 3D viewport.
Committed placements are *also* stated in UV space so they can be composited into
the exported map. The projector rejects texels outside its rectangle, beyond its
depth range, or on surfaces nearly edge-on to it (`|dot(n, forward)| < 0.08`), so
it does not spray onto the far side of the model.

While you work, up to eight committed projectors are sampled **directly in the
mesh shader** from their source images rather than through the UV target, which
keeps a decal crisp at any on-screen size; the working UV normal map is rendered
at 2048 during interaction and re-rendered at full resolution for export.

### Unfolding the surface — `core/decal_wrap.py`

A decal that crosses a UV seam is two rectangles in the atlas and one shape on
the model, so it needs a coordinate system that is continuous across the seam.
`wrapped_vertices()` builds one:

1. Flatten the anchor triangle into 2D, and fit an **affine map** from that
   flattening to its authored UVs by least squares, so the placement stays exact
   on the anchor face.
2. **Breadth-first across shared edges**, welding vertices by position through a
   spatial hash (importers split a geometric edge into separate vertices for
   normals, materials or UV seams, and their decoded positions differ by
   floating-point crumbs). Each new triangle is placed by trilateration from the
   shared edge — the two known edge lengths pin the third point to one of two
   reflections, and the side the existing triangle sits on picks which.
3. A closed mesh cannot be flattened globally without folding back over itself,
   so each candidate is tested for **overlap against already-placed triangles** —
   Sutherland–Hodgman clipping for the shared area, with a spatial hash limiting
   the tests to triangles whose bounding boxes actually share a region (testing
   against the whole chart made the traversal quadratic). A branch that would
   fold is dropped rather than producing a phantom copy of the decal on a distant
   face.
4. Each face is emitted with both its atlas UV and its unfolded surface UV, plus
   the 2×2 matrix that rotates a slope from the unfolded frame into that face's UV
   axes — otherwise a decal crossing a seam would light as though its half on the
   other side were still upright.

The same layout draws the decal's **outline on the model**: the rectangle's
border is walked (subdivided per edge, not just cornered, since a straight line
in UV bends on the surface) and each point mapped back to a world position
through the face it lands in.

### Square in UV is not square on the model

A UV unit need not cover as much surface across as it does up. A box unwrap that
gives each face a 1/3 × 1/2 cell of a square atlas makes one unit of u span 1.5×
the world one unit of v does, so a round decal stamped into a square UV rectangle
comes out an ellipse half again as wide as it is tall.

`uv_aspect()` measures the ratio off the mesh — the tangent and bitangent lengths
implied by each triangle's positions and UVs — and the decal's height divides it
back out. The image's own aspect decides the shape wanted on the surface; the
surface's decides what UV rectangle produces that shape.

The ratio is **measured from wherever the decal is now**, not remembered from
where it was placed: a layout can be dense in one island and sparse in the next,
so a number stored at placement time goes quietly wrong the moment the decal is
dragged onto another island. A decal centred in the gutter between charts has no
triangle to ask and falls back to the mesh's own average — a UV-area-weighted
*geometric* mean, because it is a ratio (a face at 2.0 and a face at 0.5 average
to 1.0, not to 1.25). The correction is clamped to 16:1 either way, and the panel
says when one is in effect rather than applying it silently.

### Edge falloff

A decal is a rectangle of an image and the surface it lands on does not stop
there, so unless the image fades to a flat normal by its own border, the
rectangle shows up as a seam. Most do not: a great many normal maps are drawn on
a white or a black canvas, and neither is flat — white decodes to a normal tilted
45° along both axes.

**Edge falloff** cuts the outer fraction of the decal entirely and spreads a
smoothstep fade over the same width again inside it, measured along whichever
axis is nearer its edge so all four sides lose the same width. Cutting rather
than merely ramping to zero at the border is the point: a ramp leaves no hard
edge but still shows most of the canvas just inside it, and the canvas is the
problem. 0 turns it off for an image that fades out by itself; the maximum is
half the decal's half-width, past which the bands from each side would meet.

### Placing, transforming, arraying

**Point at the model.** The cursor becomes a ray (`core/picking.py`), the ray
hits a triangle by **Möller–Trumbore** over every face at once, and the
triangle's own UVs are interpolated at the hit — so what you point at is what the
decal is placed on, in perspective or in an orthographic axis view alike. The
sweep is vectorised numpy rather than a trimesh ray backend, for two reasons: the
geometry being pointed at is the *rendered* one (seam-split by the unwrap), and a
few thousand triangles is a fraction of a millisecond, cheap enough to run on
every mouse move. Both facings count, since the viewport draws with culling off.
`surface_at_uv()` and `face_at_uv()` are the inverse — from a place in the texture
back to a place on the model — which is what draws outlines and measures the
local UV aspect.

**G / S / R** move, scale and rotate the selected decal with pointer motion,
Blender-style, with `X`/`Y` constraining to a decal axis, click or `Enter`
confirming and `Esc` restoring. Moving slides the projector along the surface,
following hits across faces and continuing on the decal's own plane when the
pointer runs off the mesh, so it does not jump when it comes back on.

**The library shelf** (the `decals` directory from `metadata.json`) lists every
readable image with a cached 256 px preview, keyed by path, mtime and size and
stored in the per-user cache directory; a failed cache write costs the cache, not
the asset. Images are loaded and thumbnailed on a worker pool and uploaded on the
render thread. Dragging an entry onto the model previews it live under the cursor
and drops it where you let go.

**Array modifiers** are a non-destructive stack on each decal, evaluated at draw
time without creating scene-tree entries. Each entry receives the complete result
of the one above it, exactly as an object modifier stack does, so two arrays of N
and M evaluate to (N+1)(M+1) instances. Two modes: *axes*, which steps copies
along the decal's own right/up/forward by an offset, and *radial*, which
distributes them at equal angles around a circle whose pivot is the stored decal
transform, each copy aimed inward.

A decal can also be linked to a **material** from the Material tab, in which case
it tints the surface as well as perturbing its normal.

---

## Input, navigation and state

**Navigation** follows Blender's trackpad bindings: scroll or two-finger swipe
orbits, `+shift` pans, `+ctrl`/`+alt` or a pinch zooms; left-drag orbits,
shift-drag or middle-drag pans, right-drag zooms; clicking an axis ball aligns
the view down that axis in orthographic, and orbiting away restores perspective
(Blender's Auto Perspective). The six balls in the corner are modelled on
Blender's navigation gizmo, positive axes filled and lettered, negative ones
hollow, depth sorted so the ordering reads correctly.

The orbit is a **turntable**: horizontal motion spins about world up, vertical
motion pitches about the screen horizon. Because yaw is always about the same
world axis, the horizon stays level no matter how you got there. Nothing clamps —
you can look straight down and tumble under the subject and keep going — and the
pole degeneracy is handled by Blender's horizon blend, switching the pitch axis
from the world horizon to the camera's right as a pole approaches, rather than by
the 170° clamp moderngl-window's stock `OrbitCamera` uses.

**Pinch takes a detour to get here.** macOS sends it as `magnifyWithEvent:`;
pyglet's Cocoa view implements no gesture handlers, so the event walks the
responder chain, finds nobody, and is dropped — and it is *not* ctrl+scroll
either, since that translation is a browser convention rather than an AppKit one.
`render/trackpad.py` adds the missing methods to pyglet's view class at runtime —
both `magnifyWithEvent:` and an action for an attached
`NSMagnificationGestureRecognizer`, whichever the system delivers, with the
direct event ignored where a recognizer is armed so a pinch is never counted
twice. It goes through pyglet's own `ObjCSubclass` wrapper, which keeps the
trampoline alive; a bare `class_addMethod` with a local ctypes callback would
leave AppKit calling freed memory. It is a no-op off macOS.

**Smoothing** exists for one specific defect. macOS reports scrolling one pixel
of finger travel at a time, so moving slowly produces a run of nothing punctuated
by a step: the view sits still, then hops. The motion between steps is not in the
event stream and cannot be recovered, so instead of presenting a step as though
it happened in one frame, the app measures how fast steps are arriving and moves
that much every frame. Constant input becomes constant motion. It costs about one
step of delay at a crawl and nothing once steps arrive every frame; total travel
is identical at every setting, and mouse dragging is never smoothed.

**Undo/redo** (`Cmd/Ctrl+Z`, `+Shift` to redo) snapshots the whole editable state
— materials, decals, selections — and compares snapshots structurally, so an edit
that changes nothing records nothing. Edits made by dragging a numeric field are
coalesced into one transaction per interaction rather than one per frame.

**Persistence.** Panel layout (`layout.ini`), preferences (`prefs.json` —
navigation speeds, world lighting, export folder and depth, map switches, UI
scale, panel splits) and the decal thumbnail cache all live in the per-user
config directory: `~/Library/Application Support/MeshMap` on macOS, `%APPDATA%`
on Windows, `$XDG_CONFIG_HOME` elsewhere. Delete them to reset.

**Branding** comes from `metadata.json` at the project root:

```json
{
    "title": "OpenPainter",
    "application_icon": "./assets/application_icon.png",
    "decals": "./assets/decals"
}
```

Every field is optional and none of them can stop the app opening — a file that
is missing, malformed or full of the wrong types falls back to the built-in name,
no icon and an empty shelf, because a typo in a branding file is not a reason to
refuse to start. Paths resolve against the metadata file's own directory, so the
relative forms mean the same thing wherever the app is launched from. On macOS
the process and bundle identity are set through Cocoa before pyglet builds the
application menu, so the OS shows that title rather than "Python".

**Diagnostics.** The Console view carries the session's messages, and every input
event is written as JSON lines to `debug-log.txt` at the project root
(`core/action_log.py`), replaced each run and capped at 10 MB — enough to
reproduce an intermittent input bug from the trace alone.

---

## Algorithms at a glance

| Where | What |
| --- | --- |
| `core/baking.py` | UV-space rasterisation; curvature from `dFdx`/`dFdy` of the interpolated normal; bilinear-resample smoothing; flood dilation |
| `core/occlusion.py` | Cosine-weighted, ring-stratified hemisphere sampling with golden-angle azimuth; distance-falloff ray occlusion; masked 3×3 denoise |
| `core/bevel.py` | Dihedral edge selection; per-sector corner offset solve; slerped circular arc profile; barycentric UV inheritance |
| `core/uv_unwrap.py` | Position-welded, area-and-angle-weighted vertex normals; xatlas charting with a `vmapping` |
| `core/picking.py` | Möller–Trumbore ray/triangle intersection; 2D barycentric point-in-triangle over the atlas; near/far unprojection for perspective *and* ortho |
| `core/decal_wrap.py` | BFS mesh unfolding by trilateration; spatial-hash vertex welding; Sutherland–Hodgman clipping for overlap rejection; least-squares affine fit |
| `core/decal.py` | Height-to-normal by central differences; slope-space compositing; UV-area-weighted geometric mean of the UV aspect |
| `render/shaders/noise.glsl` | Sine-hashed 3D value noise; fBm with fractional octaves and domain-warp distortion |
| `render/shaders/layer.frag` | Twelve generators, including Worley cells and dominant-axis face projection |
| `render/shaders/preview.frag` | Cook-Torrance (GGX / Smith / Schlick); Karis roughness-aware IBL Fresnel; Mikkelsen cotangent tangent frame |
| `render/viewport.py` | Turntable orbit with horizon blend; rate-tracking scroll smoothing; separable Gaussian bloom with tone rolloff |

---

## Code layout

```
main.py                  entry point, argument parsing, --selftest
metadata.json            title, icon, decal shelf
core/
  mesh_io.py             import, backend selection, units, triangulate/weld/repair
  bevel.py               Blender's Bevel modifier, approximated
  uv_unwrap.py           source UV layout, xatlas fallback, welded normals
  baking.py              the UV-space curvature bake, the smoothing, dilation
  occlusion.py           the ray-cast ambient occlusion bake
  pipeline.py            stage keying, CPU/GL split, threading, cancel
  layers.py              the material tree: masks, colours, surface channels
  edge_wear.py           the wear formula and the noise, in numpy (the reference)
  decal.py               decal images, placement maths, the CPU composite
  decal_wrap.py          unfolding connected faces into one decal chart
  decal_thumbnail.py     cached shelf previews
  picking.py             cursor ray -> triangle -> UV, and back
  export.py              PNG writers, 8- and 16-bit (incl. 16-bit RGB by hand)
  params.py              parameter blocks, split by what they cost
  metadata.py            branding, read defensively
  action_log.py          JSON-lines input trace
  macos_app.py           Cocoa process/bundle identity
render/
  viewport.py            the window: camera, events, decals, frame composition
  composite.py           the material tree, one full-screen pass per node
  trackpad.py            macOS pinch gestures, added to pyglet at runtime
  imgui_renderer.py      ImGui 1.92 texture backend
  shaders/
    gbuffer.vert/.frag   UV-space rasterisation + the curvature formula
    layer.frag           one node of the material tree: mask, then choose
    noise.glsl           the shared value noise
    edge_wear.frag       the wear formula standalone, as a reference pass
    decal.frag           one decal, as a slope in atlas coordinates
    decal_project.*      one decal, projected from a world-space frame
    decal_wrap.*         one decal, across an unfolded surface chart
    normal_encode.frag   accumulated slopes back into a normal map
    preview.frag         viewport shading, Cook-Torrance
    bright/blur/tonemap  the glow around an emissive surface
    copy/blit/background/fullscreen
ui/
  panel.py               the sidebar, navbar and status bar
  gizmo.py               the clickable axis gizmo
  decal_gizmo.py         decal and mesh selection outlines
  light_gizmo.py         the key-light arrow
tests/                   pipeline, GL, layers, decals, picking, camera, UI
tools/make_sample.py     generates the sample assets
```

---

## Tests

```bash
python -m pytest tests/ -q
python main.py assets/sample.fbx --selftest out/   # headless bake + export + renders
```

The suite checks the maths, not just that the code runs. `tests/test_gl.py`
builds a quad whose normal ramps linearly across UV space — which makes
`dFdx`/`dFdy` analytically known — and asserts the shader matches the expression
evaluated by hand across several parameter sets; others pin that a constant
normal bakes *exactly* zero, that Offset shifts the result by exactly
`offset / 10`, that Radius is monotonic as an exponent, that axis masking
multiplies per texel by `dot(n, axis)`, and that smoothing preserves the mean.

The noise gets a statistical comparison rather than a bit-exact one,
deliberately: the hash is `fract(sin(n) * 10000.0)`, chaotic enough that GPU and
CPU transcendentals genuinely diverge. The rest of the wear formula is compared
exactly, with the noise weighted out. `core/edge_wear.py` and `core/decal.py`
exist partly for this — numpy mirrors of the shaders that can be driven without a
GL context.

Elsewhere: `test_pipeline.py` sweeps every bake field and asserts it is keyed (and
that `dilation` is *not* in the curvature key, so nudging seam padding cannot
force a re-rasterise); `test_controller.py` covers cancellation and the
never-record-a-cancelled-key rule; `test_layers.py` checks the tree, the target
pool's size bound and the compositor against the numpy mirror; `test_decal.py`,
`test_decals.py` and `test_decal_wrap.py` cover placement, transforms, the aspect
correction and the unfolding; `test_camera.py`, `test_trackpad.py`,
`test_shortcuts.py`, `test_imgui_input.py`, `test_layout.py` and
`test_ui_scale.py` cover input and layout.

`--selftest` drives the real app through a headless GL context — same controller,
same shaders, same ImGui panel — and writes the maps plus a viewport render per
preview mode, with per-stage timings.

---

## Known limitations

- **Resolution changes the curvature result.** The derivative is per-texel, so
  baking at 2048 gives a thinner, dimmer curvature than 1024 for the same mesh.
  Inherent to the method.
- **A mesh with no bevels bakes black.** The bake differentiates the interpolated
  vertex normal; a perfectly hard corner has nothing to differentiate, and the
  default `offset = −2.0` crushes the resulting wash to zero. Bevel the edges (in
  Blender, or with the Bevel panel here) or use a geometric edge-detection method
  — this one genuinely cannot see them.
- **Chart borders can flare.** A 2×2 derivative quad straddling a chart edge sees
  a discontinuous normal and reports enormous curvature. Seam padding covers most
  of it.
- **Axis masking can go negative.** `curvature *= dot(n, axis)` is applied after
  the clamp, so back-facing texels store negative curvature. It clamps to black
  downstream, but the raw map is signed.
- **No convex/concave distinction.** `dot(dx, dx)` is unsigned, so a crevice and a
  ridge are indistinguishable.
- **Eight live projectors.** Beyond that, decals fall back to the UV normal target
  while you work — correct, but softer at close range until the full-resolution
  render.
- **Transparency is unsorted.** Opacity is a preview of the channel, not a correct
  transparent pass.
- **`render/imgui_renderer.py` is a workaround** for moderngl-window still
  shipping the pre-1.92 ImGui backend, which calls a method removed in
  imgui-bundle 1.92. Delete it once upstream catches up.
