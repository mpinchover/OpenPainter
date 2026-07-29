# MeshMap — ArmorPaint's EdgeWear001, standalone

A port of ArmorPaint's procedural edge wear (Browser → Cloud → Materials →
Procedural → **EdgeWear001**) as a standalone desktop app. Import a UV-mapped
FBX, bake the curvature into its UV layout, tune the wear live on the model, and
export the mask as a texture you can drop straight onto the original mesh.

Same algorithm, same parameters, same shipped defaults.

```bash
cd meshmap && source myenv/bin/activate
python main.py                       # empty, then drag an FBX onto the window
python main.py cube.fbx              # load on startup
python main.py cube.fbx --resolution 2048 --z-up
python main.py cube.fbx --auto-unwrap   # for a mesh with no UVs at all
python main.py --ui-scale 1.8        # bigger panel and text
```

Press **B** to bake, drag the *Edge wear* sliders, press **E** to export.

### UV your mesh in Blender first

MeshMap bakes into **the UV map already on the mesh**. Unwrap in Blender before
exporting (UV Editing workspace, or `U` → Smart UV Project); the FBX exporter
always includes UV layers, so nothing else is needed. Then `edge_wear.png`
applies directly to your original mesh — plug it into a `Mix Color` factor or
into Roughness, and set the image's **Color Space to Non-Color**, because it is
a mask and not a colour.

### Bevel sharp edges before baking

A perfectly sharp edge has zero width, so the curvature bake — which
differentiates the interpolated normal per texel — has nowhere to put a
gradient. Welding averages the corner normals of the faces meeting there
instead, and that swing spreads across the *whole face*, so faces bake grey and
the edge doesn't stand out at all.

The **Bevel** panel fixes that by replacing sharp edges with a narrow strip
before the bake. It approximates Blender's Bevel modifier under Affect = Edges,
Width Type = Offset and Limit Method = Angle, and exposes its three real knobs:

```bash
python main.py chair.fbx --bevel                # 1 mm, 3 segments, 30 degrees
python main.py chair.fbx --bevel 0.01,3,30      # AMOUNT,SEGMENTS,ANGLE
```

The bevel is never exported. It inherits the source UV layout, landing in the
outer rim of each island — which on your original unbeveled mesh is the texture
right up against the edge, so that is where the wear appears.

**The bevel has to be at least a texel or two wide to bake anything.** A 1 mm
bevel on a 2 m object at 1024 is 0.2 texels and falls between samples,
leaving nothing behind. The panel reports the width it lands at and warns below
two texels; raise Amount or the bake resolution. This is the one place where
Blender's usual 1 mm doesn't transfer — it's a modelling value, and here it has
to clear the sampling rate.

It is an approximation, not a port. Blender's real bevel is
`bmesh_bevel.cc`, 8,485 lines over BMesh with five corner mesh kinds and an
Eigen least-squares solve. Edge selection by dihedral angle, the Offset width
convention, loop slide and the circular profile are faithful; corners where
three or more beveled edges meet are fanned onto the corner sphere rather than
Grid Filled, so the topology there differs. Since the geometry only ever feeds
the bake, that costs a little accuracy in the corner falloff and nothing else.

**Export** writes `edge_wear.png` and `curvature.png`, plus `normal.png` if a
decal is placed — they address your original mesh through its own UV map, so
there is nothing to ship alongside them.

### Decals

The **Decal** tab stamps an imported normal map into the mesh — a scifi vent, a
hatch, a panel seam. Import a tangent-space normal map (a grayscale image is
read as a height map and converted, since that is the only reading of it that
describes a surface), and it composites into an atlas-wide normal map exported
as `normal.png`. Plug that into a Normal Map node; everywhere outside the decal
it is flat `(0.5, 0.5, 1.0)` and changes nothing.

**Place it by pointing at the model.** Click the decal preview in the panel (or
*Place on the mesh*) and it picks the decal up: it then rides the surface under
the cursor, following it across the model as you move, and a click drops it
there. `Esc` or a right-click puts it back where it was. Under the hood the
cursor becomes a ray, the ray hits a triangle, and that triangle's own UVs are
interpolated at the hit — so what you point at is what the decal is placed on,
in perspective or in an orthographic axis view alike. Two-finger orbiting still
works while placing, so you can turn the model to reach the far side.

The placement itself is stated in UV space — **Position U/V**, **Scale** (the
fraction of the atlas it spans across, with its height following the image's
aspect) and **Rotation** — and the sliders stay live for nudging afterwards.
Substance Painter projects through a 3D gizmo instead; for anything with a sane
UV layout these come to the same thing, and the *UV checker* preview shows which
part of the mesh a given placement lands on.

**Height intensity** scales the *slope* the map describes, `xy/z`, rather than
the stored vector. That is what makes 2.0 genuinely twice as steep: scaling the
vector instead runs the normal flat against the surface and then past it, which
reads as the bump inverting rather than deepening. **Flip green** is there for a
map baked DirectX-style (−Y); everything here is OpenGL (+Y up), which is what
Blender expects.

None of it involves the bake. The decal is placed in UV space, so it needs no
geometry pass behind it: import, place, export, with no mesh baked at all if the
normal map is all you want. The mesh in the viewport lights through the decal as
you drag the sliders, using a tangent frame derived per pixel from the
derivatives of position and UV — so it reads correctly on any mesh the app can
load, without a tangent attribute.

`--auto-unwrap` (or *Bake into source UVs* in Advanced bake settings) turns the
source-UV path off and lets xatlas invent an atlas instead. Only reach for it on
a mesh with no UVs at all, and understand what you get: the atlas belongs to
MeshMap's internal triangulated copy, so the texture will not fit your original
mesh. Unwrapping in Blender is almost always the better answer.

| Key | Action |
| --- | --- |
| `1`–`6` | Preview: edge wear / curvature texture / UV checker / normals / shaded / decal normals |
| `B` / `E` / `F` | Bake / export maps / frame the mesh (also resets the view) |
| `W` / `L` | Wireframe / lighting |
| `+` / `-` | Bigger / smaller UI |
| `Esc` | Cancel a decal placement |

### Navigating

| Gesture | Action |
| --- | --- |
| scroll / two-finger swipe | Orbit (turntable) |
| pinch, or ctrl (or alt) + scroll | Zoom |
| shift + scroll | Pan |
| left-drag | Orbit |
| shift-drag or middle-drag | Pan |
| right-drag | Zoom |
| click an axis ball | Align to that axis, orthographic |

The gestures are Blender's own 3D-viewport trackpad bindings, from
`blender_default.py`: `TRACKPADPAN` orbits, `+shift` pans, `+ctrl` zooms.

**Pinch** takes a detour to get here. macOS sends it as `magnifyWithEvent:`, and
pyglet's Cocoa view implements no gesture handlers, so the event found no taker
and the pinch did nothing — and it is *not* ctrl+scroll either, since that
translation is a browser convention rather than an AppKit one. `render/trackpad.py`
adds the selector to pyglet's view class at runtime, which the Objective-C
runtime permits for an already-registered class. It is a no-op off macOS.

Speeds are adjustable under **Settings → Viewport navigation** — orbit, pan and
zoom, plus a separate multiplier for scroll and trackpad gestures, since a
trackpad's stream of small deltas and a wheel's whole notches rarely want the
same rate. There are invert-axis toggles for natural scrolling. Everything is
remembered between launches in `prefs.json`. A speed of 1.0 orbits at 0.4
degrees per pixel, which is Blender's own `view_rotate_sensitivity_turntable`.

**Smoothing** is there for one specific defect. macOS reports scrolling one
pixel of finger travel at a time — `[NSEvent deltaY]`, which pyglet reads
(`pyglet/window/cocoa/pyglet_view.py`, `getMouseDelta`), carries a tenth of a
point per pixel, and `scrollingDeltaY` carries the same pixel count directly, so
neither accessor knows anything finer. Move slowly and a pixel takes several
frames to cross: what arrives is a run of zeros punctuated by a step, and the
view sits still and then hops. That is the stutter you feel at low speed and not
at high.

The motion between steps is not in the event stream and cannot be recovered, so
the repair is to stop presenting a step as though it happened in one frame. The
app measures how fast steps are arriving — one step divided by the frames since
the last one, which holds flat between steps where a per-frame average would
spike and sag — and moves that much every frame. Constant input becomes constant
motion. It costs about one step of delay at a crawl, and none at all once the
steps arrive every frame, since then there is nothing to spread. Total travel is
identical at every setting; 1.00 follows the gesture's speed exactly, 0 applies
each step whole the moment it lands, and mouse dragging is never smoothed.

The orbit is a **turntable**, ported from the non-trackball branch of Blender's
`viewrotate_apply` (`view3d_navigate_view_rotate.cc`): horizontal motion spins
about world up, vertical motion pitches about the screen horizon. Because yaw is
always about the same world axis, the horizon stays level no matter how you got
there — a trackball cannot promise that, and the roll it accumulates is what
makes one feel like it is fighting you.

Nothing clamps, so you can look straight down and tumble under the subject and
keep going. moderngl-window's stock `OrbitCamera` clamps its polar angle to a
170° band precisely to dodge the pole degeneracy; what replaces the clamp here
is Blender's own horizon blend, which switches the pitch axis from the world
horizon to the camera's right as you approach a pole.

### The axis gizmo

Six axis balls sit in the top-right corner, modelled on Blender's navigation
gizmo (`view3d_gizmo_navigate_type.cc`) down to the theme colours. Click one to
look straight down that axis in **orthographic** projection; orbiting away
restores perspective, which is Blender's Auto Perspective behaviour. Positive
axes are filled and lettered, negative ones hollow, and the balls are depth
sorted so the ordering reads correctly.

Toggle it from the *Axis gizmo* checkbox.

### Axis convention

The viewport is **Z-up, like Blender**, and the axis colours are Blender's own.
A Y-up source — which is what Blender's FBX and glTF exporters write — is
rotated into that convention on import, the same +90° about X that Blender's
importers apply. So the gizmo, the bake **Axis** dropdown and the size readout
all mean what they mean in Blender: `-Z` is down, and gravity-driven scuffing
uses `-Z`.

Pass `--z-up` (or tick *Source is Z-up*) if a file genuinely stores Z as up
already, and it will be used as-is.

The conversion only affects how the model is presented and how the bake's Axis
option is interpreted. `MeshInfo.to_world` records it, so anything needing the
source file's own coordinates can invert it.

---

# Part 1 — How the edge wear works

Two pieces of ArmorPaint, both reproduced verbatim: a **bake** that finds the
edges, and a **node graph** that turns them into wear.

## Step 1: find the edges by differentiating the normal

The insight ArmorPaint uses is that an edge is *where the surface normal changes
fastest*. Not where two faces meet at an angle, not how a vertex compares to its
neighbours — just the rate of change of the normal across the surface.

The mesh is rasterised **into UV space**: the vertex shader feeds the UV in as
clip-space position instead of projecting the model.

```glsl
// render/shaders/gbuffer.vert  (ArmorPaint: make_paint.c:193)
gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
```

Every fragment the rasteriser produces is therefore one texel of the output
atlas, holding the world position and normal of the surface point that lands
there. This matters enormously for the next line, because GPU fragment
derivatives (`dFdx`/`dFdy`) are computed across 2×2 fragment quads — and in UV
space, *adjacent fragments are adjacent texels*. `dFdx(n)` is not a screen-space
quantity here. It is "how much does the normal change between this texel and the
one next to it in the atlas."

```glsl
// render/shaders/gbuffer.frag  (ArmorPaint: make_bake.c:23-32)
vec3 dx = dFdx(n);
vec3 dy = dFdy(n);
float curvature = max(dot(dx, dx), dot(dy, dy));
curvature = clamp(pow(curvature, (1.0 / radius) * 0.25) * strength * 2.0
                  + offset / 10.0, 0.0, 1.0);
if (axis != XYZ) curvature *= dot(n, axis);
```

Line by line:

- **`dot(dx, dx)`** is the *squared* length of the derivative. Squared, so cheap
  (no `sqrt`), and unsigned — which means **this method cannot distinguish convex
  edges from concave creases**. Both read bright. ArmorPaint has no
  convex/concave control for curvature; the only directional control is the axis
  mask below.
- **`max` of the two axes** takes whichever direction bends more, so an edge
  running diagonally through the atlas registers as strongly as an axis-aligned
  one.
- **`pow(curvature, (1/radius) * 0.25)`** is the part that makes it work at all,
  and the part whose name is most misleading. See below.
- **`* strength * 2.0 + offset / 10.0`** is a plain gain and bias.
- **`clamp(..., 0, 1)`** happens *before* the axis mask, so axis masking can
  drive the stored value negative.

### Why the `pow` is the whole trick

The raw derivative has an absurd dynamic range. Measured on a tight-bevelled
cube at 1024:

| Where | raw `dot(dx,dx)` |
| --- | --- |
| Middle of a flat face | `4.8e-20` |
| On the bevel | `3.9e-03` |

That is **seventeen orders of magnitude**. Written straight to a texture it is
pure black with a few isolated white texels — useless. The exponent crushes that
range into something visible:

```
exponent = (1 / radius) * 0.25      # radius 2.0 -> 0.125

raw 1e-08  ->  pow(raw, 0.125) = 0.100
raw 1e-06  ->  pow(raw, 0.125) = 0.178
raw 1e-04  ->  pow(raw, 0.125) = 0.316
raw 1e-02  ->  pow(raw, 0.125) = 0.562
```

Four orders of magnitude of input become a factor of 5.6 in output. The same two
texels above end up at **0.0077** and **1.0** — a clean map.

So **"Radius" is not a distance.** It is the reciprocal of a gamma exponent. A
larger Radius means a smaller exponent, which lifts the faint falloff around each
edge further up, which *looks* like a wider band — hence the name. But nothing in
the code has any notion of spatial extent. Its effect is far stronger than the
name suggests: at Radius 0.5 the sample cube bakes essentially nothing, at 4.0 it
floods.

**Offset** then subtracts a floor. The shipped `-2.0` means `-0.2`, which is what
pushes the residual grey of nearly-flat regions down to true black.

### The smoothing is a resolution round-trip

`Smooth` is not a blur kernel. ArmorPaint copies the map into a render target at
**95% size** and straight back, N times, letting the hardware's bilinear filter
do the work (`render_path_paint.c:461`). Ported as-is in `core/baking.py`:

```python
for _ in range(iterations):
    blur_fbo.use();  curvature.use(0);  draw()   # down to 95%
    curv_fbo.use();  blur_texture.use(0); draw() # back up to 100%
```

Each round-trip is a gentle, slightly anisotropic smear. It redistributes rather
than brightens: a test asserts the mean is unchanged within 5%.

## Step 2: turn curvature into wear

That is the whole EdgeWear001 node group, decoded from
`cloud.armory3d.com/cloud/materials/Procedural/EdgeWear001.arm`:

```
Group Input.Strength ─┐
Group Input.Radius  ──┤
                      ▼
              [Curvature Texture]   BAKE_CURVATURE, Offset -2.0
                      │
Group Input.Value ─► [Math x10] ─► [Wear Noise]   TEX_NOISE
                                        │ Factor
                                        ▼
                                 [Wear Amount x0.6]
                                        │
              [Curvature] ─► [Break Up: A - B]
                                        │
                                 [Contrast x3, clamped]
                                        │
                                 Group Output.Mask
```

Eight nodes, which reduce to one line:

```
mask = clamp((curvature - noise * 0.6) * 3, 0, 1)
```

The **subtract** is the interesting choice, and it is why the node is called
"Break Up". A multiply would dim the wear evenly. A subtract *erodes* it: wherever
the noise runs high the edge drops out completely, so the wear is patchy and
interrupted rather than a uniform stripe tracing every bevel. That single node is
most of what makes the result look weathered instead of stamped.

The noise is ArmorPaint's own `tex_noise` from
`nodes_material/noise_texture_node.c`, ported function for function in
`core/edge_wear.py` — a sine-hashed value noise, fBM'd over `detail` octaves with
`roughness` amplitude falloff and `lacunarity` frequency step, with the sample
point pre-warped by the noise itself (`distortion`).

Two details that are easy to get wrong:

- **It is sampled at `bposition`, not the UV.** `bposition` is the object-space
  position normalised over the bounding box (`(pos + hdim) / dim` in ArmorPaint).
  Because the noise is evaluated in 3D object space, the pattern is continuous
  across atlas seams — a UV-space noise would visibly break at every chart
  border.
- **`Value` is not the noise scale.** It feeds a ×10 Math node first, so the
  actual scale is `value * 10`.

## What this method needs from your mesh

It differentiates the **interpolated vertex normal**. A perfectly hard 90° corner
— flat-shaded, constant normal per face — has nothing to differentiate and bakes
**black**.

That is not a limitation of this port. ArmorPaint behaves identically, and it is
precisely why the hard-surface workflow is "bevel your edges before you bake."

Measured here:

| Mesh | Result |
| --- | --- |
| Plain 8-vertex cube | A shallow `0.018`–`0.079` wash across every face (welding averaged the corner normals, so each face ramps gently corner to corner) — no edge peak anywhere. The shipped `offset = -2.0` then crushes the lot to zero. Black. |
| Same cube, tight chamfer | Patchy wear along every edge at the shipped defaults. |

If your model has hard edges and no bevel, add one — or use a geometric
edge-detection method, because this one genuinely cannot see them.

---

# Part 2 — How the program works

## The pipeline

```
FBX / OBJ / GLB
      │
      ▼  core/mesh_io.py      trimesh natively, else the assimp CLI via .glb
   validate ─ triangulate ─ weld (never across a UV seam) ─ repair winding
      │
      ▼  core/bevel.py       optional: a strip of geometry on every sharp edge,
   so the curvature bake has somewhere to put its gradient
      ▼  core/uv_unwrap.py    the mesh's own UV map, vertex for vertex
   (or --auto-unwrap: xatlas charts, packs and splits a new atlas,
    handing back a vmapping to the original vertices)
      │
      ▼  core/baking.py       ══ GL, on the render thread ══
   rasterise in UV space ──▶ position / normal / curvature
   dFdx,dFdy of the normal ──▶ the formula above
   95% round-trip blur × Smooth
      │
      ▼  core/baking.py       ══ CPU, on a worker thread ══
   dilate() pads chart borders outward past the seams
      │
      ▼  render/shaders/edge_wear.frag   ══ every frame ══
   mask = clamp((curvature - noise * wear) * contrast, 0, 1)
      │
      ├──▶ viewport preview, on the mesh
      └──▶ edge_wear.png  +  curvature.png
```

## The expensive/cheap split

This is the architectural decision the whole thing is built around, and it is why
the sliders feel instant.

**Baking is expensive; thresholding is cheap.** Anything that requires
re-rasterising the mesh lives in `BakeParams` and needs an explicit re-bake.
Anything that is arithmetic on already-baked textures lives in `EdgeWearParams`
and re-runs as a full-screen pass every frame a slider moves.

`core/pipeline.py` enforces the split structurally: `BakeController` has no
attribute for the wear parameters at all, so touching them *cannot* invalidate a
bake stage. A test asserts exactly that (`test_wear_params_never_dirty_the_bake`).

## Stage keying

The bake is three stages, and each is keyed on its own parameters **plus the key
of the stage before it**:

```python
unwrap_key    = (mesh_token, unwrap_params.key(), resolution)
curvature_key = unwrap_key + bake_params.curvature_key()
post_key      = curvature_key + (dilation,)
```

Cascading, so invalidation flows forward automatically and never backward:

| You change | Re-runs |
| --- | --- |
| Seam padding | `post` |
| Strength / Radius / Offset / Smooth / Axis | `curvature`, `post` |
| Bake resolution, atlas settings, or the mesh | everything |
| Any *Edge wear* slider | nothing — it is not a bake stage |

The Bake button reports how many stages are stale before you commit. A test
sweeps every bake field and asserts it is keyed; another asserts `dilation` is
*not* in the curvature key, so nudging seam padding cannot force a re-rasterise.

## Threading

Stages are tagged CPU or GL. GL work must happen on the thread that owns the
context, so `pump()` — called once per frame from the render loop — runs GL
stages inline and hands CPU stages to a worker thread, keeping the window
responsive with a live progress bar and a working Cancel.

Cancellation is handled carefully: a cancelled stage returns whatever it had
finished, which is not a valid result, so its key is deliberately *never*
recorded. Otherwise the next bake would skip it and keep the garbage. There is a
regression test for this.

## Seam padding

Single-sample rasterisation leaves a texel-wide hole around every chart border.
Left alone, bilinear filtering in a renderer pulls that empty gutter in and draws
a dark fringe along every seam. `dilate()` bleeds covered texels outward N times
to prevent it.

It always re-derives from the G-buffer rather than from a previous padded result
— otherwise re-running with a different width would pad an already-padded map,
compounding the dilation. (That was a real bug, found and fixed; it has a test.)

## Preview and export are the same pixels

The viewport samples `tex_output`, and export reads that same framebuffer back
off the GPU rather than recomputing on the CPU. The PNG is exactly what you
tuned, by construction.

`edge_wear.png` is the product. `curvature.png` is the bake behind it, exported
alongside because it is the lossless input — you can re-derive any wear from it,
but not the reverse.

---

# Parameter reference

**Bake — the Curvature Texture node. Changing these re-bakes.**

| Parameter | Default | What it does |
| --- | --- | --- |
| Bake resolution | 1024 | Also re-packs the atlas. **Changes the result**: the derivative is per-texel, so texel density is part of the measurement. |
| Strength | 0.5 | Plain gain on the lifted derivative. First knob to reach for if the bake is too dark. |
| Radius | 2.0 | **An exponent, not a distance** — `pow(c, (1/radius) * 0.25)`. Larger lifts harder and widens the apparent band. Dramatic: 0.5 gives nothing, 4.0 floods. |
| Offset | -2.0 | Added as `offset / 10`. Negative crushes near-flat areas to black. |
| Smooth | 1 | Blur iterations, each a 95% resolution round-trip. |
| Axis | XYZ | Anything else multiplies by `dot(n, axis)` — only edges facing that way wear. |
| Seam padding | 4 | Texels of dilation past each chart border. Under *Advanced*. |

**Edge wear — the node group. Live, no re-bake.**

| Parameter | Default | What it does |
| --- | --- | --- |
| Value | 1.0 | Feeds a ×10 Math node into the noise Scale, so the scale is `value * 10`. Higher = finer break-up. |
| Wear amount | 0.6 | How hard the noise erodes the curvature before the subtract. Higher = patchier. |
| Contrast | 3.0 | Final multiply, clamped to 0..1. |
| Detail / Roughness / Lacunarity / Distortion | 5.0 / 0.7 / 5.0 / 0.15 | The Wear Noise `TEX_NOISE` settings, as shipped. |

*Reset to EdgeWear001* restores every value decoded from the `.arm` file.

**If the bake comes out black:** switch to preview `2` (Curvature texture) to see
what the bake actually found before the wear graph touches it. Then raise
Strength toward 1.5–2.0 and Radius toward 3–4. If it is *still* black, your mesh
has no bevels — see "What this method needs from your mesh".

---

# Verifying it

```bash
python -m pytest tests/ -q                        # 65 tests, ~1.4s
python main.py assets/sample.fbx --selftest out/  # headless bake + render
```

The tests check the port, not just that the code runs.

`tests/test_gl.py` builds a quad whose normal ramps linearly across UV space,
which makes `dFdx`/`dFdy` analytically known, and asserts the shader matches
ArmorPaint's expression evaluated by hand across three parameter sets. Others
pin that a constant normal bakes *exactly* zero, that Offset shifts the result by
exactly `offset/10`, that Radius is monotonic as an exponent, that axis masking
multiplies per texel by `dot(n, axis)`, that smoothing preserves the mean, and
that the shipped defaults still equal the values decoded from `EdgeWear001.arm`.

The noise gets a statistical comparison rather than a bit-exact one, deliberately:
ArmorPaint's hash is `fract(sin(n) * 10000.0)`, chaotic enough that GPU and CPU
transcendentals genuinely diverge. The rest of the wear formula is compared
exactly, with the noise weighted out.

`--selftest` drives the real app through a headless GL context — same controller,
same shaders, same ImGui panel — and writes the maps plus viewport renders.

# Layout

```
main.py                  entry point, argument parsing, --selftest
core/
  mesh_io.py             import, backend selection, triangulate/weld/repair
  bevel.py               Blender's Bevel modifier, approximated
  uv_unwrap.py           source UV layout, or the xatlas fallback + vmapping
  baking.py              the UV-space curvature bake, the 95% blur, dilation
  pipeline.py            stage keying, CPU/GL split, threading, cancel
  edge_wear.py           the EdgeWear001 graph and tex_noise, in numpy
  export.py              PNG writers, 8- and 16-bit
  params.py              BakeParams / EdgeWearParams, split by cost
render/
  viewport.py            window, orbit/pan camera, live EdgeWear001 pass
  imgui_renderer.py      ImGui 1.92 texture backend (see below)
  shaders/
    gbuffer.vert/.frag   UV-space rasterisation + the curvature formula
    edge_wear.frag       the node group + ArmorPaint's tex_noise
    copy.frag            the 95% round-trip blur
    preview.frag         viewport shading modes
ui/panel.py              the slider panel
ui/gizmo.py              the clickable axis gizmo
tests/                   pipeline, GL, controller, and UI-scale tests
```

# Install

```bash
python -m venv myenv && source myenv/bin/activate
pip install -r requirements.txt
brew install assimp          # macOS; FBX parsing goes through it
```

> pyassimp's last release (5.2.5) imports `distutils`, removed in Python 3.12, so
> it cannot be imported on any current Python. `core/mesh_io.py` tries it first
> and falls back to converting through the `assimp` CLI — the same library, minus
> the broken ctypes wrapper. Formats trimesh reads natively (`.obj`, `.glb`,
> `.gltf`, `.stl`, `.ply`, `.dae`, `.off`) skip Assimp entirely.

# Known rough edges

- **Resolution changes the result.** The derivative is per-texel, so baking at
  2048 gives a thinner, dimmer curvature than 1024 for the same mesh. Inherent to
  the method; ArmorPaint has it too.
- **Chart borders can flare.** A 2×2 derivative quad straddling a chart edge sees
  a discontinuous normal and reports enormous curvature. Seam padding covers most
  of it. Same artifact in ArmorPaint.
- **Axis masking can go negative.** `curvature *= dot(n, axis)` is applied *after*
  the clamp, exactly as ArmorPaint does it, so back-facing texels store negative
  curvature. It clamps to black downstream, but the raw map is signed.
- **No convex/concave distinction.** `dot(dx, dx)` is unsigned, so a crevice and a
  ridge are indistinguishable. This is the method, not the port.
- **`render/imgui_renderer.py` is a workaround** — moderngl-window 3.1.1 still
  ships the pre-1.92 ImGui backend, which calls a method removed in imgui_bundle
  1.92, and only 1.92.x has Python 3.14 wheels. Delete it once upstream catches
  up.
- Panel layout and UI scale persist to `~/Library/Application Support/MeshMap/`
  (`layout.ini` and `prefs.json`). Delete them to reset the window arrangement.
