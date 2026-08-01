# MeshMap — ArmorPaint's EdgeWear001, standalone

A port of ArmorPaint's procedural edge wear (Browser → Cloud → Materials →
Procedural → **EdgeWear001**) as a standalone desktop app. Import a UV-mapped
FBX, bake the curvature into its UV layout, tune the wear live on the model, and
export the mask as a texture you can drop straight onto the original mesh.

Same algorithm, same parameters, same shipped defaults.

The window's name, its icon and the decal shelf come from **`metadata.json`** at
the project root:

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
refuse to start. Paths are resolved against the metadata file's own directory, so
the relative forms above mean the same thing wherever the app is launched from,
and the icon is resolved to an absolute path before it is handed over so that
moderngl-window's resource finders are bypassed rather than consulted. Whether
the icon shows at all is the platform's business: the headless backend has no
window to put one on and says so, which is caught rather than raised.

```bash
cd meshmap && source myenv/bin/activate
python main.py                       # opens on a starter cube; drag an FBX on to swap it
python main.py cube.fbx              # load on startup
python main.py cube.fbx --resolution 2048 --z-up
python main.py cube.fbx --auto-unwrap   # for a mesh with no UVs at all
python main.py --ui-scale 1.8        # bigger panel and text
```

Press **B** to bake, build a texture in the sidebar, press **E** to export.

It opens on a plain 0.5 m cube with a texture already on it — *Texture 01*, a
flat colour to build up from. The cube is box-unwrapped, so it takes the same
path an imported mesh does — bake into its own UVs, place decals on it, export maps that
fit it. It is deliberately **sharp**: the Bevel panel is what puts a bevel on,
and edge wear finds nothing to grip until it does. Drop a mesh on the window at
any point to work on that instead.

### The window

A bar across the top holds the view controls — what to look at, and how it is
drawn. The **Shaded** view is the texture, lit; **Normals** is the decal normal
map. Those are the two things the app makes, so those are the two things to
look at.

The sidebar down the left is the parameters, tabbed by which half of the
pipeline they belong to, and the status bar sits along the bottom. Drag the
sidebar's right edge to widen or narrow it — up to half the window, since the
model is the thing being worked on — and the 3D view is re-laid out around
whatever is left. Nothing floats: all three are part of the frame, the 3D view is laid out in what is
left, and the projection and the cursor rays are built from that rect rather
than from the whole window — so nothing is ever hidden behind a panel and
pointing at the model lands where you point.

The maps themselves are visible where they are being made rather than in a
window of their own: the Texture tab's tree carries a swatch or a thumbnail per
row, and the Decal tab shows the imported map. The model is the preview for
everything else -- at the size and on the surface it will be seen at.

### UV your mesh in Blender first

MeshMap bakes into **the UV map already on the mesh**. Unwrap in Blender before
exporting (UV Editing workspace, or `U` → Smart UV Project); the FBX exporter
always includes UV layers, so nothing else is needed. Then the exported maps
apply directly to your original mesh — plug `color.png` into Base Color and the
rest into the sockets they are named after, and set everything except
`color.png` to **Color Space: Non-Color**, because they are data and not colour.

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

**File → Export textures** (or **E**) writes a standard PBR set and nothing
else: `color.png`, `metallic.png` and `roughness.png` from the texture,
`normal.png` from any decals, and `ao.png` from the bake. They all address your
original mesh through its own UV map, so there is nothing to ship alongside
them.

The edge wear is not a file of its own, because it is not a separate product:
it is *in* `color.png`, wherever the texture put it. Nor is the curvature bake,
which is an input to that rather than an output of it. Alpha and emission are
there to shade the viewport by and stop at the window — a PBR set is these five.

The export itself asks one question — where to put things — and that is the only
one it asks: a folder chooser, opening wherever the last export went, with the
maps landing in a folder named after the mesh. Everything else about it is
answered in advance under **Settings → Export**: the folder it starts from, the
bit depth, and **which of the five to write**. The export lives in the **File**
menu next to **Import mesh**, because it belongs to no single tab — it takes what
the Bake, Texture and Decal tabs have each produced.

**Four of the five cost nothing to leave out and one costs a great deal.**
Colour, metallic and roughness are one composite pass over the texture; normals
are one pass over the decal. Switching those off only stops a file being
written. Ambient occlusion is the one stage that traces rays, and it is most of
what a bake costs — 4.5s of a 4.9s bake at 1024, 96s of 103s at 4096 — so
switching it off takes it out of the bake as well, which is the point of being
able to. Switching it back on needs a re-bake, and the Bake button says so.

The stage still *runs* when it is off; it just clears its map and records its
key. One that were skipped outright would leave the bake looking permanently
stale, since a stage that never records a key is a stage that is always
pending.

### Textures

The **Texture** tab is where a texture is built, and the active one resolves to
`color.png`. Press **New** and you get one flat colour, called *Texture 01*.
Change its **type** to a mask — edge wear or noise — and it grows a white and a
black side underneath, each of which can be another colour or another mask. Edge
wear picking between rust and bare metal is one mask; edge wear picking between
rust and a noise that itself picks between two greys is three.

**Every texture you make stays around.** The dropdown at the top of the tab
lists them and switches between them, so a variant is something to come back to
rather than something to lose; past five it grows a search box, which matches
anywhere in the name and ignores case. *Remove* drops the active one and falls
back to its neighbour.

A colour is a **material**, not just an RGB: alongside the colour sit
**metallic**, **roughness**, **alpha** and **emission**, and the mask blends
all five the same way — one pass with two attachments, so a surface follows its
colour wherever the tree puts it. Each comes out as its own map rather than a
packed ORM, because they are separate inputs in every shader graph they end up
in and packing is a step to undo. Emission is written as a colour (the
surface's own, times how much light it gives off), so strengths above 1 clip
there — a PNG stops at white — while the viewport keeps showing the difference.

The Shaded view shades it the way Blender's Principled BSDF does:
**Cook-Torrance** specular with a GGX distribution, Smith geometry and Schlick
Fresnel, roughness squared into the GGX alpha so the middle of the slider looks
like the middle of Blender's. Metal takes its colour from what it reflects
rather than from a diffuse term; roughness decides how sharply. The lamp is
given a small size — a light of no size puts a mirror's highlight inside a
single pixel, so polishing a surface would read as *removing* the highlight
instead of sharpening it.

**Emission is a light, not a bright colour.** Past white a surface cannot get
any brighter on screen, so cranking the slider used to do nothing but drive the
colour to white: each channel clipped at 1 in turn and the hue climbed to the
corner of the cube. Now the scene is drawn into a target with room above white,
whatever is brighter than white is blurred and added back, and the result rolls
off as a whole colour rather than channel by channel. An emissive red surface
stays red and spills a red glow, and the glow is what carries how bright it is.
The blur runs at a quarter resolution, which is most of the cost of a bloom and
none of the detail.

**World lighting**, under **Settings**, is both lights: the key light, and the
world it stands in.

**Rotation** walks the key light around the model — 0° puts it on the +X axis
and it turns towards +Y, counting the way the gizmo in the corner does. A line
under the slider names where it is standing and which way it is therefore
shining (*Light at +X, shining toward −X*), rounded to the nearest eighth,
because the exact angle is already on the slider and what you want from a line
of text is which side of the model is lit.

And while you are moving it, **an arrow appears in the viewport**: a lamp
standing at the light's own position, an arrow from it into the model, and the
faint ring of the path it travels as the slider turns. Text can say *+X*, but a
model is a 3D thing seen from an angle, and *which side of what I am looking at
is this* is answered fastest by pointing at it. It comes up on hover rather than
on the first pixel of movement — otherwise you are turning something you cannot
see yet — and fades out a second or so after you let go, so it is not another
thing permanently in the way.

It is drawn in ImGui's background draw list, above the 3D view and below every
panel, exactly as the navigation gizmo is: no shader, no depth buffer, no GL
state to hand back. Unlike the gizmo it is projected through the camera's
matrices rather than by its axes alone, because it is a thing standing in the
scene at a place rather than a widget pinned to a corner — which also means a
lamp behind the camera has to be dropped rather than divided by a negative `w`
and drawn mirrored through the middle of the view.

The light is **anchored to the model, not to the camera**. It used to be a
headlight — `eye - target`, plus an offset up and to the right — which meant it
could not be moved at all: orbiting took it along, and every view was lit
identically from the same relative angle. Anchoring it is what makes putting the
light somewhere mean anything, and it is why orbiting now shows you a
differently lit side rather than the same one from further round. Elevation is
fixed at 38°, which is where the headlight already sat; the useful control is
which side the light comes from, not how steeply.

**Strength** and **Colour** are the world around it — not an HDRI, just a
two-tone gradient, brighter above than below. Without it a metal has nothing to
reflect and a rough surface has nothing to scatter, and both render black
however their own sliders are set. A rough surface reflects as much light as a
polished one, so roughness blurs the direction it gathers from rather than
dimming what it gathers — fading it out instead is what once made a fully rough
metal a black hole.

Everything in the tree can be **renamed** by double-clicking its row — the
texture itself included, since that is the row at the top. A renamed row shows
that name and nothing else; the row being renamed shows only the field. A name describes the
part of the texture rather than the kind of thing it currently is, so it
survives changing a colour into a mask and back, and the automatic label (the
mask's kind, or the colour's hex code) follows it in brackets. Clear the name
and that label takes over again.

Two mask kinds so far: **edge wear** (the curvature bake — white on the edges)
and **noise** (the same field the wear pass breaks up with, thresholded by bias
and contrast). A node keeps both sets of settings, so switching kinds and back
finds them intact.

**A mask is a continuous field, not a stencil** — edge wear ramps up as the
surface curves, noise wanders through every value between. Mixing the two sides
by that directly gives a blend everywhere the field sits mid-way, which reads as
the colours bleeding into each other rather than as a boundary. So every mask
has a **Threshold**, the level that divides its two sides, and a **Softness**
that is zero by default: every texel belongs to one side or the other, and the
boundary is a boundary. Raise Softness to feather it back into a band.

**The Shaded view is the texture**, lit. A new mask starts as plain black and
white, so what you see on the model is the mask itself — edge wear white on the
edges, or a noise in black and white — until you put colours, or another mask,
under it. Colour is then something you add deliberately rather than something to
undo. Before a bake, or with no texture at all, Shaded draws plain grey.

**The sidebar's two halves do different jobs.** The tree is binary and its
depth is unbounded, so indenting the whole thing runs out of panel before it
runs out of tree — a node graph is the usual escape and this is the other one.
**Inspector** on top edits whatever is *selected*: its type, and either its
colour — a swatch that opens the picker in a popup, since a picker is as tall
as it is wide and is used in bursts — or that mask's own sliders, all of them,
laid out flat with nothing folded behind a header. **Texture tree** underneath
is the tree itself, one line per slot with a swatch or a thumbnail, for
selecting with. Depth costs a line rather than a column.

The two split the tab evenly, each scrolling on its own, and the divider
between them drags to give one more room than the other — which is remembered
between runs. Fixed shares rather than "whatever the contents need": the
inspector's height changes with what is selected, and a tree that jumped about
as it did would be a tree you could not keep your place in.

The thumbnails come free from how the tree is rendered: **one full-screen pass per node, bottom up**
(`render/composite.py`). Everything here is UV space already, so a node's
children are just textures by the time it runs — no recursion in GLSL, no shader
generated per tree, and depth costs passes rather than shader complexity. Each
node therefore ends up holding a real texture, which is exactly what the panel
wants to show. Targets are recycled, so a tree of depth *d* needs *d + 1* of
them however wide it is, and nesting stops at 8.

**Edge wear is tuned on the layer that uses it**, and nowhere else. The Bake tab
used to carry a second copy of the same sliders, driving a set of settings that
belonged to nothing in particular — two places to change one thing, disagreeing
by default. The layer is the only copy now, and what it produces reaches the
export the same way every other layer does: through `color.png`.

### Ambient occlusion

`ao.png` is the one map here that is not arithmetic on the G-buffer. Curvature
is a derivative of the normals at a texel — it says what the surface does
*there*. Occlusion depends on geometry that may be nowhere near it, on the
surface or in the atlas, so it is a real ray cast: 32 rays per covered texel,
into the hemisphere it faces, against the same beveled and unwrapped geometry
the bake rasterised. Through Embree where `embreex` is installed, which is what
makes a million texels a few seconds rather than a coffee break.

Three things make a modest ray count usable. The rays are **cosine-weighted**,
so an unoccluded surface comes back at exactly 1 without a `dot(n, l)` term.
They are **stratified in both directions** — elevation walks outwards one ring
per sample, azimuth turns by the golden angle — so they spread over the
hemisphere instead of clumping the way chance leaves them, and each texel's
spiral starts at its own angle so the residue looks like grain rather than a
pattern. Then a **3×3 average over covered texels only** takes the grain out,
which is far cheaper than the rays that would smooth it. The blur does not wrap
at the edges of the map: an atlas is not a tiling texture, and a chart against
the left border has nothing to do with one against the right.

Rays stop at **20% of the model's bounding-box diagonal**. Without a limit,
occlusion means "is there anything at all in this direction", which on a closed
model darkens every concave surface no matter how far away the far side is; with
one, it means "is there anything *nearby*", which is what reads as contact. A
fraction rather than a distance, so the same setting suits a bolt and a bridge.

The bevel matters here as much as it does to the curvature: a perfectly sharp
edge occludes nothing, because there is no surface turned into the corner to
catch a ray.

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
fraction of the atlas it spans across) and **Rotation** — and the sliders stay
live for nudging afterwards. Substance Painter projects through a 3D gizmo
instead; for anything with a sane UV layout these come to the same thing.

**The decal's own edge is faded out.** A decal is a rectangle of an image, and
the surface it lands on does not stop there — so unless the image fades to a
flat normal by its own border, the rectangle shows up as a seam around it. Most
do not: a great many normal maps are drawn on a canvas, and neither white nor
black is a flat normal (white decodes to one tilted 45° along both axes). **Edge
falloff** cuts the outer fraction of the decal and feathers the fade over the
same width again inside it, so what is stamped is the middle of the image.
Cutting rather than merely ramping to zero at the border matters: a ramp leaves
no hard edge but still shows most of the canvas just inside it, and the canvas is
the problem. 0 turns it off for an image that fades out by itself.

A transparent texel gets a **flat normal** before upload, too. Nothing samples
one directly — the alpha multiplies it out — but every bilinear tap and every
mipmap level around the edge of the artwork averages it in, and left as the
white the canvas happened to be, that averaging invents a tilt that was never
drawn.

**A square of UV space is not a square of the model**, and that difference is
what the decal's height is worked out from. A unit of u need not cover as much
surface as a unit of v: the starter cube's box unwrap gives each face a 1/3 × 1/2
cell of a square atlas, so one unit of u spans 1.5× the world one unit of v does,
and a round decal stamped into a square UV rectangle came out an ellipse half
again as wide as it was tall. `uv_aspect` measures that ratio off the mesh —
the tangent and bitangent lengths implied by each triangle's positions and UVs —
and the height is divided by it, so a round decal is round. The image's own
aspect still decides the shape wanted; the surface's decides what UV rectangle
produces it.

The ratio is **measured from wherever the decal is now**, not remembered from
where it was placed: `face_at_uv` finds the triangle under its centre each time
the placement moves. A layout can be dense in one island and stretched in the
next, so a number stored at placement time goes quietly wrong the moment the
decal is dragged onto another island or the mesh is re-baked into a different
atlas. A decal centred in the gutter between charts has no triangle to ask, and
falls back to the mesh's own average — a UV-area-weighted *geometric* mean,
because it is a ratio: a face at 2.0 and a face at 0.5 average to 1.0, not 1.25.
The Decal tab says when a correction is in effect rather than applying it
silently.

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
| `1` / `2` | Preview: shaded (the mask tree) / normals (the decals) |
| `B` / `E` / `F` | Bake / export maps (asks where) / frame the mesh (also resets the view) |
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
The same tab carries **World lighting** — see above — and the interface scale.

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
      ▼  core/occlusion.py    ══ CPU, on a worker thread ══
   cast rays into the hemisphere at every covered texel ──▶ ao.png
      │
      ▼  render/shaders/layer.frag  ══ once per layer, on every edit ══
   mask = clamp((curvature - noise * wear) * contrast, 0, 1)
   then choose between the two things under it
      │
      ├──▶ viewport preview, on the mesh
      └──▶ color.png + metallic.png + roughness.png
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

The bake is five stages, and each is keyed on its own parameters **plus the key
of the stage before it**:

```python
unwrap_key    = (mesh_token, unwrap_params.key(), resolution)
curvature_key = unwrap_key + bake_params.curvature_key()
occlusion_key = unwrap_key + (samples, distance)
post_key      = curvature_key + occlusion_key + (dilation,)
```

Occlusion hangs off the *unwrap* rather than off the curvature: it is a ray cast
against the geometry, and no curvature slider can change what a ray hits. So
nudging Strength re-rasterises and re-pads without spending the seconds that
re-casting would cost.

Cascading, so invalidation flows forward automatically and never backward:

| You change | Re-runs |
| --- | --- |
| Seam padding | `post` |
| Strength / Radius / Offset / Smooth / Axis | `curvature`, `post` |
| Bake resolution, atlas settings, or the mesh | everything |
| Any edge-wear setting on a layer | nothing — it is not a bake stage |

Occlusion is the slowest of them by a wide margin — seconds against fractions of
one, since it is the only stage that traces rays. It reports progress per ray
batch and honours Cancel between them, like every other CPU stage.

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

The viewport samples the compositor's own targets, and export reads those same
framebuffers back off the GPU rather than recomputing on the CPU. The PNG is
exactly what you tuned, by construction.

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

**Edge wear — the node group. Live, no re-bake.** On an edge-wear layer, in the
Texture tab.

| Parameter | Default | What it does |
| --- | --- | --- |
| Value | 1.0 | Feeds a ×10 Math node into the noise Scale, so the scale is `value * 10`. Higher = finer break-up. |
| Wear amount | 0.6 | How hard the noise erodes the curvature before the subtract. Higher = patchier. |
| Contrast | 3.0 | Final multiply, clamped to 0..1. |
| Detail / Roughness / Lacunarity / Distortion | 5.0 / 0.7 / 5.0 / 0.15 | The Wear Noise `TEX_NOISE` settings, as shipped. |

**If the wear comes out black:** the curvature bake behind it probably found
nothing. Raise Strength toward 1.5–2.0 and Radius toward 3–4. If it is *still*
black, your mesh has no bevels — see "What this method needs from your mesh".

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
  occlusion.py           the ray-cast ambient occlusion bake
  metadata.py            the title, icon and decal shelf, from metadata.json
  pipeline.py            stage keying, CPU/GL split, threading, cancel
  edge_wear.py           the EdgeWear001 graph and tex_noise, in numpy
  layers.py              the texture tree: colours, masks, and their materials
  export.py              PNG writers, 8- and 16-bit
  params.py              BakeParams / EdgeWearParams, split by cost
render/
  viewport.py            window, orbit/pan camera, the frame
  composite.py           the texture tree, one full-screen pass per layer
  imgui_renderer.py      ImGui 1.92 texture backend (see below)
  shaders/
    gbuffer.vert/.frag   UV-space rasterisation + the curvature formula
    layer.frag           one layer of the texture tree: mask, then choose
    decal.frag           one decal, as a slope added into the atlas
    normal_encode.frag   the accumulated slopes, back into a normal map
    edge_wear.frag       the node group as the reference port, not the path
    noise.glsl           ArmorPaint's tex_noise, shared by both
    copy.frag            the 95% round-trip blur
    preview.frag         viewport shading, Cook-Torrance
    bright/blur/tonemap  the glow around an emissive surface
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
