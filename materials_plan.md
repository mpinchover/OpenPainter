# Materials View Plan

The Material view should grow from a color-and-mask editor into a channel-based,
layered material system. The goal is to support convincing materials such as
painted metal, rust, rubber, glass, stone, worn plastic, and emissive panels.

## 1. Physically Based Material Channels

Each material should support:

- Base Color
- Roughness
- Metallic
- Normal
- Height
- Ambient Occlusion
- Emission
- Opacity

These channels form the foundation for physically based rendering and texture
export.

## 2. Layer Stack

Add a Photoshop- or Substance-style layer stack containing:

- Material layers
- Paint layers
- Fill layers
- Groups
- Masks
- Adjustable opacity
- Blend modes
- Drag-to-reorder support
- Per-channel enable switches

Example painted-metal stack:

```text
Scratches
  └─ Edge-wear mask
Red paint
  └─ Dirt mask
Bare steel
```

## 3. Procedural Generators

Provide reusable procedural sources such as:

- Noise
- Grunge
- Scratches
- Brushed metal
- Cells
- Clouds
- Directional streaks
- Gradients
- Brick and tile patterns
- Wood grain

Generators should expose scale, rotation, contrast, seed, distortion, and
detail controls.

## 4. Mesh-Aware Masks

Add generators that respond to the model's geometry:

- Curvature and edge wear
- Cavities
- Ambient occlusion
- Position gradients
- World-space normals
- Thickness
- Upward-facing surfaces
- Random values per mesh or UV island

These generators should remain reusable and editable instead of being
permanently baked into a single effect.

## 5. Mask Operations

Masks should support:

- Invert
- Levels
- Contrast
- Blur
- Sharpen
- Expand and shrink
- Multiply, add, subtract, and intersect
- Combining multiple generators

## 6. UV and Projection Controls

Every fill or procedural layer should offer:

- UV projection
- Triplanar projection
- Planar projection
- Scale, rotation, and offset
- Texture repetition
- Mirroring
- Per-axis triplanar controls

Triplanar projection is particularly important for reducing stretching and
visible UV seams.

## 7. Presets and Smart Materials

Allow complete materials to be saved and reused. Initial presets could include:

- Painted steel
- Rusted iron
- Scratched plastic
- Carbon fiber
- Worn leather
- Stone
- Emissive science-fiction panels

Smart materials should save their full layer stacks and mesh-aware masks so
they automatically adapt when applied to another model.

## 8. Material Preview Tools

Improve material evaluation with:

- A material sphere preview
- Individual channel previewing
- Before-and-after comparison
- Environment-map selection
- Adjustable preview lighting
- Tiling preview
- Real-time roughness and metallic response

## Recommended Implementation Order

1. Roughness and metallic channels
2. General layer stack
3. Per-layer masks
4. Procedural noise and grunge
5. Curvature, cavity, and position generators
6. Triplanar projection
7. Presets and smart materials
8. Height blending and advanced filters

The most transformative milestone is the combination of a channel-based layer
stack with procedural, mesh-aware masks. That combination moves MeshMap from
simple colored masks toward a Substance Painter-style material workflow.
