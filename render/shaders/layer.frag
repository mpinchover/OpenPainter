#version 330

// One node of the mask tree: compute a mask, then choose between the two things
// under it. Run once per node, bottom up, so a node's inputs are either a flat
// colour or the texture a previous pass already left for it.
//
// Nothing here recurses. Depth lives in the number of passes
// (render/composite.py), which is what keeps an arbitrarily deep tree from
// needing an arbitrarily complicated shader -- and leaves every node with a
// real texture the panel can show as a thumbnail.

in vec2 v_uv;

// Two attachments: the colour, and the four surface channels that travel with
// it -- metallic, roughness, alpha, emission. The mask blends both the same
// way, so a material follows its colour wherever the tree puts it.
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_material;

uniform sampler2D u_curvature;  // the bake, for an edge wear node
uniform sampler2D u_position;   // bposition: object position over the bbox, 0..1
uniform sampler2D u_white;          // used only when u_whiteIsMap is 1
uniform sampler2D u_whiteMaterial;
uniform sampler2D u_black;
uniform sampler2D u_blackMaterial;

uniform int u_kind;             // 0 = edge wear, 1 = noise, 2 = flat colour
uniform float u_threshold;      // where the two sides divide, on the mask's 0..1
uniform float u_softness;       // how wide the crossing is; 0 is a clean split
uniform float u_rangeLow;       // the slice of the field stretched across 0..1
uniform float u_rangeHigh;      // before the split -- see MaskLayer.range_low
uniform int u_whiteIsMap;
uniform int u_blackIsMap;
uniform vec3 u_whiteColor;
uniform vec3 u_blackColor;
uniform vec4 u_whiteSurface;   // metallic, roughness, alpha, emission
uniform vec4 u_blackSurface;

// Edge wear, the EdgeWear001 group: see render/shaders/edge_wear.frag.
uniform float u_value;
uniform float u_wearAmount;
uniform float u_contrast;
uniform float u_detail;
uniform float u_roughness;
uniform float u_lacunarity;
uniform float u_distortion;

// A noise mask: the same field, thresholded into black and white.
uniform float u_noiseScale;
uniform float u_noiseDetail;
uniform float u_noiseRoughness;
uniform float u_noiseLacunarity;
uniform float u_noiseDistortion;
uniform float u_noiseBias;
uniform float u_noiseContrast;

#include "noise.glsl"

float mask_value(vec3 bposition) {
    if (u_kind == 2) {
        // A texture that is only a colour: no mask at all, so everything is
        // white's side and white is that colour.
        return 1.0;
    }
    if (u_kind == 1) {
        float n = tex_noise(bposition, u_noiseScale, u_noiseDetail,
                            u_noiseRoughness, u_noiseLacunarity, u_noiseDistortion);
        // Bias picks the level that lands on the midpoint, contrast decides how
        // hard the crossing is -- 0 leaves flat grey, high values a clean edge.
        return clamp((n - u_noiseBias) * u_noiseContrast + 0.5, 0.0, 1.0);
    }

    float curvature = texture(u_curvature, v_uv).r;
    float n = tex_noise(bposition, u_value * 10.0, u_detail,
                        u_roughness, u_lacunarity, u_distortion);
    return clamp((curvature - n * u_wearAmount) * u_contrast, 0.0, 1.0);
}

void main() {
    vec3 bposition = texture(u_position, v_uv).rgb;
    float mask = mask_value(bposition);

    // A mask is a continuous field, so mixing by it directly blends the two
    // sides everywhere it sits mid-way -- the colours bleed into each other
    // instead of dividing. Cutting it at the threshold is what makes the
    // boundary a boundary; softness feathers it back if that is wanted.
    // The part of the field actually in use, stretched across the whole range.
    // Few masks use all of theirs -- noise clusters around its middle -- and
    // the threshold then lives in a few pixels of slider travel.
    mask = clamp((mask - u_rangeLow) / max(u_rangeHigh - u_rangeLow, 1e-3), 0.0, 1.0);

    mask = u_softness <= 0.0
        ? step(u_threshold, mask)
        : smoothstep(u_threshold - u_softness, u_threshold + u_softness, mask);

    vec3 white = u_whiteIsMap == 1 ? texture(u_white, v_uv).rgb : u_whiteColor;
    vec3 black = u_blackIsMap == 1 ? texture(u_black, v_uv).rgb : u_blackColor;

    vec4 white_surface = u_whiteIsMap == 1
        ? texture(u_whiteMaterial, v_uv) : u_whiteSurface;
    vec4 black_surface = u_blackIsMap == 1
        ? texture(u_blackMaterial, v_uv) : u_blackSurface;

    out_color = vec4(mix(black, white, mask), 1.0);
    out_material = mix(black_surface, white_surface, mask);
}
