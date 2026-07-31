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
out vec4 out_color;

uniform sampler2D u_curvature;  // the bake, for an edge wear node
uniform sampler2D u_position;   // bposition: object position over the bbox, 0..1
uniform sampler2D u_white;      // used only when u_whiteIsMap is 1
uniform sampler2D u_black;

uniform int u_kind;             // 0 = edge wear, 1 = noise, 2 = flat colour
uniform int u_whiteIsMap;
uniform int u_blackIsMap;
uniform vec3 u_whiteColor;
uniform vec3 u_blackColor;

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

    vec3 white = u_whiteIsMap == 1 ? texture(u_white, v_uv).rgb : u_whiteColor;
    vec3 black = u_blackIsMap == 1 ? texture(u_black, v_uv).rgb : u_blackColor;

    out_color = vec4(mix(black, white, mask), 1.0);
}
