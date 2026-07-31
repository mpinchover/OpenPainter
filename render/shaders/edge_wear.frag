#version 330

// The EdgeWear001 node group, decoded from ArmorPaint's
// cloud/materials/Procedural/EdgeWear001.arm and ported node for node.
//
//   Group Input.Strength ─┐
//   Group Input.Radius  ──┤
//                         ▼
//                 [Curvature Texture]      BAKE_CURVATURE, Offset -2.0
//                         │                (baked; sampled here)
//   Group Input.Value ─► [Math x10] ─► [Wear Noise]  TEX_NOISE
//                                          │ Factor
//                                          ▼
//                                   [Wear Amount x0.6]
//                                          │
//                 [Curvature] ─► [Break Up: A - B]
//                                          │
//                                   [Contrast x3, clamped]
//                                          │
//                                   Group Output.Mask
//
// which reduces to:  mask = clamp((curvature - noise * 0.6) * 3, 0, 1)

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_curvature;
uniform sampler2D u_position;  // bposition: object position over the bbox, 0..1

uniform float u_value;         // Group Input.Value; noise scale = value * 10
uniform float u_wearAmount;    // "Wear Amount" multiply, 0.6
uniform float u_contrast;      // "Contrast" multiply, 3.0
uniform float u_detail;        // TEX_NOISE Detail, 5.0
uniform float u_roughness;     // TEX_NOISE Roughness, 0.7
uniform float u_lacunarity;    // TEX_NOISE Lacunarity, 5.0
uniform float u_distortion;    // TEX_NOISE Distortion, 0.15
uniform int u_showCurvature;   // 1 = output the raw curvature bake instead

#include "noise.glsl"

// ---------------------------------------------------------------------------

void main() {
    float curvature = texture(u_curvature, v_uv).r;

    if (u_showCurvature == 1) {
        out_color = vec4(curvature, curvature, curvature, 1.0);
        return;
    }

    vec3 bposition = texture(u_position, v_uv).rgb;
    float noise = tex_noise(bposition, u_value * 10.0, u_detail,
                            u_roughness, u_lacunarity, u_distortion);

    float wear = noise * u_wearAmount;
    float mask = clamp((curvature - wear) * u_contrast, 0.0, 1.0);

    out_color = vec4(mask, mask, mask, 1.0);
}
