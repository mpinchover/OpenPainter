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

// -- ArmorPaint's noise, verbatim from nodes_material/noise_texture_node.c ---

float hash(float n) { return fract(sin(n) * 10000.0); }

float tex_noise_f(vec3 x) {
    vec3 step = vec3(110.0, 241.0, 171.0);
    vec3 i = floor(x);
    vec3 f = fract(x);
    float n = dot(i, step);
    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash(n + dot(step, vec3(0.0, 0.0, 0.0))), hash(n + dot(step, vec3(1.0, 0.0, 0.0))), u.x),
            mix(hash(n + dot(step, vec3(0.0, 1.0, 0.0))), hash(n + dot(step, vec3(1.0, 1.0, 0.0))), u.x), u.y),
        mix(mix(hash(n + dot(step, vec3(0.0, 0.0, 1.0))), hash(n + dot(step, vec3(1.0, 0.0, 1.0))), u.x),
            mix(hash(n + dot(step, vec3(0.0, 1.0, 1.0))), hash(n + dot(step, vec3(1.0, 1.0, 1.0))), u.x), u.y),
        u.z);
}

float tex_noise_fbm(vec3 p, float detail, float roughness, float lacunarity) {
    float fscale = 1.0;
    float amp = 1.0;
    float maxamp = 0.0;
    float sum = 0.0;
    int n = int(clamp(detail, 0.0, 8.0));
    for (int ii = 0; ii <= n; ii += 1) {
        sum = sum + amp * tex_noise_f(p * fscale);
        maxamp = maxamp + amp;
        amp = amp * roughness;
        fscale = fscale * lacunarity;
    }
    float rmd = detail - floor(detail);
    if (rmd > 0.0) {
        float t = tex_noise_f(p * fscale);
        float sum2 = sum + t * amp;
        float maxamp2 = maxamp + amp;
        return mix(sum / maxamp, sum2 / maxamp2, rmd);
    }
    return sum / maxamp;
}

float tex_noise(vec3 p, float scale, float detail, float roughness,
                float lacunarity, float distortion) {
    vec3 pp = p * scale;
    pp = pp + distortion * (vec3(tex_noise_f(pp),
                                 tex_noise_f(pp + vec3(0.5, 0.0, 0.0)),
                                 tex_noise_f(pp + vec3(0.0, 0.5, 0.0))) * 2.0 - 1.0);
    // Only the .x component is used: the node's Factor output is result.x.
    return tex_noise_fbm(pp, detail, roughness, lacunarity);
}

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
