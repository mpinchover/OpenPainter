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

// Colour, packed surface values, and material-authored ambient occlusion.
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_material;
layout(location = 2) out float out_ao;

uniform sampler2D u_curvature;  // the bake, for an edge wear node
uniform sampler2D u_position;   // bposition: object position over the bbox, 0..1
uniform sampler2D u_white;          // used only when u_whiteIsMap is 1
uniform sampler2D u_whiteMaterial;
uniform sampler2D u_whiteAO;
uniform sampler2D u_black;
uniform sampler2D u_blackMaterial;
uniform sampler2D u_blackAO;

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
uniform float u_whiteAOValue;
uniform float u_blackAOValue;

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
uniform float u_generatorRotation;
uniform float u_generatorSeed;
uniform float u_scratchWidth;
uniform float u_scratchLength;
uniform float u_scratchIrregularity;
uniform float u_brushDensity;
uniform float u_brushWaviness;
uniform float u_brushVariation;
uniform float u_cellJitter;
uniform float u_cellEdge;
uniform float u_streakLength;
uniform float u_streakWidth;
uniform float u_mortarThickness;
uniform float u_brickAspect;
uniform float u_veinWidth;

#include "noise.glsl"

vec2 rotate_generator(vec2 p) {
    float angle = radians(u_generatorRotation);
    mat2 turn = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
    return turn * (p - 0.5) + 0.5;
}

vec3 generator_position(vec3 p) {
    p.xy = rotate_generator(p.xy);
    return p + vec3(u_generatorSeed * 7.13, u_generatorSeed * 3.71, u_generatorSeed * 5.37);
}

vec3 face_projected_position(vec3 p) {
    // Position is stored in the model's normalised bounding box. Its UV-space
    // derivatives recover the face direction, letting directional 2D patterns
    // choose the matching plane instead of collapsing on faces perpendicular
    // to XY. A dominant-axis choice keeps hard-surface corners crisp.
    vec3 normal = abs(normalize(cross(dFdx(p), dFdy(p))));
    vec3 projected;
    if (normal.x >= normal.y && normal.x >= normal.z) {
        projected = vec3(p.yz, p.x);
    } else if (normal.y >= normal.z) {
        projected = vec3(p.xz, p.y);
    } else {
        projected = vec3(p.xy, p.z);
    }
    projected.xy = rotate_generator(projected.xy);
    return projected + vec3(
        u_generatorSeed * 7.13, u_generatorSeed * 3.71, u_generatorSeed * 5.37
    );
}

float cells(vec3 p) {
    vec3 cell = floor(p);
    vec3 local = fract(p);
    float nearest = 2.0;
    for (int z = -1; z <= 1; ++z) for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x) {
        vec3 offset = vec3(x, y, z);
        float h = hash(dot(cell + offset, vec3(17.0, 59.4, 15.0)));
        vec3 random_point = vec3(h, hash(h * 31.7), hash(h * 91.3));
        vec3 point = offset + mix(vec3(0.5), random_point, u_cellJitter);
        nearest = min(nearest, length(point - local));
    }
    return clamp(nearest, 0.0, 1.0);
}

float procedural_value(vec3 position) {
    vec3 p = generator_position(position);
    if (u_kind == 4 || u_kind == 5 || u_kind == 8 || u_kind == 10) {
        p = face_projected_position(position);
    }
    float scale = max(u_noiseScale, 0.001);
    float value;
    if (u_kind == 1) { // Noise
        value = tex_noise(p, scale, u_noiseDetail, u_noiseRoughness,
                          u_noiseLacunarity, u_noiseDistortion);
    } else if (u_kind == 3) { // Grunge
        float broad = tex_noise(p, scale * 0.35, u_noiseDetail,
                                u_noiseRoughness, u_noiseLacunarity, u_noiseDistortion);
        float fine = tex_noise(p + vec3(4.2), scale * 2.2, 2.0, 0.6, 2.3, 0.4);
        value = clamp(broad * 1.25 - fine * 0.45 + 0.15, 0.0, 1.0);
    } else if (u_kind == 4) { // Scratches
        vec2 q = p.xy * scale;
        float lane = floor(q.x);
        float offset = hash(lane + floor(p.z * 17.0)) * u_scratchIrregularity;
        float line = abs(fract(q.x + offset) - 0.5);
        float broken = tex_noise_f(vec3(
            q.y * mix(0.08, 0.8, u_scratchIrregularity), lane, p.z * 7.0
        ));
        float width = max(u_scratchWidth, 0.002);
        float line_mask = 1.0 - smoothstep(width * 0.45, width, line);
        float length_mask = smoothstep(1.0 - u_scratchLength, 1.0, broken);
        value = line_mask * length_mask;
    } else if (u_kind == 5) { // Brushed metal
        vec2 q = p.xy * scale;
        float warp = tex_noise_f(vec3(q.y * 0.07, p.z * 2.0, 0.0))
                   * u_brushWaviness * 8.0;
        float bands = sin(q.x * u_brushDensity + warp) * 0.5 + 0.5;
        float variation = tex_noise_f(vec3(q.x * 2.0, q.y * 0.035, p.z));
        value = mix(bands, variation, u_brushVariation);
    } else if (u_kind == 6) { // Cells
        value = smoothstep(u_cellEdge, min(u_cellEdge + 0.18, 1.0),
                           1.0 - cells(p * scale));
    } else if (u_kind == 7) { // Clouds
        value = smoothstep(0.22, 0.78, tex_noise(p, scale * 0.35,
                           max(u_noiseDetail, 5.0), u_noiseRoughness,
                           u_noiseLacunarity, u_noiseDistortion));
    } else if (u_kind == 8) { // Directional streaks
        vec3 q = vec3(
            p.x * scale * max(u_streakWidth, 0.01),
            p.y * scale * max(u_streakLength, 0.1),
            p.z * scale * max(u_streakWidth, 0.01)
        );
        value = tex_noise(
            q, 1.0, u_noiseDetail, u_noiseRoughness,
            u_noiseLacunarity, u_noiseDistortion
        );
    } else if (u_kind == 9) { // Gradient
        value = clamp((p.x - 0.5) * max(scale * 0.1, 1.0) + 0.5, 0.0, 1.0);
    } else if (u_kind == 10) { // Brick / tile
        vec2 q = p.xy * scale;
        q.x /= max(u_brickAspect, 0.1);
        q.x += mod(floor(q.y), 2.0) * 0.5;
        vec2 edge = min(fract(q), 1.0 - fract(q));
        float mortar = clamp(u_mortarThickness, 0.001, 0.45);
        value = smoothstep(mortar, min(mortar + 0.025, 0.49), min(edge.x, edge.y));
    } else if (u_kind == 11) { // Wood grain
        vec2 q = (p.xy - 0.5) * scale;
        float wobble = tex_noise(
            p, scale * 0.35, u_noiseDetail, 0.6, 2.0, u_noiseDistortion
        );
        float ring = abs(sin((length(q) + wobble * 1.8) * 10.0));
        value = smoothstep(1.0 - u_veinWidth, 1.0, ring);
    } else { // Marble: continuous 3D veins, independent of UV seams.
        float turbulence = tex_noise(
            p, scale * 0.45, max(u_noiseDetail, 4.0), 0.58, 2.1,
            max(u_noiseDistortion, 0.35)
        );
        float vein = (p.x + p.y * 0.32 + p.z * 0.18) * scale * 3.5;
        float wave = abs(sin(vein + turbulence * 9.0));
        value = smoothstep(1.0 - u_veinWidth, 1.0, wave);
    }
    return clamp((value - u_noiseBias) * u_noiseContrast + 0.5, 0.0, 1.0);
}

float mask_value(vec3 bposition) {
    if (u_kind == 2) {
        // A texture that is only a colour: no mask at all, so everything is
        // white's side and white is that colour.
        return 1.0;
    }
    if (u_kind >= 1) {
        return procedural_value(bposition);
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
    float white_ao = u_whiteIsMap == 1 ? texture(u_whiteAO, v_uv).r : u_whiteAOValue;
    float black_ao = u_blackIsMap == 1 ? texture(u_blackAO, v_uv).r : u_blackAOValue;

    out_color = vec4(mix(black, white, mask), 1.0);
    out_material = mix(black_surface, white_surface, mask);
    out_ao = mix(black_ao, white_ao, mask);
}
