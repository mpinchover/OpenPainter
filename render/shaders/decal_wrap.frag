#version 330

in vec2 v_surface_uv;
flat in mat2 v_slope_transform;
out vec4 out_color;

uniform sampler2D u_decal;
uniform vec2 u_center;
uniform vec2 u_size;
uniform float u_rotation;
uniform float u_falloff;
uniform float u_intensity;
uniform float u_flipGreen;

float fade(vec2 decal_uv) {
    if (u_falloff <= 0.0) return 1.0;
    vec2 centred = abs(decal_uv - 0.5) * 2.0;
    float edge = max(centred.x, centred.y);
    return 1.0 - smoothstep(1.0 - 2.0 * u_falloff, 1.0 - u_falloff, edge);
}

void main() {
    vec2 offset = v_surface_uv - u_center;
    float cosine = cos(u_rotation);
    float sine = sin(u_rotation);
    vec2 local = vec2(offset.x * cosine + offset.y * sine,
                      -offset.x * sine + offset.y * cosine);
    vec2 decal_uv = local / max(u_size, vec2(1e-6)) + 0.5;

    vec2 slope = vec2(0.0);
    if (all(greaterThanEqual(decal_uv, vec2(0.0)))
        && all(lessThanEqual(decal_uv, vec2(1.0)))) {
        vec4 texel = texture(u_decal, decal_uv);
        vec3 normal = texel.rgb * 2.0 - 1.0;
        normal.y = mix(normal.y, -normal.y, u_flipGreen);
        slope = normal.xy / max(normal.z, 1e-4)
                * (u_intensity * texel.a * fade(decal_uv));
        slope = vec2(slope.x * cosine - slope.y * sine,
                     slope.x * sine + slope.y * cosine);
        slope = v_slope_transform * slope;
    }
    out_color = vec4(slope, 0.0, 0.0);
}
