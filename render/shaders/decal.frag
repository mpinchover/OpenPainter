#version 330

// Stamps a decal into an atlas-wide tangent-space normal map. One full-screen
// pass over the atlas, so every knob in the Decal tab stays live.
//
// Mirrored on the CPU by core/decal.py's composite_normal_map(), which is what
// the tests check this against.
//
// Outside the decal's rectangle the output is (0.5, 0.5, 1.0): a normal
// pointing straight out, which a Normal Map node treats as no change at all.

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_decal;

uniform vec2 u_center;     // where the middle of the decal sits, in UV
uniform vec2 u_size;       // its width and height, in UV units
uniform float u_rotation;  // radians, counter-clockwise in UV space
uniform float u_intensity; // multiplies the surface slope, not the vector
uniform float u_flipGreen; // 1 = the map was baked DirectX-style (-Y)

void main() {
    vec2 offset = v_uv - u_center;

    // The decal turns one way, so the lookup into it turns the other.
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

        // xy/z is the slope of the surface the normal describes. Scaling that
        // and rebuilding is what makes intensity 2.0 twice as steep; scaling
        // the vector instead would tip the normal flat and then past flat.
        slope = normal.xy / max(normal.z, 1e-4) * (u_intensity * texel.a);

        // Express the decal's own slope in the atlas's frame.
        slope = vec2(slope.x * cosine - slope.y * sine,
                     slope.x * sine + slope.y * cosine);
    }

    vec3 result = normalize(vec3(slope, 1.0));
    out_color = vec4(result * 0.5 + 0.5, 1.0);
}
