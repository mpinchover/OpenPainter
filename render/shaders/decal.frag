#version 330

// Stamps one decal into the atlas's surface gradient. One full-screen pass per
// decal, so every knob in the Decal tab stays live.
//
// What comes out is a *slope*, not a normal, and the pass runs with additive
// blending: several decals on one surface accumulate. A vent stamped across a
// panel line should read as both, and adding the two surfaces' gradients is
// what "both" means -- adding encoded normals would average them towards flat,
// and the second decal would rub out the first as much as it added itself.
// render/shaders/normal_encode.frag turns the total back into a normal once.
//
// Outside the decal's rectangle the output is zero, which adds nothing.
//
// Mirrored on the CPU by core/decal.py's composite(), which is what the tests
// check this against.

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_decal;

uniform vec2 u_center;     // where the middle of the decal sits, in UV
uniform vec2 u_size;       // its width and height, in UV units
uniform float u_rotation;  // radians, counter-clockwise in UV space
uniform float u_falloff;   // how much of the edge fades out, 0..1
uniform float u_intensity; // multiplies the surface slope, not the vector
uniform float u_flipGreen; // 1 = the map was baked DirectX-style (-Y)

// How much of the decal survives at this point. Cut, then feathered: the outer
// u_falloff of the way out is gone entirely, with the fade spread over the same
// width again just inside it, so the middle is untouched. Measured along
// whichever axis is nearer its edge, so all four sides lose the same width
// rather than the corners going first.
float fade(vec2 decal_uv) {
    if (u_falloff <= 0.0) {
        return 1.0;
    }
    vec2 centred = abs(decal_uv - 0.5) * 2.0;
    float edge = max(centred.x, centred.y);
    return 1.0 - smoothstep(1.0 - 2.0 * u_falloff, 1.0 - u_falloff, edge);
}

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
        slope = normal.xy / max(normal.z, 1e-4) * (u_intensity * texel.a * fade(decal_uv));

        // Express the decal's own slope in the atlas's frame.
        slope = vec2(slope.x * cosine - slope.y * sine,
                     slope.x * sine + slope.y * cosine);
    }

    out_color = vec4(slope, 0.0, 0.0);
}
