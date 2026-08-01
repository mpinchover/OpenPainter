#version 330

// The scene comes back out of its headroom, the glow is added on top, and the
// result is brought down to what a display can show.

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_hdrScale;
uniform float u_bloomStrength;

// Everything above 1 rolls off instead of clipping, and the whole colour is
// scaled together rather than channel by channel. Clipping per channel is what
// turns a bright red emission white: red pins first, then green, and the hue
// climbs to the top corner of the cube. This keeps the hue and lets the
// brightness saturate -- the surface stays its own colour, and the glow spilling
// past its edge is what says how bright it is.
vec3 rolloff(vec3 colour) {
    float peak = max(max(colour.r, colour.g), colour.b);
    return colour / (1.0 + max(peak - 1.0, 0.0));
}

void main() {
    vec3 scene = texture(u_scene, v_uv).rgb * u_hdrScale;
    vec3 glow = texture(u_bloom, v_uv).rgb * u_hdrScale * u_bloomStrength;
    out_color = vec4(rolloff(scene + glow), 1.0);
}
