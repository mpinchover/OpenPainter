#version 330

// The first half of the glow around an emissive surface: keep only what is
// brighter than white, throw the rest away. Everything the lamp and the world
// produce sits at or below 1, so what survives here is emission and nothing
// else -- the glow belongs to the material, not to any bright highlight.
//
// Sampled at quarter resolution with a four-tap box, so this doubles as the
// downsample the blur runs on.

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_scene;
uniform float u_hdrScale;   // the headroom the scene target is divided by
uniform vec2 u_texel;       // one source pixel, in UV

void main() {
    vec3 total = texture(u_scene, v_uv + u_texel * vec2(-0.5, -0.5)).rgb
               + texture(u_scene, v_uv + u_texel * vec2( 0.5, -0.5)).rgb
               + texture(u_scene, v_uv + u_texel * vec2(-0.5,  0.5)).rgb
               + texture(u_scene, v_uv + u_texel * vec2( 0.5,  0.5)).rgb;
    vec3 scene = total * 0.25 * u_hdrScale;

    // Subtract the threshold rather than masking around it: a surface that
    // creeps past 1 starts to glow faintly instead of switching on.
    vec3 excess = max(scene - vec3(1.0), vec3(0.0));
    out_color = vec4(excess / u_hdrScale, 1.0);
}
