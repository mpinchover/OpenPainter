#version 330

// Downsamples a bake target into the small RGBA8 texture the ImGui map
// inspector displays. Single-channel float textures show up as pure red
// through ImGui's shader, so the grayscale expansion happens here.

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex;
uniform int u_rgb;  // 1 for a colour map (the decal normals), 0 for a mask

void main() {
    vec4 texel = texture(u_tex, v_uv);
    out_color = vec4(u_rgb == 1 ? texel.rgb : vec3(texel.r), 1.0);
}
