#version 330

// Downsamples a bake target into the small RGBA8 texture the ImGui map
// inspector displays. Single-channel float textures show up as pure red
// through ImGui's shader, so the grayscale expansion happens here.

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex;

void main() {
    out_color = vec4(vec3(texture(u_tex, v_uv).r), 1.0);
}
