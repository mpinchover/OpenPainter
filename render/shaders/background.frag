#version 330

in vec2 v_uv;
out vec4 out_color;

uniform vec3 u_top;
uniform vec3 u_bottom;

void main() {
    out_color = vec4(mix(u_bottom, u_top, v_uv.y), 1.0);
}
