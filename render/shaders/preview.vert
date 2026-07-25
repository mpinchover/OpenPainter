#version 330

uniform mat4 u_mvp;

in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;

out vec3 v_normal;
out vec2 v_uv;

void main() {
    v_normal = in_normal;
    v_uv = in_uv;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
