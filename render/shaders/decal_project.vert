#version 330

in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;

out vec3 v_world;
out vec3 v_normal;
out vec2 v_atlas;

void main() {
    v_world = in_position;
    v_normal = in_normal;
    v_atlas = in_uv;
    gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
}
