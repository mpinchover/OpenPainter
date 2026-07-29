#version 330

uniform mat4 u_mvp;

in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;

out vec3 v_normal;
out vec3 v_world;
out vec2 v_uv;

void main() {
    v_normal = in_normal;
    // The fragment stage differentiates this to build a tangent frame for the
    // normal map, so it has to be the position the UVs actually belong to.
    v_world = in_position;
    v_uv = in_uv;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
