#version 330

out vec2 v_uv;

void main() {
    // Full-screen triangle synthesised from gl_VertexID -- no vertex buffer,
    // no attribute bindings, three vertices covering the whole target.
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    v_uv = corner;
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
