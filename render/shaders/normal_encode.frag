#version 330

// The last step of the decal composite: the accumulated surface gradient,
// turned back into a tangent-space normal map.
//
// A slope of (a, b) is the normal (a, b, 1) once renormalised, so the sum of
// every decal's gradient rebuilds into the surface they describe between them.
// Where nothing was stamped the slope is zero, which encodes to (0.5, 0.5, 1.0)
// -- a normal pointing straight out, which a Normal Map node treats as no
// change at all.

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_slope;

void main() {
    vec2 slope = texture(u_slope, v_uv).rg;
    out_color = vec4(normalize(vec3(slope, 1.0)) * 0.5 + 0.5, 1.0);
}
