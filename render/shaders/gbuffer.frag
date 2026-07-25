#version 330

// ArmorPaint's curvature bake, ported from the shader it generates in
// paint/sources/render/make_bake.c (BAKE_TYPE_CURVATURE):
//
//     var dx: float3 = ddx3(n);
//     var dy: float3 = ddy3(n);
//     var curvature: float = max(dot(dx, dx), dot(dy, dy));
//     curvature = clamp(pow(curvature, (1.0 / radius) * 0.25)
//                       * strength * 2.0 + offset / 10.0, 0.0, 1.0);
//     curvature *= dot(n, axis);        // only when a bake axis is chosen
//
// Because the vertex stage rasterises in UV space, dFdx/dFdy are the change in
// the interpolated normal between *neighbouring texels of the atlas* -- not
// screen space. Where the normal swings fast (a bevel) the derivative is large;
// across a flat face it is zero.

in vec3 v_position;
in vec3 v_normal;

// Attachment 0: world-space position, alpha doubles as the coverage mask.
// Attachment 1: world-space normal.
// Attachment 2: curvature, kept separate so the smoothing passes can blur it
//               without dragging the normals along.
layout(location = 0) out vec4 out_position;
layout(location = 1) out vec4 out_normal;
layout(location = 2) out vec4 out_curvature;

uniform float u_strength;
uniform float u_radius;
uniform float u_offset;
uniform vec3 u_axis;  // zero vector = no axis masking (ArmorPaint's "XYZ")

void main() {
    vec3 n = normalize(v_normal);

    vec3 dx = dFdx(n);
    vec3 dy = dFdy(n);
    float curvature = max(dot(dx, dx), dot(dy, dy));

    // radius is an exponent, not a distance: it gamma-lifts the derivative,
    // which is what turns a one-texel spike into a visible band. Guarded
    // against zero, which the slider allows and which would divide by 0.
    float exponent = (1.0 / max(u_radius, 1e-4)) * 0.25;
    curvature = clamp(pow(curvature, exponent) * u_strength * 2.0 + u_offset / 10.0, 0.0, 1.0);

    if (dot(u_axis, u_axis) > 0.0) {
        curvature *= dot(n, u_axis);
    }

    out_position = vec4(v_position, 1.0);
    out_normal = vec4(n, 1.0);
    out_curvature = vec4(curvature, curvature, curvature, 1.0);
}
