#version 330

in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;

out vec3 v_position;
out vec3 v_normal;

void main() {
    v_position = in_position;
    v_normal = in_normal;

    // The bake trick: feed the UV in as clip-space position so the rasteriser
    // walks the unwrapped surface instead of the projected 3D one. Every texel
    // it touches therefore corresponds to a point on the mesh -- and, crucially
    // for the curvature pass, dFdx/dFdy in the fragment shader become per-texel
    // derivatives across the atlas rather than screen-space ones.
    //
    // ArmorPaint does exactly this in make_paint.c:
    //     var tpos = float2(tex.x * 2 - 1, (1 - tex.y) * 2 - 1);
    //     output.pos = float4(tpos, 0, 1);
    gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
}
