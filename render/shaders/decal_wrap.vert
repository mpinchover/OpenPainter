#version 330

in vec2 in_atlas_uv;
in vec2 in_surface_uv;
in vec4 in_slope_transform;

out vec2 v_surface_uv;
flat out mat2 v_slope_transform;

void main() {
    v_surface_uv = in_surface_uv;
    v_slope_transform = mat2(
        in_slope_transform.x, in_slope_transform.z,
        in_slope_transform.y, in_slope_transform.w
    );
    gl_Position = vec4(in_atlas_uv * 2.0 - 1.0, 0.0, 1.0);
}
