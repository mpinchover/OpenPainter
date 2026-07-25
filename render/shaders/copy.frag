#version 330

// ArmorPaint's curvature smoothing is a resolution round-trip rather than a
// kernel: render_path_paint.c copies the map into a target at 95% size and
// straight back, N times. Each round-trip resamples through the hardware's
// bilinear filter, which is a cheap, slightly anisotropic blur.
//
//     for (i32 i = 0; i < blurs; ++i) {
//         set_target("texpaint_blur"); bind(texpaint); draw(copy_pass);
//         set_target(texpaint); bind("texpaint_blur"); draw(copy_pass);
//     }

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex;

void main() {
    out_color = texture(u_tex, v_uv);
}
