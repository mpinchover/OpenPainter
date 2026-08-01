#version 330

// One axis of a Gaussian blur, run twice: horizontal, then vertical. Separable,
// so a 9-tap kernel costs 18 samples instead of 81. The taps sit between texel
// centres so the hardware's bilinear filter pairs them up for free -- five
// samples covering nine texels, the standard trick.

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex;
uniform vec2 u_direction;  // one texel along the axis being blurred

const float OFFSETS[3] = float[](0.0, 1.3846153846, 3.2307692308);
const float WEIGHTS[3] = float[](0.2270270270, 0.3162162162, 0.0702702703);

void main() {
    vec3 total = texture(u_tex, v_uv).rgb * WEIGHTS[0];
    for (int i = 1; i < 3; ++i) {
        vec2 step = u_direction * OFFSETS[i];
        total += texture(u_tex, v_uv + step).rgb * WEIGHTS[i];
        total += texture(u_tex, v_uv - step).rgb * WEIGHTS[i];
    }
    out_color = vec4(total, 1.0);
}
