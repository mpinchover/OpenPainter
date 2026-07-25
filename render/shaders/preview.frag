#version 330

in vec3 v_normal;
in vec2 v_uv;

out vec4 out_color;

uniform sampler2D u_map;

// 0 = sample u_map as greyscale
// 1 = UV checker
// 2 = world normals
// 3 = plain shaded, no map
uniform int u_mode;
uniform float u_lighting;
uniform float u_checkerScale;
uniform vec3 u_lightDir;

void main() {
    vec3 base;

    if (u_mode == 1) {
        vec2 cell = floor(v_uv * u_checkerScale);
        float check = mod(cell.x + cell.y, 2.0);
        base = mix(vec3(0.20), vec3(0.80), check);
    } else if (u_mode == 2) {
        base = normalize(v_normal) * 0.5 + 0.5;
    } else if (u_mode == 3) {
        base = vec3(0.72);
    } else {
        base = vec3(texture(u_map, v_uv).r);
    }

    vec3 normal = normalize(v_normal);
    if (!gl_FrontFacing) {
        normal = -normal;
    }

    // Headlight plus a low fill, so form reads without washing out the map.
    float lambert = max(dot(normal, normalize(u_lightDir)), 0.0);
    float shading = 0.35 + 0.65 * lambert;

    out_color = vec4(base * mix(1.0, shading, u_lighting), 1.0);
}
