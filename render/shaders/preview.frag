#version 330

in vec3 v_normal;
in vec3 v_world;
in vec2 v_uv;

out vec4 out_color;

uniform sampler2D u_map;
uniform sampler2D u_normalMap;

// 0 = sample u_map as greyscale
// 1 = UV checker
// 2 = world normals
// 3 = plain shaded, no map
// 4 = sample u_map as colour
uniform int u_mode;
uniform float u_lighting;
uniform float u_checkerScale;
uniform vec3 u_lightDir;
uniform float u_useNormalMap;

// Per-pixel tangent frame, from the derivatives of position and UV across the
// triangle (Mikkelsen's cotangent frame). The mesh carries no tangent
// attribute, and deriving one here means a decal lights correctly on anything
// the app can load, however it was unwrapped.
vec3 apply_normal_map(vec3 normal, vec2 uv) {
    vec3 dp1 = dFdx(v_world);
    vec3 dp2 = dFdy(v_world);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);

    vec3 dp2perp = cross(dp2, normal);
    vec3 dp1perp = cross(normal, dp1);
    vec3 tangent = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 bitangent = dp2perp * duv1.y + dp1perp * duv2.y;

    float longest = max(dot(tangent, tangent), dot(bitangent, bitangent));
    if (longest <= 0.0) {
        return normal;  // degenerate UVs -- no frame to speak of
    }

    float invmax = inversesqrt(longest);
    mat3 tbn = mat3(tangent * invmax, bitangent * invmax, normal);
    vec3 tangent_normal = texture(u_normalMap, uv).rgb * 2.0 - 1.0;
    return normalize(tbn * tangent_normal);
}

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
    } else if (u_mode == 4) {
        base = texture(u_map, v_uv).rgb;
    } else {
        base = vec3(texture(u_map, v_uv).r);
    }

    vec3 normal = normalize(v_normal);
    if (!gl_FrontFacing) {
        normal = -normal;
    }
    if (u_useNormalMap > 0.5) {
        normal = apply_normal_map(normal, v_uv);
    }

    // Headlight plus a low fill, so form reads without washing out the map.
    float lambert = max(dot(normal, normalize(u_lightDir)), 0.0);
    float shading = 0.35 + 0.65 * lambert;

    out_color = vec4(base * mix(1.0, shading, u_lighting), 1.0);
}
