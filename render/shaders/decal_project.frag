#version 330

in vec3 v_world;
in vec3 v_normal;
in vec2 v_atlas;
out vec4 out_color;

uniform sampler2D u_decal;
uniform vec3 u_projectorCenter;
uniform vec3 u_projectorRight;
uniform vec3 u_projectorUp;
uniform vec3 u_projectorForward;
uniform vec2 u_projectorSize;
uniform float u_intensity;
uniform float u_flipGreen;
uniform float u_falloff;

float fade(vec2 uv) {
    if (u_falloff <= 0.0) return 1.0;
    float edge = max(abs(uv.x - 0.5), abs(uv.y - 0.5)) * 2.0;
    return 1.0 - smoothstep(1.0 - 2.0 * u_falloff, 1.0 - u_falloff, edge);
}

void main() {
    vec3 offset = v_world - u_projectorCenter;
    vec2 decal_uv = vec2(dot(offset, u_projectorRight), dot(offset, u_projectorUp))
                    / max(u_projectorSize, vec2(1e-6)) + 0.5;
    vec3 surface_normal = normalize(v_normal);
    if (any(lessThan(decal_uv, vec2(0.0))) || any(greaterThan(decal_uv, vec2(1.0)))
            || abs(dot(offset, u_projectorForward)) > max(u_projectorSize.x, u_projectorSize.y)
            || abs(dot(surface_normal, u_projectorForward)) < 0.08) {
        out_color = vec4(0.0);
        return;
    }

    vec4 texel = texture(u_decal, decal_uv);
    vec3 encoded = texel.rgb * 2.0 - 1.0;
    if (u_flipGreen > 0.5) encoded.y = -encoded.y;
    vec2 decal_slope = encoded.xy / max(encoded.z, 0.05) * u_intensity;

    vec3 project_tangent = normalize(
        u_projectorRight - surface_normal * dot(u_projectorRight, surface_normal)
    );
    vec3 project_bitangent = normalize(cross(surface_normal, project_tangent));
    vec3 projected = normalize(
        surface_normal + project_tangent * decal_slope.x + project_bitangent * decal_slope.y
    );

    vec3 dp1 = dFdx(v_world);
    vec3 dp2 = dFdy(v_world);
    vec2 duv1 = dFdx(v_atlas);
    vec2 duv2 = dFdy(v_atlas);
    vec3 dp2perp = cross(dp2, surface_normal);
    vec3 dp1perp = cross(surface_normal, dp1);
    vec3 tangent = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 bitangent = dp2perp * duv1.y + dp1perp * duv2.y;
    float longest = max(dot(tangent, tangent), dot(bitangent, bitangent));
    if (longest < 1e-12) {
        out_color = vec4(0.0);
        return;
    }
    float invmax = inversesqrt(longest);
    tangent *= invmax;
    bitangent *= invmax;
    mat3 tbn = mat3(tangent, bitangent, surface_normal);
    vec3 tangent_normal = inverse(tbn) * projected;
    vec2 slope = tangent_normal.xy / max(tangent_normal.z, 0.05);
    slope *= texel.a * fade(decal_uv);
    out_color = vec4(slope, 0.0, 0.0);
}
