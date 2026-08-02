#version 330

in vec3 v_normal;
in vec3 v_world;
in vec2 v_uv;

out vec4 out_color;

uniform sampler2D u_map;
uniform sampler2D u_normalMap;
uniform sampler2D u_material;   // metallic, roughness, alpha, emission
uniform sampler2D u_liveDecal;
uniform float u_useLiveDecal;
uniform vec3 u_projectorCenter;
uniform vec3 u_projectorRight;
uniform vec3 u_projectorUp;
uniform vec3 u_projectorForward;
uniform vec2 u_projectorSize;
uniform float u_projectorIntensity;
uniform float u_projectorFlipGreen;

// 0 = sample u_map as greyscale
// 1 = UV checker
// 2 = world normals
// 3 = plain shaded, no map
// 4 = sample u_map as colour
uniform int u_mode;
uniform float u_lighting;
uniform float u_checkerScale;
uniform vec3 u_lightDir;
uniform vec3 u_eye;
uniform float u_useNormalMap;
uniform float u_useMaterial;
uniform float u_emissionScale;  // the material map stores emission as a fraction
uniform vec3 u_worldColor;      // the light everything sits in
uniform float u_worldStrength;
uniform float u_hdrScale;       // the headroom this target is divided down by

const float PI = 3.14159265359;

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

vec3 apply_live_decal(vec3 normal) {
    vec3 offset = v_world - u_projectorCenter;
    vec2 decal_uv = vec2(
        dot(offset, u_projectorRight) / u_projectorSize.x + 0.5,
        dot(offset, u_projectorUp) / u_projectorSize.y + 0.5
    );
    float depth = abs(dot(offset, u_projectorForward));
    float facing = dot(normal, u_projectorForward);
    if (any(lessThan(decal_uv, vec2(0.0))) || any(greaterThan(decal_uv, vec2(1.0)))
            || depth > max(u_projectorSize.x, u_projectorSize.y)
            || abs(facing) < 0.08) {
        return normal;
    }
    vec4 decal = texture(u_liveDecal, decal_uv);
    if (decal.a <= 0.001) {
        return normal;
    }
    vec3 encoded = decal.rgb * 2.0 - 1.0;
    if (u_projectorFlipGreen > 0.5) {
        encoded.y = -encoded.y;
    }
    vec2 slope = encoded.xy / max(encoded.z, 0.05) * u_projectorIntensity;
    vec3 tangent = u_projectorRight - normal * dot(u_projectorRight, normal);
    if (dot(tangent, tangent) < 1e-8) {
        return normal;
    }
    tangent = normalize(tangent);
    vec3 bitangent = normalize(cross(normal, tangent));
    vec3 projected = normalize(normal + tangent * slope.x + bitangent * slope.y);
    float edge = min(min(decal_uv.x, decal_uv.y), min(1.0 - decal_uv.x, 1.0 - decal_uv.y));
    float weight = decal.a * smoothstep(0.0, 0.025, edge);
    return normalize(mix(normal, projected, weight));
}

// -- Cook-Torrance, the same specular model Blender's Principled BSDF uses ---
//
// GGX for the distribution of microfacets, Smith for how they shadow each
// other, Schlick for Fresnel. Roughness is Blender's: squared into the GGX
// alpha, so the slider's middle looks like the slider's middle there too.

float distribution_ggx(float n_dot_h, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

float geometry_smith(float n_dot_v, float n_dot_l, float roughness) {
    float r = roughness + 1.0;
    float k = r * r / 8.0;
    float view = n_dot_v / (n_dot_v * (1.0 - k) + k);
    float light = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return view * light;
}

vec3 fresnel_schlick(float cosine, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cosine, 0.0, 1.0), 5.0);
}

// Fresnel for light arriving from the whole world rather than one direction.
// The plain Schlick form drives a rough surface's reflection to full strength
// at grazing angles, which a rough surface does not do; the roughness ceiling
// holds it back. Karis' approximation, as used for image-based lighting.
vec3 fresnel_schlick_roughness(float cosine, vec3 f0, float roughness) {
    vec3 ceiling = max(vec3(1.0 - roughness), f0);
    return f0 + (ceiling - f0) * pow(clamp(1.0 - cosine, 0.0, 1.0), 5.0);
}

// The world, as a two-colour gradient: brighter above, dimmer below. A single
// lamp leaves a metal with nothing to reflect and a rough surface with nothing
// to scatter, so both read as black however their sliders are set -- this is
// what gives them something to be made of.
vec3 world_light(vec3 direction) {
    float up = direction.z * 0.5 + 0.5;
    return u_worldColor * u_worldStrength * mix(0.25, 1.0, up);
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
    if (u_useLiveDecal > 0.5) {
        normal = apply_live_decal(normal);
    }

    float alpha = 1.0;
    if (u_lighting < 0.5) {
        // Unlit: the map itself, nothing added.
        out_color = vec4(base / u_hdrScale, alpha);
        return;
    }

    vec3 light = normalize(u_lightDir);
    vec3 view = normalize(u_eye - v_world);
    vec3 halfway = normalize(light + view);

    float n_dot_l = max(dot(normal, light), 0.0);
    float n_dot_v = max(dot(normal, view), 1e-4);
    float n_dot_h = max(dot(normal, halfway), 0.0);
    float v_dot_h = max(dot(view, halfway), 0.0);

    float metallic = 0.0;
    float roughness = 0.5;
    float emission = 0.0;
    if (u_useMaterial > 0.5) {
        vec4 surface = texture(u_material, v_uv);
        metallic = clamp(surface.r, 0.0, 1.0);
        roughness = clamp(surface.g, 0.03, 1.0);
        alpha = clamp(surface.b, 0.0, 1.0);
        emission = max(surface.a, 0.0) * u_emissionScale;
    }

    // A metal has no diffuse of its own: its colour is what it reflects.
    vec3 f0 = mix(vec3(0.04), base, metallic);
    vec3 fresnel = fresnel_schlick(v_dot_h, f0);
    vec3 diffuse_share = (1.0 - fresnel) * (1.0 - metallic);

    // The lamp has a size, the way every light in Blender does. A light of no
    // size puts a mirror's highlight inside a single pixel, so polishing a
    // surface reads as removing the highlight rather than sharpening it; the
    // floor is the smallest the highlight is allowed to get.
    float direct_roughness = max(roughness, 0.07);
    float distribution = distribution_ggx(n_dot_h, direct_roughness);
    float geometry = geometry_smith(n_dot_v, n_dot_l, direct_roughness);
    vec3 specular = distribution * geometry * fresnel
                  / max(4.0 * n_dot_v * n_dot_l, 1e-4);

    vec3 key = (diffuse_share * base + specular) * n_dot_l;

    // The world: light arriving from every direction at once. Diffuse takes
    // what the surface faces. The reflection looks along the mirror direction,
    // bent back towards the normal as roughness climbs -- a rough surface
    // gathers from everywhere rather than from one spot. Blurring the lookup
    // rather than dimming it is what keeps a rough metal a metal: it reflects
    // as much light as a polished one, just none of it in a straight line.
    vec3 reflected = reflect(-view, normal);
    vec3 gathered = normalize(mix(reflected, normal, roughness * roughness));
    vec3 env_fresnel = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    vec3 ambient = diffuse_share * base * world_light(normal)
                 + env_fresnel * world_light(gathered);

    // Divided down into the target's headroom, not clipped at 1. An emissive
    // surface brighter than white stays brighter than white all the way to the
    // bloom pass, which is what makes it read as a light rather than a colour.
    vec3 shaded = key + ambient + base * emission;
    out_color = vec4(min(shaded, vec3(u_hdrScale)) / u_hdrScale, alpha);
}
