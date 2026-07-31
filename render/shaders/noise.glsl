// ArmorPaint's noise, verbatim from nodes_material/noise_texture_node.c.
// Shared by the wear pass and the mask tree, so both sample the same field.
// Included through the #include support in render/shaders/__init__.py.

float hash(float n) { return fract(sin(n) * 10000.0); }

float tex_noise_f(vec3 x) {
    vec3 step = vec3(110.0, 241.0, 171.0);
    vec3 i = floor(x);
    vec3 f = fract(x);
    float n = dot(i, step);
    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash(n + dot(step, vec3(0.0, 0.0, 0.0))), hash(n + dot(step, vec3(1.0, 0.0, 0.0))), u.x),
            mix(hash(n + dot(step, vec3(0.0, 1.0, 0.0))), hash(n + dot(step, vec3(1.0, 1.0, 0.0))), u.x), u.y),
        mix(mix(hash(n + dot(step, vec3(0.0, 0.0, 1.0))), hash(n + dot(step, vec3(1.0, 0.0, 1.0))), u.x),
            mix(hash(n + dot(step, vec3(0.0, 1.0, 1.0))), hash(n + dot(step, vec3(1.0, 1.0, 1.0))), u.x), u.y),
        u.z);
}

float tex_noise_fbm(vec3 p, float detail, float roughness, float lacunarity) {
    float fscale = 1.0;
    float amp = 1.0;
    float maxamp = 0.0;
    float sum = 0.0;
    int n = int(clamp(detail, 0.0, 8.0));
    for (int ii = 0; ii <= n; ii += 1) {
        sum = sum + amp * tex_noise_f(p * fscale);
        maxamp = maxamp + amp;
        amp = amp * roughness;
        fscale = fscale * lacunarity;
    }
    float rmd = detail - floor(detail);
    if (rmd > 0.0) {
        float t = tex_noise_f(p * fscale);
        float sum2 = sum + t * amp;
        float maxamp2 = maxamp + amp;
        return mix(sum / maxamp, sum2 / maxamp2, rmd);
    }
    return sum / maxamp;
}

float tex_noise(vec3 p, float scale, float detail, float roughness,
                float lacunarity, float distortion) {
    vec3 pp = p * scale;
    pp = pp + distortion * (vec3(tex_noise_f(pp),
                                 tex_noise_f(pp + vec3(0.5, 0.0, 0.0)),
                                 tex_noise_f(pp + vec3(0.0, 0.5, 0.0))) * 2.0 - 1.0);
    // Only the .x component is used: the node's Factor output is result.x.
    return tex_noise_fbm(pp, detail, roughness, lacunarity);
}
