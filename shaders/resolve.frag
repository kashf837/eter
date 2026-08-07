#version 450
// ETER §6.2 Crack Filling — Pass 2 (resolve) fragment shader
// Valid pixels: lighting from G-buffer pos/norm (fast path).
// Crack pixels (rasterizer missed): SSBO Bézier evaluation on neighbor patch
// to fill the gap with pixel-accurate surface position and normal.

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D gbufPos;
layout(set = 0, binding = 1) uniform sampler2D gbufNorm;
layout(set = 0, binding = 2) uniform usampler2D gbufID;
layout(set = 0, binding = 3) uniform sampler2D gbufUV;

layout(set = 1, binding = 0) uniform Lighting {
    float intensity;
    float lightAz;
    float lightEl;
    float checkerboard;
    float boundary;
} lighting;

// Bézier control points: flat array, 16×vec4 per patch (xyz + w)
layout(std430, set = 2, binding = 0) buffer BezierData {
    vec4 bezierCP[];
};

// ============================================================
// Cubic Bernstein polynomials and derivatives
// ============================================================
void bernstein(float t, out float B[4])
{
    float t1 = 1.0 - t;
    B[0] = t1 * t1 * t1;
    B[1] = 3.0 * t * t1 * t1;
    B[2] = 3.0 * t * t * t1;
    B[3] = t * t * t;
}

void bernsteinDeriv(float t, out float dB[4])
{
    float t1 = 1.0 - t;
    dB[0] = -3.0 * t1 * t1;
    dB[1] =  3.0 * t1 * t1 - 6.0 * t * t1;
    dB[2] =  6.0 * t * t1 - 3.0 * t * t;
    dB[3] =  3.0 * t * t;
}

// Evaluate rational bicubic Bézier patch at (u,v).
// Returns surface position and normal via SSBO control points.
void evalBezier(uint patchID, vec2 uv, out vec3 pos, out vec3 norm)
{
    float Bu[4], Bv[4], dBu[4], dBv[4];
    bernstein(uv.x, Bu);
    bernstein(uv.y, Bv);
    bernsteinDeriv(uv.x, dBu);
    bernsteinDeriv(uv.y, dBv);

    uint base = patchID * 16u;
    vec4 Sw  = vec4(0.0);
    vec4 dSu = vec4(0.0);
    vec4 dSv = vec4(0.0);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            vec4 cp   = bezierCP[base + uint(i * 4 + j)];
            float wb  = Bu[i]  * Bv[j]  * cp.w;
            float wbu = dBu[i] * Bv[j]  * cp.w;
            float wbv = Bu[i]  * dBv[j] * cp.w;
            Sw.xyz  += cp.xyz * wb;   Sw.w  += wb;
            dSu.xyz += cp.xyz * wbu;  dSu.w += wbu;
            dSv.xyz += cp.xyz * wbv;  dSv.w += wbv;
        }
    }

    pos = Sw.xyz / Sw.w;

    // Rational surface derivatives (quotient rule)
    float invW2 = 1.0 / (Sw.w * Sw.w);
    vec3 du = (dSu.xyz * Sw.w - Sw.xyz * dSu.w) * invW2;
    vec3 dv = (dSv.xyz * Sw.w - Sw.xyz * dSv.w) * invW2;
    norm = normalize(cross(du, dv));
}

// ============================================================
// Lighting
// ============================================================
vec3 evalLighting(vec3 worldNorm, vec3 lightDir)
{
    float ambient  = 0.4;
    float diffuse  = max(dot(normalize(worldNorm), lightDir), 0.0);
    return vec3(0.85) * (ambient + diffuse * 0.6) * lighting.intensity;
}

float decodeChecker(uint pid)
{
    uint pi = pid >> 12;
    uint pj = pid & 0xFFFu;
    return (((pi + pj) & 1u) != 0u) ? 0.6 : 1.0;
}

// ============================================================
// main
// ============================================================
void main()
{
    ivec2 uv = ivec2(gl_FragCoord.xy);

    vec4 posSample = texelFetch(gbufPos, uv, 0);
    bool isFg = (posSample.a >= 0.5);

    vec3 lightDir = vec3(
        cos(lighting.lightEl) * sin(lighting.lightAz),
        sin(lighting.lightEl),
        cos(lighting.lightEl) * cos(lighting.lightAz)
    );

    // ============================================================
    // Fast path: valid pixel → G-buffer lighting directly
    // ============================================================
    if (isFg) {
        vec3 worldPos   = posSample.xyz;
        vec3 worldNorm  = texelFetch(gbufNorm, uv, 0).xyz;
        uint patchID    = texelFetch(gbufID,   uv, 0).r;

        vec3 colorOut = evalLighting(worldNorm, lightDir)
                      * mix(1.0, decodeChecker(patchID), lighting.checkerboard);

        // Boundary debug overlay (checks neighbors for different patchID)
        uint n0 = texelFetchOffset(gbufID, uv, 0, ivec2( 1,  0)).r;
        uint n1 = texelFetchOffset(gbufID, uv, 0, ivec2(-1,  0)).r;
        uint n2 = texelFetchOffset(gbufID, uv, 0, ivec2( 0,  1)).r;
        uint n3 = texelFetchOffset(gbufID, uv, 0, ivec2( 0, -1)).r;
        bool isBoundary = (n0 != patchID || n1 != patchID || n2 != patchID || n3 != patchID);

        outColor = (lighting.boundary > 0.5 && isBoundary)
            ? vec4(mix(colorOut, vec3(1.0, 0.0, 0.0), 0.6), 1.0)
            : vec4(colorOut, 1.0);
        return;
    }

    // ============================================================
    // Crack pixel: fill via SSBO Bézier evaluation on neighbor
    // ============================================================
    // Check 4 neighbors for a valid pixel to borrow patchID and (u,v) from
    vec4  np  = texelFetchOffset(gbufPos, uv, 0, ivec2( 1,  0));
    vec4  nn  = texelFetchOffset(gbufPos, uv, 0, ivec2(-1,  0));
    vec4  nt  = texelFetchOffset(gbufPos, uv, 0, ivec2( 0,  1));
    vec4  nb  = texelFetchOffset(gbufPos, uv, 0, ivec2( 0, -1));

    uint  selID  = 0u;
    vec2  selUV  = vec2(0.0);
    float bestDist = 1e10;

    // Right neighbor: map crack's position to left edge of neighbor patch
    if (np.a >= 0.5) {
        uint  nid = texelFetchOffset(gbufID, uv, 0, ivec2( 1,  0)).r;
        vec2  nuv = texelFetchOffset(gbufUV, uv, 0, ivec2( 1,  0)).rg;
        // Crack is to the LEFT of the neighbor → u ≈ 0 on neighbor
        float d = abs(nuv.x);
        if (d < bestDist) { bestDist = d; selID = nid; selUV = vec2(0.0, nuv.y); }
    }
    // Left neighbor
    if (nn.a >= 0.5) {
        uint  nid = texelFetchOffset(gbufID, uv, 0, ivec2(-1,  0)).r;
        vec2  nuv = texelFetchOffset(gbufUV, uv, 0, ivec2(-1,  0)).rg;
        float d = abs(1.0 - nuv.x);
        if (d < bestDist) { bestDist = d; selID = nid; selUV = vec2(1.0, nuv.y); }
    }
    // Top neighbor
    if (nt.a >= 0.5) {
        uint  nid = texelFetchOffset(gbufID, uv, 0, ivec2( 0,  1)).r;
        vec2  nuv = texelFetchOffset(gbufUV, uv, 0, ivec2( 0,  1)).rg;
        float d = abs(nuv.y);
        if (d < bestDist) { bestDist = d; selID = nid; selUV = vec2(nuv.x, 0.0); }
    }
    // Bottom neighbor
    if (nb.a >= 0.5) {
        uint  nid = texelFetchOffset(gbufID, uv, 0, ivec2( 0, -1)).r;
        vec2  nuv = texelFetchOffset(gbufUV, uv, 0, ivec2( 0, -1)).rg;
        float d = abs(1.0 - nuv.y);
        if (d < bestDist) { bestDist = d; selID = nid; selUV = vec2(nuv.x, 1.0); }
    }

    if (selID != 0u) {
        // Pixel-accurate surface evaluation via SSBO
        vec3 pos, norm;
        evalBezier(selID, selUV, pos, norm);
        vec3 col = evalLighting(norm, lightDir)
                 * mix(1.0, decodeChecker(selID), lighting.checkerboard);
        outColor = vec4(col, 1.0);
    } else {
        // No valid neighbor — true background
        outColor = vec4(0.1, 0.1, 0.15, 1.0);
    }
}
