#version 450
// ETER §6.2 Crack Filling — Pass 2 (resolve) fragment shader
//
// All valid pixels:  SSBO Bézier evaluation (pure, pixel-accurate normals).
//                    CPs are centered per-patch → FP32-safe at any scale.
//                    Consistent across all patch boundaries → no seams.
// Crack pixels:      SSBO Bézier from nearest valid neighbor.
//
// Checkerboard:      encoded as sign bit of patchID float in vertex buffer,
//                    stored in gbufNorm.w by visibility.frag.
//                    patchID in gbufID (uint) = abs(original) = clean seq ID.

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D gbufPos;
layout(set = 0, binding = 1) uniform sampler2D gbufNorm;
layout(set = 0, binding = 2) uniform usampler2D gbufID;
layout(set = 0, binding = 3) uniform sampler2D gbufUV;

layout(std140, set = 1, binding = 0) uniform Lighting {
    float intensity;
    float lightDirX;
    float lightDirY;
    float lightDirZ;
    float checkerboard;
    float boundary;
    float patchCols;   // unused (kept for UBO compat)
    float _pad;
} lighting;

layout(std430, set = 2, binding = 0) buffer BezierData {
    vec4 bezierCP[];
};

// ============================================================
// Cubic Bernstein polynomials
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

void evalBezier(uint patchID, vec2 uv, out vec3 pos, out vec3 norm)
{
    float u = clamp(uv.x, 0.0, 1.0);
    float v = clamp(uv.y, 0.0, 1.0);

    float Bu[4], Bv[4], dBu[4], dBv[4];
    bernstein(u, Bu);
    bernstein(v, Bv);
    bernsteinDeriv(u, dBu);
    bernsteinDeriv(v, dBv);

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

    float invW2 = 1.0 / (Sw.w * Sw.w);
    vec3 du = (dSu.xyz * Sw.w - Sw.xyz * dSu.w) * invW2;
    vec3 dv = (dSv.xyz * Sw.w - Sw.xyz * dSv.w) * invW2;
    norm = normalize(cross(du, dv));
}

// ============================================================
// Lighting
// ============================================================
vec3 shadeNormal(vec3 N, vec3 L, float checker)
{
    float ambient = 0.4;
    float NdotL = dot(normalize(N), L);
    float diffuse = abs(NdotL);  // view-independent: front and back equally lit
    return vec3(0.85) * (ambient + diffuse * 0.6 * checker) * lighting.intensity;
}

// detail: tessellation density [0,1] from |gbufNorm.w|
// isEven: sign of gbufNorm.w → checkerboard parity (encoded via sign bit of patchID)
// Binary checker modulated by detail brightness:
//   high tess = brighter, low tess = dimmer, but checker pattern always visible
float checkerFromParity(bool isEven)
{
    return isEven ? 1.15 : 0.85;  // checkerboard, average ≈ 1.0
}

vec3 shadeSSBO(uint patchID, vec2 uv, vec3 L, bool isEven)
{
    vec3 pos, norm;
    evalBezier(patchID, uv, pos, norm);
    float checker = mix(1.0, checkerFromParity(isEven), lighting.checkerboard);
    return shadeNormal(norm, L, checker);
}

// ============================================================
// main
// ============================================================
void main()
{
    ivec2 coord = ivec2(gl_FragCoord.xy);

    vec4 posSample = texelFetch(gbufPos, coord, 0);
    bool isFg = (posSample.a >= 0.5);

    vec3 lightDir = normalize(vec3(lighting.lightDirX, lighting.lightDirY, lighting.lightDirZ));

    // ============================================================
    // Valid pixel
    // ============================================================
    if (isFg) {
        uint  patchID  = texelFetch(gbufID, coord, 0).r;
        vec2  paramUV  = texelFetch(gbufUV, coord, 0).rg;
        float rawW     = texelFetch(gbufNorm, coord, 0).w;
        bool  isEven   = rawW >= 0.0;     // sign bit → checkerboard parity

        // Detect patch boundary
        uint n0 = texelFetchOffset(gbufID, coord, 0, ivec2( 1,  0)).r;
        uint n1 = texelFetchOffset(gbufID, coord, 0, ivec2(-1,  0)).r;
        uint n2 = texelFetchOffset(gbufID, coord, 0, ivec2( 0,  1)).r;
        uint n3 = texelFetchOffset(gbufID, coord, 0, ivec2( 0, -1)).r;
        bool isBoundary = (n0 != patchID || n1 != patchID || n2 != patchID || n3 != patchID);

        // SSBO Bézier for all pixels → pixel-accurate, boundary-consistent normals.
        vec3 colorOut = shadeSSBO(patchID, paramUV, lightDir, isEven);

        // Boundary debug overlay
        outColor = (lighting.boundary > 0.5 && isBoundary)
            ? vec4(mix(colorOut, vec3(1.0, 0.0, 0.0), 0.6), 1.0)
            : vec4(colorOut, 1.0);
        return;
    }

    // ============================================================
    // Crack pixel: fill via SSBO Bézier from nearest valid neighbor
    // ============================================================
    vec4 np = texelFetchOffset(gbufPos, coord, 0, ivec2( 1,  0));
    vec4 nn = texelFetchOffset(gbufPos, coord, 0, ivec2(-1,  0));
    vec4 nt = texelFetchOffset(gbufPos, coord, 0, ivec2( 0,  1));
    vec4 nb = texelFetchOffset(gbufPos, coord, 0, ivec2( 0, -1));

    uint  selID     = 0u;
    vec2  selUV     = vec2(0.0);
    float selRawW   = 0.0;

    if (np.a >= 0.5) {
        selID = texelFetchOffset(gbufID, coord, 0, ivec2( 1,  0)).r;
        selUV = vec2(0.0, texelFetchOffset(gbufUV, coord, 0, ivec2( 1,  0)).rg.y);
        selRawW = texelFetchOffset(gbufNorm, coord, 0, ivec2( 1,  0)).w;
    } else if (nn.a >= 0.5) {
        selID = texelFetchOffset(gbufID, coord, 0, ivec2(-1,  0)).r;
        selUV = vec2(1.0, texelFetchOffset(gbufUV, coord, 0, ivec2(-1,  0)).rg.y);
        selRawW = texelFetchOffset(gbufNorm, coord, 0, ivec2(-1,  0)).w;
    } else if (nt.a >= 0.5) {
        selID = texelFetchOffset(gbufID, coord, 0, ivec2( 0,  1)).r;
        selUV = vec2(texelFetchOffset(gbufUV, coord, 0, ivec2( 0,  1)).rg.x, 0.0);
        selRawW = texelFetchOffset(gbufNorm, coord, 0, ivec2( 0,  1)).w;
    } else if (nb.a >= 0.5) {
        selID = texelFetchOffset(gbufID, coord, 0, ivec2( 0, -1)).r;
        selUV = vec2(texelFetchOffset(gbufUV, coord, 0, ivec2( 0, -1)).rg.x, 1.0);
        selRawW = texelFetchOffset(gbufNorm, coord, 0, ivec2( 0, -1)).w;
    }

    if (selID != 0u) {
        vec3 pos, norm;
        evalBezier(selID, selUV, pos, norm);
        bool  selEven   = selRawW >= 0.0;
        float checker = mix(1.0, checkerFromParity(selEven), lighting.checkerboard);
        outColor = vec4(shadeNormal(norm, lightDir, checker), 1.0);
    } else {
        outColor = vec4(0.1, 0.1, 0.15, 1.0);
    }
}
