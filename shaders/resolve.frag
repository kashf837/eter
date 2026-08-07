#version 450
// ETER §6.2 Crack Filling — Pass 2 (resolve) fragment shader
// Boundary pixels: blend lighting from both adjacent patches to fix
// z-fighting caused by conservative raster overlap.

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D gbufPos;
layout(set = 0, binding = 1) uniform sampler2D gbufNorm;
layout(set = 0, binding = 2) uniform usampler2D gbufID;

layout(set = 1, binding = 0) uniform Lighting {
    float intensity;
    float lightAz;
    float lightEl;
    float checkerboard;
    float boundary;
} lighting;

// Compute Lambertian lighting for a single G-buffer sample
vec3 evalLighting(vec3 worldNorm, vec3 lightDir)
{
    float ambient  = 0.4;
    float diffuse  = max(dot(normalize(worldNorm), lightDir), 0.0);
    return vec3(0.85) * (ambient + diffuse * 0.6) * lighting.intensity;
}

// Decode 2D checkerboard value from encoded patchID
float decodeChecker(uint pid)
{
    uint pi = pid >> 12;
    uint pj = pid & 0xFFFu;
    return (((pi + pj) & 1u) != 0u) ? 0.6 : 1.0;
}

void main()
{
    ivec2 uv = ivec2(gl_FragCoord.xy);

    // Background pixels: G-buffer was cleared to zero → skip lighting
    vec4 posSample  = texelFetch(gbufPos, uv, 0);
    if (posSample.a < 0.5) {
        outColor = vec4(0.1, 0.1, 0.15, 1.0);
        return;
    }

    vec3 worldPos   = posSample.xyz;
    vec3 worldNorm  = texelFetch(gbufNorm, uv, 0).xyz;
    uint patchID    = texelFetch(gbufID,   uv, 0).r;

    vec3 lightDir = vec3(
        cos(lighting.lightEl) * sin(lighting.lightAz),
        sin(lighting.lightEl),
        cos(lighting.lightEl) * cos(lighting.lightAz)
    );

    // Check 4-neighborhood for patch boundary
    uint n0 = texelFetch(gbufID, uv + ivec2( 1,  0), 0).r;
    uint n1 = texelFetch(gbufID, uv + ivec2(-1,  0), 0).r;
    uint n2 = texelFetch(gbufID, uv + ivec2( 0,  1), 0).r;
    uint n3 = texelFetch(gbufID, uv + ivec2( 0, -1), 0).r;
    bool isBoundary = (n0 != patchID || n1 != patchID || n2 != patchID || n3 != patchID);

    // Lighting for current pixel
    vec3 colorCur = evalLighting(worldNorm, lightDir)
                    * mix(1.0, decodeChecker(patchID), lighting.checkerboard);

    // At patch boundaries: blend lighting from both adjacent patches.
    // Only blend when the neighbor also has mesh data (not background/silhouette).
    vec3 colorOut = colorCur;
    if (isBoundary) {
        ivec2 nuv = uv;
        if      (n0 != patchID && n0 != 0u) nuv = uv + ivec2( 1,  0);
        else if (n1 != patchID && n1 != 0u) nuv = uv + ivec2(-1,  0);
        else if (n2 != patchID && n2 != 0u) nuv = uv + ivec2( 0,  1);
        else if (n3 != patchID && n3 != 0u) nuv = uv + ivec2( 0, -1);
        else nuv = uv; // fallback: all differing neighbors are background → no blend

        if (nuv != uv) {
            vec3  neighNorm  = texelFetch(gbufNorm, nuv, 0).xyz;
            uint  neighPID   = texelFetch(gbufID,   nuv, 0).r;
            vec3 colorNeigh  = evalLighting(neighNorm, lightDir)
                              * mix(1.0, decodeChecker(neighPID), lighting.checkerboard);
            colorOut = mix(colorCur, colorNeigh, 0.5);
        }
    }

    // Debug: red highlight on patch boundaries when toggle is ON
    outColor = (lighting.boundary > 0.5 && isBoundary)
        ? vec4(mix(colorOut, vec3(1.0, 0.0, 0.0), 0.6), 1.0)
        : vec4(colorOut, 1.0);
}
