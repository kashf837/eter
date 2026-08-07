#version 450
// ETER §6.2 Crack Filling — Pass 2 (resolve) fragment shader
// Reads G-buffer textures, detects patch boundaries, computes lighting.

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D gbufPos;
layout(set = 0, binding = 1) uniform sampler2D gbufNorm;
layout(set = 0, binding = 2) uniform usampler2D gbufID;

layout(set = 1, binding = 0) uniform Lighting {
    float intensity;
    float lightAz;
    float lightEl;
    float checkerboard;
    float boundary;     // >0.5 → show patch boundaries in red
} lighting;

void main()
{
    ivec2 uv = ivec2(gl_FragCoord.xy);

    vec3 worldPos   = texelFetch(gbufPos,  uv, 0).xyz;
    vec3 worldNorm  = texelFetch(gbufNorm, uv, 0).xyz;
    uint patchID    = texelFetch(gbufID,   uv, 0).r;

    // --- §6.2 Boundary detection: check 4-neighborhood ---
    bool isBoundary = false;
    if (lighting.boundary > 0.5) {
        uint n0 = texelFetch(gbufID, uv + ivec2( 1,  0), 0).r;
        uint n1 = texelFetch(gbufID, uv + ivec2(-1,  0), 0).r;
        uint n2 = texelFetch(gbufID, uv + ivec2( 0,  1), 0).r;
        uint n3 = texelFetch(gbufID, uv + ivec2( 0, -1), 0).r;
        isBoundary = (n0 != patchID || n1 != patchID || n2 != patchID || n3 != patchID);
    }

    vec3 N = normalize(worldNorm);
    vec3 lightDir = vec3(
        cos(lighting.lightEl) * sin(lighting.lightAz),
        sin(lighting.lightEl),
        cos(lighting.lightEl) * cos(lighting.lightAz)
    );

    float ambient  = 0.4;
    float diffuse  = max(dot(N, lightDir), 0.0);
    float brightness = ambient + diffuse * 0.6;

    // 2D checkerboard: decode (pi,pj) from patchID = (pi<<12)|pj
    uint pi = patchID >> 12;
    uint pj = patchID & 0xFFFu;
    float checker = (((pi + pj) & 1u) != 0u) ? 0.6 : 1.0;

    vec3 baseColor = vec3(0.85) * brightness * lighting.intensity
                     * mix(1.0, checker, lighting.checkerboard);

    // Boundary visualization: red tint on patch edges
    outColor = isBoundary
        ? vec4(mix(baseColor, vec3(1.0, 0.0, 0.0), 0.7), 1.0)
        : vec4(baseColor, 1.0);
}
