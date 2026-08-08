#version 450
// ETER §6.1 Visibility Buffer — Pass 1 fragment shader
// Writes per-pixel attributes to G-buffer attachments.

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragWorldNormal;
layout(location = 2) in vec2 fragUV;
layout(location = 3) flat in float fragPatchID;
layout(location = 4) in vec2 fragGrid;  // interpolated grid (x,y) within patch

layout(location = 0) out vec4 outPos;   // gbufPos: world-space position
layout(location = 1) out vec4 outNorm;  // gbufNorm: world-space normal
layout(location = 2) out uint outID;    // gbufID: patch index
layout(location = 3) out vec2 outUV;    // gbufUV: parametric (u,v) coordinates

void main()
{
    outPos  = vec4(fragWorldPos, 1.0);
    float raw    = fragPatchID;
    float detail = fract(abs(raw)) * 10.0;  // recover detail [0,1] from fractional

    // Per-quad checkerboard: snap interpolated grid coords to nearest integer
    int cx = int(round(fragGrid.x));
    int cy = int(round(fragGrid.y));
    bool isEvenQuad = ((cx + cy) & 1) == 0;

    outNorm = vec4(normalize(fragWorldNormal), isEvenQuad ? detail : -detail);
    outID   = uint(abs(raw) + 0.5);
    outUV   = fragUV;
}
