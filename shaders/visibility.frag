#version 450
// ETER §6.1 Visibility Buffer — Pass 1 fragment shader
// Writes per-pixel attributes to G-buffer attachments.

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragWorldNormal;
layout(location = 2) flat in float fragPatchID;

layout(location = 0) out vec4 outPos;   // gbufPos: world-space position
layout(location = 1) out vec4 outNorm;  // gbufNorm: world-space normal
layout(location = 2) out uint outID;    // gbufID: patch index

void main()
{
    outPos  = vec4(fragWorldPos, 1.0);
    outNorm = vec4(normalize(fragWorldNormal), 1.0);
    outID   = uint(fragPatchID + 0.5);  // float→int: safe for <16M patches
}
