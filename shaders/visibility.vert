#version 450
// ETER §6.1 Visibility Buffer — Pass 1 vertex shader
// Transforms vertices to world/clip space and passes per-pixel attributes through.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;       // parametric (u,v) within Bézier patch
layout(location = 3) in float inPatchID;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragWorldNormal;
layout(location = 2) out vec2 fragUV;
layout(location = 3) flat out float fragPatchID;

layout(set = 0, binding = 0) uniform MVP {
    mat4 model;
    mat4 view;
    mat4 proj;
} mvp;

void main()
{
    vec4 worldPos = mvp.model * vec4(inPosition, 1.0);
    fragWorldPos   = worldPos.xyz;
    fragWorldNormal = mat3(mvp.model) * inNormal;
    fragUV          = inUV;
    fragPatchID     = inPatchID;
    gl_Position     = mvp.proj * mvp.view * worldPos;
}
