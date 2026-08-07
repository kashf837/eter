#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in float inChecker;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec3 fragWorldPos;
layout(location = 2) out float fragChecker;

layout(set = 0, binding = 0) uniform MVP {
    mat4 model;
    mat4 view;
    mat4 proj;
} mvp;

void main()
{
    vec4 worldPos = mvp.model * vec4(inPosition, 1.0);
    fragWorldPos = worldPos.xyz;
    // Transform normal to world space (no non-uniform scale, so just mat3)
    fragNormal = mat3(mvp.model) * inNormal;
    fragChecker = inChecker;
    gl_Position = mvp.proj * mvp.view * worldPos;
}
