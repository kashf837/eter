#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragWorldPos;
layout(location = 2) in float fragChecker;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform MVP {
    mat4 model, view, proj;
    float intensity, lightAz, lightEl, checkerboard;
} mvp;

void main()
{
    vec3 N = normalize(fragNormal);
    vec3 lightDir = vec3(cos(mvp.lightEl)*sin(mvp.lightAz), sin(mvp.lightEl), cos(mvp.lightEl)*cos(mvp.lightAz));
    float ambient = 0.4;
    float diffuse = max(dot(N, lightDir), 0.0);
    float brightness = ambient + diffuse * 0.6;
    // Checkerboard: alternate patch brightness to visualize tessellation
    float checker = 0.6 + 0.4 * fragChecker;  // 1.0 → bright, 0.0 → dim
    outColor = vec4(vec3(0.85) * brightness * mvp.intensity * mix(1.0, checker, mvp.checkerboard), 1.0);
}
