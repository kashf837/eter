#version 450

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec3 fragPos;  // pass world position to fragment shader

layout(set = 0, binding = 0) uniform MVP {
    mat4 model;
    mat4 view;
    mat4 proj;
} mvp;

void main()
{
    fragPos = inPosition;  // world-space position = input (model matrix is identity)
    gl_Position = mvp.proj * mvp.view * mvp.model * vec4(inPosition, 1.0);
}
