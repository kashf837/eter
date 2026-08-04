#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragWorldPos;

layout(location = 0) out vec4 outColor;

void main()
{
    vec3 N = normalize(fragNormal);
    // Light from upper-right-front
    vec3 lightDir = normalize(vec3(1.0, 2.0, 1.5));
    // Ambient + diffuse Lambertian
    float ambient = 0.4;
    float diffuse = max(dot(N, lightDir), 0.0);
    float brightness = ambient + diffuse * 0.6;
    vec3 gray = vec3(0.85);
    outColor = vec4(gray * brightness, 1.0);
}
