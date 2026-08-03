#version 450

layout(location = 0) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

void main()
{
    // Diverging colormap centered at z=0
    // Peaks (z>0): white → yellow → red
    // Valleys (z<0): white → cyan → blue
    float z = fragPos.z;
    float t = clamp(z / 0.4 + 0.5, 0.0, 1.0);  // map [-0.2, 0.2] to [0, 1]
    vec3 col;
    if (t < 0.5) {
        float s = t * 2.0;
        col = mix(vec3(0.0, 0.2, 1.0), vec3(0.8, 0.8, 0.8), s);  // blue → light gray
    } else {
        float s = (t - 0.5) * 2.0;
        col = mix(vec3(0.8, 0.8, 0.8), vec3(1.0, 0.3, 0.0), s);  // light gray → orange
    }
    outColor = vec4(col, 1.0);
}
