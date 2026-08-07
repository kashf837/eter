#version 450
// ETER §6.1 Visibility Buffer — Pass 2 (resolve) vertex shader
// Full-screen triangle: 3 vertices cover the entire viewport.

vec2 positions[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);

void main()
{
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
