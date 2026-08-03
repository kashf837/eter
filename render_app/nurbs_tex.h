#pragma once

// Hardcoded test Bezier patch: 3x3 CPs, p=2, pillow shape
struct TestPatch {
    int   p_u, p_v;     // degree
    int   n_cp;          // total CP count = (p_u+1)*(p_v+1)
    float CP[48];        // flat: n_cp * 4 floats (x,y,z,w), max 12 CPs
};

// Evaluate the Bezier surface at WxH points, map (x,y,z) -> (R,G,B,A).
// Writes result to cpu_data[W*H*4].
void launch_nurbs_texture(float* cpu_data, int W, int H, const TestPatch* patch);
