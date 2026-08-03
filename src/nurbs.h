#pragma once

// Full NURBS pipeline: load .surf → knot insertion → Bezier extraction
// → GPU evaluation → heatmap texture in cpu_data[W*H*4] (RGBA).
int nurbs_to_texture(float* cpu_data, int W, int H, const char* surf_path);

// Wireframe helper: evaluate surface at coarse grid, return x,y positions.
// `res` samples per patch per direction. Returns flat array of (res*res*2) floats
// per patch: [x0,y0, x1,y1, ...]. Caller frees with free().
// `np_u`/`np_v` set to patch count.
float* nurbs_wireframe(const char* surf_path, int res,
                       int* np_u_out, int* np_v_out);
