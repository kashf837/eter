#pragma once

// Full NURBS pipeline: load .surf → knot insertion → Bezier extraction
// → GPU evaluation → heatmap texture in cpu_data[W*H*4] (RGBA).
// Returns 0 on success, -1 on error.
int nurbs_to_texture(float* cpu_data, int W, int H, const char* surf_path);
