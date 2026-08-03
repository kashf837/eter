#pragma once

// Full NURBS pipeline: load .surf → KI → Bezier eval → heatmap texture.
// Returns 0 on success, -1 on failure.
int launch_nurbs_texture(float* cpu_data, int W, int H, const char* surf_path);
