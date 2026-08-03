#pragma once

// Evaluate NURBS surface at coarse grid, return x,y,z positions for triangle mesh.
// Returns flat array of (res*res*3) floats per patch: [x0,y0,z0, x1,y1,z1, ...].
// `np_u_out`/`np_v_out` set to patch count. Caller frees with free().
float* nurbs_wireframe(const char* surf_path, int res,
                       int* np_u_out, int* np_v_out);
