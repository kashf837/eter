#pragma once
#include "adaptive_tess.h"

// Fixed-resolution mesh (original)
float* nurbs_wireframe(const char* surf_path, int res,
                       int* np_u_out, int* np_v_out);

// Adaptive-resolution mesh: per-patch tess factors from camera + curvature.
// Returns vertex positions (xyz), indices, and counts. Caller frees with free().
float* nurbs_adaptive_mesh(const char* surf_path, const CameraParams* cam,
                           uint32_t** indicesOut, int* vertCountOut, int* idxCountOut);
