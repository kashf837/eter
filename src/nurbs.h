#pragma once
#include <stdint.h>
#include "adaptive_tess.h"  // for CameraParams

// Uniform-resolution mesh. Caller frees with free().
float* nurbs_wireframe(const char* surf_path, int res,
                       int* np_u_out, int* np_v_out);

// Compute global adaptive resolution: scans all Bezier patches, returns
// unified tu=tv resolution based on worst-case epsilon. Safe for boundary matching.
int compute_adaptive_res(const char* surf_path, const CameraParams* cam);
