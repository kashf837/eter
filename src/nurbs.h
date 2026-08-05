#pragma once
#include <stdint.h>
#include "adaptive_tess.h"  // for CameraParams

// Evaluate NURBS surface at uniform grid, return flat xyz array.
// Caller frees with free(). np_u_out/np_v_out: number of Bezier patches.
float* nurbs_wireframe(const char* surf_path, int res,
                       int* np_u_out, int* np_v_out);
