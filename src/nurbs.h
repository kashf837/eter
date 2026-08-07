#pragma once
#include <stdint.h>
#include "adaptive_tess.h"  // for CameraParams

// Uniform-resolution mesh. Caller frees with free().
float* nurbs_wireframe(const char* surf_path, int res,
                       int* np_u_out, int* np_v_out);

// Per-patch adaptive evaluation (ETER §4). Each patch gets its own (tu,tv).
// Returns flat vertex array + per-patch metadata. Caller frees verts and info.
// If cp_out is non-NULL, also returns flat Bézier CP array (64 floats/patch, xyz+w).
typedef struct { int tu, tv, offset; } PatchInfo;
float* nurbs_adaptive_eval(const char* surf_path, const CameraParams* cam,
                           int* np_u, int* np_v, PatchInfo** info, int* total_verts,
                           float** cp_out);
