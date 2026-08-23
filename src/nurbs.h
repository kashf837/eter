#pragma once
#include <stdint.h>
#include "adaptive_tess.h"  // for CameraParams

// Uniform-resolution mesh. Caller frees with free().
// All patches of one surface share the same CP grid size ru×rv
// (orders = p_u+1, p_v+1), returned via ru_out/rv_out.
// If cp_out is non-NULL, also returns flat Bézier CP array
// ((ru*rv)*4 floats/patch, xyz+w; e.g. 64 for bicubic).
float* nurbs_wireframe(const char* surf_path, int res,
                       int* np_u_out, int* np_v_out,
                       int* ru_out, int* rv_out,
                       float** cp_out);

// Per-patch adaptive evaluation (ETER §4). Each patch gets its own (tu,tv).
// Returns flat vertex array + per-patch metadata. Caller frees verts and info.
// ru_out/rv_out/cp_out semantics are the same as nurbs_wireframe.
typedef struct { int tu, tv, offset; } PatchInfo;
float* nurbs_adaptive_eval(const char* surf_path, const CameraParams* cam,
                           int* np_u, int* np_v, PatchInfo** info, int* total_verts,
                           int* ru_out, int* rv_out, float** cp_out);
