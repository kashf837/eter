#pragma once
#include <stdint.h>
#include "adaptive_tess.h"  // for CameraParams
#include "bezier_eval.h"    // for PatchInfo

// Per-patch adaptive evaluation (ETER §4). Each patch gets its own (tu,tv).
// Returns flat vertex array + per-patch metadata. Caller frees verts and info.
// All patches of one surface share the same CP grid size ru×rv
// (orders = p_u+1, p_v+1), returned via ru_out/rv_out.
// If cp_out is non-NULL, also returns flat uncentered Bézier CP array
// ((ru*rv)*4 floats/patch, xyz+w; e.g. 64 for bicubic).
float* nurbs_adaptive_eval(const char* surf_path, const CameraParams* cam,
                           int* np_u, int* np_v, PatchInfo** info, int* total_verts,
                           int* ru_out, int* rv_out, float** cp_out);

// Adaptive evaluation from pre-converted Bézier CPs — skips file I/O and
// NURBS→Bézier knot insertion (ETER §4 rebuild cache). Caller frees verts/info.
// cps: nu*nv patches × ru*rv vec4s (xyz+w), uncentered, patch-major layout.
float* nurbs_adaptive_eval_cached(const float* cps, int nu, int nv, int ru, int rv,
                                  const CameraParams* cam,
                                  PatchInfo** info_out, int* total_verts);
