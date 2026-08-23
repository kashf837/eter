#pragma once
#include <stdint.h>
#include "adaptive_tess.h"  // for CameraParams
#include "bezier_eval.h"    // for ScenePatchMeta

// One surface of the scene: flat uncentered Bézier CPs + grid dims.
// cps: nu*nv patches × ru*rv vec4s (xyz+w), patch-major.
typedef struct { const float* cps; int nu, nv, ru, rv; } SceneSurf;

// Per-patch adaptive evaluation (ETER §4) of a single .surf file. Each patch
// gets its own (tu,tv). Returns flat vertex array + per-patch metadata
// (ScenePatchMeta, surface-local offsets). Caller frees verts and info.
// If cp_out is non-NULL, also returns flat uncentered Bézier CP array
// ((ru*rv)*4 floats/patch, xyz+w; e.g. 64 for bicubic) — caller frees it.
float* nurbs_adaptive_eval(const char* surf_path, const CameraParams* cam,
                           int* np_u, int* np_v, ScenePatchMeta** info, int* total_verts,
                           int* ru_out, int* rv_out, float** cp_out);

// Scene-level adaptive evaluation: pass-1 (host) tess factors for ALL patches
// of ALL surfaces, then ONE GPU launch for the whole scene. Surfaces keep
// their array order; meta offsets/cpOffs are scene-global (the same order
// addPatchGrid consumes batches). Caller frees verts and *meta_out.
float* nurbs_adaptive_eval_scene(const SceneSurf* surfs, int nSurfs,
                                 const CameraParams* cam,
                                 ScenePatchMeta** meta_out, int* total_verts);

// Read one .surf, NURBS→Bézier decompose, extract flat uncentered CPs — no
// evaluation. Returns NULL on any error, incl. degree > 7 (the eval kernel's
// shared Bernstein tables cap orders at 7). Caller frees the CP array.
float* nurbs_read_cps(const char* path, int* nu, int* nv, int* ru, int* rv);

// Read a packed binary .surfb file (written by scripts/step2surf.py --all):
// the whole scene in one file — ONE open replaces 3506 fopen/close pairs
// (the per-open cost of the per-file text path dominated its wall clock).
// All surfaces are decomposed in parallel (OpenMP). Returns per-surface flat
// uncentered Bézier CP arrays (same layout as nurbs_read_cps); entry i is
// NULL if surface i was skipped (degree > 7 etc.). Returns the surface count
// or -1 if the file cannot be read. Caller frees *cpsOut and each non-NULL
// entry, plus the four int arrays.
int nurbs_read_packed(const char* path, float*** cpsOut, int** nuOut, int** nvOut,
                      int** ruOut, int** rvOut);
