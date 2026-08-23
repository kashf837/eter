#ifndef BEZIER_EVAL_H
#define BEZIER_EVAL_H

// Per-patch adaptive-tessellation metadata (ETER §4), scene-wide:
//   tu/tv     = tess factors (multiples of 8, clamped [8,64])
//   offset    = GLOBAL offset of this patch's vertices in the scene vertex
//               array (v-major: index = offset + v*tu + u)
//   ru/rv     = CP grid per patch (orders = ru-1, rv-1)
//   cpOff     = GLOBAL offset of this patch's CPs in the scene CP array,
//               in vec4 units (float index = cpOff*4)
typedef struct { int tu, tv, offset, ru, rv, cpOff; } ScenePatchMeta;

// Evaluate all patches of a scene in ONE GPU round trip (gridDim.x = np).
// cps:    flat uncentered CP array, patch-major in meta order; byte size is
//         derived from the meta (Σ ru*rv*4 floats) so mixed CP grids are fine
// meta:   np ScenePatchMeta records, same patch order as cps
// V_out:  output, total_verts*6 floats (pos+normal), per patch v-major,
//         caller allocates
void eval_bezier_patches_scene(const float* cps, const ScenePatchMeta* meta, int np,
                               float* V_out, int total_verts);

#endif
