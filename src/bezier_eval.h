#ifndef BEZIER_EVAL_H
#define BEZIER_EVAL_H

// Per-patch adaptive-tessellation metadata (ETER §4): tess factors + vertex
// offset of this patch inside the surface's flat vertex array.
typedef struct { int tu, tv, offset; } PatchInfo;

// Batch-evaluate all patches of one surface in a single GPU round trip.
// cps:    np patches × ru*rv vec4s (xyz+w), uncentered, patch-major
// info:   np PatchInfo records {tu, tv, offset}, same patch order as cps
// V_out:  output, total_verts*6 floats (pos+normal), per patch v-major
//         (index = offset + v*tu + u), caller allocates
void eval_bezier_patches_batched(const float* cps, const PatchInfo* info, int np,
                                 int ru, int rv, float* V_out, int total_verts);

#endif
