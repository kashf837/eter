#ifndef BEZIER_EVAL_H
#define BEZIER_EVAL_H

#include "knot_insert.h"    // for BezierPatch

// Evaluate one Bezier patch on GPU.
// bp:  BezierPatch with flat CP array
// V_out: output array, nu*nv*3 floats, caller allocates
// nu, nv: sampling grid resolution (u-direction, v-direction)
void eval_bezier_patch(BezierPatch* bp, float* V_out, int nu, int nv);

#endif
