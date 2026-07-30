#ifndef BEZIER_EVAL_H
#define BEZIER_EVAL_H

#include <stdio.h>
#include "knot_insert.h"    // for BezierPatch

// Evaluate one Bezier patch on GPU.
// bp: BezierPatch with flat CP array and degree info
// V_out: output array, N*N*3 floats, caller allocates
// N: grid resolution (N x N sampling points)
void eval_bezier_patch(BezierPatch* bp, float* V_out, int nu, int nv);

// Compute binomial coefficients
void compute_binom(int order, int* binom);

#endif
