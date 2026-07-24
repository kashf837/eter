#ifndef BEZIER_EVAL_H
#define BEZIER_EVAL_H

#include <stdio.h>

// Evaluate one Bezier patch on GPU.
// CP: flat array of (order_u+1)*(order_v+1)*4 floats, format [x,y,z,w,...]
// V_out: output array, N*N*3 floats, caller allocates
// N: grid resolution (N x N sampling points)
void eval_bezier_patch(float* CP, int order_u, int order_v,
                       float* V_out, int N);

// Compute binomial coefficients
void compute_binom(int order, int* binom);

#endif
