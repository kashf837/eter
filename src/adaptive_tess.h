#ifndef ADAPTIVE_TESS_H
#define ADAPTIVE_TESS_H

#include "knot_insert.h"    // for BezierPatch

// Screen-space error → object-space epsilon (hardcoded camera)
float compute_epsilon(BezierPatch* bp);

// second-difference → tu, tv from epsilon
void compute_tess_factors(BezierPatch* bp, float eps, int* tu, int* tv);

// Scale tu, tv to (7p+1)x(7q+1) for 8x8 meshlet compatibility
void adapt_tess_factors(int* tu, int* tv);

#endif
