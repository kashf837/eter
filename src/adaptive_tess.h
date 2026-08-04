#pragma once
#include "knot_insert.h"    // for BezierPatch

typedef struct {
    float camX, camY, camZ;  // camera position in world space
    float fovY;               // vertical field of view (degrees)
    int   scrW, scrH;         // viewport resolution
} CameraParams;

// Screen-space error → object-space epsilon
float compute_epsilon(BezierPatch* bp, const CameraParams* cam);

// Zheng & Sederberg (2000): second-difference → tu, tv
void compute_tess_factors(BezierPatch* bp, float eps, int* tu, int* tv);

// Scale tu, tv to (7p+1)x(7q+1) for 8x8 meshlet compatibility
void adapt_tess_factors(int* tu, int* tv);
