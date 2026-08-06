#include "adaptive_tess.h"
#include <math.h>

float compute_epsilon(BezierPatch* bp, const CameraParams* cam) {
    float B = 1.0f / tanf((cam->fovY * 3.14159265f / 180.0f) * 0.5f);
    float A = B * (float)cam->scrW / (float)cam->scrH;
    float tau_x = 1.0f / (float)cam->scrW;
    float tau_y = 1.0f / (float)cam->scrH;

    float z_min = 1e30f, xz_max = 0.0f, yz_max = 0.0f;
    int rows = bp->p_u + 1, cols = bp->p_v + 1;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float xw = bp->CP[(i * cols + j) * 4 + 0];
            float yw = bp->CP[(i * cols + j) * 4 + 1];
            float zw = bp->CP[(i * cols + j) * 4 + 2];
            float w  = bp->CP[(i * cols + j) * 4 + 3];

            float x = xw / w, y = yw / w, z = zw / w;
            // Use actual distance from camera, not camera-space z
            float dx = x - cam->camX;
            float dy = y - cam->camY;
            float dz = z - cam->camZ;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);
            if (dist < 0.01f) dist = 0.01f;

            // Project onto view direction (approximate: use min distance as z_min)
            if (dist < z_min) z_min = dist;
            float axz = fabsf(dx) / dist;
            float ayz = fabsf(dy) / dist;
            if (axz > xz_max) xz_max = axz;
            if (ayz > yz_max) yz_max = ayz;
        }
    }

    float ex = (z_min * tau_x) / (A * (1.0f + xz_max) + tau_x);
    float ey = (z_min * tau_y) / (B * (1.0f + yz_max) + tau_y);
    return (ex < ey) ? ex : ey;
}

// ========== §4.1: Zheng & Sederberg (2000) ε → tu, tv ==========

void compute_tess_factors(BezierPatch* bp, float eps, int* tu, int* tv) {
    int m = bp->p_u + 1,  n = bp->p_v + 1;
    int cols = n;

    // find min weight and max CP norm for r*
    float min_w = 1e30f, max_norm = 0.0f;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float w = bp->CP[(i * cols + j) * 4 + 3];
            if (w < min_w) min_w = w;
            float x = bp->CP[(i * cols + j) * 4 + 0] / w;
            float y = bp->CP[(i * cols + j) * 4 + 1] / w;
            float z = bp->CP[(i * cols + j) * 4 + 2] / w;
            float nd = sqrtf(x*x + y*y + z*z);
            if (nd > max_norm) max_norm = nd;
        }
    float r_star = max_norm;

    // u direction: second difference along rows
    float max_D2u = 0.0f, max_D2w_u = 0.0f;
    for (int i = 0; i <= m - 3; i++)
        for (int j = 0; j < n; j++) {
            float* p0 = &bp->CP[( i      * cols + j) * 4];
            float* p1 = &bp->CP[((i + 1) * cols + j) * 4];
            float* p2 = &bp->CP[((i + 2) * cols + j) * 4];
            float sum2 = 0.0f;
            for (int d = 0; d < 4; d++) {
                float d2 = p2[d] - 2.0f * p1[d] + p0[d];
                sum2 += d2 * d2;
            }
            float nd = sqrtf(sum2);
            if (nd > max_D2u) max_D2u = nd;
            float d2w = p2[3] - 2.0f * p1[3] + p0[3];
            float aw = fabsf(d2w);
            if (aw > max_D2w_u) max_D2w_u = aw;
        }
    float denom_u = m*(m-1)*max_D2u + (r_star - eps)*m*(m-1)*max_D2w_u;
    float delta_u = sqrtf(8.0f * eps * min_w / denom_u);
    if (delta_u < 1e-10f) delta_u = 1e-10f;
    *tu = (int)ceilf(1.0f / delta_u);

    // v direction: second difference along columns
    float max_D2v = 0.0f, max_D2w_v = 0.0f;
    for (int i = 0; i < m; i++)
        for (int j = 0; j <= n - 3; j++) {
            float* p0 = &bp->CP[(i * cols +  j     ) * 4];
            float* p1 = &bp->CP[(i * cols + (j + 1)) * 4];
            float* p2 = &bp->CP[(i * cols + (j + 2)) * 4];
            float sum2 = 0.0f;
            for (int d = 0; d < 4; d++) {
                float d2 = p2[d] - 2.0f * p1[d] + p0[d];
                sum2 += d2 * d2;
            }
            float nd = sqrtf(sum2);
            if (nd > max_D2v) max_D2v = nd;
            float d2w = p2[3] - 2.0f * p1[3] + p0[3];
            float aw = fabsf(d2w);
            if (aw > max_D2w_v) max_D2w_v = aw;
        }
    float denom_v = n*(n-1)*max_D2v + (r_star - eps)*n*(n-1)*max_D2w_v;
    float delta_v = sqrtf(8.0f * eps * min_w / denom_v);
    if (delta_v < 1e-10f) delta_v = 1e-10f;
    *tv = (int)ceilf(1.0f / delta_v);
}

// ========== §4.2: Round to multiples of BLOCK_SIZE (8) ==========
// Paper uses (7p+1) for meshlet compatibility; we round to 8x for
// clean GPU grid coverage (avoids __syncthreads UB on non-multiples).

void adapt_tess_factors(int* tu, int* tv) {
    *tu = ((*tu + 7) / 8) * 8;  if (*tu < 8) *tu = 8; if (*tu > 64) *tu = 64;
    *tv = ((*tv + 7) / 8) * 8;  if (*tv < 8) *tv = 8; if (*tv > 64) *tv = 64;
}
