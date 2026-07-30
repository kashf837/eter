#include "render.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ========== Projection helpers ==========

__device__ float edge(float ax, float ay, float bx, float by, float px, float py) {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

// Project 3D point to screen. Returns false if behind camera.
__device__ bool project(float x, float y, float z,
                         float* sx, float* sy, float* sz,
                         float A, float B, float cx, float cy, float cz) {
    float ex = x - cx, ey = y - cy, ez = z - cz;
    if (ez <= 0.01f) return false;
    *sz = ez;
    *sx = (A * ex / ez * 0.5f + 0.5f) * (float)SCR_W;
    *sy = (0.5f - B * ey / ez * 0.5f) * (float)SCR_H;
    return true;
}

// ========== Rasterization kernel ==========

__global__ void rasterize_patch_kernel(float* V, int tu, int tv,
                                        float* depth, float A, float B,
                                        float cx, float cy, float cz) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    int nq = (tu - 1) * (tv - 1);
    if (q >= nq) return;

    int i = q / (tv - 1);
    int j = q % (tv - 1);

    // Four corners of this quad
    float *v00 = &V[( i      * tv +  j     ) * 3];
    float *v10 = &V[((i + 1) * tv +  j     ) * 3];
    float *v01 = &V[( i      * tv + (j + 1)) * 3];
    float *v11 = &V[((i + 1) * tv + (j + 1)) * 3];

    // Project to screen
    float s00x, s00y, s00z, s10x, s10y, s10z;
    float s01x, s01y, s01z, s11x, s11y, s11z;
    if (!project(v00[0],v00[1],v00[2], &s00x,&s00y,&s00z, A,B, cx,cy,cz) ||
        !project(v10[0],v10[1],v10[2], &s10x,&s10y,&s10z, A,B, cx,cy,cz) ||
        !project(v01[0],v01[1],v01[2], &s01x,&s01y,&s01z, A,B, cx,cy,cz) ||
        !project(v11[0],v11[1],v11[2], &s11x,&s11y,&s11z, A,B, cx,cy,cz))
        return;

    // Bounding box
    int x0 = (int)fminf(fminf(s00x,s10x), fminf(s01x,s11x));
    int x1 = (int)fmaxf(fmaxf(s00x,s10x), fmaxf(s01x,s11x)) + 1;
    int y0 = (int)fminf(fminf(s00y,s10y), fminf(s01y,s11y));
    int y1 = (int)fmaxf(fmaxf(s00y,s10y), fmaxf(s01y,s11y)) + 1;
    if (x0 < 0) x0 = 0; if (x1 > SCR_W) x1 = SCR_W;
    if (y0 < 0) y0 = 0; if (y1 > SCR_H) y1 = SCR_H;

    for (int py = y0; py < y1; py++)
        for (int px = x0; px < x1; px++) {
            float sx = px + 0.5f, sy = py + 0.5f;

            // Lower triangle: (v00, v10, v11)
            float e0 = edge(s00x,s00y, s10x,s10y, sx, sy);
            float e1 = edge(s10x,s10y, s11x,s11y, sx, sy);
            float e2 = edge(s11x,s11y, s00x,s00y, sx, sy);
            bool in = (e0>=-1e-6f&&e1>=-1e-6f&&e2>=-1e-6f)
                   || (e0<= 1e-6f&&e1<= 1e-6f&&e2<= 1e-6f);

            if (!in) {
                // Upper triangle: (v00, v11, v01)
                e0 = edge(s00x,s00y, s11x,s11y, sx, sy);
                e1 = edge(s11x,s11y, s01x,s01y, sx, sy);
                e2 = edge(s01x,s01y, s00x,s00y, sx, sy);
                in = (e0>=-1e-6f&&e1>=-1e-6f&&e2>=-1e-6f)
                  || (e0<= 1e-6f&&e1<= 1e-6f&&e2<= 1e-6f);
            }

            if (in) {
                // Conservative depth: use nearest corner
                float z = fminf(fminf(s00z,s10z), fminf(s01z,s11z));
                int* dbuf = (int*)depth;
                atomicMin(&dbuf[py * SCR_W + px], __float_as_int(z));
            }
        }
}

// ========== Public API ==========

DepthBuffer* create_depth_buffer(int w, int h) {
    DepthBuffer* db = (DepthBuffer*)malloc(sizeof(DepthBuffer));
    db->W = w; db->H = h;
    cudaMalloc(&db->d_depth, w * h * sizeof(float));
    int n = w * h;
    float* init = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) init[i] = 1e10f;
    cudaMemcpy(db->d_depth, init, n * sizeof(float), cudaMemcpyHostToDevice);
    free(init);
    return db;
}

void free_depth_buffer(DepthBuffer* db) {
    cudaFree(db->d_depth);
    free(db);
}

void rasterize_patch(float* d_V, int tu, int tv, DepthBuffer* db) {
    float B = 1.0f / tanf((FOV_Y * 3.14159265f / 180.0f) * 0.5f);
    float A = B * (float)SCR_W / (float)SCR_H;
    float cx = CAM_X, cy = CAM_Y, cz = CAM_Z;

    int nq = (tu - 1) * (tv - 1);
    int th = 256, blk = (nq + th - 1) / th;
    rasterize_patch_kernel<<<blk, th>>>(d_V, tu, tv, db->d_depth, A, B, cx, cy, cz);
}

void write_ppm(const char* path, DepthBuffer* db) {
    int n = db->W * db->H;
    float* buf = (float*)malloc(n * sizeof(float));
    cudaMemcpy(buf, db->d_depth, n * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* f = fopen(path, "w");
    fprintf(f, "P2\n%d %d\n255\n", db->W, db->H);
    float zmin = 1e10f, zmax = 0.0f;
    for (int i = 0; i < n; i++)
        if (buf[i] < 1e9f) {
            if (buf[i] < zmin) zmin = buf[i];
            if (buf[i] > zmax) zmax = buf[i];
        }
    float range = zmax - zmin;
    if (range < 0.001f) range = 0.001f;
    for (int y = 0; y < db->H; y++) {
        for (int x = 0; x < db->W; x++) {
            float z = buf[y * db->W + x];
            int g = (z >= 1e9f) ? 0 : (int)((zmax - z) / range * 255.0f);
            if (g > 255) g = 255; if (g < 0) g = 0;
            fprintf(f, "%d ", g);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    free(buf);
}
