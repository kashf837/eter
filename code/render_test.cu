#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "knot_insert.h"
#include "bezier_eval.h"
#include "adaptive_tess.h"

#define SCR_W 640
#define SCR_H 640

// Camera
#define FOV_Y  60.0f
#define CAM_X  0.5f
#define CAM_Y  0.5f
#define CAM_Z -2.0f

// Edge function
__device__ float edge(float ax, float ay, float bx, float by, float px, float py) {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

// Project 3D to screen
__device__ int project(float x, float y, float z,
                        float* sx, float* sy, float* sz,
                        float A, float B) {
    float ex = x - CAM_X, ey = y - CAM_Y, ez = z - CAM_Z;
    if (ez <= 0.01f) return 0;
    *sz = ez;
    *sx = (A * ex / ez * 0.5f + 0.5f) * SCR_W;
    *sy = (0.5f - B * ey / ez * 0.5f) * SCR_H;
    return 1;
}

// Rasterize a vertex grid into depth buffer
__global__ void rasterize_grid(float* V, int nu, int nv,
                                float* depth, float A, float B) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    int total_q = (nu - 1) * (nv - 1);
    if (q >= total_q) return;

    int i = q / (nv - 1);
    int j = q % (nv - 1);

    float *v00 = &V[( i      * nv +  j     ) * 3];
    float *v10 = &V[((i + 1) * nv +  j     ) * 3];
    float *v01 = &V[( i      * nv + (j + 1)) * 3];
    float *v11 = &V[((i + 1) * nv + (j + 1)) * 3];

    float s00x,s00y,s00z, s10x,s10y,s10z, s01x,s01y,s01z, s11x,s11y,s11z;
    if (!project(v00[0],v00[1],v00[2], &s00x,&s00y,&s00z, A,B)) return;
    if (!project(v10[0],v10[1],v10[2], &s10x,&s10y,&s10z, A,B)) return;
    if (!project(v01[0],v01[1],v01[2], &s01x,&s01y,&s01z, A,B)) return;
    if (!project(v11[0],v11[1],v11[2], &s11x,&s11y,&s11z, A,B)) return;

    int x0 = (int)fminf(fminf(s00x,s10x), fminf(s01x,s11x));
    int x1 = (int)fmaxf(fmaxf(s00x,s10x), fmaxf(s01x,s11x)) + 1;
    int y0 = (int)fminf(fminf(s00y,s10y), fminf(s01y,s11y));
    int y1 = (int)fmaxf(fmaxf(s00y,s10y), fmaxf(s01y,s11y)) + 1;
    if (x0 < 0) x0 = 0; if (x1 > SCR_W) x1 = SCR_W;
    if (y0 < 0) y0 = 0; if (y1 > SCR_H) y1 = SCR_H;

    for (int py = y0; py < y1; py++)
        for (int px = x0; px < x1; px++) {
            float sx = px + 0.5f, sy = py + 0.5f;

            float e0 = edge(s00x,s00y, s10x,s10y, sx,sy);
            float e1 = edge(s10x,s10y, s11x,s11y, sx,sy);
            float e2 = edge(s11x,s11y, s00x,s00y, sx,sy);
            bool in = (e0>=-1e-6f&&e1>=-1e-6f&&e2>=-1e-6f)
                   || (e0<= 1e-6f&&e1<= 1e-6f&&e2<= 1e-6f);

            if (!in) {
                e0 = edge(s00x,s00y, s11x,s11y, sx,sy);
                e1 = edge(s11x,s11y, s01x,s01y, sx,sy);
                e2 = edge(s01x,s01y, s00x,s00y, sx,sy);
                in = (e0>=-1e-6f&&e1>=-1e-6f&&e2>=-1e-6f)
                  || (e0<= 1e-6f&&e1<= 1e-6f&&e2<= 1e-6f);
            }

            if (in) {
                float z = fminf(fminf(s00z,s10z), fminf(s01z,s11z));
                int* dbuf = (int*)depth;
                atomicMin(&dbuf[py * SCR_W + px], __float_as_int(z));
            }
        }
}

static int next_line(FILE* f, char* buf, int sz) {
    while (fgets(buf, sz, f))
        if (buf[0] != '#' && buf[0] != '\n') return 1;
    return 0;
}

static NurbsSurf* read_surf(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return NULL;
    NurbsSurf* s = (NurbsSurf*)malloc(sizeof(NurbsSurf));
    char line[256];

    next_line(f, line, 256);
    sscanf(line, "%d %d %d %d", &s->p_u, &s->p_v, &s->n_u, &s->n_v);
    s->m_u = s->n_u + s->p_u + 1;
    s->m_v = s->n_v + s->p_v + 1;

    s->U = (float*)malloc((s->m_u + 1) * sizeof(float));
    next_line(f, line, 256);
    char* p = line;
    for (int i = 0; i <= s->m_u; i++, p = strchr(p, ' ') + 1) sscanf(p, "%f", &s->U[i]);

    s->V = (float*)malloc((s->m_v + 1) * sizeof(float));
    next_line(f, line, 256);
    p = line;
    for (int i = 0; i <= s->m_v; i++, p = strchr(p, ' ') + 1) sscanf(p, "%f", &s->V[i]);

    s->Ps = new_cpgrid(s->n_u, s->n_v, 4);
    for (int i = 0; i <= s->n_u; i++)
        for (int j = 0; j <= s->n_v; j++) {
            next_line(f, line, 256);
            sscanf(line, "%f %f %f %f", &s->Ps[i][j][0], &s->Ps[i][j][1],
                   &s->Ps[i][j][2], &s->Ps[i][j][3]);
        }
    fclose(f);
    return s;
}

void write_ppm(const char* path, float* d, int W, int H) {
    FILE* f = fopen(path, "w");
    fprintf(f, "P2\n%d %d\n255\n", W, H);
    // find min and max depth among written pixels
    float zmin = 1e10f, zmax = 0.0f;
    for (int i = 0; i < W*H; i++) {
        if (d[i] < 1e9f) {
            if (d[i] < zmin) zmin = d[i];
            if (d[i] > zmax) zmax = d[i];
        }
    }
    float range = zmax - zmin;
    if (range < 0.001f) range = 0.001f;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float z = d[y*W + x];
            int g = (z >= 1e9f) ? 0 : (int)((zmax - z) / range * 255.0f);
            if (g > 255) g = 255; if (g < 0) g = 0;
            fprintf(f, "%d ", g);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

static void free_surf(NurbsSurf* s) {
    free_cpgrid(s->Ps, s->n_u, s->n_v);
    free(s->U); free(s->V);
    free(s);
}

int main() {
    int np = SCR_W * SCR_H;
    float* depth = (float*)malloc(np * sizeof(float));
    for (int i = 0; i < np; i++) depth[i] = 1e10f;
    float* d_depth;
    cudaMalloc(&d_depth, np * sizeof(float));
    cudaMemcpy(d_depth, depth, np * sizeof(float), cudaMemcpyHostToDevice);

    // Camera projection params
    float B = 1.0f / tanf((FOV_Y * 3.14159265f/180.0f) * 0.5f);
    float A = B * (float)SCR_W / (float)SCR_H;

    // Load surface
    NurbsSurf* s = read_surf("data/test.surf");
    if (!s) { printf("Cannot open surf\n"); return 1; }

    // Knot insertion
    for (int i = s->p_u+1; i <= s->m_u - s->p_u - 1; i++) {
        float u = s->U[i];
        while (mult_of(s->U, s->m_u, u) < s->p_u + 1) knot_insert_surf(s, 0, u);
    }
    for (int i = s->p_v+1; i <= s->m_v - s->p_v - 1; i++) {
        float v = s->V[i];
        while (mult_of(s->V, s->m_v, v) < s->p_v + 1) knot_insert_surf(s, 1, v);
    }

    // Render each patch
    int ru = s->p_u + 1, rv = s->p_v + 1;
    int np_u = (s->n_u + 1 - ru) / ru + 1;
    int np_v = (s->n_v + 1 - rv) / rv + 1;

    BezierPatch bp;
    bp.p_u = s->p_u; bp.p_v = s->p_v;
    bp.CP = (float*)malloc(ru * rv * 4 * sizeof(float));
    float* V = NULL, *d_V = NULL; int Vcap = 0;

    for (int pi = 0; pi < np_u; pi++)
        for (int pj = 0; pj < np_v; pj++) {
            for (int i = 0; i < ru; i++)
                for (int j = 0; j < rv; j++)
                    memcpy(&bp.CP[(i*rv+j)*4], s->Ps[pi*ru+i][pj*rv+j], 4*sizeof(float));

            float eps = compute_epsilon(&bp);
            int tu, tv;
            compute_tess_factors(&bp, eps, &tu, &tv);
            adapt_tess_factors(&tu, &tv);

            int nV = tu * tv;
            if (nV > Vcap) {
                free(V); cudaFree(d_V);
                V = (float*)malloc(nV*3*sizeof(float));
                cudaMalloc(&d_V, nV*3*sizeof(float));
                Vcap = nV;
            }
            eval_bezier_patch(&bp, V, tu, tv);
            cudaMemcpy(d_V, V, nV*3*sizeof(float), cudaMemcpyHostToDevice);

            int nq = (tu-1)*(tv-1);
            int th = 256, blk = (nq + th - 1) / th;
            rasterize_grid<<<blk, th>>>(d_V, tu, tv, d_depth, A, B);
        }

    cudaDeviceSynchronize();
    cudaMemcpy(depth, d_depth, np * sizeof(float), cudaMemcpyDeviceToHost);
    write_ppm("build/render.ppm", depth, SCR_W, SCR_H);

    cudaFree(d_depth); cudaFree(d_V);
    free(depth); free(V); free(bp.CP);
    free_surf(s);
    printf("Done -> build/render.ppm\n");
    return 0;
}
