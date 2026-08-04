#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nurbs.h"
#include "knot_insert.h"
#include "bezier_eval.h"

static int next_line(FILE* f, char* buf, int size) {
    while (fgets(buf, size, f))
        if (buf[0] != '#' && buf[0] != '\n') return 1;
    return 0;
}

static NurbsSurf* read_surf(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) { printf("Cannot open %s\n", path); return NULL; }
    NurbsSurf* s = (NurbsSurf*)calloc(1, sizeof(NurbsSurf));
    char line[256];

    next_line(f, line, 256);
    sscanf(line, "%d %d %d %d", &s->p_u, &s->p_v, &s->n_u, &s->n_v);
    s->m_u = s->n_u + s->p_u + 1;
    s->m_v = s->n_v + s->p_v + 1;

    s->U = (float*)malloc((s->m_u + 1) * sizeof(float));
    next_line(f, line, 256);
    char* p = line;
    for (int i = 0; i <= s->m_u; i++, p = strchr(p, ' ') + 1)
        sscanf(p, "%f", &s->U[i]);

    s->V = (float*)malloc((s->m_v + 1) * sizeof(float));
    next_line(f, line, 256);
    p = line;
    for (int i = 0; i <= s->m_v; i++, p = strchr(p, ' ') + 1)
        sscanf(p, "%f", &s->V[i]);

    s->Ps = new_cpgrid(s->n_u, s->n_v, 4);
    for (int i = 0; i <= s->n_u; i++)
        for (int j = 0; j <= s->n_v; j++) {
            next_line(f, line, 256);
            sscanf(line, "%f %f %f %f",
                   &s->Ps[i][j][0], &s->Ps[i][j][1],
                   &s->Ps[i][j][2], &s->Ps[i][j][3]);
        }
    fclose(f);
    return s;
}

static void free_surf(NurbsSurf* s) {
    free_cpgrid(s->Ps, s->n_u, s->n_v);
    free(s->U); free(s->V);
    free(s);
}

// ---- Evaluate all patches at coarse grid, return x,y,z positions for mesh ----

float* nurbs_wireframe(const char* surf_path, int res,
                       int* np_u_out, int* np_v_out)
{
    NurbsSurf* s = read_surf(surf_path);
    if (!s) return NULL;

    // KI
    for (int i = s->p_u + 1; i <= s->m_u - s->p_u - 1; i++) {
        float u = s->U[i];
        while (mult_of(s->U, s->m_u, u) < s->p_u + 1)
            knot_insert_surf(s, 0, u);
    }
    for (int i = s->p_v + 1; i <= s->m_v - s->p_v - 1; i++) {
        float v = s->V[i];
        while (mult_of(s->V, s->m_v, v) < s->p_v + 1)
            knot_insert_surf(s, 1, v);
    }

    int ru = s->p_u + 1, rv = s->p_v + 1;
    int np_u = (s->n_u + 1) / ru;
    int np_v = (s->n_v + 1) / rv;
    *np_u_out = np_u; *np_v_out = np_v;

    float* verts = (float*)malloc(np_u * np_v * res * res * 3 * sizeof(float));
    BezierPatch bp;
    bp.p_u = s->p_u; bp.p_v = s->p_v;
    bp.CP = (float*)malloc(ru * rv * 4 * sizeof(float));

    for (int pi = 0; pi < np_u; pi++) {
        for (int pj = 0; pj < np_v; pj++) {
            for (int i = 0; i < ru; i++)
                for (int j = 0; j < rv; j++)
                    memcpy(&bp.CP[(i * rv + j) * 4],
                           s->Ps[pi * ru + i][pj * rv + j], 4 * sizeof(float));

            float* V = (float*)malloc(res * res * 3 * sizeof(float));
            eval_bezier_patch(&bp, V, res, res);

            float* dst = verts + (pi * np_v + pj) * res * res * 3;
            for (int y = 0; y < res; y++)
                for (int x = 0; x < res; x++) {
                    dst[(y * res + x) * 3 + 0] = V[(x * res + y) * 3 + 0]; // x
                    dst[(y * res + x) * 3 + 1] = V[(x * res + y) * 3 + 1]; // y
                    dst[(y * res + x) * 3 + 2] = V[(x * res + y) * 3 + 2]; // z
                }
            free(V);
        }
    }

    free(bp.CP);
    free_surf(s);
    return verts;
}

// ---- Adaptive mesh: per-patch tess factors from camera + curvature ----

float* nurbs_adaptive_mesh(const char* surf_path, const CameraParams* cam,
                           uint32_t** indicesOut, int* vertCountOut, int* idxCountOut)
{
    // 1. Load + KI
    NurbsSurf* s = read_surf(surf_path);
    if (!s) return NULL;
    for (int i = s->p_u + 1; i <= s->m_u - s->p_u - 1; i++) {
        float u = s->U[i];
        while (mult_of(s->U, s->m_u, u) < s->p_u + 1) knot_insert_surf(s, 0, u);
    }
    for (int i = s->p_v + 1; i <= s->m_v - s->p_v - 1; i++) {
        float v = s->V[i];
        while (mult_of(s->V, s->m_v, v) < s->p_v + 1) knot_insert_surf(s, 1, v);
    }

    int ru = s->p_u + 1, rv = s->p_v + 1;
    int np_u = (s->n_u + 1) / ru, np_v = (s->n_v + 1) / rv;
    printf("Adaptive mesh: %dx%d patches\n", np_u, np_v);

    // 2. Compute per-patch tess factors
    int* tu_arr = (int*)malloc(np_u * np_v * sizeof(int));
    int* tv_arr = (int*)malloc(np_u * np_v * sizeof(int));
    int* voff   = (int*)malloc((np_u * np_v + 1) * sizeof(int)); // vertex offsets
    voff[0] = 0;

    BezierPatch bp;
    bp.p_u = s->p_u; bp.p_v = s->p_v;
    bp.CP = (float*)malloc(ru * rv * 4 * sizeof(float));

    for (int pi = 0; pi < np_u; pi++) {
        for (int pj = 0; pj < np_v; pj++) {
            for (int i = 0; i < ru; i++)
                for (int j = 0; j < rv; j++)
                    memcpy(&bp.CP[(i * rv + j) * 4],
                           s->Ps[pi * ru + i][pj * rv + j], 4 * sizeof(float));

            float eps = compute_epsilon(&bp, cam);
            int tu = 0, tv = 0;
            compute_tess_factors(&bp, eps, &tu, &tv);
            adapt_tess_factors(&tu, &tv);
            if (tu < 2) tu = 2; if (tv < 2) tv = 2;  // minimum 2x2

            int idx = pi * np_v + pj;
            tu_arr[idx] = tu; tv_arr[idx] = tv;
            voff[idx + 1] = voff[idx] + tu * tv;
        }
    }

    *vertCountOut = voff[np_u * np_v];
    float* verts = (float*)malloc(*vertCountOut * 3 * sizeof(float));
    printf("Total adaptive verts: %d (was %d fixed)\n", *vertCountOut,
           np_u * np_v * 12 * 12);  // compare with old res=12

    // 3. Evaluate each patch at its adaptive resolution
    for (int pi = 0; pi < np_u; pi++) {
        for (int pj = 0; pj < np_v; pj++) {
            int idx = pi * np_v + pj;
            int tu = tu_arr[idx], tv = tv_arr[idx];

            for (int i = 0; i < ru; i++)
                for (int j = 0; j < rv; j++)
                    memcpy(&bp.CP[(i * rv + j) * 4],
                           s->Ps[pi * ru + i][pj * rv + j], 4 * sizeof(float));

            float* V = (float*)malloc(tu * tv * 3 * sizeof(float));
            eval_bezier_patch(&bp, V, tu, tv);

            float* dst = verts + voff[idx] * 3;
            for (int y = 0; y < tv; y++)
                for (int x = 0; x < tu; x++) {
                    *dst++ = V[(x * tv + y) * 3 + 0];
                    *dst++ = V[(x * tv + y) * 3 + 1];
                    *dst++ = V[(x * tv + y) * 3 + 2];
                }
            free(V);
        }
    }

    // 4. Build index buffer with correct per-patch offsets
    *idxCountOut = 0;
    for (int pi = 0; pi < np_u; pi++)
        for (int pj = 0; pj < np_v; pj++) {
            int idx = pi * np_v + pj;
            *idxCountOut += (tu_arr[idx] - 1) * (tv_arr[idx] - 1) * 6;
        }

    *indicesOut = (uint32_t*)malloc(*idxCountOut * sizeof(uint32_t));
    uint32_t* idxPtr = *indicesOut;

    for (int pi = 0; pi < np_u; pi++) {
        for (int pj = 0; pj < np_v; pj++) {
            int pidx = pi * np_v + pj;
            int tu = tu_arr[pidx], tv = tv_arr[pidx];
            uint32_t base = voff[pidx];
            for (int y = 0; y < tv - 1; y++)
                for (int x = 0; x < tu - 1; x++) {
                    uint32_t a = base + y * tu + x, b = a + 1;
                    uint32_t c = a + tu, d = c + 1;
                    *idxPtr++ = a; *idxPtr++ = b; *idxPtr++ = c;
                    *idxPtr++ = b; *idxPtr++ = d; *idxPtr++ = c;
                }
        }
    }

    free(tu_arr); free(tv_arr); free(voff);
    free(bp.CP); free_surf(s);
    return verts;
}
