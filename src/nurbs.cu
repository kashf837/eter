#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nurbs.h"
#include "knot_insert.h"
#include "bezier_eval.h"

// Forward declaration — read_surf uses it before free_surf is defined
static void free_surf(NurbsSurf* s);

static int next_line(FILE* f, char* buf, int size) {
    while (fgets(buf, size, f))
        if (buf[0] != '#' && buf[0] != '\n') return 1;
    return 0;
}

static NurbsSurf* read_surf(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) { printf("  [SKIP] Cannot open %s\n", path); return NULL; }
    NurbsSurf* s = (NurbsSurf*)calloc(1, sizeof(NurbsSurf));
    char line[256];

    if (!next_line(f, line, 256)) { fclose(f); free(s); return NULL; }
    sscanf(line, "%d %d %d %d", &s->p_u, &s->p_v, &s->n_u, &s->n_v);
    // Sanity check: reject obviously corrupted files (negative / huge dimensions)
    if (s->p_u <= 0 || s->p_v <= 0 || s->n_u <= 0 || s->n_v <= 0 ||
        s->p_u > 20 || s->p_v > 20 || s->n_u > 10000 || s->n_v > 10000) {
        printf("  [SKIP] %s: bad dimensions p=(%d,%d) n=(%d,%d)\n", path, s->p_u, s->p_v, s->n_u, s->n_v);
        fclose(f); free(s); return NULL;
    }
    // Read U knot vector — count actual values, set m_u from that
    if (!next_line(f, line, 256)) { fclose(f); free_surf(s); return NULL; }
    {
        int cnt = 1;  // count spaces to get number of values
        for (char* c = line; *c; c++) if (*c == ' ') cnt++;
        s->m_u = cnt - 1;  // last index
        s->U = (float*)malloc(cnt * sizeof(float));
        char* p = line;
        for (int i = 0; i < cnt; i++) {
            sscanf(p, "%f", &s->U[i]);
            char* next = strchr(p, ' ');
            if (!next) { p += strlen(p); break; }
            p = next + 1;
        }
    }

    // Read V knot vector similarly
    if (!next_line(f, line, 256)) { fclose(f); free_surf(s); return NULL; }
    {
        int cnt = 1;
        for (char* c = line; *c; c++) if (*c == ' ') cnt++;
        s->m_v = cnt - 1;
        s->V = (float*)malloc(cnt * sizeof(float));
        char* p = line;
        for (int i = 0; i < cnt; i++) {
            sscanf(p, "%f", &s->V[i]);
            char* next = strchr(p, ' ');
            if (!next) { p += strlen(p); break; }
            p = next + 1;
        }
    }

    s->Ps = new_cpgrid(s->n_u, s->n_v, 4);
    for (int i = 0; i <= s->n_u; i++)
        for (int j = 0; j <= s->n_v; j++) {
            if (!next_line(f, line, 256)) {
                printf("  [SKIP] %s: missing CP at (%d,%d)\n", path, i, j);
                fclose(f); free_surf(s); return NULL;
            }
            int got = sscanf(line, "%f %f %f %f",
                   &s->Ps[i][j][0], &s->Ps[i][j][1],
                   &s->Ps[i][j][2], &s->Ps[i][j][3]);
            if (got < 4) {
                printf("  [SKIP] %s: bad CP line at (%d,%d) got=%d\n", path, i, j, got);
                fclose(f); free_surf(s); return NULL;
            }
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
        }
    }

    // Crack prevention: all patches must share the same tu and tv.
    // (Full fix requires §6 crack-filling pipeline; this is the simple fallback.)
    int max_tu = 0, max_tv = 0;
    for (int i = 0; i < np_u * np_v; i++) {
        if (tu_arr[i] > max_tu) max_tu = tu_arr[i];
        if (tv_arr[i] > max_tv) max_tv = tv_arr[i];
    }
    for (int i = 0; i < np_u * np_v; i++) {
        tu_arr[i] = max_tu; tv_arr[i] = max_tv;
    }
    int tu = max_tu, tv = max_tv;

    // Shared-boundary vertex grid: unique vertices = (np_u*(tu-1)+1) × (np_v*(tv-1)+1)
    int gCols = np_u * (tu - 1) + 1;  // global grid columns
    int gRows = np_v * (tv - 1) + 1;  // global grid rows
    *vertCountOut = gCols * gRows;
    float* verts = (float*)calloc(*vertCountOut * 3, sizeof(float));

    printf("Global grid: %dx%d = %d verts (was %d per-patch)\n",
           gCols, gRows, *vertCountOut, np_u * np_v * tu * tv);

    // Evaluate each patch and write interior + boundary vertices
    // Each patch covers: col [pi*(tu-1), (pi+1)*(tu-1)], row [pj*(tv-1), (pj+1)*(tv-1)]
    for (int pi = 0; pi < np_u; pi++) {
        for (int pj = 0; pj < np_v; pj++) {
            for (int i = 0; i < ru; i++)
                for (int j = 0; j < rv; j++)
                    memcpy(&bp.CP[(i * rv + j) * 4],
                           s->Ps[pi * ru + i][pj * rv + j], 4 * sizeof(float));

            float* V = (float*)malloc(tu * tv * 3 * sizeof(float));
            eval_bezier_patch(&bp, V, tu, tv);

            int cx = pi * (tu - 1);  // column offset in global grid
            int cy = pj * (tv - 1);  // row offset in global grid
            for (int y = 0; y < tv; y++)
                for (int x = 0; x < tu; x++) {
                    int gx = cx + x, gy = cy + y;
                    float* dst = verts + (gy * gCols + gx) * 3;
                    dst[0] = V[(x * tv + y) * 3 + 0];
                    dst[1] = V[(x * tv + y) * 3 + 1];
                    dst[2] = V[(x * tv + y) * 3 + 2];
                }
            free(V);
        }
    }

    // Build index buffer: each patch references shared global grid vertices
    *idxCountOut = np_u * np_v * (tu - 1) * (tv - 1) * 6;
    *indicesOut = (uint32_t*)malloc(*idxCountOut * sizeof(uint32_t));
    uint32_t* idxPtr = *indicesOut;
    for (int pi = 0; pi < np_u; pi++) {
        for (int pj = 0; pj < np_v; pj++) {
            int cx = pi * (tu - 1);  // column offset in global grid
            int cy = pj * (tv - 1);
            for (int y = 0; y < tv - 1; y++)
                for (int x = 0; x < tu - 1; x++) {
                    uint32_t a = (cy + y) * gCols + (cx + x);
                    uint32_t b = a + 1;
                    uint32_t c = a + gCols, d = c + 1;
                    *idxPtr++ = a; *idxPtr++ = b; *idxPtr++ = c;
                    *idxPtr++ = b; *idxPtr++ = d; *idxPtr++ = c;
                }
        }
    }

    free(tu_arr); free(tv_arr);
    free(bp.CP); free_surf(s);
    return verts;
}
