#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nurbs.h"
#include "knot_insert.h"
#include "bezier_eval.h"

// ============================================================
// .surf file I/O
// ============================================================

static void free_surf(NurbsSurf* s);

static int next_line(FILE* f, char* buf, int size) {
    while (fgets(buf, size, f))
        if (buf[0] != '#' && buf[0] != '\n') return 1;
    return 0;
}

static int count_floats(const char* line) {
    int n = 1;
    for (const char* c = line; *c; c++) if (*c == ' ') n++;
    return n;
}

static void parse_floats(char* line, float* out, int n) {
    char* p = line;
    for (int i = 0; i < n; i++) {
        sscanf(p, "%f", &out[i]);
        char* next = strchr(p, ' ');
        if (!next) break;
        p = next + 1;
    }
}

static NurbsSurf* read_surf(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "  [SKIP] Cannot open %s\n", path); return NULL; }
    NurbsSurf* s = (NurbsSurf*)calloc(1, sizeof(NurbsSurf));
    char line[4096];

    if (!next_line(f, line, sizeof(line))) { fclose(f); free(s); return NULL; }
    sscanf(line, "%d %d %d %d", &s->p_u, &s->p_v, &s->n_u, &s->n_v);
    if (s->p_u <= 0 || s->p_v <= 0 || s->n_u <= 0 || s->n_v <= 0 ||
        s->p_u > 20 || s->p_v > 20 || s->n_u > 10000 || s->n_v > 10000) {
        fclose(f); free(s); return NULL;
    }

    if (!next_line(f, line, sizeof(line))) { fclose(f); free_surf(s); return NULL; }
    {   int n = count_floats(line); s->m_u = n - 1;
        s->U = (float*)malloc(n * sizeof(float)); parse_floats(line, s->U, n); }

    if (!next_line(f, line, sizeof(line))) { fclose(f); free_surf(s); return NULL; }
    {   int n = count_floats(line); s->m_v = n - 1;
        s->V = (float*)malloc(n * sizeof(float)); parse_floats(line, s->V, n); }

    s->Ps = new_cpgrid(s->n_u, s->n_v, 4);
    for (int i = 0; i <= s->n_u; i++)
        for (int j = 0; j <= s->n_v; j++) {
            if (!next_line(f, line, sizeof(line))) { fclose(f); free_surf(s); return NULL; }
            int got = sscanf(line, "%f %f %f %f",
                   &s->Ps[i][j][0], &s->Ps[i][j][1],
                   &s->Ps[i][j][2], &s->Ps[i][j][3]);
            if (got < 1) { fclose(f); free_surf(s); return NULL; }
            if (got < 3) { s->Ps[i][j][1] = 0; s->Ps[i][j][2] = 0; }
            if (got < 4) s->Ps[i][j][3] = 1.0f;
        }
    fclose(f);
    return s;
}

static void free_surf(NurbsSurf* s) {
    free_cpgrid(s->Ps, s->n_u, s->n_v);
    free(s->U); free(s->V); free(s);
}

// ============================================================
// NURBS → triangle mesh
// ============================================================

static void nurbs_to_beziers(NurbsSurf* s) {
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
}

float* nurbs_wireframe(const char* surf_path, int res,
                       int* np_u_out, int* np_v_out)
{
    NurbsSurf* s = read_surf(surf_path);
    if (!s) return NULL;

    nurbs_to_beziers(s);

    int ru = s->p_u + 1, rv = s->p_v + 1;
    int np_u = (s->n_u + 1) / ru, np_v = (s->n_v + 1) / rv;
    *np_u_out = np_u; *np_v_out = np_v;

    float* verts = (float*)malloc(np_u * np_v * res * res * 6 * sizeof(float));
    BezierPatch bp = {s->p_u, s->p_v, (float*)malloc(ru * rv * 4 * sizeof(float))};

    for (int pi = 0; pi < np_u; pi++) {
        for (int pj = 0; pj < np_v; pj++) {
            for (int i = 0; i < ru; i++)
                for (int j = 0; j < rv; j++)
                    memcpy(&bp.CP[(i * rv + j) * 4],
                           s->Ps[pi * ru + i][pj * rv + j], 4 * sizeof(float));

            float* pv = (float*)malloc(res * res * 6 * sizeof(float));
            eval_bezier_patch(&bp, pv, res, res);

            float* dst = verts + (pi * np_v + pj) * res * res * 6;
            for (int y = 0; y < res; y++)
                for (int x = 0; x < res; x++)
                    memcpy(&dst[(y * res + x) * 6], &pv[(x * res + y) * 6], 6 * sizeof(float));
            free(pv);
        }
    }
    free(bp.CP);
    free_surf(s);
    return verts;
}

// ============================================================
// Global adaptive resolution (ETER §4)
// ============================================================

int compute_adaptive_res(const char* surf_path, const CameraParams* cam) {
    NurbsSurf* s = read_surf(surf_path);
    if (!s) return 32;

    nurbs_to_beziers(s);

    int ru = s->p_u + 1, rv = s->p_v + 1;
    int nu = (s->n_u + 1) / ru, nv = (s->n_v + 1) / rv;
    BezierPatch bp = {s->p_u, s->p_v, (float*)malloc(ru * rv * 4 * sizeof(float))};
    int g_tu = 8, g_tv = 8;

    for (int i = 0; i < nu; i++) {
        for (int j = 0; j < nv; j++) {
            for (int ii = 0; ii < ru; ii++)
                for (int jj = 0; jj < rv; jj++)
                    memcpy(&bp.CP[(ii*rv+jj)*4],
                           s->Ps[i*ru+ii][j*rv+jj], 4*sizeof(float));

            float eps = compute_epsilon(&bp, cam);
            int tu = 0, tv = 0;
            compute_tess_factors(&bp, eps, &tu, &tv);
            adapt_tess_factors(&tu, &tv);
            if (tu < 2) tu = 2; if (tv < 2) tv = 2;
            if (tu > 64) tu = 64; if (tv > 64) tv = 64;
            if (tu > g_tu) g_tu = tu;
            if (tv > g_tv) g_tv = tv;
        }
    }
    free(bp.CP);
    free_surf(s);

    int res = (g_tu > g_tv) ? g_tu : g_tv;
    printf("  [ADAPT] %s: %dx%d patches, max tu=%d tv=%d -> res=%d\n",
           surf_path, nu, nv, g_tu, g_tv, res);
    return res;
}
