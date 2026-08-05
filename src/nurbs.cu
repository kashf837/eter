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
    char line[256];

    if (!next_line(f, line, 256)) { fclose(f); free(s); return NULL; }
    sscanf(line, "%d %d %d %d", &s->p_u, &s->p_v, &s->n_u, &s->n_v);
    if (s->p_u <= 0 || s->p_v <= 0 || s->n_u <= 0 || s->n_v <= 0 ||
        s->p_u > 20 || s->p_v > 20 || s->n_u > 10000 || s->n_v > 10000) {
        fprintf(stderr, "  [SKIP] %s: bad dims p=(%d,%d) n=(%d,%d)\n",
                path, s->p_u, s->p_v, s->n_u, s->n_v);
        fclose(f); free(s); return NULL;
    }

    // U knot vector — count actual values, derive m_u from count
    if (!next_line(f, line, 256)) { fclose(f); free_surf(s); return NULL; }
    {   int n = count_floats(line); s->m_u = n - 1;
        s->U = (float*)malloc(n * sizeof(float)); parse_floats(line, s->U, n); }

    // V knot vector
    if (!next_line(f, line, 256)) { fclose(f); free_surf(s); return NULL; }
    {   int n = count_floats(line); s->m_v = n - 1;
        s->V = (float*)malloc(n * sizeof(float)); parse_floats(line, s->V, n); }

    // Control points
    s->Ps = new_cpgrid(s->n_u, s->n_v, 4);
    for (int i = 0; i <= s->n_u; i++)
        for (int j = 0; j <= s->n_v; j++) {
            if (!next_line(f, line, 256) ||
                sscanf(line, "%f %f %f %f",
                       &s->Ps[i][j][0], &s->Ps[i][j][1],
                       &s->Ps[i][j][2], &s->Ps[i][j][3]) < 4) {
                fprintf(stderr, "  [SKIP] %s: bad CP at (%d,%d)\n", path, i, j);
                fclose(f); free_surf(s); return NULL;
            }
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

    float* verts = (float*)malloc(np_u * np_v * res * res * 3 * sizeof(float));
    BezierPatch bp = {s->p_u, s->p_v, (float*)malloc(ru * rv * 4 * sizeof(float))};

    for (int pi = 0; pi < np_u; pi++) {
        for (int pj = 0; pj < np_v; pj++) {
            for (int i = 0; i < ru; i++)
                for (int j = 0; j < rv; j++)
                    memcpy(&bp.CP[(i * rv + j) * 4],
                           s->Ps[pi * ru + i][pj * rv + j], 4 * sizeof(float));

            float* pv = (float*)malloc(res * res * 3 * sizeof(float));
            eval_bezier_patch(&bp, pv, res, res);

            // Transpose: GPU output is x-major, we need y-major
            float* dst = verts + (pi * np_v + pj) * res * res * 3;
            for (int y = 0; y < res; y++)
                for (int x = 0; x < res; x++)
                    memcpy(&dst[(y * res + x) * 3], &pv[(x * res + y) * 3], 3 * sizeof(float));
            free(pv);
        }
    }
    free(bp.CP);
    free_surf(s);
    return verts;
}
