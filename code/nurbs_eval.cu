#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "knot_insert.h"
#include "bezier_eval.h"

#define N 32

// ========== File I/O ==========

static int next_line(FILE* f, char* buf, int size) {
    while (fgets(buf, size, f))
        if (buf[0] != '#' && buf[0] != '\n') return 1;
    return 0;
}

typedef struct {
    int p_u, p_v, n_u, n_v, m_u, m_v;
    float* U, *V;
    CPGrid Ps;
} NurbsSurf;

static NurbsSurf* read_surf(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) { printf("Cannot open %s\n", path); return NULL; }

    NurbsSurf* s = (NurbsSurf*)malloc(sizeof(NurbsSurf));
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

// ========== Main ==========

int main() {
    // 1. Load
    NurbsSurf* s = read_surf("data/test.surf");
    if (!s) return 1;
    printf("Loaded: p=(%d,%d)  CPs=%dx%d\n", s->p_u, s->p_v, s->n_u+1, s->n_v+1);

    // 2. Knot insertion in u and v
    for (int i = s->p_u + 1; i <= s->m_u - s->p_u - 1; i++) {
        float u = s->U[i];
        while (mult_of(s->U, s->m_u, u) < s->p_u + 1) {
            int k = find_span(s->U, s->m_u, u);
            knot_insert_surf_u(&s->U, &s->m_u, &s->Ps, &s->n_u, s->n_v,
                               s->p_u, 4, u, k);
        }
    }
    for (int i = s->p_v + 1; i <= s->m_v - s->p_v - 1; i++) {
        float v = s->V[i];
        while (mult_of(s->V, s->m_v, v) < s->p_v + 1) {
            int k = find_span(s->V, s->m_v, v);
            knot_insert_surf_v(&s->V, &s->m_v, &s->Ps, s->n_u, &s->n_v,
                               s->p_v, 4, v, k);
        }
    }
    printf("After KI: CPs=%dx%d\n", s->n_u+1, s->n_v+1);

    // 3. Extract Bezier patches, evaluate each on GPU, write OBJ
    int ru = s->p_u + 1;  // rows per patch
    int rv = s->p_v + 1;  // cols per patch
    int np_u = (s->n_u + 1) / ru;  // number of patches in u
    int np_v = (s->n_v + 1) / rv;  // number of patches in v

    float* V   = (float*)malloc(N * N * 3 * sizeof(float));
    float* CP  = (float*)malloc(ru * rv * 4 * sizeof(float));

    FILE* obj = fopen("build/nurbs_output.obj", "w");
    int vert_offset = 0;

    for (int pi = 0; pi < np_u; pi++) {
        for (int pj = 0; pj < np_v; pj++) {
            // Flatten this patch's CPs
            for (int i = 0; i < ru; i++)
                for (int j = 0; j < rv; j++)
                    memcpy(&CP[(i * rv + j) * 4],
                           s->Ps[pi * ru + i][pj * rv + j],
                           4 * sizeof(float));

            // GPU eval
            eval_bezier_patch(CP, s->p_u, s->p_v, V, N);

            // Write vertices
            for (int vi = 0; vi < N * N; vi++)
                fprintf(obj, "v %f %f %f\n", V[vi*3+0], V[vi*3+1], V[vi*3+2]);

            // Write faces with vertex offset
            for (int i = 0; i < N - 1; i++)
                for (int j = 0; j < N - 1; j++) {
                    int a = vert_offset + i * N + j + 1, b = a + 1;
                    int c = vert_offset + (i + 1) * N + j + 1, d = c + 1;
                    fprintf(obj, "f %d %d %d\n", a, b, c);
                    fprintf(obj, "f %d %d %d\n", b, d, c);
                }

            vert_offset += N * N;
        }
    }

    fclose(obj);
    free(V); free(CP);
    free_surf(s);
    printf("Done: %d patches -> build/nurbs_output.obj\n", np_u * np_v);
    return 0;
}
