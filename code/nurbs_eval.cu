#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "knot_insert.h"

// ========== File I/O ==========

static int next_line(FILE* f, char* buf, int size) {
    while (fgets(buf, size, f))
        if (buf[0] != '#' && buf[0] != '\n') return 1;
    return 0;
}

typedef struct {
    int p_u, p_v;
    int n_u, n_v;
    int m_u, m_v;
    float* U;
    float* V;
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
    NurbsSurf* s = read_surf("data/test.surf");
    if (!s) return 1;

    printf("Before: p=(%d,%d) CPs=%dx%d\n", s->p_u, s->p_v, s->n_u+1, s->n_v+1);

    // Knot insertion in u-direction
    for (int i = s->p_u + 1; i <= s->m_u - s->p_u - 1; i++) {
        float u = s->U[i];
        while (mult_of(s->U, s->m_u, u) < s->p_u + 1) {
            int k = find_span(s->U, s->m_u, u);
            knot_insert_surf_u(&s->U, &s->m_u, &s->Ps, &s->n_u, s->n_v,
                               s->p_u, 4, u, k);
        }
    }

    // Knot insertion in v-direction
    for (int i = s->p_v + 1; i <= s->m_v - s->p_v - 1; i++) {
        float v = s->V[i];
        while (mult_of(s->V, s->m_v, v) < s->p_v + 1) {
            int k = find_span(s->V, s->m_v, v);
            knot_insert_surf_v(&s->V, &s->m_v, &s->Ps, s->n_u, &s->n_v,
                               s->p_v, 4, v, k);
        }
    }

    printf("After: CPs=%dx%d\n", s->n_u+1, s->n_v+1);
    printf("U: "); for (int i = 0; i <= s->m_u; i++) printf("%.1f ", s->U[i]);
    printf("\nV: "); for (int i = 0; i <= s->m_v; i++) printf("%.1f ", s->V[i]);
    printf("\n");

    free_surf(s);
    return 0;
}
