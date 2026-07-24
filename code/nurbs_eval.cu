#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ========== Types & memory ==========
typedef float*  CP;
typedef CP*     CPArray;
typedef CPArray* CPGrid;

static CP new_cp(int dim) {
    return (float*)malloc(dim * sizeof(float));
}

static CPArray new_cparray(int n, int dim) {
    CPArray arr = (CPArray)malloc((n + 1) * sizeof(CP));
    for (int i = 0; i <= n; i++) arr[i] = new_cp(dim);
    return arr;
}

static CPGrid new_cpgrid(int nr, int nc, int dim) {
    CPGrid g = (CPGrid)malloc((nr + 1) * sizeof(CPArray));
    for (int i = 0; i <= nr; i++) g[i] = new_cparray(nc, dim);
    return g;
}

static void free_cpgrid(CPGrid g, int nr, int nc) {
    for (int i = 0; i <= nr; i++) {
        for (int j = 0; j <= nc; j++) free(g[i][j]);
        free(g[i]);
    }
    free(g);
}

// ========== File I/O ==========

static int next_line(FILE* f, char* buf, int size) {
    while (fgets(buf, size, f))
        if (buf[0] != '#' && buf[0] != '\n') return 1;
    return 0;
}

typedef struct {
    int p_u, p_v;       // degree in u, v
    int n_u, n_v;       // CP count in u, v minus 1
    int m_u, m_v;       // knot count minus 1 (= n + p + 1)
    float* U;           // knot vector U [m_u+1]
    float* V;           // knot vector V [m_v+1]
    CPGrid Ps;          // control point grid [n_u+1][n_v+1]
} NurbsSurf;

static NurbsSurf* read_surf(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) { printf("Cannot open %s\n", path); return NULL; }

    NurbsSurf* s = (NurbsSurf*)malloc(sizeof(NurbsSurf));
    char line[256];

    // header: p_u p_v  n_u n_v
    next_line(f, line, 256);
    sscanf(line, "%d %d %d %d", &s->p_u, &s->p_v, &s->n_u, &s->n_v);

    s->m_u = s->n_u + s->p_u + 1;
    s->m_v = s->n_v + s->p_v + 1;

    // U knot vector
    s->U = (float*)malloc((s->m_u + 1) * sizeof(float));
    next_line(f, line, 256);
    char* p = line;
    for (int i = 0; i <= s->m_u; i++, p = strchr(p, ' ') + 1)
        sscanf(p, "%f", &s->U[i]);

    // V knot vector
    s->V = (float*)malloc((s->m_v + 1) * sizeof(float));
    next_line(f, line, 256);
    p = line;
    for (int i = 0; i <= s->m_v; i++, p = strchr(p, ' ') + 1)
        sscanf(p, "%f", &s->V[i]);

    // control points
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

    printf("NURBS surface: p_u=%d p_v=%d  n_u=%d n_v=%d  m_u=%d m_v=%d\n",
           s->p_u, s->p_v, s->n_u, s->n_v, s->m_u, s->m_v);

    printf("U: "); for (int i = 0; i <= s->m_u; i++) printf("%.1f ", s->U[i]);
    printf("\nV: "); for (int i = 0; i <= s->m_v; i++) printf("%.1f ", s->V[i]);
    printf("\nCPs (%dx%d):\n", s->n_u + 1, s->n_v + 1);
    for (int i = 0; i <= s->n_u; i++) {
        printf("  row %d: ", i);
        for (int j = 0; j <= s->n_v; j++)
            printf("(%.1f %.1f %.1f w=%.1f) ",
                s->Ps[i][j][0], s->Ps[i][j][1], s->Ps[i][j][2], s->Ps[i][j][3]);
        printf("\n");
    }

    free_surf(s);
    return 0;
}
