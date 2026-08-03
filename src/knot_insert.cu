#include "knot_insert.h"
#include <string.h>

// ========== Knot vector helpers ==========

int find_span(KnotVector U, int m, float u) {
    for (int i = 0; i < m; i++)
        if (u >= U[i] && u < U[i + 1]) return i;
    return -1;
}

int mult_of(KnotVector U, int m, float u) {
    int cnt = 0;
    for (int i = 0; i <= m; i++)
        if (U[i] == u) cnt++;
    return cnt;
}

static void expand_grid(CPGrid* Ps, int nr, int nc, int dim, int dir) {
    if (dir == 0) {
        *Ps = (CPGrid)realloc(*Ps, (nr + 1) * sizeof(CPArray));
        (*Ps)[nr] = new_cparray(nc, dim);
    } else {
        for (int r = 0; r <= nc; r++) {
            (*Ps)[r] = (CPArray)realloc((*Ps)[r], (nr + 1) * sizeof(CP));
            (*Ps)[r][nr] = new_cp(dim);
        }
    }
}

static void expand_knot(KnotVector* K, int* m, int k, float x) {
    int old = *m;
    (*m)++;
    *K = (float*)realloc(*K, (*m + 1) * sizeof(float));
    for (int i = old; i > k; i--) (*K)[i + 1] = (*K)[i];
    (*K)[k + 1] = x;
}

static void precompute_alpha(float* alpha, float* K_old, int n_old,
                              int p, float x, int k) {
    for (int i = 0; i <= n_old + 1; i++) {
        if      (i <= k - p)  alpha[i] = 1.0f;
        else if (i >= k + 1)  alpha[i] = 0.0f;
        else alpha[i] = (x - K_old[i]) / (K_old[i + p] - K_old[i]);
    }
}

// ========== Surface knot insertion ==========

// forward declaration
static void surf_insert_along(KnotVector* K, int* m_k, CPGrid* Ps,
                               int* n_along, int n_across,
                               int p, int dim, float x, int k, int dir);

void knot_insert_surf(NurbsSurf* s, int dir, float x) {
    KnotVector* K  = (dir == 0) ? &s->U  : &s->V;
    int* m_k       = (dir == 0) ? &s->m_u : &s->m_v;
    int* n_along   = (dir == 0) ? &s->n_u : &s->n_v;
    int  n_across  = (dir == 0) ?  s->n_v :  s->n_u;
    int  p         = (dir == 0) ?  s->p_u :  s->p_v;

    int k = find_span(*K, *m_k, x);
    surf_insert_along(K, m_k, &s->Ps, n_along, n_across, p, 4, x, k, dir);
}

// ========== Internal helpers ==========

// Process one 1D slice of the control grid.
// dir=0: t is column, rows 0..old_n are the varying index
// dir=1: t is row,    cols 0..old_n are the varying index
static void process_slice(CPGrid* Ps, int t, int old_n, int first, int last,
                           float* alpha, int dim, int dir) {
    // save all old CPs in this slice: P_0 .. P_{old_n}
    float saved[64][8];
    for (int v = 0; v <= old_n; v++)
        for (int d = 0; d < dim; d++) {
            float* cp = (dir == 0) ? &(*Ps)[v][t][d] : &(*Ps)[t][v][d];
            saved[v][d] = *cp;
        }

    // write all new CPs: Q_v = alpha[v] * P_v + (1-alpha[v]) * P_{v-1}
    // alpha[0]=1 (clamp P_{-1}→P_0 harmless), alpha[old_n+1]=0 (clamp P_{old_n+1}→P_{old_n} harmless)
    for (int v = 0; v <= old_n + 1; v++) {
        int pi  = (v <= old_n) ? v : old_n;      // P_i, clamp high end
        int pi1 = (v - 1 >= 0) ? v - 1 : 0;      // P_{i-1}, clamp low end
        for (int d = 0; d < dim; d++) {
            float* dst = (dir == 0) ? &(*Ps)[v][t][d] : &(*Ps)[t][v][d];
            *dst = alpha[v] * saved[pi][d] + (1.0f - alpha[v]) * saved[pi1][d];
        }
    }
}

// ========== Internal implementation ==========

static void surf_insert_along(KnotVector* K, int* m_k, CPGrid* Ps,
                        int* n_along, int n_across,
                        int p, int dim, float x, int k, int dir) {
    int old_m = *m_k, old_n = *n_along;
    int first = k - p + 1;
    int last  = k;

    float* K_old = (float*)malloc((old_m + 1) * sizeof(float));
    memcpy(K_old, *K, (old_m + 1) * sizeof(float));

    float* alpha = (float*)malloc((old_n + p + 2) * sizeof(float));
    precompute_alpha(alpha, K_old, old_n, p, x, k);

    expand_knot(K, m_k, k, x);

    (*n_along)++;
    expand_grid(Ps, *n_along, n_across, dim, dir);

    for (int t = 0; t <= n_across; t++)
        process_slice(Ps, t, old_n, first, last, alpha, dim, dir);

    free(K_old);
    free(alpha);
}
