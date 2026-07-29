#include "knot_insert.h"

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

// ========== Internal implementation ==========

static void surf_insert_along(KnotVector* K, int* m_k, CPGrid* Ps,
                        int* n_along, int n_across,
                        int p, int dim, float x, int k, int dir) {
    int old_m = *m_k, old_n = *n_along;
    int first = k - p + 1;
    int last  = k;

    float K_old[20];
    memcpy(K_old, *K, (old_m + 1) * sizeof(float));

    float alpha[20];
    precompute_alpha(alpha, K_old, old_n, p, x, k);

    expand_knot(K, m_k, k, x);

    (*n_along)++;
    if (dir == 0) {
        // u-direction: add one row
        *Ps = (CPGrid)realloc(*Ps, (*n_along + 1) * sizeof(CPArray));
        (*Ps)[*n_along] = new_cparray(n_across, dim);
    } else {
        // v-direction: add one column to every row
        for (int r = 0; r <= n_across; r++) {
            (*Ps)[r] = (CPArray)realloc((*Ps)[r], (*n_along + 1) * sizeof(CP));
            (*Ps)[r][*n_along] = new_cp(dim);
        }
    }

    for (int t = 0; t <= n_across; t++) {
        float saved[10][8];

        if (dir == 0) {
            // slice = column t
            for (int i = first - 1; i <= last; i++)
                for (int d = 0; d < dim; d++)
                    saved[i - (first - 1)][d] = (*Ps)[i][t][d];
            for (int i = old_n; i > last; i--)
                for (int d = 0; d < dim; d++)
                    (*Ps)[i + 1][t][d] = (*Ps)[i][t][d];
            for (int i = last + 1; i >= first; i--) {
                if (alpha[i] == 1.0f) continue;
                if (alpha[i] == 0.0f) {
                    for (int d = 0; d < dim; d++)
                        (*Ps)[i][t][d] = (*Ps)[i - 1][t][d];
                    continue;
                }
                int sc = i - (first - 1), sp = i - 1 - (first - 1);
                for (int d = 0; d < dim; d++)
                    (*Ps)[i][t][d] = alpha[i] * saved[sc][d]
                                   + (1.0f - alpha[i]) * saved[sp][d];
            }
        } else {
            // slice = row t
            for (int j = first - 1; j <= last; j++)
                for (int d = 0; d < dim; d++)
                    saved[j - (first - 1)][d] = (*Ps)[t][j][d];
            for (int j = old_n; j > last; j--)
                for (int d = 0; d < dim; d++)
                    (*Ps)[t][j + 1][d] = (*Ps)[t][j][d];
            for (int j = last + 1; j >= first; j--) {
                if (alpha[j] == 1.0f) continue;
                if (alpha[j] == 0.0f) {
                    for (int d = 0; d < dim; d++)
                        (*Ps)[t][j][d] = (*Ps)[t][j - 1][d];
                    continue;
                }
                int sc = j - (first - 1), sp = j - 1 - (first - 1);
                for (int d = 0; d < dim; d++)
                    (*Ps)[t][j][d] = alpha[j] * saved[sc][d]
                                   + (1.0f - alpha[j]) * saved[sp][d];
            }
        }
    }
}
