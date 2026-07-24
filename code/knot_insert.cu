#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ========== Type aliases (clean up the pointer soup) ==========
typedef float*  CP;        // single control point:  dim floats
typedef CP*     CPArray;   // 1D array of control points
typedef CPArray* CPGrid;   // 2D grid (rows x cols) of control points

// ========== Knot vector helpers ==========

int find_span(float* U, int m, float u) {
    for (int i = 0; i < m; i++)
        if (u >= U[i] && u < U[i + 1]) return i;
    return -1;
}

int mult_of(float* U, int m, float u) {
    int cnt = 0;
    for (int i = 0; i <= m; i++)
        if (U[i] == u) cnt++;
    return cnt;
}

// Insert u_bar into U, expand by 1. Returns new U, *m += 1.
static void expand_knot(float** U, int* m, int k, float u_bar) {
    int old_m = *m;
    (*m)++;
    *U = (float*)realloc(*U, (*m + 1) * sizeof(float));
    for (int i = old_m; i > k; i--)
        (*U)[i + 1] = (*U)[i];
    (*U)[k + 1] = u_bar;
}

// Precompute alpha[i] for i = 0..n+1 from old knot vector.
static void precompute_alpha(float* alpha, float* U_old, int n_old,
                              int p, float u_bar, int k) {
    for (int i = 0; i <= n_old + 1; i++) {
        if      (i <= k - p)  alpha[i] = 1.0f;
        else if (i >= k + 1)  alpha[i] = 0.0f;
        else alpha[i] = (u_bar - U_old[i]) / (U_old[i + p] - U_old[i]);
    }
}

// ========== Memory helpers ==========

static CP new_cp(int dim) {
    CP p = (float*)malloc(dim * sizeof(float));
    return p;
}

static CPArray new_cparray(int n, int dim) {
    CPArray arr = (CPArray)malloc((n + 1) * sizeof(CP));
    for (int i = 0; i <= n; i++) arr[i] = new_cp(dim);
    return arr;
}

static CPGrid new_cpgrid(int nr, int nc, int dim) {
    CPGrid g = (CPGrid)malloc((nr + 1) * sizeof(CPArray));
    for (int i = 0; i <= nr; i++)
        g[i] = new_cparray(nc, dim);
    return g;
}

static void free_cparray(CPArray arr, int n) {
    for (int i = 0; i <= n; i++) free(arr[i]);
    free(arr);
}

static void free_cpgrid(CPGrid g, int nr, int nc) {
    for (int i = 0; i <= nr; i++) free_cparray(g[i], nc);
    free(g);
}

// ========== Curve knot insertion ==========

void knot_insert_curve(float** U, int* m, CPArray* P, int* n,
                        int p, int dim, float u_bar, int k) {
    int old_m = *m, old_n = *n;
    int first  = k - p + 1;   // first affected CP index
    int last   = k;           // last  affected CP index

    // snapshot old knot vector (needed for alpha)
    float U_old[20];
    memcpy(U_old, *U, (old_m + 1) * sizeof(float));

    // save old CPs in [first-1 .. last]
    float saved[10][8];
    for (int i = first - 1; i <= last; i++)
        for (int d = 0; d < dim; d++)
            saved[i - (first - 1)][d] = (*P)[i][d];

    expand_knot(U, m, k, u_bar);

    // expand CP array
    (*n)++;
    *P = (CPArray)realloc(*P, (*n + 1) * sizeof(CP));
    (*P)[*n] = new_cp(dim);

    // shift un-affected right side by 1
    for (int i = old_n; i > last; i--)
        for (int d = 0; d < dim; d++)
            (*P)[i + 1][d] = (*P)[i][d];

    // compute new CPs in [first .. last+1], backwards
    for (int i = last + 1; i >= first; i--) {
        float alpha;
        if      (i <= k - p)  continue;                      // Q = P (already OK)
        else if (i >= k + 1)  alpha = 0.0f;
        else alpha = (u_bar - U_old[i]) / (U_old[i + p] - U_old[i]);

        if (alpha == 0.0f) {
            for (int d = 0; d < dim; d++)
                (*P)[i][d] = (*P)[i - 1][d];
            continue;
        }

        int si_cur  = i     - (first - 1);
        int si_prev = i - 1 - (first - 1);
        for (int d = 0; d < dim; d++)
            (*P)[i][d] = alpha * saved[si_cur][d] + (1.0f - alpha) * saved[si_prev][d];
    }
}

// ========== Surface knot insertion ==========

// Insert u_bar in u-direction (affects rows, one column at a time)
void knot_insert_surf_u(float** U, int* m_u, CPGrid* Ps, int* n_u, int n_v,
                         int p_u, int dim, float u_bar, int k) {
    int old_m = *m_u, old_n = *n_u;
    int first = k - p_u + 1;
    int last  = k;

    float U_old[20];
    memcpy(U_old, *U, (old_m + 1) * sizeof(float));

    float alpha[20];
    precompute_alpha(alpha, U_old, old_n, p_u, u_bar, k);

    expand_knot(U, m_u, k, u_bar);

    // add one row to the grid
    (*n_u)++;
    *Ps = (CPGrid)realloc(*Ps, (*n_u + 1) * sizeof(CPArray));
    (*Ps)[*n_u] = new_cparray(n_v, dim);

    // process each column independently
    for (int j = 0; j <= n_v; j++) {
        float saved[10][8];
        for (int i = first - 1; i <= last; i++)
            for (int d = 0; d < dim; d++)
                saved[i - (first - 1)][d] = (*Ps)[i][j][d];

        for (int i = old_n; i > last; i--)
            for (int d = 0; d < dim; d++)
                (*Ps)[i + 1][j][d] = (*Ps)[i][j][d];

        for (int i = last + 1; i >= first; i--) {
            if (alpha[i] == 1.0f) continue;
            if (alpha[i] == 0.0f) {
                for (int d = 0; d < dim; d++)
                    (*Ps)[i][j][d] = (*Ps)[i - 1][j][d];
                continue;
            }
            int si_cur  = i     - (first - 1);
            int si_prev = i - 1 - (first - 1);
            for (int d = 0; d < dim; d++)
                (*Ps)[i][j][d] = alpha[i] * saved[si_cur][d]
                               + (1.0f - alpha[i]) * saved[si_prev][d];
        }
    }
}

// Insert v_bar in v-direction (affects columns, one row at a time)
void knot_insert_surf_v(float** V, int* m_v, CPGrid* Ps, int n_u, int* n_v,
                         int p_v, int dim, float v_bar, int k) {
    int old_m = *m_v, old_n = *n_v;
    int first = k - p_v + 1;
    int last  = k;

    float V_old[20];
    memcpy(V_old, *V, (old_m + 1) * sizeof(float));

    float alpha[20];
    precompute_alpha(alpha, V_old, old_n, p_v, v_bar, k);

    expand_knot(V, m_v, k, v_bar);

    // add one column to every row
    (*n_v)++;
    for (int i = 0; i <= n_u; i++) {
        (*Ps)[i] = (CPArray)realloc((*Ps)[i], (*n_v + 1) * sizeof(CP));
        (*Ps)[i][*n_v] = new_cp(dim);
    }

    // process each row independently
    for (int i = 0; i <= n_u; i++) {
        float saved[10][8];
        for (int j = first - 1; j <= last; j++)
            for (int d = 0; d < dim; d++)
                saved[j - (first - 1)][d] = (*Ps)[i][j][d];

        for (int j = old_n; j > last; j--)
            for (int d = 0; d < dim; d++)
                (*Ps)[i][j + 1][d] = (*Ps)[i][j][d];

        for (int j = last + 1; j >= first; j--) {
            if (alpha[j] == 1.0f) continue;
            if (alpha[j] == 0.0f) {
                for (int d = 0; d < dim; d++)
                    (*Ps)[i][j][d] = (*Ps)[i][j - 1][d];
                continue;
            }
            int sj_cur  = j     - (first - 1);
            int sj_prev = j - 1 - (first - 1);
            for (int d = 0; d < dim; d++)
                (*Ps)[i][j][d] = alpha[j] * saved[sj_cur][d]
                               + (1.0f - alpha[j]) * saved[sj_prev][d];
        }
    }
}

// ========== Print helpers ==========

static void print_cp_array(CPArray arr, int n, int dim) {
    for (int i = 0; i <= n; i++) {
        printf("  [%d] (", i);
        for (int d = 0; d < dim; d++)
            printf("%.1f%s", arr[i][d], d < dim - 1 ? "," : "");
        printf(")\n");
    }
}

static void print_cp_grid(CPGrid g, int nr, int nc, int dim) {
    for (int i = 0; i <= nr; i++) {
        printf("  row %d: ", i);
        for (int j = 0; j <= nc; j++) {
            printf("(");
            for (int d = 0; d < dim; d++)
                printf("%.1f%s", g[i][j][d], d < dim - 1 ? "," : "");
            printf(") ");
        }
        printf("\n");
    }
}

// ========== Main (curve + surface demos) ==========

static void demo_curve(void) {
    printf("=== NURBS Curve ===\n");
    int p = 2, dim = 4;

    int m = 6, n = 3;
    float* U = (float*)malloc((m + 1) * sizeof(float));
    float U_init[] = {0,0,0,1,2,2,2};
    memcpy(U, U_init, (m + 1) * sizeof(float));

    CPArray P = new_cparray(n, dim);
    // (x, y, 0, w) — 2D NURBS curve in z=0 plane
    float init[4][4] = {
        {0,0, 0,1}, {2,4, 0,2}, {4,-2, 0,2}, {6,0, 0,1}
    };
    for (int i = 0; i <= n; i++)
        memcpy(P[i], init[i], dim * sizeof(float));

    float u = 1.0;
    while (mult_of(U, m, u) < p + 1) {
        int k = find_span(U, m, u);
        knot_insert_curve(&U, &m, &P, &n, p, dim, u, k);
    }

    printf("Knots: "); for (int i = 0; i <= m; i++) printf("%.1f ", U[i]);
    printf("\n%d CPs (x y z w):\n", n + 1);
    print_cp_array(P, n, dim);

    free_cparray(P, n); free(U);
}

static void demo_surface(void) {
    printf("\n=== NURBS Surface ===\n");
    int p = 2, dim = 4;

    int m_u = 6, m_v = 6, n_u = 3, n_v = 3;
    float* U = (float*)malloc((m_u + 1) * sizeof(float));
    float* V = (float*)malloc((m_v + 1) * sizeof(float));
    float U_init[] = {0,0,0,1,2,2,2};
    memcpy(U, U_init, (m_u + 1) * sizeof(float));
    memcpy(V, U_init, (m_v + 1) * sizeof(float));

    CPGrid Ps = new_cpgrid(n_u, n_v, dim);
    for (int i = 0; i <= n_u; i++)
        for (int j = 0; j <= n_v; j++) {
            Ps[i][j][0] = (float)i + (float)j * 0.5f;
            Ps[i][j][1] = (float)i - (float)j;
            Ps[i][j][2] = 0.0f;
            Ps[i][j][3] = ((i == 1 || i == 2) && (j == 1 || j == 2)) ? 2.0f : 1.0f;
        }

    printf("Before: %dx%d  dim=%d  U: ", n_u + 1, n_v + 1, dim);
    for (int i = 0; i <= m_u; i++) printf("%.1f ", U[i]);
    printf("\n");

    float u = 1.0;
    while (mult_of(U, m_u, u) < p + 1) {
        int k = find_span(U, m_u, u);
        knot_insert_surf_u(&U, &m_u, &Ps, &n_u, n_v, p, dim, u, k);
    }

    float v = 1.0;
    while (mult_of(V, m_v, v) < p + 1) {
        int k = find_span(V, m_v, v);
        knot_insert_surf_v(&V, &m_v, &Ps, n_u, &n_v, p, dim, v, k);
    }

    printf("After:  %dx%d  U: ", n_u + 1, n_v + 1);
    for (int i = 0; i <= m_u; i++) printf("%.1f ", U[i]);
    printf("  V: ");
    for (int i = 0; i <= m_v; i++) printf("%.1f ", V[i]);
    printf("\n");
    print_cp_grid(Ps, n_u, n_v, dim);

    free_cpgrid(Ps, n_u, n_v); free(U); free(V);
}

int main() {
    demo_curve();
    demo_surface();
    return 0;
}
