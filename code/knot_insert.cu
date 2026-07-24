#include <stdio.h>
#include <stdlib.h>

int find_span(float* U, int m, float u) {
    for (int i = 0; i < m; i++)
        if (u >= U[i] && u < U[i + 1]) return i;
    return -1;
}

int multiplicity(float* U, int m, float u) {
    int cnt = 0;
    for (int i = 0; i <= m; i++)
        if (U[i] == u) cnt++;
    return cnt;
}

// Curve knot insertion (in-place on U and P)
void knot_insert_curve(float** U, int* m, float*** P_curve, int* n, int p, int dim,
                        float u_bar, int k) {
    int m_old = *m, n_old = *n;
    int first = k - p + 1;
    int last  = k;

    float U_old_snap[20];
    for (int i = 0; i <= m_old; i++) U_old_snap[i] = (*U)[i];

    float saved[10][8];
    for (int i = first - 1; i <= last; i++) {
        int si = i - (first - 1);
        for (int d = 0; d < dim; d++)
            saved[si][d] = (*P_curve)[i][d];
    }

    (*m)++;
    *U = (float*)realloc(*U, (*m + 1) * sizeof(float));
    for (int i = m_old; i > k; i--)
        (*U)[i + 1] = (*U)[i];
    (*U)[k + 1] = u_bar;

    (*n)++;
    *P_curve = (float**)realloc(*P_curve, (*n + 1) * sizeof(float*));
    (*P_curve)[*n] = (float*)malloc(dim * sizeof(float));

    for (int i = n_old; i > last; i--)
        for (int d = 0; d < dim; d++)
            (*P_curve)[i + 1][d] = (*P_curve)[i][d];

    for (int i = last + 1; i >= first; i--) {
        float alpha;
        if (i <= k - p)
            continue;
        else if (i >= k + 1) {
            for (int d = 0; d < dim; d++)
                (*P_curve)[i][d] = (*P_curve)[i - 1][d];
            continue;
        } else {
            alpha = (u_bar - U_old_snap[i]) / (U_old_snap[i + p] - U_old_snap[i]);
        }

        int si_cur  = i     - (first - 1);
        int si_prev = i - 1 - (first - 1);
        for (int d = 0; d < dim; d++)
            (*P_curve)[i][d] = alpha * saved[si_cur][d] + (1.0f - alpha) * saved[si_prev][d];
    }
}

// Surface knot insertion in u-direction
// P_surf[nu+1][nv+1], each CP has dim floats
void knot_insert_surf_u(float** U, int* m_u,
                         float**** P_surf, int* n_u, int n_v, int p_u, int dim,
                         float u_bar, int k) {
    int m_old = *m_u, n_old = *n_u;
    int first = k - p_u + 1;
    int last  = k;

    // snapshot old U for alpha
    float U_old_snap[20];
    for (int i = 0; i <= m_old; i++) U_old_snap[i] = (*U)[i];

    // precompute alpha for each row
    float alpha[20];
    for (int i = 0; i <= n_old + 1; i++) {
        if (i <= k - p_u)       alpha[i] = 1.0f;
        else if (i >= k + 1)    alpha[i] = 0.0f;
        else alpha[i] = (u_bar - U_old_snap[i]) / (U_old_snap[i + p_u] - U_old_snap[i]);
    }

    // expand U
    (*m_u)++;
    *U = (float*)realloc(*U, (*m_u + 1) * sizeof(float));
    for (int i = m_old; i > k; i--)
        (*U)[i + 1] = (*U)[i];
    (*U)[k + 1] = u_bar;

    // expand surface: add one row
    (*n_u)++;
    *P_surf = (float***)realloc(*P_surf, (*n_u + 1) * sizeof(float**));
    (*P_surf)[*n_u] = (float**)malloc((n_v + 1) * sizeof(float*));
    for (int j = 0; j <= n_v; j++)
        (*P_surf)[*n_u][j] = (float*)malloc(dim * sizeof(float));

    // compute new control points for each column
    for (int j = 0; j <= n_v; j++) {
        // save old column values
        float saved[10][8];
        for (int i = first - 1; i <= last; i++) {
            int si = i - (first - 1);
            for (int d = 0; d < dim; d++)
                saved[si][d] = (*P_surf)[i][j][d];
        }

        // shift right
        for (int i = n_old; i > last; i--)
            for (int d = 0; d < dim; d++)
                (*P_surf)[i + 1][j][d] = (*P_surf)[i][j][d];

        // compute backwards
        for (int i = last + 1; i >= first; i--) {
            if (alpha[i] == 1.0f) continue;
            if (alpha[i] == 0.0f) {
                for (int d = 0; d < dim; d++)
                    (*P_surf)[i][j][d] = (*P_surf)[i - 1][j][d];
                continue;
            }
            int si_cur  = i     - (first - 1);
            int si_prev = i - 1 - (first - 1);
            for (int d = 0; d < dim; d++)
                (*P_surf)[i][j][d] = alpha[i] * saved[si_cur][d]
                                   + (1.0f - alpha[i]) * saved[si_prev][d];
        }
    }
}

// Surface knot insertion in v-direction
void knot_insert_surf_v(float** V, int* m_v,
                         float**** P_surf, int n_u, int* n_v, int p_v, int dim,
                         float v_bar, int k) {
    int m_old = *m_v, n_old = *n_v;
    int first = k - p_v + 1;
    int last  = k;

    float V_old_snap[20];
    for (int i = 0; i <= m_old; i++) V_old_snap[i] = (*V)[i];

    float alpha[20];
    for (int i = 0; i <= n_old + 1; i++) {
        if (i <= k - p_v)       alpha[i] = 1.0f;
        else if (i >= k + 1)    alpha[i] = 0.0f;
        else alpha[i] = (v_bar - V_old_snap[i]) / (V_old_snap[i + p_v] - V_old_snap[i]);
    }

    (*m_v)++;
    *V = (float*)realloc(*V, (*m_v + 1) * sizeof(float));
    for (int i = m_old; i > k; i--)
        (*V)[i + 1] = (*V)[i];
    (*V)[k + 1] = v_bar;

    // add one column to each row
    (*n_v)++;
    for (int i = 0; i <= n_u; i++) {
        (*P_surf)[i] = (float**)realloc((*P_surf)[i], (*n_v + 1) * sizeof(float*));
        (*P_surf)[i][*n_v] = (float*)malloc(dim * sizeof(float));
    }

    for (int i = 0; i <= n_u; i++) {
        float saved[10][8];
        for (int j = first - 1; j <= last; j++) {
            int si = j - (first - 1);
            for (int d = 0; d < dim; d++)
                saved[si][d] = (*P_surf)[i][j][d];
        }
        for (int j = n_old; j > last; j--)
            for (int d = 0; d < dim; d++)
                (*P_surf)[i][j + 1][d] = (*P_surf)[i][j][d];

        for (int j = last + 1; j >= first; j--) {
            if (alpha[j] == 1.0f) continue;
            if (alpha[j] == 0.0f) {
                for (int d = 0; d < dim; d++)
                    (*P_surf)[i][j][d] = (*P_surf)[i][j - 1][d];
                continue;
            }
            int sj_cur  = j     - (first - 1);
            int sj_prev = j - 1 - (first - 1);
            for (int d = 0; d < dim; d++)
                (*P_surf)[i][j][d] = alpha[j] * saved[sj_cur][d]
                                   + (1.0f - alpha[j]) * saved[sj_prev][d];
        }
    }
}

void free_surf(float*** P, int n_u, int n_v) {
    for (int i = 0; i <= n_u; i++) {
        for (int j = 0; j <= n_v; j++) free(P[i][j]);
        free(P[i]);
    }
    free(P);
}

void print_surf(float*** P, int n_u, int n_v, int dim) {
    for (int i = 0; i <= n_u; i++) {
        printf("  row %d: ", i);
        for (int j = 0; j <= n_v; j++) {
            printf("(");
            for (int d = 0; d < dim; d++)
                printf("%.1f%s", P[i][j][d], d < dim - 1 ? "," : "");
            printf(") ");
        }
        printf("\n");
    }
}

int main() {
    int p = 2, dim = 2;

    // Curve test
    printf("=== Curve Knot Insertion ===\n");
    {
        int m = 6, n = 3;
        float* U = (float*)malloc((m + 1) * sizeof(float));
        float U_init[] = {0,0,0,1,2,2,2};
        for (int i = 0; i <= m; i++) U[i] = U_init[i];

        float** P = (float**)malloc((n + 1) * sizeof(float*));
        float P_init[4][2] = {{0,0}, {1,2}, {2,-1}, {3,0}};
        for (int i = 0; i <= n; i++) {
            P[i] = (float*)malloc(dim * sizeof(float));
            for (int d = 0; d < dim; d++) P[i][d] = P_init[i][d];
        }

        float u_target = 1.0;
        while (1) {
            int s = multiplicity(U, m, u_target);
            if (s >= p + 1) break;
            int k = find_span(U, m, u_target);
            knot_insert_curve(&U, &m, &P, &n, p, dim, u_target, k);
        }
        printf("Knots: ");
        for (int i = 0; i <= m; i++) printf("%.1f ", U[i]);
        printf("\nCPs (%d):\n", n + 1);
        for (int i = 0; i <= n; i++)
            printf("  Q%d=(%.1f,%.1f)\n", i, P[i][0], P[i][1]);

        for (int i = 0; i <= n; i++) free(P[i]);
        free(P); free(U);
    }

    // Surface test
    printf("\n=== Surface Knot Insertion ===\n");
    {
        int p_u = 2, p_v = 2;
        int m_u = 6, m_v = 6;
        int n_u = 3, n_v = 3;  // 4x4 CP grid

        float* U = (float*)malloc((m_u + 1) * sizeof(float));
        float* V = (float*)malloc((m_v + 1) * sizeof(float));
        float U_init[] = {0,0,0,1,2,2,2};
        float V_init[] = {0,0,0,1,2,2,2};
        for (int i = 0; i <= m_u; i++) U[i] = U_init[i];
        for (int i = 0; i <= m_v; i++) V[i] = V_init[i];

        // 4x4 CP grid, each CP is (x, y)
        float*** Ps = (float***)malloc((n_u + 1) * sizeof(float**));
        for (int i = 0; i <= n_u; i++) {
            Ps[i] = (float**)malloc((n_v + 1) * sizeof(float*));
            for (int j = 0; j <= n_v; j++) {
                Ps[i][j] = (float*)malloc(dim * sizeof(float));
                Ps[i][j][0] = (float)i + (float)j * 0.5f;
                Ps[i][j][1] = (float)i - (float)j;
            }
        }

        printf("Before: n_u=%d n_v=%d\n", n_u, n_v);
        printf("U: "); for (int i = 0; i <= m_u; i++) printf("%.1f ", U[i]);
        printf("\nV: "); for (int i = 0; i <= m_v; i++) printf("%.1f ", V[i]);
        printf("\n");

        // Insert u=1.0 until multiplicity reaches p_u+1
        float u_target = 1.0;
        while (1) {
            int s = multiplicity(U, m_u, u_target);
            if (s >= p_u + 1) break;
            int k = find_span(U, m_u, u_target);
            knot_insert_surf_u(&U, &m_u, &Ps, &n_u, n_v, p_u, dim, u_target, k);
            printf("  u-insert: n_u=%d\n", n_u);
        }

        // Insert v=1.0 until multiplicity reaches p_v+1
        float v_target = 1.0;
        while (1) {
            int s = multiplicity(V, m_v, v_target);
            if (s >= p_v + 1) break;
            int k = find_span(V, m_v, v_target);
            knot_insert_surf_v(&V, &m_v, &Ps, n_u, &n_v, p_v, dim, v_target, k);
            printf("  v-insert: n_v=%d\n", n_v);
        }

        printf("\nAfter: n_u=%d n_v=%d\n", n_u, n_v);
        printf("U: "); for (int i = 0; i <= m_u; i++) printf("%.1f ", U[i]);
        printf("\nV: "); for (int i = 0; i <= m_v; i++) printf("%.1f ", V[i]);
        printf("\nCP grid (%dx%d):\n", n_u + 1, n_v + 1);
        print_surf(Ps, n_u, n_v, dim);

        free_surf(Ps, n_u, n_v);
        free(U); free(V);
    }

    return 0;
}
