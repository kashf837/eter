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

void knot_insert(float** U, int* m, float*** P, int* n, int p,
                    float u_bar, int k) {
    int m_old = *m, n_old = *n;
    int first = k - p + 1;
    int last  = k;

    // save old knot vector, needed for alpha computation
    float U_old_snap[20];
    for (int i = 0; i <= m_old; i++) U_old_snap[i] = (*U)[i];

    // save P[first-1] through P[last] (p+1 old values)
    float saved[10][2];
    for (int i = first - 1; i <= last; i++) {
        int si = i - (first - 1);
        saved[si][0] = (*P)[i][0];
        saved[si][1] = (*P)[i][1];
    }

    // expand knot vector
    (*m)++;
    *U = (float*)realloc(*U, (*m + 1) * sizeof(float));
    for (int i = m_old; i > k; i--)
        (*U)[i + 1] = (*U)[i];
    (*U)[k + 1] = u_bar;

    // expand control point array
    (*n)++;
    *P = (float**)realloc(*P, (*n + 1) * sizeof(float*));
    (*P)[*n] = (float*)malloc(2 * sizeof(float));

    // shift right: indices after 'last' move by 1
    for (int i = n_old; i > last; i--) {
        (*P)[i + 1][0] = (*P)[i][0];
        (*P)[i + 1][1] = (*P)[i][1];
    }

    // compute Q[first..last+1] backwards to avoid overwrite
    for (int i = last + 1; i >= first; i--) {
        float alpha;
        if (i <= k - p)
            { (*P)[i][0] = (*P)[i][0]; (*P)[i][1] = (*P)[i][1]; continue; }
        else if (i >= k + 1)
            { (*P)[i][0] = (*P)[i - 1][0]; (*P)[i][1] = (*P)[i - 1][1]; continue; }
        else
            alpha = (u_bar - U_old_snap[i]) / (U_old_snap[i + p] - U_old_snap[i]);

        // i is in the middle range, read old P[i] and P[i-1] from saved
        int si_cur  = i     - (first - 1);
        int si_prev = i - 1 - (first - 1);

        (*P)[i][0] = alpha * saved[si_cur][0] + (1.0f - alpha) * saved[si_prev][0];
        (*P)[i][1] = alpha * saved[si_cur][1] + (1.0f - alpha) * saved[si_prev][1];
    }
}

void free_cp(float** P, int n) {
    for (int i = 0; i <= n; i++) free(P[i]);
    free(P);
}

int main() {
    int p = 2, m = 6, n = 3;

    float* U = (float*)malloc((m + 1) * sizeof(float));
    float U_init[] = {0,0,0,1,2,2,2};
    for (int i = 0; i <= m; i++) U[i] = U_init[i];

    float** P = (float**)malloc((n + 1) * sizeof(float*));
    for (int i = 0; i <= n; i++) P[i] = (float*)malloc(2 * sizeof(float));
    float P_init[4][2] = {{0,0}, {1,2}, {2,-1}, {3,0}};
    for (int i = 0; i <= n; i++) {
        P[i][0] = P_init[i][0];
        P[i][1] = P_init[i][1];
    }

    float u_target = 1.0;
    while (1) {
        int s = multiplicity(U, m, u_target);
        printf("s=%d  m=%d  n=%d\n", s, m, n);
        if (s >= p + 1) break;

        int k = find_span(U, m, u_target);
        knot_insert(&U, &m, &P, &n, p, u_target, k);
    }

    printf("\nKnots: ");
    for (int i = 0; i <= m; i++) printf("%.1f ", U[i]);
    printf("\nCPs:\n");
    for (int i = 0; i <= n; i++)
        printf("  Q%d = (%.3f, %.3f)\n", i, P[i][0], P[i][1]);

    free(U); free_cp(P, n);
    return 0;
}
