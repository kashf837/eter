#ifndef KNOT_INSERT_H
#define KNOT_INSERT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ========== Type aliases ==========
typedef float*  CP;        // single control point: dim floats
typedef CP*     CPArray;   // 1D array of control points
typedef CPArray* CPGrid;   // 2D grid (rows x cols) of control points

// ========== Memory ==========
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

static void free_cparray(CPArray arr, int n) {
    for (int i = 0; i <= n; i++) free(arr[i]);
    free(arr);
}

static void free_cpgrid(CPGrid g, int nr, int nc) {
    for (int i = 0; i <= nr; i++) free_cparray(g[i], nc);
    free(g);
}

// ========== Knot vector helpers ==========
int find_span(float* U, int m, float u);
int mult_of(float* U, int m, float u);

// ========== Surface types ==========

typedef struct {
    int p_u, p_v;         // degree in u, v
    int n_u, n_v;         // CP count minus 1 in u, v (= p_u, p_v for a Bezier patch)
    int m_u, m_v;         // knot count minus 1 (= n + p + 1)
    float* U, *V;         // knot vectors
    CPGrid Ps;            // control point grid [n_u+1][n_v+1]
} NurbsSurf;

typedef struct {
    int p_u, p_v;         // degree
    int n_u, n_v;         // CP count minus 1
    float* CP;            // flat array: (n_u+1)*(n_v+1)*4 floats
} BezierPatch;

// ========== Knot insertion (curve + surface) ==========
void knot_insert_curve(float** U, int* m, CPArray* P, int* n,
                       int p, int dim, float u_bar, int k);

void knot_insert_surf_u(float** U, int* m_u, CPGrid* Ps, int* n_u, int n_v,
                         int p_u, int dim, float u_bar, int k);

void knot_insert_surf_v(float** V, int* m_v, CPGrid* Ps, int n_u, int* n_v,
                         int p_v, int dim, float v_bar, int k);

#endif
