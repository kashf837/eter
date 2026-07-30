#ifndef KNOT_INSERT_H
#define KNOT_INSERT_H

#include <stdlib.h>

// ========== Type aliases ==========
typedef float*  CP;          // single control point:  dim floats
typedef CP*     CPArray;     // 1D array of control points
typedef CPArray* CPGrid;     // 2D grid (rows x cols) of control points
typedef float*  KnotVector;  // knot values array (reallocated by knot insertion)

// ========== Memory helpers ==========
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
int find_span(KnotVector U, int m, float u);
int mult_of(KnotVector U, int m, float u);

// ========== Surface types ==========

typedef struct {
    int p_u, p_v;           // degree in u, v
    int n_u, n_v;           // CP count minus 1 in u, v
    int m_u, m_v;           // knot count minus 1 (= n + p + 1)
    KnotVector U, V;        // knot vectors (may be reallocated)
    CPGrid Ps;              // control point grid [n_u+1][n_v+1]
} NurbsSurf;

typedef struct {
    int p_u, p_v;           // degree (= CP count minus 1 for Bezier)
    float* CP;              // flat array: (p_u+1)*(p_v+1)*4 floats
} BezierPatch;

// ========== Surface knot insertion ==========
// dir=0: insert into U (affects rows), dir=1: insert into V (affects columns)
// x = knot value to insert; finds k automatically
void knot_insert_surf(NurbsSurf* s, int dir, float x);

#endif
