#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nurbs.h"
#include "knot_insert.h"
#include "bezier_eval.h"

// ============================================================
// .surf file I/O
// ============================================================

static void free_surf(NurbsSurf* s);

static int next_line(FILE* f, char* buf, int size) {
    while (fgets(buf, size, f))
        if (buf[0] != '#' && buf[0] != '\n') return 1;
    return 0;
}

static int count_floats(const char* line) {
    int n = 1;
    for (const char* c = line; *c; c++) if (*c == ' ') n++;
    return n;
}

static void parse_floats(char* line, float* out, int n) {
    char* p = line;
    for (int i = 0; i < n; i++) {
        sscanf(p, "%f", &out[i]);
        char* next = strchr(p, ' ');
        if (!next) break;
        p = next + 1;
    }
}

static NurbsSurf* read_surf(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "  [SKIP] Cannot open %s\n", path); return NULL; }
    NurbsSurf* s = (NurbsSurf*)calloc(1, sizeof(NurbsSurf));
    char line[4096];

    if (!next_line(f, line, sizeof(line))) { fclose(f); free(s); return NULL; }
    sscanf(line, "%d %d %d %d", &s->p_u, &s->p_v, &s->n_u, &s->n_v);
    if (s->p_u <= 0 || s->p_v <= 0 || s->n_u <= 0 || s->n_v <= 0 ||
        s->p_u > 20 || s->p_v > 20 || s->n_u > 10000 || s->n_v > 10000) {
        fclose(f); free(s); return NULL;
    }

    if (!next_line(f, line, sizeof(line))) { fclose(f); free_surf(s); return NULL; }
    {   int n = count_floats(line); s->m_u = n - 1;
        s->U = (float*)malloc(n * sizeof(float)); parse_floats(line, s->U, n); }

    if (!next_line(f, line, sizeof(line))) { fclose(f); free_surf(s); return NULL; }
    {   int n = count_floats(line); s->m_v = n - 1;
        s->V = (float*)malloc(n * sizeof(float)); parse_floats(line, s->V, n); }

    s->Ps = new_cpgrid(s->n_u, s->n_v, 4);
    for (int i = 0; i <= s->n_u; i++)
        for (int j = 0; j <= s->n_v; j++) {
            if (!next_line(f, line, sizeof(line))) { fclose(f); free_surf(s); return NULL; }
            int got = sscanf(line, "%f %f %f %f",
                   &s->Ps[i][j][0], &s->Ps[i][j][1],
                   &s->Ps[i][j][2], &s->Ps[i][j][3]);
            if (got < 1) { fclose(f); free_surf(s); return NULL; }
            if (got < 3) { s->Ps[i][j][1] = 0; s->Ps[i][j][2] = 0; }
            if (got < 4) s->Ps[i][j][3] = 1.0f;
        }
    fclose(f);
    return s;
}

static void free_surf(NurbsSurf* s) {
    free_cpgrid(s->Ps, s->n_u, s->n_v);
    free(s->U); free(s->V); free(s);
}

// ============================================================
// NURBS → Bézier decomposition
// ============================================================

static void nurbs_to_beziers(NurbsSurf* s) {
    for (int i = s->p_u + 1; i <= s->m_u - s->p_u - 1; i++) {
        float u = s->U[i];
        while (mult_of(s->U, s->m_u, u) < s->p_u + 1)
            knot_insert_surf(s, 0, u);
    }
    for (int i = s->p_v + 1; i <= s->m_v - s->p_v - 1; i++) {
        float v = s->V[i];
        while (mult_of(s->V, s->m_v, v) < s->p_v + 1)
            knot_insert_surf(s, 1, v);
    }
}

// ============================================================
// Per-patch adaptive evaluation (ETER §4)
// ============================================================

// Core of adaptive evaluation: takes pre-converted Bézier CPs (flat array,
// patch-major), computes per-patch tess factors and evaluates vertices.
// Shared by nurbs_adaptive_eval (fresh load) and nurbs_adaptive_eval_cached
// (camera rebuild — no file I/O, no knot insertion).
static float* adaptive_eval_core(const float* cps, int nu, int nv, int ru, int rv,
                                 const CameraParams* cam,
                                 PatchInfo** info_out, int* total_verts)
{
    int np = nu * nv;
    int cpsPerPatch = ru * rv * 4;  // e.g. 64 for bicubic

    PatchInfo* info = (PatchInfo*)malloc(np * sizeof(PatchInfo));
    BezierPatch bp = {ru - 1, rv - 1, (float*)malloc(cpsPerPatch * sizeof(float))};
    int offset = 0;

    // First pass: compute tess factors and offsets
    for (int i = 0; i < nu; i++) {
        for (int j = 0; j < nv; j++) {
            for (int ii = 0; ii < ru; ii++)
                for (int jj = 0; jj < rv; jj++)
                    memcpy(&bp.CP[(ii*rv+jj)*4],
                           cps + ((i*nv+j)*ru*rv + ii*rv+jj)*4, 4*sizeof(float));

            float eps = compute_epsilon(&bp, cam);
            int tu = 0, tv = 0;
            compute_tess_factors(&bp, eps, &tu, &tv);
            adapt_tess_factors(&tu, &tv);

            int idx = i*nv+j;
            info[idx].tu = tu; info[idx].tv = tv; info[idx].offset = offset;
            offset += tu * tv;
        }
    }

    // Second pass: batch-evaluate every patch at its own resolution in ONE
    // GPU round trip (one launch for the whole surface). The old per-patch
    // eval_bezier_patch made 13 CUDA API calls per patch (~680µs each on
    // WDDM = 39s for 57k patches); this is one cudaMalloc group + one H2D +
    // one launch + one D2H per surface. The kernel writes v-major directly
    // (index = offset + v*tu + u), so the old per-patch transpose loop is
    // gone too.
    float* verts = (float*)malloc(offset * 6 * sizeof(float));
    eval_bezier_patches_batched(cps, info, np, ru, rv, verts, offset);

    free(bp.CP);
    *info_out = info;
    *total_verts = offset;

    return verts;
}

float* nurbs_adaptive_eval(const char* surf_path, const CameraParams* cam,
                           int* np_u, int* np_v, PatchInfo** info_out, int* total_verts,
                           int* ru_out, int* rv_out, float** cp_out)
{
    NurbsSurf* s = read_surf(surf_path);
    if (!s) return NULL;

    nurbs_to_beziers(s);

    int ru = s->p_u + 1, rv = s->p_v + 1;
    int nu = (s->n_u + 1) / ru, nv = (s->n_v + 1) / rv;
    *np_u = nu; *np_v = nv;
    *ru_out = ru; *rv_out = rv;
    int np = nu * nv;
    int cpsPerPatch = ru * rv * 4;  // e.g. 64 for bicubic

    // Extract all patches' CPs into one flat array (uncentered) up front.
    // If the caller wants them (cp_out), hand over this allocation; otherwise
    // free it after the core finishes. The core only reads this array.
    float* cps;
    if (cp_out) { *cp_out = (float*)malloc(np * cpsPerPatch * sizeof(float)); cps = *cp_out; }
    else        { cps = (float*)malloc(np * cpsPerPatch * sizeof(float)); }
    for (int i = 0; i < nu; i++)
        for (int j = 0; j < nv; j++)
            for (int ii = 0; ii < ru; ii++)
                for (int jj = 0; jj < rv; jj++)
                    memcpy(&cps[((i*nv+j)*ru*rv + ii*rv+jj)*4],
                           s->Ps[i*ru+ii][j*rv+jj], 4*sizeof(float));
    free_surf(s);

    float* verts = adaptive_eval_core(cps, nu, nv, ru, rv, cam, info_out, total_verts);
    if (!cp_out) free(cps);
    return verts;
}

float* nurbs_adaptive_eval_cached(const float* cps, int nu, int nv, int ru, int rv,
                                  const CameraParams* cam,
                                  PatchInfo** info_out, int* total_verts)
{
    if (!cps || nu <= 0 || nv <= 0 || ru <= 1 || rv <= 1) return NULL;
    return adaptive_eval_core(cps, nu, nv, ru, rv, cam, info_out, total_verts);
}
