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
// Single-surface path (loadFile); the scene path nurbs_adaptive_eval_scene
// evaluates many surfaces in ONE launch and is the rebuild workhorse.
static float* adaptive_eval_core(const float* cps, int nu, int nv, int ru, int rv,
                                 const CameraParams* cam,
                                 ScenePatchMeta** info_out, int* total_verts)
{
    int np = nu * nv;
    int cpsPerPatch = ru * rv * 4;  // e.g. 64 for bicubic

    ScenePatchMeta* info = (ScenePatchMeta*)malloc(np * sizeof(ScenePatchMeta));
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
            info[idx].ru = ru; info[idx].rv = rv; info[idx].cpOff = idx * ru * rv;
            offset += tu * tv;
        }
    }

    // Second pass: batch-evaluate every patch at its own resolution in ONE
    // GPU round trip. The old per-patch eval_bezier_patch made 13 CUDA API
    // calls per patch (~680µs each on WDDM = 39s for 57k patches); the scene
    // wrapper needs only 2 H2D + 1 launch + 1 D2H. The kernel writes v-major
    // directly (index = offset + v*tu + u), so the old per-patch transpose
    // loop is gone too.
    float* verts = (float*)malloc(offset * 6 * sizeof(float));
    eval_bezier_patches_scene(cps, info, np, verts, offset);

    free(bp.CP);
    *info_out = info;
    *total_verts = offset;

    return verts;
}

float* nurbs_adaptive_eval(const char* surf_path, const CameraParams* cam,
                           int* np_u, int* np_v, ScenePatchMeta** info_out, int* total_verts,
                           int* ru_out, int* rv_out, float** cp_out)
{
    int nu, nv, ru, rv;
    float* cps = nurbs_read_cps(surf_path, &nu, &nv, &ru, &rv);
    if (!cps) return NULL;

    *np_u = nu; *np_v = nv; *ru_out = ru; *rv_out = rv;
    if (cp_out) *cp_out = cps;   // hand over the CP array; the core only reads it

    float* verts = adaptive_eval_core(cps, nu, nv, ru, rv, cam, info_out, total_verts);
    if (!cp_out) free(cps);
    return verts;
}

// Read + decompose one .surf without evaluating: NURBS→Bézier knot insertion,
// flat uncentered CP extraction. The scene path (directory loads) calls this
// for every file first, then evaluates everything in one launch.
float* nurbs_read_cps(const char* path, int* nu, int* nv, int* ru, int* rv)
{
    NurbsSurf* s = read_surf(path);
    if (!s) return NULL;

    int r_u = s->p_u + 1, r_v = s->p_v + 1;
    int n_u = (s->n_u + 1) / r_u, n_v = (s->n_v + 1) / r_v;
    if (n_u <= 0 || n_v <= 0) { free_surf(s); return NULL; }
    // The eval kernel's shared Bernstein tables cap degrees at 7 (MAX_ORDER).
    if (r_u - 1 > 7 || r_v - 1 > 7) {
        fprintf(stderr, "  [SKIP] %s: degree (%d,%d) > 7 not supported\n", path, r_u - 1, r_v - 1);
        free_surf(s); return NULL;
    }
    int np = n_u * n_v;
    int cpsPerPatch = r_u * r_v * 4;  // e.g. 64 for bicubic

    float* cps = (float*)malloc((size_t)np * cpsPerPatch * sizeof(float));
    for (int i = 0; i < n_u; i++)
        for (int j = 0; j < n_v; j++)
            for (int ii = 0; ii < r_u; ii++)
                for (int jj = 0; jj < r_v; jj++)
                    memcpy(&cps[((i*n_v+j)*r_u*r_v + ii*r_v+jj)*4],
                           s->Ps[i*r_u+ii][j*r_v+jj], 4*sizeof(float));
    free_surf(s);

    *nu = n_u; *nv = n_v; *ru = r_u; *rv = r_v;
    return cps;
}

// Scene-level adaptive evaluation: one pass-1 (host tess factors, OpenMP-
// parallel over surfaces) over ALL surfaces, then ONE GPU launch for the
// whole scene (ETER §4 rebuild path).
// Patch order = surface order in surfs[], then k = i*nv+j inside each surface
// — identical to the order addPatchGrid appends batches, so cpOff matches
// addPatchGrid's cpVec4Base + k*ru*rv exactly and SSBO meta offsets stay valid.
float* nurbs_adaptive_eval_scene(const SceneSurf* surfs, int nSurfs,
                                 const CameraParams* cam,
                                 ScenePatchMeta** meta_out, int* total_verts)
{
    if (nSurfs <= 0) return NULL;

    // One-time sweep: per-surface bases (meta/flatCps indices) + scene totals.
    int npTotal = 0;
    size_t cpsFloatsTotal = 0;
    int* metaBase = (int*)malloc((size_t)nSurfs * sizeof(int));
    size_t* cpsBase = (size_t*)malloc((size_t)nSurfs * sizeof(size_t));
    for (int s = 0; s < nSurfs; s++) {
        metaBase[s] = npTotal;
        cpsBase[s] = cpsFloatsTotal;
        npTotal += surfs[s].nu * surfs[s].nv;
        cpsFloatsTotal += (size_t)surfs[s].nu * surfs[s].nv * surfs[s].ru * surfs[s].rv * 4;
    }
    if (npTotal <= 0) { free(metaBase); free(cpsBase); return NULL; }

    ScenePatchMeta* meta = (ScenePatchMeta*)malloc((size_t)npTotal * sizeof(ScenePatchMeta));
    float* flatCps = (float*)malloc(cpsFloatsTotal * sizeof(float));

    // Pass 1 (host): per-patch eps + tess factors. Patches are independent,
    // so surfaces run in parallel (OpenMP); each thread uses a stack-local
    // BezierPatch whose CP points straight into the shared source array — the
    // compute functions only READ CPs (verified), so the old per-patch memcpy
    // into a scratch buffer is gone too. Only tu/tv/ru/rv are written here;
    // the scene-global offsets are filled by the serial prefix sums below
    // (57k iterations of += — negligible vs the ~150ms of math they replace:
    // measured 3.7-5.0 ms on 24 threads).
    #pragma omp parallel for schedule(dynamic)
    for (int s = 0; s < nSurfs; s++) {
        const SceneSurf& sf = surfs[s];
        BezierPatch bp;                            // thread-local
        bp.p_u = sf.ru - 1; bp.p_v = sf.rv - 1;    // compute_epsilon/tess read these
        ScenePatchMeta* sm = meta + metaBase[s];
        int np = sf.nu * sf.nv;
        for (int k = 0; k < np; k++) {
            bp.CP = (float*)(sf.cps + (size_t)k * sf.ru * sf.rv * 4);
            float eps = compute_epsilon(&bp, cam);
            int tu = 0, tv = 0;
            compute_tess_factors(&bp, eps, &tu, &tv);
            adapt_tess_factors(&tu, &tv);
            sm[k].tu = tu; sm[k].tv = tv;
            sm[k].ru = sf.ru; sm[k].rv = sf.rv;
        }
        // Each surface owns a disjoint slice of flatCps — parallel-safe.
        memcpy(flatCps + cpsBase[s], sf.cps,
               (size_t)np * sf.ru * sf.rv * 4 * sizeof(float));
    }

    // Serial prefix sums: scene-global vertex offset and vec4 CP offset.
    // Same arithmetic as the old in-loop accumulation — bit-identical.
    int offset = 0, cpOff = 0;
    for (int mi = 0; mi < npTotal; mi++) {
        meta[mi].offset = offset;
        meta[mi].cpOff = cpOff;
        offset += meta[mi].tu * meta[mi].tv;
        cpOff += meta[mi].ru * meta[mi].rv;
    }
    free(metaBase); free(cpsBase);

    // Pass 2 (GPU): ONE launch for the whole scene.
    float* verts = (float*)malloc((size_t)offset * 6 * sizeof(float));
    eval_bezier_patches_scene(flatCps, meta, npTotal, verts, offset);

    free(flatCps);
    *meta_out = meta;
    *total_verts = offset;
    return verts;
}
