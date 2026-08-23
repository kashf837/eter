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

// Advance the in-memory cursor to the next content line (skips comments and
// blank lines), NUL-terminate it in place (CRLF -> LF). Returns the line or
// NULL at end of buffer. Replaces the per-line fgets: 3506 files meant
// ~500k fgets calls; one fread per file + pointer scans is far cheaper.
static char* mem_next_line(char** p) {
    while (**p) {
        char* start = *p;
        char* eol = strchr(start, '\n');
        if (!eol) { *p = start + strlen(start); eol = *p; }
        else *p = eol + 1;
        *eol = '\0';
        if (eol > start && eol[-1] == '\r') eol[-1] = '\0';
        if (start[0] != '#' && start[0] != '\0') return start;
    }
    return NULL;
}

static int count_floats(const char* line) {
    int n = 1;
    for (const char* c = line; *c; c++) if (*c == ' ') n++;
    return n;
}

static void parse_floats(char* line, float* out, int n) {
    // strtof instead of sscanf: same correctly-rounded value per float
    // (knot insertion compares U[i]==u exactly, so the bits must not
    // change), but ~4x faster — parsing was 98% of the load cost.
    char* p = line;
    for (int i = 0; i < n; i++) {
        out[i] = strtof(p, &p);
        if (*p == '\0') break;   // malformed tail: mirror the old break
    }
}

static NurbsSurf* read_surf(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "  [SKIP] Cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[got] = '\0';

    NurbsSurf* s = (NurbsSurf*)calloc(1, sizeof(NurbsSurf));
    char* cursor = buf;
    char* line;

    if (!(line = mem_next_line(&cursor))) { free(buf); free(s); return NULL; }
    sscanf(line, "%d %d %d %d", &s->p_u, &s->p_v, &s->n_u, &s->n_v);
    if (s->p_u <= 0 || s->p_v <= 0 || s->n_u <= 0 || s->n_v <= 0 ||
        s->p_u > 20 || s->p_v > 20 || s->n_u > 10000 || s->n_v > 10000) {
        free(buf); free(s); return NULL;
    }

    if (!(line = mem_next_line(&cursor))) { free(buf); free_surf(s); return NULL; }
    {   int n = count_floats(line); s->m_u = n - 1;
        s->U = (float*)malloc(n * sizeof(float)); parse_floats(line, s->U, n); }

    if (!(line = mem_next_line(&cursor))) { free(buf); free_surf(s); return NULL; }
    {   int n = count_floats(line); s->m_v = n - 1;
        s->V = (float*)malloc(n * sizeof(float)); parse_floats(line, s->V, n); }

    s->Ps = new_cpgrid(s->n_u, s->n_v, 4);
    for (int i = 0; i <= s->n_u; i++)
        for (int j = 0; j <= s->n_v; j++) {
            if (!(line = mem_next_line(&cursor))) { free(buf); free_surf(s); return NULL; }
            // Same strtof swap as parse_floats; got<1/3/4 defaults unchanged.
            const char* q = line;
            int got = 0;
            for (int d = 0; d < 4; d++) {
                while (*q == ' ') q++;
                if (*q == '\0') break;   // no more numbers on this line
                s->Ps[i][j][d] = strtof(q, (char**)&q);
                got++;
            }
            if (got < 1) { free(buf); free_surf(s); return NULL; }
            if (got < 3) { s->Ps[i][j][1] = 0; s->Ps[i][j][2] = 0; }
            if (got < 4) s->Ps[i][j][3] = 1.0f;
        }
    free(buf);
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

    nurbs_to_beziers(s);

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

// ============================================================
// Packed .surfb reader — the whole scene in one open
// ============================================================
// Format (little-endian, written by scripts/step2surf.py --all):
//   "SURF" u32 version u32 nSurf, then per surface:
//   i32 p_u p_v n_u n_v | i32 m_u (m_u+1)xf32 U | i32 m_v (m_v+1)xf32 V |
//   (n_u+1)*(n_v+1) x 4xf32 CPs (i-major, xyz+w).
// The floats are the exact float32s the text parser produces (the writer
// rounds the same .10f/.6f strings correctly), so knot insertion's U[i]==u
// compares behave bit-identically to a text load.

int nurbs_read_packed(const char* path, float*** cpsOut, int** nuOut, int** nvOut,
                      int** ruOut, int** rvOut)
{
    *cpsOut = NULL; *nuOut = NULL; *nvOut = NULL; *ruOut = NULL; *rvOut = NULL;

    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 12) { fclose(f); return -1; }
    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return -1; }
    fclose(f);
    buf[sz] = '\0';

    const char* q = buf;
    const char* end = buf + sz;
    if (memcmp(q, "SURF", 4) != 0) { free(buf); return -1; }
    q += 4;
    uint32_t ver = 0, nSurf = 0;
    memcpy(&ver, q, 4); q += 4;
    memcpy(&nSurf, q, 4); q += 4;
    if (ver != 1 || nSurf == 0 || nSurf > 1000000) { free(buf); return -1; }

    // Serial sweep: record each surface's byte span so the parallel decode
    // below can hand every thread a disjoint buffer slice. The skip
    // arithmetic here and the memcpy arithmetic in the parallel loop must
    // stay identical — they are each other's bounds guarantee.
    size_t* offs = (size_t*)malloc(((size_t)nSurf + 1) * sizeof(size_t));
    if (!offs) { free(buf); return -1; }
    bool ok = true;
    for (uint32_t i = 0; i < nSurf && ok; i++) {
        offs[i] = (size_t)(q - buf);
        int h[4];
        if (end - q < 16) { ok = false; break; }
        memcpy(h, q, 16); q += 16;
        if (h[0] <= 0 || h[1] <= 0 || h[2] <= 0 || h[3] <= 0 ||
            h[0] > 20 || h[1] > 20 || h[2] > 10000 || h[3] > 10000) { ok = false; break; }
        int m_u = 0, m_v = 0;
        if (end - q < 4) { ok = false; break; }
        memcpy(&m_u, q, 4); q += 4;
        if (m_u < 0 || m_u > 100000 || end - q < (long)(m_u + 1) * 4) { ok = false; break; }
        q += (m_u + 1) * 4;
        if (end - q < 4) { ok = false; break; }
        memcpy(&m_v, q, 4); q += 4;
        if (m_v < 0 || m_v > 100000 || end - q < (long)(m_v + 1) * 4) { ok = false; break; }
        q += (m_v + 1) * 4;
        long ncp = (long)(h[2] + 1) * (h[3] + 1) * 16;
        if (end - q < ncp) { ok = false; break; }
        q += ncp;
    }
    if (!ok) { free(buf); free(offs); return -1; }
    offs[nSurf] = (size_t)(q - buf);

    float** cps = (float**)calloc(nSurf, sizeof(float*));
    int* nu = (int*)calloc(nSurf, sizeof(int));
    int* nv = (int*)calloc(nSurf, sizeof(int));
    int* ru = (int*)calloc(nSurf, sizeof(int));
    int* rv = (int*)calloc(nSurf, sizeof(int));
    if (!cps || !nu || !nv || !ru || !rv) {
        free(cps); free(nu); free(nv); free(ru); free(rv);
        free(buf); free(offs); return -1;
    }

    // Decompose every surface in parallel. Each thread owns a disjoint
    // buffer slice and its own NURBS structs — fully independent, the same
    // pattern as the per-file text path.
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)nSurf; i++) {
        const char* p = buf + offs[(size_t)i];
        NurbsSurf s;
        memcpy(&s.p_u, p, 16); p += 16;
        memcpy(&s.m_u, p, 4);  p += 4;
        s.U = (float*)malloc(((size_t)s.m_u + 1) * 4);
        memcpy(s.U, p, ((size_t)s.m_u + 1) * 4); p += ((size_t)s.m_u + 1) * 4;
        memcpy(&s.m_v, p, 4);  p += 4;
        s.V = (float*)malloc(((size_t)s.m_v + 1) * 4);
        memcpy(s.V, p, ((size_t)s.m_v + 1) * 4); p += ((size_t)s.m_v + 1) * 4;
        s.Ps = new_cpgrid(s.n_u, s.n_v, 4);
        for (int ii = 0; ii <= s.n_u; ii++)
            for (int jj = 0; jj <= s.n_v; jj++)
                memcpy(s.Ps[ii][jj], p + ((size_t)ii * (s.n_v + 1) + jj) * 16, 16);

        nurbs_to_beziers(&s);

        int r_u = s.p_u + 1, r_v = s.p_v + 1;
        int n_u = (s.n_u + 1) / r_u, n_v = (s.n_v + 1) / r_v;
        // NOTE: s lives on the STACK here — do NOT call free_surf(&s) (it
        // does free(s) and would hand a stack pointer to the heap, which is
        // exactly the 0xC0000374 heap corruption this path hit). Free the
        // three heap-owned parts manually instead.
        if (n_u <= 0 || n_v <= 0 || r_u - 1 > 7 || r_v - 1 > 7) {
            if (r_u - 1 > 7 || r_v - 1 > 7)
                fprintf(stderr, "  [SKIP] surface %d: degree (%d,%d) > 7 not supported\n",
                        i, r_u - 1, r_v - 1);
            free_cpgrid(s.Ps, s.n_u, s.n_v);
            free(s.U); free(s.V);
            continue;   // cps[i] stays NULL -> caller skips it
        }
        int np = n_u * n_v;
        float* flat = (float*)malloc((size_t)np * r_u * r_v * 4 * sizeof(float));
        for (int ii = 0; ii < n_u; ii++)
            for (int jj = 0; jj < n_v; jj++)
                for (int k = 0; k < r_u; k++)
                    for (int l = 0; l < r_v; l++)
                        memcpy(&flat[((size_t)(ii * n_v + jj) * r_u * r_v + k * r_v + l) * 4],
                               s.Ps[ii * r_u + k][jj * r_v + l], 4 * sizeof(float));
        free_cpgrid(s.Ps, s.n_u, s.n_v);
        free(s.U); free(s.V);
        cps[i] = flat;
        nu[i] = n_u; nv[i] = n_v; ru[i] = r_u; rv[i] = r_v;
    }
    free(buf); free(offs);

    *cpsOut = cps; *nuOut = nu; *nvOut = nv; *ruOut = ru; *rvOut = rv;
    return (int)nSurf;
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
