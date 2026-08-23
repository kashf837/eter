#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nurbs.h"
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
// NURBS → triangle mesh
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

float* nurbs_wireframe(const char* surf_path, int res,
                       int* np_u_out, int* np_v_out,
                       int* ru_out, int* rv_out,
                       float** cp_out)
{
    NurbsSurf* s = read_surf(surf_path);
    if (!s) return NULL;

    nurbs_to_beziers(s);

    int ru = s->p_u + 1, rv = s->p_v + 1;
    int np_u = (s->n_u + 1) / ru, np_v = (s->n_v + 1) / rv;
    *np_u_out = np_u; *np_v_out = np_v;
    *ru_out = ru; *rv_out = rv;
    int np = np_u * np_v;
    int cpsPerPatch = ru * rv * 4;  // e.g. 64 for bicubic

    float* verts = (float*)malloc(np * res * res * 6 * sizeof(float));
    BezierPatch bp = {s->p_u, s->p_v, (float*)malloc(cpsPerPatch * sizeof(float))};

    // Allocate CP output if requested (ETER §6.2 SSBO)
    if (cp_out) {
        *cp_out = (float*)malloc(np * cpsPerPatch * sizeof(float));
    }

    for (int pi = 0; pi < np_u; pi++) {
        for (int pj = 0; pj < np_v; pj++) {
            for (int i = 0; i < ru; i++)
                for (int j = 0; j < rv; j++)
                    memcpy(&bp.CP[(i * rv + j) * 4],
                           s->Ps[pi * ru + i][pj * rv + j], 4 * sizeof(float));

            // Copy CPs if requested
            if (cp_out) {
                memcpy(*cp_out + (pi * np_v + pj) * cpsPerPatch,
                       bp.CP, cpsPerPatch * sizeof(float));
            }

            float* pv = (float*)malloc(res * res * 6 * sizeof(float));
            eval_bezier_patch(&bp, pv, res, res);

            float* dst = verts + (pi * np_v + pj) * res * res * 6;
            for (int y = 0; y < res; y++)
                for (int x = 0; x < res; x++)
                    memcpy(&dst[(y * res + x) * 6], &pv[(x * res + y) * 6], 6 * sizeof(float));
            free(pv);
        }
    }
    free(bp.CP);
    free_surf(s);
    return verts;
}

// ============================================================
// Per-patch adaptive evaluation (ETER §4)
// ============================================================

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

    PatchInfo* info = (PatchInfo*)malloc(np * sizeof(PatchInfo));
    BezierPatch bp = {s->p_u, s->p_v, (float*)malloc(cpsPerPatch * sizeof(float))};
    int offset = 0;

    // Allocate CP output if requested (ETER §6.2 SSBO)
    if (cp_out) {
        *cp_out = (float*)malloc(np * cpsPerPatch * sizeof(float));
    }

    // First pass: compute tess factors and offsets
    for (int i = 0; i < nu; i++) {
        for (int j = 0; j < nv; j++) {
            for (int ii = 0; ii < ru; ii++)
                for (int jj = 0; jj < rv; jj++)
                    memcpy(&bp.CP[(ii*rv+jj)*4],
                           s->Ps[i*ru+ii][j*rv+jj], 4*sizeof(float));

            float eps = compute_epsilon(&bp, cam);
            int tu = 0, tv = 0;
            compute_tess_factors(&bp, eps, &tu, &tv);
            adapt_tess_factors(&tu, &tv);

            int idx = i*nv+j;
            info[idx].tu = tu; info[idx].tv = tv; info[idx].offset = offset;
            offset += tu * tv;
        }
    }

    // Second pass: evaluate each patch at its own resolution
    float* verts = (float*)malloc(offset * 6 * sizeof(float));

    for (int i = 0; i < nu; i++) {
        for (int j = 0; j < nv; j++) {
            for (int ii = 0; ii < ru; ii++)
                for (int jj = 0; jj < rv; jj++)
                    memcpy(&bp.CP[(ii*rv+jj)*4],
                           s->Ps[i*ru+ii][j*rv+jj], 4*sizeof(float));

            // Save original CPs before eval_bezier_patch offsets them
            if (cp_out) {
                int cpBase = (i*nv+j) * cpsPerPatch;
                memcpy(*cp_out + cpBase, bp.CP, cpsPerPatch * sizeof(float));
            }

            int idx = i*nv+j;
            int tu = info[idx].tu, tv = info[idx].tv;
            float* pv = (float*)malloc(tu * tv * 6 * sizeof(float));
            eval_bezier_patch(&bp, pv, tu, tv);

            float* dst = verts + info[idx].offset * 6;
            for (int y = 0; y < tv; y++)
                for (int x = 0; x < tu; x++)
                    memcpy(&dst[(y*tu+x)*6], &pv[(x*tv+y)*6], 6*sizeof(float));
            free(pv);
        }
    }

    free(bp.CP);
    free_surf(s);
    *info_out = info;
    *total_verts = offset;

    int tu_min=999,tu_max=0,tv_min=999,tv_max=0;
    for(int k=0;k<np;k++){if(info[k].tu<tu_min)tu_min=info[k].tu;if(info[k].tu>tu_max)tu_max=info[k].tu;if(info[k].tv<tv_min)tv_min=info[k].tv;if(info[k].tv>tv_max)tv_max=info[k].tv;}
    printf("  [ADAPT] %dx%d patches, tu=[%d,%d] tv=[%d,%d] total=%d verts\n",
           nu, nv, tu_min, tu_max, tv_min, tv_max, offset);
    return verts;
}
