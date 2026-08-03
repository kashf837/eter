#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nurbs.h"
#include "knot_insert.h"
#include "bezier_eval.h"

static int next_line(FILE* f, char* buf, int size) {
    while (fgets(buf, size, f))
        if (buf[0] != '#' && buf[0] != '\n') return 1;
    return 0;
}

static NurbsSurf* read_surf(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) { printf("Cannot open %s\n", path); return NULL; }
    NurbsSurf* s = (NurbsSurf*)calloc(1, sizeof(NurbsSurf));
    char line[256];

    next_line(f, line, 256);
    sscanf(line, "%d %d %d %d", &s->p_u, &s->p_v, &s->n_u, &s->n_v);
    s->m_u = s->n_u + s->p_u + 1;
    s->m_v = s->n_v + s->p_v + 1;

    s->U = (float*)malloc((s->m_u + 1) * sizeof(float));
    next_line(f, line, 256);
    char* p = line;
    for (int i = 0; i <= s->m_u; i++, p = strchr(p, ' ') + 1)
        sscanf(p, "%f", &s->U[i]);

    s->V = (float*)malloc((s->m_v + 1) * sizeof(float));
    next_line(f, line, 256);
    p = line;
    for (int i = 0; i <= s->m_v; i++, p = strchr(p, ' ') + 1)
        sscanf(p, "%f", &s->V[i]);

    s->Ps = new_cpgrid(s->n_u, s->n_v, 4);
    for (int i = 0; i <= s->n_u; i++)
        for (int j = 0; j <= s->n_v; j++) {
            next_line(f, line, 256);
            sscanf(line, "%f %f %f %f",
                   &s->Ps[i][j][0], &s->Ps[i][j][1],
                   &s->Ps[i][j][2], &s->Ps[i][j][3]);
        }
    fclose(f);
    return s;
}

static void free_surf(NurbsSurf* s) {
    free_cpgrid(s->Ps, s->n_u, s->n_v);
    free(s->U); free(s->V);
    free(s);
}

int nurbs_to_texture(float* cpu_data, int W, int H, const char* surf_path)
{
    // 1. Load .surf
    NurbsSurf* s = read_surf(surf_path);
    if (!s) return -1;
    printf("Loaded: p=(%d,%d) CPs=%dx%d\n", s->p_u, s->p_v, s->n_u+1, s->n_v+1);

    // 2. Knot insertion -> Bezier patches
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
    printf("After KI: CPs=%dx%d\n", s->n_u+1, s->n_v+1);

    // 3. Evaluate all patches, find z range
    int ru = s->p_u + 1, rv = s->p_v + 1;
    int np_u = (s->n_u + 1) / ru;
    int np_v = (s->n_v + 1) / rv;
    int pw = W / np_u, ph = H / np_v;
    printf("Patches: %dx%d, each %dx%d pixels\n", np_u, np_v, pw, ph);

    float zmin = 1e30f, zmax = -1e30f;

    BezierPatch bp;
    bp.p_u = s->p_u; bp.p_v = s->p_v;
    bp.CP = (float*)malloc(ru * rv * 4 * sizeof(float));

    for (int pi = 0; pi < np_u; pi++) {
        for (int pj = 0; pj < np_v; pj++) {
            // Extract patch from CP grid
            for (int i = 0; i < ru; i++)
                for (int j = 0; j < rv; j++)
                    memcpy(&bp.CP[(i * rv + j) * 4],
                           s->Ps[pi * ru + i][pj * rv + j], 4 * sizeof(float));

            // Evaluate at pw x ph points
            float* V = (float*)malloc(pw * ph * 3 * sizeof(float));
            eval_bezier_patch(&bp, V, pw, ph);

            // Map to texture region: x=pi*width (no flip), y = flip v (image y↓ vs NURBS v↑)
            int x0 = pi * pw;
            int y0 = pj * ph;
            for (int y = 0; y < ph; y++) {
                for (int x = 0; x < pw; x++) {
                    float z = V[(x * ph + y) * 3 + 2];
                    if (z < zmin) zmin = z;
                    if (z > zmax) zmax = z;
                    int tx = x0 + x, ty = y0 + y;
                    cpu_data[(ty * W + tx) * 4 + 0] = z;  // stash z in R channel
                }
            }
            free(V);
        }
    }
    printf("z range: [%.4f, %.4f]\n", zmin, zmax);

    // 4. Convert z -> heatmap RGBA
    for (int i = 0; i < W * H; i++) {
        float z = cpu_data[i * 4 + 0];
        float t = (z - zmin) / (zmax - zmin + 1e-6f);
        float r, g, b;
        if (t < 0.5f) {
            float s = t * 2.0f;
            r = 0.0f;  g = s;     b = 1.0f - s;
        } else {
            float s = (t - 0.5f) * 2.0f;
            r = s;     g = 1.0f - s;  b = 0.0f;
        }
        cpu_data[i * 4 + 0] = r; cpu_data[i * 4 + 1] = g;
        cpu_data[i * 4 + 2] = b; cpu_data[i * 4 + 3] = 1.0f;
    }

    free(bp.CP);
    free_surf(s);
    return 0;
}

// ---- Wireframe: evaluate all patches at coarse grid, return x,y positions ----

float* nurbs_wireframe(const char* surf_path, int res,
                       int* np_u_out, int* np_v_out)
{
    NurbsSurf* s = read_surf(surf_path);
    if (!s) return NULL;

    // KI
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

    int ru = s->p_u + 1, rv = s->p_v + 1;
    int np_u = (s->n_u + 1) / ru;
    int np_v = (s->n_v + 1) / rv;
    *np_u_out = np_u; *np_v_out = np_v;

    float* verts = (float*)malloc(np_u * np_v * res * res * 2 * sizeof(float));
    BezierPatch bp;
    bp.p_u = s->p_u; bp.p_v = s->p_v;
    bp.CP = (float*)malloc(ru * rv * 4 * sizeof(float));

    for (int pi = 0; pi < np_u; pi++) {
        for (int pj = 0; pj < np_v; pj++) {
            for (int i = 0; i < ru; i++)
                for (int j = 0; j < rv; j++)
                    memcpy(&bp.CP[(i * rv + j) * 4],
                           s->Ps[pi * ru + i][pj * rv + j], 4 * sizeof(float));

            float* V = (float*)malloc(res * res * 3 * sizeof(float));
            eval_bezier_patch(&bp, V, res, res);

            float* dst = verts + (pi * np_v + pj) * res * res * 2;
            for (int y = 0; y < res; y++)
                for (int x = 0; x < res; x++) {
                    dst[(y * res + x) * 2 + 0] = V[(x * res + y) * 3 + 0]; // x
                    dst[(y * res + x) * 2 + 1] = V[(x * res + y) * 3 + 1]; // y
                }
            free(V);
        }
    }

    free(bp.CP);
    free_surf(s);
    return verts;
}
