#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nurbs_tex.h"
#include "knot_insert.h"
#include "bezier_eval.h"

// ---- File I/O helpers (from nurbs_eval.cu) ----

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

// ---- Main pipeline ----

int launch_nurbs_texture(float* cpu_data, int W, int H, const char* surf_path)
{
    // 1. Load .surf
    NurbsSurf* s = read_surf(surf_path);
    if (!s) return -1;
    printf("Loaded: p=(%d,%d) CPs=%dx%d\n", s->p_u, s->p_v, s->n_u+1, s->n_v+1);

    // 2. Knot insertion → Bezier patches
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

    // 3. Extract first Bezier patch
    int ru = s->p_u + 1, rv = s->p_v + 1;
    BezierPatch bp;
    bp.p_u = s->p_u; bp.p_v = s->p_v;
    bp.CP = (float*)malloc(ru * rv * 4 * sizeof(float));
    for (int i = 0; i < ru; i++)
        for (int j = 0; j < rv; j++)
            memcpy(&bp.CP[(i * rv + j) * 4], s->Ps[i][j], 4 * sizeof(float));

    // 4. Evaluate on GPU at W×H sampling points
    float* V = (float*)malloc(W * H * 3 * sizeof(float));
    eval_bezier_patch(&bp, V, W, H);

    // 5. Find z range for heatmap
    float zmin = 1e30f, zmax = -1e30f;
    for (int i = 0; i < W * H; i++) {
        float z = V[i * 3 + 2];
        if (z < zmin) zmin = z;
        if (z > zmax) zmax = z;
    }
    printf("z range: [%.4f, %.4f]\n", zmin, zmax);

    // 6. Convert xyz → heatmap RGBA
    for (int i = 0; i < W * H; i++) {
        float z = V[i * 3 + 2];
        float t = (z - zmin) / (zmax - zmin + 1e-6f);
        float r, g, b;
        if (t < 0.5f) {
            float s = t * 2.0f;
            r = 0.0f;  g = s;     b = 1.0f - s;
        } else {
            float s = (t - 0.5f) * 2.0f;
            r = s;     g = 1.0f - s;  b = 0.0f;
        }
        float* pixel = cpu_data + i * 4;
        pixel[0] = r; pixel[1] = g; pixel[2] = b; pixel[3] = 1.0f;
    }

    // 7. Cleanup
    free(V); free(bp.CP);
    free_surf(s);
    return 0;
}
