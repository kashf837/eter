#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "knot_insert.h"
#include "bezier_eval.h"

#define N 32

// ========== File I/O ==========

static int next_line(FILE* f, char* buf, int size) {
    while (fgets(buf, size, f))
        if (buf[0] != '#' && buf[0] != '\n') return 1;
    return 0;
}

typedef struct {
    int p_u, p_v, n_u, n_v, m_u, m_v;
    float* U, *V;
    CPGrid Ps;
} NurbsSurf;

static NurbsSurf* read_surf(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) { printf("Cannot open %s\n", path); return NULL; }

    NurbsSurf* s = (NurbsSurf*)malloc(sizeof(NurbsSurf));
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

// ========== Process one surface, write into open OBJ file ==========
static int process_surf(const char* path, FILE* obj, int* voff) {
    NurbsSurf* s = read_surf(path);
    if (!s) return 0;
    if (s->p_u < 0 || s->p_v < 0 || s->n_u < 0 || s->n_v < 0) {
        printf("  SKIP (invalid)\n"); free_surf(s); return 0;
    }
    printf("  p=(%d,%d) CPs=%dx%d", s->p_u, s->p_v, s->n_u+1, s->n_v+1);

    // 2. Knot insertion in u and v
    for (int i = s->p_u + 1; i <= s->m_u - s->p_u - 1; i++) {
        float u = s->U[i];
        while (mult_of(s->U, s->m_u, u) < s->p_u + 1) {
            int k = find_span(s->U, s->m_u, u);
            knot_insert_surf_u(&s->U, &s->m_u, &s->Ps, &s->n_u, s->n_v,
                               s->p_u, 4, u, k);
        }
    }
    for (int i = s->p_v + 1; i <= s->m_v - s->p_v - 1; i++) {
        float v = s->V[i];
        while (mult_of(s->V, s->m_v, v) < s->p_v + 1) {
            int k = find_span(s->V, s->m_v, v);
            knot_insert_surf_v(&s->V, &s->m_v, &s->Ps, s->n_u, &s->n_v,
                               s->p_v, 4, v, k);
        }
    }
    printf("After KI: CPs=%dx%d\n", s->n_u+1, s->n_v+1);

    // Pad degenerate directions so each patch has enough CPs
    int ru = s->p_u + 1, rv = s->p_v + 1;
    while (s->n_u + 1 < ru) {
        // duplicate last row
        s->n_u++; s->m_u++;
        s->Ps = (CPGrid)realloc(s->Ps, (s->n_u + 1) * sizeof(CPArray));
        s->Ps[s->n_u] = new_cparray(s->n_v, 4);
        s->U = (float*)realloc(s->U, (s->m_u + 1) * sizeof(float));
        s->U[s->m_u] = s->U[s->m_u - 1];
        for (int j = 0; j <= s->n_v; j++)
            memcpy(s->Ps[s->n_u][j], s->Ps[s->n_u - 1][j], 4 * sizeof(float));
    }
    while (s->n_v + 1 < rv) {
        s->n_v++; s->m_v++;
        for (int i = 0; i <= s->n_u; i++) {
            s->Ps[i] = (CPArray)realloc(s->Ps[i], (s->n_v + 1) * sizeof(CP));
            s->Ps[i][s->n_v] = new_cp(4);
            memcpy(s->Ps[i][s->n_v], s->Ps[i][s->n_v - 1], 4 * sizeof(float));
        }
        s->V = (float*)realloc(s->V, (s->m_v + 1) * sizeof(float));
        s->V[s->m_v] = s->V[s->m_v - 1];
    }
    printf("After pad: CPs=%dx%d\n", s->n_u+1, s->n_v+1);

    // 3. Extract Bezier patches, evaluate each on GPU, write OBJ
    int np_u = (s->n_u + 1 - ru) / ru + 1;
    int np_v = (s->n_v + 1 - rv) / rv + 1;

    float* V   = (float*)malloc(N * N * 3 * sizeof(float));
    float* CP  = (float*)malloc(ru * rv * 4 * sizeof(float));

    for (int pi = 0; pi < np_u; pi++) {
        for (int pj = 0; pj < np_v; pj++) {
            // Flatten this patch's CPs
            for (int i = 0; i < ru; i++)
                for (int j = 0; j < rv; j++)
                    memcpy(&CP[(i * rv + j) * 4],
                           s->Ps[pi * ru + i][pj * rv + j],
                           4 * sizeof(float));

            // GPU eval
            eval_bezier_patch(CP, s->p_u, s->p_v, V, N);

            // Write vertices
            for (int vi = 0; vi < N * N; vi++)
                fprintf(obj, "v %f %f %f\n", V[vi*3+0], V[vi*3+1], V[vi*3+2]);

            // Write faces with vertex offset
            for (int i = 0; i < N - 1; i++)
                for (int j = 0; j < N - 1; j++) {
                    int off = *voff;
                    int a = off + i * N + j + 1, b = a + 1;
                    int c = off + (i + 1) * N + j + 1, d = c + 1;
                    fprintf(obj, "f %d %d %d\n", a, b, c);
                    fprintf(obj, "f %d %d %d\n", b, d, c);
                }

            *voff += N * N;
        }
    }

    int total = np_u * np_v;
    free(V); free(CP);
    free_surf(s);
    printf(" -> %d patches\n", total);
    return total;
}

// ========== Main ==========

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: nurbs_eval.exe <surf1.surf> [surf2.surf ...]\n");
        printf("       nurbs_eval.exe --batch <dir> <output.obj>\n");
        return 1;
    }

    FILE* obj;
    int vert_offset = 0;
    int total_patches = 0, total_surfs = 0;

    if (strcmp(argv[1], "--batch") == 0) {
        // Batch mode: read all .surf from directory
        if (argc < 4) { printf("Usage: nurbs_eval.exe --batch <dir> <out.obj>\n"); return 1; }
        obj = fopen(argv[3], "w");
        printf("Batch from %s -> %s\n", argv[2], argv[3]);
        // Use system call to list files, read output
        char cmd[512];
        snprintf(cmd, 512, "dir /b \"%s\\*.surf\" 2>nul", argv[2]);
        FILE* fp = _popen(cmd, "r");
        if (!fp) { printf("Cannot list directory\n"); return 1; }
        char path[512];
        while (fgets(path, 512, fp)) {
            path[strcspn(path, "\n")] = 0;
            if (strlen(path) == 0) continue;
            char full[512];
            snprintf(full, 512, "%s/%s", argv[2], path);
            printf("[%d]", total_surfs);
            int p = process_surf(full, obj, &vert_offset);
            if (p > 0) { total_patches += p; total_surfs++; }
        }
        _pclose(fp);
    } else {
        obj = fopen("build/nurbs_output.obj", "w");
        for (int i = 1; i < argc; i++) {
            printf("[%d]", total_surfs);
            int p = process_surf(argv[i], obj, &vert_offset);
            if (p > 0) { total_patches += p; total_surfs++; }
        }
    }

    fclose(obj);
    printf("\nDone: %d surfaces, %d patches, %d verts -> OBJ\n",
           total_surfs, total_patches, vert_offset);
    return 0;
}
