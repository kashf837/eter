#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "knot_insert.h"
#include "bezier_eval.h"

// ========== Scale tess factors to (7p+1)x(7q+1) for meshlets ==========

static void adapt_tess_factors(int* tu, int* tv) {
    int p = (*tu + 6) / 7;  if (p < 1) p = 1;
    int q = (*tv + 6) / 7;  if (q < 1) q = 1;
    *tu = 7 * p + 1;
    *tv = 7 * q + 1;
}

// ========== File I/O ==========

static int next_line(FILE* f, char* buf, int size) {
    while (fgets(buf, size, f))
        if (buf[0] != '#' && buf[0] != '\n') return 1;
    return 0;
}

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

//  ε → tu, tv ==========

// Compute Δu, Δv from ε using 2nd-difference formula on 4D homogeneous CPs.
static void compute_tess_factors(BezierPatch* bp, float eps, int* tu, int* tv) {
    int m = bp->p_u + 1,  n = bp->p_v + 1;
    int cols = n;

    float min_w = 1e30f, max_norm = 0.0f;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float w = bp->CP[(i * cols + j) * 4 + 3];
            if (w < min_w) min_w = w;
            float x = bp->CP[(i * cols + j) * 4 + 0] / w;
            float y = bp->CP[(i * cols + j) * 4 + 1] / w;
            float z = bp->CP[(i * cols + j) * 4 + 2] / w;
            float nd = sqrtf(x*x + y*y + z*z);
            if (nd > max_norm) max_norm = nd;
        }
    float r_star = max_norm;

    // u direction: 2nd difference along rows
    float max_D2u = 0.0f, max_D2w_u = 0.0f;
    for (int i = 0; i <= m - 3; i++)
        for (int j = 0; j < n; j++) {
            float* p0 = &bp->CP[( i      * cols + j) * 4];
            float* p1 = &bp->CP[((i + 1) * cols + j) * 4];
            float* p2 = &bp->CP[((i + 2) * cols + j) * 4];
            float sum2 = 0.0f;
            for (int d = 0; d < 4; d++) {
                float d2 = p2[d] - 2.0f * p1[d] + p0[d];
                sum2 += d2 * d2;
            }
            float nd = sqrtf(sum2);
            if (nd > max_D2u) max_D2u = nd;
            float d2w = p2[3] - 2.0f * p1[3] + p0[3];
            float aw = fabsf(d2w);
            if (aw > max_D2w_u) max_D2w_u = aw;
        }
    float denom_u = m*(m-1)*max_D2u + (r_star - eps)*m*(m-1)*max_D2w_u;
    float delta_u = sqrtf(8.0f * eps * min_w / denom_u);
    if (delta_u < 1e-10f) delta_u = 1e-10f;
    *tu = (int)ceilf(1.0f / delta_u);

    // v direction: 2nd difference along columns
    float max_D2v = 0.0f, max_D2w_v = 0.0f;
    for (int i = 0; i < m; i++)
        for (int j = 0; j <= n - 3; j++) {
            float* p0 = &bp->CP[(i * cols +  j     ) * 4];
            float* p1 = &bp->CP[(i * cols + (j + 1)) * 4];
            float* p2 = &bp->CP[(i * cols + (j + 2)) * 4];
            float sum2 = 0.0f;
            for (int d = 0; d < 4; d++) {
                float d2 = p2[d] - 2.0f * p1[d] + p0[d];
                sum2 += d2 * d2;
            }
            float nd = sqrtf(sum2);
            if (nd > max_D2v) max_D2v = nd;
            float d2w = p2[3] - 2.0f * p1[3] + p0[3];
            float aw = fabsf(d2w);
            if (aw > max_D2w_v) max_D2w_v = aw;
        }
    float denom_v = n*(n-1)*max_D2v + (r_star - eps)*n*(n-1)*max_D2w_v;
    float delta_v = sqrtf(8.0f * eps * min_w / denom_v);
    if (delta_v < 1e-10f) delta_v = 1e-10f;
    *tv = (int)ceilf(1.0f / delta_v);
}

// ========== Screen error → object-space error ==========

// Camera model (hardcoded for now — replace with proper camera later)
#define CAM_FOV_Y  60.0f                         // vertical FOV in degrees
#define CAM_WIDTH  1920                          // screen resolution
#define CAM_HEIGHT 1080
#define CAM_POS_X  0.5f                          // camera position (world space)
#define CAM_POS_Y  0.5f
#define CAM_POS_Z  -2.0f                         // camera looking down +Z

static float compute_epsilon(BezierPatch* bp) {
    // projection matrix coefficients
    float B = 1.0f / tanf((CAM_FOV_Y * 3.14159265f / 180.0f) * 0.5f);  // B = cot(fov_y/2)
    float A = B * (float)CAM_WIDTH / (float)CAM_HEIGHT;                // A = B * aspect
    float tau_x = 1.0f / (float)CAM_WIDTH;                             // half-pixel in clip space
    float tau_y = 1.0f / (float)CAM_HEIGHT;

    // compute patch bounding box in camera space
    float z_min = 1e30f, xz_max = 0.0f, yz_max = 0.0f;
    int rows = bp->p_u + 1, cols = bp->p_v + 1;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float xw = bp->CP[(i * cols + j) * 4 + 0];
            float yw = bp->CP[(i * cols + j) * 4 + 1];
            float zw = bp->CP[(i * cols + j) * 4 + 2];
            float w  = bp->CP[(i * cols + j) * 4 + 3];

            // perspective division to get 3D position
            float x = xw / w, y = yw / w, z = zw / w;

            // transform to camera space (camera at CAM_POS looking +Z)
            float cx = x - CAM_POS_X;
            float cy = y - CAM_POS_Y;
            float cz = z - CAM_POS_Z;           // depth = distance along view direction

            if (cz < 0.01f) cz = 0.01f;          // clamp to avoid division by zero

            if (cz < z_min) z_min = cz;
            float axz = fabsf(cx / cz);
            float ayz = fabsf(cy / cz);
            if (axz > xz_max) xz_max = axz;
            if (ayz > yz_max) yz_max = ayz;
        }
    }

    // equation from §4.1
    float ex = (z_min * tau_x) / (A * (1.0f + xz_max) + tau_x);
    float ey = (z_min * tau_y) / (B * (1.0f + yz_max) + tau_y);
    float eps = (ex < ey) ? ex : ey;

    return eps;
}

// ========== Process one surface, write into open OBJ file ==========
static int process_surf(const char* path, FILE* obj, int* voff) {
    NurbsSurf* s = read_surf(path);
    if (!s) return 0;
    if (s->p_u < 0 || s->p_v < 0 || s->n_u < 0 || s->n_v < 0) {
        printf("  SKIP (invalid)\n"); free_surf(s); return 0;
    }
    printf("  p=(%d,%d) CPs=%dx%d\n", s->p_u, s->p_v, s->n_u+1, s->n_v+1);

    // 2. Knot insertion in u and v
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

    // 3. Extract Bezier patches, evaluate each on GPU, write OBJ
    // Each patch is ru x rv control points, flattened into bp.CP
    int ru = s->p_u + 1, rv = s->p_v + 1;                // rows/cols per patch
    int np_u = (s->n_u + 1 - ru) / ru + 1;               // number of patches in u
    int np_v = (s->n_v + 1 - rv) / rv + 1;               // number of patches in v

    BezierPatch bp;
    bp.p_u = s->p_u; bp.p_v = s->p_v;
    bp.CP  = (float*)malloc(ru * rv * 4 * sizeof(float));
    float* V = NULL;  // allocate per patch (size varies)
    int V_cap = 0;

    for (int pi = 0; pi < np_u; pi++) {
        for (int pj = 0; pj < np_v; pj++) {
            // Flatten this patch from the grid into bp.CP
            for (int i = 0; i < ru; i++)
                for (int j = 0; j < rv; j++)
                    memcpy(&bp.CP[(i * rv + j) * 4],
                           s->Ps[pi * ru + i][pj * rv + j],
                           4 * sizeof(float));

            // §4: compute epsilon and tessellation factors
            float eps = compute_epsilon(&bp);
            int tu = 0, tv = 0;
            compute_tess_factors(&bp, eps, &tu, &tv);
            adapt_tess_factors(&tu, &tv);
            // GPU eval with dynamic resolution
            int nV = tu * tv;
            if (nV > V_cap) { free(V); V = (float*)malloc(nV * 3 * sizeof(float)); V_cap = nV; }
            eval_bezier_patch(&bp, V, tu, tv);

            // Write vertices
            for (int vi = 0; vi < nV; vi++)
                fprintf(obj, "v %f %f %f\n", V[vi*3+0], V[vi*3+1], V[vi*3+2]);

            // Write faces with vertex offset
            int off = *voff;
            for (int i = 0; i < tu - 1; i++)
                for (int j = 0; j < tv - 1; j++) {
                    int a = off + i * tv + j + 1, b = a + 1;
                    int c = off + (i + 1) * tv + j + 1, d = c + 1;
                    fprintf(obj, "f %d %d %d\n", a, b, c);
                    fprintf(obj, "f %d %d %d\n", b, d, c);
                }
            *voff += nV;
        }
    }

    int total = np_u * np_v;
    free(V); free(bp.CP);
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
