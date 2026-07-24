#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define N 32
#define BLOCK_SIZE 8
#define MAX_ORDER 7

// ========== Math helpers ==========

static void compute_binom(int order, int* binom) {
    binom[0] = 1;
    for (int k = 1; k <= order; k++)
        binom[k] = binom[k - 1] * (order - k + 1) / k;
}

// ========== File I/O ==========

// Skip comment lines (#) and blank lines. Returns 1 if a line was read.
static int next_line(FILE* f, char* buf, int size) {
    while (fgets(buf, size, f))
        if (buf[0] != '#' && buf[0] != '\n') return 1;
    return 0;
}

// Read control points from .cp file. Format:
//   line 1: order_u order_v
//   rest:   x y z w  (one CP per line, dim=4)
static float* load_cp_file(const char* path, int* order_u, int* order_v, int* num_cp) {
    FILE* f = fopen(path, "r");
    if (!f) { printf("Cannot open %s\n", path); return NULL; }

    char line[256];
    next_line(f, line, 256);
    sscanf(line, "%d %d", order_u, order_v);

    int rows = *order_u + 1, cols = *order_v + 1;
    *num_cp = rows * cols;

    float* CP = (float*)malloc(*num_cp * 4 * sizeof(float));
    for (int i = 0; i < *num_cp; i++) {
        next_line(f, line, 256);
        sscanf(line, "%f %f %f %f", &CP[i*4+0], &CP[i*4+1], &CP[i*4+2], &CP[i*4+3]);
    }
    fclose(f);
    return CP;
}

// Write triangle mesh to .obj
static void write_obj(const char* path, float* V, int n) {
    FILE* f = fopen(path, "w");
    for (int i = 0; i < n * n; i++)
        fprintf(f, "v %f %f %f\n", V[i*3+0], V[i*3+1], V[i*3+2]);
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - 1; j++) {
            int a = i * n + j + 1, b = a + 1;
            int c = (i + 1) * n + j + 1, d = c + 1;
            fprintf(f, "f %d %d %d\n", a, b, c);
            fprintf(f, "f %d %d %d\n", b, d, c);
        }
    fclose(f);
}

// ========== GPU kernel ==========

__global__ void eval_surface(
    float* d_CP, float* d_V, int n,
    int order_u, int order_v, int cols,
    int* d_binom_u, int* d_binom_v
) {
    int i  = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int j  = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int idx = i * n + j;

    float u = (float)i / (n - 1);
    float v = (float)j / (n - 1);

    // shared memory for Bernstein basis functions
    __shared__ float s_Bu[BLOCK_SIZE][MAX_ORDER + 1];
    __shared__ float s_Bv[BLOCK_SIZE][MAX_ORDER + 1];

    if (threadIdx.x <= order_u)
        s_Bu[threadIdx.y][threadIdx.x] =
            powf(u, threadIdx.x) * powf(1.0f - u, order_u - threadIdx.x)
            * (float)d_binom_u[threadIdx.x];

    if (threadIdx.y <= order_v)
        s_Bv[threadIdx.x][threadIdx.y] =
            powf(v, threadIdx.y) * powf(1.0f - v, order_v - threadIdx.y)
            * (float)d_binom_v[threadIdx.y];

    __syncthreads();

    // accumulate weighted sum:  S(u,v) = sum(B_i(u) B_j(v) * w_ij * P_ij) / sum(B_i(u) B_j(v) * w_ij)
    float Sw[4] = {0.0f};  // homogeneous: Sx, Sy, Sz, Sw
    for (int ii = 0; ii <= order_u; ii++) {
        for (int jj = 0; jj <= order_v; jj++) {
            float wb    = s_Bu[threadIdx.y][ii] * s_Bv[threadIdx.x][jj];  // Bernstein product
            float w_cp  = d_CP[(ii * cols + jj) * 4 + 3];                 // control point weight
            float w     = wb * w_cp;

            for (int k = 0; k < 3; k++)
                Sw[k] += w * d_CP[(ii * cols + jj) * 4 + k];              // w * (x,y,z)
            Sw[3] += w;                                                    // w * w_cp
        }
    }

    // perspective division
    float inv = 1.0f / Sw[3];
    d_V[idx * 3 + 0] = Sw[0] * inv;
    d_V[idx * 3 + 1] = Sw[1] * inv;
    d_V[idx * 3 + 2] = Sw[2] * inv;
}

// ========== Main ==========

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: bezier_eval.exe <cp_file>\n");
        return 1;
    }

    // --- load control points ---
    int order_u, order_v, num_cp;
    float* CP = load_cp_file(argv[1], &order_u, &order_v, &num_cp);
    if (!CP) return 1;

    int cols = order_v + 1;

    // --- precompute binomial coefficients ---
    int* binom_u = (int*)malloc((order_u + 1) * sizeof(int));
    int* binom_v = (int*)malloc((order_v + 1) * sizeof(int));
    compute_binom(order_u, binom_u);
    compute_binom(order_v, binom_v);

    // --- GPU memory ---
    float* V = (float*)malloc(N * N * 3 * sizeof(float));

    float *d_CP, *d_V;
    int   *d_binom_u, *d_binom_v;

    cudaMalloc(&d_CP,      num_cp * 4 * sizeof(float));
    cudaMalloc(&d_V,       N * N * 3 * sizeof(float));
    cudaMalloc(&d_binom_u, (order_u + 1) * sizeof(int));
    cudaMalloc(&d_binom_v, (order_v + 1) * sizeof(int));

    cudaMemcpy(d_CP,      CP,      num_cp * 4       * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_binom_u, binom_u, (order_u + 1)     * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_binom_v, binom_v, (order_v + 1)     * sizeof(int),   cudaMemcpyHostToDevice);

    // --- launch kernel ---
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid (N / BLOCK_SIZE, N / BLOCK_SIZE);
    eval_surface<<<grid, block>>>(d_CP, d_V, N,
        order_u, order_v, cols, d_binom_u, d_binom_v);

    // --- copy back and write OBJ ---
    cudaMemcpy(V, d_V, N * N * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    write_obj("build/output.obj", V, N);

    // --- cleanup ---
    cudaFree(d_CP);      cudaFree(d_V);
    cudaFree(d_binom_u);  cudaFree(d_binom_v);
    free(CP); free(V); free(binom_u); free(binom_v);

    printf("Done: %d CPs, %d verts -> build/output.obj\n", num_cp, N * N);
    return 0;
}
