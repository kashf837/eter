#include "bezier_eval.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define BLOCK_SIZE 8
#define MAX_ORDER 7

void compute_binom(int order, int* binom) {
    binom[0] = 1;
    for (int k = 1; k <= order; k++)
        binom[k] = binom[k - 1] * (order - k + 1) / k;
}

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

    float Sw[4] = {0.0f};
    for (int ii = 0; ii <= order_u; ii++)
        for (int jj = 0; jj <= order_v; jj++) {
            float wb   = s_Bu[threadIdx.y][ii] * s_Bv[threadIdx.x][jj];
            float w_cp = d_CP[(ii * cols + jj) * 4 + 3];
            float w    = wb * w_cp;
            for (int k = 0; k < 3; k++)
                Sw[k] += w * d_CP[(ii * cols + jj) * 4 + k];
            Sw[3] += w;
        }

    float inv = 1.0f / Sw[3];
    d_V[idx * 3 + 0] = Sw[0] * inv;
    d_V[idx * 3 + 1] = Sw[1] * inv;
    d_V[idx * 3 + 2] = Sw[2] * inv;
}

// GPU-accelerated Bezier surface evaluation.
//   bp:  BezierPatch with flat CP array (x, y, z, w per point)
//   V_out: output vertex buffer, N*N*3 floats
//   N:  sampling grid resolution (N x N vertices)
void eval_bezier_patch(BezierPatch* bp, float* V_out, int N) {
    int order_u = bp->p_u, order_v = bp->p_v;
    int cols   = order_v + 1;                       // control points per row
    int num_cp = (order_u + 1) * cols;

    // precompute binomial coefficients for Bernstein basis
    int* binom_u = (int*)malloc((order_u + 1) * sizeof(int));
    int* binom_v = (int*)malloc((order_v + 1) * sizeof(int));
    compute_binom(order_u, binom_u);
    compute_binom(order_v, binom_v);

    // allocate GPU memory: control points, output vertices, binom tables
    float *d_CP, *d_V;
    int   *d_binom_u, *d_binom_v;
    cudaMalloc(&d_CP,      num_cp * 4 * sizeof(float));
    cudaMalloc(&d_V,       N * N * 3 * sizeof(float));
    cudaMalloc(&d_binom_u, (order_u + 1) * sizeof(int));
    cudaMalloc(&d_binom_v, (order_v + 1) * sizeof(int));

    // copy data to GPU
    cudaMemcpy(d_CP, bp->CP, num_cp * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_binom_u, binom_u, (order_u + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_binom_v, binom_v, (order_v + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // launch kernel: BLOCK_SIZE x BLOCK_SIZE threads per block
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(N / BLOCK_SIZE, N / BLOCK_SIZE);
    eval_surface<<<grid, block>>>(d_CP, d_V, N, order_u, order_v, cols,
                                   d_binom_u, d_binom_v);

    // copy results back to CPU
    cudaMemcpy(V_out, d_V, N * N * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(d_CP); cudaFree(d_V);
    cudaFree(d_binom_u); cudaFree(d_binom_v);
    free(binom_u); free(binom_v);
}

// ========== Standalone CLI ==========
#ifdef STANDALONE
static int next_line(FILE* f, char* buf, int size) {
    while (fgets(buf, size, f))
        if (buf[0] != '#' && buf[0] != '\n') return 1;
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) { printf("Usage: bezier_eval.exe <cp_file>\n"); return 1; }
    FILE* f = fopen(argv[1], "r");
    if (!f) { printf("Cannot open %s\n", argv[1]); return 1; }

    char line[256];
    int order_u, order_v;
    next_line(f, line, 256);
    sscanf(line, "%d %d", &order_u, &order_v);

    BezierPatch bp;
    bp.p_u = order_u; bp.p_v = order_v;
    bp.n_u = order_u; bp.n_v = order_v;
    int num_cp = (order_u + 1) * (order_v + 1);
    bp.CP = (float*)malloc(num_cp * 4 * sizeof(float));
    for (int i = 0; i < num_cp; i++) {
        next_line(f, line, 256);
        sscanf(line, "%f %f %f %f",
            &bp.CP[i*4+0], &bp.CP[i*4+1], &bp.CP[i*4+2], &bp.CP[i*4+3]);
    }
    fclose(f);

    int N = 32;
    float* V = (float*)malloc(N * N * 3 * sizeof(float));
    eval_bezier_patch(&bp, V, N);

    free(bp.CP);

    FILE* out = fopen("build/output.obj", "w");
    for (int i = 0; i < N * N; i++)
        fprintf(out, "v %f %f %f\n", V[i*3+0], V[i*3+1], V[i*3+2]);
    for (int i = 0; i < N - 1; i++)
        for (int j = 0; j < N - 1; j++) {
            int a = i * N + j + 1, b = a + 1;
            int c = (i + 1) * N + j + 1, d = c + 1;
            fprintf(out, "f %d %d %d\n", a, b, c);
            fprintf(out, "f %d %d %d\n", b, d, c);
        }
    fclose(out);

    free(V);
    printf("Done: %d CPs, %d verts -> build/output.obj\n", num_cp, N * N);
    return 0;
}
#endif
