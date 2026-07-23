#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define N 32
#define BLOCK_SIZE 8
#define MAX_ORDER 7

void compute_binom(int order, int* binom) {
    binom[0] = 1;
    for (int k = 1; k <= order; k++)
        binom[k] = binom[k - 1] * (order - k + 1) / k;
}

__global__ void eval_surface(float* d_CP, float* d_V, int n,
                            int order_u, int order_v, int cols,
                            int* d_binom_u, int* d_binom_v) {
    int i = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int j = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float u = (float)i / (n - 1);
    float v = (float)j / (n - 1);
    int idx = i * n + j;

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

    float s[4] = {0.0f};
    for (int ii = 0; ii <= order_u; ii++)
        for (int jj = 0; jj <= order_v; jj++) {
            float w = s_Bu[threadIdx.y][ii] * s_Bv[threadIdx.x][jj];
            for (int k = 0; k < 4; k++)
                s[k] += w * d_CP[(ii * cols + jj) * 4 + k];
        }

    float inv = 1.0f / s[3];
    d_V[idx*3+0] = s[0] * inv;
    d_V[idx*3+1] = s[1] * inv;
    d_V[idx*3+2] = s[2] * inv;
}

int main(int argc, char* argv[]) {
    if (argc < 2) { printf("Usage: bezier_eval.exe <cp_file>\n"); return 1; }

    FILE* fin = fopen(argv[1], "r");
    if (!fin) { printf("Cannot open %s\n", argv[1]); return 1; }

    char line[256];
    int order_u, order_v;
    do { if (!fgets(line, 256, fin)) break; }
    while (line[0] == '#' || line[0] == '\n');
    sscanf(line, "%d %d", &order_u, &order_v);

    int rows = order_u + 1, cols = order_v + 1, num_cp = rows * cols;
    float* CP = (float*)malloc(num_cp * 4 * sizeof(float));
    for (int i = 0; i < num_cp; i++) {
        do { if (!fgets(line, 256, fin)) break; }
        while (line[0] == '#' || line[0] == '\n');
        sscanf(line, "%f %f %f %f", &CP[i*4+0], &CP[i*4+1], &CP[i*4+2], &CP[i*4+3]);
    }
    fclose(fin);

    int* binom_u = (int*)malloc((order_u+1) * sizeof(int));
    int* binom_v = (int*)malloc((order_v+1) * sizeof(int));
    compute_binom(order_u, binom_u);
    compute_binom(order_v, binom_v);

    float* V = (float*)malloc(N * N * 3 * sizeof(float));
    float *d_CP, *d_V;
    int *d_binom_u, *d_binom_v;

    cudaMalloc(&d_CP, num_cp * 4 * sizeof(float));
    cudaMalloc(&d_V, N * N * 3 * sizeof(float));
    cudaMalloc(&d_binom_u, (order_u+1) * sizeof(int));
    cudaMalloc(&d_binom_v, (order_v+1) * sizeof(int));

    cudaMemcpy(d_CP, CP, num_cp * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_binom_u, binom_u, (order_u+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_binom_v, binom_v, (order_v+1)*sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(N / BLOCK_SIZE, N / BLOCK_SIZE);
    eval_surface<<<numBlocks, threadsPerBlock>>>(d_CP, d_V, N,
        order_u, order_v, cols, d_binom_u, d_binom_v);

    cudaMemcpy(V, d_V, N * N * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* f = fopen("build/output.obj", "w");
    for (int i = 0; i < N * N; i++)
        fprintf(f, "v %f %f %f\n", V[i*3+0], V[i*3+1], V[i*3+2]);
    for (int i = 0; i < N-1; i++)
        for (int j = 0; j < N-1; j++) {
            int a = i*N+j+1, b = i*N+j+2;
            int c = (i+1)*N+j+1, d = (i+1)*N+j+2;
            fprintf(f, "f %d %d %d\n", a, b, c);
            fprintf(f, "f %d %d %d\n", b, d, c);
        }
    fclose(f);

    cudaFree(d_CP); cudaFree(d_V);
    cudaFree(d_binom_u); cudaFree(d_binom_v);
    free(CP); free(V); free(binom_u); free(binom_v);
    printf("Done: %d CPs, %d verts -> build/output.obj\n", num_cp, N*N);
    return 0;
}
