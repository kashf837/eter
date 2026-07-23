#include <stdio.h>
#include <math.h>
#include <time.h>

#define N 2048
#define BLOCK_SIZE 8

float CP[16 * 3] = {
    0.0,  0.0,   0.0,   0.33, 0.0,   0.5,   0.67, 0.0,   0.5,   1.0,  0.0,   0.0,
    0.0,  0.33,  0.5,   0.33, 0.33,  1.0,   0.67, 0.33,  1.0,   1.0,  0.33,  0.5,
    0.0,  0.67,  0.5,   0.33, 0.67,  1.0,   0.67, 0.67,  1.0,   1.0,  0.67,  0.5,
    0.0,  1.0,   0.0,   0.33, 1.0,   0.5,   0.67, 1.0,   0.5,   1.0,  1.0,   0.0
};

__constant__ float d_CP_const[16 * 3];

void cpu_eval(float* V, int n) {
    int binom[4] = {1, 3, 3, 1};
    for (int i = 0; i < n; i++) {
        float u = (float)i / (n - 1);
        float Bu[4];
        for (int k = 0; k <= 3; k++)
            Bu[k] = binom[k] * powf(u, k) * powf(1 - u, 3 - k);
        for (int j = 0; j < n; j++) {
            float v = (float)j / (n - 1);
            float Bv[4];
            for (int k = 0; k <= 3; k++)
                Bv[k] = binom[k] * powf(v, k) * powf(1 - v, 3 - k);
            float sx = 0, sy = 0, sz = 0;
            for (int ii = 0; ii <= 3; ii++)
                for (int jj = 0; jj <= 3; jj++) {
                    float w = Bu[ii] * Bv[jj];
                    sx += w * CP[(ii * 4 + jj) * 3 + 0];
                    sy += w * CP[(ii * 4 + jj) * 3 + 1];
                    sz += w * CP[(ii * 4 + jj) * 3 + 2];
                }
            int idx = i * n + j;
            V[idx * 3 + 0] = sx; V[idx * 3 + 1] = sy; V[idx * 3 + 2] = sz;
        }
    }
}

__global__ void gpu_naive(float* d_CP, float* d_V, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    int i = idx / n, j = idx % n;
    float u = (float)i / (n - 1), v = (float)j / (n - 1);
    int binom[4] = {1, 3, 3, 1};
    float Bu[4], Bv[4];
    for (int k = 0; k <= 3; k++) {
        Bu[k] = binom[k] * powf(u, k) * powf(1 - u, 3 - k);
        Bv[k] = binom[k] * powf(v, k) * powf(1 - v, 3 - k);
    }
    float sx = 0, sy = 0, sz = 0;
    for (int ii = 0; ii <= 3; ii++)
        for (int jj = 0; jj <= 3; jj++) {
            float w = Bu[ii] * Bv[jj];
            sx += w * d_CP[(ii * 4 + jj) * 3 + 0];
            sy += w * d_CP[(ii * 4 + jj) * 3 + 1];
            sz += w * d_CP[(ii * 4 + jj) * 3 + 2];
        }
    d_V[idx * 3 + 0] = sx; d_V[idx * 3 + 1] = sy; d_V[idx * 3 + 2] = sz;
}

__global__ void gpu_shared(float* d_CP, float* d_V, int n) {
    int i = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int j = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int idx = i * n + j;
    float u = (float)i / (n - 1), v = (float)j / (n - 1);
    int binom[4] = {1, 3, 3, 1};
    __shared__ float s_Bu[BLOCK_SIZE][4];
    __shared__ float s_Bv[BLOCK_SIZE][4];
    if (threadIdx.x < 4)
        s_Bu[threadIdx.y][threadIdx.x] = binom[threadIdx.x] * powf(u, threadIdx.x) * powf(1 - u, 3 - threadIdx.x);
    if (threadIdx.y < 4)
        s_Bv[threadIdx.x][threadIdx.y] = binom[threadIdx.y] * powf(v, threadIdx.y) * powf(1 - v, 3 - threadIdx.y);
    __syncthreads();
    float sx = 0, sy = 0, sz = 0;
    for (int ii = 0; ii <= 3; ii++)
        for (int jj = 0; jj <= 3; jj++) {
            float w = s_Bu[threadIdx.y][ii] * s_Bv[threadIdx.x][jj];
            sx += w * d_CP[(ii * 4 + jj) * 3 + 0];
            sy += w * d_CP[(ii * 4 + jj) * 3 + 1];
            sz += w * d_CP[(ii * 4 + jj) * 3 + 2];
        }
    d_V[idx * 3 + 0] = sx; d_V[idx * 3 + 1] = sy; d_V[idx * 3 + 2] = sz;
}

__global__ void gpu_shared_const(float* d_V, int n) {
    int i = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int j = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int idx = i * n + j;
    float u = (float)i / (n - 1), v = (float)j / (n - 1);
    int binom[4] = {1, 3, 3, 1};
    __shared__ float s_Bu[BLOCK_SIZE][4];
    __shared__ float s_Bv[BLOCK_SIZE][4];
    if (threadIdx.x < 4)
        s_Bu[threadIdx.y][threadIdx.x] = binom[threadIdx.x] * powf(u, threadIdx.x) * powf(1 - u, 3 - threadIdx.x);
    if (threadIdx.y < 4)
        s_Bv[threadIdx.x][threadIdx.y] = binom[threadIdx.y] * powf(v, threadIdx.y) * powf(1 - v, 3 - threadIdx.y);
    __syncthreads();
    float sx = 0, sy = 0, sz = 0;
    for (int ii = 0; ii <= 3; ii++)
        for (int jj = 0; jj <= 3; jj++) {
            float w = s_Bu[threadIdx.y][ii] * s_Bv[threadIdx.x][jj];
            sx += w * d_CP_const[(ii * 4 + jj) * 3 + 0];
            sy += w * d_CP_const[(ii * 4 + jj) * 3 + 1];
            sz += w * d_CP_const[(ii * 4 + jj) * 3 + 2];
        }
    d_V[idx * 3 + 0] = sx; d_V[idx * 3 + 1] = sy; d_V[idx * 3 + 2] = sz;
}

float time_gpu_naive(float* d_CP, float* d_V, int n, int runs) {
    int t = 256, b = (n * n + t - 1) / t;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int k = 0; k < runs; k++) gpu_naive<<<b, t>>>(d_CP, d_V, n);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms / runs;
}

float time_gpu_shared(float* d_CP, float* d_V, int n, int runs) {
    dim3 thr(BLOCK_SIZE, BLOCK_SIZE), blk(n / BLOCK_SIZE, n / BLOCK_SIZE);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int k = 0; k < runs; k++) gpu_shared<<<blk, thr>>>(d_CP, d_V, n);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms / runs;
}

float time_gpu_shared_const(float* d_V, int n, int runs) {
    dim3 thr(BLOCK_SIZE, BLOCK_SIZE), blk(n / BLOCK_SIZE, n / BLOCK_SIZE);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int k = 0; k < runs; k++) gpu_shared_const<<<blk, thr>>>(d_V, n);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms / runs;
}

double time_cpu(float* V, int n) {
    clock_t s = clock();
    cpu_eval(V, n);
    return (double)(clock() - s) / CLOCKS_PER_SEC * 1000.0;
}

int main() {
    int total = N * N, runs = 20;
    float* V_cpu = (float*)malloc(total * 3 * sizeof(float));
    float *d_V;
    cudaMalloc(&d_V, total * 3 * sizeof(float));

    float *d_CP;
    cudaMalloc(&d_CP, 16 * 3 * sizeof(float));
    cudaMemcpy(d_CP, CP, 16 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_CP_const, CP, 16 * 3 * sizeof(float));

    printf("=== Bezier Surface Evaluation Benchmark ===\n");
    printf("N=%d, total=%d, runs=%d\n\n", N, total, runs);

    printf("[CPU] running...\n");
    double cpu_ms = time_cpu(V_cpu, N);
    printf("[CPU]                avg: %7.3f ms\n\n", cpu_ms);

    gpu_naive<<<(total+255)/256, 256>>>(d_CP, d_V, N);
    cudaDeviceSynchronize();

    printf("[GPU Naive]          running...\n");
    float nv = time_gpu_naive(d_CP, d_V, N, runs);
    printf("[GPU Naive]          avg: %7.3f ms  (vs CPU: %.1fx)\n\n", nv, cpu_ms / nv);

    printf("[GPU Shared]         running...\n");
    float sh = time_gpu_shared(d_CP, d_V, N, runs);
    printf("[GPU Shared]         avg: %7.3f ms  (vs CPU: %.1fx, vs Naive: %.1fx)\n\n",
           sh, cpu_ms / sh, nv / sh);

    printf("[GPU Shared+Const]   running...\n");
    float sc = time_gpu_shared_const(d_V, N, runs);
    printf("[GPU Shared+Const]   avg: %7.3f ms  (vs CPU: %.1fx, vs Naive: %.1fx)\n\n",
           sc, cpu_ms / sc, nv / sc);

    cudaFree(d_CP); cudaFree(d_V);
    free(V_cpu);
    return 0;
}
