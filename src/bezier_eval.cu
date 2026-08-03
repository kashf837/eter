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
    float* d_CP, float* d_V, int nu, int nv,
    int order_u, int order_v, int cols,
    int* d_binom_u, int* d_binom_v
) {
    int i  = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int j  = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= nu || j >= nv) return;
    int idx = i * nv + j;

    float u = (float)i / (nu - 1);
    float v = (float)j / (nv - 1);

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

void eval_bezier_patch(BezierPatch* bp, float* V_out, int nu, int nv) {
    int order_u = bp->p_u, order_v = bp->p_v;
    int cols   = order_v + 1;
    int num_cp = (order_u + 1) * cols;
    int total  = nu * nv;

    int* binom_u = (int*)malloc((order_u + 1) * sizeof(int));
    int* binom_v = (int*)malloc((order_v + 1) * sizeof(int));
    compute_binom(order_u, binom_u);
    compute_binom(order_v, binom_v);

    float *d_CP, *d_V;
    int   *d_binom_u, *d_binom_v;
    cudaMalloc(&d_CP,      num_cp * 4 * sizeof(float));
    cudaMalloc(&d_V,       total * 3 * sizeof(float));
    cudaMalloc(&d_binom_u, (order_u + 1) * sizeof(int));
    cudaMalloc(&d_binom_v, (order_v + 1) * sizeof(int));

    cudaMemcpy(d_CP, bp->CP, num_cp * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_binom_u, binom_u, (order_u + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_binom_v, binom_v, (order_v + 1) * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((nv + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (nu + BLOCK_SIZE - 1) / BLOCK_SIZE);
    eval_surface<<<grid, block>>>(d_CP, d_V, nu, nv, order_u, order_v, cols,
                                   d_binom_u, d_binom_v);

    cudaMemcpy(V_out, d_V, total * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_CP); cudaFree(d_V);
    cudaFree(d_binom_u); cudaFree(d_binom_v);
    free(binom_u); free(binom_v);
}
