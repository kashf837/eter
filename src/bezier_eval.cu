#include "bezier_eval.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define BLOCK_SIZE 8
#define MAX_ORDER 7

void compute_binom(int order, int* binom) {
    binom[0] = 1;
    for (int k = 1; k <= order; k++)
        binom[k] = binom[k - 1] * (order - k + 1) / k;
}

// ---- Original shared-memory kernel ----
__global__ void eval_surface(
    float* d_CP, float* d_V, int nu, int nv,
    int order_u, int order_v, int cols,
    int* d_binom_u, int* d_binom_v)
{
    int i = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int j = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= nu || j >= nv) return;
    int idx = i * nv + j;
    float u = (float)i / (nu - 1), v = (float)j / (nv - 1);

    __shared__ float s_Bu[BLOCK_SIZE][MAX_ORDER + 1];
    __shared__ float s_Bv[BLOCK_SIZE][MAX_ORDER + 1];

    if (threadIdx.x <= order_u)
        s_Bu[threadIdx.y][threadIdx.x] = powf(u, threadIdx.x)*powf(1.f-u, order_u-threadIdx.x)*(float)d_binom_u[threadIdx.x];
    if (threadIdx.y <= order_v)
        s_Bv[threadIdx.x][threadIdx.y] = powf(v, threadIdx.y)*powf(1.f-v, order_v-threadIdx.y)*(float)d_binom_v[threadIdx.y];
    __syncthreads();

    float Sw[4] = {0};
    for (int ii = 0; ii <= order_u; ii++)
        for (int jj = 0; jj <= order_v; jj++) {
            float wb = s_Bu[threadIdx.y][ii] * s_Bv[threadIdx.x][jj];
            float w = wb * d_CP[(ii*cols+jj)*4+3];
            for (int k = 0; k < 3; k++) Sw[k] += w * d_CP[(ii*cols+jj)*4+k];
            Sw[3] += w;
        }
    float inv = 1.f / Sw[3];
    d_V[idx*3+0] = Sw[0]*inv; d_V[idx*3+1] = Sw[1]*inv; d_V[idx*3+2] = Sw[2]*inv;
}

// ---- Tensor Core kernel (ETER §5.1) ----
// Step1: [B;B'](16xm)×C(mxn) → [T;Tu]
// Step2: [T;Tu](16xn)×Bv(nx8) → [S; dS/du]
// Step3: [T;0](16xn)×Bv'(nx8) → [dS/dv; 0]
__global__ void eval_surface_tc(
    float* d_CP, float* d_V, int nu, int nv,
    int order_u, int order_v, int cols,
    int* d_binom_u, int* d_binom_v)
{
    int i = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int j = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= nu || j >= nv) return;
    int idx = i * nv + j;
    float u = (float)i/(nu-1), v = (float)j/(nv-1);
    int m = order_u+1, n = order_v+1;

    // Bernstein bases + derivatives
    __shared__ float s_Bu[BLOCK_SIZE][MAX_ORDER+1];
    __shared__ float s_BuP[BLOCK_SIZE][MAX_ORDER+1];
    __shared__ float s_Bv[BLOCK_SIZE][MAX_ORDER+1];
    __shared__ float s_BvP[BLOCK_SIZE][MAX_ORDER+1];

    if (threadIdx.x <= order_u) {
        float b = (float)d_binom_u[threadIdx.x];
        s_Bu[threadIdx.y][threadIdx.x] = powf(u,threadIdx.x)*powf(1.f-u,order_u-threadIdx.x)*b;
    }
    if (threadIdx.y <= order_v) {
        float b = (float)d_binom_v[threadIdx.y];
        s_Bv[threadIdx.x][threadIdx.y] = powf(v,threadIdx.y)*powf(1.f-v,order_v-threadIdx.y)*b;
    }
    // Compute derivatives: B'_k,p(t) = (k - p·t) / (t·(1-t)) * B_k,p(t)
    // Simpler: B'_k,p = p*(B_{k-1,p-1} - B_{k,p-1}) using lower-order Bernstein
    __syncthreads();
    if (order_u > 0 && threadIdx.x <= order_u-1 && threadIdx.y < BLOCK_SIZE) {
        int n1 = order_u-1;
        int b1 = 1;
        for (int k=1; k<=threadIdx.x; k++) b1 = b1*(n1-k+1)/k;
        s_BuP[threadIdx.y][threadIdx.x] = order_u * b1 * powf(u,threadIdx.x)*powf(1.f-u,n1-threadIdx.x);
    }
    if (order_v > 0 && threadIdx.y <= order_v-1 && threadIdx.x < BLOCK_SIZE) {
        int n1 = order_v-1;
        int b1 = 1;
        for (int k=1; k<=threadIdx.y; k++) b1 = b1*(n1-k+1)/k;
        s_BvP[threadIdx.x][threadIdx.y] = order_v * b1 * powf(v,threadIdx.y)*powf(1.f-v,n1-threadIdx.y);
    }
    __syncthreads();

    __shared__ half  s_A[256];
    __shared__ half  s_B[256];
    __shared__ float s_C[256];
    __shared__ float s_T[128]; // save Bu*C (rows 0-7) for step 3
    bool warp0 = (threadIdx.y*BLOCK_SIZE+threadIdx.x) < 32;

    float pos[4]={0}, dSu[4]={0}, dSv[4]={0};

    for (int comp = 0; comp < 4; comp++) {
        // ---- Build A = [Bu; Bu'](16xm) padded to 16x16 ----
        for (int t = threadIdx.y*BLOCK_SIZE+threadIdx.x; t < 256; t += BLOCK_SIZE*BLOCK_SIZE) {
            int r = t/16, c = t%16;
            float val = 0;
            if (c < m) {
                if      (r < 8)                     val = s_Bu[r][c];
                else if (r < 16 && r-8 < 8 && c <= order_u-1) val = s_BuP[r-8][c];
            }
            s_A[t] = __float2half(val);
        }
        __syncthreads();

        // ---- Build B = C_weighted(mxn) padded to 16x16 ----
        for (int t = threadIdx.y*BLOCK_SIZE+threadIdx.x; t < 256; t += BLOCK_SIZE*BLOCK_SIZE) {
            int r = t/16, c = t%16;
            float val = 0;
            if (r < m && c < n) {
                float w = d_CP[(r*cols+c)*4+3];
                val = (comp<3) ? w*d_CP[(r*cols+c)*4+comp] : w;
            }
            s_B[t] = __float2half(val);
        }
        __syncthreads();

        // MMA1: [Bu;Bu'](16x16) * C(16x16) → [T;Tu](16x16)
        if (warp0) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b;
            wmma::fragment<wmma::accumulator,16,16,16,float> c;
            wmma::fill_fragment(c,0);
            wmma::load_matrix_sync(a,s_A,16); wmma::load_matrix_sync(b,s_B,16);
            wmma::mma_sync(c,a,b,c);
            wmma::store_matrix_sync(s_C,c,16,wmma::mem_row_major);
        }
        __syncthreads();
        // Save T = Bu*C (first 8 rows) for step 3
        for (int t = threadIdx.y*BLOCK_SIZE+threadIdx.x; t < 128; t += BLOCK_SIZE*BLOCK_SIZE)
            s_T[t] = s_C[t];

        // ---- MMA2: [T;Tu] × Bv → [S; dS/du] ----
        // Copy intermediate → A (float→half), rows 0-7=T, rows 8-15=Tu
        for (int t = threadIdx.y*BLOCK_SIZE+threadIdx.x; t < 256; t += BLOCK_SIZE*BLOCK_SIZE)
            s_A[t] = __float2half(s_C[t]);
        __syncthreads();

        // B = Bv^T (nx8 → 16x16 padded)
        for (int t = threadIdx.y*BLOCK_SIZE+threadIdx.x; t < 256; t += BLOCK_SIZE*BLOCK_SIZE) {
            int r = t/16, c = t%16;
            s_B[t] = (c < BLOCK_SIZE && r < n) ? __float2half(s_Bv[c][r]) : __float2half(0.f);
        }
        __syncthreads();

        if (warp0) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b;
            wmma::fragment<wmma::accumulator,16,16,16,float> c;
            wmma::fill_fragment(c,0);
            wmma::load_matrix_sync(a,s_A,16); wmma::load_matrix_sync(b,s_B,16);
            wmma::mma_sync(c,a,b,c);
            wmma::store_matrix_sync(s_C,c,16,wmma::mem_row_major);
        }
        __syncthreads();

        // pos from row threadIdx.y (T*Bv^T), dSu from row threadIdx.y+8 (Tu*Bv^T)
        pos[comp] = s_C[threadIdx.y*16+threadIdx.x];
        dSu[comp] = (threadIdx.y < BLOCK_SIZE) ? s_C[(8+threadIdx.y)*16+threadIdx.x] : 0;

        // ---- MMA3: [T;0] × Bv' → [dS/dv; 0] ----
        // Rebuild A: rows 0-7 = T (saved from MMA1), rows 8-15 = 0
        for (int t = threadIdx.y*BLOCK_SIZE+threadIdx.x; t < 256; t += BLOCK_SIZE*BLOCK_SIZE) {
            int r = t/16, c = t%16;
            s_A[t] = (r < 8) ? __float2half(s_T[r*16+c]) : __float2half(0.f);
        }
        __syncthreads();

        // B = Bv'^T
        for (int t = threadIdx.y*BLOCK_SIZE+threadIdx.x; t < 256; t += BLOCK_SIZE*BLOCK_SIZE) {
            int r = t/16, c = t%16;
            s_B[t] = (c < BLOCK_SIZE && r < n) ? __float2half(s_BvP[c][r]) : __float2half(0.f);
        }
        __syncthreads();

        if (warp0) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b;
            wmma::fragment<wmma::accumulator,16,16,16,float> c;
            wmma::fill_fragment(c,0);
            wmma::load_matrix_sync(a,s_A,16); wmma::load_matrix_sync(b,s_B,16);
            wmma::mma_sync(c,a,b,c);
            wmma::store_matrix_sync(s_C,c,16,wmma::mem_row_major);
        }
        __syncthreads();

        dSv[comp] = s_C[threadIdx.y*16+threadIdx.x];
    }

    // Perspective division
    float iw = 1.f/(pos[3]+1e-10f);
    d_V[idx*3+0] = pos[0]*iw;
    d_V[idx*3+1] = pos[1]*iw;
    d_V[idx*3+2] = pos[2]*iw;
}

void eval_bezier_patch(BezierPatch* bp, float* V_out, int nu, int nv) {
    int order_u = bp->p_u, order_v = bp->p_v;
    int cols = order_v+1, num_cp = (order_u+1)*cols, total = nu*nv;
    int* binom_u = (int*)malloc((order_u+1)*sizeof(int));
    int* binom_v = (int*)malloc((order_v+1)*sizeof(int));
    compute_binom(order_u,binom_u); compute_binom(order_v,binom_v);
    float *d_CP, *d_V; int *d_binom_u, *d_binom_v;
    cudaMalloc(&d_CP, num_cp*4*sizeof(float));
    cudaMalloc(&d_V, total*3*sizeof(float));
    cudaMalloc(&d_binom_u, (order_u+1)*sizeof(int));
    cudaMalloc(&d_binom_v, (order_v+1)*sizeof(int));
    cudaMemcpy(d_CP,bp->CP,num_cp*4*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_binom_u,binom_u,(order_u+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_binom_v,binom_v,(order_v+1)*sizeof(int),cudaMemcpyHostToDevice);
    dim3 block(BLOCK_SIZE,BLOCK_SIZE);
    dim3 grid((nv+BLOCK_SIZE-1)/BLOCK_SIZE, (nu+BLOCK_SIZE-1)/BLOCK_SIZE);
    bool useTC = (order_u>=3 && order_v>=3);
    if (useTC)
        eval_surface_tc<<<grid,block>>>(d_CP,d_V,nu,nv,order_u,order_v,cols,d_binom_u,d_binom_v);
    else
        eval_surface<<<grid,block>>>(d_CP,d_V,nu,nv,order_u,order_v,cols,d_binom_u,d_binom_v);
    cudaMemcpy(V_out,d_V,total*3*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(d_CP); cudaFree(d_V);
    cudaFree(d_binom_u); cudaFree(d_binom_v);
    free(binom_u); free(binom_v);
}
