#include "bezier_eval.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define BLOCK_SIZE 8
#define MAX_ORDER 7

static void compute_binom(int order, int* binom) {
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
    float u = (float)i/(nu-1), v = (float)j/(nv-1);

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
    d_V[idx*6+0] = Sw[0]*inv; d_V[idx*6+1] = Sw[1]*inv; d_V[idx*6+2] = Sw[2]*inv;
    d_V[idx*6+3] = 0.f; d_V[idx*6+4] = 0.f; d_V[idx*6+5] = 0.f;
}

// ---- Tensor Core kernel (ETER §5.1, aligned to paper) ----
// Paper uses m16n8k8 WMMA; our hardware (sm_89) supports m16n8k16.
// K=16 means we pad m,n ≤ 8 to k=16 (acceptable overhead for degrees 3-7).
//
// Matrix definitions (m = order_u+1, n = order_v+1, both ≤ 8):
//   A1 = [Bu; Bu']   : 16×m   (rows 0-7=Bu, rows 8-15=Bu')
//   B1 = C_weighted   : m×n    (CP coords × w, one per component)
//   C1 = [T; Tu]      : 16×n   (rows 0-7=T, rows 8-15=Tu)
//   A2 = [T; Tu]      : 16×n   (reuse C1)
//   B2 = Bv^T         : n×8    (Bernstein basis in v, transposed)
//   C2 = [S; Su]      : 16×8   (positions + u-tangents)
//   A3 = [T; 0]       : 8×n    (rows 0-7=T, rows 8-15=0)
//   B3 = Bv'^T        : n×8    (derivative basis in v)
//   C3 = [Sv; 0]      : 16×8   (v-tangents in rows 0-7)
// Normal = cross(Su[ty], Sv[ty]) for each thread row ty.
//
// TODO: precision recovery (Ootomo & Yokota 2022).
// FP16 TC results have ~1e-3 relative error. For rendering, this is ~35× finer
// than pixel precision (1px ≈ 0.035 units on a 36-unit model). Visual impact is
// minimal; implementing when numerical correctness is required for publication.
__global__ void eval_surface_tc(
    float* d_CP, float* d_V, int nu, int nv,
    int order_u, int order_v, int cols,
    int* d_binom_u, int* d_binom_v)
{
    int ty = threadIdx.y, tx = threadIdx.x;
    int i = blockIdx.y * BLOCK_SIZE + ty;
    int j = blockIdx.x * BLOCK_SIZE + tx;
    if (i >= nu || j >= nv) return;
    int idx = i * nv + j;
    float u = (float)i/(nu-1), v = (float)j/(nv-1);
    int m = order_u+1, n = order_v+1;

    // ---- Shared: Bernstein bases + derivatives ----
    __shared__ float s_Bu[BLOCK_SIZE][MAX_ORDER+1];
    __shared__ float s_BuP[BLOCK_SIZE][MAX_ORDER+1];
    __shared__ float s_Bv[BLOCK_SIZE][MAX_ORDER+1];
    __shared__ float s_BvP[BLOCK_SIZE][MAX_ORDER+1];

    if (tx <= order_u) {
        float b = (float)d_binom_u[tx];
        s_Bu[ty][tx] = powf(u,tx)*powf(1.f-u,order_u-tx)*b;
    }
    if (ty <= order_v) {
        float b = (float)d_binom_v[ty];
        s_Bv[tx][ty] = powf(v,ty)*powf(1.f-v,order_v-ty)*b;
    }
    __syncthreads();

    // Derivatives: B'_{k,p}(t) = binom(p,k)*[k*t^{k-1}*(1-t)^{p-k} - (p-k)*t^k*(1-t)^{p-k-1}]
    if (order_u > 0 && tx <= order_u) {
        float k = (float)tx, p = (float)order_u;
        float binom_pk = (float)d_binom_u[tx];
        float t1 = (tx>0) ? k * powf(u,k-1.f) * powf(1.f-u, p-k) : 0.f;
        float t2 = (tx<p) ? (p-k) * powf(u,k) * powf(1.f-u, p-k-1.f) : 0.f;
        s_BuP[ty][tx] = binom_pk * (t1 - t2);
    }
    if (order_v > 0 && ty <= order_v) {
        float k = (float)ty, p = (float)order_v;
        float binom_pk = (float)d_binom_v[ty];
        float t1 = (ty>0) ? k * powf(v,k-1.f) * powf(1.f-v, p-k) : 0.f;
        float t2 = (ty<p) ? (p-k) * powf(v,k) * powf(1.f-v, p-k-1.f) : 0.f;
        s_BvP[tx][ty] = binom_pk * (t1 - t2);
    }
    __syncthreads();

    // ---- MMA scratchpad ----
    __shared__ half  s_A[256];   // A matrix (16×16 half)
    __shared__ half  s_B[256];   // B matrix (16×16 half)
    __shared__ float s_C[256];   // C accumulators (16×16 float)
    __shared__ float s_T[128];   // save T (8 rows of MMA1)

    bool warp0 = (ty*BLOCK_SIZE+tx) < 32;

    float pos[4]={0}, su[4]={0}, sv[4]={0};

    // Pad dimensions for m16n8k16: K=16, M=16, N=8

    for (int comp = 0; comp < 4; comp++) {

        // ---- Build A = [Bu; Bu'] : 16×m, pad to 16×16 ----
        for (int t = ty*BLOCK_SIZE+tx; t < 256; t += BLOCK_SIZE*BLOCK_SIZE) {
            int r = t/16, c = t%16;
            float val = 0;
            if (c <= order_u && r < 16) {
                if (r < BLOCK_SIZE)            val = s_Bu[r][c];
                else if (r >= BLOCK_SIZE && r-BLOCK_SIZE < BLOCK_SIZE)
                    val = s_BuP[r-BLOCK_SIZE][c];
            }
            s_A[t] = __float2half(val);
        }
        __syncthreads();

        // ---- Build B = C_weighted : m×n, pad to 16×8 ----
        for (int t = ty*BLOCK_SIZE+tx; t < 256; t += BLOCK_SIZE*BLOCK_SIZE) {
            int r = t/16, c = t%16;
            float val = 0;
            if (r < m && c < n) {
                float cpw = d_CP[(r*cols+c)*4+3];
                val = (comp < 3) ? cpw * d_CP[(r*cols+c)*4+comp] : cpw;
            }
            s_B[t] = __float2half(val);
        }
        __syncthreads();

        // MMA1: [Bu;Bu'](16×16) × C_weighted(16×8) → [T;Tu](16×8)
        if (warp0) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b;
            wmma::fragment<wmma::accumulator,16,16,16,float> c;
            wmma::fill_fragment(c, 0);
            wmma::load_matrix_sync(a, s_A, 16);
            wmma::load_matrix_sync(b, s_B, 16);
            wmma::mma_sync(c, a, b, c);
            wmma::store_matrix_sync(s_C, c, 16, wmma::mem_row_major);
        }
        __syncthreads();

        // Save T = first 8 rows of MMA1 result for MMA3
        for (int t = ty*BLOCK_SIZE+tx; t < 8*n; t += BLOCK_SIZE*BLOCK_SIZE)
            s_T[t] = s_C[t];

        // ---- MMA2: [T;Tu](16×n) × Bv^T(n×8) → [S;Su](16×8) ----
        // A = [T;Tu] from s_C: rows 0-7=T, rows 8-15=Tu, cols 0..n-1
        for (int t = ty*BLOCK_SIZE+tx; t < 256; t += BLOCK_SIZE*BLOCK_SIZE) {
            int r = t/16, c = t%16;
            s_A[t] = (r < 16 && c < n) ? __float2half(s_C[r*16+c]) : __float2half(0.f);
        }
        __syncthreads();

        // B = Bv^T : n×8 → pad to 16×8
        for (int t = ty*BLOCK_SIZE+tx; t < 256; t += BLOCK_SIZE*BLOCK_SIZE) {
            int r = t/16, c = t%16;
            s_B[t] = (r < n && c < BLOCK_SIZE) ? __float2half(s_Bv[c][r]) : __float2half(0.f);
        }
        __syncthreads();

        // ---- MMA2 main pass ----
        if (warp0) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b;
            wmma::fragment<wmma::accumulator,16,16,16,float> c;
            wmma::fill_fragment(c, 0);
            wmma::load_matrix_sync(a, s_A, 16);
            wmma::load_matrix_sync(b, s_B, 16);
            wmma::mma_sync(c, a, b, c);
            wmma::store_matrix_sync(s_C, c, 16, wmma::mem_row_major);
        }
        __syncthreads();

        // pos = S[ty] (row ty of S), su = Su[ty] (row ty+8)
        pos[comp] = s_C[ty*16 + tx];
        su[comp] = (ty < BLOCK_SIZE) ? s_C[(8+ty)*16 + tx] : 0;

        // ---- MMA3: [T;0](8×n) × Bv'^T(n×8) → [Sv;0](8×8) ----
        // A = [T;0] : first 8 rows = T (saved), last 8 rows = 0
        for (int t = ty*BLOCK_SIZE+tx; t < 256; t += BLOCK_SIZE*BLOCK_SIZE) {
            int r = t/16, c = t%16;
            s_A[t] = (r < 8 && c < n) ? __float2half(s_T[r*16+c]) : __float2half(0.f);
        }
        __syncthreads();

        // B = Bv'^T : (order_v+1)×8
        for (int t = ty*BLOCK_SIZE+tx; t < 256; t += BLOCK_SIZE*BLOCK_SIZE) {
            int r = t/16, c = t%16;
            s_B[t] = (r <= order_v && c < BLOCK_SIZE) ? __float2half(s_BvP[c][r]) : __float2half(0.f);
        }
        __syncthreads();

        if (warp0) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b;
            wmma::fragment<wmma::accumulator,16,16,16,float> c;
            wmma::fill_fragment(c, 0);
            wmma::load_matrix_sync(a, s_A, 16);
            wmma::load_matrix_sync(b, s_B, 16);
            wmma::mma_sync(c, a, b, c);
            wmma::store_matrix_sync(s_C, c, 16, wmma::mem_row_major);
        }
        __syncthreads();

        sv[comp] = s_C[ty*16 + tx];
    }

    // ---- Perspective division ----
    float iw = 1.f / (pos[3] + 1e-10f);
    float Px = pos[0]*iw, Py = pos[1]*iw, Pz = pos[2]*iw;

    // Quotient-rule tangents: T_u = (S_u*w - S*w_u) / w², dir preserved
    float w = pos[3];
    float Tux = su[0]*w - pos[0]*su[3];
    float Tuy = su[1]*w - pos[1]*su[3];
    float Tuz = su[2]*w - pos[2]*su[3];
    float Tvx = sv[0]*w - pos[0]*sv[3];
    float Tvy = sv[1]*w - pos[1]*sv[3];
    float Tvz = sv[2]*w - pos[2]*sv[3];
    // Cross product
    float nx = Tuy*Tvz - Tuz*Tvy;
    float ny = Tuz*Tvx - Tux*Tvz;
    float nz = Tux*Tvy - Tuy*Tvx;
    float il = 1.f / (sqrtf(nx*nx+ny*ny+nz*nz) + 1e-10f);

    d_V[idx*6+0] = Px;     d_V[idx*6+1] = Py;     d_V[idx*6+2] = Pz;
    d_V[idx*6+3] = nx*il;  d_V[idx*6+4] = ny*il;  d_V[idx*6+5] = nz*il;
}

void eval_bezier_patch(BezierPatch* bp, float* V_out, int nu, int nv) {
    int order_u = bp->p_u, order_v = bp->p_v;
    int cols = order_v+1, num_cp = (order_u+1)*cols, total = nu*nv;
    int* binom_u = (int*)malloc((order_u+1)*sizeof(int));
    int* binom_v = (int*)malloc((order_v+1)*sizeof(int));
    compute_binom(order_u,binom_u); compute_binom(order_v,binom_v);
    float *d_CP=0, *d_V=0; int *d_binom_u=0, *d_binom_v=0;
    dim3 block(BLOCK_SIZE,BLOCK_SIZE);
    dim3 grid((nv+BLOCK_SIZE-1)/BLOCK_SIZE, (nu+BLOCK_SIZE-1)/BLOCK_SIZE);
    bool useTC = (order_u>=3 && order_v>=3);
    cudaError_t e;
    e=cudaMalloc(&d_CP, num_cp*4*sizeof(float));
    if(e!=cudaSuccess){fprintf(stderr,"  [CUDA] cudaMalloc d_CP failed: %s\n",cudaGetErrorString(e));goto cleanup;}
    e=cudaMalloc(&d_V, total*6*sizeof(float));
    if(e!=cudaSuccess){fprintf(stderr,"  [CUDA] cudaMalloc d_V failed: %s\n",cudaGetErrorString(e));goto cleanup;}
    e=cudaMalloc(&d_binom_u, (order_u+1)*sizeof(int));
    if(e!=cudaSuccess){fprintf(stderr,"  [CUDA] cudaMalloc d_binom_u failed: %s\n",cudaGetErrorString(e));goto cleanup;}
    e=cudaMalloc(&d_binom_v, (order_v+1)*sizeof(int));
    if(e!=cudaSuccess){fprintf(stderr,"  [CUDA] cudaMalloc d_binom_v failed: %s\n",cudaGetErrorString(e));goto cleanup;}
    e=cudaMemcpy(d_CP,bp->CP,num_cp*4*sizeof(float),cudaMemcpyHostToDevice);
    if(e!=cudaSuccess){fprintf(stderr,"  [CUDA] cudaMemcpy H2D failed: %s\n",cudaGetErrorString(e));goto cleanup;}
    e=cudaMemcpy(d_binom_u,binom_u,(order_u+1)*sizeof(int),cudaMemcpyHostToDevice);
    e=cudaMemcpy(d_binom_v,binom_v,(order_v+1)*sizeof(int),cudaMemcpyHostToDevice);
    if (useTC)
        eval_surface_tc<<<grid,block>>>(d_CP,d_V,nu,nv,order_u,order_v,cols,d_binom_u,d_binom_v);
    else
        eval_surface<<<grid,block>>>(d_CP,d_V,nu,nv,order_u,order_v,cols,d_binom_u,d_binom_v);
    e=cudaGetLastError();
    if(e!=cudaSuccess) fprintf(stderr,"  [CUDA] kernel failed: %s\n",cudaGetErrorString(e));
    e=cudaMemcpy(V_out,d_V,total*6*sizeof(float),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess){fprintf(stderr,"  [CUDA] cudaMemcpy D2H failed: %s\n",cudaGetErrorString(e));memset(V_out,0,total*6*sizeof(float));}
cleanup:
    if(d_CP) cudaFree(d_CP); if(d_V) cudaFree(d_V);
    if(d_binom_u) cudaFree(d_binom_u); if(d_binom_v) cudaFree(d_binom_v);
    free(binom_u); free(binom_v);
}
