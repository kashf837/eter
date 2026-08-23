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

// ---- Batched patch evaluation kernel ----
// One block per patch (gridDim.x = np), block = 8×8. Each block loops over
// the 8×8 tiles of its patch. tu/tv are always multiples of 8 (adapt_tess_
// factors quantizes to [8,64]), so every tile is full — the i<tu && j<tv
// guard on the final write is defensive only. Threads must NEVER early-return
// (the old per-point kernel could): __syncthreads pairing depends on all 64
// threads running every tile's pipeline.
//
// Per-patch CP centering (ETER §5 precision recovery) happens in-kernel now:
// every thread redundantly computes the bbox centroid (min+max)/2 over the
// patch's ru*rv CPs — bit-identical to the old host-side centering, so the
// output matches the previous per-patch kernels exactly.
//
// Tensor Core path (order ≥ 3, ETER §5.1, aligned to paper):
//   A1 = [Bu; Bu'] 16×m, B1 = C_weighted m×n, C1 = [T; Tu] 16×n
//   A2 = [T; Tu],   B2 = Bv^T,             C2 = [S; Su] 16×8
//   A3 = [T; 0],    B3 = Bv'^T,            C3 = [Sv; 0] 8×8
//   normal = cross(Su[ty], Sv[ty])
// NOTE: s_C is only read in columns 0..7 (MMA store_matrix writes 8 columns) —
// keep it that way if this pipeline is ever extended.
__global__ void eval_patches(
    const float* d_CP,     // np patches × ru*rv vec4s (uncentered), patch-major
    float* d_V,            // total_verts*6 floats (pos+normal), per patch v-major
    const int* d_meta,     // np × 3 ints: {tu, tv, outOff in verts} (= PatchInfo)
    int ru, int rv,        // CP grid per patch (orders = ru-1, rv-1), surface-uniform
    int useTC,             // order ≥ 3 in both directions (block-uniform branch)
    const int* d_binom_u, const int* d_binom_v)
{
    int patch = blockIdx.x;
    int order_u = ru - 1, order_v = rv - 1;
    int cols = rv;
    int tu = d_meta[patch*3+0], tv = d_meta[patch*3+1];
    int outBase = d_meta[patch*3+2];
    const float* cpBase = d_CP + (size_t)patch * ru * rv * 4;
    int ty = threadIdx.y, tx = threadIdx.x;

    // ---- Per-patch bbox centroid (min+max)/2, same as the old host code ----
    float cpMin[3]={1e30f,1e30f,1e30f}, cpMax[3]={-1e30f,-1e30f,-1e30f};
    for (int k = 0; k < ru * rv; k++) {
        for (int a = 0; a < 3; a++) {
            float v = cpBase[k*4+a];
            if (v < cpMin[a]) cpMin[a] = v;
            if (v > cpMax[a]) cpMax[a] = v;
        }
    }
    float cent[3] = { (cpMin[0]+cpMax[0])*0.5f,
                      (cpMin[1]+cpMax[1])*0.5f,
                      (cpMin[2]+cpMax[2])*0.5f };

    // ---- Shared: Bernstein bases + derivatives + MMA scratchpad ----
    __shared__ float s_Bu[BLOCK_SIZE][MAX_ORDER+1];
    __shared__ float s_BuP[BLOCK_SIZE][MAX_ORDER+1];
    __shared__ float s_Bv[BLOCK_SIZE][MAX_ORDER+1];
    __shared__ float s_BvP[BLOCK_SIZE][MAX_ORDER+1];
    __shared__ half  s_A[256];   // A matrix (16×16 half)
    __shared__ half  s_B[256];   // B matrix (16×16 half)
    __shared__ float s_C[256];   // C accumulators (16×16 float)
    __shared__ float s_T[128];   // save T (8 rows of MMA1)

    bool warp0 = (ty*BLOCK_SIZE+tx) < 32;

    for (int uTile = 0; uTile < tu/BLOCK_SIZE; uTile++)
    for (int vTile = 0; vTile < tv/BLOCK_SIZE; vTile++) {
        int i = uTile*BLOCK_SIZE + ty;   // u index
        int j = vTile*BLOCK_SIZE + tx;   // v index
        float u = (float)i/(tu-1), v = (float)j/(tv-1);
        __syncthreads();  // tile top: previous tile's shared reads are done

        // ---- Bernstein bases (verbatim from the old per-patch kernel) ----
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

        float pos[4]={0}, su[4]={0}, sv[4]={0};

        if (useTC) {

            // ---- Tensor Core path (ETER §5.1, verbatim pipeline) ----
            // Pad dimensions for m16n8k16: K=16, M=16, N=8
            int m = order_u+1, n = order_v+1;

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

                // ---- Build B = C_weighted : m×n, pad to 16×8 (CPs offset by centroid) ----
                for (int t = ty*BLOCK_SIZE+tx; t < 256; t += BLOCK_SIZE*BLOCK_SIZE) {
                    int r = t/16, c = t%16;
                    float val = 0;
                    if (r < m && c < n) {
                        float cpw = cpBase[(r*cols+c)*4+3];
                        float coord = cpBase[(r*cols+c)*4+comp] - (comp < 3 ? cent[comp] : 0.f);
                        val = (comp < 3) ? cpw * coord : cpw;
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

            // Restore centroid offset: S' = S - C*w → add C*w back
            pos[0] += cent[0] * pos[3];
            pos[1] += cent[1] * pos[3];
            pos[2] += cent[2] * pos[3];

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

            if (i < tu && j < tv) {
                int oi = outBase + j*tu + i;  // v-major: v outer, u inner
                d_V[oi*6+0] = Px;     d_V[oi*6+1] = Py;     d_V[oi*6+2] = Pz;
                d_V[oi*6+3] = nx*il;  d_V[oi*6+4] = ny*il;  d_V[oi*6+5] = nz*il;
            }

        } else {

            // ---- Plain path (order < 3): weighted sum, in-kernel centering ----
            // Normals are written as 0 (same as the old plain kernel) — the
            // mesh builder fills them via finite differences if needed.
            float Sw[4] = {0};
            for (int ii = 0; ii <= order_u; ii++)
                for (int jj = 0; jj <= order_v; jj++) {
                    float wb = s_Bu[ty][ii] * s_Bv[tx][jj];
                    float w = wb * cpBase[(ii*cols+jj)*4+3];
                    Sw[0] += w * (cpBase[(ii*cols+jj)*4+0] - cent[0]);
                    Sw[1] += w * (cpBase[(ii*cols+jj)*4+1] - cent[1]);
                    Sw[2] += w * (cpBase[(ii*cols+jj)*4+2] - cent[2]);
                    Sw[3] += w;
                }
            // Restore centroid offset
            Sw[0] += cent[0] * Sw[3]; Sw[1] += cent[1] * Sw[3]; Sw[2] += cent[2] * Sw[3];
            float iw = 1.f / Sw[3];

            if (i < tu && j < tv) {
                int oi = outBase + j*tu + i;
                d_V[oi*6+0] = Sw[0]*iw; d_V[oi*6+1] = Sw[1]*iw; d_V[oi*6+2] = Sw[2]*iw;
                d_V[oi*6+3] = 0.f;      d_V[oi*6+4] = 0.f;      d_V[oi*6+5] = 0.f;
            }
        }
    }
}

// Evaluate all patches of one surface in a single GPU round trip. The old
// per-patch eval_bezier_patch made 13 CUDA API calls per patch (4 allocs +
// 3 H2D + 1 launch + 1 blocking D2H + 4 frees ≈ 680µs each on WDDM = 39s
// for 57k patches); this needs one cudaMalloc group + one H2D + one launch +
// one D2H for the whole surface.
void eval_bezier_patches_batched(const float* cps, const PatchInfo* info, int np,
                                 int ru, int rv, float* V_out, int total_verts)
{
    if (np <= 0 || total_verts <= 0) return;
    int order_u = ru - 1, order_v = rv - 1;
    int* binom_u = (int*)malloc(ru * sizeof(int));
    int* binom_v = (int*)malloc(rv * sizeof(int));
    compute_binom(order_u, binom_u); compute_binom(order_v, binom_v);

    size_t cpsBytes    = (size_t)np * ru * rv * 4 * sizeof(float);
    size_t metaBytes   = (size_t)np * 3 * sizeof(int);   // PatchInfo = 3 ints, no padding
    size_t outBytes    = (size_t)total_verts * 6 * sizeof(float);
    size_t binomUBytes = (size_t)ru * sizeof(int), binomVBytes = (size_t)rv * sizeof(int);

    float *d_CP=0, *d_V=0; int *d_meta=0, *d_binom_u=0, *d_binom_v=0;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);   // declared before the gotos (C++ init bypass rule)
    int useTC = (order_u >= 3 && order_v >= 3);
    cudaError_t e;
    e=cudaMalloc(&d_CP, cpsBytes);
    if(e!=cudaSuccess){fprintf(stderr,"  [CUDA] cudaMalloc d_CP failed: %s\n",cudaGetErrorString(e));goto cleanup;}
    e=cudaMalloc(&d_V, outBytes);
    if(e!=cudaSuccess){fprintf(stderr,"  [CUDA] cudaMalloc d_V failed: %s\n",cudaGetErrorString(e));goto cleanup;}
    e=cudaMalloc(&d_meta, metaBytes);
    if(e!=cudaSuccess){fprintf(stderr,"  [CUDA] cudaMalloc d_meta failed: %s\n",cudaGetErrorString(e));goto cleanup;}
    e=cudaMalloc(&d_binom_u, binomUBytes);
    if(e!=cudaSuccess){fprintf(stderr,"  [CUDA] cudaMalloc d_binom_u failed: %s\n",cudaGetErrorString(e));goto cleanup;}
    e=cudaMalloc(&d_binom_v, binomVBytes);
    if(e!=cudaSuccess){fprintf(stderr,"  [CUDA] cudaMalloc d_binom_v failed: %s\n",cudaGetErrorString(e));goto cleanup;}
    e=cudaMemcpy(d_CP, cps, cpsBytes, cudaMemcpyHostToDevice);
    if(e!=cudaSuccess){fprintf(stderr,"  [CUDA] memcpy H2D cps failed: %s\n",cudaGetErrorString(e));goto cleanup;}
    e=cudaMemcpy(d_meta, info, metaBytes, cudaMemcpyHostToDevice);
    if(e!=cudaSuccess){fprintf(stderr,"  [CUDA] memcpy H2D meta failed: %s\n",cudaGetErrorString(e));goto cleanup;}
    e=cudaMemcpy(d_binom_u, binom_u, binomUBytes, cudaMemcpyHostToDevice);
    e=cudaMemcpy(d_binom_v, binom_v, binomVBytes, cudaMemcpyHostToDevice);

    eval_patches<<<np, block>>>(d_CP, d_V, d_meta, ru, rv, useTC, d_binom_u, d_binom_v);
    e=cudaGetLastError();
    if(e!=cudaSuccess) fprintf(stderr,"  [CUDA] kernel failed: %s\n",cudaGetErrorString(e));
    e=cudaMemcpy(V_out, d_V, outBytes, cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess){fprintf(stderr,"  [CUDA] memcpy D2H failed: %s\n",cudaGetErrorString(e));memset(V_out,0,outBytes);}
cleanup:
    if(d_CP) cudaFree(d_CP); if(d_V) cudaFree(d_V); if(d_meta) cudaFree(d_meta);
    if(d_binom_u) cudaFree(d_binom_u); if(d_binom_v) cudaFree(d_binom_v);
    free(binom_u); free(binom_v);
}
