#include "bezier_eval.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define BLOCK_SIZE 8
#define MAX_ORDER 7

// Binomial coefficient C(order, k) via the recurrence v_i = v_{i-1}*(order-i+1)/i.
// Every division is exact (standard Pascal identity), so the values are
// bit-identical to the old host-side compute_binom table — max C(7,3)=35 fits
// int exactly and converts to float losslessly.
__device__ int binom_k(int order, int k) {
    int v = 1;
    for (int i = 1; i <= k; i++)
        v = v * (order - i + 1) / i;
    return v;
}

// ---- Scene batched patch evaluation kernel ----
// One block per patch (gridDim.x = np of the WHOLE scene), block = 8×8. Each
// block loops over the 8×8 tiles of its patch. tu/tv are always multiples of
// 8 (adapt_tess_factors quantizes to [8,64]), so every tile is full — the
// i<tu && j<tv guard on the final write is defensive only. Threads must NEVER
// early-return (the old per-point kernel could): __syncthreads pairing depends
// on all 64 threads running every tile's pipeline.
//
// Per-patch CP centering (ETER §5 precision recovery) happens in-kernel now:
// every thread redundantly computes the bbox centroid (min+max)/2 over the
// patch's ru*rv CPs — bit-identical to the old host-side centering, so the
// output matches the previous per-patch kernels exactly.
//
// Every patch brings its own ru/rv (ScenePatchMeta), so order/cols/useTC are
// block-local — all threads of a block read the same meta, so useTC stays a
// block-uniform branch (no divergence across __syncthreads boundaries).
//
// Tensor Core path (order ≥ 3, ETER §5.1, aligned to paper):
//   A1 = [Bu; Bu'] 16×m, B1 = C_weighted m×n, C1 = [T; Tu] 16×n
//   A2 = [T; Tu],   B2 = Bv^T,             C2 = [S; Su] 16×8
//   A3 = [T; 0],    B3 = Bv'^T,            C3 = [Sv; 0] 8×8
//   normal = cross(Su[ty], Sv[ty])
// NOTE: s_C is only read in columns 0..7 (MMA store_matrix writes 8 columns) —
// keep it that way if this pipeline is ever extended.
__global__ void eval_patches(
    const float* d_CP,     // scene CP array, patch-major (global cpOff in meta)
    float* d_V,            // total_verts*6 floats (pos+normal), per patch v-major
    const int* d_meta)     // np × 6 ints: {tu, tv, outOff, ru, rv, cpOff in vec4s}
{
    int patch = blockIdx.x;
    const int* m = d_meta + patch * 6;
    int tu = m[0], tv = m[1];
    int outBase = m[2];
    int ru = m[3], rv = m[4];
    int order_u = ru - 1, order_v = rv - 1;
    int cols = rv;
    const float* cpBase = d_CP + (size_t)m[5] * 4;   // cpOff in vec4 units
    bool useTC = (order_u >= 3 && order_v >= 3);     // block-uniform (same meta)
    int ty = threadIdx.y, tx = threadIdx.x;
    int bu = binom_k(order_u, tx), bv = binom_k(order_v, ty);  // C(order, idx), safe for any order ≤ 7

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
            float b = (float)bu;
            s_Bu[ty][tx] = powf(u,tx)*powf(1.f-u,order_u-tx)*b;
        }
        if (ty <= order_v) {
            float b = (float)bv;
            s_Bv[tx][ty] = powf(v,ty)*powf(1.f-v,order_v-ty)*b;
        }
        __syncthreads();

        // Derivatives: B'_{k,p}(t) = binom(p,k)*[k*t^{k-1}*(1-t)^{p-k} - (p-k)*t^k*(1-t)^{p-k-1}]
        if (order_u > 0 && tx <= order_u) {
            float k = (float)tx, p = (float)order_u;
            float binom_pk = (float)bu;
            float t1 = (tx>0) ? k * powf(u,k-1.f) * powf(1.f-u, p-k) : 0.f;
            float t2 = (tx<p) ? (p-k) * powf(u,k) * powf(1.f-u, p-k-1.f) : 0.f;
            s_BuP[ty][tx] = binom_pk * (t1 - t2);
        }
        if (order_v > 0 && ty <= order_v) {
            float k = (float)ty, p = (float)order_v;
            float binom_pk = (float)bv;
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

// ---- Process-lifetime device buffer pool ----
// Per-surface eval used to allocate + free 5 device buffers per surface
// (3506 surfaces ≈ 35k cudaMalloc/cudaFree ≈ ~1s of WDDM driver latency).
// With scene-level batching the pool needs only 3 buffers (CP / vertex out /
// meta), allocated once at the scene's largest size and reused forever —
// fully overwritten on every use, so grow-only with no resets is correct.
// The CUDA context reclaims the memory at process exit — no explicit frees.
namespace {
struct DevPool {
    float* d_CP     = nullptr; size_t cpCap   = 0;
    float* d_V      = nullptr; size_t vCap    = 0;
    int*   d_meta   = nullptr; size_t metaCap = 0;
};
DevPool g_devPool;

// Grow-only pool access: reallocates only on record-breaking sizes (typically
// <10 per buffer over a whole run — every later surface reuses the largest).
template<typename T>
bool ensurePool(T*& ptr, size_t& cap, size_t bytes) {
    if (bytes <= cap) return true;
    if (ptr) cudaFree(ptr);
    cudaError_t e = cudaMalloc(&ptr, bytes);
    if (e == cudaSuccess) { cap = bytes; return true; }
    ptr = nullptr; cap = 0;
    fprintf(stderr, "  [CUDA] pool grow to %zu bytes failed: %s\n", bytes, cudaGetErrorString(e));
    return false;
}
}  // namespace

// Evaluate ALL patches of the scene in a single GPU round trip. The old
// per-surface eval_bezier_patches_batched made 6 CUDA API calls per surface
// (4 memcpys + 1 launch + 1 blocking D2H × 3506 surfaces ≈ 21k calls ≈ 0.5s
// of WDDM driver latency on top of the actual work); this needs one cudaMalloc
// group (from the process-lifetime pool above) + 2 H2D memcpys + one launch +
// one blocking D2H for the whole scene (the D2H waits for the kernel).
void eval_bezier_patches_scene(const float* cps, const ScenePatchMeta* meta, int np,
                               float* V_out, int total_verts)
{
    if (np <= 0 || total_verts <= 0) return;

    // cps byte size follows from the meta (Σ ru*rv*4 floats) — mixed CP grids
    // per patch are allowed since each block reads its own ru/rv/cpOff.
    size_t cpsBytes = 0;
    for (int i = 0; i < np; i++)
        cpsBytes += (size_t)meta[i].ru * meta[i].rv * 4 * sizeof(float);
    size_t metaBytes = (size_t)np * 6 * sizeof(int);   // ScenePatchMeta = 6 ints, no padding
    size_t outBytes  = (size_t)total_verts * 6 * sizeof(float);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);   // declared before the gotos (C++ init bypass rule)

    // One-time (grow-only) allocations from the process-lifetime pool.
    if (!ensurePool(g_devPool.d_CP,   g_devPool.cpCap,   cpsBytes)
     || !ensurePool(g_devPool.d_V,    g_devPool.vCap,    outBytes)
     || !ensurePool(g_devPool.d_meta, g_devPool.metaCap, metaBytes)) {
        memset(V_out, 0, outBytes);  // zeros beat garbage on the (OOM-only) failure path
        return;
    }

    cudaError_t e;
    e=cudaMemcpy(g_devPool.d_CP, cps, cpsBytes, cudaMemcpyHostToDevice);
    if(e!=cudaSuccess){fprintf(stderr,"  [CUDA] memcpy H2D cps failed: %s\n",cudaGetErrorString(e));goto cleanup;}
    e=cudaMemcpy(g_devPool.d_meta, meta, metaBytes, cudaMemcpyHostToDevice);
    if(e!=cudaSuccess){fprintf(stderr,"  [CUDA] memcpy H2D meta failed: %s\n",cudaGetErrorString(e));goto cleanup;}

    eval_patches<<<np, block>>>(g_devPool.d_CP, g_devPool.d_V, g_devPool.d_meta);
    e=cudaGetLastError();
    if(e!=cudaSuccess) fprintf(stderr,"  [CUDA] kernel failed: %s\n",cudaGetErrorString(e));
    e=cudaMemcpy(V_out, g_devPool.d_V, outBytes, cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess){fprintf(stderr,"  [CUDA] memcpy D2H failed: %s\n",cudaGetErrorString(e));memset(V_out,0,outBytes);}
cleanup:
    ;
}
