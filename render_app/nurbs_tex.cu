#include <cuda_runtime.h>
#include <math.h>
#include "nurbs_tex.h"

// Evaluate a Bezier patch at W*H parameter points, output RGBA colors.
// Each thread handles one pixel.
__global__ void bezier_to_texture(
    float* data, int W, int H,
    const float* CP, int n_u, int n_v,    // n_u = p_u+1 control points per direction
    float xmin, float xmax, float ymin, float ymax, float zmin, float zmax)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W * H) return;

    int px = idx % W;
    int py = idx / W;
    float u = (float)px / (float)(W - 1);
    float v = (float)py / (float)(H - 1);

    // Evaluate Bezier at (u,v) — rational form
    float sx = 0, sy = 0, sz = 0, sw = 0;
    int p_u = n_u - 1, p_v = n_v - 1;

    for (int i = 0; i < n_u; i++) {
        // Bernstein basis B_{i,p_u}(u)
        float bu = (p_u == 2) ?
            ((i == 0) ? (1-u)*(1-u) : (i == 1) ? 2*u*(1-u) : u*u)
            : powf(u, i) * powf(1-u, p_u-i);  // fallback for higher orders
        for (int j = 0; j < n_v; j++) {
            float bv = (p_v == 2) ?
                ((j == 0) ? (1-v)*(1-v) : (j == 1) ? 2*v*(1-v) : v*v)
                : powf(v, j) * powf(1-v, p_v-j);

            float basis = bu * bv;
            float w     = CP[(i * n_v + j) * 4 + 3];
            sx += basis * w * CP[(i * n_v + j) * 4 + 0];
            sy += basis * w * CP[(i * n_v + j) * 4 + 1];
            sz += basis * w * CP[(i * n_v + j) * 4 + 2];
            sw += basis * w;
        }
    }
    sx /= sw; sy /= sw; sz /= sw;

    // Heat map: z height → color  (blue=low, green=mid, red=high)
    float t = (sz - zmin) / (zmax - zmin + 1e-6f);  // normalize z to [0,1]
    float r, g, b;
    if (t < 0.5f) {
        // blue → cyan (t: 0 → 0.5)
        float s = t * 2.0f;
        r = 0.0f;  g = s;     b = 1.0f - s;
    } else {
        // green → red (t: 0.5 → 1.0)
        float s = (t - 0.5f) * 2.0f;
        r = s;     g = 1.0f - s;  b = 0.0f;
    }

    float* pixel = data + idx * 4;
    pixel[0] = r;
    pixel[1] = g;
    pixel[2] = b;
    pixel[3] = 1.0f;
}

// CPU wrapper — main.cpp calls this
void launch_nurbs_texture(float* cpu_data, int W, int H, const TestPatch* patch)
{
    int n_cp = (patch->p_u + 1) * (patch->p_v + 1);
    int bytes = n_cp * 4 * sizeof(float);

    float* d_CP = nullptr;
    cudaMalloc(&d_CP, bytes);
    cudaMemcpy(d_CP, patch->CP, bytes, cudaMemcpyHostToDevice);

    int img_bytes = W * H * 4 * sizeof(float);
    float* d_data = nullptr;
    cudaMalloc(&d_data, img_bytes);

    int threads = 256;
    int blocks  = (W * H + threads - 1) / threads;

    // Pillow CP: center CP z=1, but Bernstein weight at u=v=0.5 is 0.25
    // Actual z range is ~[0, 0.25], set zmax=0.3 for full heatmap spectrum
    float xmin = 0.0f, xmax = 1.0f;
    float ymin = 0.0f, ymax = 1.0f;
    float zmin = 0.0f, zmax = 0.3f;

    bezier_to_texture<<<blocks, threads>>>(
        d_data, W, H,
        d_CP, patch->p_u + 1, patch->p_v + 1,
        xmin, xmax, ymin, ymax, zmin, zmax);

    cudaDeviceSynchronize();
    cudaMemcpy(cpu_data, d_data, img_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_CP);
}
