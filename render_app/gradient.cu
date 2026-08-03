#include <cuda_runtime.h>
#include "gradient.h"

// Each thread fills one pixel's RGBA values.
__global__ void fill_gradient_kernel(float* data, int W, int H, float blue)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W * H) return;

    int x = idx % W;
    int y = idx / W;

    float* pixel = data + idx * 4;
    pixel[0] = (float)x / (float)(W - 1);
    pixel[1] = (float)y / (float)(H - 1);
    pixel[2] = blue;
    pixel[3] = 1.0f;
}

// Wrapper: main.cpp calls this, all CUDA is inside here
void launch_fill_gradient(float* cpu_data, int W, int H, float blue)
{
    float* d_data = nullptr;
    cudaMalloc(&d_data, W * H * 4 * sizeof(float));

    int threads = 256;
    int blocks = (W * H + threads - 1) / threads;
    fill_gradient_kernel<<<blocks, threads>>>(d_data, W, H, blue);
    cudaDeviceSynchronize();
    cudaMemcpy(cpu_data, d_data, W * H * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}
