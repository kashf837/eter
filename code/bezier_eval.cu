#include <stdio.h>
#include <math.h>

#define N 32
#define BLOCK_SIZE 8

__global__ void eval_surface(float* d_CP, float* d_V, int n) {

    int i = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int j = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float u = (float)i / (n - 1);
    float v = (float)j / (n - 1);

    int idx = i * N + j;

    int binom[4] = {1, 3, 3, 1};
    __shared__ float s_Bu[BLOCK_SIZE][4]; 
    __shared__ float s_Bv[BLOCK_SIZE][4];

    if(threadIdx.x < 4) {
        s_Bu[threadIdx.y][threadIdx.x] = binom[threadIdx.x] * pow(u, threadIdx.x) * pow(1 - u, 3 - threadIdx.x);
    }

    if(threadIdx.y < 4) {
        s_Bv[threadIdx.x][threadIdx.y] = binom[threadIdx.y] * pow(v, threadIdx.y) * pow(1 - v, 3 - threadIdx.y);
    }

    __syncthreads();

    d_V[idx * 3 + 0] = 0.0; d_V[idx * 3 + 1] = 0.0; d_V[idx * 3 + 2] = 0.0;

    for(int ii = 0; ii <= 3; ii++){
        for(int jj = 0; jj <= 3; jj++){
            d_V[idx * 3 + 0] += s_Bu[threadIdx.y][ii] * s_Bv[threadIdx.x][jj] * d_CP[(ii * 4 + jj)*3 + 0];
            d_V[idx * 3 + 1] += s_Bu[threadIdx.y][ii] * s_Bv[threadIdx.x][jj] * d_CP[(ii * 4 + jj)*3 + 1];
            d_V[idx * 3 + 2] += s_Bu[threadIdx.y][ii] * s_Bv[threadIdx.x][jj] * d_CP[(ii * 4 + jj)*3 + 2];
        }
    }

}

int main() {
    float CP[16 * 3] = {
        0.0,  0.0,   0.0,
        0.33, 0.0,   0.5,
        0.67, 0.0,   0.5,
        1.0,  0.0,   0.0,

        0.0,  0.33,  0.5,
        0.33, 0.33,  1.0,
        0.67, 0.33,  1.0,
        1.0,  0.33,  0.5,

        0.0,  0.67,  0.5,
        0.33, 0.67,  1.0,
        0.67, 0.67,  1.0,
        1.0,  0.67,  0.5,

        0.0,  1.0,   0.0,
        0.33, 1.0,   0.5,
        0.67, 1.0,   0.5,
        1.0,  1.0,   0.0
    };

    FILE* f = fopen("output.obj", "w"); 
    float *d_CP, *d_V;
    float V[N * N * 3];

    cudaMalloc(&d_CP, 16 * 3 * sizeof(float));
    cudaMalloc(&d_V, N * N * 3 * sizeof(float));

    cudaMemcpy(d_CP, CP, 16 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(N / BLOCK_SIZE, N / BLOCK_SIZE);
    eval_surface<<<numBlocks, threadsPerBlock>>>(d_CP, d_V, N);

    cudaMemcpy(V, d_V, N * N * 3 *sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N * N; i++){
        fprintf(f, "v %f %f %f\n", V[i * 3 + 0], V[i * 3 + 1], V[i * 3 + 2]);
    }

    for(int i = 0; i < N - 1; i++){
        for(int j = 0; j < N - 1; j++){
            fprintf(f, "f %d %d %d\n", i * N + j + 1, i * N + j + 2, (i + 1) * N + j + 1);
            fprintf(f, "f %d %d %d\n", i * N + j + 2, (i + 1) * N + j + 2, (i + 1) * N + j + 1);
        }
    }

    cudaFree(d_CP);
    cudaFree(d_V);

    fclose(f);

    return 0;
}
