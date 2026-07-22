#include <stdio.h>
#include <math.h>

#define N 32
#define THREADS_PER_BLOCK 256

__global__ void eval_surface(float* d_CP, float* d_V, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;

    int i = idx / n;
    int j = idx % n;
    float u = (float)i / (n - 1);
    float v = (float)j / (n - 1);

    int binom[4] = {1, 3, 3, 1};
    float B_u[4], B_v[4];

    for(int k = 0; k <= 3; k++){
        B_u[k] = binom[k] * pow(u, k) * pow(1 - u, 3 - k);
        B_v[k] = binom[k] * pow(v, k) * pow(1 - v, 3 - k);
    }

    d_V[idx * 3 + 0] = 0.0; d_V[idx * 3 + 1] = 0.0; d_V[idx * 3 + 2] = 0.0;

    for(int ii = 0; ii <= 3; ii++){
        for(int jj = 0; jj <= 3; jj++){
            d_V[idx * 3 + 0] += B_u[ii] * B_v[jj] * d_CP[(ii * 4 + jj)*3 + 0];
            d_V[idx * 3 + 1] += B_u[ii] * B_v[jj] * d_CP[(ii * 4 + jj)*3 + 1];
            d_V[idx * 3 + 2] += B_u[ii] * B_v[jj] * d_CP[(ii * 4 + jj)*3 + 2];
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

    int numBlocks = (N * N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    eval_surface<<<numBlocks, THREADS_PER_BLOCK>>>(d_CP, d_V, N);

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
