#include <stdio.h>
#include <math.h>

#define N 32

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

    float u, v;
    int binom[4] = {1, 3, 3, 1};
    float B_u[4], B_v[4];
    float Sx, Sy, Sz;
    FILE* f = fopen("output.obj", "w"); 

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            u = (float)i / (N - 1);
            v = (float)j / (N - 1);
            for(int k = 0; k <= 3; k++){
                B_u[k] = binom[k] * pow(u, k) * pow(1 - u, 3 - k);
                B_v[k] = binom[k] * pow(v, k) * pow(1 - v, 3 - k);
            }
            Sx = 0.0; Sy = 0.0; Sz = 0.0;
            for(int ii = 0; ii <= 3; ii++){
                for(int jj = 0; jj <= 3; jj++){
                    Sx += B_u[ii] * B_v[jj] * CP[(ii*4 + jj)*3 + 0];
                    Sy += B_u[ii] * B_v[jj] * CP[(ii*4 + jj)*3 + 1];
                    Sz += B_u[ii] * B_v[jj] * CP[(ii*4 + jj)*3 + 2];
                }
            }
            fprintf(f, "v %f %f %f\n", Sx, Sy, Sz);
        }
    }

    for(int i = 0; i < N - 1; i++){
        for(int j = 0; j < N - 1; j++){
            fprintf(f, "f %d %d %d\n", i * N + j + 1, i * N + j + 2, (i + 1) * N + j + 1);
            fprintf(f, "f %d %d %d\n", i * N + j + 2, (i + 1) * N + j + 2, (i + 1) * N + j + 1);
        }
    }

    fclose(f);

    return 0;
}
