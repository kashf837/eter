#ifndef RENDER_H
#define RENDER_H

// Screen resolution
#define SCR_W 640
#define SCR_H 640

// Camera parameters (hardcoded — same as adaptive_tess.cu)
#define FOV_Y  60.0f
#define CAM_X  0.5f
#define CAM_Y  0.5f
#define CAM_Z -2.0f

// Depth buffer (GPU-side)
typedef struct {
    int W, H;
    float* d_depth;
} DepthBuffer;

// Allocate and initialize depth buffer on GPU
DepthBuffer* create_depth_buffer(int w, int h);
void free_depth_buffer(DepthBuffer* db);

// Copy depth buffer from GPU and write PPM file
void write_ppm(const char* path, DepthBuffer* db);

// Rasterize a vertex grid (tu x tv) into the depth buffer.
// V is a GPU-side array of tu*tv*3 floats (x,y,z per vertex)
void rasterize_patch(float* d_V, int tu, int tv, DepthBuffer* db);

#endif
