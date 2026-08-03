#pragma once

// main.cpp calls this; all CUDA code is hidden inside gradient.cu
void launch_fill_gradient(float* cpu_data, int W, int H, float blue);
