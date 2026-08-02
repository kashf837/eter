cmake_minimum_required(VERSION 3.22)
project(eter LANGUAGES CXX CUDA)

# C++20，跟 nvpro_core2 保持一致
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# ============================================================
# 主程序：nurbs_eval
# ============================================================
add_executable(nurbs_eval
  code/nurbs_eval.cu
  code/knot_insert.cu
  code/bezier_eval.cu
  code/adaptive_tess.cu
  code/render.cu
)

# ============================================================
# 编译选项
# ============================================================
# 压制编码警告（你之前 build.ps1 里也有这个）
target_compile_options(nurbs_eval PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819>)

# 输出到 build/ 根目录
set_target_properties(nurbs_eval PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
