cmake_minimum_required(VERSION 3.22)
project(eter LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 89)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/openmp>)

# ---- nvpro_core2 ----
option(NVPRO2_ENABLE_nvgl "" OFF)
option(NVPRO2_ENABLE_nvgpu_monitor "" OFF)
option(NVPRO2_ENABLE_nvslang "" OFF)
option(NVPRO2_ENABLE_nvvkglsl "" OFF)
option(NVPRO2_ENABLE_nvvkgltf "" OFF)
option(NVPRO2_ENABLE_nvshaders_host "" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(NvproCore2 REQUIRED)

# ---- Source files ----
add_executable(eter
  src/main.cpp
  src/nurbs.cu
  src/knot_insert.cu
  src/bezier_eval.cu
  src/adaptive_tess.cu
)

# ---- Link ----
target_link_libraries(eter PRIVATE nvpro2::nvapp)
add_project_definitions(eter)
copy_to_runtime_and_install(eter AUTO)
