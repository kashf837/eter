@echo off
set GLSLC=C:\VulkanSDK\1.4.357.0\Bin\glslc.exe
%GLSLC% shaders\mesh.vert -o shaders\spv\mesh.vert.spv
%GLSLC% shaders\mesh.frag -o shaders\spv\mesh.frag.spv
if %ERRORLEVEL% EQU 0 (echo Shaders compiled OK) else (echo Shader compile FAILED)
