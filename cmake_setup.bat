@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul
if exist build rmdir /s /q build
set "CUDACXX=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe"
echo CUDACXX=%CUDACXX%
cmake -S D:\project\eter -B D:\project\eter\build -G Ninja
