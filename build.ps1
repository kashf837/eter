# Build nurbs_eval.exe
# Usage: powershell -File build.ps1

$NVCC = "nvcc"
$SRC = "code/nurbs_eval.cu", "code/knot_insert.cu", "code/bezier_eval.cu", "code/adaptive_tess.cu"
$OUT = "build/nurbs_eval.exe"
$FLAGS = "-Xcompiler", "/wd4819"

& $NVCC $SRC $FLAGS -o $OUT
if ($LASTEXITCODE -eq 0) { Write-Host "=== BUILD OK ===" }
