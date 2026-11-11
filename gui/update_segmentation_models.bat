@echo off
setlocal
cd /d "%~dp0"
echo Updating segmentation model support...
REM Keep the installed CUDA/CPU torch and torchvision packages unchanged.
python -m pip install --no-deps timm==1.0.31
if errorlevel 1 goto :failed
python -c "import timm, torch, torchvision"
if errorlevel 1 goto :failed
python ..\python\prepare_builtin_assets.py --only mobilenetv4_small_encoder,efficientnet_b1_encoder,shufflenet_v2_x1_0,bisenetv2_backbone,stdc1_backbone --strict
if errorlevel 1 goto :failed
echo Ready. Restart the Python app and select a model in Training.
if not defined DVS_WEIGHTS_NONINTERACTIVE pause
exit /b 0
:failed
echo Setup failed. See the error above. Completed downloads are preserved.
if not defined DVS_WEIGHTS_NONINTERACTIVE pause
exit /b 1
