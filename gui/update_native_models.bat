@echo off
setlocal
cd /d "%~dp0"
python -c "import sys; print('Python: ' + sys.executable)"
if errorlevel 1 exit /b 1
python -m pip install libreyolo==1.5.0 pycocotools==2.0.11
if errorlevel 1 exit /b 1
python ..\python\prepare_builtin_assets.py --only libreyolo_detect_dfine_nano,libreyolo_classify_mobilenetv4_medium --strict
if errorlevel 1 exit /b 1
echo Native model weights are ready. Restart the Python app.
exit /b 0
