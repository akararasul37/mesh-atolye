@echo off
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" goto missing
if not exist "runtime\installed.json" goto missing
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
"venv\Scripts\python.exe" -m mesh_atolye.launch %*
exit /b %ERRORLEVEL%
:missing
echo Mesh Atolye kurulumu eksik. Once kurulum.bat dosyasini calistirin.
exit /b 1
