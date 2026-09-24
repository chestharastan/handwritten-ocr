@echo off
rem Train the Khmer handwriting model on an NVIDIA GPU (Windows).
rem
rem   train.bat                 new run (previous checkpoints are backed up to checkpoints\old_<time>\)
rem   train.bat --resume        continue from checkpoints\last.pt
rem   train.bat --height 96     any train.py option is passed through
rem
rem The first run creates .venv and installs the CUDA build of PyTorch.
setlocal EnableDelayedExpansion
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo Creating .venv and installing requirements...
  python -m venv .venv || exit /b 1
  .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124 || exit /b 1
  .venv\Scripts\pip install -r requirements.txt || exit /b 1
)

.venv\Scripts\python -c "import torch; print('CUDA:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available')"

echo %* | findstr /c:"--resume" >nul
if errorlevel 1 if exist checkpoints\*.pt (
  for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set OLD=checkpoints\old_%%t
  mkdir "!OLD!"
  copy /y checkpoints\*.pt "!OLD!\" >nul
  if exist train.log move train.log "!OLD!\" >nul
  echo Backed up previous checkpoints to !OLD!
)

rem PowerShell Tee-Object shows progress and also writes train.log
powershell -NoProfile -Command ".venv\Scripts\python -u train.py %* 2>&1 | Tee-Object -FilePath train.log -Append"
